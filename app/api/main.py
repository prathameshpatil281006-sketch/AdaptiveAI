"""FastAPI prediction service foundation for House Price predictions (Step 21-24, 29, 40-42).

Thin layer reusing existing model persistence and prediction logic.
Successful predictions are persisted to SQLite history (Step 24).
Monitoring checks are exposed as endpoints (Step 29). Drift-triggered
retraining is exposed as a thin adapter over DriftRetrainingOrchestrator
(Step 40): the endpoint only forwards the validated promote flag and
returns the orchestrator result unchanged. Retraining/promotion history
is exposed as a thin read adapter over PromotionRepository (Step 41).
Retraining scheduler status is exposed as a thin read-only adapter over
RetrainingScheduler.get_status() (Step 42). No training, MLflow, drift,
scheduling, or background work lives here.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

from app.core.exceptions import (
    AdaptiveAIError,
    DataValidationError,
    ModelNotFoundError,
    PredictionError,
)
from app.database.database import DatabaseManager
from app.database.monitoring_repository import MonitoringRepository
from app.database.promotion_repository import PromotionRepository
from app.database.repository import PredictionRepository
from app.monitoring.monitoring_service import create_house_price_monitoring_service
from app.retraining.model_promoter import create_house_price_promoter
from app.use_cases.house_price.model_persistence import (
    get_production_pointer_path,
    load_house_price_model,
    read_production_pointer,
)
from app.use_cases.house_price.predict import predict_house_price

logger = logging.getLogger(__name__)

# Configurable model location for tests (monkeypatch-safe).
# Defaults to the standard persisted model; tests override with tmp paths.
HOUSE_PRICE_MODEL_NAME = os.getenv("HOUSE_PRICE_MODEL_NAME", "LinearRegression")
HOUSE_PRICE_MODELS_DIR: Optional[Union[str, Path]] = os.getenv("HOUSE_PRICE_MODELS_DIR")


class HousePriceRequest(BaseModel):
    """Pydantic request schema for Pune-aware House Price prediction (schema v2)."""

    area_sqft: float = Field(..., gt=0, description="Carpet area in sq ft, must be > 0")
    bedrooms: int = Field(..., ge=1, le=6, description="Number of bedrooms, 1-6")
    bathrooms: int = Field(..., ge=1, le=6, description="Number of bathrooms, 1-6")
    parking: int = Field(..., ge=0, le=3, description="Dedicated parking slots, 0-3")
    location: str = Field(..., description="Pune locality, e.g. Wakad")
    property_age: float = Field(..., ge=0, le=60, description="Property age in years, 0-60")
    floor_number: int = Field(..., ge=0, description="Floor of the unit, 0 = ground")
    total_floors: int = Field(..., ge=1, description="Total floors in the building, >= 1")
    furnishing_status: str = Field(..., description="Unfurnished, Semi-Furnished or Fully-Furnished")
    property_type: str = Field(..., description="Apartment, Villa or Row House")

    @model_validator(mode="after")
    def _check_schema_v2(self):
        from app.use_cases.house_price.market_config import (
            FURNISHING_VALUES,
            LOCALITY_RATES,
            PROPERTY_TYPE_VALUES,
        )

        if self.location not in LOCALITY_RATES:
            raise ValueError(f"Unknown location: {self.location}")
        if self.furnishing_status not in FURNISHING_VALUES:
            raise ValueError(f"Unknown furnishing_status: {self.furnishing_status}")
        if self.property_type not in PROPERTY_TYPE_VALUES:
            raise ValueError(f"Unknown property_type: {self.property_type}")
        if self.floor_number > self.total_floors:
            raise ValueError("floor_number must not exceed total_floors")
        if self.bathrooms > self.bedrooms + 2:
            raise ValueError("bathrooms must be at most bedrooms + 2")
        return self


class HousePriceResponse(BaseModel):
    """Pydantic response schema containing the prediction (Phase 2 enriched).

    ``prediction`` is the stable Phase 1 contract (ML point estimate in
    INR). All other fields are additive Phase 2 valuation data: market
    context (locality benchmark), the empirical residual interval, and
    actual serving metadata. They default to ``None`` and are populated
    on successful valuation so older clients keep working unchanged.
    """

    prediction: float
    predicted_price_per_sqft: Optional[float] = None
    locality_benchmark_price: Optional[float] = None
    benchmark_price_per_sqft: Optional[float] = None
    benchmark_difference: Optional[float] = None
    benchmark_difference_percent: Optional[float] = None
    prediction_lower: Optional[float] = None
    prediction_upper: Optional[float] = None
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    schema_version: Optional[int] = None
    prediction_id: Optional[int] = None
    inferred_at: Optional[str] = None
    location: Optional[str] = None
    area_sqft: Optional[float] = None


class DriftRetrainingRequest(BaseModel):
    """Pydantic request schema for drift-triggered retraining."""

    promote: bool = True


class LifecycleStartRunRequest(BaseModel):
    """Pydantic request schema for starting an explicit lifecycle run."""

    production_limit: int = Field(default=1000, ge=1, description="Monitoring window size")
    trigger: str = Field(default="manual", description="Run trigger label")


class LifecyclePromotionCheckRequest(BaseModel):
    """Pydantic request schema for a fresh promotion-gate evaluation."""

    window_limit: int = Field(default=1000, ge=1, description="Monitoring window size")


class LifecycleRollbackRequest(BaseModel):
    """Pydantic request schema for explicit pointer rollback."""

    pointer_history_id: int = Field(..., ge=1, description="Pointer-history entry to restore")


class LifecyclePromoteRequest(BaseModel):
    """Pydantic request schema for explicit promotion confirmation.

    Schema-migration runs additionally require ``allow_migration``
    (explicit human confirmation that the incompatible production
    baseline may be replaced on absolute validation). Defaults to
    False, so normal upgrades and unconfirmed migrations never move
    the pointer. Irrelevant for schema-compatible upgrades.
    """

    allow_migration: bool = Field(
        default=False, description="Confirm schema-migration promotion"
    )


class DemoCreateSessionRequest(BaseModel):
    """Pydantic request schema for creating a demo simulator session."""

    scenario: str = Field(..., description="One of the supported demo scenarios")
    sample_count: int = Field(default=100, ge=1, le=1000, description="Demo samples to generate")
    seed: int = Field(default=42, ge=0, description="Deterministic seed")
    params: Optional[dict] = Field(default=None, description="Optional scenario tunables")


class DemoRetrainRequest(BaseModel):
    """Pydantic request schema for demo-triggered retraining."""

    production_limit: int = Field(default=1000, ge=1, description="Monitoring window size")


def create_house_price_drift_orchestrator(
    prediction_repo: PredictionRepository,
    monitoring_repo: Optional[MonitoringRepository] = None,
    promotion_repo: Optional[PromotionRepository] = None,
    tracker=None,
) -> "DriftRetrainingOrchestrator":
    """Builds the House Price drift-to-retraining orchestrator on demand.

    Wires the existing layers without duplicating business logic:

        MonitoringService -> RetrainingTriggerDecision
            -> RetrainingScheduler -> RetrainingService
                -> evaluator/decision gate -> promoter
                -> registry + PromotionRepository

    The existing promoter (with its evaluator/decision safety gate)
    is wired so promote=True can execute the real safe promotion
    flow; promote=False still performs no promotion side effect.
    Called per request inside the endpoint only, so importing or
    creating the app never touches MLflow, the database, or models.
    Tests should monkeypatch this factory to inject a stub
    orchestrator instead of building real services.
    """
    from app.engine.mlflow_tracker import MLflowTracker
    from app.monitoring.retraining_trigger import RetrainingTriggerDecision
    from app.retraining.retraining_service import create_house_price_retraining_service
    from app.scheduler.drift_retraining_orchestrator import DriftRetrainingOrchestrator
    from app.scheduler.retraining_scheduler import RetrainingScheduler
    from app.use_cases.house_price.use_case import HousePriceUseCase

    monitoring_service = create_house_price_monitoring_service(
        repository=prediction_repo,
        monitoring_repository=monitoring_repo,
        source="production",
    )
    trigger = RetrainingTriggerDecision()
    tracker = tracker or MLflowTracker(experiment_name="house_price_retraining")
    promoter = create_house_price_promoter(tracker=tracker)
    retraining_service = create_house_price_retraining_service(
        tracker=tracker,
        promoter=promoter,
        promotion_repository=promotion_repo,
    )
    scheduler = RetrainingScheduler(
        retraining_service=retraining_service,
        use_case=HousePriceUseCase(),
    )
    return DriftRetrainingOrchestrator(
        monitoring_service=monitoring_service,
        trigger_decision=trigger,
        retraining_scheduler=scheduler,
    )


def create_house_price_retraining_scheduler() -> "RetrainingScheduler":
    """Builds the House Price retraining scheduler on demand.

    Reuses the existing scheduler configuration mechanism
    (``scheduler.enabled`` / ``scheduler.interval_minutes`` from
    platform.yaml via ``RetrainingScheduler.from_config``). Adds no
    configuration keys and never enables scheduling: defaults remain
    disabled.

    Called per request inside the status endpoint only, so importing
    or creating the app never constructs a scheduler. Never starts
    background work. Tests should monkeypatch this factory to inject
    a stub scheduler instead of building real services.
    """
    from app.core.config import load_config
    from app.retraining.retraining_service import create_house_price_retraining_service
    from app.scheduler.retraining_scheduler import RetrainingScheduler
    from app.use_cases.house_price.use_case import HousePriceUseCase

    try:
        scheduler_config = load_config().get("scheduler")
    except (FileNotFoundError, ValueError):
        scheduler_config = None
    retraining_service = create_house_price_retraining_service()
    return RetrainingScheduler.from_config(
        retraining_service=retraining_service,
        use_case=HousePriceUseCase(),
        scheduler_config=scheduler_config,
    )


def _sanitize_orchestrator_result_for_http(result):
    """Drops live estimator objects so the retraining result is JSON safe.

    The service keeps the fitted estimator under
    ``retraining_result["best_pipeline"]`` for in-process promotion
    use. That object cannot cross the HTTP boundary, so a copy
    without that single key is returned. Every other field,
    including ``all_results`` entries and any promotion outcome, is
    preserved unchanged. Non-dict results and a ``None``
    retraining result (no-retraining path) pass through untouched.
    """
    if not isinstance(result, dict):
        return result
    inner = result.get("retraining_result")
    if inner is None or not isinstance(inner, dict):
        return result
    if "best_pipeline" not in inner:
        return result
    sanitized_inner = {k: v for k, v in inner.items() if k != "best_pipeline"}
    sanitized = dict(result)
    sanitized["retraining_result"] = sanitized_inner
    return sanitized


def _load_model():
    """Loads the current production House Price model and its serving info.

    Resolution order (Step 55 serving contract):

    1. Production serving reference (``production.json`` in the models
       directory) when present: loads exactly the artifact + version the
       last promotion designated. A broken reference (designated
       artifact missing) never falls back silently.
    2. Legacy named artifact (``HOUSE_PRICE_MODEL_NAME``) only when no
       production reference exists, preserving pre-pointer deployments.

    Returns:
        Tuple of (pipeline, serving dict with model_name/model_version
        or None when serving the legacy artifact).

    Raises:
        ModelNotFoundError: If no production model is deployed.
        AdaptiveAIError: If the production reference is corrupt.
    """
    try:
        pointer = read_production_pointer(models_dir=HOUSE_PRICE_MODELS_DIR)
    except ModelNotFoundError:
        pointer = None
    if pointer is not None:
        model = load_house_price_model(
            pointer["model_name"], models_dir=HOUSE_PRICE_MODELS_DIR
        )
        return model, {
            "model_name": pointer["model_name"],
            "model_version": pointer.get("model_version"),
            "registered_model_name": pointer.get("registered_model_name"),
            "run_id": pointer.get("run_id"),
        }
    model = load_house_price_model(
        HOUSE_PRICE_MODEL_NAME, models_dir=HOUSE_PRICE_MODELS_DIR
    )
    return model, None


def _production_pointer_exists() -> bool:
    """Reports whether a production serving reference file exists."""
    try:
        return get_production_pointer_path(
            models_dir=HOUSE_PRICE_MODELS_DIR
        ).is_file()
    except Exception:
        return False


def create_house_price_deploy_service(
    promotion_repo: Optional[PromotionRepository] = None,
) -> "RetrainingService":
    """Builds the House Price explicit-deployment service on demand.

    Wires the existing retraining service with its evaluator/decision
    gate and promoter, sharing one tracker backend so registration
    uses the same store that tracked the candidate. Called per request
    inside the deploy-initial endpoint only, so importing or creating
    the app never touches MLflow, the database, or models. Tests
    should monkeypatch this factory to inject a stub service instead
    of building real collaborators.
    """
    from app.engine.mlflow_tracker import MLflowTracker
    from app.retraining.retraining_service import (
        create_house_price_retraining_service,
    )

    tracker = MLflowTracker(experiment_name="house_price_retraining")
    promoter = create_house_price_promoter(
        tracker=tracker, models_dir=HOUSE_PRICE_MODELS_DIR
    )
    return create_house_price_retraining_service(
        tracker=tracker,
        promoter=promoter,
        promotion_repository=promotion_repo,
    )


def create_house_price_lifecycle_service(
    db_manager: Optional[DatabaseManager] = None,
    tracker=None,
    promoter=None,
) -> "RetrainingLifecycleService":
    """Builds the House Price explicit lifecycle service on demand.

    Wires the lifecycle service with the existing tracker/promoter
    plus lifecycle repositories bound to the given database manager
    (defaults to the configured database) and a monitoring provider
    for gate data-quality evidence. Called per request inside the
    lifecycle endpoints only, so importing or creating the app never
    touches MLflow, the database, or models. Tests should monkeypatch
    this factory to inject a stub service instead of building real
    collaborators.
    """
    from app.database.lifecycle_repository import (
        LifecycleEventRepository,
        PointerHistoryRepository,
        RetrainingRunRepository,
    )
    from app.engine.mlflow_tracker import MLflowTracker
    from app.monitoring.data_provider import create_house_price_provider
    from app.retraining.lifecycle import create_house_price_lifecycle

    manager = db_manager if db_manager is not None else DatabaseManager()
    tracker = tracker or MLflowTracker(experiment_name="house_price_retraining")
    promoter = promoter or create_house_price_promoter(
        tracker=tracker, models_dir=HOUSE_PRICE_MODELS_DIR
    )
    monitoring_provider = create_house_price_provider(
        repository=PredictionRepository(manager),
        source="production",
    )
    return create_house_price_lifecycle(
        tracker=tracker,
        promoter=promoter,
        run_repository=RetrainingRunRepository(manager),
        event_repository=LifecycleEventRepository(manager),
        pointer_history_repository=PointerHistoryRepository(manager),
        monitoring_provider=monitoring_provider,
        models_dir=HOUSE_PRICE_MODELS_DIR,
    )


def create_house_price_demo_service(
    db_manager: Optional[DatabaseManager] = None,
) -> "DemoService":
    """Builds the House Price demo simulator service on demand.

    Wires prediction + session + audit repositories bound to the given
    database manager (defaults to the configured database). Called per
    request inside the demo endpoints only, so importing or creating
    the app never touches the database or models. Tests should
    monkeypatch this factory to inject a stub service instead of
    building real collaborators.
    """
    from app.demo.service import create_house_price_demo_service as _factory

    return _factory(db_manager=db_manager, models_dir=HOUSE_PRICE_MODELS_DIR)


def _sanitize_retraining_result_for_http(result):
    """Drops the live estimator so a retraining result is JSON safe.

    Removes ``best_pipeline`` (kept in-process for promotion use)
    from a top-level service result; every other field, including all
    candidate results and any promotion outcome, is preserved
    unchanged. Non-dict results pass through untouched.
    """
    if not isinstance(result, dict):
        return result
    if "best_pipeline" not in result:
        return result
    sanitized = dict(result)
    del sanitized["best_pipeline"]
    return sanitized


def create_app(db_manager: Optional[DatabaseManager] = None) -> FastAPI:
    """Application factory returning an independent FastAPI instance.

    Args:
        db_manager: Optional DatabaseManager for prediction history.
            Defaults to the configured development database.
            Tests should pass a temporary DatabaseManager.
            Tables are created lazily on successful prediction so that
            importing or creating the app never touches the real DB file.
    """
    if db_manager is None:
        db_manager = DatabaseManager()
    prediction_repo = PredictionRepository(db_manager)
    monitoring_repo = MonitoringRepository(db_manager)
    promotion_repo = PromotionRepository(db_manager)

    application = FastAPI(
        title="AdaptiveAI API",
        description="House Price prediction service.",
        version="1.0.0",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    application.state.db_manager = db_manager
    application.state.prediction_repo = prediction_repo
    application.state.monitoring_repo = monitoring_repo
    application.state.promotion_repo = promotion_repo

    @application.get("/health")
    def health_check():
        """Simple health check endpoint."""
        return {"status": "healthy"}

    @application.post("/predict/house-price", response_model=HousePriceResponse)
    def predict_house_price_endpoint(payload: HousePriceRequest, request: Request):
        """Predicts a house price, persisting successful predictions to history."""
        try:
            model, serving = _load_model()
        except ModelNotFoundError as e:
            logger.warning(f"House Price model not available: {e}")
            if _production_pointer_exists():
                raise HTTPException(
                    status_code=503,
                    detail="House Price production model is unavailable.",
                ) from e
            raise HTTPException(
                status_code=503,
                detail="No production House Price model is deployed. "
                "Deploy a model before requesting predictions.",
            ) from e
        except Exception as e:
            logger.error(f"Failed to load House Price model: {e}")
            raise HTTPException(status_code=503, detail="House Price model not available.") from e

        try:
            input_data = {
                "area_sqft": payload.area_sqft,
                "bedrooms": payload.bedrooms,
                "bathrooms": payload.bathrooms,
                "parking": payload.parking,
                "location": payload.location,
                "property_age": payload.property_age,
                "floor_number": payload.floor_number,
                "total_floors": payload.total_floors,
                "furnishing_status": payload.furnishing_status,
                "property_type": payload.property_type,
            }
            input_df = pd.DataFrame([input_data])
            predictions = predict_house_price(model, input_df)
            prediction_value = float(predictions[0])
        except DataValidationError as e:
            logger.warning(f"House Price prediction validation failed: {e}")
            raise HTTPException(status_code=422, detail="Invalid House Price input.") from e
        except HTTPException:
            raise
        except PredictionError as e:
            # Step 70 (I-20): a loaded model that crashes on valid input
            # is explicitly unavailable for serving (503), not a generic
            # 500 and not a client input error (422). Internals stay
            # out of the response; nothing is persisted below.
            logger.error(f"House Price prediction failed: {e}")
            raise HTTPException(
                status_code=503, detail="House Price prediction model is unavailable."
            ) from e
        except Exception as e:
            logger.error(f"House Price prediction failed: {e}")
            raise HTTPException(status_code=500, detail="House Price prediction failed.") from e

        try:
            repo: PredictionRepository = request.app.state.prediction_repo
            repo.init()
            serving_name = (
                serving.get("model_name")
                if isinstance(serving, dict) and serving.get("model_name")
                else HOUSE_PRICE_MODEL_NAME
            )
            # Step 71 (I-21): record the authoritative served version
            # from the production reference whenever one is available;
            # legacy fallback (no reference) persists None per contract.
            # Never inferred from filenames; unusable values degrade to
            # None rather than breaking a successful prediction.
            serving_version = None
            if isinstance(serving, dict):
                candidate_version = serving.get("model_version")
                if isinstance(candidate_version, str) and candidate_version.strip():
                    serving_version = candidate_version.strip()
            prediction_id = repo.save_prediction(
                use_case="house_price",
                model_name=serving_name,
                input_data=input_data,
                prediction=prediction_value,
                model_version=serving_version,
                source="production",
            )
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to persist House Price prediction: {e}")
            raise HTTPException(status_code=500, detail="Failed to save prediction.") from e

        # Phase 2 valuation enrichment (market context, residual
        # interval, serving metadata). Enrichment never breaks the core
        # prediction: interval failures degrade to None bounds inside
        # the service, and any unexpected enrichment error degrades to
        # the stable Phase 1 shape instead of failing the request.
        try:
            from app.use_cases.house_price.valuation import build_valuation

            valuation = build_valuation(
                point_prediction=prediction_value,
                input_data=input_data,
                pipeline=model,
                model_name=serving_name,
                model_version=serving_version,
            )
            return HousePriceResponse(
                prediction=prediction_value,
                predicted_price_per_sqft=valuation.get("predicted_price_per_sqft"),
                locality_benchmark_price=valuation.get("locality_benchmark_price"),
                benchmark_price_per_sqft=valuation.get("benchmark_price_per_sqft"),
                benchmark_difference=valuation.get("benchmark_difference"),
                benchmark_difference_percent=valuation.get(
                    "benchmark_difference_percent"
                ),
                prediction_lower=valuation.get("prediction_lower"),
                prediction_upper=valuation.get("prediction_upper"),
                model_name=serving_name,
                model_version=serving_version,
                schema_version=valuation.get("schema_version"),
                prediction_id=prediction_id
                if isinstance(prediction_id, int) and not isinstance(prediction_id, bool)
                else None,
                inferred_at=valuation.get("inferred_at"),
                location=input_data.get("location"),
                area_sqft=input_data.get("area_sqft"),
            )
        except Exception as e:
            logger.warning(f"House Price valuation enrichment degraded: {e}")
            return HousePriceResponse(prediction=prediction_value)

    @application.post("/monitor/house-price", tags=["monitoring"])
    def monitor_house_price_endpoint(request: Request):
        """Runs the House Price monitoring check and persists the result."""
        try:
            prediction_repo: PredictionRepository = request.app.state.prediction_repo
            monitoring_repo: MonitoringRepository = request.app.state.monitoring_repo
            prediction_repo.init()
            monitoring_repo.init()
            service = create_house_price_monitoring_service(
                repository=prediction_repo,
                monitoring_repository=monitoring_repo,
                source="production",
            )
            return service.run_check()
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"House Price monitoring check failed: {e}")
            raise HTTPException(
                status_code=500, detail="House Price monitoring check failed."
            ) from e
        except Exception as e:
            logger.error(f"House Price monitoring check failed: {e}")
            raise HTTPException(
                status_code=500, detail="House Price monitoring check failed."
            ) from e

    @application.get("/monitor/house-price/summary", tags=["monitoring"])
    def monitor_house_price_summary(
        request: Request,
        limit: int = Query(default=1000, ge=1),
    ):
        """Runs a full Phase 4 health check and returns the summary.

        Thin adapter only: resolves the actually-serving model identity
        (read-only; missing reference yields null model fields) and
        the whole-table prediction count, then delegates to
        ``MonitoringService.run_health_check``, which persists the run
        and returns the summary unchanged. Starts no background work.
        """
        try:
            from app.use_cases.house_price.model_persistence import (
                read_production_pointer,
            )

            prediction_repo: PredictionRepository = request.app.state.prediction_repo
            monitoring_repo: MonitoringRepository = request.app.state.monitoring_repo
            prediction_repo.init()
            monitoring_repo.init()
            try:
                pointer = read_production_pointer(models_dir=HOUSE_PRICE_MODELS_DIR)
                serving_info = {
                    "model_name": pointer.get("model_name"),
                    "model_version": pointer.get("model_version"),
                }
            except Exception:
                serving_info = {"model_name": None, "model_version": None}
            try:
                total_predictions = prediction_repo.count_predictions(
                    use_case="house_price", source="production"
                )
            except Exception:
                total_predictions = None
            service = create_house_price_monitoring_service(
                repository=prediction_repo,
                monitoring_repository=monitoring_repo,
                source="production",
            )
            return service.run_health_check(
                production_limit=limit,
                serving_info=serving_info,
                total_predictions=total_predictions,
            )
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"House Price health summary failed: {e}")
            raise HTTPException(
                status_code=500, detail="House Price health summary failed."
            ) from e
        except Exception as e:
            logger.error(f"House Price health summary failed: {e}")
            raise HTTPException(
                status_code=500, detail="House Price health summary failed."
            ) from e

    @application.get("/monitor/house-price/history", tags=["monitoring"])
    def monitor_house_price_history(
        request: Request,
        limit: int = Query(default=100, ge=1),
        offset: int = Query(default=0, ge=0),
    ):
        """Returns persisted House Price monitoring results."""
        try:
            repo: MonitoringRepository = request.app.state.monitoring_repo
            repo.init()
            items = repo.list_monitoring_results(
                limit=limit, offset=offset, use_case="house_price"
            )
            return {"items": items, "limit": limit, "offset": offset}
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"Failed to retrieve monitoring history: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve monitoring history."
            ) from e
        except Exception as e:
            logger.error(f"Failed to retrieve monitoring history: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve monitoring history."
            ) from e

    @application.post("/monitor/house-price/retrain", tags=["monitoring"])
    def monitor_house_price_retrain_endpoint(
        payload: DriftRetrainingRequest, request: Request
    ):
        """Runs one drift-triggered retraining cycle via the orchestrator.

        Thin adapter only: forwards the validated promote flag to
        ``DriftRetrainingOrchestrator.run_check_and_retrain`` exactly
        once and returns its result. The live estimator object kept
        inside the service result for in-process promotion use is
        not JSON serializable, so it is omitted at this HTTP
        boundary; every JSON-safe field is returned unchanged.
        Starts no background work and duplicates no business logic.
        """
        try:
            prediction_repo: PredictionRepository = request.app.state.prediction_repo
            monitoring_repo: MonitoringRepository = request.app.state.monitoring_repo
            promotion_repo: PromotionRepository = request.app.state.promotion_repo
            prediction_repo.init()
            monitoring_repo.init()
            orchestrator = create_house_price_drift_orchestrator(
                prediction_repo=prediction_repo,
                monitoring_repo=monitoring_repo,
                promotion_repo=promotion_repo,
            )
            result = orchestrator.run_check_and_retrain(promote=payload.promote)
            return _sanitize_orchestrator_result_for_http(result)
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"House Price drift-triggered retraining failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="House Price drift-triggered retraining failed.",
            ) from e
        except Exception as e:
            logger.error(f"House Price drift-triggered retraining failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="House Price drift-triggered retraining failed.",
            ) from e

    @application.get("/monitor/house-price/retrain/history", tags=["monitoring"])
    def monitor_house_price_retrain_history(
        request: Request,
        limit: int = Query(default=100, ge=1),
        offset: int = Query(default=0, ge=0),
    ):
        """Returns persisted House Price retraining/promotion history.

        Thin read adapter only: forwards limit/offset to
        ``PromotionRepository.list_promotion_history`` for the
        house_price use case and returns its items unchanged.
        Triggers no retraining and duplicates no business logic.
        """
        try:
            repo: PromotionRepository = request.app.state.promotion_repo
            repo.init()
            items = repo.list_promotion_history(
                limit=limit, offset=offset, use_case="house_price"
            )
            return {"items": items, "limit": limit, "offset": offset}
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"Failed to retrieve retraining history: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve retraining history."
            ) from e
        except Exception as e:
            logger.error(f"Failed to retrieve retraining history: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve retraining history."
            ) from e

    @application.post("/monitor/house-price/deploy-initial", tags=["monitoring"])
    def monitor_house_price_deploy_initial_endpoint(request: Request):
        """Deploys the first production model through an explicit cycle.

        Thin adapter only: builds the deployment service via
        ``create_house_price_deploy_service`` and runs one retraining
        cycle with explicit initial deployment allowed. With no
        production model present, the best candidate is deployed and
        the production serving reference is published, so prediction
        becomes available. With production already present, the cycle
        behaves like a normal promotion-gated run. The live estimator
        object is omitted at this HTTP boundary; every JSON-safe
        field is returned unchanged. Starts no background work and
        duplicates no business logic.
        """
        try:
            from app.use_cases.house_price.use_case import HousePriceUseCase

            promotion_repo: PromotionRepository = request.app.state.promotion_repo
            promotion_repo.init()
            deploy_service = create_house_price_deploy_service(
                promotion_repo=promotion_repo
            )
            result = deploy_service.retrain(
                use_case=HousePriceUseCase(),
                promote=True,
                allow_initial_promotion=True,
                models_dir=HOUSE_PRICE_MODELS_DIR,
            )
            return _sanitize_retraining_result_for_http(result)
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"House Price initial deployment failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="House Price initial deployment failed.",
            ) from e
        except Exception as e:
            logger.error(f"House Price initial deployment failed: {e}")
            raise HTTPException(
                status_code=500,
                detail="House Price initial deployment failed.",
            ) from e

    @application.get("/monitor/house-price/retrain/status", tags=["monitoring"])
    def monitor_house_price_retrain_status():
        """Returns the retraining scheduler status (read-only).

        Thin observer only: calls ``RetrainingScheduler.get_status``
        exactly once and returns its result unchanged. Modifies no
        state, triggers no retraining, and starts no background work.
        """
        try:
            scheduler = create_house_price_retraining_scheduler()
            return scheduler.get_status()
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"Failed to retrieve retraining scheduler status: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve retraining scheduler status.",
            ) from e
        except Exception as e:
            logger.error(f"Failed to retrieve retraining scheduler status: {e}")
            raise HTTPException(
                status_code=500,
                detail="Failed to retrieve retraining scheduler status.",
            ) from e

    @application.get("/retrain/house-price/decision", tags=["retraining"])
    def retrain_house_price_decision(
        request: Request,
        limit: int = Query(default=1000, ge=1),
    ):
        """Returns an explicit retraining recommendation (advice only).

        Thin adapter: runs the Phase 4 health check for the window,
        resolves the serving production identity (read-only), and maps
        both through the deterministic recommendation rules. Trains
        nothing, promotes nothing, moves no pointers.
        """
        try:
            from app.use_cases.house_price.model_persistence import (
                read_production_pointer,
            )

            prediction_repo: PredictionRepository = request.app.state.prediction_repo
            monitoring_repo: MonitoringRepository = request.app.state.monitoring_repo
            prediction_repo.init()
            monitoring_repo.init()
            try:
                pointer = read_production_pointer(models_dir=HOUSE_PRICE_MODELS_DIR)
                serving_info = {
                    "model_name": pointer.get("model_name"),
                    "model_version": pointer.get("model_version"),
                }
            except Exception:
                serving_info = {"model_name": None, "model_version": None}
            try:
                total_predictions = prediction_repo.count_predictions(
                    use_case="house_price", source="production"
                )
            except Exception:
                total_predictions = None
            service = create_house_price_monitoring_service(
                repository=prediction_repo,
                monitoring_repository=monitoring_repo,
                source="production",
            )
            summary = service.run_health_check(
                production_limit=limit,
                serving_info=serving_info,
                total_predictions=total_predictions,
            )
            lifecycle = create_house_price_lifecycle_service(
                db_manager=request.app.state.db_manager
            )
            decision = lifecycle.recommend(
                summary,
                production_model=serving_info.get("model_name"),
                production_version=serving_info.get("model_version"),
            )
            return {
                "decision": decision,
                "monitoring": {
                    "status": summary.get("status"),
                    "severity": summary.get("severity"),
                    "overall_health": summary.get("overall_health"),
                    "data_quality_status": summary.get("data_quality_status"),
                    "production_rows": summary.get("production_rows"),
                    "drifted_count": summary.get("drifted_count"),
                    "total_monitored": summary.get("total_monitored"),
                },
            }
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"House Price retraining decision failed: {e}")
            raise HTTPException(
                status_code=500, detail="House Price retraining decision failed."
            ) from e
        except Exception as e:
            logger.error(f"House Price retraining decision failed: {e}")
            raise HTTPException(
                status_code=500, detail="House Price retraining decision failed."
            ) from e

    @application.post("/retrain/house-price/runs", tags=["retraining"])
    def retrain_house_price_start_run(
        payload: LifecycleStartRunRequest, request: Request
    ):
        """Starts an explicit retraining run (trains, stages, gates).

        Training failures are recorded as failed runs and returned;
        the production pointer is never touched here.
        """
        try:
            lifecycle = create_house_price_lifecycle_service(
                db_manager=request.app.state.db_manager
            )
            return lifecycle.start_run(
                production_limit=payload.production_limit,
                trigger=payload.trigger,
            )
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"House Price retraining run failed: {e}")
            raise HTTPException(
                status_code=500, detail="House Price retraining run failed."
            ) from e
        except Exception as e:
            logger.error(f"House Price retraining run failed: {e}")
            raise HTTPException(
                status_code=500, detail="House Price retraining run failed."
            ) from e

    @application.get("/retrain/house-price/runs", tags=["retraining"])
    def retrain_house_price_list_runs(
        request: Request,
        limit: int = Query(default=100, ge=1),
        offset: int = Query(default=0, ge=0),
    ):
        """Lists recorded retraining runs, newest first."""
        try:
            lifecycle = create_house_price_lifecycle_service(
                db_manager=request.app.state.db_manager
            )
            items = lifecycle.list_runs(
                limit=limit, offset=offset, use_case="house_price"
            )
            return {"items": items, "limit": limit, "offset": offset}
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"Failed to list retraining runs: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to list retraining runs."
            ) from e
        except Exception as e:
            logger.error(f"Failed to list retraining runs: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to list retraining runs."
            ) from e

    @application.get("/retrain/house-price/runs/{run_id}", tags=["retraining"])
    def retrain_house_price_get_run(run_id: str, request: Request):
        """Returns one recorded retraining run with comparison and gate."""
        try:
            lifecycle = create_house_price_lifecycle_service(
                db_manager=request.app.state.db_manager
            )
            run = lifecycle.get_run(run_id)
            if run is None:
                raise HTTPException(status_code=404, detail="Retraining run not found.")
            return run
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"Failed to retrieve retraining run: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve retraining run."
            ) from e
        except Exception as e:
            logger.error(f"Failed to retrieve retraining run: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve retraining run."
            ) from e

    @application.post(
        "/retrain/house-price/runs/{run_id}/promotion-check", tags=["retraining"]
    )
    def retrain_house_price_promotion_check(
        run_id: str, payload: LifecyclePromotionCheckRequest, request: Request
    ):
        """Re-evaluates the promotion gate vs current production.

        Records a promotion_check audit event. Never promotes.
        """
        try:
            lifecycle = create_house_price_lifecycle_service(
                db_manager=request.app.state.db_manager
            )
            checked = lifecycle.promotion_check(
                run_id, window_limit=payload.window_limit
            )
            if checked is None:
                raise HTTPException(status_code=404, detail="Retraining run not found.")
            return checked
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"House Price promotion check failed: {e}")
            raise HTTPException(
                status_code=500, detail="House Price promotion check failed."
            ) from e
        except Exception as e:
            logger.error(f"House Price promotion check failed: {e}")
            raise HTTPException(
                status_code=500, detail="House Price promotion check failed."
            ) from e

    @application.post(
        "/retrain/house-price/runs/{run_id}/promote", tags=["retraining"]
    )
    def retrain_house_price_promote(
        run_id: str,
        request: Request,
        payload: Optional[LifecyclePromoteRequest] = None,
    ):
        """Explicitly promotes a gated candidate to production.

        Re-runs the promotion gate first; any failure leaves the
        production pointer untouched. Schema-migration runs
        additionally require ``{"allow_migration": true}`` confirmation
        in the request body; without it the pointer is unchanged.
        """
        try:
            lifecycle = create_house_price_lifecycle_service(
                db_manager=request.app.state.db_manager
            )
            allow_migration = bool(
                payload is not None and payload.allow_migration is True
            )
            outcome = lifecycle.promote(run_id, allow_migration=allow_migration)
            if outcome is None:
                raise HTTPException(status_code=404, detail="Retraining run not found.")
            return outcome
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"House Price promotion failed: {e}")
            raise HTTPException(
                status_code=500, detail="House Price promotion failed."
            ) from e
        except Exception as e:
            logger.error(f"House Price promotion failed: {e}")
            raise HTTPException(
                status_code=500, detail="House Price promotion failed."
            ) from e

    @application.post("/retrain/house-price/rollback", tags=["retraining"])
    def retrain_house_price_rollback(
        payload: LifecycleRollbackRequest, request: Request
    ):
        """Restores a history-recorded previous production pointer.

        Validates the target artifact and refuses no-op rollbacks; the
        current artifact is always preserved.
        """
        try:
            lifecycle = create_house_price_lifecycle_service(
                db_manager=request.app.state.db_manager
            )
            outcome = lifecycle.rollback(payload.pointer_history_id)
            if outcome is None:
                raise HTTPException(
                    status_code=404, detail="Pointer history entry not found."
                )
            return outcome
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"House Price rollback failed: {e}")
            raise HTTPException(
                status_code=500, detail="House Price rollback failed."
            ) from e
        except Exception as e:
            logger.error(f"House Price rollback failed: {e}")
            raise HTTPException(
                status_code=500, detail="House Price rollback failed."
            ) from e

    @application.get("/demo/house-price/scenarios", tags=["demo"])
    def demo_house_price_scenarios():
        """Lists supported simulator scenarios with expectations.

        Read-only metadata (names, titles, expected monitoring
        outcomes). No state, no side effects.
        """
        try:
            from app.demo.scenarios import describe_scenarios

            return {"items": describe_scenarios()}
        except AdaptiveAIError as e:
            logger.error(f"Failed to list demo scenarios: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to list demo scenarios."
            ) from e

    @application.post("/demo/house-price/sessions", tags=["demo"])
    def demo_house_price_create_session(
        payload: DemoCreateSessionRequest, request: Request
    ):
        """Creates an isolated demo session (no samples yet).

        Thin adapter: validates the scenario eagerly and records a
        ``demo_session_created`` audit event. Creates no samples and
        touches no production state.
        """
        try:
            service = create_house_price_demo_service(
                db_manager=request.app.state.db_manager
            )
            return service.create_session(
                scenario=payload.scenario,
                sample_count=payload.sample_count,
                seed=payload.seed,
                params=payload.params,
            )
        except AdaptiveAIError as e:
            logger.warning(f"Demo session creation failed: {e}")
            raise HTTPException(status_code=422, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Demo session creation failed: {e}")
            raise HTTPException(
                status_code=500, detail="Demo session creation failed."
            ) from e

    @application.get("/demo/house-price/sessions", tags=["demo"])
    def demo_house_price_list_sessions(
        request: Request,
        limit: int = Query(default=50, ge=1),
        offset: int = Query(default=0, ge=0),
    ):
        """Lists demo sessions, newest first (read-only)."""
        try:
            service = create_house_price_demo_service(
                db_manager=request.app.state.db_manager
            )
            items = service.list_sessions(limit=limit, offset=offset)
            return {"items": items, "limit": limit, "offset": offset}
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"Failed to list demo sessions: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to list demo sessions."
            ) from e
        except Exception as e:
            logger.error(f"Failed to list demo sessions: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to list demo sessions."
            ) from e

    @application.get("/demo/house-price/sessions/{session_id}", tags=["demo"])
    def demo_house_price_get_session(session_id: str, request: Request):
        """Returns one demo session with its monitoring snapshot."""
        try:
            service = create_house_price_demo_service(
                db_manager=request.app.state.db_manager
            )
            session = service.get_session(session_id)
            if session is None:
                raise HTTPException(status_code=404, detail="Demo session not found.")
            return session
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"Failed to retrieve demo session: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve demo session."
            ) from e
        except Exception as e:
            logger.error(f"Failed to retrieve demo session: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve demo session."
            ) from e

    @application.post(
        "/demo/house-price/sessions/{session_id}/generate", tags=["demo"]
    )
    def demo_house_price_generate(session_id: str, request: Request):
        """Generates deterministic isolated demo samples for a session.

        Persists rows with ``source='demo'`` only; the production
        prediction count is unchanged. Regenerating replaces the
        session's previous demo rows.
        """
        try:
            service = create_house_price_demo_service(
                db_manager=request.app.state.db_manager
            )
            return service.generate_samples(session_id)
        except AdaptiveAIError as e:
            logger.warning(f"Demo generation failed: {e}")
            status = 404 if "not found" in str(e).lower() else 422
            raise HTTPException(status_code=status, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Demo generation failed: {e}")
            raise HTTPException(
                status_code=500, detail="Demo generation failed."
            ) from e

    @application.post(
        "/demo/house-price/sessions/{session_id}/monitor", tags=["demo"]
    )
    def demo_house_price_monitor(session_id: str, request: Request):
        """Runs the real Phase 4 health check over demo traffic only.

        Reuses the production monitoring engine scoped to this
        session's rows. Writes NOTHING to ``monitoring_results`` — the
        result is stored on the session record — so production
        monitoring history is unchanged. Also returns the existing
        Phase 5 retraining recommendation (advice only).
        """
        try:
            service = create_house_price_demo_service(
                db_manager=request.app.state.db_manager
            )
            return service.monitor_session(session_id)
        except AdaptiveAIError as e:
            logger.warning(f"Demo monitoring failed: {e}")
            status = 404 if "not found" in str(e).lower() else 422
            raise HTTPException(status_code=status, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Demo monitoring failed: {e}")
            raise HTTPException(
                status_code=500, detail="Demo monitoring failed."
            ) from e

    @application.post(
        "/demo/house-price/sessions/{session_id}/reset", tags=["demo"]
    )
    def demo_house_price_reset(session_id: str, request: Request):
        """Resets a demo session (removes ONLY its simulated rows).

        Never removes production predictions, models, reference data,
        the production pointer, or audit history.
        """
        try:
            service = create_house_price_demo_service(
                db_manager=request.app.state.db_manager
            )
            return service.reset_session(session_id)
        except AdaptiveAIError as e:
            logger.warning(f"Demo reset failed: {e}")
            status = 404 if "not found" in str(e).lower() else 422
            raise HTTPException(status_code=status, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Demo reset failed: {e}")
            raise HTTPException(status_code=500, detail="Demo reset failed.") from e

    @application.post(
        "/demo/house-price/sessions/{session_id}/retrain", tags=["demo"]
    )
    def demo_house_price_retrain(
        session_id: str, payload: DemoRetrainRequest, request: Request
    ):
        """Starts an explicit lifecycle run tagged as demo-triggered.

        Reuses the existing Phase 5 lifecycle (shared training path,
        evaluation, gate) with trigger ``demo:<scenario>:<session>``.
        Never promotes: the production pointer is unchanged and only
        the existing explicit promotion call (with its own warning)
        can move it.
        """
        try:
            service = create_house_price_demo_service(
                db_manager=request.app.state.db_manager
            )
            lifecycle = create_house_price_lifecycle_service(
                db_manager=request.app.state.db_manager
            )
            return service.start_retraining(
                session_id,
                lifecycle,
                production_limit=payload.production_limit,
            )
        except AdaptiveAIError as e:
            logger.warning(f"Demo retraining failed: {e}")
            status = 404 if "not found" in str(e).lower() else 422
            raise HTTPException(status_code=status, detail=str(e)) from e
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Demo retraining failed: {e}")
            raise HTTPException(
                status_code=500, detail="Demo retraining failed."
            ) from e

    @application.get("/predict/house-price/history", tags=["prediction"])
    def predict_house_price_history(
        request: Request,
        limit: int = Query(default=100, ge=1),
        offset: int = Query(default=0, ge=0),
    ):
        """Returns persisted House Price prediction history.

        Thin read adapter over PredictionRepository for the
        house_price use case. Triggers no prediction.
        """
        try:
            repo: PredictionRepository = request.app.state.prediction_repo
            repo.init()
            items = repo.list_predictions(
                limit=limit,
                offset=offset,
                use_case="house_price",
                source="production",
            )
            return {"items": items, "limit": limit, "offset": offset}
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"Failed to retrieve prediction history: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve prediction history."
            ) from e
        except Exception as e:
            logger.error(f"Failed to retrieve prediction history: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve prediction history."
            ) from e

    @application.get("/predict/house-price/benchmarks", tags=["prediction"])
    def predict_house_price_benchmarks(
        location: Optional[str] = Query(default=None),
    ):
        """Returns Pune locality market benchmarks (Phase 2).

        Thin read adapter over the benchmark service (single source:
        Phase 1 calibration). Without ``location`` returns all 12
        supported localities; with ``location`` returns that locality
        only (unknown localities are a 422). Benchmarks are market
        calibration references, not predictions.
        """
        try:
            from app.use_cases.house_price.benchmarks import (
                get_all_benchmarks,
                get_locality_benchmark,
            )
            from app.use_cases.house_price.market_config import (
                DATA_SCHEMA_VERSION,
            )

            if location is None:
                items = get_all_benchmarks()
            else:
                items = [get_locality_benchmark(location)]
            return {
                "items": items,
                "schema_version": int(DATA_SCHEMA_VERSION),
            }
        except DataValidationError as e:
            logger.warning(f"House Price benchmark validation failed: {e}")
            raise HTTPException(status_code=422, detail="Unknown location.") from e
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"Failed to retrieve benchmarks: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve benchmarks."
            ) from e
        except Exception as e:
            logger.error(f"Failed to retrieve benchmarks: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve benchmarks."
            ) from e

    @application.get("/models/house-price/production", tags=["models"])
    def models_house_price_production():
        """Returns the production model specification with metrics.

        Thin adapter over the production info service. Performs no
        training and writes nothing.
        """
        try:
            from app.use_cases.house_price.production_info import (
                get_house_price_production_info,
            )

            return get_house_price_production_info(models_dir=HOUSE_PRICE_MODELS_DIR)
        except ModelNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"Failed to retrieve production info: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve production info."
            ) from e
        except Exception as e:
            logger.error(f"Failed to retrieve production info: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve production info."
            ) from e

    @application.get("/models/house-price/registry", tags=["models"])
    def models_house_price_registry(
        limit: int = Query(default=100, ge=1),
    ):
        """Returns production pointer plus version history.

        Thin adapter over the registry info service. Read-only.
        """
        try:
            from app.retraining.registry_info import get_house_price_registry_info

            return get_house_price_registry_info(
                limit=limit, models_dir=HOUSE_PRICE_MODELS_DIR
            )
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"Failed to retrieve registry info: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve registry info."
            ) from e
        except Exception as e:
            logger.error(f"Failed to retrieve registry info: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve registry info."
            ) from e

    @application.post("/predict/house-price/explain", tags=["prediction"])
    def predict_house_price_explain(payload: HousePriceRequest):
        """Explains one prediction with linear contributions.

        Thin adapter over the explain service using the current
        production artifact. Performs no persistence.
        """
        try:
            from app.use_cases.house_price.explain import (
                explain_house_price_prediction,
            )

            return explain_house_price_prediction(
                {
                    "area_sqft": payload.area_sqft,
                    "bedrooms": payload.bedrooms,
                    "bathrooms": payload.bathrooms,
                    "parking": payload.parking,
                    "location": payload.location,
                    "property_age": payload.property_age,
                    "floor_number": payload.floor_number,
                    "total_floors": payload.total_floors,
                    "furnishing_status": payload.furnishing_status,
                    "property_type": payload.property_type,
                },
                models_dir=HOUSE_PRICE_MODELS_DIR,
            )
        except ModelNotFoundError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"Failed to explain prediction: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to explain prediction."
            ) from e
        except Exception as e:
            logger.error(f"Failed to explain prediction: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to explain prediction."
            ) from e

    @application.get("/monitor/house-price/distributions", tags=["monitoring"])
    def monitor_house_price_distributions(
        limit: int = Query(default=1000, ge=1),
    ):
        """Returns reference vs production distributions for charts.

        Thin adapter over the distribution service. Read-only and
        performs no drift decision here.
        """
        try:
            from app.monitoring.distribution_service import (
                get_house_price_distributions,
            )

            return get_house_price_distributions(
                production_limit=limit, source="production"
            )
        except HTTPException:
            raise
        except AdaptiveAIError as e:
            logger.error(f"Failed to retrieve distributions: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve distributions."
            ) from e
        except Exception as e:
            logger.error(f"Failed to retrieve distributions: {e}")
            raise HTTPException(
                status_code=500, detail="Failed to retrieve distributions."
            ) from e

    return application


app = create_app()
