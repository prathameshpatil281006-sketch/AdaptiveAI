"""Retraining orchestration foundation for AdaptiveAI (Steps 30 + 34 + 36).

Reusable layer that trains/evaluates model candidates for a use case
and returns a structured candidate result:

    use_case
        |
        v
    load/validate training data
        |
        v
    split data
        |
        v
    candidate models (use_case.get_model_candidates)
        |
        v
    MLEngine.train_and_evaluate (once per candidate)
        |
        v
    MLflowTracker.track_candidate (once per candidate)
        |
        v
    select best candidate (highest R2 for regression)
        |
        v
    structured retraining result (no registry, no persistence by default)
        |
        v
    optional promotion phase (only when promote=True with a promoter)
        |
        v
    optional audit persistence (only with an injected history repository)

Default behavior (promote=False) is unchanged from Step 30: no
evaluation of production, no persistence, no registry versions, no
SQLite work. An opt-in promotion phase (Step 34) evaluates the best
candidate against the current production artifact via the existing
evaluator and delegates any actual promotion exclusively to the
existing promoter. An opt-in audit phase (Step 36) records the
promoter's structured outcome via an injected history repository. The
service never performs persistence or registry writes itself and never
opens database connections directly.

The service performs no CSV/SQL/scheduler/API/dashboard work itself
beyond loading the training CSV via pandas. It touches SQLite only
through an explicitly injected history repository.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd

from app.core.config import resolve_project_path
from app.core.exceptions import AdaptiveAIError, ModelPromotionError, RetrainingError
from app.engine.ml_engine import MLEngine
from app.engine.mlflow_tracker import (
    HOUSE_PRICE_EXPERIMENT_NAME,
    HOUSE_PRICE_USE_CASE_NAME,
    TRAINING_CONTEXT_RETRAINING,
    MLflowTracker,
)
from app.retraining.model_evaluator import ModelEvaluator
from app.retraining.model_promoter import ModelPromoter
from app.use_cases.base import BaseUseCase

logger = logging.getLogger(__name__)

# Absolute project-root anchored default (Step 56): retraining resolves the
# same reference dataset regardless of the process working directory.
DEFAULT_DATA_PATH = str(
    resolve_project_path(
        Path("data") / "reference" / "house_price_reference.csv"
    )
)


def _default_preprocessor_factory(use_case: BaseUseCase):
    """Resolves the preprocessor for a known use case using existing utilities."""
    if use_case.name == "house_price":
        from app.use_cases.house_price.preprocessing import get_preprocessor

        return get_preprocessor()
    raise RetrainingError(
        f"Unsupported use case for retraining: '{use_case.name}'. "
        "Provide a preprocessor_factory for custom use cases."
    )


def _default_regressor_factory(model_type: str, hyperparameters: Dict[str, Any]):
    """Instantiates a regressor via the authoritative House Price path.

    Step 60 (I-11): delegates to the single shared factory in
    ``app.use_cases.house_price.training`` so orchestration cannot
    diverge from selection/legacy training. This avoids duplicating ML
    algorithms inside RetrainingService.
    """
    from app.use_cases.house_price.training import create_regressor

    return create_regressor(model_type, hyperparameters)


def _load_training_dataframe(data_path: Union[str, Path, pd.DataFrame]) -> pd.DataFrame:
    """Loads training data from a CSV path or DataFrame.

    Step 60 (I-11): delegates to the authoritative shared loader so all
    House Price paths resolve/load identically. Maps the shared
    TrainingError contract to RetrainingError to preserve this
    service's public error behavior.

    Relative CSV paths resolve against the project root (Step 56 path
    contract); absolute paths pass through unchanged.
    """
    from app.core.exceptions import TrainingError
    from app.use_cases.house_price.training import load_house_price_dataframe

    try:
        return load_house_price_dataframe(data_path)
    except RetrainingError:
        raise
    except TrainingError as e:
        message = str(e)
        if "not found" in message.lower():
            # Preserve the legacy "Training data not found" wording.
            raise RetrainingError(message) from e
        if "invalid data_path type" in message.lower():
            raise RetrainingError(
                f"Invalid data_path type: {type(data_path).__name__}. "
                "Expected a CSV path or pandas DataFrame."
            ) from e
        raise RetrainingError(f"Failed to load training data: {e}") from e


def _split_training_data(use_case: BaseUseCase, df: pd.DataFrame):
    """Validates and splits training data via the authoritative path.

    Step 60 (I-11): delegates to the shared House Price split (same
    validation, test_size, and seed as legacy training/selection).
    """
    if use_case.name == "house_price":
        from app.use_cases.house_price.training import prepare_house_price_splits

        try:
            return prepare_house_price_splits(df)
        except RetrainingError:
            raise
        except Exception as e:
            message = str(e)
            lowered = message.lower()
            if "missing" in lowered or "duplicate" in lowered or "must be" in lowered or "non-numeric" in lowered:
                raise RetrainingError(f"Training data validation failed: {e}") from e
            # split_house_price_data validates first, so validation
            # failures surface here; anything else is a split failure.
            # Distinguish by attempting to classify DataValidationError.
            from app.core.exceptions import DataValidationError

            if isinstance(e, DataValidationError):
                raise RetrainingError(f"Training data validation failed: {e}") from e
            raise RetrainingError(f"Failed to split training data: {e}") from e
    raise RetrainingError(
        f"Unsupported use case for retraining: '{use_case.name}'. "
        "Only 'house_price' is supported in this step."
    )


class RetrainingService:
    """Use-case-agnostic orchestration for candidate retraining."""

    def __init__(
        self,
        engine: Optional[MLEngine] = None,
        tracker: Optional[MLflowTracker] = None,
        preprocessor_factory: Optional[Callable[[BaseUseCase], Any]] = None,
        regressor_factory: Optional[Callable[[str, Dict[str, Any]], Any]] = None,
        evaluator: Optional[Any] = None,
        promoter: Optional[Any] = None,
        promotion_repository: Optional[Any] = None,
    ) -> None:
        """Wires the retraining dependencies with injection for testing.

        Args:
            engine: MLEngine used for train/evaluate. Defaults to MLEngine().
            tracker: MLflowTracker used for per-candidate tracking.
                Defaults to an MLflowTracker on a retraining experiment.
                Tests should inject a tracker with a temporary tracking URI.
            preprocessor_factory: Optional callable resolving a preprocessor
                for a use case. Defaults to existing House Price preprocessing.
            regressor_factory: Optional callable instantiating a regressor
                from (model_type, hyperparameters). Defaults to reusing the
                existing House Price candidate mapping.
            evaluator: Optional candidate-vs-production evaluator exposing
                ``evaluate_candidate(...)``. Defaults to None (a default
                evaluator is created lazily only when promote=True).
                Tests should inject a stub to avoid filesystem access.
            promoter: Optional promotion executor exposing
                ``promote(...)``. Defaults to None (promotion disabled).
                Required when retrain() is called with promote=True.
                Tests should inject a stub to avoid real side effects.
            promotion_repository: Optional audit store exposing
                ``save_promotion_result(...)``. Defaults to None (no audit
                persistence). Used only when promote=True and a promoter
                returned a structured outcome. Never instantiated
                automatically; tests should inject a stub or a repository
                bound to a temporary database.
        """
        if engine is not None and not isinstance(engine, MLEngine):
            raise RetrainingError("engine must be an MLEngine instance or None.")
        if tracker is not None and not isinstance(tracker, MLflowTracker):
            raise RetrainingError("tracker must be an MLflowTracker instance or None.")
        if preprocessor_factory is not None and not callable(preprocessor_factory):
            raise RetrainingError("preprocessor_factory must be callable or None.")
        if regressor_factory is not None and not callable(regressor_factory):
            raise RetrainingError("regressor_factory must be callable or None.")
        if evaluator is not None:
            evaluate_fn = getattr(evaluator, "evaluate_candidate", None)
            if not callable(evaluate_fn):
                raise RetrainingError(
                    "evaluator must expose a callable evaluate_candidate() or be None."
                )
        if promoter is not None:
            promote_fn = getattr(promoter, "promote", None)
            if not callable(promote_fn):
                raise RetrainingError(
                    "promoter must expose a callable promote() or be None."
                )
        if promotion_repository is not None:
            save_fn = getattr(promotion_repository, "save_promotion_result", None)
            if not callable(save_fn):
                raise RetrainingError(
                    "promotion_repository must expose a callable "
                    "save_promotion_result() or be None."
                )
        self._engine = engine if engine is not None else MLEngine()
        self._tracker = (
            tracker
            if tracker is not None
            else MLflowTracker(experiment_name=HOUSE_PRICE_EXPERIMENT_NAME)
        )
        self._preprocessor_factory = preprocessor_factory or _default_preprocessor_factory
        self._regressor_factory = regressor_factory or _default_regressor_factory
        self._evaluator = evaluator
        self._promoter = promoter
        self._promotion_repository = promotion_repository

    @property
    def engine(self) -> MLEngine:
        """Returns the injected MLEngine."""
        return self._engine

    @property
    def tracker(self) -> MLflowTracker:
        """Returns the injected MLflowTracker."""
        return self._tracker

    @property
    def evaluator(self) -> Optional[Any]:
        """Returns the injected evaluator (None if not configured)."""
        return self._evaluator

    @property
    def promoter(self) -> Optional[Any]:
        """Returns the injected promoter (None if promotion disabled)."""
        return self._promoter

    @property
    def promotion_repository(self) -> Optional[Any]:
        """Returns the injected audit repository (None if not configured)."""
        return self._promotion_repository

    def retrain(
        self,
        use_case: BaseUseCase,
        data_path: Union[str, Path, pd.DataFrame] = DEFAULT_DATA_PATH,
        promote: bool = False,
        production_model_name: Optional[str] = None,
        production_model: Optional[Any] = None,
        models_dir: Optional[Union[str, Path]] = None,
        allow_initial_promotion: bool = False,
    ) -> Dict[str, Any]:
        """Trains/evaluates every candidate and returns the retraining result.

        Args:
            use_case: Use-case abstraction providing candidates/metadata.
            data_path: CSV path or DataFrame with training data.
            promote: Opt-in promotion phase. Defaults to False (legacy
                behavior: train/track/select only, no production
                evaluation and no promotion side effects).
            production_model_name: Identity of the production artifact to
                compare against (``<name>.joblib`` convention). When both
                this and ``production_model`` are None, the best candidate
                name is reused for lookup; a missing artifact preserves
                the evaluator's "no production" semantics and never
                auto-promotes.
            production_model: Already-loaded production pipeline for tests.
                Takes precedence over name-based loading when provided.
            models_dir: Optional production directory override for the
                promotion phase. ``None`` uses the default location.
            allow_initial_promotion: Explicit opt-in for first deployment
                (Step 55). Only meaningful with promote=True. When the
                evaluator reports no production model and this flag is
                set, the best candidate is deployed through the
                promoter's explicit initial-deployment operation
                (preserving its run identity) and the production serving
                reference is published. Defaults to False, which
                preserves the legacy safe behavior: no production model
                yields ``requires_initial_promotion`` with zero
                production side effects.

        Returns:
            Dict with use_case, best_model_name, best_pipeline,
            best_metrics {mae, rmse, r2}, all_results
            [{model_name, metrics, run_id}], and best_run_id. When
            promote=True, an additional ``promotion`` key holds the actual
            promoter result (propagated unchanged, never reconstructed).
            When promote=True with an injected history repository, an
            additional ``promotion_history_id`` key holds the persisted
            audit record id.

        Raises:
            RetrainingError: If loading, validation, training, or tracking
                fails, if promote=True without a promoter, if production
                evaluation fails, if promotion fails (with the
                promotion failure chained as cause and candidate identity
                retained in the message), or if audit persistence fails
                (with the underlying failure chained as cause).
        """
        if not isinstance(use_case, BaseUseCase):
            raise RetrainingError("use_case must be a BaseUseCase instance.")
        if not isinstance(allow_initial_promotion, bool):
            raise RetrainingError("allow_initial_promotion must be a boolean.")
        if promote and self._promoter is None:
            raise RetrainingError(
                "A promoter is required when promote=True. "
                "Inject one via RetrainingService(promoter=...)."
            )

        try:
            df = _load_training_dataframe(data_path)
        except RetrainingError:
            raise
        except Exception as e:
            raise RetrainingError(f"Failed to load training data: {e}") from e

        try:
            X_train, X_test, y_train, y_test = _split_training_data(use_case, df)
        except RetrainingError:
            raise
        except Exception as e:
            raise RetrainingError(f"Failed to prepare training data: {e}") from e

        try:
            candidates = use_case.get_model_candidates()
        except Exception as e:
            raise RetrainingError(f"Failed to obtain model candidates: {e}") from e
        if not isinstance(candidates, list) or not candidates:
            raise RetrainingError("No candidate models were configured for retraining.")

        all_results: List[Dict[str, Any]] = []
        pipelines: Dict[str, Any] = {}
        run_ids: Dict[str, str] = {}
        metrics_by_name: Dict[str, Dict[str, float]] = {}

        for candidate in candidates:
            if not isinstance(candidate, dict):
                raise RetrainingError("Each model candidate must be a dict.")
            model_name = candidate.get("model_type")
            if not model_name or not str(model_name).strip():
                raise RetrainingError("Each candidate must define a 'model_type'.")
            model_name = str(model_name)
            hyperparams = candidate.get("hyperparameters", {})
            if hyperparams is None:
                hyperparams = {}
            if not isinstance(hyperparams, dict):
                raise RetrainingError(
                    f"Hyperparameters for '{model_name}' must be a dict."
                )

            try:
                regressor = self._regressor_factory(model_name, dict(hyperparams))
            except RetrainingError:
                raise
            except Exception as e:
                raise RetrainingError(
                    f"Failed to create candidate '{model_name}': {e}"
                ) from e

            try:
                preprocessor = self._preprocessor_factory(use_case)
            except RetrainingError:
                raise
            except Exception as e:
                raise RetrainingError(
                    f"Failed to resolve preprocessor for '{use_case.name}': {e}"
                ) from e

            try:
                engine_output = self._train_candidate_via_shared_path(
                    X_train=X_train,
                    X_test=X_test,
                    y_train=y_train,
                    y_test=y_test,
                    model_name=model_name,
                    hyperparams=dict(hyperparams),
                    preprocessor=preprocessor,
                    regressor=regressor,
                )
            except Exception as e:
                if isinstance(e, RetrainingError):
                    raise
                raise RetrainingError(
                    f"Training failed for candidate '{model_name}': {e}"
                ) from e

            try:
                pipeline = engine_output["pipeline"]
                metrics = engine_output["metrics"]
                # Re-validate the shared-path output shape so a future
                # divergence fails closed with the legacy message.
                metrics = {
                    "mae": float(metrics["mae"]),
                    "rmse": float(metrics["rmse"]),
                    "r2": float(metrics["r2"]),
                }
            except Exception as e:
                raise RetrainingError(
                    f"Invalid training output for candidate '{model_name}': {e}"
                ) from e

            try:
                run_id = self._tracker.track_candidate(
                    model_type=model_name,
                    hyperparameters=dict(hyperparams),
                    metrics=dict(metrics),
                    pipeline=pipeline,
                    run_name=model_name,
                    use_case=HOUSE_PRICE_USE_CASE_NAME,
                    training_context=TRAINING_CONTEXT_RETRAINING,
                    extra_tags={"candidate_role": "candidate"},
                )
            except Exception as e:
                if isinstance(e, RetrainingError):
                    raise
                raise RetrainingError(
                    f"MLflow tracking failed for candidate '{model_name}': {e}"
                ) from e

            self._add_lineage_tags(run_id, use_case.name, model_name)

            all_results.append(
                {
                    "model_name": model_name,
                    "metrics": dict(metrics),
                    "run_id": str(run_id),
                }
            )
            pipelines[model_name] = pipeline
            run_ids[model_name] = str(run_id)
            metrics_by_name[model_name] = dict(metrics)

        if not all_results:
            raise RetrainingError("No candidate models were evaluated.")

        # Deterministic best selection: highest R2 (first max wins).
        best_entry = max(all_results, key=lambda item: item["metrics"]["r2"])
        best_model_name = best_entry["model_name"]
        logger.info(
            "Retraining completed for use case '%s': best=%s r2=%.4f",
            use_case.name,
            best_model_name,
            best_entry["metrics"]["r2"],
        )
        # Step 61 (I-12): mark the winning run without altering run_id
        # lineage (best-effort; tracking-only policy unchanged).
        try:
            add_tags = getattr(self._tracker, "add_tags_to_run", None)
            if callable(add_tags):
                add_tags(run_ids[best_model_name], {"candidate_role": "best"})
        except Exception:
            pass
        retraining_result: Dict[str, Any] = {
            "use_case": use_case.name,
            "best_model_name": best_model_name,
            "best_pipeline": pipelines[best_model_name],
            "best_metrics": dict(metrics_by_name[best_model_name]),
            "all_results": all_results,
            "best_run_id": run_ids[best_model_name],
        }

        if not promote:
            return retraining_result

        return self._run_promotion_phase(
            retraining_result=retraining_result,
            use_case_name=use_case.name,
            X_test=X_test,
            y_test=y_test,
            production_model_name=production_model_name,
            production_model=production_model,
            models_dir=models_dir,
            allow_initial_promotion=allow_initial_promotion,
        )

    def _resolve_production_model_name(
        self,
        best_model_name: str,
        models_dir: Optional[Union[str, Path]],
    ) -> Optional[str]:
        """Resolves which production artifact to compare against.

        Prefers the model designated by the production serving
        reference (Step 55 single source of truth) so a promoted
        candidate under a different name is compared against the live
        production model instead of a same-named file. Falls back to
        the best candidate name (legacy behavior) when no production
        reference exists. A corrupt reference fails closed.
        """
        from app.core.exceptions import ModelNotFoundError
        from app.use_cases.house_price.model_persistence import (
            read_production_pointer,
        )

        try:
            pointer = read_production_pointer(models_dir=models_dir)
        except ModelNotFoundError:
            return best_model_name
        except Exception as e:
            raise RetrainingError(
                "Production serving reference is unreadable; refusing to "
                f"evaluate against an ambiguous production model: {e}"
            ) from e
        designated = pointer.get("model_name")
        if not isinstance(designated, str) or not designated.strip():
            raise RetrainingError(
                "Production serving reference is missing model_name; refusing "
                "to evaluate against an ambiguous production model."
            )
        return designated.strip()

    def _run_promotion_phase(
        self,
        retraining_result: Dict[str, Any],
        use_case_name: str,
        X_test: Any,
        y_test: Any,
        production_model_name: Optional[str],
        production_model: Optional[Any],
        models_dir: Optional[Union[str, Path]],
        allow_initial_promotion: bool = False,
    ) -> Dict[str, Any]:
        """Evaluates the best candidate and delegates to the promoter.

        Only the promoter performs writes. A missing production artifact
        preserves the evaluator's "no production" semantics and is never
        auto-promoted unless the caller explicitly opted into first
        deployment via allow_initial_promotion. Promotion failures raise
        (never reported as success) with candidate identity retained for
        context. When a history repository is injected, the promoter's
        structured outcome (including non-promoting outcomes) is recorded
        and its record id attached; audit failures raise instead of
        reporting false success.
        """
        if self._promoter is None:  # defensive; checked earlier as well
            raise RetrainingError(
                "A promoter is required when promote=True. "
                "Inject one via RetrainingService(promoter=...)."
            )
        best_pipeline = retraining_result["best_pipeline"]
        best_model_name = retraining_result["best_model_name"]
        best_run_id = retraining_result["best_run_id"]

        if self._evaluator is None:
            evaluator = ModelEvaluator()
        else:
            evaluator = self._evaluator

        resolved_name = production_model_name
        if production_model is None and resolved_name is None:
            resolved_name = self._resolve_production_model_name(
                best_model_name, models_dir
            )

        try:
            evaluation_result = evaluator.evaluate_candidate(
                candidate_model=best_pipeline,
                production_model=production_model,
                X_test=X_test,
                y_test=y_test,
                candidate_model_name=best_model_name,
                production_model_name=resolved_name,
                models_dir=models_dir,
            )
        except RetrainingError:
            raise
        except Exception as e:
            raise RetrainingError(f"Production evaluation failed: {e}") from e

        use_initial_path = bool(
            allow_initial_promotion
            and isinstance(evaluation_result, dict)
            and evaluation_result.get("status") == "no_production_model"
        )
        try:
            if use_initial_path:
                deploy_fn = getattr(
                    self._promoter, "deploy_initial_candidate", None
                )
                if not callable(deploy_fn):
                    raise RetrainingError(
                        "The configured promoter does not support explicit "
                        "initial deployment; refusing to deploy without the "
                        "initial-deployment operation."
                    )
                promotion_result = deploy_fn(
                    candidate_model=best_pipeline,
                    evaluation_result=evaluation_result,
                    candidate_model_name=best_model_name,
                    candidate_run_id=best_run_id,
                    production_model_name=resolved_name,
                    models_dir=models_dir,
                )
            else:
                promotion_result = self._promoter.promote(
                    candidate_model=best_pipeline,
                    evaluation_result=evaluation_result,
                    candidate_model_name=best_model_name,
                    candidate_run_id=best_run_id,
                    production_model_name=resolved_name,
                    models_dir=models_dir,
                )
        except ModelPromotionError as e:
            raise RetrainingError(
                f"Promotion failed for candidate '{best_model_name}' "
                f"(run {best_run_id}); candidate retained: {e}"
            ) from e
        except Exception as e:
            raise RetrainingError(
                f"Promotion failed for candidate '{best_model_name}' "
                f"(run {best_run_id}); candidate retained: {e}"
            ) from e

        retraining_result["promotion"] = promotion_result

        if isinstance(promotion_result, dict) and promotion_result.get("decision") == "promoted":
            publish_fn = getattr(
                self._promoter, "update_production_pointer", None
            )
            if callable(publish_fn):
                try:
                    publish_fn(
                        promotion_result,
                        models_dir=models_dir,
                        use_case=use_case_name,
                    )
                except ModelPromotionError as e:
                    raise RetrainingError(
                        f"Production publication failed for candidate "
                        f"'{best_model_name}' (run {best_run_id}); serving "
                        f"reference unchanged: {e}"
                    ) from e
                except Exception as e:
                    raise RetrainingError(
                        f"Production publication failed for candidate "
                        f"'{best_model_name}' (run {best_run_id}); serving "
                        f"reference unchanged: {e}"
                    ) from e

        if self._promotion_repository is not None:
            try:
                history_id = self._promotion_repository.save_promotion_result(
                    use_case=use_case_name,
                    promotion_result=promotion_result,
                )
            except AdaptiveAIError as e:
                raise RetrainingError(
                    f"Promotion history persistence failed for candidate "
                    f"'{best_model_name}' (run {best_run_id}); "
                    f"promotion outcome not recorded: {e}"
                ) from e
            except Exception as e:
                raise RetrainingError(
                    f"Promotion history persistence failed for candidate "
                    f"'{best_model_name}' (run {best_run_id}); "
                    f"promotion outcome not recorded: {e}"
                ) from e
            retraining_result["promotion_history_id"] = history_id

        return retraining_result

    def _train_candidate_via_shared_path(
        self,
        X_train,
        X_test,
        y_train,
        y_test,
        model_name: str,
        hyperparams: Dict[str, Any],
        preprocessor: Any,
        regressor: Any,
    ) -> Dict[str, Any]:
        """Trains one candidate via the authoritative shared path.

        Step 60 (I-11): single engine-interaction implementation shared
        with legacy training and selection (metrics normalization and
        pipeline construction cannot diverge). Uses the injected engine
        so tests with stub engines keep working.
        """
        from app.core.exceptions import TrainingError
        from app.use_cases.house_price.training import (
            train_candidate_with_instances,
        )

        try:
            return train_candidate_with_instances(
                X_train,
                X_test,
                y_train,
                y_test,
                model_name=model_name,
                hyperparameters=dict(hyperparams),
                preprocessor=preprocessor,
                regressor=regressor,
                engine=self._engine,
            )
        except RetrainingError:
            raise
        except TrainingError as e:
            # Preserve the legacy "Invalid training output" contract
            # without it being re-wrapped as a generic training failure.
            if "Invalid training output" in str(e):
                raise RetrainingError(str(e)) from e
            raise

    def _add_lineage_tags(
        self,
        run_id: str,
        use_case_name: str,
        model_name: Optional[str] = None,
    ) -> None:
        """Adds coherent lineage tags on a best-effort basis (Step 61 I-12).

        Preserves the legacy ``training_stage``/``use_case`` keys while
        also ensuring the shared ``training_context``/``model_name`` keys
        exist even for trackers that predate inside-run tagging. Never
        changes run_id lineage and never touches the registry.
        """
        try:
            get_client = getattr(self._tracker, "_get_client", None)
            if get_client is None:
                return
            client = get_client()
            client.set_tag(str(run_id), "training_stage", TRAINING_CONTEXT_RETRAINING)
            client.set_tag(str(run_id), "training_context", TRAINING_CONTEXT_RETRAINING)
            client.set_tag(str(run_id), "use_case", str(use_case_name))
            if model_name is not None and str(model_name).strip():
                client.set_tag(str(run_id), "model_name", str(model_name).strip())
        except Exception as e:
            logger.warning(f"Failed to set retraining lineage tags: {e}")


def create_house_price_retraining_service(
    engine: Optional[MLEngine] = None,
    tracker: Optional[MLflowTracker] = None,
    evaluator: Optional[Any] = None,
    promoter: Optional[Any] = None,
    promotion_repository: Optional[Any] = None,
) -> RetrainingService:
    """Small factory wiring RetrainingService for House Price.

    Reuses existing House Price data/validation/split/preprocessing and
    candidate definitions through the default factories.
    """
    return RetrainingService(
        engine=engine,
        tracker=tracker,
        evaluator=evaluator,
        promoter=promoter,
        promotion_repository=promotion_repository,
    )
