"""Registry info service for Model Registry page (Phase 5 extended).

Aggregates real backend state without inventing versions:

- production pointer (serving authority) + artifact availability
- promotion history (persisted decisions, newest-first)
- retraining runs (lifecycle candidates with metrics, schema,
  artifact availability, eligibility — newest-first)
- production pointer history (previous → new transitions backing
  validated rollback, newest-first)
- MLflow registry versions for HousePriceModel (best-effort; empty
  when the tracking store is unavailable)

No API, SQL, or HTTP work lives here. The FastAPI layer calls
``get_house_price_registry_info`` as a thin adapter.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _artifact_available(model_name: Any, models_dir: Optional[Union[str, Path]]) -> Optional[bool]:
    """Reports whether a model artifact file exists (None when unnamed)."""
    if not isinstance(model_name, str) or not model_name.strip():
        return None
    try:
        from app.use_cases.house_price.model_persistence import _get_default_models_dir

        base = Path(models_dir) if models_dir is not None else Path(_get_default_models_dir())
        return bool((base / f"{model_name.strip()}.joblib").is_file())
    except Exception:
        return None


def get_house_price_registry_info(
    limit: int = 100,
    db_manager: Any = None,
    models_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Returns production + runs + pointer history + version history."""
    from app.core.exceptions import AdaptiveAIError
    from app.database.database import DatabaseManager
    from app.database.lifecycle_repository import (
        PointerHistoryRepository,
        RetrainingRunRepository,
    )
    from app.database.promotion_repository import PromotionRepository
    from app.use_cases.house_price.model_persistence import read_production_pointer

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise AdaptiveAIError("limit must be a positive integer.")

    try:
        pointer = read_production_pointer(models_dir=models_dir)
    except Exception:
        pointer = None

    manager = db_manager if db_manager is not None else DatabaseManager()

    versions: List[Dict[str, Any]] = []
    try:
        repo = PromotionRepository(manager)
        repo.init()
        items = repo.list_promotion_history(
            limit=limit, offset=0, use_case="house_price"
        )
        # Keep promoted + rejected + initial records; newest-first already.
        versions = items
    except Exception as e:
        logger.warning(f"Failed to read promotion history: {e}")
        versions = []

    runs: List[Dict[str, Any]] = []
    try:
        run_repo = RetrainingRunRepository(manager)
        run_repo.init()
        for run in run_repo.list_runs(limit=limit, offset=0, use_case="house_price"):
            runs.append(_run_summary(run, models_dir))
    except Exception as e:
        logger.warning(f"Failed to read retraining runs: {e}")
        runs = []

    pointer_history: List[Dict[str, Any]] = []
    try:
        history_repo = PointerHistoryRepository(manager)
        history_repo.init()
        pointer_history = history_repo.list_transitions(
            limit=limit, offset=0, use_case="house_price"
        )
    except Exception as e:
        logger.warning(f"Failed to read pointer history: {e}")
        pointer_history = []

    mlflow_versions: List[Dict[str, Any]] = []
    try:
        import mlflow
        from mlflow.tracking import MlflowClient

        from app.core.config import resolve_project_path

        tracking_path = resolve_project_path("mlruns")
        try:
            mlflow.set_tracking_uri(f"file:{tracking_path}")
        except Exception:
            pass
        client = MlflowClient()
        try:
            registered = client.search_registered_models(
                filter_string="name='HousePriceModel'"
            )
            if registered:
                model = registered[0]
                for v in getattr(model, "latest_versions", []) or []:
                    mlflow_versions.append(
                        {
                            "name": getattr(v, "name", "HousePriceModel"),
                            "version": getattr(v, "version", ""),
                            "run_id": getattr(v, "run_id", ""),
                            "status": getattr(v, "status", ""),
                            "stage": getattr(v, "current_stage", ""),
                        }
                    )
        except Exception as e:
            logger.warning(f"MLflow registry lookup failed: {e}")
    except Exception as e:
        logger.warning(f"MLflow unavailable for registry info: {e}")

    production: Optional[Dict[str, Any]] = None
    if isinstance(pointer, dict):
        production = {
            "model_name": pointer.get("model_name"),
            "model_version": pointer.get("model_version"),
            "registered_model_name": pointer.get("registered_model_name", "HousePriceModel"),
            "run_id": pointer.get("run_id"),
            "deployed_at": pointer.get("deployed_at"),
            "artifact_available": _artifact_available(
                pointer.get("model_name"), models_dir
            ),
        }

    return {
        "production": production,
        "runs": runs,
        "pointer_history": pointer_history,
        "versions": versions,
        "mlflow_versions": mlflow_versions,
        "total": len(versions),
    }


def _run_summary(run: Dict[str, Any], models_dir: Optional[Union[str, Path]]) -> Dict[str, Any]:
    """Projects a run row to the registry-facing candidate summary."""
    best_metrics = run.get("best_metrics") if isinstance(run.get("best_metrics"), dict) else None
    metrics = None
    if isinstance(best_metrics, dict):
        metrics = {
            key: best_metrics.get(key)
            for key in ("mae", "rmse", "r2")
            if isinstance(best_metrics.get(key), (int, float))
            and not isinstance(best_metrics.get(key), bool)
        } or None
    comparison = run.get("comparison") if isinstance(run.get("comparison"), dict) else None
    deltas = None
    if isinstance(comparison, dict) and isinstance(comparison.get("deltas"), dict):
        deltas = dict(comparison["deltas"])
    return {
        "run_id": run.get("run_id"),
        "status": run.get("status"),
        "started_at": run.get("started_at"),
        "completed_at": run.get("completed_at"),
        "trigger": run.get("trigger"),
        "dataset_rows": run.get("dataset_rows"),
        "schema_version": run.get("schema_version"),
        "candidate_model": run.get("best_model"),
        "candidate_metrics": metrics,
        "candidate_run_id": run.get("candidate_run_id"),
        "candidate_version": run.get("candidate_version"),
        "candidate_artifact": run.get("candidate_artifact"),
        "artifact_available": _staged_artifact_available(
            run.get("candidate_artifact"), models_dir
        ),
        "comparison_deltas": deltas,
        "promotion_eligible": run.get("promotion_eligible"),
        "promotion_decision": run.get("promotion_decision"),
        "failure_reason": run.get("failure_reason"),
    }


def _staged_artifact_available(
    artifact_name: Any, models_dir: Optional[Union[str, Path]]
) -> Optional[bool]:
    """Reports whether a staged candidate file exists (None when unnamed)."""
    if not isinstance(artifact_name, str) or not artifact_name.strip():
        return None
    try:
        from app.use_cases.house_price.model_persistence import _get_default_models_dir

        base = Path(models_dir) if models_dir is not None else Path(_get_default_models_dir())
        return bool((base / f"{artifact_name.strip()}.joblib").is_file())
    except Exception:
        return None
