import logging
from typing import Dict, Any, Union, Optional
import pandas as pd
from sklearn.pipeline import Pipeline

from app.core.exceptions import TrainingError
from app.engine.ml_engine import MLEngine
from app.engine.mlflow_tracker import (
    HOUSE_PRICE_EXPERIMENT_NAME,
    HOUSE_PRICE_USE_CASE_NAME,
    TRAINING_CONTEXT_SELECTION,
    MLflowTracker,
)
from app.use_cases.house_price.training import (
    create_regressor as _authoritative_create_regressor,
    evaluate_all_candidates,
    get_house_price_candidates,
    prepare_house_price_data,
    select_best_candidate,
)

logger = logging.getLogger(__name__)


def _create_regressor(model_type: str, hyperparameters: Dict[str, Any]):
    """Instantiates a regressor model based on model type and hyperparameters.

    Backwards-compatible alias for the authoritative factory in
    ``app.use_cases.house_price.training``. Preserved so existing
    callers keep working; new code should import from the shared path.
    """
    return _authoritative_create_regressor(model_type, hyperparameters)


def select_best_house_price_model(
    data_path: Union[str, pd.DataFrame] = "data/reference/house_price_reference.csv",
    engine: Optional[MLEngine] = None,
    tracker: Optional[MLflowTracker] = None
) -> Dict[str, Any]:
    """
    Trains and evaluates all candidate models for House Price prediction using
    MLEngine, tracks each candidate experiment in MLflow, and selects the model
    with the highest R² score.

    Step 60 (I-11): data loading, splitting, candidate definitions, and
    per-candidate train/evaluate all delegate to the authoritative shared
    path (``app.use_cases.house_price.training``) so selection shares the
    exact preprocessing, split, seed, candidate configuration, and metrics
    implementation used by legacy training and the orchestration layer.

    Step 61 (I-12): candidate runs use the ONE canonical House Price
    experiment (``HOUSE_PRICE_EXPERIMENT_NAME``) with coherent lineage
    tags (``use_case``, ``model_name``, ``training_context``/
    ``training_stage``, ``candidate_role``); the winning run is marked
    ``best`` without changing any run_id lineage.

    Steps:
    1. Load reference dataset (if string path provided).
    2. Validate dataset integrity.
    3. Split into train and test sets.
    4. Retrieve candidate models from the shared candidate source.
    5. Train each model using MLEngine via the shared evaluation path.
    6. Log model type, hyperparameters, metrics (MAE, RMSE, R²), and pipeline to MLflow.
    7. Select the candidate model with the highest R².
    8. Create NO registry versions (Step 58 registration policy):
       candidate evaluation produces experiment runs only. Registry
       registration happens exclusively through ModelPromoter when a
       candidate is actually promoted/deployed, preserving exact
       run_id lineage at that point.

    Args:
        data_path (Union[str, pd.DataFrame]): Path to CSV file or a DataFrame.
            String paths resolve against the project root (CWD independent).
        engine (Optional[MLEngine]): Optional MLEngine instance.
        tracker (Optional[MLflowTracker]): Optional MLflowTracker instance.

    Returns:
        Dict[str, Any]: Dictionary containing 'best_model_name', 'best_pipeline',
                        and 'all_results'.

    Raises:
        TrainingError: If training, data loading, or evaluation fails.
    """
    logger.info("House Price model selection process started using MLEngine and MLflow.")

    if engine is None:
        engine = MLEngine()

    if tracker is None:
        tracker = MLflowTracker(experiment_name=HOUSE_PRICE_EXPERIMENT_NAME)

    try:
        # 1-3. Load/validate/split via the authoritative shared path
        # (project-root anchored, CWD independent; same seed/split as
        # legacy training and the orchestration layer).
        X_train, X_test, y_train, y_test = prepare_house_price_data(data_path)

        # 4. Fetch candidate models via the shared candidate source
        # (single source: HousePriceUseCase definitions).
        candidates = get_house_price_candidates()

        # 5. Train & evaluate all candidates via the shared path
        # (shared MLEngine invocation + metrics normalization).
        evaluated = evaluate_all_candidates(
            X_train, X_test, y_train, y_test,
            candidates=candidates,
            engine=engine,
        )

        all_results = []
        pipelines: Dict[str, Pipeline] = {}
        run_ids: Dict[str, str] = {}

        # 6. MLflow experiment tracking for candidate (runs only;
        #    registry versions are created exclusively at promotion).
        #    Step 61 (I-12): coherent lineage tags inside the shared
        #    experiment; run_ids retained for best designation.
        for entry in evaluated:
            model_name = entry["model_name"]
            hyperparams = entry["hyperparameters"]
            pipeline = entry["pipeline"]
            metrics = entry["metrics"]
            run_id = tracker.track_candidate(
                model_type=model_name,
                hyperparameters=dict(hyperparams),
                metrics={
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"]
                },
                pipeline=pipeline,
                run_name=model_name,
                use_case=HOUSE_PRICE_USE_CASE_NAME,
                training_context=TRAINING_CONTEXT_SELECTION,
                extra_tags={"candidate_role": "candidate"},
            )
            run_ids[model_name] = str(run_id)

            all_results.append({
                "model_name": model_name,
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
            })
            pipelines[model_name] = pipeline

        if not all_results:
            raise TrainingError("No candidate models were evaluated.")

        # 7. Select model with highest R² via the shared selector.
        best_entry = select_best_candidate(evaluated)
        best_model_name = best_entry["model_name"]
        best_pipeline = pipelines[best_model_name]
        best_r2 = best_entry["metrics"]["r2"]

        # Step 61 (I-12): mark the winning run without altering run_id
        # lineage (best-effort; tracking-only policy unchanged).
        try:
            best_run_id = run_ids.get(best_model_name, "")
            if best_run_id:
                tracker.add_tags_to_run(best_run_id, {"candidate_role": "best"})
        except Exception:
            pass

        logger.info(
            f"Best model selected: {best_model_name} with R2: {best_r2:.4f}"
        )

        # 8. No registry write: evaluation ends at experiment runs.
        #    Promotion (ModelPromoter) registers the exact promoted run.
        logger.info(
            f"Selection complete for '{best_model_name}'; no registry version created."
        )

        return {
            "best_model_name": best_model_name,
            "best_pipeline": best_pipeline,
            "all_results": all_results
        }

    except Exception as e:
        logger.error(f"House Price model selection failed: {e}")
        if isinstance(e, TrainingError):
            raise
        raise TrainingError(f"Failed to select best House Price model: {str(e)}") from e
