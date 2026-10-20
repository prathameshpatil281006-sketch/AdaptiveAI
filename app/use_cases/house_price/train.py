import logging
from typing import Tuple, Dict, Union
from pathlib import Path
from sklearn.pipeline import Pipeline
import pandas as pd

from app.core.exceptions import TrainingError
from app.use_cases.house_price.training import (
    get_house_price_candidates,
    prepare_house_price_data,
    train_single_candidate,
)


def train_house_price_model(
    data_path: Union[str, Path, pd.DataFrame] = "data/reference/house_price_reference.csv",
) -> Tuple[Pipeline, Dict[str, float]]:
    """
    Trains and evaluates a linear regression model for House Price prediction.

    Step 60 (I-11): delegates to the authoritative House Price training
    path (``app.use_cases.house_price.training``) so legacy training
    shares the exact preprocessing, split, candidate configuration,
    and metrics implementation used by selection and orchestration.
    The legacy single-model contract is preserved: LinearRegression
    only, no tracking, ``(pipeline, {mae, rmse, r2})`` return shape.

    Args:
        data_path (str): Path to the CSV dataset. Relative paths resolve
            against the project root, so training works regardless of the
            process working directory.

    Returns:
        Tuple[Pipeline, Dict[str, float]]: The trained scikit-learn pipeline and evaluation metrics.

    Raises:
        TrainingError: If training or evaluation fails.
    """
    logger = logging.getLogger(__name__)
    logger.info("House Price model training started.")

    try:
        # 1-2. Load/validate/split via the authoritative path
        # (project-root anchored, CWD independent; validation + seed
        # behavior identical to selection/orchestration).
        X_train, X_test, y_train, y_test = prepare_house_price_data(data_path)

        # 3. Resolve the canonical LinearRegression configuration so the
        # legacy path cannot diverge from the candidate definition.
        candidates = get_house_price_candidates()
        lr_entry = next(
            (c for c in candidates if c.get("model_type") == "LinearRegression"),
            {"model_type": "LinearRegression", "hyperparameters": {"fit_intercept": True}},
        )
        lr_hyperparams = dict(lr_entry.get("hyperparameters", {}) or {})

        # 4-5. Train/evaluate via the authoritative single-candidate path
        # (shared MLEngine + preprocessor + metrics implementation).
        result = train_single_candidate(
            X_train,
            X_test,
            y_train,
            y_test,
            "LinearRegression",
            lr_hyperparams,
        )
        model = result["pipeline"]
        metrics = {
            "mae": float(result["metrics"]["mae"]),
            "rmse": float(result["metrics"]["rmse"]),
            "r2": float(result["metrics"]["r2"]),
        }

        logger.info(f"House Price model training completed. Metrics: {metrics}")

        return model, metrics

    except Exception as e:
        logger.error(f"House Price model training failed: {e}")
        if isinstance(e, TrainingError):
            raise
        raise TrainingError(f"Failed to train House Price model: {str(e)}") from e
