"""Production info service for House Price (frontend migration).

Computes the production model specification shown in the Overview page
without duplicating training logic:

- reads the production pointer (serving authority)
- reports algorithm family, sklearn version, run id, dataset size
- evaluates the production artifact on the shared holdout split to
  produce R2 / MAE / RMSE (same split/seed as training)
- reports input feature signatures (name + dtype)

No API, SQL, or HTTP work lives here. The FastAPI layer calls
``get_house_price_production_info`` as a thin adapter.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def get_house_price_production_info(
    models_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Returns the production model specification for House Price.

    Returns:
        Dict with model_name, model_version, registered_model_name,
        run_id, algorithm_family, sklearn_version, training_dataset
        (rows + hash prefix), metrics (r2/mae/rmse), features
        ([{name, dtype}]), and pointer metadata.

    Raises:
        ModelNotFoundError: If no production model is deployed.
        AdaptiveAIError: If the pointer is corrupt or evaluation fails.
    """
    from sklearn import __version__ as sklearn_version

    from app.core.exceptions import AdaptiveAIError, ModelNotFoundError
    from app.use_cases.house_price.model_persistence import read_production_pointer
    from app.use_cases.house_price.use_case import HousePriceUseCase

    try:
        pointer = read_production_pointer(models_dir=models_dir)
    except ModelNotFoundError:
        raise
    except Exception as e:
        raise AdaptiveAIError(f"Failed to read production pointer: {e}") from e

    model_name = pointer.get("model_name", "LinearRegression")
    use_case = HousePriceUseCase()
    feature_names: List[str] = list(use_case.get_feature_names())

    # Reference dataset size (training dataset card).
    training_rows: Optional[int] = None
    reference_hash: Optional[str] = None
    try:
        from app.core.config import resolve_project_path

        import pandas as pd

        ref_path = resolve_project_path(
            Path("data") / "reference" / "house_price_reference.csv"
        )
        if ref_path.is_file():
            df_ref = pd.read_csv(ref_path)
            training_rows = int(len(df_ref))
            try:
                import hashlib

                with open(ref_path, "rb") as f:
                    digest = hashlib.sha256(f.read()).hexdigest()
                reference_hash = f"sha256:{digest[:10]}..."
            except Exception:
                reference_hash = None
    except Exception as e:
        logger.warning(f"Failed to inspect reference dataset: {e}")

    # Evaluate production artifact on the shared holdout split.
    metrics: Dict[str, Optional[float]] = {"r2": None, "mae": None, "rmse": None}
    try:
        from app.retraining.model_evaluator import compute_regression_metrics
        from app.use_cases.house_price.model_persistence import load_house_price_model
        from app.use_cases.house_price.training import prepare_house_price_data

        pipeline = load_house_price_model(model_name, models_dir=models_dir)
        _, X_test, _, y_test = prepare_house_price_data()
        y_pred = pipeline.predict(X_test)
        computed = compute_regression_metrics(y_test, y_pred)
        metrics = {
            "r2": float(computed["r2"]),
            "mae": float(computed["mae"]),
            "rmse": float(computed["rmse"]),
        }
    except ModelNotFoundError:
        raise
    except Exception as e:
        logger.warning(f"Failed to evaluate production model: {e}")

    # Feature signatures (dtype from reference CSV when available).
    features: List[Dict[str, str]] = []
    try:
        from app.core.config import resolve_project_path

        import pandas as pd

        ref_path = resolve_project_path(
            Path("data") / "reference" / "house_price_reference.csv"
        )
        dtypes: Dict[str, str] = {}
        if ref_path.is_file():
            df = pd.read_csv(ref_path, nrows=5)
            for col in feature_names:
                if col in df.columns:
                    dtypes[col] = str(df[col].dtype)
        for name in feature_names:
            features.append({"name": name, "dtype": dtypes.get(name, "float64")})
    except Exception:
        features = [{"name": n, "dtype": "float64"} for n in feature_names]

    return {
        "model_name": model_name,
        "model_version": pointer.get("model_version"),
        "registered_model_name": pointer.get("registered_model_name", "HousePriceModel"),
        "run_id": pointer.get("run_id"),
        "algorithm_family": model_name,
        "sklearn_version": sklearn_version,
        "training_dataset": {
            "rows": training_rows,
            "reference": reference_hash,
        },
        "metrics": metrics,
        "features": features,
        "deployed_at": pointer.get("deployed_at"),
        "use_case": pointer.get("use_case", "house_price"),
    }
