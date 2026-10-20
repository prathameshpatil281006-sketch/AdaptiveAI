"""Authoritative House Price training and evaluation path (Step 60, I-11 fix).

Single reusable implementation for House Price data handling, candidate
definition access, per-candidate training/evaluation, full candidate
evaluation, and best-candidate selection.

Consolidated paths (all delegate here, no duplicated logic):
  - legacy single-model training (``train.train_house_price_model``)
  - candidate selection (``model_selection.select_best_house_price_model``)
  - orchestration layer (``RetrainingService.retrain``)

Preserved exactly:
  - candidate set: LinearRegression, RandomForestRegressor,
    GradientBoostingRegressor with the hyperparameters defined by
    ``HousePriceUseCase.get_model_candidates`` (that method remains the
    single source of candidate definitions).
  - preprocessing: ``get_preprocessor`` (median impute + standard scale).
  - validation: ``validate_house_price_data`` (via ``split``).
  - split: ``split_house_price_data`` (test_size 0.20, config random_seed).
  - metrics: MAE / RMSE / R2 via ``MLEngine.train_and_evaluate``.
  - tracking-only policy: this module performs NO MLflow registry writes
    and NO production pointer writes; callers track runs only.

This module performs no CSV/SQL/scheduler/API/dashboard work beyond
loading the training CSV via pandas, and never touches the registry,
production artifacts, or databases.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

from app.core.config import resolve_project_path
from app.core.exceptions import TrainingError

logger = logging.getLogger(__name__)

DEFAULT_HOUSE_PRICE_DATA_PATH = "data/reference/house_price_reference.csv"


def create_regressor(model_type: str, hyperparameters: Dict[str, Any]):
    """Instantiates a regressor for the canonical House Price candidate set.

    Args:
        model_type: One of ``LinearRegression``, ``RandomForestRegressor``,
            ``GradientBoostingRegressor``.
        hyperparameters: Constructor kwargs for the regressor.

    Returns:
        An unfitted scikit-learn regressor.

    Raises:
        TrainingError: If the model type is unsupported.
    """
    if hyperparameters is None:
        hyperparameters = {}
    if model_type == "LinearRegression":
        return LinearRegression(**hyperparameters)
    elif model_type == "RandomForestRegressor":
        return RandomForestRegressor(**hyperparameters)
    elif model_type == "GradientBoostingRegressor":
        return GradientBoostingRegressor(**hyperparameters)
    else:
        raise TrainingError(f"Unsupported model type: {model_type}")


def load_house_price_dataframe(
    data_path: Union[str, Path, pd.DataFrame],
) -> pd.DataFrame:
    """Loads House Price training data from a CSV path or DataFrame.

    Relative CSV paths resolve against the project root (CWD independent);
    absolute paths pass through unchanged. DataFrames are copied.

    Raises:
        TrainingError: If the path type is invalid or loading fails.
    """
    if isinstance(data_path, pd.DataFrame):
        return data_path.copy()
    if isinstance(data_path, (str, Path)):
        path = resolve_project_path(data_path)
        try:
            return pd.read_csv(path)
        except FileNotFoundError as e:
            raise TrainingError(f"Training data not found: {path}") from e
        except Exception as e:
            raise TrainingError(f"Failed to load training data: {e}") from e
    raise TrainingError(
        f"Invalid data_path type: {type(data_path)}. "
        "Expected a CSV path or pandas DataFrame."
    )


def prepare_house_price_splits(df: pd.DataFrame):
    """Validates and splits House Price data via the shared split utility.

    Uses ``split_house_price_data`` (validation + test_size 0.20 + config
    random_seed). Lets validation/split errors propagate so callers can
    map them to their own error contract.
    """
    from app.use_cases.house_price.split import split_house_price_data

    return split_house_price_data(df)


def prepare_house_price_data(
    data_path: Union[str, Path, pd.DataFrame] = DEFAULT_HOUSE_PRICE_DATA_PATH,
):
    """Loads and splits House Price data in one authoritative step."""
    df = load_house_price_dataframe(data_path)
    return prepare_house_price_splits(df)


def get_house_price_candidates() -> List[Dict[str, Any]]:
    """Returns the canonical House Price candidate definitions.

    Delegates to ``HousePriceUseCase.get_model_candidates`` so the use
    case remains the single source of candidate names/configurations.
    """
    from app.use_cases.house_price.use_case import HousePriceUseCase

    return HousePriceUseCase().get_model_candidates()


def train_candidate_with_instances(
    X_train,
    X_test,
    y_train,
    y_test,
    *,
    model_name: str,
    hyperparameters: Dict[str, Any],
    preprocessor: Any,
    regressor: Any,
    engine: Any,
) -> Dict[str, Any]:
    """Trains/evaluates one candidate via the shared engine interaction.

    Single place where ``MLEngine.train_and_evaluate`` is invoked for
    House Price and where raw engine metrics are normalized to
    ``{mae, rmse, r2}`` floats. All House Price paths (legacy training,
    selection, orchestration) share this function so preprocessing,
    metrics, and pipeline construction cannot diverge.

    Returns:
        Dict with ``model_name``, ``hyperparameters``, ``pipeline``,
        ``metrics`` ({mae, rmse, r2} floats).

    Raises:
        TrainingError: If the engine output is malformed.
    """
    engine_output = engine.train_and_evaluate(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        preprocessor=preprocessor,
        model=regressor,
    )
    try:
        pipeline = engine_output["pipeline"]
        raw_metrics = engine_output["metrics"]
        metrics = {
            "mae": float(raw_metrics["mae"]),
            "rmse": float(raw_metrics["rmse"]),
            "r2": float(raw_metrics["r2"]),
        }
    except Exception as e:
        raise TrainingError(
            f"Invalid training output for candidate '{model_name}': {e}"
        ) from e
    return {
        "model_name": model_name,
        "hyperparameters": dict(hyperparameters or {}),
        "pipeline": pipeline,
        "metrics": metrics,
    }


def train_single_candidate(
    X_train,
    X_test,
    y_train,
    y_test,
    model_type: str,
    hyperparameters: Optional[Dict[str, Any]] = None,
    engine: Any = None,
    preprocessor: Any = None,
) -> Dict[str, Any]:
    """Trains/evaluates a single named candidate with default resolution.

    Resolves a default ``MLEngine`` and ``get_preprocessor()`` when not
    supplied, instantiates the regressor via ``create_regressor``, then
    delegates to ``train_candidate_with_instances``.
    """
    from app.engine.ml_engine import MLEngine
    from app.use_cases.house_price.preprocessing import get_preprocessor

    if hyperparameters is None:
        hyperparameters = {}
    resolved_engine = engine if engine is not None else MLEngine()
    resolved_preprocessor = (
        preprocessor if preprocessor is not None else get_preprocessor()
    )
    regressor = create_regressor(model_type, dict(hyperparameters))
    return train_candidate_with_instances(
        X_train,
        X_test,
        y_train,
        y_test,
        model_name=model_type,
        hyperparameters=dict(hyperparameters),
        preprocessor=resolved_preprocessor,
        regressor=regressor,
        engine=resolved_engine,
    )


def evaluate_all_candidates(
    X_train,
    X_test,
    y_train,
    y_test,
    candidates: Optional[List[Dict[str, Any]]] = None,
    engine: Any = None,
) -> List[Dict[str, Any]]:
    """Trains/evaluates every candidate via the authoritative single path.

    Args:
        X_train/X_test/y_train/y_test: Already-split data.
        candidates: Candidate dicts (``model_type`` + ``hyperparameters``).
            Defaults to ``get_house_price_candidates()``.
        engine: Shared engine instance. Defaults to a new ``MLEngine``.

    Returns:
        List of per-candidate dicts (``model_name``, ``hyperparameters``,
        ``pipeline``, ``metrics``) in candidate order.
    """
    from app.engine.ml_engine import MLEngine

    resolved_candidates = (
        candidates if candidates is not None else get_house_price_candidates()
    )
    resolved_engine = engine if engine is not None else MLEngine()
    evaluated: List[Dict[str, Any]] = []
    for candidate in resolved_candidates:
        model_name = candidate.get("model_type")
        hyperparams = candidate.get("hyperparameters", {}) or {}
        evaluated.append(
            train_single_candidate(
                X_train,
                X_test,
                y_train,
                y_test,
                str(model_name),
                dict(hyperparams),
                engine=resolved_engine,
            )
        )
    return evaluated


def select_best_candidate(
    evaluated: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Selects the candidate with the highest R2 (first max wins).

    Raises:
        TrainingError: If no candidates were evaluated.
    """
    if not evaluated:
        raise TrainingError("No candidate models were evaluated.")
    return max(evaluated, key=lambda item: item["metrics"]["r2"])


def train_and_evaluate_house_price_data(
    data_path: Union[str, Path, pd.DataFrame] = DEFAULT_HOUSE_PRICE_DATA_PATH,
    engine: Any = None,
) -> Tuple[List[Dict[str, Any]], Tuple]:
    """Loads, splits, and evaluates all candidates (no tracking).

    Convenience combining ``prepare_house_price_data`` +
    ``evaluate_all_candidates`` for callers that only need the shared
    evaluation without MLflow or orchestration concerns.

    Returns:
        Tuple of (evaluated_candidates, splits) where splits is
        ``(X_train, X_test, y_train, y_test)``.
    """
    X_train, X_test, y_train, y_test = prepare_house_price_data(data_path)
    evaluated = evaluate_all_candidates(
        X_train, X_test, y_train, y_test, engine=engine
    )
    return evaluated, (X_train, X_test, y_train, y_test)
