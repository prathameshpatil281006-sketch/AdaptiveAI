"""Explain service for House Price predictions (Phase 2).

Per-feature influence for the currently deployed production model,
computed from the ACTUAL loaded pipeline — never hardcoded:

- Linear models (``coef_``): exact decomposition
  ``prediction = intercept + sum(contribution_i)`` where each
  contribution is ``coef * scaled_feature`` summed over the
  feature's transformed columns (one-hot categoricals included).
- Any other regressor (RandomForest, GradientBoosting, ...):
  baseline-substitution influence. Each feature is replaced, one at a
  time, by a reference baseline (median for numeric, mode for
  categoricals, taken from the versioned reference CSV) while all
  others are held fixed::

      contribution_i = predict(x) - predict(x with feature i = baseline_i)

  These are marginal effects against a documented baseline property —
  they do NOT sum to the prediction, and no exactness is claimed.
  ``intercept`` in this mode is the baseline-property prediction
  ``predict(all features = baseline)``, i.e. the reference point the
  influences are measured against.

``direction`` is ``positive``/``negative``/``neutral`` from the sign of
the contribution; ``neutral`` covers |contribution| within
``NEUTRAL_TOLERANCE_FRACTION`` of |prediction| (numerical noise, not a
claim of zero effect).

No API, SQL, or HTTP work lives here.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

# Explanation method identifiers (stable contract).
EXPLAIN_METHOD_LINEAR_EXACT = "linear_exact"
EXPLAIN_METHOD_BASELINE_SUBSTITUTION = "baseline_substitution"

# |contribution| within this fraction of |prediction| reads as neutral.
NEUTRAL_TOLERANCE_FRACTION = 0.005

FEATURE_LABELS = {
    "area_sqft": "Living Area",
    "bedrooms": "Bedrooms",
    "bathrooms": "Bathrooms",
    "parking": "Parking Spots",
    "location": "Locality",
    "property_age": "Age Depreciation",
    "floor_number": "Floor",
    "total_floors": "Building Height",
    "furnishing_status": "Furnishing",
    "property_type": "Property Type",
}


def explain_house_price_prediction(
    input_data: Dict[str, Any],
    models_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Explains one prediction with per-feature contributions.

    Args:
        input_data: Mapping with the schema-v2 House Price features
            (area_sqft, bedrooms, bathrooms, parking, location,
            property_age, floor_number, total_floors, furnishing_status,
            property_type).
        models_dir: Optional models directory override.

    Returns:
        Dict with prediction, intercept (model intercept for linear
        models; baseline-property prediction for substitution mode),
        contributions ([{feature, label, value, contribution,
        direction}]), method ("linear_exact" or
        "baseline_substitution"), model_name, model_version.
    """
    from app.core.exceptions import AdaptiveAIError, ModelNotFoundError
    from app.use_cases.house_price.model_persistence import (
        load_house_price_model,
        read_production_pointer,
    )
    from app.use_cases.house_price.use_case import HousePriceUseCase

    import pandas as pd

    try:
        pointer = read_production_pointer(models_dir=models_dir)
        model_name = pointer.get("model_name", "LinearRegression")
        model_version = pointer.get("model_version")
    except ModelNotFoundError:
        raise
    except Exception as e:
        raise AdaptiveAIError(f"Failed to read production pointer: {e}") from e

    try:
        pipeline = load_house_price_model(model_name, models_dir=models_dir)
    except Exception as e:
        raise AdaptiveAIError(f"Failed to load production model: {e}") from e

    use_case = HousePriceUseCase()
    feature_names = list(use_case.get_feature_names())
    numeric_names = list(use_case.get_numeric_feature_names())
    try:
        row: Dict[str, Any] = {}
        for k in feature_names:
            if k in numeric_names:
                row[k] = float(input_data[k])
            else:
                row[k] = str(input_data[k])
    except Exception as e:
        raise AdaptiveAIError(f"Invalid input for explanation: {e}") from e

    df = pd.DataFrame([{k: row[k] for k in feature_names}])
    try:
        prediction = float(pipeline.predict(df)[0])
    except Exception as e:
        raise AdaptiveAIError(f"Explanation prediction failed: {e}") from e

    regressor = _find_regressor(pipeline)
    method = EXPLAIN_METHOD_BASELINE_SUBSTITUTION
    intercept: Optional[float] = None
    per_feature: Dict[str, float] = {}
    if regressor is not None and hasattr(regressor, "coef_"):
        try:
            intercept, per_feature = _linear_contributions(
                pipeline, regressor, df, feature_names, numeric_names
            )
            method = EXPLAIN_METHOD_LINEAR_EXACT
        except AdaptiveAIError as e:
            # Exact path not honorably available: fall back to the
            # model-agnostic substitution instead of fabricating values.
            logger.warning(f"Linear explanation unavailable, substituting: {e}")
    if method == EXPLAIN_METHOD_BASELINE_SUBSTITUTION:
        intercept, per_feature = _substitution_contributions(
            pipeline, df, row, feature_names, numeric_names
        )

    tolerance = abs(prediction) * NEUTRAL_TOLERANCE_FRACTION
    contributions = [
        {
            "feature": name,
            "label": FEATURE_LABELS.get(name, name),
            "value": row[name],
            "contribution": float(per_feature.get(name, 0.0)),
            "direction": _direction(float(per_feature.get(name, 0.0)), tolerance),
        }
        for name in feature_names
    ]

    return {
        "prediction": prediction,
        "intercept": intercept,
        "contributions": contributions,
        "method": method,
        "model_name": model_name,
        "model_version": model_version,
    }


def _direction(contribution: float, tolerance: float) -> str:
    """Maps a contribution sign to positive/negative/neutral."""
    if contribution > tolerance:
        return "positive"
    if contribution < -tolerance:
        return "negative"
    return "neutral"


def _find_regressor(pipeline: Any) -> Any:
    """Finds the estimator step of a pipeline (or the model itself).

    Training builds ``('preprocessor', ...) -> ('model', ...)`` while
    some callers/tests use a ``'regressor'`` step name; both are
    honored so explanation stays compatible with whichever supported
    model type is deployed. A bare estimator is returned as-is.
    """
    steps = getattr(pipeline, "named_steps", None)
    if isinstance(steps, dict):
        for key in ("model", "regressor"):
            if steps.get(key) is not None:
                return steps.get(key)
        return None
    if hasattr(pipeline, "predict"):
        return pipeline
    return None


def _linear_contributions(
    pipeline: Any,
    regressor: Any,
    df: Any,
    feature_names: List[str],
    numeric_names: List[str],
) -> Any:
    """Exact coef-based decomposition (linear models only).

    Returns (intercept, {feature: contribution}). Raises
    AdaptiveAIError when the exact path cannot be honored — the caller
    then falls back to substitution rather than fabricating values.
    """
    from app.core.exceptions import AdaptiveAIError

    import numpy as np

    preprocessor = None
    steps = getattr(pipeline, "named_steps", None)
    if isinstance(steps, dict):
        preprocessor = steps.get("preprocessor")
    coef = np.asarray(regressor.coef_).ravel()
    try:
        intercept = float(np.asarray(regressor.intercept_).ravel()[0])
    except Exception as e:
        raise AdaptiveAIError(f"Linear model has no usable intercept: {e}") from e
    if preprocessor is not None:
        try:
            scaled = np.asarray(preprocessor.transform(df)).ravel()
            out_names = list(preprocessor.get_feature_names_out())
        except Exception as e:
            raise AdaptiveAIError(
                f"Failed to map linear contributions: {e}"
            ) from e
    else:
        # Bare linear estimator: only numeric features are mappable.
        if set(feature_names) != set(numeric_names):
            raise AdaptiveAIError(
                "Bare linear estimator cannot map categorical features."
            )
        scaled = np.asarray(df[numeric_names].to_numpy(dtype=float)).ravel()
        out_names = [f"manual__{n}" for n in numeric_names]
    if len(coef) != len(scaled) or not out_names:
        raise AdaptiveAIError("Coefficient/transform shape mismatch.")
    per_feature: Dict[str, float] = {n: 0.0 for n in feature_names}
    for coef_i, scaled_i, out in zip(coef, scaled, out_names):
        # ColumnTransformer names look like "num__area_sqft"
        # or "cat__location_Wakad".
        base = str(out).split("__", 1)[-1] if "__" in str(out) else str(out)
        owner = base
        for cand in feature_names:
            if base == cand or base.startswith(cand + "_"):
                owner = cand
                break
        if owner in per_feature:
            per_feature[owner] += float(coef_i * scaled_i)
    return intercept, per_feature


def _reference_baselines(
    feature_names: List[str], numeric_names: List[str]
) -> Dict[str, Any]:
    """Loads substitution baselines from the versioned reference CSV.

    Median for numeric features, mode (first) for categoricals.
    The reference dataset is the documented baseline source so every
    explanation measures against the same baseline property.
    """
    from app.core.config import resolve_project_path
    from app.core.exceptions import AdaptiveAIError

    import pandas as pd

    ref_path = resolve_project_path(
        "data/reference/house_price_reference.csv"
    )
    if not ref_path.is_file():
        raise AdaptiveAIError(
            f"Reference dataset not found for explanation baselines: {ref_path}"
        )
    try:
        ref = pd.read_csv(ref_path)
    except Exception as e:
        raise AdaptiveAIError(
            f"Failed to read reference dataset for baselines: {e}"
        ) from e
    baselines: Dict[str, Any] = {}
    for name in feature_names:
        if name not in ref.columns:
            raise AdaptiveAIError(
                f"Reference dataset is missing feature '{name}'."
            )
        if name in numeric_names:
            baselines[name] = float(ref[name].median())
        else:
            modes = ref[name].mode()
            if len(modes) == 0:
                raise AdaptiveAIError(
                    f"No baseline value for feature '{name}'."
                )
            baselines[name] = str(modes.iloc[0])
    return baselines


def _substitution_contributions(
    pipeline: Any,
    df: Any,
    row: Dict[str, Any],
    feature_names: List[str],
    numeric_names: List[str],
) -> Any:
    """Baseline-substitution influence for non-linear models.

    contribution_i = predict(x) − predict(x with feature i = baseline).
    Returns (baseline_prediction, {feature: contribution}).
    """
    from app.core.exceptions import AdaptiveAIError

    baselines = _reference_baselines(feature_names, numeric_names)
    base_row = {k: baselines[k] for k in feature_names}

    import pandas as pd

    try:
        point = float(pipeline.predict(df)[0])
        baseline_prediction = float(
            pipeline.predict(pd.DataFrame([base_row]))[0]
        )
    except Exception as e:
        raise AdaptiveAIError(f"Substitution explanation failed: {e}") from e
    per_feature: Dict[str, float] = {}
    for name in feature_names:
        perturbed = dict(row)
        perturbed[name] = baselines[name]
        try:
            altered = float(pipeline.predict(pd.DataFrame([perturbed]))[0])
        except Exception as e:
            raise AdaptiveAIError(
                f"Substitution explanation failed for '{name}': {e}"
            ) from e
        per_feature[name] = float(point - altered)
    return baseline_prediction, per_feature
