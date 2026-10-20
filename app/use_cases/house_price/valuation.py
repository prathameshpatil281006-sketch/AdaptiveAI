"""Prediction-valuation service for House Price (Phase 2).

Enriches a point prediction with market context, a defensible
uncertainty range, and serving metadata:

- **Model Estimate**: ``pipeline.predict`` output for the property.
- **Locality Benchmark**: calibration-anchor reference from
  ``benchmarks`` (rate x area) — market context, NOT a prediction.
- **Prediction range**: empirical residual interval (see below).
- **Metadata**: actual serving metadata passed in by the caller
  (model name/version, schema version, inference timestamp).

Prediction-range methodology (empirical residual interval):

1. Take the shared holdout split (``prepare_house_price_data``:
   test_size 0.20, config seed — the exact split training and
   evaluation use).
2. Score the LOADED production pipeline on it:
   ``residuals = y_test − pipeline.predict(X_test)``.
3. Take empirical residual quantiles Q05 and Q95
   (``RESIDUAL_LOWER_QUANTILE`` / ``RESIDUAL_UPPER_QUANTILE``).
4. ``prediction_lower = max(0, point + Q05)``,
   ``prediction_upper = point + Q95``.

By construction, 90% of the holdout residuals fall inside [Q05, Q95].
That coverage statement describes the holdout set only — it is NOT a
"95% confidence" claim and no confidence level is reported. The width
is data-derived (it varies with the deployed model and reference data),
never an arbitrary fixed percentage.

No API, SQL, training, registry, or HTTP work lives here. This module
never writes models, pointers, or history.
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from app.core.exceptions import AdaptiveAIError, PredictionError
from app.use_cases.house_price.market_config import DATA_SCHEMA_VERSION

logger = logging.getLogger(__name__)

# Empirical residual-interval quantiles (documented methodology above).
RESIDUAL_LOWER_QUANTILE = 0.05
RESIDUAL_UPPER_QUANTILE = 0.95


def get_holdout_residuals(pipeline: Any) -> np.ndarray:
    """Scores the pipeline on the shared holdout split.

    Uses ``prepare_house_price_data`` (same validation, split, and seed
    as training/evaluation) and returns ``y_test − predict(X_test)``
    as a float array.

    Raises:
        PredictionError: If scoring fails (e.g. schema-incompatible
            pipeline — fails closed, never silently substituted).
    """
    from app.use_cases.house_price.training import prepare_house_price_data

    try:
        _, X_test, _, y_test = prepare_house_price_data()
    except Exception as e:
        raise PredictionError(
            f"Failed to prepare holdout data for residual analysis: {e}"
        ) from e
    try:
        predicted = np.asarray(pipeline.predict(X_test), dtype=float).ravel()
        actual = np.asarray(y_test, dtype=float).ravel()
    except Exception as e:
        raise PredictionError(
            f"Failed to score model on holdout data: {e}"
        ) from e
    if predicted.shape != actual.shape:
        raise PredictionError(
            "Holdout scoring returned an unexpected shape: "
            f"predicted {predicted.shape} vs actual {actual.shape}."
        )
    residuals = actual - predicted
    if residuals.size == 0 or not bool(np.all(np.isfinite(residuals))):
        raise PredictionError("Holdout residuals are empty or non-finite.")
    return residuals


def compute_residual_interval(
    point_prediction: float, residuals: np.ndarray
) -> Dict[str, float]:
    """Builds the empirical residual interval around a point prediction.

    Args:
        point_prediction: Model output in INR (finite number).
        residuals: Holdout residuals from :func:`get_holdout_residuals`.

    Returns:
        Dict with ``prediction_lower`` and ``prediction_upper``
        (floats, lower floored at 0.0, lower <= point <= upper
        whenever the residual spread is non-degenerate).
    """
    import math

    point = float(point_prediction)
    if not math.isfinite(point):
        raise PredictionError("'point_prediction' must be finite.")
    resid = np.asarray(residuals, dtype=float).ravel()
    if resid.size == 0 or not bool(np.all(np.isfinite(resid))):
        raise PredictionError("Residuals must be a non-empty finite array.")
    lower_q = float(np.quantile(resid, RESIDUAL_LOWER_QUANTILE))
    upper_q = float(np.quantile(resid, RESIDUAL_UPPER_QUANTILE))
    return {
        "prediction_lower": max(0.0, point + lower_q),
        "prediction_upper": point + upper_q,
    }


def build_valuation(
    point_prediction: Any,
    input_data: Dict[str, Any],
    pipeline: Any,
    model_name: Optional[str] = None,
    model_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Builds the enriched valuation for one predicted property.

    Combines the point prediction with per-sqft pricing, the locality
    benchmark comparison, the empirical residual interval, and serving
    metadata. The residual interval degrades to ``None`` bounds (logged)
    if holdout scoring fails, so enrichment can never break the core
    point prediction.

    Args:
        point_prediction: Model output in INR.
        input_data: Validated schema-v2 feature mapping.
        pipeline: Loaded production pipeline (used for residuals only;
            the point prediction itself is NOT recomputed here).
        model_name: Actual served model name (or None when unknown).
        model_version: Actual served registry version (or None).

    Returns:
        Dict with ``prediction``, ``predicted_price_per_sqft``,
        ``locality_benchmark_price``, ``benchmark_price_per_sqft``,
        ``benchmark_difference``, ``benchmark_difference_percent``,
        ``prediction_lower``/``prediction_upper`` (or None on
        degradation), ``model_name``, ``model_version``,
        ``schema_version``, ``location``, ``area_sqft``, ``inferred_at``.
    """
    from app.use_cases.house_price.benchmarks import (
        compare_to_benchmark,
        get_locality_benchmark,
    )

    import math

    try:
        point = float(point_prediction)
    except (TypeError, ValueError) as e:
        raise PredictionError(
            f"'point_prediction' must be numeric, got {point_prediction!r}."
        ) from e
    if not math.isfinite(point):
        raise PredictionError("'point_prediction' must be finite.")
    if not isinstance(input_data, dict):
        raise PredictionError("'input_data' must be a dict.")
    location = input_data.get("location")
    area = input_data.get("area_sqft")
    comparison = compare_to_benchmark(point, location, area)

    lower: Optional[float] = None
    upper: Optional[float] = None
    try:
        residuals = get_holdout_residuals(pipeline)
        interval = compute_residual_interval(point, residuals)
        lower = float(interval["prediction_lower"])
        upper = float(interval["prediction_upper"])
    except AdaptiveAIError as e:
        logger.warning(f"Prediction interval unavailable, degrading: {e}")

    try:
        area_value = float(area)
    except (TypeError, ValueError):
        area_value = area
    return {
        "prediction": point,
        "predicted_price_per_sqft": float(comparison["predicted_price_per_sqft"]),
        "locality_benchmark_price": float(comparison["locality_benchmark_price"]),
        "benchmark_price_per_sqft": float(
            get_locality_benchmark(location)["benchmark_price_per_sqft"]
        ),
        "benchmark_difference": float(comparison["benchmark_difference"]),
        "benchmark_difference_percent": float(
            comparison["benchmark_difference_percent"]
        ),
        "prediction_lower": lower,
        "prediction_upper": upper,
        "model_name": model_name,
        "model_version": model_version,
        "schema_version": int(DATA_SCHEMA_VERSION),
        "location": location,
        "area_sqft": area_value,
        "inferred_at": datetime.now(timezone.utc).isoformat(),
    }


def list_valuation_fields() -> List[str]:
    """Returns the response-field names :func:`build_valuation` produces."""
    return [
        "prediction",
        "predicted_price_per_sqft",
        "locality_benchmark_price",
        "benchmark_price_per_sqft",
        "benchmark_difference",
        "benchmark_difference_percent",
        "prediction_lower",
        "prediction_upper",
        "model_name",
        "model_version",
        "schema_version",
        "location",
        "area_sqft",
        "inferred_at",
    ]
