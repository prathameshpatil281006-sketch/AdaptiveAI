"""Locality market-benchmark service for House Price (Phase 2).

Single home for Pune locality benchmark data. The ONLY pricing source is
the Phase 1 calibration table (``LOCALITY_RATES`` in
``app.use_cases.house_price.market_config``); nothing here invents a
second pricing table.

Concepts (kept strictly apart):

- **ML prediction** ("Model Estimate"): the regressor pipeline output for
  a specific property. Property-specific.
- **Locality benchmark** ("Locality Benchmark"): ``base_rate_per_sqft``
  for the property's locality, i.e. the synthetic market-calibration
  anchor. Locality-level, identical for every property in that locality
  with the same area.
- **Benchmark price**: ``benchmark_price_per_sqft * area_sqft`` — a
  market-context reference, NOT a prediction.

The dataset is FULLY SYNTHETIC and the calibration anchors are
approximate asking-rate references, NOT transaction records.

No API, SQL, training, or HTTP work lives here.
"""

from typing import Any, Dict, List

from app.core.exceptions import DataValidationError
from app.use_cases.house_price.market_config import LOCALITY_RATES

BENCHMARK_SOURCE = (
    "Synthetic market-calibration reference: approximate 2026 Pune "
    "locality asking-rate anchors from Phase 1 (market_config.py). "
    "Not transaction data; verify against current listings before any "
    "real-world use."
)


def list_supported_localities() -> List[str]:
    """Returns the supported Pune locality names (sorted)."""
    return sorted(LOCALITY_RATES.keys())


def get_locality_benchmark(location: Any) -> Dict[str, Any]:
    """Returns the market benchmark for one locality.

    Args:
        location: Locality name; must be a supported Pune locality.

    Returns:
        Dict with ``location``, ``benchmark_price_per_sqft`` (float),
        ``benchmark_source`` and ``calibration_note`` (strings).

    Raises:
        DataValidationError: If the location is unknown.
    """
    if not isinstance(location, str) or location not in LOCALITY_RATES:
        raise DataValidationError(f"Unknown location: {location!r}.")
    return {
        "location": location,
        "benchmark_price_per_sqft": float(
            LOCALITY_RATES[location]["base_rate_per_sqft"]
        ),
        "benchmark_source": BENCHMARK_SOURCE,
        "calibration_note": BENCHMARK_SOURCE,
    }


def get_all_benchmarks() -> List[Dict[str, Any]]:
    """Returns benchmarks for every supported locality (sorted by name)."""
    return [get_locality_benchmark(loc) for loc in list_supported_localities()]


def compute_benchmark_price(location: Any, area_sqft: Any) -> float:
    """Computes the locality benchmark price for a property area.

    Args:
        location: Supported Pune locality.
        area_sqft: Carpet area in sq ft; must be finite and > 0.

    Returns:
        ``benchmark_price_per_sqft * area_sqft`` as a float.

    Raises:
        DataValidationError: For unknown locations or non-positive area.
    """
    benchmark = get_locality_benchmark(location)
    try:
        area = float(area_sqft)
    except (TypeError, ValueError) as e:
        raise DataValidationError(
            f"'area_sqft' must be a number, got {area_sqft!r}."
        ) from e
    import math

    if not math.isfinite(area) or area <= 0:
        raise DataValidationError("'area_sqft' must be strictly greater than 0.")
    return float(benchmark["benchmark_price_per_sqft"]) * area


def compare_to_benchmark(
    predicted_price: Any, location: Any, area_sqft: Any
) -> Dict[str, float]:
    """Compares an ML point prediction against the locality benchmark.

    Args:
        predicted_price: Model output in INR (finite number).
        location: Supported Pune locality.
        area_sqft: Carpet area in sq ft (> 0).

    Returns:
        Dict with ``predicted_price_per_sqft``,
        ``locality_benchmark_price`` (rate x area),
        ``benchmark_difference`` (predicted − benchmark, INR) and
        ``benchmark_difference_percent``
        (``100 * difference / benchmark``).

    Raises:
        DataValidationError: For invalid prediction/location/area.
    """
    import math

    try:
        predicted = float(predicted_price)
    except (TypeError, ValueError) as e:
        raise DataValidationError(
            f"'predicted_price' must be a number, got {predicted_price!r}."
        ) from e
    if not math.isfinite(predicted):
        raise DataValidationError("'predicted_price' must be finite.")
    try:
        area = float(area_sqft)
    except (TypeError, ValueError) as e:
        raise DataValidationError(
            f"'area_sqft' must be a number, got {area_sqft!r}."
        ) from e
    if not math.isfinite(area) or area <= 0:
        raise DataValidationError("'area_sqft' must be strictly greater than 0.")
    benchmark_price = compute_benchmark_price(location, area)
    difference = predicted - benchmark_price
    return {
        "predicted_price_per_sqft": predicted / area,
        "locality_benchmark_price": benchmark_price,
        "benchmark_difference": difference,
        "benchmark_difference_percent": (100.0 * difference / benchmark_price)
        if benchmark_price > 0
        else 0.0,
    }
