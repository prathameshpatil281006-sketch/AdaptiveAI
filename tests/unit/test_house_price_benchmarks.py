"""Phase 2 tests: locality market-benchmark service.

The benchmark service must stay a thin view over the Phase 1
calibration table (market_config.LOCALITY_RATES) — no second pricing
table. Uses no models, databases, or production state.
"""

import math

import pytest

from app.core.exceptions import DataValidationError
from app.use_cases.house_price.benchmarks import (
    compare_to_benchmark,
    compute_benchmark_price,
    get_all_benchmarks,
    get_locality_benchmark,
    list_supported_localities,
)
from app.use_cases.house_price.market_config import LOCALITY_RATES


def test_supported_localities_match_calibration():
    assert set(list_supported_localities()) == set(LOCALITY_RATES.keys())
    assert len(list_supported_localities()) == 12


def test_benchmark_single_source_no_second_table():
    for loc, cfg in LOCALITY_RATES.items():
        bench = get_locality_benchmark(loc)
        assert bench["location"] == loc
        assert bench["benchmark_price_per_sqft"] == float(
            cfg["base_rate_per_sqft"]
        )
        assert isinstance(bench["benchmark_source"], str) and bench["benchmark_source"]
        assert isinstance(bench["calibration_note"], str) and bench["calibration_note"]


def test_all_benchmarks_covers_every_locality():
    items = get_all_benchmarks()
    assert len(items) == 12
    assert {i["location"] for i in items} == set(LOCALITY_RATES.keys())


def test_unknown_location_rejected():
    with pytest.raises(DataValidationError):
        get_locality_benchmark("Atlantis")
    with pytest.raises(DataValidationError):
        compute_benchmark_price("Atlantis", 1000.0)


def test_benchmark_price_is_rate_times_area():
    assert compute_benchmark_price("Wakad", 1200.0) == pytest.approx(8800 * 1200.0)
    assert compute_benchmark_price("Kothrud", 500.0) == pytest.approx(11500 * 500.0)


def test_benchmark_price_rejects_bad_area():
    for bad in (0.0, -100.0, float("nan"), float("inf"), "huge", None):
        with pytest.raises(DataValidationError):
            compute_benchmark_price("Wakad", bad)


def test_compare_math_is_model_minus_benchmark():
    out = compare_to_benchmark(11_000_000.0, "Wakad", 1000.0)
    assert out["locality_benchmark_price"] == pytest.approx(8_800_000.0)
    assert out["predicted_price_per_sqft"] == pytest.approx(11_000.0)
    assert out["benchmark_difference"] == pytest.approx(2_200_000.0)
    assert out["benchmark_difference_percent"] == pytest.approx(25.0)
    for key in (
        "predicted_price_per_sqft",
        "locality_benchmark_price",
        "benchmark_difference",
        "benchmark_difference_percent",
    ):
        assert math.isfinite(out[key])


def test_compare_negative_difference():
    out = compare_to_benchmark(5_000_000.0, "Baner", 1000.0)
    assert out["benchmark_difference"] == pytest.approx(-5_500_000.0)
    assert out["benchmark_difference_percent"] == pytest.approx(
        -5_500_000.0 / 10_500_000.0 * 100.0
    )
