"""Phase 6 tests: deterministic demo scenario generation.

Verifies the five simulator scenarios, seed determinism, schema
validity (v2 categories only), expected shift signatures, and input
validation. Never touches production state (pure generation, no I/O
beyond the read-only reference CSV).
"""

import pytest

from app.core.exceptions import AdaptiveAIError
from app.demo.scenarios import (
    SUPPORTED_SCENARIOS,
    describe_scenarios,
    generate_scenario_samples,
)
from app.use_cases.house_price.market_config import (
    FEATURE_COLUMNS,
    FURNISHING_VALUES,
    LOCALITY_RATES,
    PROPERTY_TYPE_VALUES,
)


def _assert_valid_v2_row(row, allow_relationship_violations=False):
    assert set(row.keys()) == set(FEATURE_COLUMNS)
    assert isinstance(row["area_sqft"], float) and row["area_sqft"] > 0
    assert isinstance(row["bedrooms"], int)
    assert isinstance(row["bathrooms"], int)
    assert isinstance(row["parking"], int) and 0 <= row["parking"] <= 3
    assert isinstance(row["property_age"], float) and row["property_age"] >= 0
    assert isinstance(row["floor_number"], int) and row["floor_number"] >= 0
    assert isinstance(row["total_floors"], int) and row["total_floors"] >= 1
    assert row["location"] in LOCALITY_RATES
    assert row["furnishing_status"] in FURNISHING_VALUES
    assert row["property_type"] in PROPERTY_TYPE_VALUES
    if not allow_relationship_violations:
        assert row["floor_number"] <= row["total_floors"]
        assert row["bathrooms"] <= row["bedrooms"] + 2


def test_supported_scenarios_complete():
    assert set(SUPPORTED_SCENARIOS) == {
        "healthy_baseline",
        "area_shift",
        "locality_shift",
        "config_shift",
        "data_quality",
    }


def test_describe_scenarios_metadata():
    items = describe_scenarios()
    assert len(items) == 5
    for item in items:
        assert set(item.keys()) == {"name", "title", "description", "expected"}
        assert item["name"] in SUPPORTED_SCENARIOS
        assert item["title"] and item["expected"]


def test_every_scenario_deterministic():
    for scenario in SUPPORTED_SCENARIOS:
        first = generate_scenario_samples(scenario, 25, 42)
        second = generate_scenario_samples(scenario, 25, 42)
        assert first == second, scenario
        assert len(first) == 25


def test_same_seed_scenario_reproducible_at_scale():
    first = generate_scenario_samples("area_shift", 200, 7)
    second = generate_scenario_samples("area_shift", 200, 7)
    assert first == second


def test_different_seeds_differ():
    first = generate_scenario_samples("area_shift", 50, 1)
    second = generate_scenario_samples("area_shift", 50, 2)
    assert first != second


def test_baseline_rows_are_valid_and_in_distribution():
    import pandas as pd

    from app.core.config import resolve_project_path

    ref = pd.read_csv(resolve_project_path("data/reference/house_price_reference.csv"))
    rows = generate_scenario_samples("healthy_baseline", 50, 42)
    for row in rows:
        _assert_valid_v2_row(row)
    areas = [r["area_sqft"] for r in rows]
    assert min(areas) >= float(ref["area_sqft"].min())
    assert max(areas) <= float(ref["area_sqft"].max())
    assert len({r["location"] for r in rows}) > 1


def test_area_shift_concentrates_high_area():
    rows = generate_scenario_samples("area_shift", 50, 7)
    for row in rows:
        _assert_valid_v2_row(row)
    areas = [r["area_sqft"] for r in rows]
    assert min(areas) >= 2400.0
    assert max(areas) <= 3000.0


def test_area_shift_custom_band_and_validation():
    rows = generate_scenario_samples(
        "area_shift", 10, 3, {"area_min": 1500.0, "area_max": 1800.0}
    )
    assert all(1500.0 <= r["area_sqft"] <= 1800.0 for r in rows)
    with pytest.raises(AdaptiveAIError):
        generate_scenario_samples("area_shift", 10, 3, {"area_min": 3000.0, "area_max": 1000.0})
    with pytest.raises(AdaptiveAIError):
        generate_scenario_samples("area_shift", 10, 3, {"bogus": 1})


def test_locality_shift_concentrates_valid_locality():
    rows = generate_scenario_samples("locality_shift", 100, 7)
    for row in rows:
        _assert_valid_v2_row(row)
    kothrud = sum(1 for r in rows if r["location"] == "Kothrud")
    assert kothrud >= 80


def test_locality_shift_custom_params_and_validation():
    rows = generate_scenario_samples(
        "locality_shift", 100, 9, {"locality": "Wakad", "concentration": 1.0}
    )
    assert all(r["location"] == "Wakad" for r in rows)
    with pytest.raises(AdaptiveAIError):
        generate_scenario_samples("locality_shift", 10, 1, {"locality": "Atlantis"})
    with pytest.raises(AdaptiveAIError):
        generate_scenario_samples("locality_shift", 10, 1, {"concentration": 0.0})
    with pytest.raises(AdaptiveAIError):
        generate_scenario_samples("locality_shift", 10, 1, {"concentration": 1.5})


def test_config_shift_uses_valid_shifted_mix():
    rows = generate_scenario_samples("config_shift", 100, 11)
    for row in rows:
        _assert_valid_v2_row(row)
    furnished = sum(1 for r in rows if r["furnishing_status"] == "Fully-Furnished")
    villas = sum(1 for r in rows if r["property_type"] == "Villa")
    assert furnished >= 60
    assert villas >= 50


def test_data_quality_contains_documented_issues():
    rows = generate_scenario_samples("data_quality", 100, 5)
    assert len(rows) == 100
    for row in rows:
        _assert_valid_v2_row(row, allow_relationship_violations=True)
    texts = [str(sorted(r.items())) for r in rows]
    assert len(set(texts)) < len(texts)  # duplicates present
    floor_bad = sum(1 for r in rows if r["floor_number"] > r["total_floors"])
    bath_bad = sum(1 for r in rows if r["bathrooms"] > r["bedrooms"] + 2)
    assert floor_bad > 0
    assert bath_bad > 0


def test_generation_rejects_bad_inputs():
    with pytest.raises(AdaptiveAIError):
        generate_scenario_samples("nope", 10, 1)
    with pytest.raises(AdaptiveAIError):
        generate_scenario_samples("healthy_baseline", 0, 1)
    with pytest.raises(AdaptiveAIError):
        generate_scenario_samples("healthy_baseline", 1001, 1)
    with pytest.raises(AdaptiveAIError):
        generate_scenario_samples("healthy_baseline", 10, -1)
    with pytest.raises(AdaptiveAIError):
        generate_scenario_samples("healthy_baseline", "10", 1)
