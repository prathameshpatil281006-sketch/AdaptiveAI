"""Phase 4 tests: data-quality checks over the monitoring window.

Each rule is exercised with a crafted window: clean data passes,
violations fail/warn with row counts, legacy and malformed rows are
counted (never raised on), and an empty window is unknown. Read-only;
no database involved.
"""

from app.monitoring.data_quality import run_house_price_data_quality


def _row(**overrides):
    row = {
        "area_sqft": 1200.0,
        "bedrooms": 3,
        "bathrooms": 2,
        "parking": 1,
        "location": "Wakad",
        "property_age": 8.0,
        "floor_number": 2,
        "total_floors": 12,
        "furnishing_status": "Semi-Furnished",
        "property_type": "Apartment",
    }
    row.update(overrides)
    return row


def _status_of(result, name):
    return next(c for c in result["checks"] if c["name"] == name)


def test_clean_window_passes():
    result = run_house_price_data_quality([_row(), _row(area_sqft=900.0, bedrooms=2)])
    assert result["status"] == "pass"
    assert result["total_rows"] == 2
    assert result["v2_rows"] == 2
    assert all(c["status"] == "pass" for c in result["checks"])
    assert all(c["affected"] == 0 for c in result["checks"])


def test_empty_window_unknown():
    result = run_house_price_data_quality([])
    assert result["status"] == "unknown"
    assert result["total_rows"] == 0


def test_missing_values_fail():
    result = run_house_price_data_quality([_row(), _row(property_age=None)])
    assert result["status"] == "fail"
    bad = _status_of(result, "missing_values")
    assert bad["status"] == "fail" and bad["affected"] == 1


def test_duplicates_warn_not_fail():
    result = run_house_price_data_quality([_row(), _row()])
    assert result["status"] == "warn"
    dup = _status_of(result, "duplicates")
    assert dup["status"] == "warn" and dup["affected"] == 1


def test_bad_numeric_ranges_fail():
    rows = [
        _row(area_sqft=0),
        _row(parking=5),
        _row(property_age=-1),
        _row(bedrooms=0),
    ]
    result = run_house_price_data_quality(rows)
    assert result["status"] == "fail"
    num = _status_of(result, "numeric_ranges")
    assert num["status"] == "fail" and num["affected"] == 4


def test_unknown_categoricals_fail():
    rows = [_row(location="Atlantis"), _row(furnishing_status="Bare")]
    result = run_house_price_data_quality(rows)
    assert result["status"] == "fail"
    cat = _status_of(result, "categorical_values")
    assert cat["status"] == "fail" and cat["affected"] == 2


def test_broken_relationships_fail():
    rows = [_row(floor_number=13, total_floors=12), _row(bedrooms=2, bathrooms=5)]
    result = run_house_price_data_quality(rows)
    assert result["status"] == "fail"
    rel = _status_of(result, "relationships")
    assert rel["status"] == "fail" and rel["affected"] == 2


def test_out_of_band_area_warns():
    # 3 BHK band is 1100-1600 sqft; 5000 sqft is an outlier heuristic flag.
    result = run_house_price_data_quality([_row(area_sqft=5000.0)])
    assert result["status"] == "warn"
    out = _status_of(result, "outliers")
    assert out["status"] == "warn" and out["affected"] == 1
    assert result["v2_rows"] == 1


def test_legacy_rows_counted_and_warned():
    legacy = {
        "area": 1000.0,
        "bedrooms": 2,
        "bathrooms": 1,
        "property_age": 5.0,
        "parking": 1,
    }
    result = run_house_price_data_quality([_row(), legacy])
    assert result["legacy_rows"] == 1
    assert result["v2_rows"] == 1
    leg = _status_of(result, "legacy_schema")
    assert leg["status"] == "warn" and leg["affected"] == 1
    # Legacy rows are reported, not treated as v2 failures.
    assert _status_of(result, "numeric_ranges")["affected"] == 0


def test_malformed_rows_fail_schema():
    result = run_house_price_data_quality([_row(), "not-a-dict", {"area": 1.0}])
    assert result["status"] == "fail"
    assert result["malformed_rows"] == 1
    schema = _status_of(result, "schema")
    assert schema["status"] == "fail" and schema["affected"] == 2


def test_messages_carry_counts():
    result = run_house_price_data_quality([_row(parking=9)])
    num = _status_of(result, "numeric_ranges")
    assert "1 v2 row(s)" in num["message"]
    for check in result["checks"]:
        assert isinstance(check["message"], str) and check["message"]
