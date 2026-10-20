"""Phase 4 tests: deterministic severity classification.

Thresholds are project conventions documented in
``app/monitoring/severity.py``: KS D >= 0.30 and PSI >= 0.25 read as
large effects; 3+ drifted features read as a broad shift. Pure
functions; no I/O.
"""

from app.monitoring.severity import (
    classify_drift_severity,
    classify_overall_health,
)


def _num(name, drifted, statistic):
    return {"feature_name": name, "type": "numeric", "drifted": drifted, "statistic": statistic}


def _cat(name, drifted, psi):
    return {"feature_name": name, "type": "categorical", "drifted": drifted, "effect_size": psi}


def test_healthy_when_nothing_drifted():
    out = classify_drift_severity(
        numeric_features=[_num("area_sqft", False, 0.05)],
        categorical_features=[_cat("location", False, 0.02)],
        sufficient_data=True,
    )
    assert out["severity"] == "healthy"
    assert out["drifted_count"] == 0


def test_unknown_when_insufficient():
    out = classify_drift_severity(sufficient_data=False)
    assert out["severity"] == "unknown"
    assert out["drifted_features"] == []


def test_warning_for_small_numeric_drift():
    out = classify_drift_severity(numeric_features=[_num("area_sqft", True, 0.12)])
    assert out["severity"] == "warning"
    assert out["drifted_features"] == ["area_sqft"]


def test_critical_at_large_numeric_boundary():
    out = classify_drift_severity(numeric_features=[_num("area_sqft", True, 0.30)])
    assert out["severity"] == "critical"
    assert out["large_effect_features"] == ["area_sqft"]


def test_just_below_large_is_warning():
    out = classify_drift_severity(numeric_features=[_num("area_sqft", True, 0.299)])
    assert out["severity"] == "warning"


def test_critical_at_large_psi_boundary():
    out = classify_drift_severity(categorical_features=[_cat("location", True, 0.25)])
    assert out["severity"] == "critical"


def test_critical_for_broad_small_shift():
    out = classify_drift_severity(
        numeric_features=[
            _num("a", True, 0.11),
            _num("b", True, 0.12),
            _num("c", True, 0.13),
        ]
    )
    assert out["severity"] == "critical"
    assert out["drifted_count"] == 3


def test_two_small_drifts_stay_warning():
    out = classify_drift_severity(
        numeric_features=[_num("a", True, 0.11), _num("b", True, 0.12)]
    )
    assert out["severity"] == "warning"


def test_overall_health_combinations():
    assert classify_overall_health("healthy", "pass")["overall_health"] == "healthy"
    assert classify_overall_health("warning", "pass")["overall_health"] == "warning"
    assert classify_overall_health("healthy", "warn")["overall_health"] == "warning"
    assert classify_overall_health("critical", "pass")["overall_health"] == "critical"
    assert classify_overall_health("healthy", "fail")["overall_health"] == "critical"
    assert classify_overall_health("warning", "fail")["overall_health"] == "critical"
    assert classify_overall_health("unknown", "unknown")["overall_health"] == "unknown"
    assert classify_overall_health("unknown", None)["overall_health"] == "unknown"


def test_overall_health_unknown_drift_pass_dq():
    # Drift unknown but quality assessed: unknown, never invented healthy.
    assert classify_overall_health("unknown", "pass")["overall_health"] == "unknown"
