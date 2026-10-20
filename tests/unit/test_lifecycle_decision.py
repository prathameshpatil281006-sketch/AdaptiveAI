"""Phase 5 tests: retraining recommendation rules.

Pure-function tests over crafted monitoring summaries: critical and
broad drift recommend; lone small drifts do not; data-quality failure
and insufficient data block. No I/O.
"""

from app.retraining.decision import recommend_retraining


def _summary(**overrides):
    base = {
        "status": "ok",
        "sufficient_data": True,
        "production_rows": 60,
        "severity": "healthy",
        "drifted_count": 0,
        "drifted_features": [],
        "data_quality_status": "pass",
        "features": [],
    }
    base.update(overrides)
    return base


def _drifted(name, kind="numeric"):
    return {"feature_name": name, "type": kind, "drifted": True}


def test_healthy_summary_not_recommended():
    out = recommend_retraining(_summary(), "LinearRegression", "311")
    assert out["recommended"] is False
    assert out["blocked_by"] == []
    assert out["triggers"] == []
    assert out["production_model"] == "LinearRegression"
    assert out["production_version"] == "311"
    assert out["monitored_prediction_count"] == 60
    assert isinstance(out["reason"], str) and out["reason"]


def test_critical_severity_recommends():
    summary = _summary(
        severity="critical",
        drifted_count=1,
        drifted_features=["location"],
        features=[_drifted("location", "categorical")],
    )
    out = recommend_retraining(summary)
    assert out["recommended"] is True
    assert out["blocked_by"] == []
    assert "categorical-drift:location" in out["triggers"]


def test_broad_shift_recommends():
    summary = _summary(
        severity="warning",
        drifted_count=2,
        drifted_features=["area_sqft", "bedrooms"],
        features=[_drifted("area_sqft"), _drifted("bedrooms")],
    )
    out = recommend_retraining(summary)
    assert out["recommended"] is True
    assert "numeric-drift:area_sqft" in out["triggers"]
    assert "numeric-drift:bedrooms" in out["triggers"]


def test_single_small_drift_does_not_recommend():
    summary = _summary(
        severity="warning",
        drifted_count=1,
        drifted_features=["parking"],
        features=[_drifted("parking")],
    )
    out = recommend_retraining(summary)
    assert out["recommended"] is False
    assert out["blocked_by"] == []
    assert "numeric-drift:parking" in out["triggers"]


def test_data_quality_failure_blocks():
    summary = _summary(
        severity="critical",
        drifted_count=3,
        data_quality_status="fail",
        features=[_drifted("area_sqft")],
    )
    out = recommend_retraining(summary)
    assert out["recommended"] is False
    assert out["blocked_by"] == ["data-quality-failed"]


def test_data_quality_warning_advises_without_blocking():
    summary = _summary(
        severity="critical",
        drifted_count=1,
        data_quality_status="warn",
        features=[_drifted("location", "categorical")],
    )
    out = recommend_retraining(summary)
    assert out["recommended"] is True
    assert out["blocked_by"] == []
    assert "data-quality-warning" in out["triggers"]


def test_insufficient_data_blocks():
    for summary in (
        _summary(status="insufficient_data", sufficient_data=False, production_rows=5),
        _summary(production_rows=10),
        None,
        {"unexpected": "shape"},
    ):
        out = recommend_retraining(summary)
        assert out["recommended"] is False
        assert out["blocked_by"] == ["insufficient-data"]
        assert out["monitored_prediction_count"] in (5, 10, None)


def test_unknown_quality_blocks():
    out = recommend_retraining(_summary(data_quality_status="unknown"))
    assert out["recommended"] is False
    assert out["blocked_by"] == ["data-quality-unknown"]
