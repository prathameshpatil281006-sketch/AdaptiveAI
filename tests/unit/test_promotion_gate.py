"""Phase 5 tests: promotion eligibility gate (pure function).

Every documented check is exercised: eligibility requires all eight
checks; ties and regressions block; missing baselines fail closed;
unknown quality blocks while warnings pass. No I/O.
"""

from app.retraining.promotion_gate import check_promotion_eligibility

GOOD_CANDIDATE = {"mae": 1.0, "rmse": 2.0, "r2": 0.9}
GOOD_PRODUCTION = {"mae": 2.0, "rmse": 3.0, "r2": 0.8}


def _names(result):
    return [c["name"] for c in result["checks"]]


def test_all_checks_pass_is_eligible():
    result = check_promotion_eligibility(
        run_completed=True,
        artifact_available=True,
        artifact_loadable=True,
        schema_compatible=True,
        candidate_metrics=dict(GOOD_CANDIDATE),
        production_metrics=dict(GOOD_PRODUCTION),
        improvement=0.1,
        data_quality_status="pass",
    )
    assert result["eligible"] is True
    assert _names(result) == [
        "run_completed",
        "artifact_available",
        "artifact_loadable",
        "schema_compatible",
        "metrics_available",
        "production_baseline",
        "regression_tolerance",
        "data_quality",
    ]
    assert all(c["passed"] for c in result["checks"])
    for check in result["checks"]:
        assert isinstance(check["message"], str) and check["message"]


def test_incomplete_run_blocks():
    result = check_promotion_eligibility(
        run_completed=False,
        artifact_available=True,
        artifact_loadable=True,
        schema_compatible=True,
        candidate_metrics=dict(GOOD_CANDIDATE),
        production_metrics=dict(GOOD_PRODUCTION),
        improvement=0.1,
        data_quality_status="pass",
    )
    assert result["eligible"] is False
    by_name = {c["name"]: c for c in result["checks"]}
    assert by_name["run_completed"]["passed"] is False


def test_missing_artifact_blocks():
    result = check_promotion_eligibility(
        run_completed=True,
        artifact_available=False,
        artifact_loadable=False,
        schema_compatible=False,
        candidate_metrics=dict(GOOD_CANDIDATE),
        production_metrics=dict(GOOD_PRODUCTION),
        improvement=0.1,
        data_quality_status="pass",
    )
    assert result["eligible"] is False


def test_schema_incompatible_blocks():
    result = check_promotion_eligibility(
        run_completed=True,
        artifact_available=True,
        artifact_loadable=True,
        schema_compatible=False,
        candidate_metrics=dict(GOOD_CANDIDATE),
        production_metrics=dict(GOOD_PRODUCTION),
        improvement=0.5,
        data_quality_status="pass",
    )
    assert result["eligible"] is False
    by_name = {c["name"]: c for c in result["checks"]}
    assert by_name["schema_compatible"]["passed"] is False
    assert by_name["regression_tolerance"]["passed"] is True


def test_tie_and_regression_block():
    for improvement in (0.0, -0.05):
        result = check_promotion_eligibility(
            run_completed=True,
            artifact_available=True,
            artifact_loadable=True,
            schema_compatible=True,
            candidate_metrics=dict(GOOD_CANDIDATE),
            production_metrics=dict(GOOD_PRODUCTION),
            improvement=improvement,
            data_quality_status="pass",
        )
        assert result["eligible"] is False


def test_missing_baseline_fails_closed():
    result = check_promotion_eligibility(
        run_completed=True,
        artifact_available=True,
        artifact_loadable=True,
        schema_compatible=True,
        candidate_metrics=dict(GOOD_CANDIDATE),
        production_metrics=None,
        improvement=None,
        data_quality_status="pass",
    )
    assert result["eligible"] is False
    by_name = {c["name"]: c for c in result["checks"]}
    assert by_name["production_baseline"]["passed"] is False
    assert "initial deployment" in by_name["production_baseline"]["message"]


def test_non_finite_metrics_block():
    bad = {"mae": float("inf"), "rmse": 2.0, "r2": 0.9}
    result = check_promotion_eligibility(
        run_completed=True,
        artifact_available=True,
        artifact_loadable=True,
        schema_compatible=True,
        candidate_metrics=bad,
        production_metrics=dict(GOOD_PRODUCTION),
        improvement=0.1,
        data_quality_status="pass",
    )
    assert result["eligible"] is False


def test_failed_quality_blocks_warn_passes():
    blocked = check_promotion_eligibility(
        run_completed=True,
        artifact_available=True,
        artifact_loadable=True,
        schema_compatible=True,
        candidate_metrics=dict(GOOD_CANDIDATE),
        production_metrics=dict(GOOD_PRODUCTION),
        improvement=0.1,
        data_quality_status="fail",
    )
    assert blocked["eligible"] is False
    warned = check_promotion_eligibility(
        run_completed=True,
        artifact_available=True,
        artifact_loadable=True,
        schema_compatible=True,
        candidate_metrics=dict(GOOD_CANDIDATE),
        production_metrics=dict(GOOD_PRODUCTION),
        improvement=0.1,
        data_quality_status="warn",
    )
    assert warned["eligible"] is True
    unknown = check_promotion_eligibility(
        run_completed=True,
        artifact_available=True,
        artifact_loadable=True,
        schema_compatible=True,
        candidate_metrics=dict(GOOD_CANDIDATE),
        production_metrics=dict(GOOD_PRODUCTION),
        improvement=0.1,
        data_quality_status=None,
    )
    assert unknown["eligible"] is False


def test_custom_tolerance_respected():
    strict = check_promotion_eligibility(
        run_completed=True,
        artifact_available=True,
        artifact_loadable=True,
        schema_compatible=True,
        candidate_metrics=dict(GOOD_CANDIDATE),
        production_metrics=dict(GOOD_PRODUCTION),
        improvement=0.05,
        data_quality_status="pass",
        regression_tolerance=0.10,
    )
    assert strict["eligible"] is False
    loose = check_promotion_eligibility(
        run_completed=True,
        artifact_available=True,
        artifact_loadable=True,
        schema_compatible=True,
        candidate_metrics=dict(GOOD_CANDIDATE),
        production_metrics=dict(GOOD_PRODUCTION),
        improvement=0.05,
        data_quality_status="pass",
        regression_tolerance=0.01,
    )
    assert loose["eligible"] is True
