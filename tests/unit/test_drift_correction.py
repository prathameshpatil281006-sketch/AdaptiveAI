"""Step 62 (I-13) focused tests: Holm-Bonferroni multiple-testing correction.

No database/API/MLflow usage. Deterministic; KS is mocked where exact
p-values are required, real KS is used for end-to-end sanity.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from app.monitoring.drift_detector import (
    CORRECTION_METHOD,
    DriftDetector,
    holm_adjusted_pvalues,
)


def _frames(shift=0.0, n_ref=200, n_prod=200):
    ref = np.random.RandomState(42).normal(0, 1, n_ref)
    prod = np.random.RandomState(1).normal(shift, 1, n_prod)
    return pd.DataFrame({"f1": ref}), pd.DataFrame({"f1": prod})


def _mock_ks(p_values, statistic=0.1):
    vals = list(p_values)

    def _fake(a, b, alternative="two-sided"):
        p = vals.pop(0) if vals else 1.0
        return SimpleNamespace(statistic=statistic, pvalue=p)

    return _fake


def _five_feature_frames(n=100):
    rng = np.random.RandomState(0)
    cols = {f"f{i}": rng.normal(0, 1, n) for i in range(1, 6)}
    ref = pd.DataFrame(cols)
    rng2 = np.random.RandomState(1)
    cols2 = {f"f{i}": rng2.normal(0, 1, n) for i in range(1, 6)}
    return ref, pd.DataFrame(cols2)


def test_holm_math_known_values():
    # m=5, smallest raw 0.011 -> adjusted 0.055 -> not significant at 0.05.
    adj = holm_adjusted_pvalues([0.011, 0.5, 0.6, 0.7, 0.8])
    assert adj[0] == abs(0.011 * 5) / 1  # 0.055
    assert abs(adj[0] - 0.055) < 1e-12
    # Step-down monotonicity: [0.005, 0.02, 0.5, 0.6, 0.7]
    adj2 = holm_adjusted_pvalues([0.005, 0.02, 0.5, 0.6, 0.7])
    assert abs(adj2[0] - 0.025) < 1e-12  # 0.005*5
    assert abs(adj2[1] - 0.08) < 1e-12  # max(0.025, 0.02*4)
    # Single test reduces to identity.
    assert holm_adjusted_pvalues([0.03]) == [0.03]
    # Empty input.
    assert holm_adjusted_pvalues([]) == []


def test_borderline_raw_p_suppressed_by_correction():
    ref, prod = _five_feature_frames()
    with patch(
        "app.monitoring.drift_detector.ks_2samp",
        side_effect=_mock_ks([0.03, 0.5, 0.6, 0.7, 0.8]),
    ):
        result = DriftDetector().detect(ref, prod)
    assert result["correction"] == CORRECTION_METHOD
    by_name = {f["feature_name"]: f for f in result["features"]}
    # Raw 0.03 < 0.05 would have drifted before I-13; Holm gives 0.15.
    assert by_name["f1"]["p_value"] == 0.03
    assert abs(by_name["f1"]["p_value_corrected"] - 0.15) < 1e-12
    assert by_name["f1"]["drifted"] is False
    assert result["overall_drifted"] is False


def test_clearly_significant_drift_still_detected():
    ref, prod = _five_feature_frames()
    with patch(
        "app.monitoring.drift_detector.ks_2samp",
        side_effect=_mock_ks([1e-10, 0.5, 0.6, 0.7, 0.8]),
    ):
        result = DriftDetector().detect(ref, prod)
    by_name = {f["feature_name"]: f for f in result["features"]}
    assert by_name["f1"]["drifted"] is True
    assert by_name["f1"]["p_value_corrected"] < 0.05
    assert result["overall_drifted"] is True


def test_no_drift_remains_no_drift():
    ref, prod = _five_feature_frames()
    with patch(
        "app.monitoring.drift_detector.ks_2samp",
        side_effect=_mock_ks([0.5, 0.6, 0.7, 0.8, 0.9]),
    ):
        result = DriftDetector().detect(ref, prod)
    assert result["overall_drifted"] is False
    assert all(f["drifted"] is False for f in result["features"])


def test_family_wise_step_down_partial_rejection():
    ref, prod = _five_feature_frames()
    with patch(
        "app.monitoring.drift_detector.ks_2samp",
        side_effect=_mock_ks([0.005, 0.02, 0.5, 0.6, 0.7]),
    ):
        result = DriftDetector().detect(ref, prod)
    by_name = {f["feature_name"]: f for f in result["features"]}
    assert by_name["f1"]["drifted"] is True  # adj 0.025
    assert by_name["f2"]["drifted"] is False  # adj 0.08
    assert result["overall_drifted"] is True


def test_insufficient_data_behavior_preserved():
    ref, _ = _frames()
    empty = pd.DataFrame(columns=["f1"])
    result = DriftDetector().detect(ref, empty)
    assert result["overall_drifted"] is False
    assert result["sufficient_data"] is False
    assert result["status"] == "insufficient_data"
    assert result["features"] == []
    assert result["production_rows"] == 0


def test_non_finite_p_values_handled_safely():
    ref, prod = _five_feature_frames()
    with patch(
        "app.monitoring.drift_detector.ks_2samp",
        side_effect=_mock_ks([float("nan"), float("inf"), 0.6, 0.7, 0.8]),
    ):
        result = DriftDetector().detect(ref, prod)
    by_name = {f["feature_name"]: f for f in result["features"]}
    assert by_name["f1"]["p_value_corrected"] == 1.0
    assert by_name["f1"]["drifted"] is False
    assert result["overall_drifted"] is False


def test_deterministic_behavior():
    ref, prod = _five_feature_frames()
    with patch(
        "app.monitoring.drift_detector.ks_2samp",
        side_effect=_mock_ks([0.03, 0.5, 0.6, 0.7, 0.8]),
    ):
        r1 = DriftDetector().detect(ref, prod)
    with patch(
        "app.monitoring.drift_detector.ks_2samp",
        side_effect=_mock_ks([0.03, 0.5, 0.6, 0.7, 0.8]),
    ):
        r2 = DriftDetector().detect(ref, prod)
    assert r1 == r2


def test_real_ks_end_to_end_sanity():
    ref, prod = _frames(shift=0.0)
    ok_result = DriftDetector().detect(ref, prod)
    assert ok_result["overall_drifted"] is False
    ref2, prod2 = _frames(shift=5.0)
    drift_result = DriftDetector().detect(ref2, prod2)
    assert drift_result["overall_drifted"] is True
