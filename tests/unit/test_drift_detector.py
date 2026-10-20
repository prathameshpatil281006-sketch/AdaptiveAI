"""Focused tests for KS drift detector (Step 26).

Deterministic via fixed random seeds. No database/API/MLflow usage.
"""

import numpy as np
import pandas as pd
import pytest

from app.core.exceptions import AdaptiveAIError
from app.monitoring.drift_detector import DriftDetector


def _frames(shift: float = 0.0, n_ref: int = 200, n_prod: int = 200):
    ref_vals = np.random.RandomState(42).normal(0, 1, n_ref)
    prod_vals = np.random.RandomState(1).normal(shift, 1, n_prod)
    return (
        pd.DataFrame({"f1": ref_vals}),
        pd.DataFrame({"f1": prod_vals}),
    )


def test_similar_distributions_no_drift():
    ref, prod = _frames(shift=0.0)
    result = DriftDetector().detect(ref, prod)
    assert result["overall_drifted"] is False
    assert result["features"][0]["drifted"] is False


def test_different_distributions_drift_detected():
    ref, prod = _frames(shift=5.0)
    result = DriftDetector().detect(ref, prod)
    assert result["overall_drifted"] is True
    assert result["features"][0]["drifted"] is True
    assert result["features"][0]["p_value"] < 0.05


def test_only_one_feature_drifts():
    ref_base, prod_same = _frames(shift=0.0)
    _, prod_shifted = _frames(shift=5.0)
    ref = pd.DataFrame({"f1": ref_base["f1"], "f2": ref_base["f1"]})
    prod = pd.DataFrame({"f1": prod_same["f1"], "f2": prod_shifted["f1"]})
    result = DriftDetector().detect(ref, prod)
    by_name = {f["feature_name"]: f for f in result["features"]}
    assert by_name["f1"]["drifted"] is False
    assert by_name["f2"]["drifted"] is True
    assert result["overall_drifted"] is True


def test_statistics_are_python_floats_and_serializable():
    import json

    ref, prod = _frames(shift=5.0)
    result = DriftDetector().detect(ref, prod)
    assert isinstance(result["alpha"], float)
    assert isinstance(result["overall_drifted"], bool)
    for feat in result["features"]:
        assert isinstance(feat["statistic"], float)
        assert isinstance(feat["p_value"], float)
        assert isinstance(feat["drifted"], bool)
        assert isinstance(feat["feature_name"], str)
    json.dumps(result)


def test_alpha_is_configurable():
    ref, prod = _frames(shift=0.0)  # p ~ 0.27 deterministically
    strict = DriftDetector(alpha=0.05).detect(ref, prod)
    loose = DriftDetector(alpha=0.5).detect(ref, prod)
    assert strict["alpha"] == pytest.approx(0.05)
    assert loose["alpha"] == pytest.approx(0.5)
    assert strict["features"][0]["drifted"] is False
    assert loose["features"][0]["drifted"] is True
    with pytest.raises(AdaptiveAIError):
        DriftDetector(alpha=0.0)
    with pytest.raises(AdaptiveAIError):
        DriftDetector(alpha=1.5)


def test_overall_drifted_correct():
    ref, prod = _frames(shift=0.0)
    assert DriftDetector().detect(ref, prod)["overall_drifted"] is False
    ref2, prod2 = _frames(shift=5.0)
    assert DriftDetector().detect(ref2, prod2)["overall_drifted"] is True


def test_missing_feature_column_rejected():
    ref = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    prod = pd.DataFrame({"other": [1.0, 2.0, 3.0]})
    with pytest.raises(AdaptiveAIError):
        DriftDetector().detect(ref, prod, features=["f1"])
    with pytest.raises(AdaptiveAIError):
        DriftDetector().detect(ref, prod)


def test_mismatched_feature_sets_rejected():
    ref = pd.DataFrame({"f1": np.random.RandomState(0).normal(0, 1, 50)})
    prod = pd.DataFrame(
        {
            "f1": np.random.RandomState(1).normal(0, 1, 50),
            "f2": np.random.RandomState(2).normal(0, 1, 50),
        }
    )
    with pytest.raises(AdaptiveAIError):
        DriftDetector().detect(ref, prod)


def test_non_numeric_values_rejected():
    ref = pd.DataFrame({"f1": ["a", "b", "c"]})
    prod = pd.DataFrame({"f1": ["a", "b", "c"]})
    with pytest.raises(AdaptiveAIError):
        DriftDetector(min_production_rows=1).detect(ref, prod)


def test_nan_values_rejected():
    ref = pd.DataFrame({"f1": np.random.RandomState(0).normal(0, 1, 50)})
    prod = pd.DataFrame({"f1": np.random.RandomState(1).normal(0, 1, 50)})
    prod.loc[0, "f1"] = float("nan")
    with pytest.raises(AdaptiveAIError):
        DriftDetector().detect(ref, prod)


def test_empty_production_handled_as_insufficient():
    ref, _ = _frames()
    empty = pd.DataFrame(columns=["f1"])
    result = DriftDetector().detect(ref, empty)
    assert result["overall_drifted"] is False
    assert result["sufficient_data"] is False
    assert result["status"] == "insufficient_data"
    assert result["features"] == []
    assert result["production_rows"] == 0


def test_below_minimum_rows_handled_as_insufficient():
    ref, _ = _frames()
    small = pd.DataFrame({"f1": np.random.RandomState(1).normal(5, 1, 10)})
    result = DriftDetector(min_production_rows=30).detect(ref, small)
    assert result["overall_drifted"] is False
    assert result["sufficient_data"] is False
    assert result["status"] == "insufficient_data"
    assert result["features"] == []
    # Same data with a lower minimum produces real per-feature results.
    ok_result = DriftDetector(min_production_rows=5).detect(ref, small)
    assert ok_result["sufficient_data"] is True
    assert ok_result["status"] == "ok"
    assert len(ok_result["features"]) == 1


def test_sufficient_data_produces_per_feature_results():
    ref, prod = _frames(shift=5.0, n_prod=30)
    result = DriftDetector(min_production_rows=30).detect(ref, prod)
    assert result["sufficient_data"] is True
    assert result["production_rows"] == 30
    assert len(result["features"]) == 1
    feat = result["features"][0]
    assert {"feature_name", "statistic", "p_value", "drifted"} <= set(feat.keys())
    # Step 62 (I-13): Holm-adjusted p-value exposed additively.
    assert "p_value_corrected" in feat
    assert isinstance(feat["p_value_corrected"], float)
    assert result.get("correction") == "holm-bonferroni"


def test_detector_independent_of_house_price():
    rng_ref = np.random.RandomState(10)
    rng_prod = np.random.RandomState(11)
    ref = pd.DataFrame(
        {"sensor_a": rng_ref.normal(0, 1, 100), "sensor_b": rng_ref.normal(0, 1, 100)}
    )
    prod = pd.DataFrame(
        {"sensor_a": rng_prod.normal(0, 1, 100), "sensor_b": rng_prod.normal(4, 1, 100)}
    )
    result = DriftDetector().detect(ref, prod)
    assert {f["feature_name"] for f in result["features"]} == {"sensor_a", "sensor_b"}
