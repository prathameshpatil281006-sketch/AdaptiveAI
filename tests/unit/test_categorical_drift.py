"""Phase 4 tests: categorical (chi-square) drift detector.

Verifies distribution comparison without numeric encoding, unseen /
missing / zero-count category handling, insufficient-sample behavior,
and Holm correction confined to the categorical family. Uses small
hand-built frames and temporary resources only.
"""

import math

import pandas as pd
import pytest

from app.core.exceptions import AdaptiveAIError
from app.monitoring.categorical_drift import (
    CategoricalDriftDetector,
    population_stability_index,
)


def _frame(values, feature="location"):
    return pd.DataFrame({feature: list(values)})


def test_identical_distributions_no_drift():
    ref = _frame(["A"] * 400 + ["B"] * 300 + ["C"] * 300)
    prod = _frame(["A"] * 40 + ["B"] * 30 + ["C"] * 30)
    result = CategoricalDriftDetector().detect(ref, prod)
    assert result["status"] == "ok"
    assert result["sufficient_data"] is True
    assert result["overall_drifted"] is False
    assert result["method"] == "chi-square"
    assert result["correction"] == "holm-bonferroni"
    entry = result["features"][0]
    assert entry["type"] == "categorical"
    assert entry["drifted"] is False
    assert entry["status"] == "ok"
    assert entry["dof"] >= 1
    assert math.isfinite(entry["effect_size"])


def test_concentrated_shift_detected_with_large_effect():
    ref = _frame(["A"] * 400 + ["B"] * 300 + ["C"] * 300)
    prod = _frame(["A"] * 60)
    result = CategoricalDriftDetector().detect(ref, prod)
    assert result["status"] == "ok"
    assert result["overall_drifted"] is True
    entry = result["features"][0]
    assert entry["drifted"] is True
    assert entry["effect_size"] is not None and entry["effect_size"] > 0.25
    assert entry["p_value_corrected"] is not None
    assert entry["p_value_corrected"] < 0.05


def test_unseen_production_category_handled():
    ref = _frame(["A"] * 500 + ["B"] * 500)
    prod = _frame(["A"] * 30 + ["Z"] * 30)
    result = CategoricalDriftDetector().detect(ref, prod)
    assert result["status"] == "ok"
    entry = result["features"][0]
    assert entry["status"] == "ok"
    assert isinstance(entry["drifted"], bool)
    assert math.isfinite(entry["effect_size"])


def test_reference_only_category_zero_production_count():
    ref = _frame(["A"] * 500 + ["B"] * 500)
    prod = _frame(["A"] * 60)
    result = CategoricalDriftDetector().detect(ref, prod)
    assert result["status"] == "ok"
    assert result["features"][0]["status"] == "ok"


def test_insufficient_rows_returns_insufficient_data():
    ref = _frame(["A"] * 500 + ["B"] * 500)
    prod = _frame(["A"] * 10)
    result = CategoricalDriftDetector(min_production_rows=30).detect(ref, prod)
    assert result["status"] == "insufficient_data"
    assert result["sufficient_data"] is False
    assert result["overall_drifted"] is False
    assert result["features"] == []


def test_degenerate_single_bucket_is_untestable_not_drifted():
    ref = _frame(["A"] * 500)
    prod = _frame(["A"] * 60)
    result = CategoricalDriftDetector().detect(ref, prod)
    assert result["status"] == "ok"
    assert result["overall_drifted"] is False
    entry = result["features"][0]
    assert entry["status"] == "insufficient"
    assert entry["drifted"] is False
    assert entry["p_value"] is None


def test_holm_adjustment_properties():
    ref = _frame(["A"] * 500 + ["B"] * 500)
    prod = _frame(["A"] * 30 + ["B"] * 30)
    other_ref = pd.DataFrame({"furnishing_status": ["X"] * 1000})
    other_prod = pd.DataFrame({"furnishing_status": ["X"] * 40 + ["Y"] * 20})
    frame_ref = pd.concat([ref.rename(columns={"location": "location"}), other_ref], axis=1)
    frame_prod = pd.concat([prod, other_prod], axis=1)
    result = CategoricalDriftDetector().detect(
        frame_ref, frame_prod, features=["location", "furnishing_status"]
    )
    assert len(result["features"]) == 2
    for entry in result["features"]:
        if entry["p_value"] is not None:
            assert 0.0 <= entry["p_value"] <= 1.0
            assert entry["p_value_corrected"] is not None
            # Holm never makes a p-value smaller.
            assert entry["p_value_corrected"] >= entry["p_value"] - 1e-12
            assert entry["p_value_corrected"] <= 1.0


def test_invalid_inputs_raise():
    detector = CategoricalDriftDetector()
    with pytest.raises(AdaptiveAIError):
        detector.detect("nope", _frame(["A"] * 40))
    with pytest.raises(AdaptiveAIError):
        detector.detect(_frame(["A"] * 40), _frame(["A"] * 40), features=[])
    with pytest.raises(AdaptiveAIError):
        detector.detect(pd.DataFrame({"a": []}), _frame(["A"] * 40))
    with pytest.raises(AdaptiveAIError):
        CategoricalDriftDetector(alpha=1.5)
    with pytest.raises(AdaptiveAIError):
        CategoricalDriftDetector(min_production_rows=0)


def test_psi_zero_for_identical_shares():
    psi = population_stability_index({"A": 50, "B": 50}, {"A": 5, "B": 5})
    assert psi == pytest.approx(0.0, abs=1e-9)


def test_psi_large_for_concentration():
    psi = population_stability_index({"A": 900, "B": 100}, {"A": 60, "B": 0})
    assert psi > 0.25


def test_explicit_features_subset():
    ref = pd.DataFrame({"location": ["A"] * 500, "property_type": ["X"] * 500})
    prod = pd.DataFrame({"location": ["A"] * 40, "property_type": ["X"] * 40})
    result = CategoricalDriftDetector().detect(ref, prod, features=["location"])
    assert [f["feature_name"] for f in result["features"]] == ["location"]
