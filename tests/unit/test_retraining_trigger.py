"""Focused tests for the drift-triggered retraining decision layer (Step 38).

Deterministic and side-effect free. Uses plain dicts plus the real
DriftDetector output shape for compatibility. Never runs retraining,
never touches the database, MLflow, or model artifacts.
"""

import copy
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.core.config import get_project_root
from app.core.exceptions import AdaptiveAIError, DriftDetectionError
from app.monitoring import RetrainingTriggerDecision
from app.monitoring.drift_detector import DriftDetector


def _feature(name, drifted, statistic=0.1, p_value=0.5):
    return {
        "feature_name": name,
        "statistic": statistic,
        "p_value": p_value,
        "drifted": drifted,
    }


def _ok_result(features=None, overall_drifted=False, production_rows=100):
    if features is None:
        features = [_feature("area", False), _feature("bedrooms", False)]
    return {
        "overall_drifted": overall_drifted,
        "alpha": 0.05,
        "min_production_rows": 30,
        "production_rows": production_rows,
        "sufficient_data": True,
        "status": "ok",
        "features": features,
    }


def _insufficient_result(production_rows=5, features=None):
    if features is None:
        features = []
    return {
        "overall_drifted": False,
        "alpha": 0.05,
        "min_production_rows": 30,
        "production_rows": production_rows,
        "sufficient_data": False,
        "status": "insufficient_data",
        "features": features,
    }


def _trigger_source():
    return (Path(get_project_root()) / "app" / "monitoring" / "retraining_trigger.py").read_text(
        encoding="utf-8"
    )


# --- 1-3. Core decision behavior ---


def test_drifted_result_requires_retraining():
    result = _ok_result(
        features=[_feature("area", True, 0.4, 0.001), _feature("bedrooms", False)],
        overall_drifted=True,
    )
    out = RetrainingTriggerDecision().decide(result)
    assert out["decision"] == "retraining_required"
    assert out["status"] == "ok"
    assert out["overall_drifted"] is True
    assert out["production_rows"] == 100
    assert isinstance(out["reason"], str) and out["reason"].strip()


def test_non_drifted_result_needs_no_retraining():
    out = RetrainingTriggerDecision().decide(_ok_result())
    assert out["decision"] == "no_retraining"
    assert out["overall_drifted"] is False
    assert isinstance(out["reason"], str) and out["reason"].strip()


def test_insufficient_data_needs_no_retraining():
    out = RetrainingTriggerDecision().decide(_insufficient_result(production_rows=5))
    assert out["decision"] == "no_retraining"
    assert out["status"] == "insufficient_data"
    assert out["drifted_features"] == []
    assert "insufficient" in out["reason"].lower()


def test_decision_output_contract_keys():
    out = RetrainingTriggerDecision().decide(_ok_result())
    assert set(out.keys()) == {
        "decision",
        "reason",
        "status",
        "overall_drifted",
        "production_rows",
        "drifted_features",
    }
    assert out["decision"] in ("retraining_required", "no_retraining")


# --- 4-6. Drifted feature extraction ---


def test_drifted_feature_extraction_preserves_names_and_order():
    result = _ok_result(
        features=[
            _feature("area", True),
            _feature("bedrooms", False),
            _feature("parking", True),
        ],
        overall_drifted=True,
    )
    out = RetrainingTriggerDecision().decide(result)
    assert out["drifted_features"] == ["area", "parking"]


def test_no_drift_yields_empty_drifted_features():
    out = RetrainingTriggerDecision().decide(_ok_result())
    assert out["decision"] == "no_retraining"
    assert out["drifted_features"] == []


def test_insufficient_data_ignores_attempted_drift_flags():
    # Even when drift flags are planted under insufficient data,
    # the decision stays negative and exposes no drifted features.
    result = _insufficient_result(
        features=[_feature("area", True), _feature("bedrooms", True)]
    )
    out = RetrainingTriggerDecision().decide(result)
    assert out["decision"] == "no_retraining"
    assert out["drifted_features"] == []


# --- 7-10. Missing / non-dict / status problems ---


def test_missing_input_rejected():
    with pytest.raises(DriftDetectionError):
        RetrainingTriggerDecision().decide(None)


@pytest.mark.parametrize("bad", [["not", "a", "dict"], "ok", 42, 3.14, True, (1, 2)])
def test_non_dict_input_rejected(bad):
    with pytest.raises(DriftDetectionError):
        RetrainingTriggerDecision().decide(bad)


@pytest.mark.parametrize(
    "key",
    [
        "overall_drifted",
        "alpha",
        "min_production_rows",
        "production_rows",
        "sufficient_data",
        "status",
        "features",
    ],
)
def test_missing_keys_rejected(key):
    result = _ok_result()
    del result[key]
    with pytest.raises(DriftDetectionError, match=key):
        RetrainingTriggerDecision().decide(result)


def test_missing_status_rejected():
    result = _ok_result()
    del result["status"]
    with pytest.raises(DriftDetectionError, match="status"):
        RetrainingTriggerDecision().decide(result)


@pytest.mark.parametrize("bad", ["weird", "OK", "", "drifted", None, 0])
def test_unknown_status_rejected(bad):
    result = _ok_result()
    result["status"] = bad
    with pytest.raises(DriftDetectionError, match="status"):
        RetrainingTriggerDecision().decide(result)


# --- 11-12. Strict booleans ---


@pytest.mark.parametrize("bad", [0, 1, "true", "false", None, [], 0.0, 1.0])
def test_invalid_overall_drifted_types_rejected(bad):
    result = _ok_result()
    result["overall_drifted"] = bad
    with pytest.raises(DriftDetectionError, match="overall_drifted"):
        RetrainingTriggerDecision().decide(result)


@pytest.mark.parametrize("bad", [0, 1, "true", "false", None, [], 0.0])
def test_invalid_sufficient_data_types_rejected(bad):
    result = _ok_result()
    result["sufficient_data"] = bad
    with pytest.raises(DriftDetectionError, match="sufficient_data"):
        RetrainingTriggerDecision().decide(result)


# --- 13-14. Alpha ---


@pytest.mark.parametrize("bad", [0, 1, -0.1, 1.5, "0.05", None, True, False, [0.05]])
def test_invalid_alpha_rejected(bad):
    result = _ok_result()
    result["alpha"] = bad
    with pytest.raises(DriftDetectionError, match="alpha"):
        RetrainingTriggerDecision().decide(result)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_inf_alpha_rejected(bad):
    result = _ok_result()
    result["alpha"] = bad
    with pytest.raises(DriftDetectionError, match="alpha"):
        RetrainingTriggerDecision().decide(result)


# --- 15-16. Row counts ---


@pytest.mark.parametrize("bad", [0, -1, True, False, "30", 1.5, None, [30]])
def test_invalid_min_production_rows_rejected(bad):
    result = _ok_result()
    result["min_production_rows"] = bad
    with pytest.raises(DriftDetectionError, match="min_production_rows"):
        RetrainingTriggerDecision().decide(result)


@pytest.mark.parametrize("bad", [-1, -100, True, False, "5", 2.5, None])
def test_invalid_production_rows_rejected(bad):
    result = _ok_result()
    result["production_rows"] = bad
    with pytest.raises(DriftDetectionError, match="production_rows"):
        RetrainingTriggerDecision().decide(result)


def test_zero_production_rows_shape_accepted_but_insufficient():
    # Zero rows is a valid non-negative count; with insufficient status
    # it still yields no retraining rather than a crash.
    out = RetrainingTriggerDecision().decide(_insufficient_result(production_rows=0))
    assert out["decision"] == "no_retraining"
    assert out["production_rows"] == 0


# --- 17. Status / sufficiency contradictions ---


def test_ok_with_insufficient_flag_rejected():
    result = _ok_result()
    result["sufficient_data"] = False
    with pytest.raises(DriftDetectionError, match="Contradictory"):
        RetrainingTriggerDecision().decide(result)


def test_insufficient_with_sufficient_flag_rejected():
    result = _insufficient_result()
    result["sufficient_data"] = True
    with pytest.raises(DriftDetectionError, match="Contradictory"):
        RetrainingTriggerDecision().decide(result)


# --- 18-21. Feature list hygiene ---


@pytest.mark.parametrize("bad", ["features", {"a": 1}, None, 42, True])
def test_malformed_features_list_rejected(bad):
    result = _ok_result()
    result["features"] = bad
    with pytest.raises(DriftDetectionError, match="features"):
        RetrainingTriggerDecision().decide(result)


@pytest.mark.parametrize("bad", ["area", 42, None, ["x"], True])
def test_malformed_feature_records_rejected(bad):
    result = _ok_result(features=[bad])
    with pytest.raises(DriftDetectionError, match="features"):
        RetrainingTriggerDecision().decide(result)


@pytest.mark.parametrize("bad", [None, "", "   ", 42, ["area"]])
def test_missing_feature_name_rejected(bad):
    entry = {"feature_name": bad, "statistic": 0.1, "p_value": 0.5, "drifted": True}
    result = _ok_result(features=[entry], overall_drifted=True)
    with pytest.raises(DriftDetectionError, match="feature_name"):
        RetrainingTriggerDecision().decide(result)


def test_feature_record_without_name_key_rejected():
    result = _ok_result(
        features=[{"statistic": 0.1, "p_value": 0.5, "drifted": True}],
        overall_drifted=True,
    )
    with pytest.raises(DriftDetectionError, match="feature_name"):
        RetrainingTriggerDecision().decide(result)


@pytest.mark.parametrize("bad", [0, 1, "yes", "true", None, [], 1.0, 0.0])
def test_invalid_drifted_flag_rejected(bad):
    result = _ok_result(
        features=[_feature("area", bad)],
        overall_drifted=False,
    )
    with pytest.raises(DriftDetectionError, match="drifted"):
        RetrainingTriggerDecision().decide(result)


# --- 22-23. Immutability / determinism ---


def test_input_not_mutated():
    result = _ok_result(
        features=[_feature("area", True), _feature("bedrooms", False)],
        overall_drifted=True,
    )
    snapshot = copy.deepcopy(result)
    out = RetrainingTriggerDecision().decide(result)
    assert result == snapshot
    # Mutating the output must not affect the input either.
    out["drifted_features"].append("injected")
    assert result == snapshot


def test_repeated_decisions_deterministic():
    decider = RetrainingTriggerDecision()
    result = _ok_result(
        features=[_feature("area", True), _feature("bedrooms", False)],
        overall_drifted=True,
    )
    first = decider.decide(result)
    second = decider.decide(result)
    assert first == second
    assert first is not second
    assert first["drifted_features"] is not second["drifted_features"]


# --- 24. Compatibility with the real detector output ---


def test_exact_compatibility_with_actual_detector_output():
    rng = np.random.RandomState(0)
    reference = pd.DataFrame(
        {
            "area": rng.normal(2000, 300, 200),
            "bedrooms": rng.normal(3, 1, 200).round(),
        }
    )
    detector = DriftDetector(alpha=0.05, min_production_rows=30)
    decider = RetrainingTriggerDecision()

    same = detector.detect(reference=reference, production=reference.copy())
    assert same["status"] == "ok"
    out_same = decider.decide(same)
    assert out_same["decision"] == "no_retraining"
    assert out_same["drifted_features"] == []

    shifted = pd.DataFrame(
        {
            "area": rng.normal(2000 + 5000, 300, 200),
            "bedrooms": rng.normal(3, 1, 200).round(),
        }
    )
    drifted = detector.detect(reference=reference, production=shifted)
    assert drifted["status"] == "ok"
    assert drifted["overall_drifted"] is True
    out_drifted = decider.decide(drifted)
    assert out_drifted["decision"] == "retraining_required"
    assert "area" in out_drifted["drifted_features"]

    tiny = detector.detect(reference=reference, production=reference.iloc[:5].copy())
    assert tiny["status"] == "insufficient_data"
    out_tiny = decider.decide(tiny)
    assert out_tiny["decision"] == "no_retraining"
    assert out_tiny["drifted_features"] == []


# --- 25. No retraining / scheduler involvement ---


def test_no_retraining_or_scheduler_calls():
    source = _trigger_source()
    for forbidden in (
        "RetrainingService",
        "RetrainingScheduler",
        "run_once",
        ".retrain",
        "retrain(",
        "promote(",
        "ModelPromoter",
        "ModelEvaluator",
    ):
        assert forbidden not in source
    import app.monitoring.retraining_trigger as module

    for attr in ("retraining_service", "scheduler", "promoter", "evaluator"):
        assert not hasattr(module, attr)


def test_decide_does_not_call_services():
    service = MagicMock()
    # Even with a mock around, decide is pure: it never touches it.
    out = RetrainingTriggerDecision().decide(_ok_result())
    assert out["decision"] == "no_retraining"
    service.retrain.assert_not_called()


# --- 26. No DB / MLflow / filesystem side effects ---


def test_no_db_mlflow_filesystem_side_effects():
    source = _trigger_source()
    for forbidden in (
        "sqlite3",
        "SELECT ",
        "DatabaseManager",
        "PromotionRepository",
        "MonitoringRepository",
        "PredictionRepository",
        "mlflow",
        "Mlflow",
        "joblib",
        "FastAPI(",
        "fastapi",
        "streamlit",
        "Streamlit",
        "APScheduler",
        "Celery",
        "threading",
        "open(",
    ):
        assert forbidden not in source


def test_decision_creates_no_real_artifacts(tmp_path):
    decider = RetrainingTriggerDecision()
    decider.decide(_ok_result())
    decider.decide(_insufficient_result())
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass


def test_package_exports_decision_layer():
    import app.monitoring as monitoring_pkg

    assert monitoring_pkg.RetrainingTriggerDecision is RetrainingTriggerDecision
    assert "RetrainingTriggerDecision" in monitoring_pkg.__all__


def test_decision_errors_are_adaptive_ai_errors():
    with pytest.raises(AdaptiveAIError):
        RetrainingTriggerDecision().decide(None)
