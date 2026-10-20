"""Focused tests for ModelPromotionDecision (Step 32).

Decision-only: never touches real production artifacts, the real
database, or the MLflow registry. All data is in-memory or temporary.
"""

import copy
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.core.config import get_project_root
from app.core.exceptions import AdaptiveAIError, ModelPromotionError
from app.retraining import ModelPromotionDecision
from app.retraining.model_evaluator import ModelEvaluator
from app.retraining.model_promotion import (
    DECISION_PROMOTE,
    DECISION_REJECT,
    DECISION_REQUIRES_INITIAL_PROMOTION,
)


def _candidate_better_result(**overrides):
    base = {
        "candidate": {"model_name": "cand", "metrics": {"mae": 1.0, "rmse": 2.0, "r2": 0.9}},
        "production": {"model_name": "prod", "metrics": {"mae": 2.0, "rmse": 3.0, "r2": 0.7}},
        "comparison": {
            "primary_metric": "r2",
            "candidate_score": 0.9,
            "production_score": 0.7,
            "improvement": 0.2,
            "candidate_better": True,
        },
        "status": "candidate_better",
    }
    if "comparison" in overrides and isinstance(overrides["comparison"], dict):
        merged = dict(base["comparison"])
        merged.update(overrides.pop("comparison"))
        base["comparison"] = merged
    base.update(overrides)
    return base


def _production_better_result(**overrides):
    base = {
        "candidate": {"model_name": "cand", "metrics": {"mae": 2.0, "rmse": 3.0, "r2": 0.5}},
        "production": {"model_name": "prod", "metrics": {"mae": 1.0, "rmse": 2.0, "r2": 0.8}},
        "comparison": {
            "primary_metric": "r2",
            "candidate_score": 0.5,
            "production_score": 0.8,
            "improvement": -0.3,
            "candidate_better": False,
        },
        "status": "production_better",
    }
    if "comparison" in overrides and isinstance(overrides["comparison"], dict):
        merged = dict(base["comparison"])
        merged.update(overrides.pop("comparison"))
        base["comparison"] = merged
    base.update(overrides)
    return base


def _no_production_result(**overrides):
    base = {
        "candidate": {"model_name": "cand", "metrics": {"mae": 1.0, "rmse": 2.0, "r2": 0.85}},
        "production": None,
        "comparison": {
            "primary_metric": "r2",
            "candidate_score": 0.85,
            "production_score": None,
            "improvement": None,
            "candidate_better": False,
        },
        "status": "no_production_model",
    }
    if "comparison" in overrides and isinstance(overrides["comparison"], dict):
        merged = dict(base["comparison"])
        merged.update(overrides.pop("comparison"))
        base["comparison"] = merged
    base.update(overrides)
    return base


def _stub_model(y_pred):
    model = MagicMock()
    pred = np.asarray(y_pred, dtype=float)
    model.predict = MagicMock(side_effect=lambda X: pred)
    return model


def _xy_simple():
    X_test = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y_test = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    return X_test, y_test


# --- Core mapping ---


def test_candidate_better_promotes():
    decider = ModelPromotionDecision()
    out = decider.decide(_candidate_better_result())
    assert out["decision"] == "promote"
    assert out["decision"] == DECISION_PROMOTE
    assert isinstance(out["reason"], str) and out["reason"].strip()
    assert out["primary_metric"] == "r2"
    assert out["candidate_score"] == pytest.approx(0.9)
    assert out["production_score"] == pytest.approx(0.7)
    assert out["improvement"] == pytest.approx(0.2)


def test_production_better_rejects():
    out = ModelPromotionDecision().decide(_production_better_result())
    assert out["decision"] == "reject"
    assert out["decision"] == DECISION_REJECT
    assert isinstance(out["reason"], str) and out["reason"].strip()
    assert out["primary_metric"] == "r2"
    assert out["candidate_score"] == pytest.approx(0.5)
    assert out["production_score"] == pytest.approx(0.8)
    assert out["improvement"] == pytest.approx(-0.3)


def test_no_production_requires_initial_promotion():
    out = ModelPromotionDecision().decide(_no_production_result())
    assert out["decision"] == "requires_initial_promotion"
    assert out["decision"] == DECISION_REQUIRES_INITIAL_PROMOTION
    assert out["decision"] != "promote"
    assert isinstance(out["reason"], str) and out["reason"].strip()
    assert out["primary_metric"] == "r2"
    assert out["candidate_score"] == pytest.approx(0.85)
    assert out["production_score"] is None
    assert out["improvement"] is None


def test_no_production_never_auto_promotes_even_with_high_score():
    res = _no_production_result(comparison={"candidate_score": 0.99})
    out = ModelPromotionDecision().decide(res)
    assert out["decision"] == "requires_initial_promotion"
    assert out["decision"] != "promote"


# --- Candidate_better strict validation ---


def test_candidate_better_flag_false_not_approved():
    res = _candidate_better_result(comparison={"candidate_better": False})
    out = ModelPromotionDecision().decide(res)
    assert out["decision"] != "promote"
    assert out["decision"] == "reject"


@pytest.mark.parametrize("bad_improvement", [0.0, -0.1, -1.5])
def test_candidate_better_non_positive_improvement_not_approved(bad_improvement):
    res = _candidate_better_result(comparison={"improvement": bad_improvement})
    out = ModelPromotionDecision().decide(res)
    assert out["decision"] != "promote"
    assert out["decision"] == "reject"


def test_candidate_better_truthy_int_is_not_approved():
    # Must be exactly True, not 1.
    res = _candidate_better_result(comparison={"candidate_better": 1})
    out = ModelPromotionDecision().decide(res)
    assert out["decision"] == "reject"


def test_missing_comparison_handled_safely():
    decider = ModelPromotionDecision()
    # candidate_better without comparison must never promote.
    res = _candidate_better_result()
    del res["comparison"]
    with pytest.raises(ModelPromotionError):
        decider.decide(res)
    # production_better without comparison must still safely reject.
    res2 = _production_better_result()
    del res2["comparison"]
    out = decider.decide(res2)
    assert out["decision"] == "reject"


def test_comparison_not_dict_handled_safely():
    with pytest.raises(ModelPromotionError):
        ModelPromotionDecision().decide(_candidate_better_result(comparison="bad"))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_metric", [None, "", "   ", 123, ["r2"]])
def test_missing_invalid_primary_metric_handled_safely(bad_metric):
    res = _candidate_better_result(comparison={"primary_metric": bad_metric})
    with pytest.raises(ModelPromotionError):
        ModelPromotionDecision().decide(res)


@pytest.mark.parametrize(
    "field,value",
    [
        ("candidate_score", "0.9"),
        ("candidate_score", None),
        ("candidate_score", True),
        ("candidate_score", float("nan")),
        ("candidate_score", float("inf")),
        ("production_score", "0.7"),
        ("production_score", None),
        ("production_score", False),
        ("production_score", float("nan")),
        ("improvement", "0.2"),
        ("improvement", None),
        ("improvement", True),
        ("improvement", float("nan")),
        ("improvement", float("inf")),
    ],
)
def test_non_numeric_scores_handled_safely(field, value):
    res = _candidate_better_result(comparison={field: value})
    with pytest.raises(ModelPromotionError):
        ModelPromotionDecision().decide(res)


# --- Production_better robustness ---


def test_production_better_malformed_still_rejects():
    decider = ModelPromotionDecision()
    # Inconsistent flag/improvement must not flip to promote.
    res = _production_better_result(
        comparison={"candidate_better": True, "improvement": 0.5}
    )
    out = decider.decide(res)
    assert out["decision"] == "reject"

    res2 = _production_better_result(comparison={"candidate_score": "bad"})
    out2 = decider.decide(res2)
    assert out2["decision"] == "reject"
    assert out2["candidate_score"] is None


def test_production_better_equal_scores_rejects():
    res = _production_better_result(
        comparison={
            "candidate_score": 0.7,
            "production_score": 0.7,
            "improvement": 0.0,
            "candidate_better": False,
        }
    )
    out = ModelPromotionDecision().decide(res)
    assert out["decision"] == "reject"


# --- Unknown status ---


@pytest.mark.parametrize("bad_status", ["unknown", "better", "", None, 123])
def test_unknown_status_raises_adaptive_ai_error(bad_status):
    res = _candidate_better_result(status=bad_status)
    with pytest.raises(AdaptiveAIError):
        ModelPromotionDecision().decide(res)
    with pytest.raises(ModelPromotionError):
        ModelPromotionDecision().decide(res)


def test_missing_status_raises():
    res = _candidate_better_result()
    del res["status"]
    with pytest.raises(AdaptiveAIError):
        ModelPromotionDecision().decide(res)


def test_non_dict_input_raises():
    with pytest.raises(AdaptiveAIError):
        ModelPromotionDecision().decide("candidate_better")  # type: ignore[arg-type]
    with pytest.raises(AdaptiveAIError):
        ModelPromotionDecision().decide(None)  # type: ignore[arg-type]


def test_model_promotion_error_is_adaptive_ai_error():
    assert issubclass(ModelPromotionError, AdaptiveAIError)


# --- Contract: no mutation, determinism, structure ---


def test_input_not_mutated():
    decider = ModelPromotionDecision()
    for factory in (
        _candidate_better_result,
        _production_better_result,
        _no_production_result,
    ):
        original = factory()
        snapshot = copy.deepcopy(original)
        decider.decide(original)
        assert original == snapshot


def test_result_structure_and_determinism():
    decider = ModelPromotionDecision()
    res = _candidate_better_result()
    first = decider.decide(res)
    second = decider.decide(res)
    assert first == second
    assert set(first.keys()) == {
        "decision",
        "reason",
        "primary_metric",
        "candidate_score",
        "production_score",
        "improvement",
    }
    assert first["decision"] in {
        "promote",
        "reject",
        "requires_initial_promotion",
    }
    # Fresh dict each call.
    assert first is not second


def test_never_promotes_on_candidate_existence_alone():
    decider = ModelPromotionDecision()
    # A result with a strong candidate block but production_better status.
    res = _production_better_result()
    res["candidate"] = {
        "model_name": "super",
        "metrics": {"mae": 0.1, "rmse": 0.2, "r2": 0.999},
    }
    out = decider.decide(res)
    assert out["decision"] == "reject"


# --- Real evaluator shape accepted directly ---


def test_existing_evaluator_result_shape_accepted_directly():
    X_test, y_test = _xy_simple()
    evaluator = ModelEvaluator()

    cand = _stub_model([1.0, 2.0, 3.0, 4.0, 5.0])
    prod = _stub_model([1.0, 1.0, 1.0, 1.0, 1.0])
    better = evaluator.evaluate_candidate(
        cand, prod, X_test, y_test,
        candidate_model_name="cand", production_model_name="prod",
    )
    out = ModelPromotionDecision().decide(better)
    assert out["decision"] == "promote"
    assert out["primary_metric"] == better["comparison"]["primary_metric"]
    assert out["candidate_score"] == pytest.approx(better["comparison"]["candidate_score"])
    assert out["production_score"] == pytest.approx(better["comparison"]["production_score"])
    assert out["improvement"] == pytest.approx(better["comparison"]["improvement"])

    cand2 = _stub_model([1.0, 1.0, 1.0, 1.0, 1.0])
    prod2 = _stub_model([1.0, 2.0, 3.0, 4.0, 5.0])
    worse = evaluator.evaluate_candidate(
        cand2, prod2, X_test, y_test,
        candidate_model_name="cand", production_model_name="prod",
    )
    assert ModelPromotionDecision().decide(worse)["decision"] == "reject"

    missing = evaluator.evaluate_candidate(
        cand, None, X_test, y_test,
        candidate_model_name="cand", production_model_name=None,
    )
    assert missing["status"] == "no_production_model"
    out3 = ModelPromotionDecision().decide(missing)
    assert out3["decision"] == "requires_initial_promotion"


def test_package_export_pattern():
    from app import retraining as retraining_pkg

    assert "ModelPromotionDecision" in retraining_pkg.__all__
    assert retraining_pkg.ModelPromotionDecision is ModelPromotionDecision


# --- No side effects ---


def test_promotion_source_has_no_side_effect_wiring():
    source = (
        Path(get_project_root()) / "app" / "retraining" / "model_promotion.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "save_house_price_model",
        "load_house_price_model",
        "register_model",
        "ensure_registered_model",
        "create_model_version",
        "joblib.dump",
        "joblib.load",
        "import sqlite3",
        "sqlite3",
        "mlflow",
        "Mlflow",
        "APScheduler",
        "Celery",
        "streamlit",
        "Streamlit",
        "FastAPI(",
    ):
        assert forbidden not in source


def test_decide_creates_no_files_and_touches_nothing(tmp_path: Path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    before = set(p.name for p in workdir.iterdir())
    decider = ModelPromotionDecision()
    decider.decide(_candidate_better_result())
    decider.decide(_production_better_result())
    decider.decide(_no_production_result())
    assert set(p.name for p in workdir.iterdir()) == before

    models_dir = Path(get_project_root()) / "models" / "house_price"
    prod_before = (
        set(p.name for p in models_dir.glob("*.joblib")) if models_dir.exists() else set()
    )
    decider.decide(_candidate_better_result())
    prod_after = (
        set(p.name for p in models_dir.glob("*.joblib")) if models_dir.exists() else set()
    )
    assert prod_before == prod_after
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass
