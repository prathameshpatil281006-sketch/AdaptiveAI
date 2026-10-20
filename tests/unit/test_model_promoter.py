"""Focused tests for ModelPromoter execution layer (Step 33).

Execution gating tests: all side effects are mocked. Never touches the
real production model, the real MLflow registry, or the real database.
Filesystem assertions use tmp_path only.
"""

import copy
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.core.config import get_project_root
from app.core.exceptions import AdaptiveAIError, ModelPromotionError
from app.retraining import ModelPromoter, ModelPromotionDecision
from app.retraining.model_evaluator import ModelEvaluator
from app.retraining.model_promoter import (
    FAILED,
    PROMOTED,
    REJECTED,
    REQUIRES_INITIAL_PROMOTION,
)


# --- Helpers ---


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


def _stub_candidate():
    model = MagicMock()
    model.predict = MagicMock(return_value=np.asarray([1.0, 2.0]))
    return model


def _make_promoter(save_return=None, register_return=None, decision=None):
    save_mock = MagicMock(
        return_value=save_return if save_return is not None else Path("/tmp/fake.joblib")
    )
    tracker_mock = MagicMock()
    tracker_mock.register_model = MagicMock(
        return_value=register_return
        if register_return is not None
        else {
            "registered_model_name": "HousePriceModel",
            "version": "3",
            "run_id": "run-abc",
            "source": "runs:/run-abc/model",
        }
    )
    promoter = ModelPromoter(
        decision=decision or ModelPromotionDecision(),
        save_fn=save_mock,
        tracker=tracker_mock,
    )
    return promoter, save_mock, tracker_mock


def _stub_eval_model(y_pred):
    model = MagicMock()
    pred = np.asarray(y_pred, dtype=float)
    model.predict = MagicMock(side_effect=lambda X: pred)
    return model


def _xy_simple():
    X_test = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y_test = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    return X_test, y_test


# --- Decision gating ---


def test_candidate_better_promotes_with_both_side_effects():
    promoter, save_mock, tracker_mock = _make_promoter()
    candidate = _stub_candidate()
    out = promoter.promote(
        candidate,
        _candidate_better_result(),
        candidate_model_name="LinearRegression",
        candidate_run_id="run-abc",
    )
    assert out["decision"] == "promoted"
    assert out["decision"] == PROMOTED
    assert save_mock.call_count == 1
    assert tracker_mock.register_model.call_count == 1
    assert out["model_version"] == "3"
    assert out["registered_model_name"] == "HousePriceModel"


def test_production_better_no_side_effects():
    promoter, save_mock, tracker_mock = _make_promoter()
    out = promoter.promote(
        _stub_candidate(),
        _production_better_result(),
        candidate_model_name="cand",
        candidate_run_id="run-xyz",
    )
    assert out["decision"] == "rejected"
    assert out["decision"] == REJECTED
    save_mock.assert_not_called()
    tracker_mock.register_model.assert_not_called()
    assert out["model_version"] is None


def test_no_production_model_no_side_effects():
    promoter, save_mock, tracker_mock = _make_promoter()
    out = promoter.promote(
        _stub_candidate(),
        _no_production_result(),
        candidate_model_name="cand",
        candidate_run_id="run-xyz",
    )
    assert out["decision"] == "requires_initial_promotion"
    assert out["decision"] == REQUIRES_INITIAL_PROMOTION
    assert out["decision"] != "promoted"
    save_mock.assert_not_called()
    tracker_mock.register_model.assert_not_called()
    assert out["model_version"] is None


def test_malformed_evaluation_no_promotion():
    promoter, save_mock, tracker_mock = _make_promoter()
    with pytest.raises(ModelPromotionError):
        promoter.promote(
            _stub_candidate(),
            {"status": "unknown_status", "comparison": {}},
            candidate_model_name="cand",
            candidate_run_id="run-xyz",
        )
    save_mock.assert_not_called()
    tracker_mock.register_model.assert_not_called()


def test_candidate_better_structurally_invalid_no_promotion():
    promoter, save_mock, tracker_mock = _make_promoter()
    bad = _candidate_better_result()
    del bad["comparison"]
    with pytest.raises(ModelPromotionError):
        promoter.promote(
            _stub_candidate(), bad, candidate_model_name="cand", candidate_run_id="r1"
        )
    save_mock.assert_not_called()
    tracker_mock.register_model.assert_not_called()


# --- Validation ---


@pytest.mark.parametrize("bad_name", [None, "", "   ", 123])
def test_missing_invalid_candidate_model_name_raises(bad_name):
    promoter, save_mock, tracker_mock = _make_promoter()
    with pytest.raises(ModelPromotionError):
        promoter.promote(
            _stub_candidate(),
            _candidate_better_result(),
            candidate_model_name=bad_name,
            candidate_run_id="run-abc",
        )
    save_mock.assert_not_called()
    tracker_mock.register_model.assert_not_called()


def test_missing_candidate_model_raises():
    promoter, save_mock, tracker_mock = _make_promoter()
    with pytest.raises(ModelPromotionError):
        promoter.promote(
            None,
            _candidate_better_result(),
            candidate_model_name="cand",
            candidate_run_id="run-abc",
        )
    save_mock.assert_not_called()
    tracker_mock.register_model.assert_not_called()


@pytest.mark.parametrize("bad_run", [None, "", "   ", 123, ["r1"], {}])
def test_missing_invalid_run_id_raises_no_side_effects(bad_run):
    promoter, save_mock, tracker_mock = _make_promoter()
    with pytest.raises(ModelPromotionError):
        promoter.promote(
            _stub_candidate(),
            _candidate_better_result(),
            candidate_model_name="cand",
            candidate_run_id=bad_run,
        )
    save_mock.assert_not_called()
    tracker_mock.register_model.assert_not_called()


@pytest.mark.parametrize("bad_eval", [None, "candidate_better", [], {"comparison": {}}])
def test_malformed_evaluation_result_raises(bad_eval):
    promoter, save_mock, tracker_mock = _make_promoter()
    with pytest.raises(ModelPromotionError):
        promoter.promote(
            _stub_candidate(),
            bad_eval,
            candidate_model_name="cand",
            candidate_run_id="run-abc",
        )
    save_mock.assert_not_called()
    tracker_mock.register_model.assert_not_called()


def test_promotion_requires_candidate_better_status_even_if_decision_stub_says_promote():
    class _LyingDecision:
        def decide(self, result):
            return {
                "decision": "promote",
                "reason": "lying",
                "primary_metric": "r2",
                "candidate_score": 0.5,
                "production_score": 0.8,
                "improvement": -0.3,
            }

    promoter, save_mock, tracker_mock = _make_promoter(decision=_LyingDecision())
    with pytest.raises(ModelPromotionError):
        promoter.promote(
            _stub_candidate(),
            _production_better_result(),
            candidate_model_name="cand",
            candidate_run_id="run-abc",
        )
    save_mock.assert_not_called()
    tracker_mock.register_model.assert_not_called()


def test_missing_tracker_raises_on_promote_path_only():
    save_mock = MagicMock(return_value=Path("/tmp/x.joblib"))
    promoter = ModelPromoter(
        decision=ModelPromotionDecision(), save_fn=save_mock, tracker=None
    )
    with pytest.raises(ModelPromotionError):
        promoter.promote(
            _stub_candidate(),
            _candidate_better_result(),
            candidate_model_name="cand",
            candidate_run_id="run-abc",
        )
    save_mock.assert_not_called()
    # Gating paths still return without a tracker.
    out = promoter.promote(
        _stub_candidate(),
        _production_better_result(),
        candidate_model_name="cand",
        candidate_run_id="run-abc",
    )
    assert out["decision"] == "rejected"


# --- Persistence ---


def test_save_called_only_after_promote_decision():
    promoter, save_mock, tracker_mock = _make_promoter()
    promoter.promote(
        _stub_candidate(),
        _production_better_result(),
        candidate_model_name="cand",
        candidate_run_id="r1",
    )
    save_mock.assert_not_called()
    promoter.promote(
        _stub_candidate(),
        _candidate_better_result(),
        candidate_model_name="cand",
        candidate_run_id="r1",
    )
    assert save_mock.call_count == 1


def test_correct_candidate_model_and_name_passed_to_persistence(tmp_path: Path):
    target = tmp_path / "LinearRegression.joblib"
    promoter, save_mock, tracker_mock = _make_promoter(save_return=target)
    candidate = _stub_candidate()
    promoter.promote(
        candidate,
        _candidate_better_result(),
        candidate_model_name="LinearRegression",
        candidate_run_id="run-abc",
        models_dir=tmp_path,
    )
    assert save_mock.call_count == 1
    args, kwargs = save_mock.call_args
    assert args[0] is candidate
    assert args[1] == "LinearRegression"
    assert kwargs.get("models_dir") == tmp_path


# --- MLflow ---


def test_register_called_only_after_promote_decision():
    promoter, save_mock, tracker_mock = _make_promoter()
    promoter.promote(
        _stub_candidate(),
        _no_production_result(),
        candidate_model_name="cand",
        candidate_run_id="r1",
    )
    tracker_mock.register_model.assert_not_called()


def test_exact_candidate_run_id_passed_to_registry():
    promoter, save_mock, tracker_mock = _make_promoter()
    promoter.promote(
        _stub_candidate(),
        _candidate_better_result(),
        candidate_model_name="cand",
        candidate_run_id="candidate-run-999",
    )
    _, kwargs = tracker_mock.register_model.call_args
    assert kwargs["run_id"] == "candidate-run-999"


def test_registry_version_propagated_verbatim():
    register_return = {
        "registered_model_name": "HousePriceModel",
        "version": "7",
        "run_id": "run-xyz",
        "source": "runs:/run-xyz/model",
    }
    promoter, save_mock, tracker_mock = _make_promoter(register_return=register_return)
    out = promoter.promote(
        _stub_candidate(),
        _candidate_better_result(),
        candidate_model_name="cand",
        candidate_run_id="run-xyz",
    )
    assert out["decision"] == "promoted"
    assert out["model_version"] == "7"
    assert out["registered_model_name"] == "HousePriceModel"
    assert out["run_id"] == "run-xyz"
    assert out["model_name"] == "cand"


def test_registry_uses_existing_naming_convention_by_default():
    promoter, save_mock, tracker_mock = _make_promoter()
    promoter.promote(
        _stub_candidate(),
        _candidate_better_result(),
        candidate_model_name="cand",
        candidate_run_id="run-1",
    )
    _, kwargs = tracker_mock.register_model.call_args
    assert kwargs["registered_model_name"] == "HousePriceModel"
    assert kwargs["artifact_path"] == "model"
    assert tracker_mock.register_model.call_count == 1


def test_registry_failure_not_reported_as_success():
    tracker_mock = MagicMock()
    tracker_mock.register_model = MagicMock(
        side_effect=ModelPromotionError("registry boom")
    )
    save_mock = MagicMock(return_value=Path("/tmp/x.joblib"))
    promoter = ModelPromoter(
        decision=ModelPromotionDecision(), save_fn=save_mock, tracker=tracker_mock
    )
    with pytest.raises(ModelPromotionError):
        promoter.promote(
            _stub_candidate(),
            _candidate_better_result(),
            candidate_model_name="cand",
            candidate_run_id="run-abc",
        )


def test_registry_invalid_response_not_reported_as_success():
    for bad_response in [
        {},
        {"registered_model_name": "HousePriceModel"},
        {"version": "1"},
        {"registered_model_name": "", "version": "1"},
        {"registered_model_name": "HousePriceModel", "version": ""},
        "not-a-dict",
    ]:
        promoter, save_mock, tracker_mock = _make_promoter(register_return=bad_response)
        with pytest.raises(ModelPromotionError):
            promoter.promote(
                _stub_candidate(),
                _candidate_better_result(),
                candidate_model_name="cand",
                candidate_run_id="run-abc",
            )


# --- Failure behavior ---


def test_persistence_failure_raises_and_skips_registry():
    save_mock = MagicMock(side_effect=ModelPromotionError("disk full"))
    tracker_mock = MagicMock()
    tracker_mock.register_model = MagicMock()
    promoter = ModelPromoter(
        decision=ModelPromotionDecision(), save_fn=save_mock, tracker=tracker_mock
    )
    with pytest.raises(ModelPromotionError):
        promoter.promote(
            _stub_candidate(),
            _candidate_better_result(),
            candidate_model_name="cand",
            candidate_run_id="run-abc",
        )
    tracker_mock.register_model.assert_not_called()


def test_persistence_generic_exception_wrapped_as_promotion_error():
    save_mock = MagicMock(side_effect=OSError("io error"))
    tracker_mock = MagicMock()
    tracker_mock.register_model = MagicMock()
    promoter = ModelPromoter(
        decision=ModelPromotionDecision(), save_fn=save_mock, tracker=tracker_mock
    )
    with pytest.raises(ModelPromotionError, match="persistence failed"):
        promoter.promote(
            _stub_candidate(),
            _candidate_better_result(),
            candidate_model_name="cand",
            candidate_run_id="run-abc",
        )


def test_partial_failure_registry_after_persistence_is_clear():
    save_mock = MagicMock(return_value=Path("/tmp/partial.joblib"))
    tracker_mock = MagicMock()
    tracker_mock.register_model = MagicMock(
        side_effect=RuntimeError("registry down")
    )
    promoter = ModelPromoter(
        decision=ModelPromotionDecision(), save_fn=save_mock, tracker=tracker_mock
    )
    assert save_mock.call_count == 0
    with pytest.raises(ModelPromotionError) as exc_info:
        promoter.promote(
            _stub_candidate(),
            _candidate_better_result(),
            candidate_model_name="cand",
            candidate_run_id="run-abc",
        )
    assert save_mock.call_count == 1  # persistence did happen
    msg = str(exc_info.value).lower()
    assert "registry" in msg
    assert "persistence succeeded" in msg or "partial" in msg
    assert "rollback" in msg


def test_no_false_promoted_after_side_effect_failure():
    save_mock = MagicMock(side_effect=ModelPromotionError("nope"))
    tracker_mock = MagicMock()
    tracker_mock.register_model = MagicMock()
    promoter = ModelPromoter(
        decision=ModelPromotionDecision(), save_fn=save_mock, tracker=tracker_mock
    )
    try:
        promoter.promote(
            _stub_candidate(),
            _candidate_better_result(),
            candidate_model_name="cand",
            candidate_run_id="run-abc",
        )
    except ModelPromotionError:
        pass
    else:
        pytest.fail("expected ModelPromotionError")
    tracker_mock.register_model.assert_not_called()


# --- Safety ---


def test_input_evaluation_dict_not_mutated():
    promoter, _, _ = _make_promoter()
    for factory in (
        _candidate_better_result,
        _production_better_result,
        _no_production_result,
    ):
        original = factory()
        snapshot = copy.deepcopy(original)
        try:
            promoter.promote(
                _stub_candidate(),
                original,
                candidate_model_name="cand",
                candidate_run_id="run-abc",
            )
        except ModelPromotionError:
            pass
        assert original == snapshot


def test_rejected_decisions_zero_calls_across_statuses():
    promoter, save_mock, tracker_mock = _make_promoter()
    for res in (_production_better_result(), _no_production_result()):
        promoter.promote(
            _stub_candidate(), res, candidate_model_name="cand", candidate_run_id="r1"
        )
    save_mock.assert_not_called()
    tracker_mock.register_model.assert_not_called()


def test_result_structure():
    promoter, _, _ = _make_promoter()
    out = promoter.promote(
        _stub_candidate(),
        _candidate_better_result(),
        candidate_model_name="cand",
        candidate_run_id="run-abc",
    )
    assert set(out.keys()) == {
        "decision",
        "model_name",
        "run_id",
        "registered_model_name",
        "model_version",
        "reason",
    }
    assert out["decision"] in {"promoted", "rejected", "requires_initial_promotion", "failed"}
    assert isinstance(out["reason"], str) and out["reason"].strip()

    out2 = promoter.promote(
        _stub_candidate(),
        _production_better_result(),
        candidate_model_name="cand",
        candidate_run_id="run-abc",
    )
    assert set(out2.keys()) == set(out.keys())


def test_integration_with_real_decision_and_evaluator_shapes():
    X_test, y_test = _xy_simple()
    evaluator = ModelEvaluator()
    cand = _stub_eval_model([1.0, 2.0, 3.0, 4.0, 5.0])
    prod = _stub_eval_model([1.0, 1.0, 1.0, 1.0, 1.0])
    better = evaluator.evaluate_candidate(
        cand, prod, X_test, y_test,
        candidate_model_name="cand", production_model_name="prod",
    )
    promoter, save_mock, tracker_mock = _make_promoter()
    out = promoter.promote(
        cand, better, candidate_model_name="cand", candidate_run_id="run-real"
    )
    assert out["decision"] == "promoted"
    assert save_mock.call_count == 1
    assert tracker_mock.register_model.call_count == 1

    worse = evaluator.evaluate_candidate(
        prod, cand, X_test, y_test,
        candidate_model_name="cand", production_model_name="prod",
    )
    out2 = promoter.promote(
        prod, worse, candidate_model_name="cand", candidate_run_id="run-real"
    )
    assert out2["decision"] == "rejected"

    missing = evaluator.evaluate_candidate(
        cand, None, X_test, y_test,
        candidate_model_name="cand", production_model_name=None,
    )
    out3 = promoter.promote(
        cand, missing, candidate_model_name="cand", candidate_run_id="run-real"
    )
    assert out3["decision"] == "requires_initial_promotion"
    # Only the first call promoted.
    assert save_mock.call_count == 1
    assert tracker_mock.register_model.call_count == 1


def test_package_export_pattern():
    from app import retraining as retraining_pkg

    assert "ModelPromoter" in retraining_pkg.__all__
    assert retraining_pkg.ModelPromoter is ModelPromoter


def test_promoter_source_hygiene_no_db_api_scheduler_dashboard():
    source = (
        Path(get_project_root()) / "app" / "retraining" / "model_promoter.py"
    ).read_text(encoding="utf-8")
    # Reuses existing APIs.
    assert "save_house_price_model" in source
    assert "register_model" in source
    assert "ModelPromotionDecision" in source
    # No forbidden side-effect wiring.
    for forbidden in (
        "import sqlite3",
        "sqlite3",
        "CREATE TABLE",
        "retraining_results",
        "adaptive_ai.db",
        "FastAPI(",
        "APScheduler",
        "Celery",
        "streamlit",
        "Streamlit",
        "joblib.dump",
        "joblib.load",
        "transition_model_version_stage",
        "MlflowClient(",
        "create_registered_model",
        "create_model_version",
    ):
        assert forbidden not in source
    # Must not duplicate registry logic or invent rollback.
    assert "rollback" in source.lower()  # documents no-rollback limitation


def test_no_wiring_into_retraining_service_api_scheduler():
    retraining_src = (
        Path(get_project_root()) / "app" / "retraining" / "retraining_service.py"
    ).read_text(encoding="utf-8")
    # Step 34 wires promotion into RetrainingService on an opt-in basis only.
    assert "ModelPromoter" in retraining_src
    assert "promote" in retraining_src
    # No direct persistence/registry duplication inside the service.
    for forbidden in (
        "save_house_price_model",
        "register_model",
        "ensure_registered_model",
        "create_model_version",
        "joblib.dump",
        "joblib.load",
    ):
        assert forbidden not in retraining_src
    api_src = (Path(get_project_root()) / "app" / "api" / "main.py").read_text(
        encoding="utf-8"
    )
    assert "ModelPromoter" not in api_src
    # Step 50 wires the existing promoter through its domain-layer
    # factory only; the API must never invoke promotion behavior
    # directly. Exactly one module reference (that factory import) is
    # allowed, and direct invocation stays forbidden.
    assert api_src.lower().count("model_promoter") == 1
    assert "create_house_price_promoter" in api_src
    # Phase 5 adds an explicit lifecycle promotion path: the API may
    # delegate to RetrainingLifecycleService.promote() (thin adapter),
    # but must never invoke the ModelPromoter execution layer directly.
    assert "lifecycle.promote(" in api_src
    assert "promoter.promote(" not in api_src
    assert "register_model" not in api_src
    scheduler_src = (Path(get_project_root()) / "app" / "scheduler" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert "ModelPromoter" not in scheduler_src


def test_decide_with_mocks_touches_no_real_artifacts(tmp_path: Path):
    workdir = tmp_path / "work"
    workdir.mkdir()
    before = set(p.name for p in workdir.iterdir())
    promoter, _, _ = _make_promoter(save_return=workdir / "cand.joblib")
    promoter.promote(
        _stub_candidate(),
        _candidate_better_result(),
        candidate_model_name="cand",
        candidate_run_id="run-mock",
        models_dir=workdir,
    )
    # Mocked save must not create real files.
    assert set(p.name for p in workdir.iterdir()) == before

    models_dir = Path(get_project_root()) / "models" / "house_price"
    prod_before = (
        set(p.name for p in models_dir.glob("*.joblib")) if models_dir.exists() else set()
    )
    promoter2, _, _ = _make_promoter()
    promoter2.promote(
        _stub_candidate(),
        _production_better_result(),
        candidate_model_name="cand",
        candidate_run_id="run-mock",
    )
    prod_after = (
        set(p.name for p in models_dir.glob("*.joblib")) if models_dir.exists() else set()
    )
    assert prod_before == prod_after
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass


def test_failed_constant_documented_and_error_hierarchy():
    assert FAILED == "failed"
    assert issubclass(ModelPromotionError, AdaptiveAIError)
    with pytest.raises(AdaptiveAIError):
        raise ModelPromotionError("x")
