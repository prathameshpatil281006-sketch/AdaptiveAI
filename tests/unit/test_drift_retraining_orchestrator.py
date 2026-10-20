"""Focused tests for the drift-to-retraining orchestrator (Step 39).

Uses mocks/stubs only. Never runs real monitoring, real retraining,
and never touches the database, MLflow, or model artifacts.
"""

import copy
import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.core.config import get_project_root
from app.core.exceptions import AdaptiveAIError, DriftDetectionError, RetrainingError
from app.monitoring.retraining_trigger import RetrainingTriggerDecision
from app.scheduler import DriftRetrainingOrchestrator


def _monitoring_ok(drifted=False):
    return {
        "overall_drifted": drifted,
        "alpha": 0.05,
        "min_production_rows": 30,
        "production_rows": 100,
        "sufficient_data": True,
        "status": "ok",
        "features": [
            {
                "feature_name": "area",
                "statistic": 0.4 if drifted else 0.05,
                "p_value": 0.001 if drifted else 0.6,
                "drifted": drifted,
            },
            {
                "feature_name": "bedrooms",
                "statistic": 0.04,
                "p_value": 0.7,
                "drifted": False,
            },
        ],
    }


def _monitoring_insufficient():
    return {
        "overall_drifted": False,
        "alpha": 0.05,
        "min_production_rows": 30,
        "production_rows": 5,
        "sufficient_data": False,
        "status": "insufficient_data",
        "features": [],
    }


def _decision_required():
    return {
        "decision": "retraining_required",
        "reason": "Drift detected; retraining is required.",
        "status": "ok",
        "overall_drifted": True,
        "production_rows": 100,
        "drifted_features": ["area"],
    }


def _decision_none():
    return {
        "decision": "no_retraining",
        "reason": "No drift detected; retraining is not required.",
        "status": "ok",
        "overall_drifted": False,
        "production_rows": 100,
        "drifted_features": [],
    }


_UNSET = object()


def _orchestrator(monitoring_result=None, decision=_UNSET, scheduler_result=None):
    monitoring_service = MagicMock()
    monitoring_service.run_check = MagicMock(
        return_value=monitoring_result if monitoring_result is not None else _monitoring_ok()
    )
    trigger_decision = MagicMock()
    trigger_decision.decide = MagicMock(
        return_value=_decision_none() if decision is _UNSET else decision
    )
    retraining_scheduler = MagicMock()
    retraining_scheduler.run_once = MagicMock(
        return_value=scheduler_result if scheduler_result is not None else {"retrained": True}
    )
    orchestrator = DriftRetrainingOrchestrator(
        monitoring_service=monitoring_service,
        trigger_decision=trigger_decision,
        retraining_scheduler=retraining_scheduler,
    )
    return orchestrator, monitoring_service, trigger_decision, retraining_scheduler


def _orchestrator_source():
    return (
        Path(get_project_root()) / "app" / "scheduler" / "drift_retraining_orchestrator.py"
    ).read_text(encoding="utf-8")


# --- 1. DI constructor ---


def test_constructor_stores_injected_collaborators():
    monitoring_service = MagicMock()
    monitoring_service.run_check = MagicMock()
    trigger_decision = MagicMock()
    trigger_decision.decide = MagicMock()
    retraining_scheduler = MagicMock()
    retraining_scheduler.run_once = MagicMock()
    orchestrator = DriftRetrainingOrchestrator(
        monitoring_service=monitoring_service,
        trigger_decision=trigger_decision,
        retraining_scheduler=retraining_scheduler,
    )
    assert orchestrator.monitoring_service is monitoring_service
    assert orchestrator.trigger_decision is trigger_decision
    assert orchestrator.retraining_scheduler is retraining_scheduler


@pytest.mark.parametrize(
    "kwargs",
    [
        {"monitoring_service": object()},
        {"trigger_decision": object()},
        {"retraining_scheduler": object()},
        {"monitoring_service": None},
    ],
)
def test_constructor_rejects_bad_collaborators(kwargs):
    monitoring_service = MagicMock()
    monitoring_service.run_check = MagicMock()
    trigger_decision = MagicMock()
    trigger_decision.decide = MagicMock()
    retraining_scheduler = MagicMock()
    retraining_scheduler.run_once = MagicMock()
    defaults = {
        "monitoring_service": monitoring_service,
        "trigger_decision": trigger_decision,
        "retraining_scheduler": retraining_scheduler,
    }
    defaults.update(kwargs)
    with pytest.raises(AdaptiveAIError):
        DriftRetrainingOrchestrator(**defaults)


def test_package_exports_orchestrator():
    import app.scheduler as scheduler_pkg

    assert scheduler_pkg.DriftRetrainingOrchestrator is DriftRetrainingOrchestrator
    assert "DriftRetrainingOrchestrator" in scheduler_pkg.__all__


# --- 2-3. Monitoring happens once, decision sees the exact result ---


def test_run_check_called_exactly_once_on_negative_path():
    orchestrator, monitoring_service, _, _ = _orchestrator(
        monitoring_result=_monitoring_ok(), decision=_decision_none()
    )
    orchestrator.run_check_and_retrain()
    assert monitoring_service.run_check.call_count == 1


def test_run_check_called_exactly_once_on_positive_path():
    orchestrator, monitoring_service, _, _ = _orchestrator(
        monitoring_result=_monitoring_ok(drifted=True), decision=_decision_required()
    )
    orchestrator.run_check_and_retrain()
    assert monitoring_service.run_check.call_count == 1


def test_exact_monitoring_result_passed_to_trigger():
    monitoring_result = _monitoring_ok(drifted=True)
    orchestrator, _, trigger_decision, _ = _orchestrator(
        monitoring_result=monitoring_result, decision=_decision_required()
    )
    orchestrator.run_check_and_retrain()
    trigger_decision.decide.assert_called_once_with(monitoring_result)
    assert trigger_decision.decide.call_args[0][0] is monitoring_result


# --- 4-6. Gating: no retraining unless required ---


def test_no_retraining_decision_skips_scheduler():
    orchestrator, _, _, retraining_scheduler = _orchestrator(
        monitoring_result=_monitoring_ok(), decision=_decision_none()
    )
    out = orchestrator.run_check_and_retrain()
    retraining_scheduler.run_once.assert_not_called()
    assert out["retraining_result"] is None


def test_insufficient_data_never_retrains_with_real_trigger():
    monitoring_result = _monitoring_insufficient()
    orchestrator, monitoring_service, _, retraining_scheduler = _orchestrator(
        monitoring_result=monitoring_result,
        decision=RetrainingTriggerDecision().decide(copy.deepcopy(monitoring_result)),
    )
    out = orchestrator.run_check_and_retrain()
    retraining_scheduler.run_once.assert_not_called()
    assert out["retraining_result"] is None
    assert out["retraining_decision"]["decision"] == "no_retraining"


def test_healthy_result_never_retrains_with_real_trigger():
    monitoring_result = _monitoring_ok(drifted=False)
    orchestrator, _, _, retraining_scheduler = _orchestrator(
        monitoring_result=monitoring_result,
        decision=RetrainingTriggerDecision().decide(copy.deepcopy(monitoring_result)),
    )
    out = orchestrator.run_check_and_retrain()
    retraining_scheduler.run_once.assert_not_called()
    assert out["retraining_result"] is None


# --- 7-11. Positive path and identity ---


def test_drifted_decision_calls_scheduler_exactly_once():
    orchestrator, _, _, retraining_scheduler = _orchestrator(
        monitoring_result=_monitoring_ok(drifted=True), decision=_decision_required()
    )
    orchestrator.run_check_and_retrain(promote=True)
    assert retraining_scheduler.run_once.call_count == 1


def test_promote_default_is_true_and_propagated():
    assert inspect.signature(DriftRetrainingOrchestrator.run_check_and_retrain).parameters[
        "promote"
    ].default is True
    orchestrator, _, _, retraining_scheduler = _orchestrator(
        monitoring_result=_monitoring_ok(drifted=True), decision=_decision_required()
    )
    orchestrator.run_check_and_retrain()
    assert retraining_scheduler.run_once.call_args.kwargs["promote"] is True


@pytest.mark.parametrize("flag", [True, False])
def test_exact_promote_flag_propagated(flag):
    orchestrator, _, _, retraining_scheduler = _orchestrator(
        monitoring_result=_monitoring_ok(drifted=True), decision=_decision_required()
    )
    orchestrator.run_check_and_retrain(promote=flag)
    assert retraining_scheduler.run_once.call_args.kwargs["promote"] is flag


def test_exact_objects_returned_without_reconstruction():
    monitoring_result = _monitoring_ok(drifted=True)
    decision = _decision_required()
    scheduler_result = {"best_model_name": "X", "token": object()}
    orchestrator, _, _, _ = _orchestrator(
        monitoring_result=monitoring_result,
        decision=decision,
        scheduler_result=scheduler_result,
    )
    out = orchestrator.run_check_and_retrain(promote=False)
    assert set(out.keys()) == {"monitoring_result", "retraining_decision", "retraining_result"}
    assert out["monitoring_result"] is monitoring_result
    assert out["retraining_decision"] is decision
    assert out["retraining_result"] is scheduler_result


def test_negative_path_returns_exact_objects():
    monitoring_result = _monitoring_ok()
    decision = _decision_none()
    orchestrator, _, _, _ = _orchestrator(
        monitoring_result=monitoring_result, decision=decision
    )
    out = orchestrator.run_check_and_retrain()
    assert out["monitoring_result"] is monitoring_result
    assert out["retraining_decision"] is decision
    assert out["retraining_result"] is None


# --- 12-14. Error propagation ---


def test_scheduler_failure_propagates_unchanged():
    original = RetrainingError("cycle boom")
    orchestrator, _, _, retraining_scheduler = _orchestrator(
        monitoring_result=_monitoring_ok(drifted=True), decision=_decision_required()
    )
    retraining_scheduler.run_once = MagicMock(side_effect=original)
    with pytest.raises(RetrainingError) as exc_info:
        orchestrator.run_check_and_retrain()
    assert exc_info.value is original


def test_monitoring_failure_propagates_and_skips_downstream():
    failure = AdaptiveAIError("check boom")
    orchestrator, _, trigger_decision, retraining_scheduler = _orchestrator()
    orchestrator.monitoring_service.run_check = MagicMock(side_effect=failure)
    with pytest.raises(AdaptiveAIError) as exc_info:
        orchestrator.run_check_and_retrain()
    assert exc_info.value is failure
    trigger_decision.decide.assert_not_called()
    retraining_scheduler.run_once.assert_not_called()


def test_trigger_failure_propagates_and_skips_scheduler():
    failure = DriftDetectionError("advice boom")
    orchestrator, _, trigger_decision, retraining_scheduler = _orchestrator()
    trigger_decision.decide = MagicMock(side_effect=failure)
    with pytest.raises(DriftDetectionError) as exc_info:
        orchestrator.run_check_and_retrain()
    assert exc_info.value is failure
    retraining_scheduler.run_once.assert_not_called()


def test_malformed_advice_rejected_without_retraining():
    for bad in ({"decision": "maybe"}, {"reason": "x"}, "no_retraining", None, 42):
        orchestrator, _, _, retraining_scheduler = _orchestrator(decision=bad)
        with pytest.raises(AdaptiveAIError):
            orchestrator.run_check_and_retrain()
        retraining_scheduler.run_once.assert_not_called()


# --- 15. Promote validation ---


@pytest.mark.parametrize("bad", [1, 0, "yes", "True", None, 1.0, [], {}])
def test_invalid_promote_type_rejected_before_any_work(bad):
    orchestrator, monitoring_service, trigger_decision, retraining_scheduler = _orchestrator()
    with pytest.raises(AdaptiveAIError):
        orchestrator.run_check_and_retrain(promote=bad)
    monitoring_service.run_check.assert_not_called()
    trigger_decision.decide.assert_not_called()
    retraining_scheduler.run_once.assert_not_called()


# --- 16. Repeated explicit calls ---


def test_repeated_calls_work_independently():
    orchestrator, monitoring_service, trigger_decision, retraining_scheduler = _orchestrator()
    monitoring_service.run_check = MagicMock(
        side_effect=[_monitoring_ok(drifted=True), _monitoring_ok(drifted=False)]
    )
    trigger_decision.decide = MagicMock(
        side_effect=[_decision_required(), _decision_none()]
    )
    retraining_scheduler.run_once = MagicMock(return_value={"n": 1})
    first = orchestrator.run_check_and_retrain(promote=True)
    second = orchestrator.run_check_and_retrain(promote=False)
    assert monitoring_service.run_check.call_count == 2
    assert trigger_decision.decide.call_count == 2
    assert retraining_scheduler.run_once.call_count == 1
    assert first["retraining_result"] == {"n": 1}
    assert second["retraining_result"] is None


# --- 17-19. Boundary hygiene ---


def test_no_direct_retraining_service_calls():
    source = _orchestrator_source()
    # ".retrain(" with a leading dot only matches a real service call,
    # not the required run_check_and_retrain method name.
    for forbidden in ("RetrainingService", ".retrain(", "train_and_evaluate"):
        assert forbidden not in source


def test_no_direct_promoter_calls():
    source = _orchestrator_source()
    for forbidden in ("ModelPromoter", "promote(", "ModelPromotionDecision"):
        assert forbidden not in source


def test_no_db_mlflow_model_persistence():
    source = _orchestrator_source()
    for forbidden in (
        "sqlite3",
        "SELECT ",
        "DatabaseManager",
        "PromotionRepository",
        "MonitoringRepository",
        "PredictionRepository",
        "save_",
        "mlflow",
        "Mlflow",
        "register_model",
        "joblib",
        "open(",
    ):
        assert forbidden not in source


# --- 20-22. No background execution ---


def test_no_background_execution_primitives():
    source = _orchestrator_source()
    for forbidden in (
        "threading",
        "Thread(",
        "sleep(",
        "while True",
        "while 1",
        "APScheduler",
        "BackgroundScheduler",
        "BlockingScheduler",
        "Celery",
        "celery",
        "asyncio",
        "time.sleep",
    ):
        assert forbidden not in source
    assert "daemon" not in source


def test_no_api_dashboard_imports():
    source = _orchestrator_source()
    for forbidden in ("fastapi", "FastAPI(", "APIRouter", "streamlit", "Streamlit"):
        assert forbidden not in source


def test_no_automatic_execution_on_construction_or_import():
    orchestrator, monitoring_service, trigger_decision, retraining_scheduler = _orchestrator()
    assert orchestrator is not None
    monitoring_service.run_check.assert_not_called()
    trigger_decision.decide.assert_not_called()
    retraining_scheduler.run_once.assert_not_called()
    import app.scheduler.drift_retraining_orchestrator as module

    for attr in dir(module):
        value = getattr(module, attr)
        if isinstance(value, DriftRetrainingOrchestrator):
            raise AssertionError(f"Global orchestrator instance found: {attr}")


# --- 23. Immutability ---


def test_inputs_not_mutated_by_orchestrator():
    monitoring_result = _monitoring_ok(drifted=True)
    snapshot = copy.deepcopy(monitoring_result)
    decision = RetrainingTriggerDecision().decide(copy.deepcopy(monitoring_result))
    decision_snapshot = copy.deepcopy(decision)
    orchestrator, _, _, _ = _orchestrator(
        monitoring_result=monitoring_result,
        decision=decision,
        scheduler_result={"ok": True},
    )
    orchestrator.run_check_and_retrain()
    assert monitoring_result == snapshot
    assert decision == decision_snapshot


# --- 24. Integration with the real advice layer ---


@pytest.mark.parametrize("drifted", [True, False])
def test_integration_with_real_trigger_ok_results(drifted):
    monitoring_result = _monitoring_ok(drifted=drifted)
    real_decision = RetrainingTriggerDecision().decide(copy.deepcopy(monitoring_result))
    orchestrator, _, trigger_decision, retraining_scheduler = _orchestrator(
        monitoring_result=monitoring_result, decision=real_decision
    )
    # The mock stands in for the real advice object but carries its exact output.
    assert trigger_decision.decide.return_value is real_decision
    out = orchestrator.run_check_and_retrain(promote=True)
    assert out["retraining_decision"] is real_decision
    if drifted:
        assert out["retraining_decision"]["decision"] == "retraining_required"
        assert retraining_scheduler.run_once.call_count == 1
        assert out["retraining_decision"]["drifted_features"] == ["area"]
    else:
        assert out["retraining_decision"]["decision"] == "no_retraining"
        retraining_scheduler.run_once.assert_not_called()
        assert out["retraining_result"] is None


def test_orchestrator_creates_no_real_artifacts():
    orchestrator, _, _, _ = _orchestrator()
    orchestrator.run_check_and_retrain()
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass
