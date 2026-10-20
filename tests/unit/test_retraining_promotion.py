"""Opt-in promotion integration tests for RetrainingService (Step 34).

Uses mocks/stubs only. Never touches the real MLflow registry, the real
production model, or the real database.
"""

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.core.config import get_project_root
from app.core.exceptions import ModelPromotionError, RetrainingError
from app.engine.ml_engine import MLEngine
from app.engine.mlflow_tracker import MLflowTracker
from app.retraining.retraining_service import (
    RetrainingService,
    create_house_price_retraining_service,
)
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.use_case import HousePriceUseCase


def _small_df(n_samples: int = 120) -> pd.DataFrame:
    return generate_house_price_data(n_samples=n_samples)


def _mock_engine_with_metrics(metrics_by_model=None):
    metrics_by_model = metrics_by_model or {}
    engine = MagicMock(spec=MLEngine)

    def _side_effect(**kwargs):
        model = kwargs.get("model")
        model_name = type(model).__name__
        metrics = metrics_by_model.get(
            model_name, {"mae": 1.0, "rmse": 2.0, "r2": 0.5}
        )
        pipeline = MagicMock(spec=Pipeline)
        pipeline._mock_model_name = model_name
        return {"pipeline": pipeline, "metrics": dict(metrics)}

    engine.train_and_evaluate.side_effect = _side_effect
    return engine


def _mock_tracker_with_run_ids():
    tracker = MagicMock(spec=MLflowTracker)
    tracker.tracking_uri = "file:///tmp/fake-mlruns"

    def _track(model_type=None, hyperparameters=None, metrics=None, pipeline=None, run_name=None, **kwargs):
        return f"run-{model_type}"

    tracker.track_candidate.side_effect = _track
    tracker._get_client.return_value = MagicMock()
    return tracker


def _eval_result(status="candidate_better"):
    if status == "candidate_better":
        return {
            "candidate": {"model_name": "c", "metrics": {"mae": 1.0, "rmse": 2.0, "r2": 0.9}},
            "production": {"model_name": "p", "metrics": {"mae": 2.0, "rmse": 3.0, "r2": 0.7}},
            "comparison": {
                "primary_metric": "r2",
                "candidate_score": 0.9,
                "production_score": 0.7,
                "improvement": 0.2,
                "candidate_better": True,
            },
            "status": "candidate_better",
        }
    if status == "production_better":
        return {
            "candidate": {"model_name": "c", "metrics": {"mae": 2.0, "rmse": 3.0, "r2": 0.5}},
            "production": {"model_name": "p", "metrics": {"mae": 1.0, "rmse": 2.0, "r2": 0.8}},
            "comparison": {
                "primary_metric": "r2",
                "candidate_score": 0.5,
                "production_score": 0.8,
                "improvement": -0.3,
                "candidate_better": False,
            },
            "status": "production_better",
        }
    return {
        "candidate": {"model_name": "c", "metrics": {"mae": 1.0, "rmse": 2.0, "r2": 0.85}},
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


def _promotion_result(decision="promoted"):
    if decision == "promoted":
        return {
            "decision": "promoted",
            "model_name": "LinearRegression",
            "run_id": "run-LinearRegression",
            "registered_model_name": "HousePriceModel",
            "model_version": "5",
            "reason": "Promoted.",
        }
    if decision == "rejected":
        return {
            "decision": "rejected",
            "model_name": "LinearRegression",
            "run_id": "run-LinearRegression",
            "registered_model_name": "HousePriceModel",
            "model_version": None,
            "reason": "Rejected.",
        }
    return {
        "decision": "requires_initial_promotion",
        "model_name": "LinearRegression",
        "run_id": "run-LinearRegression",
        "registered_model_name": "HousePriceModel",
        "model_version": None,
        "reason": "Requires explicit initial deployment.",
    }


def _service_with_mocks(evaluator=None, promoter=None, metrics_by_model=None):
    engine = _mock_engine_with_metrics(metrics_by_model or {})
    tracker = _mock_tracker_with_run_ids()
    service = RetrainingService(
        engine=engine, tracker=tracker, evaluator=evaluator, promoter=promoter
    )
    return service, engine, tracker


# --- 1. Existing behavior ---


def test_promote_defaults_false_and_backward_compatible():
    sig = inspect.signature(RetrainingService.retrain)
    assert sig.parameters["promote"].default is False
    service, _, _ = _service_with_mocks()
    result = service.retrain(HousePriceUseCase(), _small_df())
    assert set(result.keys()) == {
        "use_case",
        "best_model_name",
        "best_pipeline",
        "best_metrics",
        "all_results",
        "best_run_id",
    }
    assert "promotion" not in result


def test_promote_false_does_not_call_evaluator_or_promoter():
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(return_value=_eval_result("candidate_better"))
    promoter = MagicMock()
    promoter.promote = MagicMock(return_value=_promotion_result("promoted"))
    service, _, _ = _service_with_mocks(evaluator=evaluator, promoter=promoter)

    result = service.retrain(HousePriceUseCase(), _small_df(), promote=False)
    assert "promotion" not in result
    evaluator.evaluate_candidate.assert_not_called()
    promoter.promote.assert_not_called()


def test_legacy_two_arg_call_unchanged():
    service, _, _ = _service_with_mocks()
    result = service.retrain(HousePriceUseCase(), _small_df())
    assert result["use_case"] == "house_price"
    assert result["best_run_id"].startswith("run-")


def test_service_defaults_have_no_evaluator_or_promoter():
    tracker = _mock_tracker_with_run_ids()
    engine = _mock_engine_with_metrics({})
    service = RetrainingService(engine=engine, tracker=tracker)
    assert service.evaluator is None
    assert service.promoter is None


def test_invalid_evaluator_or_promoter_rejected():
    engine = _mock_engine_with_metrics({})
    tracker = _mock_tracker_with_run_ids()
    with pytest.raises(RetrainingError):
        RetrainingService(engine=engine, tracker=tracker, evaluator=object())
    with pytest.raises(RetrainingError):
        RetrainingService(engine=engine, tracker=tracker, promoter=object())


def test_factory_forwards_evaluator_and_promoter():
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock()
    promoter = MagicMock()
    promoter.promote = MagicMock()
    tracker = _mock_tracker_with_run_ids()
    service = create_house_price_retraining_service(
        tracker=tracker, evaluator=evaluator, promoter=promoter
    )
    assert service.evaluator is evaluator
    assert service.promoter is promoter


# --- 2. Opt-in promotion ---


def test_promote_true_invokes_evaluator_and_promoter():
    evaluation = _eval_result("candidate_better")
    promotion = _promotion_result("promoted")
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(return_value=evaluation)
    promoter = MagicMock()
    promoter.promote = MagicMock(return_value=promotion)
    service, _, _ = _service_with_mocks(evaluator=evaluator, promoter=promoter)

    result = service.retrain(HousePriceUseCase(), _small_df(), promote=True)

    assert evaluator.evaluate_candidate.call_count == 1
    assert promoter.promote.call_count == 1
    assert result["promotion"] is promotion


def test_best_pipeline_and_run_id_passed_through():
    evaluation = _eval_result("candidate_better")
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(return_value=evaluation)
    captured = {}

    promoter = MagicMock()

    def _promote(**kwargs):
        captured.update(kwargs)
        return _promotion_result("promoted")

    promoter.promote = MagicMock(side_effect=_promote)
    service, _, _ = _service_with_mocks(evaluator=evaluator, promoter=promoter)

    result = service.retrain(HousePriceUseCase(), _small_df(), promote=True)

    assert captured["candidate_model"] is result["best_pipeline"]
    assert captured["candidate_run_id"] == result["best_run_id"]
    assert captured["candidate_model_name"] == result["best_model_name"]
    # Evaluation result comes from the evaluator unchanged.
    assert captured["evaluation_result"] is evaluation


def test_production_name_defaults_to_best_model_name():
    evaluation = _eval_result("candidate_better")
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(return_value=evaluation)
    promoter = MagicMock()
    promoter.promote = MagicMock(return_value=_promotion_result("promoted"))
    service, _, _ = _service_with_mocks(evaluator=evaluator, promoter=promoter)

    result = service.retrain(HousePriceUseCase(), _small_df(), promote=True)

    _, eval_kwargs = evaluator.evaluate_candidate.call_args
    assert eval_kwargs["production_model_name"] == result["best_model_name"]
    _, promo_kwargs = promoter.promote.call_args
    assert promo_kwargs["production_model_name"] == result["best_model_name"]


def test_explicit_production_name_forwarded():
    evaluation = _eval_result("production_better")
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(return_value=evaluation)
    promoter = MagicMock()
    promoter.promote = MagicMock(return_value=_promotion_result("rejected"))
    service, _, _ = _service_with_mocks(evaluator=evaluator, promoter=promoter)

    service.retrain(
        HousePriceUseCase(),
        _small_df(),
        promote=True,
        production_model_name="CustomProd",
    )
    _, eval_kwargs = evaluator.evaluate_candidate.call_args
    assert eval_kwargs["production_model_name"] == "CustomProd"


# --- 3/4/5. Decision outcomes ---


def test_candidate_better_promoted_result_propagated():
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(return_value=_eval_result("candidate_better"))
    promotion = _promotion_result("promoted")
    promoter = MagicMock()
    promoter.promote = MagicMock(return_value=promotion)
    service, _, _ = _service_with_mocks(evaluator=evaluator, promoter=promoter)

    result = service.retrain(HousePriceUseCase(), _small_df(), promote=True)
    assert result["promotion"]["decision"] == "promoted"
    assert result["promotion"]["model_version"] == "5"
    for key in (
        "use_case",
        "best_model_name",
        "best_pipeline",
        "best_metrics",
        "all_results",
        "best_run_id",
        "promotion",
    ):
        assert key in result


def test_production_better_rejected_no_auto_promote():
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(
        return_value=_eval_result("production_better")
    )
    promoter = MagicMock()
    promoter.promote = MagicMock(return_value=_promotion_result("rejected"))
    service, _, _ = _service_with_mocks(evaluator=evaluator, promoter=promoter)

    result = service.retrain(HousePriceUseCase(), _small_df(), promote=True)
    assert result["promotion"]["decision"] == "rejected"
    assert result["promotion"]["model_version"] is None


def test_no_production_model_never_auto_promotes():
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(
        return_value=_eval_result("no_production_model")
    )
    promoter = MagicMock()
    promoter.promote = MagicMock(
        return_value=_promotion_result("requires_initial_promotion")
    )
    service, _, _ = _service_with_mocks(evaluator=evaluator, promoter=promoter)

    result = service.retrain(HousePriceUseCase(), _small_df(), promote=True)
    assert result["promotion"]["decision"] == "requires_initial_promotion"
    assert result["promotion"]["decision"] != "promoted"


# --- 6. Promotion failure ---


def test_promotion_failure_surfaced_as_retraining_error():
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(return_value=_eval_result("candidate_better"))
    promoter = MagicMock()
    promoter.promote = MagicMock(
        side_effect=ModelPromotionError("registry down after persist")
    )
    service, _, _ = _service_with_mocks(evaluator=evaluator, promoter=promoter)

    with pytest.raises(RetrainingError) as exc_info:
        service.retrain(HousePriceUseCase(), _small_df(), promote=True)
    assert isinstance(exc_info.value.__cause__, ModelPromotionError)
    # Candidate context retained.
    assert "run-" in str(exc_info.value)


def test_evaluator_failure_skips_promoter():
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(
        side_effect=RetrainingError("production broken")
    )
    promoter = MagicMock()
    promoter.promote = MagicMock(return_value=_promotion_result("promoted"))
    service, _, _ = _service_with_mocks(evaluator=evaluator, promoter=promoter)

    with pytest.raises(RetrainingError):
        service.retrain(HousePriceUseCase(), _small_df(), promote=True)
    promoter.promote.assert_not_called()


def test_promote_true_without_promoter_fails_fast():
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(return_value=_eval_result("candidate_better"))
    engine = _mock_engine_with_metrics({})
    tracker = _mock_tracker_with_run_ids()
    service = RetrainingService(engine=engine, tracker=tracker, evaluator=evaluator)

    with pytest.raises(RetrainingError, match="promoter"):
        service.retrain(HousePriceUseCase(), _small_df(), promote=True)
    engine.train_and_evaluate.assert_not_called()
    evaluator.evaluate_candidate.assert_not_called()


# --- 7/8. Safety ---


def test_promote_false_performs_zero_promotion_side_effects(tmp_path: Path, monkeypatch):
    import app.use_cases.house_price.model_persistence as persistence

    def _fail_save(*args, **kwargs):
        raise AssertionError("must not persist when promote=False")

    monkeypatch.setattr(persistence, "save_house_price_model", _fail_save)
    tracker = _mock_tracker_with_run_ids()
    tracker.register_model = MagicMock(
        side_effect=AssertionError("must not register when promote=False")
    )
    engine = _mock_engine_with_metrics({})
    service = RetrainingService(engine=engine, tracker=tracker)

    models_dir = Path(get_project_root()) / "models" / "house_price"
    before = set(p.name for p in models_dir.glob("*.joblib")) if models_dir.exists() else set()
    result = service.retrain(HousePriceUseCase(), _small_df(), promote=False)
    after = set(p.name for p in models_dir.glob("*.joblib")) if models_dir.exists() else set()
    assert before == after
    assert "promotion" not in result
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass


def test_promote_true_with_mocks_touches_no_real_artifacts():
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(return_value=_eval_result("candidate_better"))
    promoter = MagicMock()
    promoter.promote = MagicMock(return_value=_promotion_result("promoted"))
    service, _, _ = _service_with_mocks(evaluator=evaluator, promoter=promoter)

    models_dir = Path(get_project_root()) / "models" / "house_price"
    before = set(p.name for p in models_dir.glob("*.joblib")) if models_dir.exists() else set()
    service.retrain(HousePriceUseCase(), _small_df(), promote=True)
    after = set(p.name for p in models_dir.glob("*.joblib")) if models_dir.exists() else set()
    assert before == after
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass


def test_service_source_has_no_hidden_side_effect_wiring():
    source = (Path(get_project_root()) / "app" / "retraining" / "retraining_service.py").read_text(
        encoding="utf-8"
    )
    assert "ModelPromoter" in source
    assert "ModelEvaluator" in source or "evaluator" in source
    for forbidden in (
        "save_house_price_model",
        "register_model",
        "ensure_registered_model",
        "create_model_version",
        "import sqlite3",
        "FastAPI(",
        "APScheduler",
        "Celery",
        "streamlit",
    ):
        assert forbidden not in source


def test_no_scheduler_api_dashboard_db_changes():
    # Step 37 adds the scheduler foundation (explicit run_once only).
    # Guard the boundary: scheduler exports the foundation, API stays unwired.
    scheduler_src = (Path(get_project_root()) / "app" / "scheduler" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert "RetrainingScheduler" in scheduler_src
    api_src = (Path(get_project_root()) / "app" / "api" / "main.py").read_text(encoding="utf-8")
    # Step 40 adds a thin retrain adapter over DriftRetrainingOrchestrator.
    assert "/monitor/house-price/retrain" in api_src
    assert "run_check_and_retrain" in api_src
    assert "RetrainingService(" not in api_src
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass
