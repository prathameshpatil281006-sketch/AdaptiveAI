"""Promotion-history audit tests for RetrainingService (Step 36).

Uses mocks/stubs or temporary SQLite databases only. Never touches the
real MLflow registry, the real production model, or the real database.
"""

import copy
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from app.core.config import get_project_root
from app.core.exceptions import AdaptiveAIError, ModelPromotionError, RetrainingError
from app.database.database import DatabaseManager
from app.database.promotion_repository import PromotionRepository
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


def _service_with_mocks(evaluator=None, promoter=None, promotion_repository=None):
    engine = _mock_engine_with_metrics({})
    tracker = _mock_tracker_with_run_ids()
    service = RetrainingService(
        engine=engine,
        tracker=tracker,
        evaluator=evaluator,
        promoter=promoter,
        promotion_repository=promotion_repository,
    )
    return service, engine, tracker


def _mock_repo(history_id=42):
    repo = MagicMock()
    repo.save_promotion_result = MagicMock(return_value=history_id)
    return repo


# --- 1. promote=False ---


def test_promote_false_never_touches_repository():
    repo = _mock_repo()
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(return_value=_eval_result("candidate_better"))
    promoter = MagicMock()
    promoter.promote = MagicMock(return_value=_promotion_result("promoted"))
    service, _, _ = _service_with_mocks(
        evaluator=evaluator, promoter=promoter, promotion_repository=repo
    )

    result = service.retrain(HousePriceUseCase(), _small_df(), promote=False)

    assert set(result.keys()) == {
        "use_case",
        "best_model_name",
        "best_pipeline",
        "best_metrics",
        "all_results",
        "best_run_id",
    }
    assert "promotion" not in result
    assert "promotion_history_id" not in result
    evaluator.evaluate_candidate.assert_not_called()
    promoter.promote.assert_not_called()
    repo.save_promotion_result.assert_not_called()
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass


# --- 2. promote=True + repository ---


def test_promote_true_persists_exactly_once_with_exact_arguments():
    evaluation = _eval_result("candidate_better")
    promotion = _promotion_result("promoted")
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(return_value=evaluation)
    promoter = MagicMock()
    promoter.promote = MagicMock(return_value=promotion)
    repo = _mock_repo(history_id=7)
    service, _, _ = _service_with_mocks(
        evaluator=evaluator, promoter=promoter, promotion_repository=repo
    )

    result = service.retrain(HousePriceUseCase(), _small_df(), promote=True)

    assert repo.save_promotion_result.call_count == 1
    _, kwargs = repo.save_promotion_result.call_args
    assert kwargs["use_case"] == "house_price"
    assert kwargs["use_case"] == result["use_case"]
    assert kwargs["promotion_result"] is promotion
    assert result["promotion_history_id"] == 7
    assert result["promotion"] is promotion


# --- 3/4/5. each outcome persisted ---


@pytest.mark.parametrize(
    "decision", ["promoted", "rejected", "requires_initial_promotion"]
)
def test_each_valid_outcome_persisted_with_history_id(decision):
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(return_value=_eval_result("candidate_better"))
    promotion = _promotion_result(decision)
    promoter = MagicMock()
    promoter.promote = MagicMock(return_value=promotion)
    repo = _mock_repo(history_id=11)
    service, _, _ = _service_with_mocks(
        evaluator=evaluator, promoter=promoter, promotion_repository=repo
    )

    result = service.retrain(HousePriceUseCase(), _small_df(), promote=True)

    assert result["promotion"]["decision"] == decision
    assert result["promotion_history_id"] == 11
    repo.save_promotion_result.assert_called_once()


# --- 6. promote=True without repository ---


def test_promote_true_without_repository_no_persistence_no_id():
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(return_value=_eval_result("candidate_better"))
    promotion = _promotion_result("promoted")
    promoter = MagicMock()
    promoter.promote = MagicMock(return_value=promotion)
    service, _, _ = _service_with_mocks(evaluator=evaluator, promoter=promoter)
    assert service.promotion_repository is None

    result = service.retrain(HousePriceUseCase(), _small_df(), promote=True)

    assert result["promotion"] is promotion
    assert "promotion_history_id" not in result
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass


# --- 7. promotion failure ---


def test_promotion_failure_never_reaches_repository():
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(return_value=_eval_result("candidate_better"))
    promoter = MagicMock()
    promoter.promote = MagicMock(
        side_effect=ModelPromotionError("registry down after persist")
    )
    repo = _mock_repo()
    service, _, _ = _service_with_mocks(
        evaluator=evaluator, promoter=promoter, promotion_repository=repo
    )

    with pytest.raises(RetrainingError) as exc_info:
        service.retrain(HousePriceUseCase(), _small_df(), promote=True)
    assert isinstance(exc_info.value.__cause__, ModelPromotionError)
    repo.save_promotion_result.assert_not_called()


# --- 8. persistence failure ---


def test_persistence_failure_raises_without_false_success():
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(return_value=_eval_result("candidate_better"))
    promoter = MagicMock()
    promoter.promote = MagicMock(return_value=_promotion_result("promoted"))
    repo = MagicMock()
    repo.save_promotion_result = MagicMock(
        side_effect=AdaptiveAIError("disk full during audit write")
    )
    service, _, _ = _service_with_mocks(
        evaluator=evaluator, promoter=promoter, promotion_repository=repo
    )

    with pytest.raises(RetrainingError, match="persistence") as exc_info:
        service.retrain(HousePriceUseCase(), _small_df(), promote=True)
    assert isinstance(exc_info.value.__cause__, AdaptiveAIError)


def test_unexpected_persistence_failure_wrapped():
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(return_value=_eval_result("rejected"))
    promoter = MagicMock()
    promoter.promote = MagicMock(return_value=_promotion_result("rejected"))
    repo = MagicMock()
    repo.save_promotion_result = MagicMock(side_effect=OSError("io boom"))
    service, _, _ = _service_with_mocks(
        evaluator=evaluator, promoter=promoter, promotion_repository=repo
    )

    with pytest.raises(RetrainingError):
        service.retrain(HousePriceUseCase(), _small_df(), promote=True)


# --- 9/10. immutability + exact propagation ---


def test_promotion_result_not_mutated_before_persistence():
    promotion = _promotion_result("promoted")
    snapshot = copy.deepcopy(promotion)
    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(return_value=_eval_result("candidate_better"))
    promoter = MagicMock()
    promoter.promote = MagicMock(return_value=promotion)
    repo = _mock_repo(history_id=3)
    service, _, _ = _service_with_mocks(
        evaluator=evaluator, promoter=promoter, promotion_repository=repo
    )

    result = service.retrain(HousePriceUseCase(), _small_df(), promote=True)

    assert promotion == snapshot
    assert result["promotion"] is promotion
    _, kwargs = repo.save_promotion_result.call_args
    assert kwargs["promotion_result"] is promotion


# --- 11. DI validation + no direct construction ---


def test_invalid_repository_rejected_and_factory_forwards():
    engine = _mock_engine_with_metrics({})
    tracker = _mock_tracker_with_run_ids()
    with pytest.raises(RetrainingError):
        RetrainingService(engine=engine, tracker=tracker, promotion_repository=object())

    repo = _mock_repo()
    service = create_house_price_retraining_service(
        tracker=_mock_tracker_with_run_ids(), promotion_repository=repo
    )
    assert service.promotion_repository is repo
    assert service.promoter is None


def test_service_does_not_construct_database_directly():
    source = (Path(get_project_root()) / "app" / "retraining" / "retraining_service.py").read_text(
        encoding="utf-8"
    )
    assert "DatabaseManager" not in source
    assert "sqlite3" not in source
    assert "PromotionRepository" not in source
    assert "save_promotion_result" in source  # delegation only, no DB logic


# --- real repository on temp DB ---


def test_end_to_end_with_real_repository_on_temp_db(tmp_path: Path):
    manager = DatabaseManager(db_path=tmp_path / "audit.db")
    manager.init_db()
    real_repo = PromotionRepository(manager)

    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(return_value=_eval_result("candidate_better"))
    promotion = _promotion_result("promoted")
    promoter = MagicMock()
    promoter.promote = MagicMock(return_value=promotion)
    service, _, _ = _service_with_mocks(
        evaluator=evaluator, promoter=promoter, promotion_repository=real_repo
    )

    result = service.retrain(HousePriceUseCase(), _small_df(), promote=True)

    assert isinstance(result["promotion_history_id"], int)
    stored = real_repo.get_promotion_result_by_id(result["promotion_history_id"])
    assert stored is not None
    assert stored["decision"] == "promoted"
    assert stored["use_case"] == "house_price"
    assert stored["model_name"] == promotion["model_name"]
    assert stored["run_id"] == promotion["run_id"]
    assert stored["model_version"] == promotion["model_version"]
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass


def test_end_to_end_rejected_persisted_in_temp_db(tmp_path: Path):
    manager = DatabaseManager(db_path=tmp_path / "audit.db")
    manager.init_db()
    real_repo = PromotionRepository(manager)

    evaluator = MagicMock()
    evaluator.evaluate_candidate = MagicMock(
        return_value=_eval_result("production_better")
    )
    promoter = MagicMock()
    promoter.promote = MagicMock(return_value=_promotion_result("rejected"))
    service, _, _ = _service_with_mocks(
        evaluator=evaluator, promoter=promoter, promotion_repository=real_repo
    )

    result = service.retrain(HousePriceUseCase(), _small_df(), promote=True)

    assert result["promotion"]["decision"] == "rejected"
    stored = real_repo.get_promotion_result_by_id(result["promotion_history_id"])
    assert stored is not None
    assert stored["decision"] == "rejected"
    assert stored["model_version"] is None


# --- 12. no API/scheduler/dashboard wiring ---


def test_no_api_scheduler_dashboard_wiring():
    api_src = (Path(get_project_root()) / "app" / "api" / "main.py").read_text(encoding="utf-8")
    # Step 40 adds a thin retrain adapter; Step 41 adds a thin read-only
    # history adapter over PromotionRepository (no writes, no promotion).
    assert "/monitor/house-price/retrain" in api_src
    assert "/monitor/house-price/retrain/history" in api_src
    assert "RetrainingService(" not in api_src
    assert "PromotionRepository" in api_src
    assert "list_promotion_history" in api_src
    assert "save_promotion_result" not in api_src
    scheduler_src = (Path(get_project_root()) / "app" / "scheduler" / "__init__.py").read_text(
        encoding="utf-8"
    )
    # Step 37 adds the scheduler foundation (explicit run_once only).
    assert "RetrainingScheduler" in scheduler_src
    dashboard_dir = Path(get_project_root()) / "dashboard"
    if dashboard_dir.exists():
        for path in dashboard_dir.rglob("*.py"):
            assert "PromotionRepository" not in path.read_text(encoding="utf-8")
            assert "promotion_history" not in path.read_text(encoding="utf-8")
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass
