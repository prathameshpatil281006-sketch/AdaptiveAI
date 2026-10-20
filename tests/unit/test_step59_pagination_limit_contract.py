"""Step 59 (I-10) focused tests: pagination limit contract is >= 1.

Repositories must reject limit=0 at the boundary (consistent with the API
which requires ge=1), while preserving valid limits and pagination
behavior. All databases are temporary; no MLflow/model side effects.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.core.exceptions import AdaptiveAIError
from app.database.database import DatabaseManager
from app.database.monitoring_repository import MonitoringRepository
from app.database.promotion_repository import PromotionRepository
from app.database.repository import PredictionRepository


def _ok_monitoring_result(**overrides):
    result = {
        "overall_drifted": False,
        "alpha": 0.05,
        "min_production_rows": 30,
        "production_rows": 50,
        "sufficient_data": True,
        "status": "ok",
        "features": [
            {"feature_name": "area", "statistic": 0.1, "p_value": 0.4, "drifted": False},
        ],
    }
    result.update(overrides)
    return result


def _promoted(**overrides):
    result = {
        "decision": "promoted",
        "model_name": "LinearRegression",
        "run_id": "run-1",
        "registered_model_name": "HousePriceModel",
        "model_version": "1",
        "primary_metric": "r2",
        "candidate_score": 0.9,
        "production_score": 0.8,
        "improvement": 0.1,
        "reason": "better",
    }
    result.update(overrides)
    return result


def _tmp_manager(tmp_path: Path, name: str) -> DatabaseManager:
    manager = DatabaseManager(db_path=tmp_path / name)
    manager.init_db()
    return manager


# --- Repository boundary: limit=0 rejected ---


def test_prediction_repository_rejects_limit_zero(tmp_path: Path):
    repo = PredictionRepository(_tmp_manager(tmp_path, "pred.db"))
    with pytest.raises(AdaptiveAIError):
        repo.list_predictions(limit=0)


def test_monitoring_repository_rejects_limit_zero(tmp_path: Path):
    repo = MonitoringRepository(_tmp_manager(tmp_path, "mon.db"))
    with pytest.raises(AdaptiveAIError):
        repo.list_monitoring_results(limit=0)


def test_promotion_repository_rejects_limit_zero(tmp_path: Path):
    repo = PromotionRepository(_tmp_manager(tmp_path, "promo.db"))
    with pytest.raises(AdaptiveAIError):
        repo.list_promotion_history(limit=0)


@pytest.mark.parametrize("bad", [-1, -100, True, False, "5", 1.5, None])
def test_repositories_reject_other_invalid_limits(tmp_path: Path, bad):
    pred = PredictionRepository(_tmp_manager(tmp_path, f"p_{str(bad)}.db"))
    mon = MonitoringRepository(pred._db)
    promo = PromotionRepository(pred._db)
    with pytest.raises(AdaptiveAIError):
        pred.list_predictions(limit=bad)
    with pytest.raises(AdaptiveAIError):
        mon.list_monitoring_results(limit=bad)
    with pytest.raises(AdaptiveAIError):
        promo.list_promotion_history(limit=bad)


# --- Valid limits preserved ---


def test_prediction_repository_valid_limits(tmp_path: Path):
    repo = PredictionRepository(_tmp_manager(tmp_path, "pred_valid.db"))
    for i in range(3):
        repo.save_prediction(
            use_case="house_price",
            model_name="LinearRegression",
            input_data={"area": 1000.0 + i},
            prediction=300000.0 + i,
        )
    assert len(repo.list_predictions(limit=1)) == 1
    assert len(repo.list_predictions(limit=2)) == 2
    assert len(repo.list_predictions(limit=100)) == 3
    page = repo.list_predictions(limit=1, offset=1)
    assert len(page) == 1
    assert repo.list_predictions(limit=1, offset=99) == []


def test_monitoring_repository_valid_limits(tmp_path: Path):
    repo = MonitoringRepository(_tmp_manager(tmp_path, "mon_valid.db"))
    repo.save_monitoring_result(use_case="house_price", result=_ok_monitoring_result())
    repo.save_monitoring_result(use_case="house_price", result=_ok_monitoring_result())
    assert len(repo.list_monitoring_results(limit=1)) == 1
    assert len(repo.list_monitoring_results(limit=2)) == 2
    page = repo.list_monitoring_results(limit=1, offset=1)
    assert len(page) == 1
    assert repo.list_monitoring_results(limit=5, offset=99) == []


def test_promotion_repository_valid_limits(tmp_path: Path):
    repo = PromotionRepository(_tmp_manager(tmp_path, "promo_valid.db"))
    repo.save_promotion_result(use_case="house_price", promotion_result=_promoted())
    repo.save_promotion_result(use_case="house_price", promotion_result=_promoted())
    assert len(repo.list_promotion_history(limit=1)) == 1
    assert len(repo.list_promotion_history(limit=2)) == 2
    page = repo.list_promotion_history(limit=1, offset=1)
    assert len(page) == 1
    assert repo.list_promotion_history(limit=5, offset=99) == []


# --- API consistency: limit=0 -> 422, valid limits pass ---


def test_api_monitoring_history_limit_zero_is_422(tmp_path: Path):
    client = TestClient(create_app(db_manager=_tmp_manager(tmp_path, "api_mon.db")))
    assert client.get("/monitor/house-price/history?limit=0").status_code == 422
    assert client.get("/monitor/house-price/history?limit=1").status_code == 200


def test_api_retrain_history_limit_zero_is_422(tmp_path: Path):
    client = TestClient(create_app(db_manager=_tmp_manager(tmp_path, "api_ret.db")))
    assert client.get("/monitor/house-price/retrain/history?limit=0").status_code == 422
    assert client.get("/monitor/house-price/retrain/history?limit=1").status_code == 200


def test_real_development_database_untouched():
    # Isolation canary: live-state guard lives in tests/conftest.py.
    pass
