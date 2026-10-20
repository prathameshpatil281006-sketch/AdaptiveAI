"""Step 65 (I-10) focused tests: repository limit contract is >= 1.

All three history repositories must reject limit=0 (and negative /
non-integer limits) with AdaptiveAIError, matching the API contract
(Query ge=1). Valid limits and offsets keep working; empty histories
stay empty. All databases are temporary.
"""

from pathlib import Path

import pytest

from app.core.exceptions import AdaptiveAIError
from app.database.database import DatabaseManager
from app.database.monitoring_repository import MonitoringRepository
from app.database.promotion_repository import PromotionRepository
from app.database.repository import PredictionRepository


def _manager(tmp_path: Path, name: str) -> DatabaseManager:
    manager = DatabaseManager(db_path=tmp_path / name)
    manager.init_db()
    return manager


def _ok_monitoring_result():
    return {
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


def _promoted():
    return {
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


def _repos(tmp_path: Path):
    manager = _manager(tmp_path, "limits.db")
    return (
        PredictionRepository(manager),
        MonitoringRepository(manager),
        PromotionRepository(manager),
    )


def _seed(pred, mon, promo):
    for i in range(3):
        pred.save_prediction(
            use_case="house_price",
            model_name="LinearRegression",
            input_data={"area": 1000.0 + i},
            prediction=300000.0 + i,
        )
        mon.save_monitoring_result(use_case="house_price", result=_ok_monitoring_result())
        promo.save_promotion_result(use_case="house_price", promotion_result=_promoted())


def _listers(pred, mon, promo):
    return (
        pred.list_predictions,
        mon.list_monitoring_results,
        promo.list_promotion_history,
    )


def test_limit_zero_rejected_at_repository_level(tmp_path: Path):
    pred, mon, promo = _repos(tmp_path)
    _seed(pred, mon, promo)
    for list_fn in _listers(pred, mon, promo):
        with pytest.raises(AdaptiveAIError):
            list_fn(limit=0)


def test_negative_limit_rejected(tmp_path: Path):
    pred, mon, promo = _repos(tmp_path)
    _seed(pred, mon, promo)
    for list_fn in _listers(pred, mon, promo):
        with pytest.raises(AdaptiveAIError):
            list_fn(limit=-1)
        with pytest.raises(AdaptiveAIError):
            list_fn(limit=-100)


def test_non_integer_limits_rejected(tmp_path: Path):
    pred, mon, promo = _repos(tmp_path)
    for list_fn in _listers(pred, mon, promo):
        for bad in (True, False, "5", 1.5, None):
            with pytest.raises(AdaptiveAIError):
                list_fn(limit=bad)


def test_valid_limit_one_works(tmp_path: Path):
    pred, mon, promo = _repos(tmp_path)
    _seed(pred, mon, promo)
    for list_fn in _listers(pred, mon, promo):
        assert len(list_fn(limit=1)) == 1


def test_valid_larger_limits_work(tmp_path: Path):
    pred, mon, promo = _repos(tmp_path)
    _seed(pred, mon, promo)
    for list_fn in _listers(pred, mon, promo):
        assert len(list_fn(limit=2)) == 2
        assert len(list_fn(limit=100)) == 3


def test_offset_zero_and_positive_work(tmp_path: Path):
    pred, mon, promo = _repos(tmp_path)
    _seed(pred, mon, promo)
    for list_fn in _listers(pred, mon, promo):
        assert len(list_fn(limit=100, offset=0)) == 3
        assert len(list_fn(limit=100, offset=1)) == 2
        assert list_fn(limit=100, offset=99) == []
        with pytest.raises(AdaptiveAIError):
            list_fn(limit=1, offset=-1)


def test_empty_results_unchanged(tmp_path: Path):
    pred, mon, promo = _repos(tmp_path)
    for list_fn in _listers(pred, mon, promo):
        assert list_fn() == []
        assert list_fn(limit=1) == []
        assert list_fn(limit=1, offset=5) == []


def test_limit_contract_matches_api(tmp_path: Path):
    """Repository boundary mirrors the API Query(ge=1) contract."""
    pred, mon, promo = _repos(tmp_path)
    _seed(pred, mon, promo)
    for list_fn in _listers(pred, mon, promo):
        with pytest.raises(AdaptiveAIError, match="positive integer"):
            list_fn(limit=0)
