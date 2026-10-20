"""Step 63 (I-14) focused tests: history retrieval is newest-first.

Covers predictions, monitoring results, and promotion history:
newest record first, deterministic ordering, limit/offset against
newest-first, use_case filtering, and empty-history behavior.
All databases are temporary.
"""

from pathlib import Path

from app.database.database import DatabaseManager
from app.database.monitoring_repository import MonitoringRepository
from app.database.promotion_repository import PromotionRepository
from app.database.repository import PredictionRepository


def _manager(tmp_path: Path, name: str) -> DatabaseManager:
    manager = DatabaseManager(db_path=tmp_path / name)
    manager.init_db()
    return manager


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


def test_predictions_newest_first_and_deterministic(tmp_path: Path):
    repo = PredictionRepository(_manager(tmp_path, "pred.db"))
    ids = [
        repo.save_prediction(
            use_case="house_price",
            model_name="LinearRegression",
            input_data={"area": 1000.0 + i},
            prediction=300000.0 + i,
        )
        for i in range(3)
    ]
    records = repo.list_predictions()
    assert [r["id"] for r in records] == [ids[2], ids[1], ids[0]]
    # Deterministic across calls.
    assert [r["id"] for r in repo.list_predictions()] == [ids[2], ids[1], ids[0]]
    # Newest record is first.
    assert records[0]["input_data"] == {"area": 1002.0}


def test_predictions_limit_offset_newest_first(tmp_path: Path):
    repo = PredictionRepository(_manager(tmp_path, "pred_page.db"))
    for i in range(3):
        repo.save_prediction(
            use_case="house_price",
            model_name="LinearRegression",
            input_data={"area": 1000.0 + i},
            prediction=300000.0 + i,
        )
    assert [r["input_data"] for r in repo.list_predictions(limit=1)] == [{"area": 1002.0}]
    assert [r["input_data"] for r in repo.list_predictions(limit=1, offset=1)] == [
        {"area": 1001.0}
    ]
    assert [r["input_data"] for r in repo.list_predictions(limit=1, offset=2)] == [
        {"area": 1000.0}
    ]
    assert repo.list_predictions(limit=1, offset=99) == []


def test_predictions_empty_and_filter(tmp_path: Path):
    repo = PredictionRepository(_manager(tmp_path, "pred_empty.db"))
    assert repo.list_predictions() == []
    repo.save_prediction(
        use_case="house_price",
        model_name="LinearRegression",
        input_data={"area": 1.0},
        prediction=1.0,
    )
    assert len(repo.list_predictions(use_case="house_price")) == 1
    assert repo.list_predictions(use_case="missing") == []


def test_monitoring_newest_first_limit_offset(tmp_path: Path):
    repo = MonitoringRepository(_manager(tmp_path, "mon.db"))
    assert repo.list_monitoring_results() == []
    id1 = repo.save_monitoring_result(use_case="house_price", result=_ok_monitoring_result())
    id2 = repo.save_monitoring_result(use_case="house_price", result=_ok_monitoring_result())
    id3 = repo.save_monitoring_result(use_case="other_case", result=_ok_monitoring_result())

    records = repo.list_monitoring_results()
    assert [r["id"] for r in records] == [id3, id2, id1]
    assert [r["id"] for r in repo.list_monitoring_results(use_case="house_price")] == [
        id2,
        id1,
    ]
    page = repo.list_monitoring_results(limit=1, offset=1)
    assert [r["id"] for r in page] == [id2]
    assert repo.list_monitoring_results(limit=1) == records[:1]
    assert repo.list_monitoring_results(limit=1, offset=99) == []
    assert repo.list_monitoring_results(use_case="missing") == []


def test_promotion_newest_first_limit_offset(tmp_path: Path):
    repo = PromotionRepository(_manager(tmp_path, "promo.db"))
    assert repo.list_promotion_history() == []
    id1 = repo.save_promotion_result(use_case="house_price", promotion_result=_promoted())
    id2 = repo.save_promotion_result(use_case="house_price", promotion_result=_promoted())
    id3 = repo.save_promotion_result(use_case="other_case", promotion_result=_promoted())

    records = repo.list_promotion_history()
    assert [r["id"] for r in records] == [id3, id2, id1]
    assert [r["id"] for r in repo.list_promotion_history(use_case="house_price")] == [
        id2,
        id1,
    ]
    page = repo.list_promotion_history(limit=1, offset=1)
    assert [r["id"] for r in page] == [id2]
    assert repo.list_promotion_history(limit=1) == records[:1]
    assert repo.list_promotion_history(limit=1, offset=99) == []
    assert repo.list_promotion_history(use_case="missing") == []
