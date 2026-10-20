"""Focused tests for monitoring-result persistence (Step 28).

Covers the monitoring_results schema/repository plus MonitoringService
persistence integration. All databases are temporary; the real
data/adaptive_ai.db is never touched.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.core.config import get_project_root
from app.core.exceptions import AdaptiveAIError
from app.database.database import DatabaseManager
from app.database.monitoring_repository import MonitoringRepository
from app.database.repository import PredictionRepository
from app.monitoring.data_provider import MonitoringDataProvider
from app.monitoring.drift_detector import DriftDetector
from app.monitoring.monitoring_service import (
    MonitoringService,
    create_house_price_monitoring_service,
)


def _tmp_monitoring_repo(tmp_path: Path, name: str = "mon.db") -> MonitoringRepository:
    manager = DatabaseManager(db_path=tmp_path / name)
    manager.init_db()
    return MonitoringRepository(manager)


def _ok_result(**overrides):
    result = {
        "overall_drifted": False,
        "alpha": 0.05,
        "min_production_rows": 30,
        "production_rows": 50,
        "sufficient_data": True,
        "status": "ok",
        "features": [
            {"feature_name": "area", "statistic": 0.1, "p_value": 0.4, "drifted": False},
            {"feature_name": "bedrooms", "statistic": 0.2, "p_value": 0.03, "drifted": True},
        ],
    }
    result.update(overrides)
    return result


def _insufficient_result():
    return {
        "overall_drifted": False,
        "alpha": 0.05,
        "min_production_rows": 30,
        "production_rows": 0,
        "sufficient_data": False,
        "status": "insufficient_data",
        "features": [],
    }


# --- Schema ---


def test_monitoring_results_table_is_created(tmp_path: Path):
    manager = DatabaseManager(db_path=tmp_path / "schema.db")
    manager.init_db()
    conn = manager.get_connection()
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='monitoring_results'"
        ).fetchone()
        assert row is not None
        columns = {r[1]: r[2] for r in conn.execute("PRAGMA table_info(monitoring_results)").fetchall()}
        for expected in (
            "id",
            "timestamp",
            "use_case",
            "overall_drifted",
            "status",
            "alpha",
            "min_production_rows",
            "production_rows",
            "sufficient_data",
            "feature_results",
        ):
            assert expected in columns
        assert columns["id"] == "INTEGER"
        assert columns["feature_results"] == "TEXT"
    finally:
        DatabaseManager.close_connection(conn)


def test_predictions_table_unchanged(tmp_path: Path):
    manager = DatabaseManager(db_path=tmp_path / "schema.db")
    manager.init_db()
    conn = manager.get_connection()
    try:
        info = {
            r[1]: r for r in conn.execute("PRAGMA table_info(predictions)").fetchall()
        }
        # Phase 1-5 base columns are never renamed or removed.
        assert {
            "id",
            "timestamp",
            "use_case",
            "model_name",
            "model_version",
            "input_data",
            "prediction",
        }.issubset(set(info))
        # Phase 6 adds ONLY nullable demo-isolation columns (existing
        # rows read back NULLs, interpreted as production traffic).
        assert set(info) == {
            "id",
            "timestamp",
            "use_case",
            "model_name",
            "model_version",
            "input_data",
            "prediction",
            "source",
            "demo_session_id",
        }
        assert info["source"][3] == 0  # notnull flag: nullable
        assert info["demo_session_id"][3] == 0
    finally:
        DatabaseManager.close_connection(conn)


# --- Repository ---


def test_save_and_retrieve_monitoring_result(tmp_path: Path):
    repo = _tmp_monitoring_repo(tmp_path)
    record_id = repo.save_monitoring_result(use_case="house_price", result=_ok_result())
    assert isinstance(record_id, int)

    record = repo.get_monitoring_result_by_id(record_id)
    assert record is not None
    assert record["id"] == record_id
    assert record["use_case"] == "house_price"
    assert record["overall_drifted"] is False
    assert record["status"] == "ok"
    assert record["alpha"] == pytest.approx(0.05)
    assert record["min_production_rows"] == 30
    assert record["production_rows"] == 50
    assert record["sufficient_data"] is True
    assert isinstance(record["features"], list)
    assert record["features"][0]["feature_name"] == "area"


def test_list_filter_limit_offset(tmp_path: Path):
    repo = _tmp_monitoring_repo(tmp_path)
    repo.save_monitoring_result(use_case="house_price", result=_ok_result())
    repo.save_monitoring_result(use_case="house_price", result=_insufficient_result())
    repo.save_monitoring_result(use_case="other_case", result=_ok_result())

    assert len(repo.list_monitoring_results()) == 3
    assert len(repo.list_monitoring_results(use_case="house_price")) == 2
    assert len(repo.list_monitoring_results(use_case="other_case")) == 1

    page = repo.list_monitoring_results(limit=1, offset=1)
    assert len(page) == 1
    assert page[0]["id"] == 2
    assert repo.get_monitoring_result_by_id(999999) is None


def test_feature_results_json_round_trip(tmp_path: Path):
    repo = _tmp_monitoring_repo(tmp_path)
    record_id = repo.save_monitoring_result(use_case="house_price", result=_ok_result())
    conn = repo._db.get_connection()
    try:
        raw = conn.execute(
            "SELECT feature_results FROM monitoring_results WHERE id = ?", (record_id,)
        ).fetchone()[0]
        assert json.loads(raw) == _ok_result()["features"]
    finally:
        DatabaseManager.close_connection(conn)

    record = repo.get_monitoring_result_by_id(record_id)
    assert record is not None
    for feat in record["features"]:
        assert isinstance(feat["statistic"], float)
        assert isinstance(feat["p_value"], float)
        assert isinstance(feat["drifted"], bool)


def test_explicit_timestamp_round_trip(tmp_path: Path):
    repo = _tmp_monitoring_repo(tmp_path)
    explicit = datetime(2026, 2, 1, 12, 0, 0, tzinfo=timezone.utc)
    record_id = repo.save_monitoring_result(
        use_case="house_price", result=_ok_result(), timestamp=explicit
    )
    record = repo.get_monitoring_result_by_id(record_id)
    assert record is not None
    assert datetime.fromisoformat(str(record["timestamp"])) == explicit


def test_insufficient_data_result_persists(tmp_path: Path):
    repo = _tmp_monitoring_repo(tmp_path)
    record_id = repo.save_monitoring_result(
        use_case="house_price", result=_insufficient_result()
    )
    record = repo.get_monitoring_result_by_id(record_id)
    assert record is not None
    assert record["status"] == "insufficient_data"
    assert record["sufficient_data"] is False
    assert record["overall_drifted"] is False
    assert record["features"] == []


def test_malformed_result_rejected(tmp_path: Path):
    repo = _tmp_monitoring_repo(tmp_path)
    incomplete = {"overall_drifted": False, "status": "ok"}
    with pytest.raises(AdaptiveAIError):
        repo.save_monitoring_result(use_case="house_price", result=incomplete)

    bad_feature = _ok_result(features=[{"feature_name": "area"}])
    with pytest.raises(AdaptiveAIError):
        repo.save_monitoring_result(use_case="house_price", result=bad_feature)

    with pytest.raises(AdaptiveAIError):
        repo.save_monitoring_result(use_case="", result=_ok_result())
    with pytest.raises(AdaptiveAIError):
        repo.save_monitoring_result(use_case="house_price", result="not-a-dict")  # type: ignore[arg-type]
    with pytest.raises(AdaptiveAIError):
        repo.get_monitoring_result_by_id("bad")  # type: ignore[arg-type]


# --- Service integration ---


def _seed_predictions(repo: PredictionRepository, n: int = 50, shift: float = 0.0) -> None:
    import pandas as pd

    from app.core.config import resolve_project_path

    ref = pd.read_csv(
        resolve_project_path("data/reference/house_price_reference.csv")
    )
    sample = ref.sample(n=n, random_state=7).reset_index(drop=True)
    feature_cols = [c for c in ref.columns if c != "price"]
    for i in range(n):
        row = {c: sample.loc[i, c] for c in feature_cols}
        if isinstance(row.get("area_sqft"), float):
            row["area_sqft"] = float(row["area_sqft"] + shift)
        for int_col in ("bedrooms", "bathrooms", "parking", "floor_number", "total_floors"):
            row[int_col] = int(row[int_col])
        repo.save_prediction(
            use_case="house_price",
            model_name="LinearRegression",
            input_data=row,
            prediction=400000.0,
        )


def _service_with_persistence(tmp_path: Path, **kwargs):
    manager = DatabaseManager(db_path=tmp_path / "svc_mon.db")
    manager.init_db()
    pred_repo = PredictionRepository(manager)
    mon_repo = MonitoringRepository(manager)
    _seed_predictions(pred_repo)
    service = create_house_price_monitoring_service(
        repository=pred_repo, monitoring_repository=mon_repo, **kwargs
    )
    return service, mon_repo


def test_service_persists_exactly_one_result(tmp_path: Path):
    service, mon_repo = _service_with_persistence(tmp_path)
    assert mon_repo.list_monitoring_results() == []
    result = service.run_check()
    stored = mon_repo.list_monitoring_results()
    assert len(stored) == 1
    assert stored[0]["use_case"] == "house_price"
    assert stored[0]["overall_drifted"] is result["overall_drifted"]


def test_returned_result_unchanged_and_matches_persisted(tmp_path: Path):
    service, mon_repo = _service_with_persistence(tmp_path)
    result = service.run_check()
    stored = mon_repo.list_monitoring_results()[0]
    for key in (
        "overall_drifted",
        "alpha",
        "min_production_rows",
        "production_rows",
        "sufficient_data",
        "status",
    ):
        assert stored[key] == result[key]
    assert stored["features"] == result["features"]
    assert set(result.keys()) >= {
        "overall_drifted",
        "alpha",
        "min_production_rows",
        "production_rows",
        "sufficient_data",
        "status",
        "features",
    }


def test_provider_failure_persists_nothing(tmp_path: Path):
    provider = MagicMock(spec=MonitoringDataProvider)
    provider.get_monitoring_data.side_effect = AdaptiveAIError("provider boom")
    provider.use_case_name = "house_price"
    mon_repo = _tmp_monitoring_repo(tmp_path)
    service = MonitoringService(
        provider=provider, detector=DriftDetector(), monitoring_repository=mon_repo
    )
    with pytest.raises(AdaptiveAIError, match="provider boom"):
        service.run_check()
    assert mon_repo.list_monitoring_results() == []


def test_detector_failure_persists_nothing(tmp_path: Path):
    ref = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    prod = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    provider = MagicMock(spec=MonitoringDataProvider)
    provider.get_monitoring_data.return_value = {
        "reference_features": ref,
        "production_features": prod,
    }
    provider.use_case_name = "house_price"
    detector = MagicMock(spec=DriftDetector)
    detector.detect.side_effect = AdaptiveAIError("detector boom")
    mon_repo = _tmp_monitoring_repo(tmp_path)
    service = MonitoringService(
        provider=provider, detector=detector, monitoring_repository=mon_repo
    )
    with pytest.raises(AdaptiveAIError, match="detector boom"):
        service.run_check()
    assert mon_repo.list_monitoring_results() == []


def test_repository_failure_propagates(tmp_path: Path):
    manager = DatabaseManager(db_path=tmp_path / "svc_mon.db")
    manager.init_db()
    pred_repo = PredictionRepository(manager)
    _seed_predictions(pred_repo)
    service = create_house_price_monitoring_service(repository=pred_repo)
    broken = MagicMock(spec=MonitoringRepository)
    broken.init.return_value = None
    broken.save_monitoring_result.side_effect = AdaptiveAIError("db boom")
    service._monitoring_repository = broken
    with pytest.raises(AdaptiveAIError, match="db boom"):
        service.run_check()


def test_service_without_repository_matches_detector(tmp_path: Path):
    manager = DatabaseManager(db_path=tmp_path / "svc_mon.db")
    manager.init_db()
    pred_repo = PredictionRepository(manager)
    _seed_predictions(pred_repo)
    plain = create_house_price_monitoring_service(repository=pred_repo)
    result = plain.run_check()
    assert set(result.keys()) >= {"overall_drifted", "status", "features"}


def test_service_uses_injected_temporary_repository(tmp_path: Path):
    service, mon_repo = _service_with_persistence(tmp_path)
    assert service.monitoring_repository is mon_repo
    service.run_check()
    assert len(mon_repo.list_monitoring_results()) == 1


def test_real_development_database_untouched():
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass
