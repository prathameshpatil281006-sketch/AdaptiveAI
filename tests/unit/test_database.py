"""Focused unit tests for SQLite persistence foundation (Step 23).

All tests use temporary databases via tmp_path.
The real data/adaptive_ai.db must never be created or modified.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.core.config import get_project_root
from app.core.exceptions import AdaptiveAIError
from app.database.database import DatabaseManager, get_default_database_path
from app.database.repository import PredictionRepository


@pytest.fixture
def tmp_manager(tmp_path: Path) -> DatabaseManager:
    manager = DatabaseManager(db_path=tmp_path / "test.db")
    manager.init_db()
    return manager


@pytest.fixture
def tmp_repo(tmp_manager: DatabaseManager) -> PredictionRepository:
    return PredictionRepository(tmp_manager)


def test_database_can_be_created_at_temporary_path(tmp_path: Path):
    db_file = tmp_path / "nested" / "custom.db"
    assert not db_file.exists()
    manager = DatabaseManager(db_path=db_file)
    manager.init_db()
    assert db_file.exists()
    assert db_file.is_file()


def test_schema_tables_are_created(tmp_manager: DatabaseManager):
    conn = tmp_manager.get_connection()
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='predictions'"
        ).fetchone()
        assert row is not None
        columns = {
            r[1]: r[2]
            for r in conn.execute("PRAGMA table_info(predictions)").fetchall()
        }
        assert columns["id"] == "INTEGER"
        assert columns["timestamp"] == "TEXT"
        assert columns["use_case"] == "TEXT"
        assert columns["model_name"] == "TEXT"
        assert columns["model_version"] == "TEXT"
        assert columns["input_data"] == "TEXT"
        assert columns["prediction"] == "REAL"
    finally:
        DatabaseManager.close_connection(conn)


def test_prediction_can_be_inserted_and_retrieved(tmp_repo: PredictionRepository):
    payload = {"area": 2000.0, "bedrooms": 3, "bathrooms": 2, "property_age": 10.0, "parking": 1}
    record_id = tmp_repo.save_prediction(
        use_case="house_price",
        model_name="LinearRegression",
        input_data=payload,
        prediction=450000.0,
        model_version="1",
    )
    assert isinstance(record_id, int)

    record = tmp_repo.get_prediction_by_id(record_id)
    assert record is not None
    assert record["id"] == record_id
    assert record["use_case"] == "house_price"
    assert record["model_name"] == "LinearRegression"
    assert record["model_version"] == "1"
    assert record["input_data"] == payload
    assert record["prediction"] == pytest.approx(450000.0)


def test_multiple_predictions_can_be_listed(tmp_repo: PredictionRepository):
    for i in range(3):
        tmp_repo.save_prediction(
            use_case="house_price",
            model_name="LinearRegression",
            input_data={"area": 1000.0 + i},
            prediction=300000.0 + i,
        )
    records = tmp_repo.list_predictions()
    assert len(records) == 3
    # Step 63 (I-14): newest-first deterministic ordering.
    assert [r["id"] for r in records] == sorted((r["id"] for r in records), reverse=True)
    assert records[0]["input_data"] == {"area": 1002.0}
    assert records[-1]["input_data"] == {"area": 1000.0}


def test_model_version_can_be_null(tmp_repo: PredictionRepository):
    record_id = tmp_repo.save_prediction(
        use_case="house_price",
        model_name="LinearRegression",
        input_data={"area": 1500.0},
        prediction=350000.0,
    )
    record = tmp_repo.get_prediction_by_id(record_id)
    assert record is not None
    assert record["model_version"] is None


def test_json_input_data_round_trips(tmp_repo: PredictionRepository):
    payload = {
        "area": 2100.5,
        "bedrooms": 4,
        "nested": {"a": [1, 2, 3]},
        "label": "café",
    }
    record_id = tmp_repo.save_prediction(
        use_case="demand_forecasting",
        model_name="SomeModel",
        input_data=payload,
        prediction=12.5,
        model_version="2",
    )
    record = tmp_repo.get_prediction_by_id(record_id)
    assert record is not None
    assert record["input_data"] == payload

    conn = tmp_repo._db.get_connection()
    try:
        raw = conn.execute(
            "SELECT input_data FROM predictions WHERE id = ?", (record_id,)
        ).fetchone()[0]
        assert json.loads(raw) == payload
    finally:
        DatabaseManager.close_connection(conn)


def test_timestamps_are_utc_iso8601(tmp_repo: PredictionRepository):
    record_id = tmp_repo.save_prediction(
        use_case="house_price",
        model_name="LinearRegression",
        input_data={"area": 1200.0},
        prediction=200000.0,
    )
    record = tmp_repo.get_prediction_by_id(record_id)
    assert record is not None
    parsed = datetime.fromisoformat(str(record["timestamp"]))
    assert parsed.tzinfo is not None

    explicit = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    record_id2 = tmp_repo.save_prediction(
        use_case="house_price",
        model_name="LinearRegression",
        input_data={"area": 1300.0},
        prediction=210000.0,
        timestamp=explicit,
    )
    record2 = tmp_repo.get_prediction_by_id(record_id2)
    assert record2 is not None
    assert datetime.fromisoformat(str(record2["timestamp"])) == explicit


def test_database_path_is_configurable(tmp_path: Path):
    first = DatabaseManager(db_path=tmp_path / "a.db")
    second = DatabaseManager(db_path=tmp_path / "b.db")
    assert first.path != second.path
    first.init_db()
    assert (tmp_path / "a.db").exists()
    assert not (tmp_path / "b.db").exists()

    default_path = get_default_database_path()
    assert str(default_path).endswith("adaptive_ai.db")
    assert Path(get_project_root()) in default_path.parents


def test_invalid_operations_fail_cleanly(tmp_repo: PredictionRepository):
    assert tmp_repo.get_prediction_by_id(999999) is None

    with pytest.raises(AdaptiveAIError):
        tmp_repo.get_prediction_by_id("bad-id")  # type: ignore[arg-type]

    with pytest.raises(AdaptiveAIError):
        tmp_repo.save_prediction(
            use_case="house_price",
            model_name="LinearRegression",
            input_data="not-a-dict",  # type: ignore[arg-type]
            prediction=1.0,
        )

    with pytest.raises(AdaptiveAIError):
        tmp_repo.save_prediction(
            use_case="",
            model_name="LinearRegression",
            input_data={"area": 1.0},
            prediction=1.0,
        )

    with pytest.raises(AdaptiveAIError):
        tmp_repo.list_predictions(limit=-1)

    with pytest.raises(AdaptiveAIError):
        PredictionRepository(db_manager="not-a-manager")  # type: ignore[arg-type]


def test_real_development_database_untouched():
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass
