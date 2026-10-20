"""Step 64 (I-15) focused tests: close_connection() never masks exceptions.

All databases are temporary; no MLflow/model side effects.
"""

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.core.exceptions import AdaptiveAIError
from app.database.database import DatabaseManager
from app.database.repository import PredictionRepository


def _manager(tmp_path: Path, name: str = "close.db") -> DatabaseManager:
    manager = DatabaseManager(db_path=tmp_path / name)
    manager.init_db()
    return manager


def test_normal_close_returns_none_and_double_close_safe(tmp_path: Path):
    manager = _manager(tmp_path)
    conn = manager.get_connection()
    assert DatabaseManager.close_connection(conn) is None
    # Already-closed connection is a safe no-op.
    assert DatabaseManager.close_connection(conn) is None


def test_none_is_safe_noop():
    assert DatabaseManager.close_connection(None) is None


def test_sqlite_close_error_suppressed():
    broken = MagicMock()
    broken.close.side_effect = sqlite3.Error("cleanup boom")
    assert DatabaseManager.close_connection(broken) is None


def test_non_sqlite_close_error_suppressed():
    broken = MagicMock()
    broken.close.side_effect = RuntimeError("weird cleanup failure")
    assert DatabaseManager.close_connection(broken) is None


def test_cleanup_does_not_mask_active_exception():
    broken = MagicMock()
    broken.close.side_effect = sqlite3.Error("cleanup boom")
    with pytest.raises(AdaptiveAIError, match="original A"):
        try:
            raise AdaptiveAIError("original A")
        finally:
            DatabaseManager.close_connection(broken)


class _CloseBoomConn:
    """Wraps a real sqlite3 connection; close() closes then raises."""

    def __init__(self, real):
        object.__setattr__(self, "_real", real)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_real"), name)

    def close(self):
        object.__getattribute__(self, "_real").close()
        raise sqlite3.Error("cleanup boom")


def test_repository_save_masks_nothing_when_close_fails(tmp_path: Path, monkeypatch):
    manager = _manager(tmp_path, "mask.db")
    real_get = manager.get_connection

    monkeypatch.setattr(manager, "get_connection", lambda: _CloseBoomConn(real_get()))
    repo = PredictionRepository(manager)
    record_id = repo.save_prediction(
        use_case="house_price",
        model_name="LinearRegression",
        input_data={"area": 1.0},
        prediction=1.0,
    )
    assert isinstance(record_id, int)


def test_repository_operation_error_not_masked_by_close_failure(
    tmp_path: Path, monkeypatch
):
    manager = _manager(tmp_path, "mask2.db")

    failing = MagicMock()
    failing.execute.side_effect = sqlite3.Error("original boom")
    failing.close.side_effect = sqlite3.Error("cleanup boom")
    monkeypatch.setattr(manager, "get_connection", lambda: failing)

    repo = PredictionRepository(manager)
    with pytest.raises(AdaptiveAIError) as exc_info:
        repo.save_prediction(
            use_case="house_price",
            model_name="LinearRegression",
            input_data={"area": 1.0},
            prediction=1.0,
        )
    assert "original boom" in str(exc_info.value)
    assert "cleanup" not in str(exc_info.value).lower()
    assert "Failed to close" not in str(exc_info.value)


def test_init_db_close_failure_suppressed(tmp_path: Path, monkeypatch):
    manager = DatabaseManager(db_path=tmp_path / "initclose.db")
    real_connect = sqlite3.connect

    def _connect_close_boom(*args, **kwargs):
        return _CloseBoomConn(real_connect(*args, **kwargs))

    monkeypatch.setattr(sqlite3, "connect", _connect_close_boom)
    manager.init_db()  # must not raise
    assert (tmp_path / "initclose.db").exists()


def test_init_db_operation_error_not_masked(tmp_path: Path, monkeypatch):
    manager = DatabaseManager(db_path=tmp_path / "initmask.db")
    failing = MagicMock()
    failing.execute.side_effect = sqlite3.Error("init boom")
    failing.close.side_effect = sqlite3.Error("cleanup boom")
    monkeypatch.setattr(manager, "get_connection", lambda: failing)
    with pytest.raises(AdaptiveAIError) as exc_info:
        manager.init_db()
    assert "init boom" in str(exc_info.value)
    assert "cleanup" not in str(exc_info.value).lower()


def test_existing_db_operations_still_work(tmp_path: Path):
    repo = PredictionRepository(_manager(tmp_path, "ops.db"))
    record_id = repo.save_prediction(
        use_case="house_price",
        model_name="LinearRegression",
        input_data={"area": 2000.0},
        prediction=450000.0,
    )
    record = repo.get_prediction_by_id(record_id)
    assert record is not None and record["id"] == record_id
    assert repo.list_predictions()[0]["id"] == record_id
