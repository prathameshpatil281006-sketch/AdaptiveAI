"""Phase 5 tests: lifecycle repositories (runs, events, pointer history).

CRUD, validation, newest-first ordering, and JSON round-trips on
temporary databases. Never touches production data.
"""

from pathlib import Path

import pytest

from app.core.exceptions import AdaptiveAIError
from app.database.database import DatabaseManager
from app.database.lifecycle_repository import (
    LifecycleEventRepository,
    PointerHistoryRepository,
    RetrainingRunRepository,
)


def _manager(tmp_path: Path, name: str = "lifecycle.db") -> DatabaseManager:
    manager = DatabaseManager(db_path=tmp_path / name)
    manager.init_db()
    return manager


def test_run_lifecycle_round_trip(tmp_path: Path):
    repo = RetrainingRunRepository(_manager(tmp_path))
    created = repo.create_run(
        run_id="rtr_abc123",
        use_case="house_price",
        trigger="manual",
        dataset_rows=8500,
        schema_version=2,
        feature_schema=["area_sqft"],
    )
    assert created["status"] == "started"
    assert created["completed_at"] is None
    assert created["feature_schema"] == ["area_sqft"]

    fetched = repo.get_run("rtr_abc123")
    assert fetched is not None and fetched["id"] == created["id"]
    assert repo.get_run("rtr_missing") is None

    updated = repo.update_run(
        "rtr_abc123",
        status="completed",
        best_model="RandomForestRegressor",
        best_metrics={"mae": 1.0, "rmse": 2.0, "r2": 0.9},
        promotion_eligible=True,
        gate_checks=[{"name": "run_completed", "passed": True, "message": "ok"}],
    )
    assert updated["status"] == "completed"
    assert updated["best_model"] == "RandomForestRegressor"
    assert updated["best_metrics"]["r2"] == 0.9
    assert updated["promotion_eligible"] is True
    assert updated["gate_checks"][0]["name"] == "run_completed"

    listed = repo.list_runs(use_case="house_price")
    assert [r["run_id"] for r in listed] == ["rtr_abc123"]
    with pytest.raises(AdaptiveAIError):
        repo.update_run("rtr_missing", status="failed")
    # run_id is a bound parameter: rebinding it is a TypeError, so the
    # run identity can never be updated through this method.
    with pytest.raises(TypeError):
        repo.update_run("rtr_abc123", run_id="immutable")  # type: ignore[call-arg]
    with pytest.raises(AdaptiveAIError):
        repo.update_run("rtr_abc123", status="bogus")


def test_run_ids_unique(tmp_path: Path):
    repo = RetrainingRunRepository(_manager(tmp_path, "uniq.db"))
    repo.create_run(run_id="rtr_dup", use_case="house_price", trigger="manual")
    with pytest.raises(AdaptiveAIError):
        repo.create_run(run_id="rtr_dup", use_case="house_price", trigger="manual")


def test_events_append_and_list(tmp_path: Path):
    repo = LifecycleEventRepository(_manager(tmp_path, "events.db"))
    first = repo.record_event(
        use_case="house_price",
        event="retraining_started",
        run_id="rtr_1",
        reason="go",
    )
    repo.record_event(
        use_case="house_price",
        event="model_promoted",
        run_id="rtr_1",
        model_name="RandomForestRegressor",
        model_version="5",
        previous_production_version="4",
        new_production_version="5",
        reason="better",
        details={"r2": 0.93},
    )
    items = repo.list_events(use_case="house_price")
    assert [e["id"] for e in items] == sorted([e["id"] for e in items], reverse=True)
    promoted = next(e for e in items if e["event"] == "model_promoted")
    assert promoted["model_version"] == "5"
    assert promoted["previous_production_version"] == "4"
    assert promoted["details"] == {"r2": 0.93}
    assert first == items[-1]["id"]
    with pytest.raises(AdaptiveAIError):
        repo.record_event(use_case="house_price", event="bogus_event")


def test_pointer_history_round_trip(tmp_path: Path):
    repo = PointerHistoryRepository(_manager(tmp_path, "pointer.db"))
    assert repo.list_transitions(use_case="house_price") == []
    assert repo.get_transition(999) is None
    prev = {"model_name": "LinearRegression", "model_version": "311"}
    new = {"model_name": "RandomForestRegressor", "model_version": "312"}
    record_id = repo.record_transition(
        use_case="house_price",
        new_pointer=new,
        previous_pointer=prev,
        reason="promotion",
        run_id="rtr_1",
    )
    fetched = repo.get_transition(record_id)
    assert fetched is not None
    assert fetched["previous_pointer"] == prev
    assert fetched["new_pointer"] == new
    assert fetched["run_id"] == "rtr_1"
    items = repo.list_transitions(use_case="house_price")
    assert len(items) == 1 and items[0]["id"] == record_id
    with pytest.raises(AdaptiveAIError):
        repo.record_transition(use_case="house_price", new_pointer={})
