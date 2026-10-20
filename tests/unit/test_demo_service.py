"""Phase 6 tests: demo service lifecycle on isolated temporary state.

Verifies session creation/generation/monitoring/reset through
DemoService with temporary databases and model dirs: deterministic
samples, audit events, monitoring-history isolation, and the reset
safety contract. Never touches production state.
"""

from pathlib import Path

from app.database.database import DatabaseManager
from app.database.lifecycle_repository import LifecycleEventRepository
from app.database.monitoring_repository import MonitoringRepository
from app.database.repository import PredictionRepository
from app.demo.repository import DemoSessionRepository
from app.demo.service import DemoService


def _service(tmp_path: Path, name: str = "demo.db") -> dict:
    db_manager = DatabaseManager(db_path=tmp_path / name)
    db_manager.init_db()
    service = DemoService(
        prediction_repository=PredictionRepository(db_manager),
        session_repository=DemoSessionRepository(db_manager),
        event_repository=LifecycleEventRepository(db_manager),
        models_dir=str(tmp_path / "models"),
    )
    return {"service": service, "db": db_manager}


def test_create_session_fields_and_audit(tmp_path: Path):
    wired = _service(tmp_path)
    session = wired["service"].create_session("area_shift", sample_count=50, seed=9)
    assert session["session_id"].startswith("demo-")
    assert session["scenario"] == "area_shift"
    assert session["status"] == "active"
    assert session["sample_count"] == 50
    assert session["seed"] == 9
    assert session["samples_generated"] == 0
    assert session["created_at"] and session["started_at"]
    events = LifecycleEventRepository(wired["db"]).list_events(use_case="house_price")
    assert any(e["event"] == "demo_session_created" for e in events)


def test_generate_persists_isolated_demo_rows(tmp_path: Path):
    wired = _service(tmp_path)
    service = wired["service"]
    session = service.create_session("healthy_baseline", sample_count=30, seed=42)
    out = service.generate_samples(session["session_id"])
    assert out["samples_generated"] == 30
    assert out["session"]["samples_generated"] == 30
    assert out["session"]["status"] == "active"
    repo = PredictionRepository(wired["db"])
    assert repo.count_predictions("house_price", source="production") == 0
    assert repo.count_predictions("house_price", source="demo") == 30
    rows = repo.list_predictions(
        limit=30, use_case="house_price", source="demo",
        demo_session_id=session["session_id"],
    )
    assert len(rows) == 30
    assert all(r["source"] == "demo" for r in rows)
    assert all(r["demo_session_id"] == session["session_id"] for r in rows)


def test_generate_is_deterministic_across_sessions(tmp_path: Path):
    wired = _service(tmp_path)
    service = wired["service"]
    first = service.create_session("config_shift", sample_count=20, seed=123)
    second = service.create_session("config_shift", sample_count=20, seed=123)
    service.generate_samples(first["session_id"])
    service.generate_samples(second["session_id"])
    repo = PredictionRepository(wired["db"])
    rows_a = repo.list_predictions(
        limit=20, use_case="house_price", source="demo",
        demo_session_id=first["session_id"],
    )
    rows_b = repo.list_predictions(
        limit=20, use_case="house_price", source="demo",
        demo_session_id=second["session_id"],
    )
    inputs_a = sorted(str(r["input_data"]) for r in rows_a)
    inputs_b = sorted(str(r["input_data"]) for r in rows_b)
    assert inputs_a == inputs_b


def test_regenerate_replaces_session_rows(tmp_path: Path):
    wired = _service(tmp_path)
    service = wired["service"]
    session = service.create_session("area_shift", sample_count=10, seed=1)
    service.generate_samples(session["session_id"])
    out = service.generate_samples(session["session_id"])
    assert out["samples_generated"] == 10
    repo = PredictionRepository(wired["db"])
    assert repo.count_predictions(
        "house_price", source="demo", demo_session_id=session["session_id"]
    ) == 10


def test_monitor_uses_real_engine_without_persisting(tmp_path: Path):
    wired = _service(tmp_path)
    service = wired["service"]
    session = service.create_session("locality_shift", sample_count=100, seed=42)
    service.generate_samples(session["session_id"])
    before = MonitoringRepository(wired["db"]).list_monitoring_results(
        limit=100, use_case="house_price"
    )
    out = service.monitor_session(session["session_id"])
    monitoring = out["monitoring"]
    assert monitoring["demo_session_id"] == session["session_id"]
    assert monitoring["demo_scenario"] == "locality_shift"
    assert monitoring["status"] == "ok"
    assert monitoring["sufficient_data"] is True
    assert monitoring["production_rows"] == 100
    assert "decision" not in monitoring
    assert out["decision"]["recommended"] in (True, False)
    assert out["decision"]["reason"]
    after = MonitoringRepository(wired["db"]).list_monitoring_results(
        limit=100, use_case="house_price"
    )
    assert len(after) == len(before) == 0
    refreshed = service.get_session(session["session_id"])
    assert refreshed["status"] == "completed"
    assert refreshed["last_monitoring"]["demo_session_id"] == session["session_id"]
    events = LifecycleEventRepository(wired["db"]).list_events(use_case="house_price")
    assert any(e["event"] == "demo_monitoring_run" for e in events)


def test_monitor_area_shift_detects_area_drift(tmp_path: Path):
    wired = _service(tmp_path)
    service = wired["service"]
    session = service.create_session("area_shift", sample_count=100, seed=42)
    service.generate_samples(session["session_id"])
    out = service.monitor_session(session["session_id"])
    assert out["monitoring"]["overall_drifted"] is True
    assert "area_sqft" in out["monitoring"]["drifted_features"]


def test_monitor_requires_samples(tmp_path: Path):
    from app.core.exceptions import AdaptiveAIError

    wired = _service(tmp_path)
    session = wired["service"].create_session("area_shift")
    try:
        wired["service"].monitor_session(session["session_id"])
    except AdaptiveAIError:
        return
    raise AssertionError("monitoring without samples should fail")


def test_reset_removes_only_demo_rows(tmp_path: Path):
    from app.demo.scenarios import generate_scenario_samples

    wired = _service(tmp_path)
    service = wired["service"]
    repo = PredictionRepository(wired["db"])
    repo.save_prediction(
        use_case="house_price",
        model_name="Prod",
        input_data=generate_scenario_samples("healthy_baseline", 1, 1)[0],
        prediction=1e7,
        source="production",
    )
    session = service.create_session("area_shift", sample_count=25, seed=4)
    service.generate_samples(session["session_id"])
    assert repo.count_predictions("house_price", source="production") == 1
    out = service.reset_session(session["session_id"])
    assert out["removed_demo_rows"] == 25
    assert out["session"]["status"] == "reset"
    assert out["session"]["samples_generated"] == 0
    assert repo.count_predictions("house_price", source="production") == 1
    assert repo.count_predictions("house_price", source="demo") == 0
    events = LifecycleEventRepository(wired["db"]).list_events(use_case="house_price")
    assert any(e["event"] == "demo_session_reset" for e in events)
    assert any(e["event"] == "demo_session_created" for e in events)


def test_unknown_session_fails_cleanly(tmp_path: Path):
    from app.core.exceptions import AdaptiveAIError

    wired = _service(tmp_path)
    for action in (
        wired["service"].generate_samples,
        wired["service"].monitor_session,
        wired["service"].reset_session,
    ):
        try:
            action("demo-missing")
        except AdaptiveAIError:
            continue
        raise AssertionError("unknown session should fail")


def test_list_sessions_newest_first(tmp_path: Path):
    wired = _service(tmp_path)
    service = wired["service"]
    first = service.create_session("area_shift", sample_count=5, seed=1)
    second = service.create_session("locality_shift", sample_count=5, seed=1)
    items = service.list_sessions()
    assert [s["session_id"] for s in items] == [second["session_id"], first["session_id"]]
