"""Phase 6 API tests: demo endpoint contracts.

Verifies the Scenario Simulator HTTP surface on isolated temporary
databases (stubbed lifecycle for the retrain delegation): scenario
metadata, session lifecycle, generation/monitoring/reset isolation
from production reads, and error mapping. Never touches production
state.
"""

from pathlib import Path

from fastapi.testclient import TestClient

import app.api.main as api_main
from app.api.main import create_app
from app.database.database import DatabaseManager
from app.database.monitoring_repository import MonitoringRepository
from app.database.repository import PredictionRepository


def _client(tmp_path: Path, name: str = "demoapi.db", monkeypatch=None):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    if monkeypatch is not None:
        monkeypatch.setattr(api_main, "HOUSE_PRICE_MODELS_DIR", str(models_dir))
    db_manager = DatabaseManager(db_path=tmp_path / name)
    db_manager.init_db()
    return TestClient(create_app(db_manager=db_manager)), db_manager


def test_scenarios_contract(tmp_path: Path):
    client, _ = _client(tmp_path, "c1.db")
    body = client.get("/demo/house-price/scenarios").json()
    assert {s["name"] for s in body["items"]} == {
        "healthy_baseline",
        "area_shift",
        "locality_shift",
        "config_shift",
        "data_quality",
    }


def test_create_validation(tmp_path: Path):
    client, _ = _client(tmp_path, "c2.db")
    bad = client.post(
        "/demo/house-price/sessions",
        json={"scenario": "nope", "sample_count": 10, "seed": 1},
    )
    assert bad.status_code == 422
    bad_count = client.post(
        "/demo/house-price/sessions",
        json={"scenario": "area_shift", "sample_count": 0, "seed": 1},
    )
    assert bad_count.status_code == 422
    ok = client.post(
        "/demo/house-price/sessions",
        json={"scenario": "area_shift", "sample_count": 40, "seed": 42},
    )
    assert ok.status_code == 200
    session = ok.json()
    assert session["session_id"].startswith("demo-")
    assert session["status"] == "active"
    assert session["sample_count"] == 40


def test_full_flow_isolated_from_production_reads(tmp_path: Path, monkeypatch):
    client, db_manager = _client(tmp_path, "c3.db", monkeypatch)
    created = client.post(
        "/demo/house-price/sessions",
        json={"scenario": "area_shift", "sample_count": 100, "seed": 42},
    ).json()
    sid = created["session_id"]

    listed = client.get("/demo/house-price/sessions").json()
    assert listed["items"] and listed["items"][0]["session_id"] == sid

    fetched = client.get(f"/demo/house-price/sessions/{sid}").json()
    assert fetched["scenario"] == "area_shift"

    generated = client.post(f"/demo/house-price/sessions/{sid}/generate").json()
    assert generated["samples_generated"] == 100

    repo = PredictionRepository(db_manager)
    assert repo.count_predictions("house_price", source="production") == 0
    assert repo.count_predictions("house_price", source="demo") == 100
    history = client.get("/predict/house-price/history?limit=100&offset=0").json()
    assert history["items"] == []

    monitored = client.post(f"/demo/house-price/sessions/{sid}/monitor").json()
    assert monitored["monitoring"]["demo_session_id"] == sid
    assert "area_sqft" in monitored["monitoring"]["drifted_features"]
    assert monitored["decision"]["recommended"] is True

    mon_history = MonitoringRepository(db_manager).list_monitoring_results(
        limit=100, use_case="house_price"
    )
    assert mon_history == []
    summary = client.get("/monitor/house-price/summary").json()
    assert summary["production_rows"] == 0
    assert summary["status"] == "insufficient_data"

    reset = client.post(f"/demo/house-price/sessions/{sid}/reset").json()
    assert reset["removed_demo_rows"] == 100
    assert reset["session"]["status"] == "reset"
    assert repo.count_predictions("house_price", source="demo") == 0


def test_monitor_before_generate_fails(tmp_path: Path, monkeypatch):
    client, _ = _client(tmp_path, "c4.db", monkeypatch)
    sid = client.post(
        "/demo/house-price/sessions",
        json={"scenario": "area_shift", "sample_count": 10, "seed": 1},
    ).json()["session_id"]
    resp = client.post(f"/demo/house-price/sessions/{sid}/monitor")
    assert resp.status_code == 422


def test_unknown_session_404s(tmp_path: Path, monkeypatch):
    client, _ = _client(tmp_path, "c5.db", monkeypatch)
    assert client.get("/demo/house-price/sessions/demo-missing").status_code == 404
    assert client.post("/demo/house-price/sessions/demo-missing/generate").status_code == 404
    assert client.post("/demo/house-price/sessions/demo-missing/monitor").status_code == 404
    assert client.post("/demo/house-price/sessions/demo-missing/reset").status_code == 404


def test_demo_retrain_delegates_with_demo_trigger(tmp_path: Path, monkeypatch):
    calls = []

    class _StubLifecycle:
        def start_run(self, production_limit=1000, trigger="manual", data_path=None):
            calls.append((production_limit, trigger, data_path))
            return {"run_id": "rtr_demo1", "status": "completed", "trigger": trigger}

    monkeypatch.setattr(
        api_main,
        "create_house_price_lifecycle_service",
        lambda db_manager=None: _StubLifecycle(),
    )
    client, _ = _client(tmp_path, "c6.db", monkeypatch)
    sid = client.post(
        "/demo/house-price/sessions",
        json={"scenario": "locality_shift", "sample_count": 10, "seed": 1},
    ).json()["session_id"]
    client.post(f"/demo/house-price/sessions/{sid}/generate")
    body = client.post(
        f"/demo/house-price/sessions/{sid}/retrain", json={"production_limit": 1000}
    ).json()
    assert body["run_id"] == "rtr_demo1"
    assert calls and calls[0][1].startswith("demo:locality_shift:")
    assert sid in calls[0][1]


def test_demo_retrain_unknown_session_404(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        api_main,
        "create_house_price_lifecycle_service",
        lambda db_manager=None: None,
    )
    client, _ = _client(tmp_path, "c7.db", monkeypatch)
    resp = client.post(
        "/demo/house-price/sessions/demo-missing/retrain", json={"production_limit": 1000}
    )
    assert resp.status_code == 404
