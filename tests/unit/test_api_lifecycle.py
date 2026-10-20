"""Phase 5 API tests: lifecycle endpoints and extended registry.

Endpoint contracts are verified with a stubbed lifecycle service
(fast, hermetic); the registry extension is verified against real
temporary state. Never touches production state.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.api.main as api_main
from app.api.main import create_app
from app.core.config import get_project_root
from app.database.database import DatabaseManager
from app.database.lifecycle_repository import (
    LifecycleEventRepository,
    PointerHistoryRepository,
    RetrainingRunRepository,
)


class _StubLifecycle:
    def __init__(self, **overrides):
        self._overrides = overrides
        self.calls = []

    def recommend(self, summary, production_model=None, production_version=None):
        self.calls.append(("recommend", summary))
        return self._overrides.get(
            "recommend",
            {
                "recommended": True,
                "reason": "critical drift",
                "triggers": ["numeric-drift:area_sqft"],
                "blocked_by": [],
                "production_model": production_model,
                "production_version": production_version,
                "monitored_prediction_count": 60,
            },
        )

    def start_run(self, production_limit=1000, trigger="manual", data_path=None):
        self.calls.append(("start_run", trigger))
        return self._overrides.get(
            "start_run",
            {"run_id": "rtr_stub1", "status": "completed", "best_model": "LinearRegression"},
        )

    def list_runs(self, limit=100, offset=0, use_case=None):
        return self._overrides.get("list_runs", [])

    def get_run(self, run_id):
        return self._overrides.get("get_run", {"run_id": run_id, "status": "completed"})

    def promotion_check(self, run_id, window_limit=1000):
        return self._overrides.get(
            "promotion_check", {"run_id": run_id, "eligible": True, "checks": []}
        )

    def promote(self, run_id, allow_migration=False):
        self.calls.append(("promote", run_id, allow_migration))
        return self._overrides.get(
            "promote", {"run_id": run_id, "promoted": True, "eligible": True}
        )

    def rollback(self, pointer_history_id):
        return self._overrides.get(
            "rollback",
            {"pointer_history_id": pointer_history_id, "rolled_back": True},
        )


def _client(tmp_path: Path, name: str = "lcapi.db"):
    db_manager = DatabaseManager(db_path=tmp_path / name)
    return TestClient(create_app(db_manager=db_manager)), db_manager


def _stubbed(tmp_path: Path, monkeypatch, **overrides):
    stub = _StubLifecycle(**overrides)
    monkeypatch.setattr(
        api_main, "create_house_price_lifecycle_service", lambda db_manager=None: stub
    )
    client, _ = _client(tmp_path)
    return client, stub


def test_decision_endpoint_shape(tmp_path: Path, monkeypatch):
    client, stub = _stubbed(tmp_path, monkeypatch)
    body = client.get("/retrain/house-price/decision").json()
    assert body["decision"]["recommended"] is True
    assert body["decision"]["triggers"] == ["numeric-drift:area_sqft"]
    assert body["monitoring"]["production_rows"] == 0
    assert body["monitoring"]["status"] == "insufficient_data"
    assert stub.calls and stub.calls[0][0] == "recommend"


def test_start_run_endpoint(tmp_path: Path, monkeypatch):
    client, stub = _stubbed(tmp_path, monkeypatch)
    body = client.post(
        "/retrain/house-price/runs", json={"production_limit": 50, "trigger": "manual"}
    ).json()
    assert body["run_id"] == "rtr_stub1"
    assert body["status"] == "completed"
    assert ("start_run", "manual") in stub.calls


def test_start_run_validation(tmp_path: Path, monkeypatch):
    client, _ = _stubbed(tmp_path, monkeypatch)
    assert (
        client.post("/retrain/house-price/runs", json={"production_limit": 0}).status_code
        == 422
    )


def test_runs_list_and_detail(tmp_path: Path, monkeypatch):
    runs = [{"run_id": "rtr_a", "status": "completed"}]
    client, _ = _stubbed(tmp_path, monkeypatch, list_runs=runs)
    body = client.get("/retrain/house-price/runs").json()
    assert body["items"] == runs
    assert body["limit"] == 100 and body["offset"] == 0

    client2, _ = _stubbed(tmp_path, monkeypatch, get_run=None)
    assert client2.get("/retrain/house-price/runs/rtr_nope").status_code == 404


def test_promotion_check_and_promote(tmp_path: Path, monkeypatch):
    client, _ = _stubbed(tmp_path, monkeypatch)
    check = client.post(
        "/retrain/house-price/runs/rtr_stub1/promotion-check",
        json={"window_limit": 50},
    ).json()
    assert check["eligible"] is True
    promoted = client.post("/retrain/house-price/runs/rtr_stub1/promote").json()
    assert promoted["promoted"] is True

    client2, _ = _stubbed(tmp_path, monkeypatch, promote=None)
    assert client2.post("/retrain/house-price/runs/rtr_nope/promote").status_code == 404


def test_promote_forwards_migration_confirmation(tmp_path: Path, monkeypatch):
    client, stub = _stubbed(tmp_path, monkeypatch)
    body = client.post(
        "/retrain/house-price/runs/rtr_stub1/promote", json={"allow_migration": True}
    ).json()
    assert body["promoted"] is True
    assert ("promote", "rtr_stub1", True) in stub.calls
    client.post("/retrain/house-price/runs/rtr_stub1/promote").json()
    assert ("promote", "rtr_stub1", False) in stub.calls


def test_rollback_endpoint(tmp_path: Path, monkeypatch):
    client, _ = _stubbed(tmp_path, monkeypatch)
    body = client.post("/retrain/house-price/rollback", json={"pointer_history_id": 3}).json()
    assert body["rolled_back"] is True

    client2, _ = _stubbed(tmp_path, monkeypatch, rollback=None)
    assert (
        client2.post("/retrain/house-price/rollback", json={"pointer_history_id": 999}).status_code
        == 404
    )
    assert (
        client.post("/retrain/house-price/rollback", json={"pointer_history_id": 0}).status_code
        == 422
    )


def test_registry_extension_with_tmp_state(tmp_path: Path, monkeypatch):
    import app.retraining.registry_info as registry_info_module

    db_manager = DatabaseManager(db_path=tmp_path / "reg.db")
    db_manager.init_db()
    run_repo = RetrainingRunRepository(db_manager)
    run_repo.create_run(run_id="rtr_reg", use_case="house_price", trigger="manual")
    run_repo.update_run(
        "rtr_reg",
        status="completed",
        best_model="RandomForestRegressor",
        best_metrics={"mae": 1.0, "rmse": 2.0, "r2": 0.9},
        candidate_artifact=".staging/rtr_reg",
        promotion_eligible=True,
    )
    event_repo = LifecycleEventRepository(db_manager)
    event_repo.record_event(
        use_case="house_price", event="retraining_completed", run_id="rtr_reg"
    )
    pointer_repo = PointerHistoryRepository(db_manager)
    pointer_repo.record_transition(
        use_case="house_price",
        new_pointer={"model_name": "M", "model_version": "2"},
        previous_pointer={"model_name": "M", "model_version": "1"},
        run_id="rtr_reg",
    )
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    out = registry_info_module.get_house_price_registry_info(
        limit=10, db_manager=db_manager, models_dir=models_dir
    )
    assert out["production"] is None or isinstance(out["production"], dict)
    assert len(out["runs"]) == 1
    candidate = out["runs"][0]
    assert candidate["run_id"] == "rtr_reg"
    assert candidate["candidate_model"] == "RandomForestRegressor"
    assert candidate["candidate_metrics"] == {"mae": 1.0, "rmse": 2.0, "r2": 0.9}
    assert candidate["artifact_available"] is False
    assert candidate["promotion_eligible"] is True
    assert len(out["pointer_history"]) == 1
    assert out["pointer_history"][0]["run_id"] == "rtr_reg"
    assert isinstance(out["versions"], list)
    assert isinstance(out["mlflow_versions"], list)


def test_registry_endpoint_still_serves(tmp_path: Path):
    client, _ = _client(tmp_path, "regapi.db")
    body = client.get("/models/house-price/registry?limit=5").json()
    assert set(body.keys()) >= {
        "production",
        "runs",
        "pointer_history",
        "versions",
        "mlflow_versions",
        "total",
    }
    assert isinstance(body["runs"], list)
    assert isinstance(body["pointer_history"], list)


def test_real_production_pointer_unchanged():
    root = Path(get_project_root())
    pointer = (root / "models" / "house_price" / "production.json").read_text(
        encoding="utf-8"
    )
    # Sanctioned schema-migration promotion (run rtr_e82b9a983bcd):
    # production is RandomForestRegressor v313 since 2026-10-04.
    assert '"model_name": "RandomForestRegressor"' in pointer
    assert '"model_version": "313"' in pointer
    # Isolation canary: live-state guard lives in tests/conftest.py.
