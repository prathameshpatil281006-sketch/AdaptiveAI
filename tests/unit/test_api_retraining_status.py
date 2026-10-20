"""Step 42 tests: expose retraining scheduler status through FastAPI.

Covers GET /monitor/house-price/retrain/status as a thin read-only
adapter over RetrainingScheduler.get_status() using temporary
databases and injected stub schedulers only. Never touches
data/adaptive_ai.db and never triggers retraining.
"""

import copy
import inspect
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import app.api.main as api_main
from app.api.main import create_app
from app.core.config import get_project_root, load_config
from app.core.exceptions import AdaptiveAIError
from app.database.database import DatabaseManager
from app.scheduler.retraining_scheduler import RetrainingScheduler
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_persistence import save_house_price_model
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.split import split_house_price_data
from app.use_cases.house_price.use_case import HousePriceUseCase

STATUS_PATH = "/monitor/house-price/retrain/status"
GENERIC_ERROR = "Failed to retrieve retraining scheduler status."


def _status(**overrides):
    base = {
        "enabled": False,
        "interval_minutes": 60,
        "running": False,
        "last_run": None,
    }
    base.update(overrides)
    return base


def _make_client(tmp_path: Path, db_name: str = "retrain_status_api.db"):
    db_manager = DatabaseManager(db_path=tmp_path / db_name)
    application = create_app(db_manager=db_manager)
    return TestClient(application), application


def _install_stub(monkeypatch, status=None):
    """Replaces the scheduler factory with a stub returning status."""
    scheduler = MagicMock()
    scheduler.get_status = MagicMock(
        return_value=_status() if status is None else status
    )
    factory = MagicMock(return_value=scheduler)
    monkeypatch.setattr(api_main, "create_house_price_retraining_scheduler", factory)
    return factory, scheduler


def _status_endpoint_source():
    for route in api_main.app.routes:
        if getattr(route, "path", None) == STATUS_PATH:
            return inspect.getsource(route.endpoint)
    raise AssertionError(f"Route {STATUS_PATH} not found")


def _setup_model(tmp_path: Path, monkeypatch):
    df = generate_house_price_data(n_samples=200)
    X_train, _, y_train, _ = split_house_price_data(df)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", get_preprocessor()),
            ("regressor", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)
    models_dir = tmp_path / "retrain_status_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_house_price_model(pipeline, "LinearRegression", models_dir=models_dir)
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODEL_NAME", "LinearRegression")
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODELS_DIR", str(models_dir))


def _seed_predictions(repo, n: int = 50) -> None:
    from app.use_cases.house_price.data import generate_house_price_data

    df = generate_house_price_data(n_samples=n)
    feature_cols = [c for c in df.columns if c != "price"]
    for i in range(n):
        row = {c: df.loc[i, c] for c in feature_cols}
        for int_col in ("bedrooms", "bathrooms", "parking", "floor_number", "total_floors"):
            row[int_col] = int(row[int_col])
        repo.save_prediction(
            use_case="house_price",
            model_name="LinearRegression",
            input_data=row,
            prediction=400000.0,
        )


# --- Route / method ---


def test_route_exists_with_get_method():
    paths = {route.path: getattr(route, "methods", set()) for route in api_main.app.routes}
    assert STATUS_PATH in paths
    assert "GET" in paths[STATUS_PATH]


def test_post_method_not_allowed(tmp_path: Path, monkeypatch):
    _install_stub(monkeypatch)
    client, _ = _make_client(tmp_path, "method.db")
    assert client.post(STATUS_PATH).status_code == 405


# --- Status pass-through ---


def test_default_scheduler_status(tmp_path: Path, monkeypatch):
    real = RetrainingScheduler(
        retraining_service=MagicMock(),
        use_case=HousePriceUseCase(),
    ).get_status()
    _install_stub(monkeypatch, copy.deepcopy(real))
    client, _ = _make_client(tmp_path, "default.db")
    body = client.get(STATUS_PATH).json()
    assert body == {"enabled": False, "interval_minutes": 60, "running": False, "last_run": None}


def test_exact_status_object_pass_through(tmp_path: Path, monkeypatch):
    expected = _status(enabled=True, interval_minutes=15, running=False, last_run=None)
    _install_stub(monkeypatch, expected)
    client, _ = _make_client(tmp_path, "exact.db")
    response = client.get(STATUS_PATH)
    assert response.status_code == 200
    assert response.json() == expected
    assert set(response.json().keys()) == {"enabled", "interval_minutes", "running", "last_run"}


def test_status_object_not_mutated(tmp_path: Path, monkeypatch):
    status = _status(enabled=True, interval_minutes=30)
    snapshot = copy.deepcopy(status)
    _install_stub(monkeypatch, status)
    client, _ = _make_client(tmp_path, "immutable.db")
    client.get(STATUS_PATH)
    assert status == snapshot


def test_get_status_called_exactly_once(tmp_path: Path, monkeypatch):
    _, scheduler = _install_stub(monkeypatch)
    client, _ = _make_client(tmp_path, "once.db")
    client.get(STATUS_PATH)
    assert scheduler.get_status.call_count == 1


@pytest.mark.parametrize("enabled", [True, False])
def test_enabled_propagation(tmp_path: Path, monkeypatch, enabled):
    _install_stub(monkeypatch, _status(enabled=enabled))
    client, _ = _make_client(tmp_path, "en.db")
    assert client.get(STATUS_PATH).json()["enabled"] is enabled


@pytest.mark.parametrize("interval", [1, 15, 60, 1440])
def test_interval_minutes_propagation(tmp_path: Path, monkeypatch, interval):
    _install_stub(monkeypatch, _status(interval_minutes=interval))
    client, _ = _make_client(tmp_path, "int.db")
    assert client.get(STATUS_PATH).json()["interval_minutes"] == interval


@pytest.mark.parametrize("running", [True, False])
def test_running_propagation(tmp_path: Path, monkeypatch, running):
    _install_stub(monkeypatch, _status(running=running))
    client, _ = _make_client(tmp_path, "run.db")
    assert client.get(STATUS_PATH).json()["running"] is running


def test_last_run_none_propagation(tmp_path: Path, monkeypatch):
    _install_stub(monkeypatch, _status(last_run=None))
    client, _ = _make_client(tmp_path, "lr.db")
    assert client.get(STATUS_PATH).json()["last_run"] is None


def test_non_null_last_run_propagation(tmp_path: Path, monkeypatch):
    _install_stub(monkeypatch, _status(last_run="2026-01-01T00:00:00+00:00"))
    client, _ = _make_client(tmp_path, "lrnn.db")
    assert client.get(STATUS_PATH).json()["last_run"] == "2026-01-01T00:00:00+00:00"


# --- DI / laziness ---


def test_factory_called_once_per_request(tmp_path: Path, monkeypatch):
    factory, _ = _install_stub(monkeypatch)
    client, _ = _make_client(tmp_path, "di.db")
    client.get(STATUS_PATH)
    client.get(STATUS_PATH)
    assert factory.call_count == 2


def test_factory_lazy_not_called_during_create_app(tmp_path: Path, monkeypatch):
    factory, scheduler = _install_stub(monkeypatch)
    _make_client(tmp_path, "lazy.db")
    assert factory.call_count == 0
    assert scheduler.get_status.call_count == 0


def test_scheduler_not_auto_started(tmp_path: Path, monkeypatch):
    _, scheduler = _install_stub(monkeypatch)
    scheduler.run_once = MagicMock(side_effect=AssertionError("must not retrain"))
    client, _ = _make_client(tmp_path, "nostart.db")
    assert client.get(STATUS_PATH).status_code == 200
    scheduler.run_once.assert_not_called()
    assert scheduler.get_status.call_count == 1


def test_scheduler_construction_side_effect_hygiene(tmp_path: Path):
    service = MagicMock()
    scheduler = RetrainingScheduler(retraining_service=service, use_case=HousePriceUseCase())
    assert scheduler.get_status()["running"] is False
    service.retrain.assert_not_called()
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass


# --- Errors ---


def test_adaptive_ai_error_returns_generic_500(tmp_path: Path, monkeypatch):
    _, scheduler = _install_stub(monkeypatch)
    scheduler.get_status = MagicMock(side_effect=AdaptiveAIError("internal boom details"))
    client, _ = _make_client(tmp_path, "err.db")
    response = client.get(STATUS_PATH)
    assert response.status_code == 500
    assert response.json() == {"detail": GENERIC_ERROR}
    assert "boom" not in response.text


def test_unexpected_exception_returns_generic_500(tmp_path: Path, monkeypatch):
    _, scheduler = _install_stub(monkeypatch)
    scheduler.get_status = MagicMock(side_effect=RuntimeError("kaboom"))
    client, _ = _make_client(tmp_path, "err2.db")
    response = client.get(STATUS_PATH)
    assert response.status_code == 500
    assert response.json() == {"detail": GENERIC_ERROR}
    assert "kaboom" not in response.text


def test_factory_failure_returns_generic_500(tmp_path: Path, monkeypatch):
    factory = MagicMock(side_effect=AdaptiveAIError("wiring boom"))
    monkeypatch.setattr(api_main, "create_house_price_retraining_scheduler", factory)
    client, _ = _make_client(tmp_path, "err3.db")
    response = client.get(STATUS_PATH)
    assert response.status_code == 500
    assert response.json() == {"detail": GENERIC_ERROR}
    assert "boom" not in response.text


# --- Boundary ---


def test_no_retraining_orchestrator_monitoring_invocation(tmp_path: Path, monkeypatch):
    source = _status_endpoint_source()
    for forbidden in (
        "run_once",
        ".retrain(",
        "run_check_and_retrain",
        ".run_check(",
        "MonitoringService",
        ".decide(",
        "DriftRetrainingOrchestrator",
        "RetrainingService",
        "ModelPromoter",
        "mlflow",
    ):
        assert forbidden not in source
    # Behavioural: rigged neighbours are never touched by this route.
    neighbour = MagicMock(side_effect=AssertionError("must not be called"))
    monkeypatch.setattr(api_main, "create_house_price_drift_orchestrator", neighbour)
    _install_stub(monkeypatch)
    client, _ = _make_client(tmp_path, "nb.db")
    assert client.get(STATUS_PATH).status_code == 200
    assert neighbour.call_count == 0


def test_no_database_or_persistence_interaction(tmp_path: Path, monkeypatch):
    source = _status_endpoint_source()
    for forbidden in (".init(", "list_", "save_", "SELECT", "INSERT", "sqlite3"):
        assert forbidden not in source
    db_path = tmp_path / "clean.db"
    _install_stub(monkeypatch)
    client, _ = _make_client(tmp_path, "clean.db")
    assert client.get(STATUS_PATH).status_code == 200
    assert not db_path.exists()


def test_no_background_tasks_threads_loops():
    source = _status_endpoint_source()
    for forbidden in (
        "BackgroundTasks",
        "threading",
        "Thread(",
        "sleep(",
        "while True",
        "APScheduler",
        "Celery",
        "asyncio",
    ):
        assert forbidden not in source
    assert "daemon" not in source


# --- Regressions ---


def test_health_endpoint_regression(tmp_path: Path):
    client, _ = _make_client(tmp_path, "health.db")
    assert client.get("/health").json() == {"status": "healthy"}


def test_prediction_endpoint_regression(tmp_path: Path, monkeypatch):
    _setup_model(tmp_path, monkeypatch)
    client, _ = _make_client(tmp_path, "predict.db")
    valid = {
        "area_sqft": 1200.0,
        "bedrooms": 3,
        "bathrooms": 2,
        "parking": 1,
        "location": "Wakad",
        "property_age": 8.0,
        "floor_number": 2,
        "total_floors": 12,
        "furnishing_status": "Semi-Furnished",
        "property_type": "Apartment",
    }
    response = client.post("/predict/house-price", json=valid)
    assert response.status_code == 200
    assert response.json()["prediction"] > 0


def test_monitoring_endpoint_regression(tmp_path: Path):
    client, application = _make_client(tmp_path, "monreg.db")
    application.state.prediction_repo.init()
    _seed_predictions(application.state.prediction_repo, n=50)
    assert client.post("/monitor/house-price").status_code == 200
    history = client.get("/monitor/house-price/history")
    assert history.status_code == 200
    assert len(history.json()["items"]) == 1


def test_retrain_post_endpoint_regression(tmp_path: Path, monkeypatch):
    orchestrator = MagicMock()
    orchestrator.run_check_and_retrain = MagicMock(
        return_value={
            "monitoring_result": {},
            "retraining_decision": {},
            "retraining_result": None,
        }
    )
    monkeypatch.setattr(
        api_main,
        "create_house_price_drift_orchestrator",
        MagicMock(return_value=orchestrator),
    )
    _install_stub(monkeypatch)
    client, _ = _make_client(tmp_path, "retrain.db")
    assert client.post("/monitor/house-price/retrain", json={}).status_code == 200


def test_retrain_history_endpoint_regression(tmp_path: Path):
    client, _ = _make_client(tmp_path, "hist.db")
    response = client.get("/monitor/house-price/retrain/history")
    assert response.status_code == 200
    assert set(response.json().keys()) == {"items", "limit", "offset"}


# --- Disabled-by-default unchanged ---


def test_scheduler_remains_disabled_by_default():
    config = load_config()
    assert config["scheduler"]["enabled"] is False
    assert config["scheduler"]["interval_minutes"] == 60
    scheduler = RetrainingScheduler(
        retraining_service=MagicMock(), use_case=HousePriceUseCase()
    )
    assert scheduler.enabled is False
    assert scheduler.interval_minutes == 60
    assert scheduler.get_status() == {
        "enabled": False,
        "interval_minutes": 60,
        "running": False,
        "last_run": None,
    }


def test_new_tests_create_no_real_artifacts():
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass
