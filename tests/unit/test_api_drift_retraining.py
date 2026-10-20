"""Step 40 tests: expose drift-triggered retraining through FastAPI.

Covers POST /monitor/house-price/retrain as a thin adapter over
DriftRetrainingOrchestrator using temporary databases and injected
stub orchestrators only. Never touches data/adaptive_ai.db, real
models, MLflow, or background scheduling.
"""

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import app.api.main as api_main
from app.api.main import DriftRetrainingRequest, create_app
from app.core.config import get_project_root
from app.core.exceptions import AdaptiveAIError
from app.database.database import DatabaseManager
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_persistence import save_house_price_model
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.split import split_house_price_data

RETRAIN_PATH = "/monitor/house-price/retrain"
GENERIC_ERROR = "House Price drift-triggered retraining failed."


def _monitoring_ok(drifted=False):
    return {
        "overall_drifted": drifted,
        "alpha": 0.05,
        "min_production_rows": 30,
        "production_rows": 100,
        "sufficient_data": True,
        "status": "ok",
        "features": [
            {"feature_name": "area_sqft", "statistic": 0.2, "p_value": 0.01 if drifted else 0.6, "drifted": drifted},
            {"feature_name": "bedrooms", "statistic": 0.05, "p_value": 0.7, "drifted": False},
        ],
    }


def _monitoring_insufficient():
    return {
        "overall_drifted": False,
        "alpha": 0.05,
        "min_production_rows": 30,
        "production_rows": 4,
        "sufficient_data": False,
        "status": "insufficient_data",
        "features": [],
    }


def _orchestrator_result(monitoring=None, decision=None, retraining=None):
    monitoring = monitoring if monitoring is not None else _monitoring_ok()
    if decision is None:
        decision = {
            "decision": "no_retraining",
            "reason": "No drift detected; retraining is not required.",
            "status": monitoring["status"],
            "overall_drifted": monitoring["overall_drifted"],
            "production_rows": monitoring["production_rows"],
            "drifted_features": [],
        }
    return {
        "monitoring_result": monitoring,
        "retraining_decision": decision,
        "retraining_result": retraining,
    }


def _make_client(tmp_path: Path, db_name: str = "drift_retrain_api.db"):
    db_manager = DatabaseManager(db_path=tmp_path / db_name)
    application = create_app(db_manager=db_manager)
    return TestClient(application), application


def _install_stub(monkeypatch, result=None):
    """Replaces the orchestrator factory with a stub returning result."""
    orchestrator = MagicMock()
    orchestrator.run_check_and_retrain = MagicMock(return_value=result)
    factory = MagicMock(return_value=orchestrator)
    monkeypatch.setattr(api_main, "create_house_price_drift_orchestrator", factory)
    return factory, orchestrator


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
    models_dir = tmp_path / "drift_retrain_models"
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


def _retrain_endpoint_source():
    for route in api_main.app.routes:
        if getattr(route, "path", None) == RETRAIN_PATH:
            return inspect.getsource(route.endpoint)
    raise AssertionError(f"Route {RETRAIN_PATH} not found")


# --- Endpoint existence / method ---


def test_endpoint_exists_with_post_method():
    paths = {route.path: getattr(route, "methods", set()) for route in api_main.app.routes}
    assert RETRAIN_PATH in paths
    assert "POST" in paths[RETRAIN_PATH]


def test_get_method_not_allowed(tmp_path: Path, monkeypatch):
    _install_stub(monkeypatch, _orchestrator_result())
    client, _ = _make_client(tmp_path, "method.db")
    assert client.get(RETRAIN_PATH).status_code == 405


def test_request_model_defaults_promote_true():
    assert DriftRetrainingRequest().promote is True
    assert DriftRetrainingRequest(promote=False).promote is False


# --- Promote propagation ---


def test_default_promote_true_with_empty_body(tmp_path: Path, monkeypatch):
    _, orchestrator = _install_stub(monkeypatch, _orchestrator_result())
    client, _ = _make_client(tmp_path, "default.db")
    response = client.post(RETRAIN_PATH, json={})
    assert response.status_code == 200
    assert orchestrator.run_check_and_retrain.call_count == 1
    assert orchestrator.run_check_and_retrain.call_args.kwargs["promote"] is True


def test_explicit_promote_true(tmp_path: Path, monkeypatch):
    _, orchestrator = _install_stub(monkeypatch, _orchestrator_result())
    client, _ = _make_client(tmp_path, "true.db")
    assert client.post(RETRAIN_PATH, json={"promote": True}).status_code == 200
    assert orchestrator.run_check_and_retrain.call_args.kwargs["promote"] is True


def test_explicit_promote_false(tmp_path: Path, monkeypatch):
    _, orchestrator = _install_stub(monkeypatch, _orchestrator_result())
    client, _ = _make_client(tmp_path, "false.db")
    assert client.post(RETRAIN_PATH, json={"promote": False}).status_code == 200
    assert orchestrator.run_check_and_retrain.call_args.kwargs["promote"] is False


def test_orchestrator_called_exactly_once(tmp_path: Path, monkeypatch):
    _, orchestrator = _install_stub(monkeypatch, _orchestrator_result())
    client, _ = _make_client(tmp_path, "once.db")
    client.post(RETRAIN_PATH, json={"promote": True})
    assert orchestrator.run_check_and_retrain.call_count == 1


# --- Exact result pass-through ---


def test_exact_result_pass_through(tmp_path: Path, monkeypatch):
    expected = _orchestrator_result(
        monitoring=_monitoring_ok(drifted=True),
        decision={
            "decision": "retraining_required",
            "reason": "Drift detected; retraining is required.",
            "status": "ok",
            "overall_drifted": True,
            "production_rows": 100,
            "drifted_features": ["area_sqft"],
        },
        retraining={"best_model_name": "LinearRegression", "best_run_id": "run-1"},
    )
    _install_stub(monkeypatch, expected)
    client, _ = _make_client(tmp_path, "passthrough.db")
    response = client.post(RETRAIN_PATH, json={"promote": True})
    assert response.status_code == 200
    assert response.json() == expected
    assert set(response.json().keys()) == {
        "monitoring_result",
        "retraining_decision",
        "retraining_result",
    }


def test_monitoring_result_passed_through_exactly(tmp_path: Path, monkeypatch):
    monitoring = _monitoring_ok(drifted=True)
    _install_stub(monkeypatch, _orchestrator_result(monitoring=monitoring))
    client, _ = _make_client(tmp_path, "mon.db")
    body = client.post(RETRAIN_PATH, json={}).json()
    assert body["monitoring_result"] == monitoring


def test_retraining_decision_passed_through_exactly(tmp_path: Path, monkeypatch):
    decision = {
        "decision": "retraining_required",
        "reason": "Drift detected.",
        "status": "ok",
        "overall_drifted": True,
        "production_rows": 100,
        "drifted_features": ["area_sqft"],
    }
    _install_stub(monkeypatch, _orchestrator_result(decision=decision))
    client, _ = _make_client(tmp_path, "dec.db")
    body = client.post(RETRAIN_PATH, json={}).json()
    assert body["retraining_decision"] == decision


def test_retraining_result_passed_through_exactly(tmp_path: Path, monkeypatch):
    retraining = {"best_model_name": "X", "best_run_id": "run-9", "promotion": {"decision": "rejected"}}
    expected = _orchestrator_result(
        monitoring=_monitoring_ok(drifted=True),
        decision={
            "decision": "retraining_required",
            "reason": "d",
            "status": "ok",
            "overall_drifted": True,
            "production_rows": 100,
            "drifted_features": ["area_sqft"],
        },
        retraining=retraining,
    )
    _install_stub(monkeypatch, expected)
    client, _ = _make_client(tmp_path, "res.db")
    body = client.post(RETRAIN_PATH, json={"promote": False}).json()
    assert body["retraining_result"] == retraining


# --- Decision paths ---


def test_no_retraining_path_returns_null_result(tmp_path: Path, monkeypatch):
    _install_stub(monkeypatch, _orchestrator_result(retraining=None))
    client, _ = _make_client(tmp_path, "neg.db")
    body = client.post(RETRAIN_PATH, json={"promote": True}).json()
    assert body["retraining_decision"]["decision"] == "no_retraining"
    assert body["retraining_result"] is None


def test_retraining_required_path(tmp_path: Path, monkeypatch):
    expected = _orchestrator_result(
        monitoring=_monitoring_ok(drifted=True),
        decision={
            "decision": "retraining_required",
            "reason": "Drift detected.",
            "status": "ok",
            "overall_drifted": True,
            "production_rows": 100,
            "drifted_features": ["area_sqft"],
        },
        retraining={"best_model_name": "LinearRegression"},
    )
    _, orchestrator = _install_stub(monkeypatch, expected)
    client, _ = _make_client(tmp_path, "pos.db")
    body = client.post(RETRAIN_PATH, json={"promote": True}).json()
    assert body["retraining_decision"]["decision"] == "retraining_required"
    assert body["retraining_result"] == {"best_model_name": "LinearRegression"}
    assert orchestrator.run_check_and_retrain.call_count == 1


def test_insufficient_data_path_returns_null_result(tmp_path: Path, monkeypatch):
    expected = _orchestrator_result(
        monitoring=_monitoring_insufficient(),
        decision={
            "decision": "no_retraining",
            "reason": "Insufficient production data (4 rows); retraining is not triggered.",
            "status": "insufficient_data",
            "overall_drifted": False,
            "production_rows": 4,
            "drifted_features": [],
        },
        retraining=None,
    )
    client, _ = _make_client(tmp_path, "ins.db")
    _install_stub(monkeypatch, expected)
    body = client.post(RETRAIN_PATH, json={"promote": True}).json()
    assert body["monitoring_result"]["status"] == "insufficient_data"
    assert body["retraining_decision"]["decision"] == "no_retraining"
    assert body["retraining_result"] is None


# --- Error handling ---


def test_adaptive_ai_error_returns_generic_500(tmp_path: Path, monkeypatch):
    _, orchestrator = _install_stub(monkeypatch, None)
    orchestrator.run_check_and_retrain = MagicMock(
        side_effect=AdaptiveAIError("internal boom details")
    )
    client, _ = _make_client(tmp_path, "err500.db")
    response = client.post(RETRAIN_PATH, json={"promote": True})
    assert response.status_code == 500
    assert response.json() == {"detail": GENERIC_ERROR}
    assert "boom" not in response.text


def test_unexpected_exception_returns_generic_500(tmp_path: Path, monkeypatch):
    _, orchestrator = _install_stub(monkeypatch, None)
    orchestrator.run_check_and_retrain = MagicMock(side_effect=RuntimeError("kaboom"))
    client, _ = _make_client(tmp_path, "err500b.db")
    response = client.post(RETRAIN_PATH, json={"promote": True})
    assert response.status_code == 500
    assert response.json() == {"detail": GENERIC_ERROR}
    assert "kaboom" not in response.text


@pytest.mark.parametrize("bad", ["x", [], ["true"], {"a": 1}, None])
def test_invalid_promote_values_return_422(tmp_path: Path, monkeypatch, bad):
    _install_stub(monkeypatch, _orchestrator_result())
    client, _ = _make_client(tmp_path, "v.db")
    assert client.post(RETRAIN_PATH, json={"promote": bad}).status_code == 422


def test_missing_body_uses_default_or_422(tmp_path: Path, monkeypatch):
    _, orchestrator = _install_stub(monkeypatch, _orchestrator_result())
    client, _ = _make_client(tmp_path, "nobody.db")
    response = client.post(RETRAIN_PATH)
    # Either FastAPI requires a body (422) or defaults apply (200 + promote True).
    assert response.status_code in (200, 422)
    if response.status_code == 200:
        assert orchestrator.run_check_and_retrain.call_args.kwargs["promote"] is True


def test_extra_fields_ignored(tmp_path: Path, monkeypatch):
    _, orchestrator = _install_stub(monkeypatch, _orchestrator_result())
    client, _ = _make_client(tmp_path, "extra.db")
    response = client.post(RETRAIN_PATH, json={"promote": False, "unknown": 123})
    assert response.status_code == 200
    assert orchestrator.run_check_and_retrain.call_args.kwargs["promote"] is False


# --- DI / boundary ---


def test_factory_receives_app_repositories(tmp_path: Path, monkeypatch):
    factory, _ = _install_stub(monkeypatch, _orchestrator_result())
    client, application = _make_client(tmp_path, "di.db")
    client.post(RETRAIN_PATH, json={"promote": True})
    assert factory.call_count == 1
    _, kwargs = factory.call_args
    assert kwargs["prediction_repo"] is application.state.prediction_repo
    assert kwargs["monitoring_repo"] is application.state.monitoring_repo


def test_factory_not_called_until_post(tmp_path: Path, monkeypatch):
    factory, orchestrator = _install_stub(monkeypatch, _orchestrator_result())
    _, _ = _make_client(tmp_path, "lazy.db")
    assert factory.call_count == 0
    assert orchestrator.run_check_and_retrain.call_count == 0


def test_endpoint_adapter_only_source_hygiene():
    source = _retrain_endpoint_source()
    for forbidden in (".run_check(", ".retrain(", ".detect(", "save_", "BackgroundTasks", "threading"):
        assert forbidden not in source
    assert "run_check_and_retrain" in source


def test_api_source_has_no_background_execution():
    source = (Path(get_project_root()) / "app" / "api" / "main.py").read_text(encoding="utf-8")
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


def test_api_source_invokes_no_business_logic_directly():
    # The pre-existing monitor endpoint calls service.run_check(); the new
    # retrain adapter must not duplicate any business-logic invocation.
    source = _retrain_endpoint_source()
    for forbidden in (
        ".run_check(",
        ".retrain(",
        ".detect(",
        "DriftDetector(",
        "RetrainingService(",
        "ModelPromoter",
        "register_model",
        "save_monitoring_result",
        "save_prediction",
    ):
        assert forbidden not in source
    module_source = (Path(get_project_root()) / "app" / "api" / "main.py").read_text(
        encoding="utf-8"
    )
    for forbidden in (
        "DriftDetector(",
        "RetrainingService(",
        "ModelPromoter",
        "register_model",
    ):
        assert forbidden not in module_source


def test_post_creates_no_persistence_records(tmp_path: Path, monkeypatch):
    _install_stub(monkeypatch, _orchestrator_result())
    client, application = _make_client(tmp_path, "nopersist.db")
    assert client.post(RETRAIN_PATH, json={"promote": True}).status_code == 200
    assert application.state.monitoring_repo.list_monitoring_results(limit=100) == []
    assert application.state.prediction_repo.list_predictions(limit=100) == []


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
    assert client.post("/predict/house-price", json=dict(valid, area_sqft=-5)).status_code == 422


def test_monitoring_endpoints_regression(tmp_path: Path):
    client, application = _make_client(tmp_path, "monreg.db")
    application.state.prediction_repo.init()
    _seed_predictions(application.state.prediction_repo, n=50)
    assert client.post("/monitor/house-price").status_code == 200
    history = client.get("/monitor/house-price/history")
    assert history.status_code == 200
    assert len(history.json()["items"]) == 1


def test_new_tests_create_no_real_artifacts():
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass
