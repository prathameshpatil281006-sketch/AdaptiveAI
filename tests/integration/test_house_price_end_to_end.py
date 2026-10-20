"""Step 49: house-price end-to-end integration validation.

Exercises the real application wiring with isolated temporary
resources (SQLite database, model directory, MLflow tracking
directory):

    health -> prediction -> persistence -> monitoring (insufficient,
    then sufficient/drifted) -> monitoring history ->
    drift-triggered retraining (promote=False) -> scheduler status ->
    promotion/retraining history -> dashboard api_client integration.

Never touches data/adaptive_ai.db, the reference dataset, the repo
mlruns directory, or production models. Repeatable via pytest.
"""

import json
import math
from pathlib import Path
from urllib.parse import urlparse

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import app.api.main as api_main
import app.retraining.retraining_service as retraining_service_module
import app.api.http_client as dashboard_client
from app.api.main import _sanitize_orchestrator_result_for_http, create_app
from app.core.config import get_project_root
from app.database.database import DatabaseManager
from app.database.repository import PredictionRepository
from app.engine.mlflow_tracker import MLflowTracker
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_persistence import save_house_price_model
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.split import split_house_price_data

VALID_INPUT = {
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

# Statistically far from the reference mix: top-band area, max BHK and
# age, premium locality/type/furnishing. Triggers drift with 35 rows via
# the real KS detector on numeric features.
DRIFTED_INPUT = {
    "area_sqft": 2900.0,
    "bedrooms": 5,
    "bathrooms": 6,
    "parking": 3,
    "location": "Kothrud",
    "property_age": 38.0,
    "floor_number": 12,
    "total_floors": 12,
    "furnishing_status": "Fully-Furnished",
    "property_type": "Villa",
}


@pytest.fixture
def isolated_env(tmp_path, monkeypatch):
    """Trains a temp model, isolates DB/models/MLflow, returns handles."""
    df = generate_house_price_data(n_samples=200)
    X_train, _, y_train, _ = split_house_price_data(df)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", get_preprocessor()),
            ("regressor", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_house_price_model(pipeline, "LinearRegression", models_dir=models_dir)
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODEL_NAME", "LinearRegression")
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODELS_DIR", str(models_dir))

    mlruns_dir = tmp_path / "mlruns"
    real_factory = retraining_service_module.create_house_price_retraining_service

    def _isolated_factory(
        engine=None, tracker=None, evaluator=None, promoter=None,
        promotion_repository=None,
    ):
        # Force temporary tracking + promotion targets regardless of what
        # the production orchestrator factory passes, so no registry
        # write or model file can escape the temporary directory.
        from app.retraining.model_promoter import ModelPromoter

        tmp_tracker = MLflowTracker(
            tracking_uri=str(mlruns_dir), experiment_name="house_price_retraining"
        )
        tmp_promoter = ModelPromoter(
            tracker=tmp_tracker, models_dir=str(models_dir)
        )
        return real_factory(
            engine=engine,
            tracker=tmp_tracker,
            evaluator=evaluator,
            promoter=tmp_promoter,
            promotion_repository=promotion_repository,
        )

    monkeypatch.setattr(
        retraining_service_module,
        "create_house_price_retraining_service",
        _isolated_factory,
    )

    db_manager = DatabaseManager(db_path=tmp_path / "e2e.db")
    app = create_app(db_manager=db_manager)
    client = TestClient(app)
    return {
        "tmp": tmp_path,
        "models_dir": models_dir,
        "mlruns_dir": mlruns_dir,
        "db_manager": db_manager,
        "client": client,
    }


def _post_predictions(client, payload, n):
    for _ in range(n):
        response = client.post("/predict/house-price", json=payload)
        assert response.status_code == 200, response.text
    return response


# --- Health ---


def test_health(isolated_env):
    response = isolated_env["client"].get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


# --- Prediction + persistence ---


def test_prediction_flow(isolated_env):
    response = isolated_env["client"].post("/predict/house-price", json=VALID_INPUT)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], (int, float))
    assert not isinstance(body["prediction"], bool)
    assert math.isfinite(float(body["prediction"]))


def test_prediction_persistence(isolated_env):
    client = isolated_env["client"]
    client.post("/predict/house-price", json=VALID_INPUT)
    repo = PredictionRepository(isolated_env["db_manager"])
    records = repo.list_predictions(limit=10, offset=0, use_case="house_price")
    assert len(records) == 1
    record = records[0]
    assert record["use_case"] == "house_price"
    assert record["model_name"]
    assert isinstance(record["prediction"], (int, float))
    assert math.isfinite(float(record["prediction"]))
    assert isinstance(record["timestamp"], str) and record["timestamp"]
    json.loads(json.dumps(record["input_data"]))


# --- Monitoring ---


def test_monitoring_insufficient_data(isolated_env):
    client = isolated_env["client"]
    client.post("/predict/house-price", json=VALID_INPUT)
    response = client.post("/monitor/house-price")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "insufficient_data"
    assert body["sufficient_data"] is False
    assert body["production_rows"] == 1
    assert body["features"] == []


def test_monitoring_sufficient_and_drift(isolated_env):
    client = isolated_env["client"]
    _post_predictions(client, DRIFTED_INPUT, 35)
    response = client.post("/monitor/house-price")
    assert response.status_code == 200
    body = response.json()
    assert body["sufficient_data"] is True
    assert body["status"] == "ok"
    assert body["production_rows"] == 35
    assert isinstance(body["overall_drifted"], bool)
    assert body["overall_drifted"] is True
    assert len(body["features"]) > 0
    for feature in body["features"]:
        assert {"feature_name", "statistic", "p_value", "drifted"} <= set(feature)
        assert isinstance(feature["drifted"], bool)
        assert math.isfinite(float(feature["statistic"]))
        assert math.isfinite(float(feature["p_value"]))


def test_monitoring_history(isolated_env):
    client = isolated_env["client"]
    client.post("/predict/house-price", json=VALID_INPUT)
    client.post("/monitor/house-price")
    _post_predictions(client, DRIFTED_INPUT, 35)
    client.post("/monitor/house-price")
    response = client.get("/monitor/house-price/history")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["items"], list)
    assert len(body["items"]) >= 2
    assert body["limit"] == 100
    assert body["offset"] == 0
    # A failed monitoring-style request creates no new history row.
    before = len(client.get("/monitor/house-price/history").json()["items"])
    bad = client.get("/monitor/house-price")
    assert bad.status_code == 405
    after = len(client.get("/monitor/house-price/history").json()["items"])
    assert after == before


# --- Drift-triggered retraining ---


def test_retrain_promote_false_end_to_end(isolated_env):
    client = isolated_env["client"]
    _post_predictions(client, DRIFTED_INPUT, 35)
    response = client.post("/monitor/house-price/retrain", json={"promote": False})
    assert response.status_code == 200, response.text
    body = response.json()
    assert {"monitoring_result", "retraining_decision", "retraining_result"} <= set(body)
    result = body["retraining_result"]
    assert isinstance(result, dict)
    assert "best_pipeline" not in result
    assert result["use_case"] == "house_price"
    assert isinstance(result["best_model_name"], str) and result["best_model_name"]
    assert isinstance(result["best_metrics"], dict)
    for metric in ("mae", "rmse", "r2"):
        assert math.isfinite(float(result["best_metrics"][metric]))
    assert isinstance(result["best_run_id"], str) and result["best_run_id"]
    assert isinstance(result["all_results"], list) and len(result["all_results"]) >= 1
    for entry in result["all_results"]:
        assert {"model_name", "metrics", "run_id"} <= set(entry)
    # promote=False claims no promotion.
    assert "promotion" not in result


def test_orchestrator_factory_wires_promotion_safely(isolated_env, monkeypatch):
    """Step 50 regression: the factory wires promoter + audit repository."""
    import app.api.main as api_main_module

    seen = {}

    def _recording_factory(
        engine=None, tracker=None, evaluator=None, promoter=None,
        promotion_repository=None,
    ):
        from unittest.mock import MagicMock

        seen["tracker"] = tracker
        seen["promoter"] = promoter
        seen["promotion_repository"] = promotion_repository
        return MagicMock(name="retraining_service")

    monkeypatch.setattr(
        retraining_service_module,
        "create_house_price_retraining_service",
        _recording_factory,
    )
    orchestrator = api_main_module.create_house_price_drift_orchestrator(
        prediction_repo=__import__(
            "app.database.repository", fromlist=["PredictionRepository"]
        ).PredictionRepository(isolated_env["db_manager"]),
        monitoring_repo=__import__(
            "app.database.monitoring_repository",
            fromlist=["MonitoringRepository"],
        ).MonitoringRepository(isolated_env["db_manager"]),
        promotion_repo=__import__(
            "app.database.promotion_repository",
            fromlist=["PromotionRepository"],
        ).PromotionRepository(isolated_env["db_manager"]),
    )
    assert orchestrator is not None
    assert seen["promotion_repository"] is not None
    assert seen["promoter"] is not None
    assert seen["tracker"] is not None
    # The promoter shares the service tracker backend for registration.
    assert seen["promoter"].tracker is seen["tracker"]


def test_retrain_promote_true_requires_initial_promotion(isolated_env, tmp_path, monkeypatch):
    import app.use_cases.house_price.model_persistence as persistence

    # Hermetic "no production anywhere" setup: the service layer resolves
    # production through the default models directory, so point it at an
    # empty temporary directory. (The live production deployment, when
    # present, must not leak into this scenario; the isolated tmp models
    # dir holds only a legacy artifact and no production reference.)
    empty_default = tmp_path / "empty_default_models"
    empty_default.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        persistence, "_get_default_models_dir", lambda: empty_default
    )
    client = isolated_env["client"]
    models_dir = isolated_env["models_dir"]
    before = sorted(p.name for p in models_dir.iterdir())
    _post_predictions(client, DRIFTED_INPUT, 35)
    response = client.post("/monitor/house-price/retrain", json={"promote": True})
    assert response.status_code == 200, response.text
    body = response.json()
    result = body["retraining_result"]
    assert isinstance(result, dict)
    promotion = result["promotion"]
    # No production artifact exists in the default location, so the
    # existing safety gate refuses automatic initial promotion.
    assert promotion["decision"] == "requires_initial_promotion"
    assert promotion["model_version"] is None
    assert result["promotion_history_id"] == 1
    # Safety: no model file written and no registry version created.
    assert sorted(p.name for p in models_dir.iterdir()) == before
    history = client.get("/monitor/house-price/retrain/history").json()
    assert len(history["items"]) == 1
    assert history["items"][0]["decision"] == "requires_initial_promotion"
    assert history["items"][0]["model_version"] is None


def test_retrain_promote_true_without_drift_claims_no_promotion(
    isolated_env, monkeypatch
):
    # Hermetic production: resolve the default models directory at the
    # isolated models dir (schema-v2 LinearRegression artifact, no
    # production reference). The no-drift path never touches production,
    # and the data-dependent drift branch then exercises the
    # requires-initial-promotion gate instead of the live (schema-v1)
    # production model, which cannot predict on the v2 feature schema.
    import app.use_cases.house_price.model_persistence as persistence

    monkeypatch.setattr(
        persistence, "_get_default_models_dir", lambda: isolated_env["models_dir"]
    )
    client = isolated_env["client"]
    _post_predictions(client, VALID_INPUT, 35)
    monitor = client.post("/monitor/house-price").json()
    response = client.post("/monitor/house-price/retrain", json={"promote": True})
    assert response.status_code == 200
    body = response.json()
    if monitor["overall_drifted"] is False:
        assert body["retraining_result"] is None
        assert "promotion" not in body
    else:  # pragma: no cover - data-dependent branch guard
        assert isinstance(body["retraining_result"], dict)


def test_retrain_promote_true_with_production_model(isolated_env, monkeypatch):
    """Production artifact present: the real evaluator/promoter decides.

    The default model directory is pointed at the isolated models dir
    (which holds the LinearRegression artifact from setup), and an
    explicit production reference designates that artifact, so the
    candidate is compared against a real production model. Either
    outcome is accepted, but its side effects must be consistent:
    promoted implies a saved file, a registry version, and a history
    row; rejected implies none of those.
    """
    import app.use_cases.house_price.model_persistence as persistence
    from app.use_cases.house_price.model_persistence import (
        write_production_pointer,
    )

    monkeypatch.setattr(
        persistence, "_get_default_models_dir", lambda: isolated_env["models_dir"]
    )
    # Explicitly designate the setup artifact as production. Without a
    # reference the service falls back to the best-candidate name, which
    # need not match any saved artifact on the v2 schema.
    write_production_pointer(
        {
            "model_name": "LinearRegression",
            "registered_model_name": "HousePriceModel",
            "model_version": "1",
            "run_id": "run-test-prod-1",
        },
        models_dir=isolated_env["models_dir"],
    )
    client = isolated_env["client"]
    models_dir = isolated_env["models_dir"]
    before_files = sorted(p.name for p in models_dir.iterdir())
    _post_predictions(client, DRIFTED_INPUT, 35)
    response = client.post("/monitor/house-price/retrain", json={"promote": True})
    assert response.status_code == 200, response.text
    promotion = response.json()["retraining_result"]["promotion"]
    assert promotion["decision"] in ("promoted", "rejected")
    history = client.get("/monitor/house-price/retrain/history").json()
    assert history["items"][-1]["decision"] == promotion["decision"]
    if promotion["decision"] == "promoted":
        assert promotion["model_version"]
        assert promotion["registered_model_name"] == "HousePriceModel"
        assert (models_dir / f"{promotion['model_name']}.joblib").exists()
        from mlflow.tracking import MlflowClient

        registry = MlflowClient(
            tracking_uri=(isolated_env["tmp"] / "mlruns").as_uri()
        )
        versions = registry.search_model_versions(
            f"name='{promotion['registered_model_name']}'"
        )
        assert any(
            str(v.version) == str(promotion["model_version"]) for v in versions
        )
    else:
        assert sorted(p.name for p in models_dir.iterdir()) == before_files


def test_retrain_no_retraining_path_without_drift(isolated_env):
    client = isolated_env["client"]
    _post_predictions(client, VALID_INPUT, 35)
    monitor = client.post("/monitor/house-price").json()
    response = client.post("/monitor/house-price/retrain", json={"promote": False})
    assert response.status_code == 200
    body = response.json()
    if monitor["overall_drifted"] is False:
        assert body["retraining_result"] is None
        assert body["retraining_decision"]["decision"] == "no_retraining"
    else:
        assert isinstance(body["retraining_result"], dict)


def test_sanitizer_regression():
    live = object()
    result = {
        "monitoring_result": {"status": "ok"},
        "retraining_decision": {"decision": "retraining_required"},
        "retraining_result": {
            "use_case": "house_price",
            "best_model_name": "Ridge",
            "best_pipeline": live,
            "best_metrics": {"r2": 0.9},
        },
    }
    sanitized = _sanitize_orchestrator_result_for_http(result)
    assert "best_pipeline" not in sanitized["retraining_result"]
    assert sanitized["retraining_result"]["best_model_name"] == "Ridge"
    # Original is not mutated.
    assert result["retraining_result"]["best_pipeline"] is live
    # No-retraining path passes through untouched.
    none_result = {"retraining_result": None}
    assert _sanitize_orchestrator_result_for_http(none_result) == none_result


# --- Promotion history / registry / scheduler ---


def test_promotion_history_shape(isolated_env):
    client = isolated_env["client"]
    _post_predictions(client, DRIFTED_INPUT, 35)
    client.post("/monitor/house-price/retrain", json={"promote": False})
    response = client.get("/monitor/house-price/retrain/history")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["items"], list)
    assert "limit" in body and "offset" in body
    # promote=False performs no promotion phase, so no audit rows are
    # fabricated; the contract shape itself is what is validated here.
    for item in body["items"]:
        assert {"timestamp", "use_case", "decision"} <= set(item)


def test_mlflow_tracking_isolated(isolated_env):
    from mlflow.tracking import MlflowClient

    client = isolated_env["client"]
    _post_predictions(client, DRIFTED_INPUT, 35)
    response = client.post("/monitor/house-price/retrain", json={"promote": False})
    assert response.status_code == 200
    mlflow_client = MlflowClient(
        tracking_uri=(isolated_env["tmp"] / "mlruns").as_uri()
    )
    experiment = mlflow_client.get_experiment_by_name("house_price_retraining")
    assert experiment is not None
    # promote=False performs no registry promotion.
    assert "promotion" not in response.json()["retraining_result"]


def test_scheduler_status(isolated_env):
    response = isolated_env["client"].get("/monitor/house-price/retrain/status")
    assert response.status_code == 200
    body = response.json()
    assert body["enabled"] is False
    assert body["interval_minutes"] == 60
    assert body["running"] is False
    assert body["last_run"] is None


# --- Dashboard api_client integration ---


def _patched_dashboard_clients(client, monkeypatch):
    """Routes dashboard HTTP calls to the TestClient app."""

    class _Wrapped:
        def __init__(self, status_code, payload):
            self.status_code = status_code
            self._payload = payload

        def json(self):
            if isinstance(self._payload, Exception):
                raise self._payload
            return self._payload

    def _fake_post(url, json=None, timeout=None):
        path = urlparse(url).path
        resp = client.post(path, json=json) if json is not None else client.post(path)
        try:
            payload = resp.json()
        except Exception as exc:  # pragma: no cover - defensive
            payload = exc
        return _Wrapped(resp.status_code, payload)

    def _fake_get(url, params=None, timeout=None):
        path = urlparse(url).path
        resp = client.get(path, params=params)
        try:
            payload = resp.json()
        except Exception as exc:  # pragma: no cover - defensive
            payload = exc
        return _Wrapped(resp.status_code, payload)

    monkeypatch.setattr(dashboard_client.requests, "post", _fake_post)
    monkeypatch.setattr(dashboard_client.requests, "get", _fake_get)


def test_dashboard_clients_against_real_api(isolated_env, monkeypatch):
    client = isolated_env["client"]
    _post_predictions(client, DRIFTED_INPUT, 35)
    _patched_dashboard_clients(client, monkeypatch)

    prediction = dashboard_client.predict_house_price_via_api(
        base_url="http://testserver", **VALID_INPUT
    )
    assert math.isfinite(float(prediction))

    monitoring = dashboard_client.run_house_price_drift_check_via_api(
        base_url="http://testserver"
    )
    assert monitoring["sufficient_data"] is True

    history = dashboard_client.get_house_price_monitoring_history_via_api(
        base_url="http://testserver"
    )
    assert len(history["items"]) >= 1

    retraining = dashboard_client.trigger_house_price_retraining_via_api(
        promote=False, base_url="http://testserver"
    )
    assert isinstance(retraining["retraining_result"], dict)

    retraining_history = (
        dashboard_client.get_house_price_retraining_history_via_api(
            base_url="http://testserver"
        )
    )
    assert isinstance(retraining_history["items"], list)

    model_history = dashboard_client.get_house_price_model_history_via_api(
        base_url="http://testserver"
    )
    assert isinstance(model_history["items"], list)

    status = dashboard_client.get_house_price_retraining_status_via_api(
        base_url="http://testserver"
    )
    assert status["enabled"] is False


# --- Isolation ---


def test_no_production_artifacts(isolated_env):
    # Live-state isolation is enforced for every test by tests/conftest.py
    # (the live deployment legitimately exists since Step 57 verification).
    assert (isolated_env["models_dir"] / "LinearRegression.joblib").exists()
    # The isolated database lives under the temporary directory, never at
    # the production path.
    assert isolated_env["db_manager"].path.parent == isolated_env["tmp"]
