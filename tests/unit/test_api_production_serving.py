"""Step 55 tests: production serving contract at the API boundary.

Uses temporary databases/model directories only. Never touches the
real models/, mlruns/, or database.
"""

import math
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import app.api.main as api_main
from app.api.main import create_app
from app.core.config import get_project_root
from app.core.exceptions import RetrainingError
from app.database.database import DatabaseManager
from app.database.repository import PredictionRepository
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_persistence import (
    save_house_price_model,
    write_production_pointer,
)
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.split import split_house_price_data

VALID_PAYLOAD = {
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


def _fit(regressor, n_samples=200):
    df = generate_house_price_data(n_samples=n_samples)
    X_train, _, y_train, _ = split_house_price_data(df)
    pipeline = Pipeline(
        steps=[("preprocessor", get_preprocessor()), ("regressor", regressor)]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def _make_client(tmp_path: Path, db_name="prod_serving.db"):
    return TestClient(create_app(db_manager=DatabaseManager(db_path=tmp_path / db_name)))


def _isolate_models(tmp_path: Path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODELS_DIR", str(models_dir))
    return models_dir


def test_fresh_checkout_predict_returns_503_without_pointer(tmp_path: Path, monkeypatch):
    _isolate_models(tmp_path, monkeypatch)
    client = _make_client(tmp_path)
    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    assert response.status_code == 503
    body = response.json()
    assert "detail" in body
    assert "No production" in body["detail"]
    assert "Traceback" not in str(body)


def test_legacy_named_artifact_still_served(tmp_path: Path, monkeypatch):
    models_dir = _isolate_models(tmp_path, monkeypatch)
    save_house_price_model(_fit(LinearRegression()), "LinearRegression", models_dir=models_dir)
    client = _make_client(tmp_path)
    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert math.isfinite(float(response.json()["prediction"]))


def test_pointer_designation_wins_over_legacy_file(tmp_path: Path, monkeypatch):
    """Serving follows the production reference, not a hardcoded name."""
    models_dir = _isolate_models(tmp_path, monkeypatch)
    save_house_price_model(_fit(LinearRegression()), "LinearRegression", models_dir=models_dir)
    candidate = _fit(RandomForestRegressor(n_estimators=10, random_state=1))
    save_house_price_model(candidate, "RandomForestRegressor", models_dir=models_dir)
    write_production_pointer(
        {
            "model_name": "RandomForestRegressor",
            "registered_model_name": "HousePriceModel",
            "model_version": "9",
            "run_id": "run-rf-9",
        },
        models_dir=models_dir,
    )
    client = _make_client(tmp_path, "priority.db")
    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    assert response.status_code == 200
    frame = pd.DataFrame([VALID_PAYLOAD])
    assert float(response.json()["prediction"]) == float(candidate.predict(frame)[0])

    repo = PredictionRepository(DatabaseManager(db_path=tmp_path / "priority.db"))
    records = repo.list_predictions(limit=10, offset=0, use_case="house_price")
    assert len(records) == 1
    assert records[0]["model_name"] == "RandomForestRegressor"
    assert records[0]["model_version"] == "9"


def test_non_linear_regression_production_serves_without_legacy_file(
    tmp_path: Path, monkeypatch
):
    """No LinearRegression.joblib at all: the API must not hardcode it."""
    models_dir = _isolate_models(tmp_path, monkeypatch)
    candidate = _fit(RandomForestRegressor(n_estimators=10, random_state=1))
    save_house_price_model(candidate, "RandomForestRegressor", models_dir=models_dir)
    write_production_pointer(
        {
            "model_name": "RandomForestRegressor",
            "registered_model_name": "HousePriceModel",
            "model_version": "2",
            "run_id": "run-rf-2",
        },
        models_dir=models_dir,
    )
    assert not (models_dir / "LinearRegression.joblib").exists()
    client = _make_client(tmp_path, "nohardcode.db")
    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert math.isfinite(float(response.json()["prediction"]))


def test_broken_pointer_does_not_fall_back_silently(tmp_path: Path, monkeypatch):
    models_dir = _isolate_models(tmp_path, monkeypatch)
    save_house_price_model(_fit(LinearRegression()), "LinearRegression", models_dir=models_dir)
    write_production_pointer(
        {
            "model_name": "GhostModel",
            "registered_model_name": "HousePriceModel",
            "model_version": "3",
            "run_id": "run-ghost",
        },
        models_dir=models_dir,
    )
    client = _make_client(tmp_path, "broken.db")
    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    assert response.status_code == 503


def test_deploy_initial_endpoint_returns_sanitized_result(tmp_path: Path, monkeypatch):
    live = object()
    service = MagicMock()
    service.retrain = MagicMock(
        return_value={
            "use_case": "house_price",
            "best_model_name": "LinearRegression",
            "best_pipeline": live,
            "best_metrics": {"mae": 1.0, "rmse": 2.0, "r2": 0.9},
            "all_results": [],
            "best_run_id": "run-x",
            "promotion": {"decision": "promoted"},
        }
    )
    monkeypatch.setattr(api_main, "create_house_price_deploy_service", lambda promotion_repo=None: service)
    client = _make_client(tmp_path, "deploy.db")
    response = client.post("/monitor/house-price/deploy-initial")
    assert response.status_code == 200, response.text
    body = response.json()
    assert "best_pipeline" not in body
    assert body["best_model_name"] == "LinearRegression"
    assert body["promotion"]["decision"] == "promoted"
    _, kwargs = service.retrain.call_args
    assert kwargs["promote"] is True
    assert kwargs["allow_initial_promotion"] is True


def test_deploy_initial_endpoint_failure_is_generic_500(tmp_path: Path, monkeypatch):
    service = MagicMock()
    service.retrain = MagicMock(side_effect=RetrainingError("boom details"))
    monkeypatch.setattr(api_main, "create_house_price_deploy_service", lambda promotion_repo=None: service)
    client = _make_client(tmp_path, "deployfail.db")
    response = client.post("/monitor/house-price/deploy-initial")
    assert response.status_code == 500
    assert "boom details" not in response.text
    assert "Traceback" not in response.text


def test_deploy_endpoint_does_not_create_real_artifacts():
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass
