"""Focused tests for FastAPI House Price service foundation (Step 21).

Uses TestClient (no running server) and temporary model paths.
Does not train/save any production model.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import app.api.main as api_main
from app.api.main import app, create_app
from app.core.exceptions import PredictionError
from app.database.database import DatabaseManager
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_persistence import save_house_price_model
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.split import split_house_price_data


@pytest.fixture
def client(tmp_path: Path):
    isolated_db = DatabaseManager(db_path=tmp_path / "api_test.db")
    return TestClient(create_app(db_manager=isolated_db))


@pytest.fixture
def temp_model_dir(tmp_path: Path, monkeypatch):
    """Trains a small model in-memory and saves it to a temp dir.

    Patches the API to load from the temp dir only.
    """
    df = generate_house_price_data(n_samples=200)
    X_train, _, y_train, _ = split_house_price_data(df)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", get_preprocessor()),
            ("regressor", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)

    models_dir = tmp_path / "api_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_house_price_model(pipeline, "LinearRegression", models_dir=models_dir)

    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODEL_NAME", "LinearRegression")
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODELS_DIR", str(models_dir))
    return models_dir


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


def test_health_works(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_valid_house_price_prediction_works(client, temp_model_dir):
    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], float)
    assert body["prediction"] > 0


def test_request_validation_rejects_invalid_input(client, temp_model_dir):
    # Negative area violates gt=0.
    bad = dict(VALID_PAYLOAD, area_sqft=-100)
    response = client.post("/predict/house-price", json=bad)
    assert response.status_code == 422

    # Zero bedrooms violates ge=1.
    bad = dict(VALID_PAYLOAD, bedrooms=0)
    response = client.post("/predict/house-price", json=bad)
    assert response.status_code == 422

    # Parking must be 0-3.
    bad = dict(VALID_PAYLOAD, parking=5)
    response = client.post("/predict/house-price", json=bad)
    assert response.status_code == 422

    # Unknown location rejected.
    bad = dict(VALID_PAYLOAD, location="Atlantis")
    response = client.post("/predict/house-price", json=bad)
    assert response.status_code == 422

    # floor_number above total_floors rejected.
    bad = dict(VALID_PAYLOAD, floor_number=13)
    response = client.post("/predict/house-price", json=bad)
    assert response.status_code == 422

    # Missing required field.
    bad = dict(VALID_PAYLOAD)
    del bad["area_sqft"]
    response = client.post("/predict/house-price", json=bad)
    assert response.status_code == 422


def test_missing_model_handled_correctly(client, tmp_path: Path, monkeypatch):
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODEL_NAME", "LinearRegression")
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODELS_DIR", str(empty_dir))

    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    assert response.status_code == 503
    body = response.json()
    assert "detail" in body
    # Do not leak internal paths or stack traces.
    assert "Traceback" not in str(body)


def test_prediction_errors_handled_correctly(client, temp_model_dir, monkeypatch):
    def _boom(model, input_data):
        raise PredictionError("Simulated model internal crash")

    monkeypatch.setattr(api_main, "predict_house_price", _boom)

    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    # Step 70 (I-20): PredictionError maps explicitly to 503 (model
    # unavailable for serving), not the generic 500.
    assert response.status_code == 503
    body = response.json()
    assert "detail" in body
    assert "Traceback" not in str(body)
    assert "Simulated model internal crash" not in str(body)


def test_api_does_not_train_models(client, temp_model_dir, monkeypatch):
    """API prediction must not invoke any training logic."""

    def _fail_train(*args, **kwargs):
        raise AssertionError("Training must not be called inside API requests")

    # Patch all known training entry points; prediction must still succeed.
    monkeypatch.setattr(
        "app.use_cases.house_price.train.train_house_price_model", _fail_train
    )
    monkeypatch.setattr(
        "app.use_cases.house_price.model_selection.select_best_house_price_model",
        _fail_train,
    )

    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert "prediction" in response.json()
