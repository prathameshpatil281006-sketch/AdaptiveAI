"""Step 24 tests: successful House Price predictions persist to SQLite history.

Uses temporary DatabaseManager instances via create_app(db_manager=...).
Never touches data/adaptive_ai.db.
"""

from datetime import datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import app.api.main as api_main
from app.api.main import create_app
from app.core.exceptions import AdaptiveAIError, PredictionError
from app.database.database import DatabaseManager
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_persistence import save_house_price_model
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


def _setup_model(tmp_path: Path, monkeypatch):
    """Trains a small model and patches API model location to tmp dir."""
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


def _setup_client(tmp_path: Path, monkeypatch, db_filename: str = "history.db"):
    """Builds an isolated app/client with temp model and temp DB."""
    _setup_model(tmp_path, monkeypatch)
    db_manager = DatabaseManager(db_path=tmp_path / db_filename)
    db_manager.init_db()
    application = create_app(db_manager=db_manager)
    return TestClient(application), application


def test_successful_prediction_creates_exactly_one_record(tmp_path: Path, monkeypatch):
    client, application = _setup_client(tmp_path, monkeypatch)
    repo = application.state.prediction_repo
    assert repo.list_predictions() == []

    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    assert response.status_code == 200

    records = repo.list_predictions()
    assert len(records) == 1


def test_stored_fields_match_request_and_response(tmp_path: Path, monkeypatch):
    client, application = _setup_client(tmp_path, monkeypatch)
    repo = application.state.prediction_repo

    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "prediction",
        "predicted_price_per_sqft",
        "locality_benchmark_price",
        "benchmark_price_per_sqft",
        "benchmark_difference",
        "benchmark_difference_percent",
        "prediction_lower",
        "prediction_upper",
        "model_name",
        "model_version",
        "schema_version",
        "prediction_id",
        "inferred_at",
        "location",
        "area_sqft",
    }

    records = repo.list_predictions()
    assert len(records) == 1
    record = records[0]
    assert record["use_case"] == "house_price"
    assert record["model_name"] == "LinearRegression"
    assert record["model_version"] is None
    assert record["input_data"] == VALID_PAYLOAD
    assert record["prediction"] == pytest.approx(body["prediction"])
    # Timestamp is present and timezone-aware UTC ISO-8601.
    assert datetime.fromisoformat(str(record["timestamp"])).tzinfo is not None


def test_failed_model_loading_creates_no_record(tmp_path: Path, monkeypatch):
    empty_dir = tmp_path / "empty_models"
    empty_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODEL_NAME", "LinearRegression")
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODELS_DIR", str(empty_dir))

    db_manager = DatabaseManager(db_path=tmp_path / "history.db")
    db_manager.init_db()
    application = create_app(db_manager=db_manager)
    client = TestClient(application)

    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    assert response.status_code == 503
    assert application.state.prediction_repo.list_predictions() == []


def test_failed_prediction_creates_no_record(tmp_path: Path, monkeypatch):
    client, application = _setup_client(tmp_path, monkeypatch)

    def _boom(model, input_data):
        raise PredictionError("boom")

    monkeypatch.setattr(api_main, "predict_house_price", _boom)

    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    # Step 70 (I-20): PredictionError maps explicitly to 503.
    assert response.status_code == 503
    assert application.state.prediction_repo.list_predictions() == []


def test_validation_failure_creates_no_record(tmp_path: Path, monkeypatch):
    client, application = _setup_client(tmp_path, monkeypatch)
    response = client.post("/predict/house-price", json=dict(VALID_PAYLOAD, area_sqft=-1))
    assert response.status_code == 422
    assert application.state.prediction_repo.list_predictions() == []


def test_persistence_failure_returns_500_without_prediction(tmp_path: Path, monkeypatch):
    client, application = _setup_client(tmp_path, monkeypatch)

    def _fail_save(*args, **kwargs):
        raise AdaptiveAIError("simulated sqlite failure")

    monkeypatch.setattr(
        application.state.prediction_repo, "save_prediction", _fail_save
    )

    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    assert response.status_code == 500
    body = response.json()
    assert "prediction" not in body
    assert "detail" in body
    assert "Traceback" not in str(body)
    assert "SELECT" not in str(body)
    assert "sqlite" not in str(body).lower()


def test_health_and_validation_behavior_intact(tmp_path: Path, monkeypatch):
    client, application = _setup_client(tmp_path, monkeypatch)
    assert client.get("/health").status_code == 200
    assert client.get("/health").json() == {"status": "healthy"}

    bad = dict(VALID_PAYLOAD, parking=5)
    assert client.post("/predict/house-price", json=bad).status_code == 422
    assert application.state.prediction_repo.list_predictions() == []
