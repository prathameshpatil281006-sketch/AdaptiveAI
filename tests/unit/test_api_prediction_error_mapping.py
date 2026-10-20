"""Step 70 (I-20) focused tests: PredictionError maps explicitly to 503.

A loaded model that crashes on valid input is unavailable for serving
(503, prediction-model detail) — not the generic 500 and not a 422
client-input error. Success, validation, missing-model, and
persistence-failure paths are unchanged. Temporary DB/models only.
"""

from pathlib import Path

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


def _setup_client(tmp_path: Path, monkeypatch, db_filename: str = "i20.db"):
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
    db_manager = DatabaseManager(db_path=tmp_path / db_filename)
    db_manager.init_db()
    application = create_app(db_manager=db_manager)
    return TestClient(application), application


def test_prediction_error_maps_to_503_with_prediction_detail(tmp_path: Path, monkeypatch):
    client, application = _setup_client(tmp_path, monkeypatch)

    def _boom(model, input_data):
        raise PredictionError("secret-model-crash")

    monkeypatch.setattr(api_main, "predict_house_price", _boom)
    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    assert response.status_code == 503
    body = response.json()
    assert body["detail"] == "House Price prediction model is unavailable."
    assert "secret-model-crash" not in str(body)
    assert "Traceback" not in str(body)
    # Nothing persisted for a failed prediction.
    assert application.state.prediction_repo.list_predictions() == []


def test_successful_prediction_unchanged(tmp_path: Path, monkeypatch):
    client, application = _setup_client(tmp_path, monkeypatch)
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
    assert isinstance(body["prediction"], float)
    assert len(application.state.prediction_repo.list_predictions()) == 1


def test_validation_error_still_422(tmp_path: Path, monkeypatch):
    client, application = _setup_client(tmp_path, monkeypatch)
    response = client.post("/predict/house-price", json=dict(VALID_PAYLOAD, area_sqft=-1))
    assert response.status_code == 422
    assert application.state.prediction_repo.list_predictions() == []


def test_persistence_failure_still_generic_500(tmp_path: Path, monkeypatch):
    client, application = _setup_client(tmp_path, monkeypatch, "i20_persist.db")

    def _fail_save(*args, **kwargs):
        raise AdaptiveAIError("simulated sqlite failure")

    monkeypatch.setattr(
        application.state.prediction_repo, "save_prediction", _fail_save
    )
    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    assert response.status_code == 500
    assert response.json()["detail"] == "Failed to save prediction."
    assert "prediction" not in response.json()


def test_unexpected_compute_error_still_generic_500(tmp_path: Path, monkeypatch):
    client, _ = _setup_client(tmp_path, monkeypatch, "i20_unexpected.db")

    def _boom(model, input_data):
        raise RuntimeError("secret-unexpected")

    monkeypatch.setattr(api_main, "predict_house_price", _boom)
    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    assert response.status_code == 500
    assert response.json()["detail"] == "House Price prediction failed."
    assert "secret-unexpected" not in str(response.json())
