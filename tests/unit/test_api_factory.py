"""Factory tests for Step 22: create_app() configuration cleanup.

Verifies factory independence, metadata, module-level compatibility,
and unchanged endpoint behavior. Uses temp models only.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import app.api.main as api_main
from app.api.main import app, create_app
from app.database.database import DatabaseManager
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_persistence import save_house_price_model
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.split import split_house_price_data


def _make_temp_model(tmp_path: Path, monkeypatch):
    df = generate_house_price_data(n_samples=200)
    X_train, _, y_train, _ = split_house_price_data(df)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", get_preprocessor()),
            ("regressor", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)
    models_dir = tmp_path / "factory_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_house_price_model(pipeline, "LinearRegression", models_dir=models_dir)
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODEL_NAME", "LinearRegression")
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODELS_DIR", str(models_dir))
    return models_dir


def test_create_app_returns_fastapi_instance():
    first = create_app()
    second = create_app()
    assert isinstance(first, FastAPI)
    assert isinstance(second, FastAPI)
    assert first is not second


def test_application_metadata_is_correct():
    application = create_app()
    assert application.title == "AdaptiveAI API"
    assert application.version == "1.0.0"
    assert application.description
    assert "House Price" in application.description


def test_module_level_app_still_exists():
    assert isinstance(app, FastAPI)
    assert app.title == "AdaptiveAI API"
    assert app.version == "1.0.0"
    paths = {route.path for route in app.routes}
    assert "/health" in paths
    assert "/predict/house-price" in paths


def test_factory_health_behavior_unchanged(tmp_path: Path):
    isolated_db = DatabaseManager(db_path=tmp_path / "factory_health.db")
    client = TestClient(create_app(db_manager=isolated_db))
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_factory_predict_behavior_unchanged(tmp_path: Path, monkeypatch):
    _make_temp_model(tmp_path, monkeypatch)
    isolated_db = DatabaseManager(db_path=tmp_path / "factory_predict.db")
    client = TestClient(create_app(db_manager=isolated_db))

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

    # Validation behavior preserved.
    bad = dict(valid, area_sqft=-5)
    assert client.post("/predict/house-price", json=bad).status_code == 422
