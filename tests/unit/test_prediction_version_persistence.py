"""Step 71 (I-21) focused tests: history records the served model version.

A pointer-backed production prediction must persist the production
pointer's model_version (not the model name, not None). Legacy
fallback with no production reference persists None per contract.
Temporary DB/models only.
"""

import math
from pathlib import Path

from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import app.api.main as api_main
from app.api.main import create_app
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


def _isolate_models(tmp_path: Path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODELS_DIR", str(models_dir))
    return models_dir


def _make_client(tmp_path: Path, db_name="i21.db"):
    return TestClient(create_app(db_manager=DatabaseManager(db_path=tmp_path / db_name)))


def test_pointer_version_persisted_not_model_name(tmp_path: Path, monkeypatch):
    """Persisted version equals the pointer version, not the model name."""
    models_dir = _isolate_models(tmp_path, monkeypatch)
    save_house_price_model(_fit(LinearRegression()), "LinearRegression", models_dir=models_dir)
    save_house_price_model(
        _fit(RandomForestRegressor(n_estimators=10, random_state=1)),
        "RandomForestRegressor",
        models_dir=models_dir,
    )
    write_production_pointer(
        {
            "model_name": "RandomForestRegressor",
            "registered_model_name": "HousePriceModel",
            "model_version": "9",
            "run_id": "run-rf-9",
        },
        models_dir=models_dir,
    )
    client = _make_client(tmp_path)
    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert math.isfinite(float(response.json()["prediction"]))

    repo = PredictionRepository(DatabaseManager(db_path=tmp_path / "i21.db"))
    records = repo.list_predictions()
    assert len(records) == 1
    assert records[0]["model_name"] == "RandomForestRegressor"
    assert records[0]["model_version"] == "9"
    assert records[0]["model_version"] != records[0]["model_name"]


def test_legacy_fallback_persists_none_version(tmp_path: Path, monkeypatch):
    """No production reference: legacy artifact serves, version is None."""
    models_dir = _isolate_models(tmp_path, monkeypatch)
    save_house_price_model(_fit(LinearRegression()), "LinearRegression", models_dir=models_dir)
    client = _make_client(tmp_path, "i21_legacy.db")
    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    assert response.status_code == 200

    repo = PredictionRepository(DatabaseManager(db_path=tmp_path / "i21_legacy.db"))
    records = repo.list_predictions()
    assert len(records) == 1
    assert records[0]["model_name"] == "LinearRegression"
    assert records[0]["model_version"] is None


def test_version_matches_pointer_across_predictions(tmp_path: Path, monkeypatch):
    """Every successful prediction carries the current pointer version."""
    models_dir = _isolate_models(tmp_path, monkeypatch)
    save_house_price_model(_fit(LinearRegression()), "LinearRegression", models_dir=models_dir)
    write_production_pointer(
        {
            "model_name": "LinearRegression",
            "registered_model_name": "HousePriceModel",
            "model_version": "42",
            "run_id": "run-42",
        },
        models_dir=models_dir,
    )
    client = _make_client(tmp_path, "i21_multi.db")
    for _ in range(2):
        assert client.post("/predict/house-price", json=VALID_PAYLOAD).status_code == 200

    repo = PredictionRepository(DatabaseManager(db_path=tmp_path / "i21_multi.db"))
    records = repo.list_predictions()
    assert len(records) == 2
    assert [r["model_version"] for r in records] == ["42", "42"]
    assert all(isinstance(r["prediction"], float) for r in records)
