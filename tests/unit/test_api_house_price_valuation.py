"""Phase 2 API tests: enriched v2 prediction contract.

Covers the additive valuation fields, the benchmarks endpoint, extra
schema rejections, fail-closed v1 production, and isolated v2 serving.
All state is temporary (databases, model dirs); the real production
pointer and artifacts are never touched (also guarded by conftest).
"""

import hashlib
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import app.api.main as api_main
from app.api.main import create_app
from app.core.config import get_project_root
from app.database.database import DatabaseManager
from app.database.repository import PredictionRepository
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_persistence import (
    save_house_price_model,
    write_production_pointer,
)
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.split import split_house_price_data


VALID_V2 = {
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


def _deploy_v2(tmp_path: Path, monkeypatch, name="RandomForestRegressor", version="7"):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    df = generate_house_price_data(n_samples=300)
    X_train, _, y_train, _ = split_house_price_data(df)
    regressor = (
        RandomForestRegressor(n_estimators=10, random_state=3)
        if name != "LinearRegression"
        else LinearRegression()
    )
    pipe = Pipeline(
        steps=[("preprocessor", get_preprocessor()), ("model", regressor)]
    )
    pipe.fit(X_train, y_train)
    save_house_price_model(pipe, name, models_dir=models_dir)
    write_production_pointer(
        {
            "model_name": name,
            "registered_model_name": "HousePriceModel",
            "model_version": version,
            "run_id": "run-phase2-test",
        },
        models_dir=models_dir,
    )
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODELS_DIR", str(models_dir))
    return models_dir


def _client(tmp_path: Path, name="phase2.db"):
    return TestClient(create_app(db_manager=DatabaseManager(db_path=tmp_path / name)))


def test_enriched_prediction_contract(tmp_path: Path, monkeypatch):
    _deploy_v2(tmp_path, monkeypatch)
    body = _client(tmp_path).post("/predict/house-price", json=VALID_V2).json()
    assert body["prediction"] > 0
    assert body["predicted_price_per_sqft"] == pytest.approx(
        body["prediction"] / 1200.0
    )
    assert body["locality_benchmark_price"] == pytest.approx(8800 * 1200.0)
    assert body["benchmark_price_per_sqft"] == pytest.approx(8800.0)
    assert body["benchmark_difference"] == pytest.approx(
        body["prediction"] - 8800 * 1200.0
    )
    assert body["benchmark_difference_percent"] == pytest.approx(
        (body["prediction"] - 8800 * 1200.0) / (8800 * 1200.0) * 100.0
    )
    assert body["prediction_lower"] <= body["prediction"] <= body["prediction_upper"]
    assert body["prediction_upper"] > body["prediction_lower"]
    assert body["model_name"] == "RandomForestRegressor"
    assert body["model_version"] == "7"
    assert body["schema_version"] == 2
    assert isinstance(body["prediction_id"], int)
    assert isinstance(body["inferred_at"], str) and body["inferred_at"]
    assert body["location"] == "Wakad"
    assert body["area_sqft"] == pytest.approx(1200.0)


def test_benchmarks_endpoint_lists_all_localities(tmp_path: Path):
    body = _client(tmp_path).get("/predict/house-price/benchmarks").json()
    assert body["schema_version"] == 2
    assert len(body["items"]) == 12
    wakad = next(i for i in body["items"] if i["location"] == "Wakad")
    assert wakad["benchmark_price_per_sqft"] == pytest.approx(8800.0)
    assert wakad["benchmark_source"] and wakad["calibration_note"]


def test_benchmarks_endpoint_single_and_unknown(tmp_path: Path):
    client = _client(tmp_path)
    single = client.get(
        "/predict/house-price/benchmarks", params={"location": "Kothrud"}
    )
    assert single.status_code == 200
    items = single.json()["items"]
    assert len(items) == 1
    assert items[0]["location"] == "Kothrud"
    assert items[0]["benchmark_price_per_sqft"] == pytest.approx(11500.0)
    bad = client.get("/predict/house-price/benchmarks", params={"location": "Atlantis"})
    assert bad.status_code == 422


def test_schema_rejections_extended(tmp_path: Path, monkeypatch):
    _deploy_v2(tmp_path, monkeypatch)
    client = _client(tmp_path, "schema.db")

    def _bad(**overrides):
        payload = dict(VALID_V2)
        payload.update(overrides)
        return client.post("/predict/house-price", json=payload)

    assert _bad(total_floors=0).status_code == 422
    assert _bad(floor_number=-1).status_code == 422
    assert _bad(property_age=-0.5).status_code == 422
    assert _bad(bathrooms=6, bedrooms=3).status_code == 422  # > bedrooms + 2
    assert _bad(furnishing_status="Bare").status_code == 422
    assert _bad(property_type="Castle").status_code == 422
    assert _bad(area_sqft=0).status_code == 422
    assert _bad(location="Atlantis").status_code == 422


def test_explain_endpoint_upgraded_shape(tmp_path: Path, monkeypatch):
    _deploy_v2(tmp_path, monkeypatch)
    body = _client(tmp_path).post("/predict/house-price/explain", json=VALID_V2).json()
    assert body["method"] == "baseline_substitution"
    assert len(body["contributions"]) == 10
    for entry in body["contributions"]:
        assert isinstance(entry["contribution"], (int, float))
        assert entry["direction"] in {"positive", "negative", "neutral"}


def test_v1_production_fails_closed_on_v2_request(tmp_path: Path, monkeypatch):
    # Isolated schema-v1 style pipeline (legacy feature names) behind a
    # pointer: a v2 request must fail closed with a sanitized 503 and
    # persist nothing — never silently transformed or substituted.
    models_dir = tmp_path / "v1models"
    models_dir.mkdir(parents=True, exist_ok=True)
    legacy = pd.DataFrame(
        {
            "area": [1000.0, 1500.0, 2000.0, 2500.0],
            "bedrooms": [2, 3, 3, 4],
            "bathrooms": [1, 2, 2, 3],
            "property_age": [5.0, 10.0, 2.0, 20.0],
            "parking": [1, 1, 2, 2],
            "house_price": [5e6, 7e6, 9e6, 12e6],
        }
    )
    pipe = Pipeline(
        steps=[
            ("preprocessor", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    pipe.fit(legacy[["area", "bedrooms", "bathrooms", "property_age", "parking"]],
             legacy["house_price"])
    save_house_price_model(pipe, "LinearRegression", models_dir=models_dir)
    write_production_pointer(
        {
            "model_name": "LinearRegression",
            "registered_model_name": "HousePriceModel",
            "model_version": "311",
            "run_id": "run-v1-legacy",
        },
        models_dir=models_dir,
    )
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODELS_DIR", str(models_dir))
    db_path = tmp_path / "v1.db"
    client = TestClient(create_app(db_manager=DatabaseManager(db_path=db_path)))
    response = client.post("/predict/house-price", json=VALID_V2)
    assert response.status_code == 503
    assert response.json()["detail"] == "House Price prediction model is unavailable."
    assert "area" not in response.text
    history = client.get("/predict/house-price/history").json()
    assert history["items"] == []


def test_isolated_v2_serving_leaves_pointer_untouched(tmp_path: Path, monkeypatch):
    models_dir = _deploy_v2(tmp_path, monkeypatch)
    pointer_path = models_dir / "production.json"
    before = hashlib.sha256(pointer_path.read_bytes()).hexdigest()
    client = _client(tmp_path, "untouched.db")
    assert client.post("/predict/house-price", json=VALID_V2).status_code == 200
    assert client.post("/predict/house-price/explain", json=VALID_V2).status_code == 200
    assert hashlib.sha256(pointer_path.read_bytes()).hexdigest() == before


def test_v2_prediction_history_persists_and_legacy_readable(
    tmp_path: Path, monkeypatch
):
    _deploy_v2(tmp_path, monkeypatch)
    db_path = tmp_path / "hist.db"
    client = TestClient(create_app(db_manager=DatabaseManager(db_path=db_path)))
    assert client.post("/predict/house-price", json=VALID_V2).status_code == 200
    repo = PredictionRepository(DatabaseManager(db_path=db_path))
    # Legacy row stays readable alongside the new v2 row.
    repo.save_prediction(
        use_case="house_price",
        model_name="LinearRegression",
        input_data={"area": 1000.0},
        prediction=200000.0,
    )
    records = repo.list_predictions(limit=10, offset=0, use_case="house_price")
    assert len(records) == 2
    v2 = next(r for r in records if "location" in r["input_data"])
    assert v2["input_data"]["location"] == "Wakad"
    assert v2["model_version"] == "7"
    assert v2["prediction"] > 0


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
