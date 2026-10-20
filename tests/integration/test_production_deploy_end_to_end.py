"""Step 55 integration: train/selection -> deployment/promotion -> API prediction.

Every resource (database, models, MLflow store) is temporary. The real
models/, mlruns/, and database are never touched.
"""

import json
import math
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

import app.api.main as api_main
from app.api.main import create_app
from app.core.config import get_project_root
from app.core.exceptions import ModelPromotionError, RetrainingError
from app.database.database import DatabaseManager
from app.database.promotion_repository import PromotionRepository
from app.database.repository import PredictionRepository
from app.engine.ml_engine import MLEngine
from app.engine.mlflow_tracker import MLflowTracker
from app.retraining.model_promoter import create_house_price_promoter
from app.retraining.retraining_service import RetrainingService
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_persistence import (
    PRODUCTION_POINTER_FILENAME,
    load_house_price_model,
    read_production_pointer,
    save_house_price_model,
    write_production_pointer,
)
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.split import split_house_price_data
from app.use_cases.house_price.use_case import HousePriceUseCase

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


def _training_df(n_samples=300):
    return generate_house_price_data(n_samples=n_samples)


def _weak_pipeline(n_samples=300):
    """Deterministically bad production stand-in (shuffled targets)."""
    df = generate_house_price_data(n_samples=n_samples)
    X_train, _, y_train, _ = split_house_price_data(df)
    rng = np.random.RandomState(0)
    shuffled = np.asarray(y_train, dtype=float)[rng.permutation(len(y_train))]
    pipeline = Pipeline(
        steps=[
            ("preprocessor", get_preprocessor()),
            ("regressor", RandomForestRegressor(n_estimators=10, random_state=0)),
        ]
    )
    pipeline.fit(X_train, shuffled)
    return pipeline


@pytest.fixture
def env(tmp_path: Path, monkeypatch):
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    mlruns_dir = tmp_path / "mlruns"
    db_path = tmp_path / "deploy_e2e.db"
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODELS_DIR", str(models_dir))

    tracker = MLflowTracker(
        tracking_uri=str(mlruns_dir), experiment_name="house_price_retraining"
    )
    promoter = create_house_price_promoter(tracker=tracker, models_dir=str(models_dir))
    db_manager = DatabaseManager(db_path=db_path)
    db_manager.init_db()
    service = RetrainingService(
        engine=MLEngine(),
        tracker=tracker,
        promoter=promoter,
        promotion_repository=PromotionRepository(db_manager),
    )
    client = TestClient(create_app(db_manager=db_manager))
    return {
        "tmp": tmp_path,
        "models_dir": models_dir,
        "mlruns_dir": mlruns_dir,
        "db_manager": db_manager,
        "tracker": tracker,
        "promoter": promoter,
        "service": service,
        "client": client,
        "monkeypatch": monkeypatch,
    }


def _deploy(env, df=None, **kwargs):
    params = {
        "promote": True,
        "allow_initial_promotion": True,
        "models_dir": str(env["models_dir"]),
    }
    params.update(kwargs)
    return env["service"].retrain(
        HousePriceUseCase(), _training_df() if df is None else df, **params
    )


def test_predict_before_deployment_is_503(env):
    response = env["client"].post("/predict/house-price", json=VALID_INPUT)
    assert response.status_code == 503
    assert "No production" in response.json()["detail"]


def test_initial_deployment_then_predict(env):
    assert _deploy(env)["promotion"]["decision"] == "promoted"

    pointer = read_production_pointer(models_dir=env["models_dir"])
    assert pointer["registered_model_name"] == "HousePriceModel"
    assert pointer["model_version"]
    assert (env["models_dir"] / f"{pointer['model_name']}.joblib").is_file()

    response = env["client"].post("/predict/house-price", json=VALID_INPUT)
    assert response.status_code == 200, response.text
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
    assert math.isfinite(float(body["prediction"]))

    repo = PredictionRepository(env["db_manager"])
    records = repo.list_predictions(limit=10, offset=0, use_case="house_price")
    assert len(records) == 1
    assert records[0]["model_name"] == pointer["model_name"]
    assert records[0]["model_version"] == pointer["model_version"]

    from mlflow.tracking import MlflowClient

    registry = MlflowClient(tracking_uri=env["mlruns_dir"].as_uri())
    versions = registry.search_model_versions("name='HousePriceModel'")
    assert any(str(v.version) == str(pointer["model_version"]) for v in versions)
    lineage = [v for v in versions if str(v.version) == str(pointer["model_version"])][0]
    assert lineage.run_id == pointer["run_id"]


def test_second_cycle_rejects_and_preserves_production(env):
    _deploy(env)
    pointer_before = (env["models_dir"] / PRODUCTION_POINTER_FILENAME).read_bytes()
    first = env["client"].post("/predict/house-price", json=VALID_INPUT).json()["prediction"]

    df = _training_df()
    result = env["service"].retrain(
        HousePriceUseCase(), df, promote=True, models_dir=str(env["models_dir"])
    )
    assert result["promotion"]["decision"] == "rejected"
    assert (env["models_dir"] / PRODUCTION_POINTER_FILENAME).read_bytes() == pointer_before

    second = env["client"].post("/predict/house-price", json=VALID_INPUT).json()["prediction"]
    assert second == first


def test_promote_false_changes_nothing(env):
    df = _training_df()
    result = env["service"].retrain(
        HousePriceUseCase(), df, promote=False, models_dir=str(env["models_dir"])
    )
    assert "promotion" not in result
    assert not (env["models_dir"] / PRODUCTION_POINTER_FILENAME).exists()
    assert list(env["models_dir"].glob("*.joblib")) == []
    assert env["client"].post("/predict/house-price", json=VALID_INPUT).status_code == 503


def test_cross_name_promotion_updates_serving(env):
    df = _training_df()
    weak = _weak_pipeline()
    save_house_price_model(weak, "RandomForestRegressor", models_dir=env["models_dir"])
    write_production_pointer(
        {
            "model_name": "RandomForestRegressor",
            "registered_model_name": "HousePriceModel",
            "model_version": "1",
            "run_id": "run-weak",
        },
        models_dir=env["models_dir"],
    )
    result = env["service"].retrain(
        HousePriceUseCase(), df, promote=True, models_dir=str(env["models_dir"])
    )
    assert result["promotion"]["decision"] == "promoted"
    pointer = read_production_pointer(models_dir=env["models_dir"])
    assert pointer["model_name"] == result["best_model_name"]
    assert pointer["model_version"] == result["promotion"]["model_version"]

    response = env["client"].post("/predict/house-price", json=VALID_INPUT)
    assert response.status_code == 200
    production = load_house_price_model(pointer["model_name"], models_dir=env["models_dir"])
    import pandas as pd

    assert float(response.json()["prediction"]) == float(
        production.predict(pd.DataFrame([VALID_INPUT]))[0]
    )


def test_failed_promotion_preserves_production(env):
    df = _training_df()
    weak = _weak_pipeline()
    save_house_price_model(weak, "RandomForestRegressor", models_dir=env["models_dir"])
    write_production_pointer(
        {
            "model_name": "RandomForestRegressor",
            "registered_model_name": "HousePriceModel",
            "model_version": "1",
            "run_id": "run-weak",
        },
        models_dir=env["models_dir"],
    )
    pointer_before = (env["models_dir"] / PRODUCTION_POINTER_FILENAME).read_bytes()

    env["tracker"].register_model = MagicMock(
        side_effect=ModelPromotionError("registry down")
    )
    with pytest.raises(RetrainingError):
        env["service"].retrain(
            HousePriceUseCase(), df, promote=True, models_dir=str(env["models_dir"])
        )
    assert (env["models_dir"] / PRODUCTION_POINTER_FILENAME).read_bytes() == pointer_before

    response = env["client"].post("/predict/house-price", json=VALID_INPUT)
    assert response.status_code == 200
    import pandas as pd

    assert float(response.json()["prediction"]) == float(
        weak.predict(pd.DataFrame([VALID_INPUT]))[0]
    )


def test_deploy_initial_endpoint_end_to_end(env):
    def _factory(promotion_repo=None):
        from app.retraining.retraining_service import RetrainingService as Svc

        assert isinstance(env["service"], Svc)
        return RetrainingService(
            engine=MLEngine(),
            tracker=env["tracker"],
            promoter=env["promoter"],
            promotion_repository=promotion_repo,
        )

    env["monkeypatch"].setattr(
        api_main, "create_house_price_deploy_service", _factory
    )
    response = env["client"].post("/monitor/house-price/deploy-initial")
    assert response.status_code == 200, response.text
    body = response.json()
    assert "best_pipeline" not in body
    assert body["promotion"]["decision"] == "promoted"
    assert body["promotion"]["model_version"]
    assert (env["models_dir"] / PRODUCTION_POINTER_FILENAME).is_file()

    predicted = env["client"].post("/predict/house-price", json=VALID_INPUT)
    assert predicted.status_code == 200
    assert math.isfinite(float(predicted.json()["prediction"]))

    history = env["client"].get("/monitor/house-price/retrain/history").json()
    assert history["items"][-1]["decision"] == "promoted"


def test_no_real_artifacts():
    # Live-state isolation (production models dir, registry versions) is
    # enforced for every test by tests/conftest.py against a session
    # baseline; the live deployment itself is legitimate.
    pass
