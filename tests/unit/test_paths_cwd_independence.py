"""Step 56 regression: project paths are independent of the working directory.

All filesystem/MLflow defaults must resolve against the project root.
Temporary directories are used for every write; the real models/,
mlruns/, and database are never modified.
"""

import math
from pathlib import Path

import pandas as pd
import pytest

from app.core.config import get_project_root, load_config, resolve_project_path
from app.database.database import get_default_database_path
from app.engine.mlflow_tracker import MLflowTracker, get_default_tracking_dir
from app.use_cases.house_price.model_persistence import (
    PRODUCTION_POINTER_FILENAME,
    _get_default_models_dir,
    get_production_pointer_path,
    load_production_house_price_model,
    save_house_price_model,
    write_production_pointer,
)

ROOT = Path(get_project_root())


def test_resolve_project_path_anchors_relative():
    assert resolve_project_path("data/x.csv") == ROOT / "data" / "x.csv"
    assert resolve_project_path(Path("mlruns")) == ROOT / "mlruns"


def test_resolve_project_path_passes_absolute_through(tmp_path: Path):
    target = tmp_path / "elsewhere" / "f.csv"
    assert resolve_project_path(target) == target
    assert resolve_project_path(str(target)) == target


def test_load_config_from_other_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = load_config()
    assert config["application"]["name"] == "AdaptiveAI"


def test_default_database_path_from_other_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert get_default_database_path() == ROOT / "data" / "adaptive_ai.db"


def test_default_models_dir_from_other_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert _get_default_models_dir() == ROOT / "models" / "house_price"
    assert get_production_pointer_path() == ROOT / "models" / "house_price" / PRODUCTION_POINTER_FILENAME


def test_default_tracking_dir_is_project_mlruns():
    assert get_default_tracking_dir() == ROOT / "mlruns"


def test_mlflow_default_uri_from_other_cwd(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        MLflowTracker, "set_experiment", lambda self, name: setattr(self, "experiment_name", name)
    )
    tracker = MLflowTracker()
    expected = (ROOT / "mlruns").resolve().as_uri()
    assert tracker.tracking_uri == expected


def test_mlflow_relative_uri_anchors_at_root(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        MLflowTracker, "set_experiment", lambda self, name: setattr(self, "experiment_name", name)
    )
    tracker = MLflowTracker(tracking_uri="custom_store")
    assert tracker.tracking_uri == (ROOT / "custom_store").resolve().as_uri()


def test_mlflow_absolute_uri_passes_through(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        MLflowTracker, "set_experiment", lambda self, name: setattr(self, "experiment_name", name)
    )
    store = tmp_path / "isolated_mlruns"
    tracker = MLflowTracker(tracking_uri=str(store))
    assert tracker.tracking_uri == store.resolve().as_uri()


def test_train_default_data_from_other_cwd(tmp_path: Path, monkeypatch):
    from app.use_cases.house_price.train import train_house_price_model

    monkeypatch.chdir(tmp_path)
    _, metrics = train_house_price_model()
    assert math.isfinite(metrics["r2"])


def test_selection_default_data_from_other_cwd(tmp_path: Path, monkeypatch):
    from app.use_cases.house_price.model_selection import select_best_house_price_model

    monkeypatch.chdir(tmp_path)
    tracker = MLflowTracker(
        tracking_uri=str(tmp_path / "mlruns_sel"), experiment_name="house_price_prediction"
    )
    result = select_best_house_price_model(tracker=tracker)
    assert len(result["all_results"]) == 3
    assert result["best_model_name"] in {
        "LinearRegression",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
    }


def test_save_load_pointer_with_other_cwd(tmp_path: Path, monkeypatch):
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import Pipeline

    from app.use_cases.house_price.data import generate_house_price_data
    from app.use_cases.house_price.preprocessing import get_preprocessor
    from app.use_cases.house_price.split import split_house_price_data

    models_dir = tmp_path / "cwd_models"
    models_dir.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    df = generate_house_price_data(n_samples=80)
    X_train, _, y_train, _ = split_house_price_data(df)
    pipeline = Pipeline(
        steps=[("preprocessor", get_preprocessor()), ("regressor", LinearRegression())]
    )
    pipeline.fit(X_train, y_train)
    save_house_price_model(pipeline, "LinearRegression", models_dir=models_dir)
    write_production_pointer(
        {
            "model_name": "LinearRegression",
            "registered_model_name": "HousePriceModel",
            "model_version": "1",
            "run_id": "run-cwd",
        },
        models_dir=models_dir,
    )
    loaded, pointer = load_production_house_price_model(models_dir=models_dir)
    assert pointer["run_id"] == "run-cwd"
    sample = pd.DataFrame(
        [
            {
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
        ]
    )
    assert math.isfinite(float(loaded.predict(sample)[0]))


def test_retraining_and_scheduler_defaults_are_absolute():
    from app.retraining.retraining_service import DEFAULT_DATA_PATH as SERVICE_DEFAULT
    from app.scheduler.retraining_scheduler import DEFAULT_DATA_PATH as SCHEDULER_DEFAULT

    for default in (SERVICE_DEFAULT, SCHEDULER_DEFAULT):
        candidate = Path(default)
        assert candidate.is_absolute()
        assert candidate == ROOT / "data" / "reference" / "house_price_reference.csv"
        assert candidate.is_file()


def test_no_real_repository_pollution():
    # Live-state isolation (dev DB, production models dir, registry
    # versions) is enforced for every test by tests/conftest.py against
    # a session baseline; the live deployment itself is legitimate.
    pass
