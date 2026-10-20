"""Step 55 tests: production serving reference (production.json).

Uses temporary model directories only. Never touches the real
models/house_price/ directory.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from app.core.config import get_project_root
from app.core.exceptions import AdaptiveAIError, ModelNotFoundError
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_persistence import (
    PRODUCTION_POINTER_FILENAME,
    get_production_pointer_path,
    load_house_price_model,
    load_production_house_price_model,
    read_production_pointer,
    save_house_price_model,
    write_production_pointer,
)
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.split import split_house_price_data


def _pointer(**overrides):
    base = {
        "model_name": "LinearRegression",
        "registered_model_name": "HousePriceModel",
        "model_version": "1",
        "run_id": "run-abc123",
    }
    base.update(overrides)
    return base


def _trained_pipeline():
    df = generate_house_price_data(n_samples=100)
    X_train, _, y_train, _ = split_house_price_data(df)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", get_preprocessor()),
            ("regressor", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def test_pointer_path_uses_models_dir(tmp_path: Path):
    path = get_production_pointer_path(models_dir=tmp_path)
    assert path == tmp_path / PRODUCTION_POINTER_FILENAME


def test_write_then_read_round_trip(tmp_path: Path):
    target = write_production_pointer(_pointer(), models_dir=tmp_path)
    assert target.is_file()
    pointer = read_production_pointer(models_dir=tmp_path)
    assert pointer["model_name"] == "LinearRegression"
    assert pointer["registered_model_name"] == "HousePriceModel"
    assert pointer["model_version"] == "1"
    assert pointer["run_id"] == "run-abc123"


def test_write_stamps_schema_and_timestamp(tmp_path: Path):
    write_production_pointer(_pointer(), models_dir=tmp_path)
    raw = json.loads((tmp_path / PRODUCTION_POINTER_FILENAME).read_text(encoding="utf-8"))
    assert raw["schema_version"] == 1
    assert isinstance(raw["deployed_at"], str) and raw["deployed_at"]


def test_missing_pointer_raises_model_not_found(tmp_path: Path):
    with pytest.raises(ModelNotFoundError):
        read_production_pointer(models_dir=tmp_path)


def test_corrupt_pointer_raises_without_masquerading_as_missing(tmp_path: Path):
    (tmp_path / PRODUCTION_POINTER_FILENAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(AdaptiveAIError):
        read_production_pointer(models_dir=tmp_path)


def test_invalid_pointer_shape_rejected_on_write(tmp_path: Path):
    with pytest.raises(AdaptiveAIError):
        write_production_pointer({"model_name": "x"}, models_dir=tmp_path)
    assert not (tmp_path / PRODUCTION_POINTER_FILENAME).exists()


def test_invalid_pointer_shape_rejected_on_read(tmp_path: Path):
    (tmp_path / PRODUCTION_POINTER_FILENAME).write_text(
        json.dumps({"model_name": "  ", "registered_model_name": "m", "model_version": "1", "run_id": "r"}),
        encoding="utf-8",
    )
    with pytest.raises(AdaptiveAIError):
        read_production_pointer(models_dir=tmp_path)


def test_failed_write_preserves_previous_pointer(tmp_path: Path):
    write_production_pointer(_pointer(model_version="4"), models_dir=tmp_path)
    with pytest.raises(AdaptiveAIError):
        write_production_pointer({"bogus": True}, models_dir=tmp_path)
    assert read_production_pointer(models_dir=tmp_path)["model_version"] == "4"


def test_load_production_returns_pipeline_and_pointer(tmp_path: Path):
    pipeline = _trained_pipeline()
    save_house_price_model(pipeline, "LinearRegression", models_dir=tmp_path)
    write_production_pointer(_pointer(), models_dir=tmp_path)

    loaded, pointer = load_production_house_price_model(models_dir=tmp_path)
    assert pointer["model_name"] == "LinearRegression"
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
    expected = pipeline.predict(sample)
    actual = loaded.predict(sample)
    assert float(actual[0]) == float(expected[0])


def test_load_production_missing_artifact_raises(tmp_path: Path):
    write_production_pointer(_pointer(model_name="Ghost"), models_dir=tmp_path)
    with pytest.raises(ModelNotFoundError):
        load_production_house_price_model(models_dir=tmp_path)


def test_load_production_without_pointer_raises(tmp_path: Path):
    save_house_price_model(_trained_pipeline(), "LinearRegression", models_dir=tmp_path)
    with pytest.raises(ModelNotFoundError):
        load_production_house_price_model(models_dir=tmp_path)


def test_real_models_dir_untouched():
    assert load_house_price_model.__module__ == "app.use_cases.house_price.model_persistence"
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass
