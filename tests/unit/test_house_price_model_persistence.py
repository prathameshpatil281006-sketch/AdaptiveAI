from pathlib import Path
import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from app.core.exceptions import ModelNotFoundError
from app.use_cases.house_price.model_persistence import (
    save_house_price_model,
    load_house_price_model,
)
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.split import split_house_price_data


@pytest.fixture
def trained_pipeline():
    """Provides a fitted pipeline and evaluation data for testing persistence."""
    df = generate_house_price_data(n_samples=100)
    X_train, X_test, y_train, _ = split_house_price_data(df)

    preprocessor = get_preprocessor()
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])
    pipeline.fit(X_train, y_train)
    return pipeline, X_test


def test_model_can_be_saved(tmp_path: Path, trained_pipeline):
    """Verify that a trained pipeline can be saved to a specified directory."""
    pipeline, _ = trained_pipeline
    model_name = "test_model"

    saved_path = save_house_price_model(pipeline, model_name, models_dir=tmp_path)

    assert isinstance(saved_path, Path)
    assert saved_path.name == "test_model.joblib"
    assert saved_path.exists()
    assert saved_path.is_file()


def test_saved_model_can_be_loaded(tmp_path: Path, trained_pipeline):
    """Verify that a saved model can be reloaded and matches expected type."""
    pipeline, _ = trained_pipeline
    model_name = "test_model"

    save_house_price_model(pipeline, model_name, models_dir=tmp_path)
    loaded_pipeline = load_house_price_model(model_name, models_dir=tmp_path)

    assert isinstance(loaded_pipeline, Pipeline)
    assert hasattr(loaded_pipeline, "predict")


def test_loaded_model_produces_identical_predictions(tmp_path: Path, trained_pipeline):
    """Verify that the loaded model generates exactly the same predictions as the original model."""
    pipeline, X_test = trained_pipeline
    model_name = "test_prediction_equality"

    original_predictions = pipeline.predict(X_test)

    save_house_price_model(pipeline, model_name, models_dir=tmp_path)
    loaded_pipeline = load_house_price_model(model_name, models_dir=tmp_path)

    loaded_predictions = loaded_pipeline.predict(X_test)

    np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)


def test_missing_model_raises_model_not_found_error(tmp_path: Path):
    """Verify that attempting to load a non-existent model raises ModelNotFoundError."""
    with pytest.raises(ModelNotFoundError) as exc_info:
        load_house_price_model("non_existent_model", models_dir=tmp_path)

    assert "Model file not found" in str(exc_info.value)
