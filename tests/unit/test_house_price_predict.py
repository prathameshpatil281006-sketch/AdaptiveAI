import pytest
import os
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from app.use_cases.house_price.predict import predict_house_price
from app.use_cases.house_price.train import train_house_price_model
from app.use_cases.house_price.data import generate_house_price_data
from app.core.exceptions import DataValidationError, PredictionError

@pytest.fixture(scope="module")
def trained_model(tmp_path_factory):
    """Provides a fully trained House Price model for prediction tests."""
    temp_dir = tmp_path_factory.mktemp("data")
    test_path = os.path.join(temp_dir, "test_train_for_predict.csv")
    df = generate_house_price_data(n_samples=200)
    df.to_csv(test_path, index=False)
    
    model, _ = train_house_price_model(data_path=test_path)
    return model

@pytest.fixture
def prediction_data():
    """Provides a clean dataset of features without the target column."""
    df = generate_house_price_data(n_samples=20)
    return df.drop(columns=["price"])

def test_prediction_success_and_shape(trained_model, prediction_data):
    predictions = predict_house_price(trained_model, prediction_data)
    
    assert isinstance(predictions, np.ndarray)
    assert len(predictions) == len(prediction_data)
    
def test_predictions_validity(trained_model, prediction_data):
    predictions = predict_house_price(trained_model, prediction_data)
    
    # No NaNs in output
    assert not np.isnan(predictions).any()
    
    # Predictions must be strictly positive given our features
    assert (predictions > 0).all()

def test_invalid_input_raises_validation_error(trained_model, prediction_data):
    # Tamper with the area to make it negative (violating validation rules)
    prediction_data.loc[0, "area_sqft"] = -100

    with pytest.raises(DataValidationError, match="All 'area_sqft' values must be strictly greater than 0."):
        predict_house_price(trained_model, prediction_data)

def test_prediction_failures_raise_prediction_error(prediction_data):
    # Mock a model that crashes upon prediction
    broken_model = MagicMock()
    broken_model.predict.side_effect = ValueError("Simulated model internal crash")
    
    with pytest.raises(PredictionError, match="Simulated model internal crash"):
        predict_house_price(broken_model, prediction_data)
