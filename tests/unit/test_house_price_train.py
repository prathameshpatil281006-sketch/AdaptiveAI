import pytest
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from app.use_cases.house_price.train import train_house_price_model
from app.use_cases.house_price.data import generate_house_price_data

@pytest.fixture
def temp_dataset_path(tmp_path):
    """Creates a robust temporary dataset to test the training pipeline."""
    test_path = os.path.join(tmp_path, "test_training_data.csv")
    df = generate_house_price_data(n_samples=500)
    df.to_csv(test_path, index=False)
    return test_path

def test_house_price_training_completes(temp_dataset_path):
    model, metrics = train_house_price_model(data_path=temp_dataset_path)
    
    # Verify model is a sklearn Pipeline
    assert isinstance(model, Pipeline)
    
    # Verify it can generate predictions and handles NaN check
    df = pd.read_csv(temp_dataset_path)
    predictions = model.predict(df)
    assert not pd.Series(predictions).isnull().any()
    
    # Verify evaluation metrics
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2" in metrics
    
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    # Our synthetic generator has strong linear components (area, bedrooms, etc)
    # so R2 should be well within standard positive ranges.
    assert -1.0 <= metrics["r2"] <= 1.0

def test_house_price_training_reproducibility(temp_dataset_path):
    model1, metrics1 = train_house_price_model(data_path=temp_dataset_path)
    model2, metrics2 = train_house_price_model(data_path=temp_dataset_path)
    
    # Ensure exact same results when seeded properly
    assert metrics1["mae"] == metrics2["mae"]
    assert metrics1["rmse"] == metrics2["rmse"]
    assert metrics1["r2"] == metrics2["r2"]
