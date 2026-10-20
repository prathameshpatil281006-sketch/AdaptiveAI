import pytest
import pandas as pd
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.split import split_house_price_data
from app.core.exceptions import DataValidationError

def test_split_house_price_data_returns_expected_objects():
    df = generate_house_price_data(n_samples=100)
    
    X_train, X_test, y_train, y_test = split_house_price_data(df)
    
    assert isinstance(X_train, pd.DataFrame)
    assert isinstance(X_test, pd.DataFrame)
    assert isinstance(y_train, pd.Series)
    assert isinstance(y_test, pd.Series)

def test_split_house_price_data_sizes():
    df = generate_house_price_data(n_samples=100)
    
    X_train, X_test, y_train, y_test = split_house_price_data(df)
    
    # 80% train, 20% test
    assert len(X_train) == 80
    assert len(X_test) == 20
    assert len(y_train) == 80
    assert len(y_test) == 20

def test_split_house_price_data_columns():
    df = generate_house_price_data(n_samples=10)
    
    X_train, X_test, y_train, y_test = split_house_price_data(df)
    
    expected_features = {"area_sqft", "bedrooms", "bathrooms", "parking", "location",
                         "property_age", "floor_number", "total_floors",
                         "furnishing_status", "property_type"}
    assert set(X_train.columns) == expected_features
    assert set(X_test.columns) == expected_features

    assert y_train.name == "price"
    assert y_test.name == "price"

def test_split_house_price_data_reproducibility():
    df = generate_house_price_data(n_samples=100)
    
    X_train_1, X_test_1, y_train_1, y_test_1 = split_house_price_data(df)
    X_train_2, X_test_2, y_train_2, y_test_2 = split_house_price_data(df)
    
    # Since they use the same configured seed, the splits should be identical
    pd.testing.assert_frame_equal(X_train_1, X_train_2)
    pd.testing.assert_frame_equal(X_test_1, X_test_2)
    pd.testing.assert_series_equal(y_train_1, y_train_2)
    pd.testing.assert_series_equal(y_test_1, y_test_2)

def test_split_house_price_data_validation():
    # Intentionally missing the target column to trigger a DataValidationError during validation
    df_invalid = generate_house_price_data(n_samples=10).drop(columns=["price"])
    
    with pytest.raises(DataValidationError):
        split_house_price_data(df_invalid)
