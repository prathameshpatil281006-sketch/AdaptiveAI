import pytest
import pandas as pd
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.market_config import (
    FEATURE_COLUMNS,
    FURNISHING_VALUES,
    LOCALITY_RATES,
    PROPERTY_TYPE_VALUES,
    TARGET_COLUMN,
)

def test_generate_house_price_data_returns_dataframe():
    df = generate_house_price_data(n_samples=10)
    assert isinstance(df, pd.DataFrame)

def test_generate_house_price_data_requested_rows():
    df = generate_house_price_data(n_samples=105)
    assert len(df) == 105

def test_generate_house_price_data_expected_columns():
    df = generate_house_price_data(n_samples=5)
    assert set(df.columns) == set(FEATURE_COLUMNS + [TARGET_COLUMN])

def test_generate_house_price_data_positive_prices():
    df = generate_house_price_data(n_samples=100)
    assert (df["price"] > 0).all()

def test_generate_house_price_data_reproducible_seed():
    df1 = generate_house_price_data(n_samples=100)
    df2 = generate_house_price_data(n_samples=100)
    # The configured seed should ensure identical datasets
    pd.testing.assert_frame_equal(df1, df2)

def test_generate_house_price_data_reasonable_ranges():
    df = generate_house_price_data(n_samples=500)

    assert (df["area_sqft"] > 0).all()
    assert df["bedrooms"].between(1, 6).all()
    assert df["bathrooms"].between(1, 6).all()
    assert df["parking"].between(0, 3).all()
    assert (df["property_age"] >= 0).all()
    assert (df["floor_number"] >= 0).all()
    assert (df["total_floors"] >= 1).all()
    assert set(df["location"].unique()) <= set(LOCALITY_RATES.keys())
    assert set(df["furnishing_status"].unique()) <= set(FURNISHING_VALUES)
    assert set(df["property_type"].unique()) <= set(PROPERTY_TYPE_VALUES)

def test_generate_house_price_data_floor_relationships():
    df = generate_house_price_data(n_samples=500)
    assert (df["floor_number"] <= df["total_floors"]).all()
    assert (df["bathrooms"] <= df["bedrooms"] + 2).all()

def test_generate_house_price_data_no_nulls_or_duplicates():
    df = generate_house_price_data(n_samples=500)
    assert not df.isnull().any().any()
    assert not df.duplicated().any()

def test_generate_house_price_data_invalid_input():
    with pytest.raises(ValueError, match="positive integer"):
        generate_house_price_data(n_samples=0)

    with pytest.raises(ValueError, match="positive integer"):
        generate_house_price_data(n_samples=-5)
