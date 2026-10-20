import pytest
import pandas as pd
import numpy as np
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.validation import validate_house_price_data
from app.core.exceptions import DataValidationError

@pytest.fixture
def valid_df():
    # Use 50 samples to keep tests fast, which generates robust unique rows
    return generate_house_price_data(n_samples=50)

def test_valid_dataset_passes(valid_df):
    validate_house_price_data(valid_df)

def test_missing_column_fails(valid_df):
    df_invalid = valid_df.drop(columns=["area_sqft"])
    with pytest.raises(DataValidationError, match="Missing required columns: area_sqft"):
        validate_house_price_data(df_invalid)

def test_missing_value_fails(valid_df):
    df_invalid = valid_df.copy()
    df_invalid.loc[0, "bedrooms"] = np.nan
    with pytest.raises(DataValidationError, match="Missing values found in columns: bedrooms"):
        validate_house_price_data(df_invalid)

def test_duplicate_row_fails(valid_df):
    df_invalid = pd.concat([valid_df, valid_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="Duplicate rows found"):
        validate_house_price_data(df_invalid)

def test_invalid_area_fails(valid_df):
    df_invalid = valid_df.copy()
    df_invalid.loc[0, "area_sqft"] = 0
    with pytest.raises(DataValidationError, match="All 'area_sqft' values must be strictly greater than 0."):
        validate_house_price_data(df_invalid)

def test_invalid_bedrooms_fails(valid_df):
    df_invalid = valid_df.copy()
    df_invalid.loc[0, "bedrooms"] = 0
    with pytest.raises(DataValidationError, match="All 'bedrooms' values must be integers greater than or equal to 1."):
        validate_house_price_data(df_invalid)

def test_invalid_bathrooms_fails(valid_df):
    df_invalid = valid_df.copy()
    df_invalid.loc[0, "bathrooms"] = 0
    with pytest.raises(DataValidationError, match="All 'bathrooms' values must be integers greater than or equal to 1."):
        validate_house_price_data(df_invalid)

def test_invalid_property_age_fails(valid_df):
    df_invalid = valid_df.copy()
    df_invalid.loc[0, "property_age"] = -1
    with pytest.raises(DataValidationError, match="All 'property_age' values must be greater than or equal to 0."):
        validate_house_price_data(df_invalid)

def test_invalid_parking_fails(valid_df):
    df_invalid = valid_df.copy()
    df_invalid.loc[0, "parking"] = 5
    with pytest.raises(DataValidationError, match="All 'parking' values must be integers between 0 and 3."):
        validate_house_price_data(df_invalid)

def test_invalid_location_fails(valid_df):
    df_invalid = valid_df.copy()
    df_invalid.loc[0, "location"] = "Atlantis"
    with pytest.raises(DataValidationError, match="Unknown 'location' values"):
        validate_house_price_data(df_invalid)

def test_invalid_floor_relationship_fails(valid_df):
    df_invalid = valid_df.copy()
    df_invalid.loc[0, "floor_number"] = int(df_invalid.loc[0, "total_floors"]) + 1
    with pytest.raises(DataValidationError, match="'floor_number' must not exceed 'total_floors'."):
        validate_house_price_data(df_invalid)

def test_invalid_furnishing_fails(valid_df):
    df_invalid = valid_df.copy()
    df_invalid.loc[0, "furnishing_status"] = "Palatial"
    with pytest.raises(DataValidationError, match="Unknown 'furnishing_status' values"):
        validate_house_price_data(df_invalid)

def test_invalid_property_type_fails(valid_df):
    df_invalid = valid_df.copy()
    df_invalid.loc[0, "property_type"] = "Castle"
    with pytest.raises(DataValidationError, match="Unknown 'property_type' values"):
        validate_house_price_data(df_invalid)

def test_negative_price_fails(valid_df):
    df_invalid = valid_df.copy()
    df_invalid.loc[0, "price"] = 0
    with pytest.raises(DataValidationError, match="All 'price' values must be strictly greater than 0."):
        validate_house_price_data(df_invalid)
