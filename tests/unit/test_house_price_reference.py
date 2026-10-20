import os
import pytest
import pandas as pd
from unittest.mock import patch
from app.core.exceptions import DataValidationError
from app.use_cases.house_price.market_config import DEFAULT_N_SAMPLES, TARGET_COLUMN
from scripts.create_house_price_reference import create_reference_dataset

def test_house_price_reference_creation(tmp_path):
    """
    Verifies that the reference CSV can be successfully created,
    has the default row count, 11 columns, and positive prices.
    """
    test_output_path = os.path.join(tmp_path, "test_house_price_reference.csv")

    create_reference_dataset(output_path=test_output_path)

    assert os.path.exists(test_output_path)

    df = pd.read_csv(test_output_path)
    assert len(df) == DEFAULT_N_SAMPLES

    expected_columns = {"area_sqft", "bedrooms", "bathrooms", "parking", "location",
                        "property_age", "floor_number", "total_floors",
                        "furnishing_status", "property_type", "price"}
    assert set(df.columns) == expected_columns
    assert (df["price"] > 0).all()

@patch('scripts.create_house_price_reference.generate_house_price_data')
def test_house_price_reference_validation_failure(mock_generate, tmp_path):
    """
    Verifies that if the generated data is invalid, a DataValidationError is raised
    and the file is NOT saved.
    """
    # Create an invalid dataframe (missing the target column)
    invalid_df = pd.DataFrame({
        "area_sqft": [1000],
        "bedrooms": [2],
        "bathrooms": [1],
        "parking": [1],
        "location": ["Wakad"],
        "property_age": [10],
        "floor_number": [2],
        "total_floors": [10],
        "furnishing_status": ["Semi-Furnished"],
        "property_type": ["Apartment"]
        # Missing price column
    })
    mock_generate.return_value = invalid_df

    test_output_path = os.path.join(tmp_path, "invalid_reference.csv")

    # Validation should fail
    with pytest.raises(DataValidationError, match=f"Missing required columns: {TARGET_COLUMN}"):
        create_reference_dataset(output_path=test_output_path)

    # Verify file was not saved
    assert not os.path.exists(test_output_path)
