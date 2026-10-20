import pandas as pd
import numpy as np
from app.core.exceptions import DataValidationError
from app.use_cases.house_price.market_config import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    FURNISHING_VALUES,
    LOCALITY_RATES,
    NUMERIC_FEATURES,
    PROPERTY_TYPE_VALUES,
    TARGET_COLUMN,
)


def validate_house_price_data(df: pd.DataFrame) -> None:
    """Validates a Pune-aware House Price dataset (schema v2).

    Checks structure (required columns), nulls, duplicates, dtypes,
    numeric ranges, categorical membership, and cross-feature consistency
    (floor_number <= total_floors, bathrooms reasonable for bedrooms).

    Args:
        df: The DataFrame to validate.

    Raises:
        DataValidationError: If any validation check fails.
    """
    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("Input must be a pandas DataFrame.")

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise DataValidationError(f"Missing required columns: {', '.join(missing_columns)}")

    df_subset = df[required_columns]

    # Missing values
    if df_subset.isnull().any().any():
        missing_counts = df_subset.isnull().sum()
        missing_cols = missing_counts[missing_counts > 0].index.tolist()
        raise DataValidationError(f"Missing values found in columns: {', '.join(missing_cols)}")

    # Duplicate rows
    if df_subset.duplicated().any():
        duplicate_count = int(df_subset.duplicated().sum())
        raise DataValidationError(f"Duplicate rows found in the dataset. {duplicate_count} duplicates detected.")

    # Numeric dtypes for numeric columns
    for col in NUMERIC_FEATURES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DataValidationError(f"Column '{col}' must be numeric, got dtype '{df[col].dtype}'.")
        values = df[col].to_numpy(dtype=float)
        if not bool(np.all(np.isfinite(values))):
            raise DataValidationError(f"Column '{col}' contains NaN or infinite values.")

    if not (df["area_sqft"] > 0).all():
        raise DataValidationError("All 'area_sqft' values must be strictly greater than 0.")

    if not df["bedrooms"].apply(lambda v: float(v).is_integer() and int(v) >= 1).all():
        raise DataValidationError("All 'bedrooms' values must be integers greater than or equal to 1.")

    if not df["bathrooms"].apply(lambda v: float(v).is_integer() and int(v) >= 1).all():
        raise DataValidationError("All 'bathrooms' values must be integers greater than or equal to 1.")

    if not df["parking"].apply(lambda v: float(v).is_integer() and 0 <= int(v) <= 3).all():
        raise DataValidationError("All 'parking' values must be integers between 0 and 3.")

    if not (df["property_age"] >= 0).all():
        raise DataValidationError("All 'property_age' values must be greater than or equal to 0.")

    if not df["floor_number"].apply(lambda v: float(v).is_integer() and int(v) >= 0).all():
        raise DataValidationError("All 'floor_number' values must be integers greater than or equal to 0.")

    if not df["total_floors"].apply(lambda v: float(v).is_integer() and int(v) >= 1).all():
        raise DataValidationError("All 'total_floors' values must be integers greater than or equal to 1.")

    if not (df["floor_number"].to_numpy() <= df["total_floors"].to_numpy()).all():
        raise DataValidationError("'floor_number' must not exceed 'total_floors'.")

    # Bathrooms reasonable relative to bedrooms (at most bedrooms + 2).
    if not (
        df["bathrooms"].to_numpy(dtype=float) <= df["bedrooms"].to_numpy(dtype=float) + 2
    ).all():
        raise DataValidationError("'bathrooms' must be at most 'bedrooms + 2'.")

    # Categorical membership
    unknown_locations = set(df["location"].unique()) - set(LOCALITY_RATES.keys())
    if unknown_locations:
        raise DataValidationError(f"Unknown 'location' values: {sorted(unknown_locations)}.")

    unknown_furnishing = set(df["furnishing_status"].unique()) - set(FURNISHING_VALUES)
    if unknown_furnishing:
        raise DataValidationError(f"Unknown 'furnishing_status' values: {sorted(unknown_furnishing)}.")

    unknown_types = set(df["property_type"].unique()) - set(PROPERTY_TYPE_VALUES)
    if unknown_types:
        raise DataValidationError(f"Unknown 'property_type' values: {sorted(unknown_types)}.")

    if not (df[TARGET_COLUMN] > 0).all():
        raise DataValidationError(f"All '{TARGET_COLUMN}' values must be strictly greater than 0.")
