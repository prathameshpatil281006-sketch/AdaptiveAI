from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from app.use_cases.house_price.use_case import HousePriceUseCase

def get_preprocessor() -> ColumnTransformer:
    """
    Creates and returns an unfitted scikit-learn preprocessing pipeline
    for the Pune-aware House Price use case (schema v2).

    Numeric features (area_sqft, bedrooms, bathrooms, parking,
    property_age, floor_number, total_floors):
    1. Median imputation (robustness fallback).
    2. Standard scaling (zero mean, unit variance).

    Categorical features (location, furnishing_status, property_type):
    1. Most-frequent imputation (robustness fallback).
    2. One-hot encoding with ``handle_unknown="ignore"`` so unseen
       localities/categories at inference never crash prediction.

    Returns:
        ColumnTransformer: A scikit-learn transformer ready to be fitted to the training data.
    """
    use_case = HousePriceUseCase()
    numeric_features = use_case.get_numeric_feature_names()
    categorical_features = use_case.get_categorical_feature_names()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'  # Drop any unspecified columns like the target 'price'
    )

    return preprocessor
