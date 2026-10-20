import pytest
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.use_case import HousePriceUseCase

def test_get_preprocessor_returns_valid_transformer():
    preprocessor = get_preprocessor()
    assert isinstance(preprocessor, (TransformerMixin, BaseEstimator))
    assert hasattr(preprocessor, 'fit_transform')

def test_preprocessor_can_be_fitted_and_transforms():
    # Generate some robust synthetic data
    df = generate_house_price_data(n_samples=100)
    features = HousePriceUseCase().get_feature_names()
    
    # Use only the feature columns for transformation
    X = df[features]
    
    preprocessor = get_preprocessor()
    
    # Fit and transform
    X_transformed = preprocessor.fit_transform(X)
    
    # Verify the output is a numpy array
    assert isinstance(X_transformed, np.ndarray)
    
    # Numeric (7) + one-hot categoricals (> 7 columns total).
    assert X_transformed.shape[0] == 100
    assert X_transformed.shape[1] > len(features)

    # Verify scaling centered the numeric block around 0
    # (numeric block comes first in the ColumnTransformer output).
    n_numeric = len(HousePriceUseCase().get_numeric_feature_names())
    for i in range(n_numeric):
        assert abs(X_transformed[:, i].mean()) < 1e-7


def test_preprocessor_handles_unseen_categories():
    # Unseen localities at inference must not crash (handle_unknown=ignore).
    import pandas as pd

    df = generate_house_price_data(n_samples=100)
    features = HousePriceUseCase().get_feature_names()
    preprocessor = get_preprocessor()
    preprocessor.fit(df[features])

    unseen = df[features].iloc[[0]].copy()
    unseen.loc[unseen.index[0], "location"] = " brand-new-locality ".strip()
    transformed = preprocessor.transform(unseen)
    assert transformed.shape[0] == 1
