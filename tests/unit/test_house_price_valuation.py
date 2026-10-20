"""Phase 2 tests: valuation service (residual interval + enrichment).

The prediction range must be derived from actual holdout residuals of
the deployed pipeline — never an arbitrary fixed percentage. Uses
temporary models only; never touches production artifacts.
"""

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.split import split_house_price_data
from app.use_cases.house_price.training import prepare_house_price_data
from app.use_cases.house_price.use_case import HousePriceUseCase
from app.use_cases.house_price.valuation import (
    RESIDUAL_LOWER_QUANTILE,
    RESIDUAL_UPPER_QUANTILE,
    build_valuation,
    compute_residual_interval,
    get_holdout_residuals,
)


def _trained_pipeline(regressor=None, n_samples=400):
    df = generate_house_price_data(n_samples=n_samples)
    X_train, _, y_train, _ = split_house_price_data(df)
    pipe = Pipeline(
        steps=[
            ("preprocessor", get_preprocessor()),
            ("model", regressor if regressor is not None else LinearRegression()),
        ]
    )
    pipe.fit(X_train, y_train)
    return pipe


def _sample_row(**overrides):
    row = {
        "area_sqft": 1200.0,
        "bedrooms": 3,
        "bathrooms": 2,
        "parking": 1,
        "location": "Wakad",
        "property_age": 8.0,
        "floor_number": 2,
        "total_floors": 12,
        "furnishing_status": "Semi-Furnished",
        "property_type": "Apartment",
    }
    row.update(overrides)
    return row


def test_holdout_residuals_match_shared_split():
    pipeline = _trained_pipeline()
    residuals = get_holdout_residuals(pipeline)
    _, X_test, _, y_test = prepare_house_price_data()
    expected = np.asarray(y_test, dtype=float) - np.asarray(
        pipeline.predict(X_test), dtype=float
    )
    assert residuals.shape == expected.shape
    assert np.allclose(residuals, expected)
    assert residuals.size > 0


def test_interval_ordering_and_finiteness():
    pipeline = _trained_pipeline()
    residuals = get_holdout_residuals(pipeline)
    interval = compute_residual_interval(10_000_000.0, residuals)
    assert math.isfinite(interval["prediction_lower"])
    assert math.isfinite(interval["prediction_upper"])
    assert interval["prediction_lower"] <= 10_000_000.0 <= interval["prediction_upper"]
    assert interval["prediction_upper"] > interval["prediction_lower"]


def test_interval_equals_empirical_residual_quantiles():
    # The width must come from the data, not a fixed percentage.
    pipeline = _trained_pipeline()
    residuals = get_holdout_residuals(pipeline)
    point = 10_000_000.0
    interval = compute_residual_interval(point, residuals)
    assert interval["prediction_lower"] == pytest.approx(
        max(0.0, point + float(np.quantile(residuals, RESIDUAL_LOWER_QUANTILE)))
    )
    assert interval["prediction_upper"] == pytest.approx(
        point + float(np.quantile(residuals, RESIDUAL_UPPER_QUANTILE))
    )
    width = interval["prediction_upper"] - interval["prediction_lower"]
    expected_width = float(np.quantile(residuals, RESIDUAL_UPPER_QUANTILE)) - float(
        np.quantile(residuals, RESIDUAL_LOWER_QUANTILE)
    )
    assert width == pytest.approx(expected_width)
    # Same absolute width but different relative width across price
    # levels proves this is not a fixed +/- percentage.
    other = compute_residual_interval(30_000_000.0, residuals)
    other_width = other["prediction_upper"] - other["prediction_lower"]
    assert other_width == pytest.approx(width)
    assert (width / point) != pytest.approx(other_width / 30_000_000.0)


def test_interval_lower_floored_at_zero():
    residuals = np.array([-50.0, -10.0, 0.0, 10.0, 50.0])
    interval = compute_residual_interval(20.0, residuals)
    assert interval["prediction_lower"] >= 0.0


def test_build_valuation_full_contract():
    pipeline = _trained_pipeline(
        RandomForestRegressor(n_estimators=10, random_state=42)
    )
    row = _sample_row()
    point = float(pipeline.predict(pd.DataFrame([row]))[0])
    valuation = build_valuation(
        point, row, pipeline, model_name="RandomForestRegressor", model_version="9"
    )
    assert valuation["prediction"] == pytest.approx(point)
    assert valuation["predicted_price_per_sqft"] == pytest.approx(point / 1200.0)
    assert valuation["locality_benchmark_price"] == pytest.approx(8800 * 1200.0)
    assert valuation["benchmark_price_per_sqft"] == pytest.approx(8800.0)
    assert valuation["benchmark_difference"] == pytest.approx(point - 8800 * 1200.0)
    assert valuation["prediction_lower"] <= point <= valuation["prediction_upper"]
    assert valuation["model_name"] == "RandomForestRegressor"
    assert valuation["model_version"] == "9"
    assert valuation["schema_version"] == HousePriceUseCase().schema_version == 2
    assert valuation["location"] == "Wakad"
    assert valuation["area_sqft"] == pytest.approx(1200.0)
    assert isinstance(valuation["inferred_at"], str) and valuation["inferred_at"]


def test_build_valuation_rejects_bad_point():
    pipeline = _trained_pipeline()
    with pytest.raises(Exception):
        build_valuation(float("nan"), _sample_row(), pipeline)
