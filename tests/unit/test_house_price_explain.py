"""Phase 2 tests: explainability uses the actual deployed model.

Linear models get an exact coef decomposition; tree models get
baseline-substitution influences. Nothing may be hardcoded: different
inputs must yield different contributions. Uses isolated temporary
model directories only; never touches production artifacts.
"""

import math
from pathlib import Path

import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.explain import (
    EXPLAIN_METHOD_BASELINE_SUBSTITUTION,
    EXPLAIN_METHOD_LINEAR_EXACT,
    explain_house_price_prediction,
)
from app.use_cases.house_price.model_persistence import (
    save_house_price_model,
    write_production_pointer,
)
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.split import split_house_price_data
from app.use_cases.house_price.use_case import HousePriceUseCase


def _row(**overrides):
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


def _deploy(tmp_path: Path, regressor, name: str) -> Path:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    df = generate_house_price_data(n_samples=300)
    X_train, _, y_train, _ = split_house_price_data(df)
    pipe = Pipeline(
        steps=[("preprocessor", get_preprocessor()), ("model", regressor)]
    )
    pipe.fit(X_train, y_train)
    save_house_price_model(pipe, name, models_dir=models_dir)
    write_production_pointer(
        {
            "model_name": name,
            "registered_model_name": "HousePriceModel",
            "model_version": "1",
            "run_id": "run-explain-test",
        },
        models_dir=models_dir,
    )
    return models_dir


def _assert_contribution_shape(result):
    expected_features = set(HousePriceUseCase().get_feature_names())
    assert {c["feature"] for c in result["contributions"]} == expected_features
    for entry in result["contributions"]:
        assert isinstance(entry["contribution"], float)
        assert math.isfinite(entry["contribution"])
        assert entry["direction"] in {"positive", "negative", "neutral"}
        assert isinstance(entry["label"], str) and entry["label"]
    assert isinstance(result["prediction"], float) and result["prediction"] > 0


def test_linear_explanation_is_exact(tmp_path: Path):
    models_dir = _deploy(tmp_path, LinearRegression(), "LinearRegression")
    result = explain_house_price_prediction(_row(), models_dir=models_dir)
    assert result["method"] == EXPLAIN_METHOD_LINEAR_EXACT
    assert result["model_name"] == "LinearRegression"
    assert result["model_version"] == "1"
    _assert_contribution_shape(result)
    total = result["intercept"] + sum(
        c["contribution"] for c in result["contributions"]
    )
    assert total == pytest.approx(result["prediction"], rel=1e-6)


def test_tree_explanation_is_model_derived(tmp_path: Path):
    models_dir = _deploy(
        tmp_path,
        RandomForestRegressor(n_estimators=10, random_state=7),
        "RandomForestRegressor",
    )
    first = explain_house_price_prediction(_row(), models_dir=models_dir)
    assert first["method"] == EXPLAIN_METHOD_BASELINE_SUBSTITUTION
    assert first["model_name"] == "RandomForestRegressor"
    _assert_contribution_shape(first)
    assert isinstance(first["intercept"], float)  # baseline prediction

    # Not hardcoded: a different property yields different influences.
    second = explain_house_price_prediction(
        _row(location="Kothrud", area_sqft=2500.0, bedrooms=4, property_type="Villa"),
        models_dir=models_dir,
    )
    assert second["method"] == EXPLAIN_METHOD_BASELINE_SUBSTITUTION
    vec1 = [c["contribution"] for c in first["contributions"]]
    vec2 = [c["contribution"] for c in second["contributions"]]
    assert vec1 != vec2
    assert first["prediction"] != pytest.approx(second["prediction"])


def test_tree_contributions_match_predict_differences(tmp_path: Path):
    # Substitution contributions are defined as predict(x) - predict(x
    # with feature baselined); verify one feature independently.
    import joblib

    models_dir = _deploy(
        tmp_path,
        RandomForestRegressor(n_estimators=10, random_state=7),
        "RandomForestRegressor",
    )
    pipe = joblib.load(models_dir / "RandomForestRegressor.joblib")
    from app.use_cases.house_price.explain import _reference_baselines

    use_case = HousePriceUseCase()
    features = list(use_case.get_feature_names())
    baselines = _reference_baselines(features, list(use_case.get_numeric_feature_names()))
    row = _row()
    result = explain_house_price_prediction(row, models_dir=models_dir)
    by_feature = {c["feature"]: c["contribution"] for c in result["contributions"]}
    perturbed = dict(row)
    perturbed["bedrooms"] = baselines["bedrooms"]
    expected = float(pipe.predict(pd.DataFrame([row]))[0]) - float(
        pipe.predict(pd.DataFrame([perturbed]))[0]
    )
    assert by_feature["bedrooms"] == pytest.approx(expected)


def test_explanation_honors_regressor_step_name(tmp_path: Path):
    # Pipelines built with a 'regressor' step (as in API tests) must
    # still explain exactly for linear models.
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    df = generate_house_price_data(n_samples=200)
    X_train, _, y_train, _ = split_house_price_data(df)
    pipe = Pipeline(
        steps=[("preprocessor", get_preprocessor()), ("regressor", LinearRegression())]
    )
    pipe.fit(X_train, y_train)
    save_house_price_model(pipe, "LinearRegression", models_dir=models_dir)
    write_production_pointer(
        {
            "model_name": "LinearRegression",
            "registered_model_name": "HousePriceModel",
            "model_version": "1",
            "run_id": "run-regressor-step",
        },
        models_dir=models_dir,
    )
    result = explain_house_price_prediction(_row(), models_dir=models_dir)
    assert result["method"] == EXPLAIN_METHOD_LINEAR_EXACT
    total = result["intercept"] + sum(
        c["contribution"] for c in result["contributions"]
    )
    assert total == pytest.approx(result["prediction"], rel=1e-6)


def test_explain_missing_pointer_raises(tmp_path: Path):
    from app.core.exceptions import ModelNotFoundError

    with pytest.raises(ModelNotFoundError):
        explain_house_price_prediction(_row(), models_dir=tmp_path / "empty")
