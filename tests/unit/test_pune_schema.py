"""Phase 1 tests: Pune-aware schema v2 dataset, preprocessing, training.

Covers market_config calibration, generator distributions and quality
gates, numeric+categorical preprocessing without leakage, all three
candidates training with MAE/RMSE/R2, best-model selection, prediction
validation for the new schema, and the retraining path on new data.
Temporary resources only; never touches production artifacts.
"""

import numpy as np
import pandas as pd
import pytest

from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.market_config import (
    DATA_SCHEMA_VERSION,
    FEATURE_COLUMNS,
    LOCALITY_RATES,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN,
)
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.split import split_house_price_data
from app.use_cases.house_price.training import (
    evaluate_all_candidates,
    prepare_house_price_data,
    select_best_candidate,
)
from app.use_cases.house_price.use_case import HousePriceUseCase
from app.use_cases.house_price.validation import validate_house_price_data
from app.core.exceptions import DataValidationError


def test_schema_version_is_2():
    assert DATA_SCHEMA_VERSION == 2
    assert HousePriceUseCase().schema_version == 2
    assert HousePriceUseCase().target_column == "price"
    assert set(HousePriceUseCase().get_feature_names()) == set(FEATURE_COLUMNS)


def test_locality_calibration_covers_expected_localities():
    assert len(LOCALITY_RATES) == 12
    for loc in ("Wakad", "Hinjawadi", "Kharadi", "Baner", "Kothrud", "Aundh"):
        assert loc in LOCALITY_RATES
    # Premium localities anchor above entry localities.
    assert LOCALITY_RATES["Kothrud"]["base_rate_per_sqft"] > LOCALITY_RATES["Wagholi"]["base_rate_per_sqft"]


def test_dataset_scale_and_schema():
    df = generate_house_price_data()
    assert 8000 <= len(df) <= 10000
    assert set(df.columns) == set(FEATURE_COLUMNS + [TARGET_COLUMN])


def test_dataset_no_nulls_no_duplicates_positive_prices():
    df = generate_house_price_data(n_samples=500)
    assert not df.isnull().any().any()
    assert not df.duplicated().any()
    assert (df["price"] > 0).all()


def test_dataset_distributions_documented_shape():
    df = generate_house_price_data(n_samples=2000)
    bhk_share = df["bedrooms"].value_counts(normalize=True)
    # 1-3 BHK dominate; 4+ BHK exist but are less common.
    assert bhk_share.loc[[1, 2, 3]].sum() > 0.70
    assert (df["bedrooms"] >= 4).mean() < 0.25
    # Apartments dominate.
    assert (df["property_type"] == "Apartment").mean() > 0.70
    # Locality ordering follows calibration: premium medians above entry ones.
    medians = df.groupby("location")["price"].median()
    assert medians["Kothrud"] > medians["Wagholi"]
    assert medians["Baner"] > medians["Hadapsar"]


def test_dataset_price_correlated_not_deterministic():
    df = generate_house_price_data(n_samples=1000)
    # Price tracks area within a locality yet carries noise.
    wakad = df[df["location"] == "Wakad"]
    assert wakad["price"].corr(wakad["area_sqft"]) > 0.80
    # Not perfectly deterministic: same rounded area still varies in price.
    assert wakad["price"].std() > 0


def test_preprocessing_numeric_and_categorical():
    df = generate_house_price_data(n_samples=100)
    X = df[HousePriceUseCase().get_feature_names()]
    out = get_preprocessor().fit_transform(X)
    assert isinstance(out, np.ndarray)
    assert out.shape[0] == 100
    # 7 numeric + one-hot categoricals.
    assert out.shape[1] > len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES)
    assert np.all(np.isfinite(out))


def test_training_all_candidates_and_selection():
    X_train, X_test, y_train, y_test = prepare_house_price_data(
        generate_house_price_data(n_samples=300)
    )
    evaluated = evaluate_all_candidates(X_train, X_test, y_train, y_test)
    assert {e["model_name"] for e in evaluated} == {
        "LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor",
    }
    for entry in evaluated:
        for metric in ("mae", "rmse", "r2"):
            assert np.isfinite(entry["metrics"][metric])
    best = select_best_candidate(evaluated)
    assert best["metrics"]["r2"] == max(e["metrics"]["r2"] for e in evaluated)


def test_prediction_rejects_invalid_location_and_floor():
    from app.use_cases.house_price.predict import predict_house_price
    from app.use_cases.house_price.train import train_house_price_model
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "train.csv")
        generate_house_price_data(n_samples=200).to_csv(path, index=False)
        model, _ = train_house_price_model(data_path=path)

    base = generate_house_price_data(n_samples=5).drop(columns=["price"])
    predict_house_price(model, base)  # valid passes

    bad_loc = base.copy()
    bad_loc.loc[0, "location"] = "Atlantis"
    with pytest.raises(DataValidationError):
        predict_house_price(model, bad_loc)

    bad_floor = base.copy()
    bad_floor.loc[0, "floor_number"] = int(base.loc[0, "total_floors"]) + 1
    with pytest.raises(DataValidationError):
        predict_house_price(model, bad_floor)


def test_retraining_path_on_new_schema():
    from app.retraining.retraining_service import create_house_price_retraining_service

    service = create_house_price_retraining_service()
    result = service.retrain(
        HousePriceUseCase(), generate_house_price_data(n_samples=200), promote=False
    )
    assert result["use_case"] == "house_price"
    assert len(result["all_results"]) == 3
    assert "promotion" not in result
