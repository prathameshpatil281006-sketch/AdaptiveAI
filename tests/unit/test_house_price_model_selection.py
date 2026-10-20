from pathlib import Path
from unittest.mock import patch
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from app.core.exceptions import TrainingError
from app.engine.ml_engine import MLEngine
from app.engine.mlflow_tracker import MLflowTracker
from app.use_cases.house_price.model_selection import select_best_house_price_model
from app.use_cases.house_price.data import generate_house_price_data


@pytest.fixture
def tracker(tmp_path: Path) -> MLflowTracker:
    """Isolated MLflow store so selection never touches the real mlruns/."""
    return MLflowTracker(
        tracking_uri=str(tmp_path / "mlruns_selection"),
        experiment_name="house_price_prediction",
    )


def test_select_best_house_price_model_evaluates_exactly_three_models(tracker):
    """Verify that all 3 candidate models are trained and evaluated."""
    result = select_best_house_price_model(tracker=tracker)

    assert "all_results" in result
    all_results = result["all_results"]
    assert len(all_results) == 3

    evaluated_model_names = [res["model_name"] for res in all_results]
    expected_models = {"LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"}
    assert set(evaluated_model_names) == expected_models


def test_select_best_house_price_model_uses_ml_engine(tracker):
    """Verify that select_best_house_price_model delegates execution to MLEngine."""
    with patch.object(MLEngine, "train_and_evaluate", wraps=MLEngine().train_and_evaluate) as spy:
        result = select_best_house_price_model(tracker=tracker)
        # Exactly 3 candidate models should be trained and evaluated via MLEngine
        assert spy.call_count == 3
        assert result["best_model_name"] in {"LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"}


def test_select_best_house_price_model_best_model_returned(tracker):
    """Verify that the best model is identified based on the highest R² score."""
    result = select_best_house_price_model(tracker=tracker)

    assert "best_model_name" in result
    best_model_name = result["best_model_name"]
    assert best_model_name in {"LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"}

    # Confirm it matches the maximum R² in all_results
    best_r2_in_results = max(r["r2"] for r in result["all_results"])
    for res in result["all_results"]:
        if res["model_name"] == best_model_name:
            assert res["r2"] == best_r2_in_results


def test_select_best_house_price_model_best_pipeline_is_sklearn_pipeline(tracker):
    """Verify that the returned best_pipeline is an instantiated scikit-learn Pipeline."""
    result = select_best_house_price_model(tracker=tracker)

    assert "best_pipeline" in result
    best_pipeline = result["best_pipeline"]
    assert isinstance(best_pipeline, Pipeline)
    assert hasattr(best_pipeline, "predict")

    # Verify that the pipeline can generate predictions
    sample_df = generate_house_price_data(n_samples=5).drop(columns=["price"])
    preds = best_pipeline.predict(sample_df)
    assert len(preds) == 5
    assert isinstance(preds, np.ndarray)


def test_select_best_house_price_model_metrics_exist_and_valid(tracker):
    """Verify that MAE, RMSE, and R² exist and have valid numeric values."""
    result = select_best_house_price_model(tracker=tracker)

    for item in result["all_results"]:
        assert "model_name" in item
        assert "mae" in item
        assert "rmse" in item
        assert "r2" in item

        assert isinstance(item["mae"], float)
        assert isinstance(item["rmse"], float)
        assert isinstance(item["r2"], float)

        assert item["mae"] > 0
        assert item["rmse"] > 0
        # In this dataset, R² is expected to be positive and reasonably strong
        assert 0.0 <= item["r2"] <= 1.0


def test_select_best_house_price_model_reproducibility(tracker):
    """Verify that repeated runs yield reproducible metrics."""
    run1 = select_best_house_price_model(tracker=tracker)
    run2 = select_best_house_price_model(tracker=tracker)

    assert run1["best_model_name"] == run2["best_model_name"]
    for res1, res2 in zip(run1["all_results"], run2["all_results"]):
        assert res1["model_name"] == res2["model_name"]
        assert pytest.approx(res1["mae"], abs=1e-5) == res2["mae"]
        assert pytest.approx(res1["rmse"], abs=1e-5) == res2["rmse"]
        assert pytest.approx(res1["r2"], abs=1e-5) == res2["r2"]


def test_select_best_house_price_model_failure_raises_training_error():
    """Verify that TrainingError is raised when an invalid dataset is provided."""
    with pytest.raises(TrainingError):
        select_best_house_price_model("data/reference/non_existent_file.csv")

    # Invalid DataFrame schema should also trigger TrainingError
    invalid_df = pd.DataFrame({"invalid_col": [1, 2, 3]})
    with pytest.raises(TrainingError):
        select_best_house_price_model(invalid_df)
