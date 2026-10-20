from pathlib import Path
import pytest
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from app.engine.mlflow_tracker import MLflowTracker
from app.use_cases.house_price.model_selection import select_best_house_price_model
from app.use_cases.house_price.data import generate_house_price_data


@pytest.fixture
def local_mlflow_tracker(tmp_path: Path):
    """Provides an isolated MLflowTracker using a temporary directory."""
    tracking_path = str(tmp_path / "mlruns")
    tracker = MLflowTracker(
        tracking_uri=tracking_path,
        experiment_name="test_experiment"
    )
    return tracker


def test_mlflow_tracker_initialization(local_mlflow_tracker):
    """Verify that MLflowTracker sets tracking URI and creates/sets experiment."""
    tracker = local_mlflow_tracker

    assert tracker.tracking_uri.startswith("file:///")
    assert tracker.experiment_name == "test_experiment"

    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    experiment = client.get_experiment_by_name("test_experiment")
    assert experiment is not None
    assert experiment.name == "test_experiment"


def test_mlflow_tracker_track_candidate_creates_run_with_data(local_mlflow_tracker):
    """Verify that track_candidate logs model info, parameters, metrics, and pipeline."""
    tracker = local_mlflow_tracker

    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("model", LinearRegression())
    ])
    # Fit pipeline before logging
    pipeline.fit([[1, 2], [3, 4]], [10, 20])

    run_id = tracker.track_candidate(
        model_type="LinearRegression",
        hyperparameters={"fit_intercept": True},
        metrics={"mae": 15000.0, "rmse": 20000.0, "r2": 0.88},
        pipeline=pipeline,
        run_name="test_run"
    )

    assert run_id != ""

    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    run = client.get_run(run_id)

    # Check metrics
    assert run.data.metrics["mae"] == 15000.0
    assert run.data.metrics["rmse"] == 20000.0
    assert run.data.metrics["r2"] == 0.88

    # Check params
    assert run.data.params["model_type"] == "LinearRegression"
    assert run.data.params["fit_intercept"] == "True"

    # Check tags
    assert run.data.tags["model_type"] == "LinearRegression"


def test_mlflow_tracker_end_run_safe(local_mlflow_tracker):
    """Verify that ending run when no active run exists does not throw an exception."""
    tracker = local_mlflow_tracker
    tracker.end_run()  # Should complete cleanly without error


def test_house_price_model_selection_creates_three_mlflow_runs(tmp_path: Path):
    """Verify that all 3 House Price candidates create distinct MLflow runs."""
    tracking_path = str(tmp_path / "mlruns_hp")
    tracker = MLflowTracker(
        tracking_uri=tracking_path,
        experiment_name="house_price_test_exp"
    )

    sample_df = generate_house_price_data(n_samples=100)

    result = select_best_house_price_model(
        data_path=sample_df,
        tracker=tracker
    )

    assert result["best_model_name"] in {"LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"}

    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    experiment = client.get_experiment_by_name("house_price_test_exp")
    assert experiment is not None

    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 3

    logged_models = {run.data.tags.get("model_type") for run in runs}
    expected_models = {"LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor"}
    assert logged_models == expected_models
