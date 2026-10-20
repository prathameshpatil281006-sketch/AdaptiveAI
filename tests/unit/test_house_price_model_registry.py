from pathlib import Path

import pytest
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from app.core.exceptions import ModelPromotionError, AdaptiveAIError
from app.engine.mlflow_tracker import MLflowTracker
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_selection import (
    select_best_house_price_model,
)

REGISTERED_MODEL_NAME = MLflowTracker.REGISTERED_MODEL_NAME


@pytest.fixture
def registry_tracker(tmp_path: Path) -> MLflowTracker:
    """Isolated tracker using a temporary local tracking location."""
    tracking_path = str(tmp_path / "mlruns_registry")
    return MLflowTracker(
        tracking_uri=tracking_path,
        experiment_name="registry_test_exp",
    )


def _fitted_pipeline() -> Pipeline:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]
    )
    pipeline.fit([[1, 2], [3, 4], [5, 6]], [10, 20, 30])
    return pipeline


def test_register_model_creates_registered_model_and_version(registry_tracker):
    """Registering a logged model creates HousePriceModel version 1."""
    pipeline = _fitted_pipeline()
    run_id = registry_tracker.track_candidate(
        model_type="LinearRegression",
        hyperparameters={"fit_intercept": True},
        metrics={"mae": 15000.0, "rmse": 20000.0, "r2": 0.88},
        pipeline=pipeline,
        run_name="reg_run_1",
    )
    assert run_id != ""

    result = registry_tracker.register_model(
        run_id=run_id,
        registered_model_name="HousePriceModel",
    )

    assert result["registered_model_name"] == "HousePriceModel"
    assert result["version"] == "1"
    assert result["run_id"] == run_id

    client = MlflowClient(tracking_uri=registry_tracker.tracking_uri)
    registered = client.get_registered_model("HousePriceModel")
    assert registered.name == "HousePriceModel"

    versions = client.search_model_versions("name='HousePriceModel'")
    assert len(versions) == 1
    assert versions[0].run_id == run_id


def test_create_model_version_increments_version(registry_tracker):
    """A second registration creates version 2 under the same registered model."""
    pipeline = _fitted_pipeline()
    run_id_1 = registry_tracker.track_candidate(
        model_type="LinearRegression",
        hyperparameters={"fit_intercept": True},
        metrics={"mae": 15000.0, "rmse": 20000.0, "r2": 0.88},
        pipeline=pipeline,
        run_name="reg_run_1",
    )
    run_id_2 = registry_tracker.track_candidate(
        model_type="LinearRegression",
        hyperparameters={"fit_intercept": True},
        metrics={"mae": 14000.0, "rmse": 19000.0, "r2": 0.89},
        pipeline=pipeline,
        run_name="reg_run_2",
    )

    first = registry_tracker.register_model(run_id=run_id_1)
    second = registry_tracker.register_model(run_id=run_id_2)

    assert first["version"] == "1"
    assert second["version"] == "2"
    assert second["registered_model_name"] == "HousePriceModel"


def test_list_and_get_latest_model_versions(registry_tracker):
    """Listing returns all versions; latest returns the highest version."""
    pipeline = _fitted_pipeline()
    run_ids = [
        registry_tracker.track_candidate(
            model_type="LinearRegression",
            hyperparameters={"fit_intercept": True},
            metrics={"mae": 15000.0, "rmse": 20000.0, "r2": 0.88},
            pipeline=pipeline,
            run_name=f"reg_run_{i}",
        )
        for i in range(2)
    ]
    for run_id in run_ids:
        registry_tracker.register_model(run_id=run_id)

    versions = registry_tracker.list_model_versions("HousePriceModel")
    assert len(versions) == 2
    assert {v.run_id for v in versions} == set(run_ids)

    latest = registry_tracker.get_latest_model_version("HousePriceModel")
    assert latest is not None
    assert str(latest.version) == "2"

    assert registry_tracker.get_latest_model_version("NonExistentModel") is None
    assert registry_tracker.list_model_versions("NonExistentModel") == []


def test_selection_tracks_runs_without_registering(tmp_path: Path):
    """Step 58 policy: a selection cycle tracks 3 runs, registers 0 versions."""
    tracking_path = str(tmp_path / "mlruns_selection_reg")
    tracker = MLflowTracker(
        tracking_uri=tracking_path,
        experiment_name="selection_registry_exp",
    )
    sample_df = generate_house_price_data(n_samples=200)

    result = select_best_house_price_model(data_path=sample_df, tracker=tracker)

    # Existing return structure must be preserved exactly.
    assert set(result.keys()) == {"best_model_name", "best_pipeline", "all_results"}
    assert "registry" not in result
    assert "registered_model" not in result
    assert REGISTERED_MODEL_NAME == "HousePriceModel"

    client = MlflowClient(tracking_uri=tracker.tracking_uri)

    # No registered model and no versions from evaluation alone.
    assert client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'") == []

    # All three candidate runs are still tracked with lineage tags.
    experiment = client.get_experiment_by_name("selection_registry_exp")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 3
    run_id_by_model = {r.data.tags.get("model_type"): r.info.run_id for r in runs}
    assert set(run_id_by_model.keys()) == {
        "LinearRegression",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
    }


def test_registration_error_handling(registry_tracker):
    """Invalid registration inputs raise clean AdaptiveAI errors."""
    from unittest.mock import patch
    from mlflow.exceptions import MlflowException

    with pytest.raises(ModelPromotionError):
        registry_tracker.register_model(run_id="")

    with pytest.raises(ModelPromotionError):
        registry_tracker.register_model(run_id="   ")

    with pytest.raises(ModelPromotionError):
        registry_tracker.register_model(run_id="does-not-exist", registered_model_name="")

    with pytest.raises(ModelPromotionError):
        registry_tracker.create_model_version(
            registered_model_name="HousePriceModel",
            source="",
            run_id="some-run",
        )

    with pytest.raises(ModelPromotionError):
        registry_tracker.list_model_versions("")

    # Backend failures must be wrapped cleanly, not leak raw MLflow errors.
    pipeline = _fitted_pipeline()
    run_id = registry_tracker.track_candidate(
        model_type="LinearRegression",
        hyperparameters={"fit_intercept": True},
        metrics={"mae": 15000.0, "rmse": 20000.0, "r2": 0.88},
        pipeline=pipeline,
        run_name="reg_err_run",
    )
    with patch(
        "app.engine.mlflow_tracker.MlflowClient.create_model_version",
        side_effect=MlflowException("simulated backend failure"),
    ):
        with pytest.raises((ModelPromotionError, AdaptiveAIError)):
            registry_tracker.register_model(
                run_id=run_id,
                registered_model_name="HousePriceModel",
            )
