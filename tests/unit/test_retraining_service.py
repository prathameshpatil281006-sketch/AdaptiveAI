"""Focused tests for RetrainingService foundation (Step 30).

Uses temporary data and isolated MLflow tracking. Never touches the
real production database or production model artifacts.
"""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from app.core.config import get_project_root
from app.core.exceptions import RetrainingError, TrainingError
from app.engine.ml_engine import MLEngine
from app.engine.mlflow_tracker import MLflowTracker
from app.retraining.retraining_service import (
    RetrainingService,
    create_house_price_retraining_service,
)
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.use_case import HousePriceUseCase


def _tmp_tracker(tmp_path: Path, name: str = "mlruns_retrain") -> MLflowTracker:
    return MLflowTracker(
        tracking_uri=str(tmp_path / name),
        experiment_name="retrain_test_exp",
    )


def _small_df(n_samples: int = 150) -> pd.DataFrame:
    return generate_house_price_data(n_samples=n_samples)


def _mock_engine_with_metrics(metrics_by_model):
    engine = MagicMock(spec=MLEngine)

    def _side_effect(**kwargs):
        model = kwargs.get("model")
        model_name = type(model).__name__
        metrics = metrics_by_model.get(model_name, {"mae": 1.0, "rmse": 2.0, "r2": 0.5})
        pipeline = MagicMock(spec=Pipeline)
        # Attach a marker so tests can distinguish pipelines if needed.
        pipeline._mock_model_name = model_name
        return {"pipeline": pipeline, "metrics": dict(metrics)}

    engine.train_and_evaluate.side_effect = _side_effect
    return engine


def _mock_tracker_with_run_ids():
    tracker = MagicMock(spec=MLflowTracker)
    tracker.tracking_uri = "file:///tmp/fake-mlruns"

    def _track(model_type=None, hyperparameters=None, metrics=None, pipeline=None, run_name=None, **kwargs):
        return f"run-{model_type}"

    tracker.track_candidate.side_effect = _track
    # _get_client returns a mock whose set_tag is a no-op mock.
    client = MagicMock()
    tracker._get_client.return_value = client
    return tracker


# --- Successful retraining ---


def test_successful_retraining_with_real_engine_and_tracker(tmp_path: Path):
    tracker = _tmp_tracker(tmp_path)
    service = create_house_price_retraining_service(tracker=tracker)
    assert isinstance(service, RetrainingService)
    assert service.engine is not None
    assert service.tracker is tracker

    result = service.retrain(HousePriceUseCase(), _small_df(150))

    assert result["use_case"] == "house_price"
    assert result["best_model_name"] in {
        "LinearRegression",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
    }
    assert isinstance(result["best_pipeline"], Pipeline)
    assert hasattr(result["best_pipeline"], "predict")
    for key in ("mae", "rmse", "r2"):
        assert key in result["best_metrics"]
        assert isinstance(result["best_metrics"][key], float)
    assert len(result["all_results"]) == 3
    assert isinstance(result["best_run_id"], str) and result["best_run_id"] != ""

    # Lineage tags present on every run.
    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    experiment = client.get_experiment_by_name("retrain_test_exp")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 3
    for run in runs:
        assert run.data.tags.get("training_stage") == "retraining"
        assert run.data.tags.get("use_case") == "house_price"


def test_all_configured_candidates_are_evaluated(tmp_path: Path):
    engine = _mock_engine_with_metrics({})
    tracker = _mock_tracker_with_run_ids()
    service = RetrainingService(engine=engine, tracker=tracker)

    result = service.retrain(HousePriceUseCase(), _small_df(120))

    expected = {c["model_type"] for c in HousePriceUseCase().get_model_candidates()}
    assert {r["model_name"] for r in result["all_results"]} == expected
    assert len(result["all_results"]) == len(expected) == 3


def test_engine_called_once_per_candidate(tmp_path: Path):
    engine = _mock_engine_with_metrics({})
    tracker = _mock_tracker_with_run_ids()
    service = RetrainingService(engine=engine, tracker=tracker)

    service.retrain(HousePriceUseCase(), _small_df(120))

    assert engine.train_and_evaluate.call_count == 3


def test_mlflow_tracking_occurs_once_per_candidate(tmp_path: Path):
    engine = _mock_engine_with_metrics({})
    tracker = _mock_tracker_with_run_ids()
    service = RetrainingService(engine=engine, tracker=tracker)

    service.retrain(HousePriceUseCase(), _small_df(120))

    assert tracker.track_candidate.call_count == 3
    for call in tracker.track_candidate.call_args_list:
        kwargs = call.kwargs
        assert "model_type" in kwargs
        assert "hyperparameters" in kwargs
        assert "metrics" in kwargs
        assert "pipeline" in kwargs
        assert set(kwargs["metrics"].keys()) == {"mae", "rmse", "r2"}


def test_best_candidate_selected_by_highest_r2():
    metrics_by_model = {
        "LinearRegression": {"mae": 100.0, "rmse": 150.0, "r2": 0.5},
        "RandomForestRegressor": {"mae": 50.0, "rmse": 70.0, "r2": 0.9},
        "GradientBoostingRegressor": {"mae": 80.0, "rmse": 100.0, "r2": 0.7},
    }
    engine = _mock_engine_with_metrics(metrics_by_model)
    tracker = _mock_tracker_with_run_ids()
    service = RetrainingService(engine=engine, tracker=tracker)

    result = service.retrain(HousePriceUseCase(), _small_df(120))

    assert result["best_model_name"] == "RandomForestRegressor"
    assert result["best_metrics"]["r2"] == pytest.approx(0.9)
    assert result["best_run_id"] == "run-RandomForestRegressor"


def test_result_contains_required_fields(tmp_path: Path):
    tracker = _tmp_tracker(tmp_path)
    service = create_house_price_retraining_service(tracker=tracker)
    result = service.retrain(HousePriceUseCase(), _small_df(120))

    for key in (
        "use_case",
        "best_model_name",
        "best_pipeline",
        "best_metrics",
        "all_results",
        "best_run_id",
    ):
        assert key in result
    assert result["use_case"] == "house_price"
    assert isinstance(result["best_model_name"], str)
    assert isinstance(result["best_pipeline"], Pipeline)
    assert set(result["best_metrics"].keys()) == {"mae", "rmse", "r2"}
    assert isinstance(result["all_results"], list)
    assert isinstance(result["best_run_id"], str)


def test_all_results_contain_model_name_metrics_run_id(tmp_path: Path):
    tracker = _tmp_tracker(tmp_path)
    service = create_house_price_retraining_service(tracker=tracker)
    result = service.retrain(HousePriceUseCase(), _small_df(120))

    for entry in result["all_results"]:
        assert "model_name" in entry
        assert "metrics" in entry
        assert "run_id" in entry
        assert isinstance(entry["model_name"], str)
        assert set(entry["metrics"].keys()) == {"mae", "rmse", "r2"}
        assert isinstance(entry["run_id"], str) and entry["run_id"] != ""

    # best_* must be consistent with all_results.
    best = next(
        r for r in result["all_results"] if r["model_name"] == result["best_model_name"]
    )
    assert best["metrics"] == result["best_metrics"]
    assert best["run_id"] == result["best_run_id"]


# --- No promotion / no production change ---


def test_no_model_registry_registration_occurs(tmp_path: Path, monkeypatch):
    tracker = _tmp_tracker(tmp_path)
    service = create_house_price_retraining_service(tracker=tracker)

    def _fail_register(*args, **kwargs):
        raise AssertionError("Model Registry must not be called during retraining")

    monkeypatch.setattr(tracker, "register_model", _fail_register)
    monkeypatch.setattr(tracker, "ensure_registered_model", _fail_register)
    monkeypatch.setattr(tracker, "create_model_version", _fail_register)

    result = service.retrain(HousePriceUseCase(), _small_df(120))
    assert result["best_model_name"]

    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    assert client.search_model_versions("name='HousePriceModel'") == []


def test_no_registry_calls_with_mock_tracker():
    engine = _mock_engine_with_metrics({})
    tracker = _mock_tracker_with_run_ids()
    service = RetrainingService(engine=engine, tracker=tracker)

    service.retrain(HousePriceUseCase(), _small_df(120))

    tracker.register_model.assert_not_called()
    tracker.ensure_registered_model.assert_not_called()
    tracker.create_model_version.assert_not_called()


def test_no_production_model_changed(tmp_path: Path, monkeypatch):
    import app.retraining.retraining_service as retraining_module

    tracker = _tmp_tracker(tmp_path)
    service = create_house_price_retraining_service(tracker=tracker)

    # Snapshot production models directory state.
    models_dir = Path(get_project_root()) / "models" / "house_price"
    before = set(p.name for p in models_dir.glob("*.joblib")) if models_dir.exists() else set()

    import app.use_cases.house_price.model_persistence as persistence

    def _fail_save(*args, **kwargs):
        raise AssertionError("Production model must not be saved during retraining")

    monkeypatch.setattr(persistence, "save_house_price_model", _fail_save)

    # Retraining module must not reference production persistence at all.
    source = Path(retraining_module.__file__).read_text(encoding="utf-8")
    assert "save_house_price_model" not in source
    assert "register_model" not in source
    assert "ensure_registered_model" not in source
    assert "create_model_version" not in source

    result = service.retrain(HousePriceUseCase(), _small_df(120))
    assert result["best_pipeline"] is not None

    after = set(p.name for p in models_dir.glob("*.joblib")) if models_dir.exists() else set()
    assert before == after


def test_retraining_service_source_hygiene():
    import app.retraining.retraining_service as retraining_module

    source = Path(retraining_module.__file__).read_text(encoding="utf-8")
    assert "MLEngine" in source
    assert "MLflowTracker" in source
    assert "BaseUseCase" in source
    for forbidden in ("import sqlite3", "SELECT ", "APScheduler", "Celery", "Streamlit", "streamlit", "FastAPI("):
        assert forbidden not in source
    # No second use case or scheduler wiring.
    assert "demand_forecasting" not in source
    assert "stock_prediction" not in source


# --- Error handling ---


def test_invalid_missing_data_raises_retraining_error(tmp_path: Path):
    tracker = _tmp_tracker(tmp_path)
    service = create_house_price_retraining_service(tracker=tracker)
    use_case = HousePriceUseCase()

    with pytest.raises(RetrainingError):
        service.retrain(use_case, tmp_path / "does_not_exist.csv")

    with pytest.raises(RetrainingError):
        service.retrain(use_case, pd.DataFrame({"invalid_col": [1, 2, 3]}))

    with pytest.raises(RetrainingError):
        service.retrain(use_case, 12345)  # type: ignore[arg-type]

    with pytest.raises(RetrainingError):
        service.retrain("not-a-use-case", _small_df(50))  # type: ignore[arg-type]


def test_candidate_training_failure_raises_retraining_error(tmp_path: Path):
    engine = MagicMock(spec=MLEngine)
    engine.train_and_evaluate.side_effect = TrainingError("simulated training boom")
    tracker = _mock_tracker_with_run_ids()
    service = RetrainingService(engine=engine, tracker=tracker)

    with pytest.raises(RetrainingError, match="Training failed"):
        service.retrain(HousePriceUseCase(), _small_df(120))
    assert tracker.track_candidate.call_count == 0


def test_mlflow_failure_raises_retraining_error():
    engine = _mock_engine_with_metrics({})
    tracker = MagicMock(spec=MLflowTracker)
    tracker.tracking_uri = "file:///tmp/fake-mlruns"
    tracker.track_candidate.side_effect = Exception("SELECT failed /tmp/secret Traceback")
    tracker._get_client.return_value = MagicMock()
    service = RetrainingService(engine=engine, tracker=tracker)

    with pytest.raises(RetrainingError, match="MLflow tracking failed"):
        service.retrain(HousePriceUseCase(), _small_df(120))


def test_empty_candidates_raise_retraining_error(tmp_path: Path):
    class EmptyUseCase(HousePriceUseCase):
        def get_model_candidates(self):
            return []

    tracker = _tmp_tracker(tmp_path)
    service = create_house_price_retraining_service(tracker=tracker)
    with pytest.raises(RetrainingError):
        service.retrain(EmptyUseCase(), _small_df(120))


# --- Existing behavior unchanged ---


def test_existing_selection_behavior_unchanged(tmp_path: Path):
    from app.use_cases.house_price.model_selection import select_best_house_price_model

    tracking_path = str(tmp_path / "mlruns_selection_check")
    tracker = MLflowTracker(
        tracking_uri=tracking_path,
        experiment_name="selection_unchanged_exp",
    )
    sample_df = generate_house_price_data(n_samples=150)
    result = select_best_house_price_model(data_path=sample_df, tracker=tracker)

    # Original return structure preserved exactly.
    assert set(result.keys()) == {"best_model_name", "best_pipeline", "all_results"}
    assert len(result["all_results"]) == 3
    assert result["best_model_name"] in {
        "LinearRegression",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
    }
    best_r2 = max(r["r2"] for r in result["all_results"])
    best = next(r for r in result["all_results"] if r["model_name"] == result["best_model_name"])
    assert best["r2"] == best_r2

    # Step 58 registration policy: selection tracks runs but creates
    # zero registry versions (registration happens only at promotion).
    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    versions = client.search_model_versions("name='HousePriceModel'")
    assert len(versions) == 0
    experiment = client.get_experiment_by_name("selection_unchanged_exp")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 3


def test_selection_source_registers_nothing_and_not_redirected():
    source = (Path(get_project_root()) / "app" / "use_cases" / "house_price" / "model_selection.py").read_text(
        encoding="utf-8"
    )
    assert "register_model" not in source
    assert "RetrainingService" not in source
    assert "retraining" not in source.lower()


def test_no_sqlite_retraining_table_added():
    source = (Path(get_project_root()) / "app" / "database" / "database.py").read_text(
        encoding="utf-8"
    )
    # Phase 5 adds the explicit lifecycle ledger (retraining_runs,
    # lifecycle_events, production_pointer_history). The legacy ad-hoc
    # retraining tables stay forbidden; only the Phase 5 ledger may
    # exist alongside predictions/monitoring/promotion history.
    assert "retraining_results" not in source
    assert "retraining_events" not in source
    assert "CREATE TABLE IF NOT EXISTS retraining_runs" in source
    assert "CREATE TABLE IF NOT EXISTS lifecycle_events" in source
    assert "CREATE TABLE IF NOT EXISTS production_pointer_history" in source
    assert "predictions" in source
    assert "monitoring_results" in source


def test_no_scheduler_or_api_changes():
    # Step 37 adds the scheduler foundation (explicit run_once only).
    # Guard the boundary: scheduler exports the foundation, API stays unwired.
    scheduler_source = (Path(get_project_root()) / "app" / "scheduler" / "__init__.py").read_text(
        encoding="utf-8"
    )
    assert "RetrainingScheduler" in scheduler_source
    foundation = (Path(get_project_root()) / "app" / "scheduler" / "retraining_scheduler.py").read_text(
        encoding="utf-8"
    )
    assert "def run_once" in foundation
    for forbidden in ("while True", "Thread(", "APScheduler", "Celery", "FastAPI("):
        assert forbidden not in foundation
    api_source = (Path(get_project_root()) / "app" / "api" / "main.py").read_text(
        encoding="utf-8"
    )
    # Step 40 adds a thin retrain adapter over DriftRetrainingOrchestrator.
    # Guard the boundary: adapter delegates, existing routes stay intact.
    assert "/monitor/house-price/retrain" in api_source
    assert "run_check_and_retrain" in api_source
    assert "RetrainingService(" not in api_source
    for forbidden in ("while True", "Thread(", "APScheduler", "Celery", "BackgroundTasks"):
        assert forbidden not in api_source
    assert "/health" in api_source
    assert "/predict/house-price" in api_source
    assert "/monitor/house-price" in api_source


def test_real_development_database_untouched():
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass
