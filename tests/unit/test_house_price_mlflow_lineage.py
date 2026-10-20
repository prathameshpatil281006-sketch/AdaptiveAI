"""Step 61 (I-12) lineage tests: ONE experiment + coherent tags.

Proves normal selection and orchestration evaluation share the canonical
House Price experiment and lineage convention while preserving the Step 58
registry policy (runs only) and exact best_run_id lineage. All MLflow work
uses temporary stores; the real mlruns/ registry is never touched.
"""

import inspect
from pathlib import Path

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient

from app.engine.mlflow_tracker import (
    HOUSE_PRICE_EXPERIMENT_NAME,
    HOUSE_PRICE_USE_CASE_NAME,
    TRAINING_CONTEXT_SELECTION,
    TRAINING_CONTEXT_RETRAINING,
    MLflowTracker,
)
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_selection import select_best_house_price_model
from app.use_cases.house_price.use_case import HousePriceUseCase
from app.retraining.retraining_service import RetrainingService


def _small_df(n_samples: int = 100) -> pd.DataFrame:
    return generate_house_price_data(n_samples=n_samples)


def _tracker(tmp_path: Path, store: str, exp: str) -> MLflowTracker:
    return MLflowTracker(tracking_uri=str(tmp_path / store), experiment_name=exp)


def _runs(tracker: MLflowTracker, exp: str):
    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    experiment = client.get_experiment_by_name(exp)
    assert experiment is not None
    return client.search_runs(experiment_ids=[experiment.experiment_id])


def test_single_experiment_convention_defined_and_reused():
    assert HOUSE_PRICE_EXPERIMENT_NAME == "house_price"
    assert HOUSE_PRICE_USE_CASE_NAME == "house_price"
    assert TRAINING_CONTEXT_SELECTION == "selection"
    assert TRAINING_CONTEXT_RETRAINING == "retraining"

    # MLflowTracker default is the canonical experiment.
    sig = inspect.signature(MLflowTracker.__init__)
    assert sig.parameters["experiment_name"].default == HOUSE_PRICE_EXPERIMENT_NAME

    # Both callers reuse the constant (no divergent literals).
    from app.core.config import get_project_root

    root = Path(get_project_root())
    sel_src = (root / "app" / "use_cases" / "house_price" / "model_selection.py").read_text(
        encoding="utf-8"
    )
    ret_src = (root / "app" / "retraining" / "retraining_service.py").read_text(
        encoding="utf-8"
    )
    tracker_src = (root / "app" / "engine" / "mlflow_tracker.py").read_text(
        encoding="utf-8"
    )
    assert "HOUSE_PRICE_EXPERIMENT_NAME" in sel_src
    assert "HOUSE_PRICE_EXPERIMENT_NAME" in ret_src
    assert 'experiment_name="house_price_prediction"' not in sel_src
    assert 'experiment_name="house_price_retraining"' not in ret_src
    assert "HOUSE_PRICE_EXPERIMENT_NAME" in tracker_src


def test_default_tracker_uses_canonical_experiment(tmp_path: Path):
    tracker = MLflowTracker(tracking_uri=str(tmp_path / "mlruns_default"))
    assert tracker.experiment_name == HOUSE_PRICE_EXPERIMENT_NAME
    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    assert client.get_experiment_by_name(HOUSE_PRICE_EXPERIMENT_NAME) is not None


def test_selection_lineage_tags_coherent(tmp_path: Path):
    tracker = _tracker(tmp_path, "mlruns_sel", HOUSE_PRICE_EXPERIMENT_NAME)
    result = select_best_house_price_model(data_path=_small_df(), tracker=tracker)
    assert len(result["all_results"]) == 3

    runs = _runs(tracker, HOUSE_PRICE_EXPERIMENT_NAME)
    assert len(runs) == 3
    by_tag = {r.data.tags.get("model_type"): r for r in runs}
    assert set(by_tag) == {
        "LinearRegression",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
    }
    best_count = 0
    for model_name, run in by_tag.items():
        tags = run.data.tags
        assert tags.get("use_case") == "house_price"
        assert tags.get("model_name") == model_name
        assert tags.get("training_context") == "selection"
        assert tags.get("training_stage") == "selection"
        assert tags.get("candidate_role") in {"candidate", "best"}
        if tags.get("candidate_role") == "best":
            best_count += 1
    assert best_count == 1
    best_run = next(r for r in runs if r.data.tags.get("candidate_role") == "best")
    assert best_run.data.tags.get("model_type") == result["best_model_name"]


def test_retraining_lineage_tags_coherent(tmp_path: Path):
    tracker = _tracker(tmp_path, "mlruns_ret", HOUSE_PRICE_EXPERIMENT_NAME)
    service = RetrainingService(tracker=tracker)
    result = service.retrain(HousePriceUseCase(), _small_df())
    assert len(result["all_results"]) == 3

    runs = _runs(tracker, HOUSE_PRICE_EXPERIMENT_NAME)
    assert len(runs) == 3
    by_tag = {r.data.tags.get("model_type"): r for r in runs}
    assert set(by_tag) == {
        "LinearRegression",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
    }
    for model_name, run in by_tag.items():
        tags = run.data.tags
        # Legacy keys preserved.
        assert tags.get("training_stage") == "retraining"
        assert tags.get("use_case") == "house_price"
        # Shared convention keys.
        assert tags.get("training_context") == "retraining"
        assert tags.get("model_name") == model_name
    best_runs = [r for r in runs if r.data.tags.get("candidate_role") == "best"]
    assert len(best_runs) == 1
    assert best_runs[0].info.run_id == result["best_run_id"]
    assert best_runs[0].data.tags.get("model_type") == result["best_model_name"]


def test_selection_and_retraining_share_one_experiment(tmp_path: Path):
    store = str(tmp_path / "mlruns_shared")
    sel_tracker = MLflowTracker(
        tracking_uri=store, experiment_name=HOUSE_PRICE_EXPERIMENT_NAME
    )
    ret_tracker = MLflowTracker(
        tracking_uri=store, experiment_name=HOUSE_PRICE_EXPERIMENT_NAME
    )
    df = _small_df()
    sel = select_best_house_price_model(data_path=df, tracker=sel_tracker)
    service = RetrainingService(tracker=ret_tracker)
    ort = service.retrain(HousePriceUseCase(), df)

    client = MlflowClient(tracking_uri=sel_tracker.tracking_uri)
    experiment = client.get_experiment_by_name(HOUSE_PRICE_EXPERIMENT_NAME)
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 6
    contexts = sorted(r.data.tags.get("training_context") for r in runs)
    assert contexts == ["retraining"] * 3 + ["selection"] * 3
    # Same winner via the same data/seed/metrics.
    assert sel["best_model_name"] == ort["best_model_name"]


def test_selection_creates_zero_registry_versions(tmp_path: Path):
    tracker = _tracker(tmp_path, "mlruns_selreg", HOUSE_PRICE_EXPERIMENT_NAME)
    result = select_best_house_price_model(data_path=_small_df(), tracker=tracker)
    assert len(result["all_results"]) == 3
    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    assert client.search_model_versions("name='HousePriceModel'") == []


def test_retraining_evaluation_returns_exact_best_run_id(tmp_path: Path):
    tracker = _tracker(tmp_path, "mlruns_retrun", HOUSE_PRICE_EXPERIMENT_NAME)
    service = RetrainingService(tracker=tracker)
    result = service.retrain(HousePriceUseCase(), _small_df())
    assert isinstance(result["best_run_id"], str) and result["best_run_id"]

    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    run = client.get_run(result["best_run_id"])
    assert run.data.tags.get("model_type") == result["best_model_name"]
    assert run.data.tags.get("candidate_role") == "best"
    # Evaluation alone registers nothing (Step 58).
    assert client.search_model_versions("name='HousePriceModel'") == []
    # best_metrics match the best run's logged metrics.
    assert result["best_metrics"]["r2"] == pytest.approx(run.data.metrics["r2"])
