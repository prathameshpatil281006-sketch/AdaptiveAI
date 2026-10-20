"""Step 60 (I-11) consolidation tests: one authoritative House Price path.

Proves legacy training, model selection, and orchestration share the
same implementation for preprocessing, split, candidates, and
train/evaluate, while preserving candidate configuration, metrics, and
result contracts. All MLflow work uses temporary stores; the real
mlruns/ registry is never touched.
"""

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline

from app.core.exceptions import TrainingError
from app.engine.ml_engine import MLEngine
from app.engine.mlflow_tracker import MLflowTracker
from app.use_cases.house_price import training as authoritative
from app.use_cases.house_price.model_selection import (
    _create_regressor as legacy_create_regressor,
    select_best_house_price_model,
)
from app.use_cases.house_price.train import train_house_price_model
from app.use_cases.house_price.use_case import HousePriceUseCase
from app.use_cases.house_price.data import generate_house_price_data
from app.retraining.retraining_service import (
    RetrainingService,
    _default_regressor_factory,
)


def _tmp_tracker(tmp_path: Path, exp: str = "consolidation_exp") -> MLflowTracker:
    return MLflowTracker(
        tracking_uri=str(tmp_path / "mlruns_consolidation"),
        experiment_name=exp,
    )


def _small_df(n_samples: int = 150) -> pd.DataFrame:
    return generate_house_price_data(n_samples=n_samples)


# --- Candidate set preserved exactly ---


def test_authoritative_candidates_match_use_case_exactly():
    expected = [
        {"model_type": "LinearRegression", "hyperparameters": {"fit_intercept": True}},
        {
            "model_type": "RandomForestRegressor",
            "hyperparameters": {"n_estimators": 100, "max_depth": None, "random_state": 42},
        },
        {
            "model_type": "GradientBoostingRegressor",
            "hyperparameters": {
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "random_state": 42,
            },
        },
    ]
    assert authoritative.get_house_price_candidates() == expected
    assert HousePriceUseCase().get_model_candidates() == expected
    assert [c["model_type"] for c in expected] == [
        "LinearRegression",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
    ]


def test_create_regressor_mapping_preserved():
    lr = authoritative.create_regressor("LinearRegression", {"fit_intercept": True})
    assert isinstance(lr, LinearRegression)
    assert lr.fit_intercept is True

    rf = authoritative.create_regressor(
        "RandomForestRegressor",
        {"n_estimators": 100, "max_depth": None, "random_state": 42},
    )
    assert isinstance(rf, RandomForestRegressor)
    assert (rf.n_estimators, rf.max_depth, rf.random_state) == (100, None, 42)

    gbr = authoritative.create_regressor(
        "GradientBoostingRegressor",
        {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3, "random_state": 42},
    )
    assert isinstance(gbr, GradientBoostingRegressor)

    # Legacy alias delegates to the same mapping.
    legacy_lr = legacy_create_regressor("LinearRegression", {"fit_intercept": True})
    assert type(legacy_lr) is type(lr)
    assert legacy_lr.get_params() == lr.get_params()

    with pytest.raises(TrainingError):
        authoritative.create_regressor("Unsupported", {})
    with pytest.raises(TrainingError):
        legacy_create_regressor("Unsupported", {})


def test_default_regressor_factory_uses_authoritative_path():
    with patch.object(
        authoritative, "create_regressor", wraps=authoritative.create_regressor
    ) as spy:
        reg = _default_regressor_factory("LinearRegression", {"fit_intercept": True})
        assert isinstance(reg, LinearRegression)
        assert spy.call_count == 1
        assert spy.call_args[0][0] == "LinearRegression"


# --- Shared implementation proofs ---


def test_all_paths_share_authoritative_module():
    from app.core.config import get_project_root

    root = Path(get_project_root())
    train_src = (root / "app" / "use_cases" / "house_price" / "train.py").read_text(
        encoding="utf-8"
    )
    sel_src = (root / "app" / "use_cases" / "house_price" / "model_selection.py").read_text(
        encoding="utf-8"
    )
    ret_src = (root / "app" / "retraining" / "retraining_service.py").read_text(
        encoding="utf-8"
    )
    assert "house_price.training" in train_src
    assert "house_price.training" in sel_src
    assert "house_price.training" in ret_src
    # Legacy divergent duplication removed from the two wrappers:
    # train.py no longer hand-rolls metrics/pipeline.
    assert "mean_absolute_error" not in train_src
    assert "mean_squared_error" not in train_src
    # Selection no longer duplicates preprocessing/split/candidate mapping.
    assert "get_preprocessor" not in sel_src
    assert "split_house_price_data" not in sel_src
    assert "HousePriceUseCase()" not in sel_src
    # Orchestration no longer imports the legacy selector mapping.
    assert "model_selection import _create_regressor" not in ret_src


def test_legacy_train_delegates_to_authoritative_single_path(tmp_path: Path):
    df = _small_df(150)
    csv_path = tmp_path / "legacy.csv"
    df.to_csv(csv_path, index=False)
    import app.use_cases.house_price.train as train_module

    with patch.object(
        train_module, "prepare_house_price_data", wraps=authoritative.prepare_house_price_data
    ) as spy_prepare, patch.object(
        train_module, "train_single_candidate", wraps=authoritative.train_single_candidate
    ) as spy_train:
        model, metrics = train_house_price_model(data_path=str(csv_path))
        assert spy_prepare.call_count == 1
        assert spy_train.call_count == 1
        assert spy_train.call_args[0][4] == "LinearRegression"
        assert spy_train.call_args[0][5] == {"fit_intercept": True}
    assert isinstance(model, Pipeline)
    assert set(metrics.keys()) == {"mae", "rmse", "r2"}


def test_selection_delegates_to_authoritative_evaluation(tmp_path: Path):
    tracker = _tmp_tracker(tmp_path, "cons_sel_delegate")
    df = _small_df(150)
    import app.use_cases.house_price.model_selection as selection_module

    with patch.object(
        selection_module, "prepare_house_price_data", wraps=authoritative.prepare_house_price_data
    ) as spy_prepare, patch.object(
        selection_module, "evaluate_all_candidates", wraps=authoritative.evaluate_all_candidates
    ) as spy_eval, patch.object(
        selection_module, "select_best_candidate", wraps=authoritative.select_best_candidate
    ) as spy_select:
        result = select_best_house_price_model(data_path=df, tracker=tracker)
        assert spy_prepare.call_count == 1
        assert spy_eval.call_count == 1
        assert spy_select.call_count == 1
    assert set(result.keys()) == {"best_model_name", "best_pipeline", "all_results"}
    assert len(result["all_results"]) == 3


def test_orchestration_uses_shared_train_path(tmp_path: Path):
    from unittest.mock import MagicMock

    tracker = MagicMock(spec=MLflowTracker)
    tracker.tracking_uri = "file:///tmp/fake-mlruns-consolidation"
    tracker.track_candidate.side_effect = lambda **kw: f"run-{kw.get('model_type')}"
    tracker._get_client.return_value = MagicMock()
    service = RetrainingService(engine=MLEngine(), tracker=tracker)
    with patch.object(
        authoritative,
        "train_candidate_with_instances",
        wraps=authoritative.train_candidate_with_instances,
    ) as spy:
        result = service.retrain(HousePriceUseCase(), _small_df(120))
        assert spy.call_count == 3
        called_names = {c.kwargs["model_name"] for c in spy.call_args_list}
        assert called_names == {
            "LinearRegression",
            "RandomForestRegressor",
            "GradientBoostingRegressor",
        }
    assert len(result["all_results"]) == 3


def test_selection_and_orchestration_agree_on_metrics(tmp_path: Path):
    df = _small_df(150)
    sel_tracker = _tmp_tracker(tmp_path, "cons_sel_agree")
    sel = select_best_house_price_model(data_path=df, tracker=sel_tracker)
    ort_tracker = MLflowTracker(
        tracking_uri=str(tmp_path / "mlruns_consort"),
        experiment_name="cons_ort_agree",
    )
    service = RetrainingService(engine=MLEngine(), tracker=ort_tracker)
    ort = service.retrain(HousePriceUseCase(), df)
    sel_by_name = {r["model_name"]: r for r in sel["all_results"]}
    ort_by_name = {r["model_name"]: r["metrics"] for r in ort["all_results"]}
    assert set(sel_by_name) == set(ort_by_name) == {
        "LinearRegression",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
    }
    for name in sel_by_name:
        assert sel_by_name[name]["mae"] == pytest.approx(ort_by_name[name]["mae"], abs=1e-8)
        assert sel_by_name[name]["rmse"] == pytest.approx(ort_by_name[name]["rmse"], abs=1e-8)
        assert sel_by_name[name]["r2"] == pytest.approx(ort_by_name[name]["r2"], abs=1e-8)
    # Same winner via the same max-R2 rule.
    assert sel["best_model_name"] == ort["best_model_name"]


def test_legacy_train_matches_selection_linear_metrics(tmp_path: Path):
    df = _small_df(200)
    tracker = _tmp_tracker(tmp_path, "cons_legacy_match")
    _, legacy_metrics = train_house_price_model(data_path=df)
    sel = select_best_house_price_model(data_path=df, tracker=tracker)
    linear = next(r for r in sel["all_results"] if r["model_name"] == "LinearRegression")
    assert legacy_metrics["mae"] == pytest.approx(linear["mae"], abs=1e-8)
    assert legacy_metrics["rmse"] == pytest.approx(linear["rmse"], abs=1e-8)
    assert legacy_metrics["r2"] == pytest.approx(linear["r2"], abs=1e-8)


# --- Contracts preserved ---


def test_legacy_train_contract(tmp_path: Path):
    df = _small_df(120)
    model, metrics = train_house_price_model(data_path=df)
    assert isinstance(model, Pipeline)
    assert hasattr(model, "predict")
    assert set(metrics.keys()) == {"mae", "rmse", "r2"}
    for v in metrics.values():
        assert isinstance(v, float)


def test_selection_contract_unchanged(tmp_path: Path):
    tracker = _tmp_tracker(tmp_path, "cons_sel_contract")
    result = select_best_house_price_model(data_path=_small_df(150), tracker=tracker)
    assert set(result.keys()) == {"best_model_name", "best_pipeline", "all_results"}
    assert isinstance(result["best_pipeline"], Pipeline)
    assert len(result["all_results"]) == 3
    for item in result["all_results"]:
        assert set(item.keys()) == {"model_name", "mae", "rmse", "r2"}


def test_orchestration_contract_unchanged(tmp_path: Path):
    tracker = MLflowTracker(
        tracking_uri=str(tmp_path / "mlruns_conort"),
        experiment_name="cons_ort_contract",
    )
    service = RetrainingService(engine=MLEngine(), tracker=tracker)
    result = service.retrain(HousePriceUseCase(), _small_df(120))
    assert set(result.keys()) >= {
        "use_case",
        "best_model_name",
        "best_pipeline",
        "best_metrics",
        "all_results",
        "best_run_id",
    }
    assert set(result["best_metrics"].keys()) == {"mae", "rmse", "r2"}
    for entry in result["all_results"]:
        assert set(entry.keys()) == {"model_name", "metrics", "run_id"}
        assert set(entry["metrics"].keys()) == {"mae", "rmse", "r2"}


def test_consolidated_paths_remain_tracking_only(tmp_path: Path):
    df = _small_df(120)
    sel_tracker = MLflowTracker(
        tracking_uri=str(tmp_path / "mlruns_conreg_sel"),
        experiment_name="cons_reg_sel",
    )
    select_best_house_price_model(data_path=df, tracker=sel_tracker)
    ort_tracker = MLflowTracker(
        tracking_uri=str(tmp_path / "mlruns_conreg_ort"),
        experiment_name="cons_reg_ort",
    )
    service = RetrainingService(engine=MLEngine(), tracker=ort_tracker)
    service.retrain(HousePriceUseCase(), df)
    for tracker in (sel_tracker, ort_tracker):
        client = MlflowClient(tracking_uri=tracker.tracking_uri)
        assert client.search_model_versions("name='HousePriceModel'") == []
