"""Focused tests for ModelEvaluator candidate-vs-production comparison (Step 31).

Evaluation-only: never touches real production artifacts, the real
database, or the default MLflow registry. All file-based production cases
use temporary directories.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from app.core.config import get_project_root
from app.core.exceptions import AdaptiveAIError, RetrainingError
from app.retraining.model_evaluator import (
    ModelEvaluator,
    compute_regression_metrics,
    evaluate_retraining_candidate,
)
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_persistence import save_house_price_model
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.split import split_house_price_data


def _xy_simple():
    X_test = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
    y_test = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    return X_test, y_test


def _stub_model(y_pred, record=None):
    model = MagicMock()
    pred = np.asarray(y_pred, dtype=float)

    def _predict(X):
        if record is not None:
            record.append(X)
        return pred

    model.predict = MagicMock(side_effect=_predict)
    return model


# --- A. Candidate better ---


def test_a_candidate_better_when_r2_higher():
    X_test, y_test = _xy_simple()
    candidate = _stub_model([1.0, 2.0, 3.0, 4.0, 5.0])  # perfect -> r2 == 1.0
    production = _stub_model([1.0, 1.0, 1.0, 1.0, 1.0])  # poor -> low r2

    result = ModelEvaluator().evaluate_candidate(
        candidate,
        production,
        X_test,
        y_test,
        candidate_model_name="CandidateA",
        production_model_name="ProdA",
    )

    assert result["status"] == "candidate_better"
    assert result["comparison"]["primary_metric"] == "r2"
    assert result["comparison"]["candidate_better"] is True
    expected_improvement = (
        result["candidate"]["metrics"]["r2"] - result["production"]["metrics"]["r2"]
    )
    assert result["comparison"]["improvement"] == pytest.approx(expected_improvement)
    assert result["comparison"]["candidate_score"] == pytest.approx(
        result["candidate"]["metrics"]["r2"]
    )
    assert result["comparison"]["production_score"] == pytest.approx(
        result["production"]["metrics"]["r2"]
    )
    assert result["candidate"]["metrics"]["r2"] > result["production"]["metrics"]["r2"]


# --- B. Production better ---


def test_b_production_better_when_candidate_r2_lower():
    X_test, y_test = _xy_simple()
    candidate = _stub_model([1.0, 1.0, 1.0, 1.0, 1.0])
    production = _stub_model([1.0, 2.0, 3.0, 4.0, 5.0])

    result = ModelEvaluator().evaluate_candidate(
        candidate,
        production,
        X_test,
        y_test,
        candidate_model_name="CandidateB",
        production_model_name="ProdB",
    )

    assert result["status"] == "production_better"
    assert result["comparison"]["candidate_better"] is False
    assert result["comparison"]["improvement"] < 0
    assert result["comparison"]["improvement"] == pytest.approx(
        result["candidate"]["metrics"]["r2"] - result["production"]["metrics"]["r2"]
    )


# --- C. Equal score ---


def test_c_equal_r2_is_not_better():
    X_test, y_test = _xy_simple()
    candidate = _stub_model([1.5, 2.5, 3.5, 4.5, 5.5])
    production = _stub_model([1.5, 2.5, 3.5, 4.5, 5.5])

    result = ModelEvaluator().evaluate_candidate(
        candidate,
        production,
        X_test,
        y_test,
        candidate_model_name="CandidateC",
        production_model_name="ProdC",
    )

    assert result["candidate"]["metrics"]["r2"] == pytest.approx(
        result["production"]["metrics"]["r2"]
    )
    assert result["comparison"]["candidate_better"] is False
    assert result["comparison"]["improvement"] == pytest.approx(0.0)
    # Equal must NOT read as candidate_better and must not imply promotion.
    assert result["status"] == "production_better"


# --- D. Same evaluation data ---


def test_d_candidate_and_production_share_identical_evaluation_data():
    X_test, y_test = _xy_simple()
    candidate_seen = []
    production_seen = []
    candidate = _stub_model([1.0, 2.0, 3.0, 4.0, 5.0], record=candidate_seen)
    production = _stub_model([1.0, 1.0, 1.0, 1.0, 1.0], record=production_seen)

    metrics_seen_y = []
    base_fn = compute_regression_metrics

    def _spy_metrics(y_true, y_pred):
        metrics_seen_y.append(y_true)
        return base_fn(y_true, y_pred)

    evaluator = ModelEvaluator(metrics_fn=_spy_metrics)
    evaluator.evaluate_candidate(
        candidate,
        production,
        X_test,
        y_test,
        candidate_model_name="cand",
        production_model_name="prod",
    )

    assert candidate.predict.call_count == 1
    assert production.predict.call_count == 1
    # Identical objects (not copies / different splits) reach both models.
    assert candidate_seen[0] is X_test
    assert production_seen[0] is X_test
    assert candidate.predict.call_args[0][0] is X_test
    assert production.predict.call_args[0][0] is X_test
    assert len(metrics_seen_y) == 2
    assert metrics_seen_y[0] is y_test
    assert metrics_seen_y[1] is y_test


# --- E. All metrics ---


def test_e_all_metrics_present_for_both_models():
    X_test, y_test = _xy_simple()
    candidate = _stub_model([1.0, 2.0, 3.0, 4.0, 5.0])
    production = _stub_model([1.1, 2.1, 2.9, 4.2, 4.8])

    result = ModelEvaluator().evaluate_candidate(
        candidate,
        production,
        X_test,
        y_test,
        candidate_model_name="cand",
        production_model_name="prod",
    )

    for block in (result["candidate"], result["production"]):
        assert set(block["metrics"].keys()) == {"mae", "rmse", "r2"}
        for key in ("mae", "rmse", "r2"):
            assert isinstance(block["metrics"][key], float)
    # Identity labels preserved.
    assert result["candidate"]["model_name"] == "cand"
    assert result["production"]["model_name"] == "prod"
    assert result["comparison"]["primary_metric"] == "r2"


# --- F. No production model ---


def test_f_no_production_model_returns_valid_result():
    X_test, y_test = _xy_simple()
    candidate = _stub_model([1.0, 2.0, 3.0, 4.0, 5.0])

    result = ModelEvaluator().evaluate_candidate(
        candidate,
        None,
        X_test,
        y_test,
        candidate_model_name="OnlyCandidate",
        production_model_name=None,
    )

    assert result["status"] == "no_production_model"
    assert set(result["candidate"]["metrics"].keys()) == {"mae", "rmse", "r2"}
    assert isinstance(result["candidate"]["metrics"]["r2"], float)
    # No fabricated production metrics.
    assert result["production"] is None
    assert result["comparison"]["production_score"] is None
    assert result["comparison"]["improvement"] is None
    assert result["comparison"]["candidate_better"] is False
    assert result["comparison"]["candidate_score"] == pytest.approx(
        result["candidate"]["metrics"]["r2"]
    )


# --- G. Missing production file ---


def test_g_missing_production_file_is_clean_no_production(tmp_path: Path):
    X_test, y_test = _xy_simple()
    candidate = _stub_model([1.0, 2.0, 3.0, 4.0, 5.0])

    result = ModelEvaluator().evaluate_candidate(
        candidate,
        None,
        X_test,
        y_test,
        candidate_model_name="cand",
        production_model_name="DoesNotExist123",
        models_dir=tmp_path,
    )

    assert result["status"] == "no_production_model"
    assert result["production"] is None
    assert set(result["candidate"]["metrics"].keys()) == {"mae", "rmse", "r2"}

    # File-helper variant behaves the same.
    result2 = ModelEvaluator().evaluate_candidate_vs_production_file(
        candidate,
        X_test,
        y_test,
        candidate_model_name="cand",
        production_model_name="AlsoMissing",
        models_dir=tmp_path,
    )
    assert result2["status"] == "no_production_model"
    assert result2["production"] is None


# --- H. Broken production model ---


def test_h_broken_production_predict_raises_not_missing():
    X_test, y_test = _xy_simple()
    candidate = _stub_model([1.0, 2.0, 3.0, 4.0, 5.0])
    broken = MagicMock()
    broken.predict = MagicMock(side_effect=RuntimeError("boom"))

    with pytest.raises(RetrainingError):
        ModelEvaluator().evaluate_candidate(
            candidate,
            broken,
            X_test,
            y_test,
            candidate_model_name="cand",
            production_model_name="prod",
        )


def test_h_broken_production_loader_raises_not_missing():
    X_test, y_test = _xy_simple()
    candidate = _stub_model([1.0, 2.0, 3.0, 4.0, 5.0])

    def _broken_loader(model_name, models_dir=None):
        raise AdaptiveAIError("corrupt artifact")

    with pytest.raises(RetrainingError):
        ModelEvaluator(production_loader=_broken_loader).evaluate_candidate(
            candidate,
            None,
            X_test,
            y_test,
            candidate_model_name="cand",
            production_model_name="Corrupt",
            models_dir=None,
        )


def test_h_corrupt_production_file_raises_not_missing(tmp_path: Path):
    corrupt = tmp_path / "Corrupt.joblib"
    corrupt.write_text("this is not a joblib model", encoding="utf-8")
    X_test, y_test = _xy_simple()
    candidate = _stub_model([1.0, 2.0, 3.0, 4.0, 5.0])

    with pytest.raises(RetrainingError):
        ModelEvaluator().evaluate_candidate(
            candidate,
            None,
            X_test,
            y_test,
            candidate_model_name="cand",
            production_model_name="Corrupt",
            models_dir=tmp_path,
        )


# --- I. Candidate evaluation failure ---


def test_i_candidate_evaluation_failure_raises_retraining_error():
    X_test, y_test = _xy_simple()
    broken_candidate = MagicMock()
    broken_candidate.predict = MagicMock(side_effect=ValueError("bad features"))
    production = _stub_model([1.0, 2.0, 3.0, 4.0, 5.0])

    with pytest.raises(RetrainingError):
        ModelEvaluator().evaluate_candidate(
            broken_candidate,
            production,
            X_test,
            y_test,
            candidate_model_name="cand",
            production_model_name="prod",
        )


def test_i_candidate_metrics_failure_raises_retraining_error():
    X_test, y_test = _xy_simple()
    candidate = _stub_model([1.0, 2.0, 3.0, 4.0, 5.0])
    production = _stub_model([1.0, 2.0, 3.0, 4.0, 5.0])

    def _bad_metrics(y_true, y_pred):
        raise ValueError("metrics exploded")

    with pytest.raises(RetrainingError):
        ModelEvaluator(metrics_fn=_bad_metrics).evaluate_candidate(
            candidate,
            production,
            X_test,
            y_test,
            candidate_model_name="cand",
            production_model_name="prod",
        )


def test_i_candidate_failure_without_production_still_raises():
    X_test, y_test = _xy_simple()
    broken_candidate = MagicMock()
    broken_candidate.predict = MagicMock(side_effect=RuntimeError("nope"))

    with pytest.raises(RetrainingError):
        ModelEvaluator().evaluate_candidate(
            broken_candidate,
            None,
            X_test,
            y_test,
            candidate_model_name="cand",
            production_model_name=None,
        )


# --- J. No side effects ---


def test_j_evaluator_source_has_no_side_effect_wiring():
    source = (Path(get_project_root()) / "app" / "retraining" / "model_evaluator.py").read_text(
        encoding="utf-8"
    )
    assert "load_house_price_model" in source
    for forbidden in (
        "save_house_price_model",
        "register_model",
        "ensure_registered_model",
        "create_model_version",
        "joblib.dump",
        "joblib.load",
        "import sqlite3",
        "sqlite3",
        "mlflow",
        "Mlflow",
        "APScheduler",
        "Celery",
        "streamlit",
        "Streamlit",
        "FastAPI(",
    ):
        assert forbidden not in source


def test_j_evaluation_creates_no_files_and_calls_no_save(tmp_path: Path, monkeypatch):
    import app.use_cases.house_price.model_persistence as persistence

    def _fail_save(*args, **kwargs):
        raise AssertionError("ModelEvaluator must not save models")

    monkeypatch.setattr(persistence, "save_house_price_model", _fail_save)

    X_test, y_test = _xy_simple()
    candidate = _stub_model([1.0, 2.0, 3.0, 4.0, 5.0])
    production = _stub_model([1.0, 1.0, 1.0, 1.0, 1.0])

    workdir = tmp_path / "work"
    workdir.mkdir()
    before = set(p.name for p in workdir.iterdir())

    result = ModelEvaluator().evaluate_candidate(
        candidate,
        production,
        X_test,
        y_test,
        candidate_model_name="cand",
        production_model_name="prod",
        models_dir=workdir,
    )
    assert result["status"] in {"candidate_better", "production_better"}
    assert set(p.name for p in workdir.iterdir()) == before

    # Real production directory untouched.
    models_dir = Path(get_project_root()) / "models" / "house_price"
    prod_before = set(p.name for p in models_dir.glob("*.joblib")) if models_dir.exists() else set()
    ModelEvaluator().evaluate_candidate(
        candidate,
        production,
        X_test,
        y_test,
        candidate_model_name="cand",
        production_model_name="prod",
    )
    prod_after = set(p.name for p in models_dir.glob("*.joblib")) if models_dir.exists() else set()
    assert prod_before == prod_after


def test_j_no_registry_or_db_side_effects():
    evaluator_source = (
        Path(get_project_root()) / "app" / "retraining" / "model_evaluator.py"
    ).read_text(encoding="utf-8")
    assert "search_model_versions" not in evaluator_source
    assert "CREATE TABLE" not in evaluator_source
    assert "retraining_results" not in evaluator_source
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass


# --- K. Real House Price pipeline ---


def _fit_house_price_pipeline(estimator, X_train, y_train) -> Pipeline:
    pipeline = Pipeline(
        steps=[("preprocessor", get_preprocessor()), ("model", estimator)]
    )
    pipeline.fit(X_train, y_train)
    return pipeline


def test_k_real_house_price_pipelines_share_test_split():
    df = generate_house_price_data(n_samples=150)
    X_train, X_test, y_train, y_test = split_house_price_data(df)

    candidate = _fit_house_price_pipeline(LinearRegression(), X_train, y_train)
    production = _fit_house_price_pipeline(
        DummyRegressor(strategy="mean"), X_train, y_train
    )

    result = ModelEvaluator().evaluate_candidate(
        candidate,
        production,
        X_test,
        y_test,
        candidate_model_name="LinearRegression",
        production_model_name="DummyMean",
    )

    for block in (result["candidate"], result["production"]):
        assert set(block["metrics"].keys()) == {"mae", "rmse", "r2"}
        for key in ("mae", "rmse", "r2"):
            assert isinstance(block["metrics"][key], float)
            assert np.isfinite(block["metrics"][key])
    assert result["comparison"]["primary_metric"] == "r2"
    assert result["comparison"]["improvement"] == pytest.approx(
        result["candidate"]["metrics"]["r2"] - result["production"]["metrics"]["r2"]
    )
    assert result["comparison"]["candidate_better"] == (
        result["candidate"]["metrics"]["r2"] > result["production"]["metrics"]["r2"]
    )
    assert result["status"] in {"candidate_better", "production_better"}
    assert result["candidate"]["model_name"] == "LinearRegression"
    assert result["production"]["model_name"] == "DummyMean"


def test_k_dataframe_helper_reuses_validated_split(tmp_path: Path):
    df = generate_house_price_data(n_samples=150)
    X_train, X_test, y_train, y_test = split_house_price_data(df)
    candidate = _fit_house_price_pipeline(LinearRegression(), X_train, y_train)

    # Persist a real production artifact only inside tmp_path.
    production = _fit_house_price_pipeline(LinearRegression(), X_train, y_train)
    save_house_price_model(production, "ProdModel", models_dir=tmp_path)

    evaluator = ModelEvaluator()
    result = evaluator.evaluate_house_price_candidate(
        candidate,
        df,
        candidate_model_name="LinearRegression",
        production_model_name="ProdModel",
        models_dir=tmp_path,
    )

    assert set(result["candidate"]["metrics"].keys()) == {"mae", "rmse", "r2"}
    assert set(result["production"]["metrics"].keys()) == {"mae", "rmse", "r2"}
    assert result["production"]["model_name"] == "ProdModel"
    assert result["status"] in {"candidate_better", "production_better"}

    # Missing file through the same helper -> no production, candidate intact.
    missing = evaluator.evaluate_house_price_candidate(
        candidate,
        df,
        candidate_model_name="LinearRegression",
        production_model_name="MissingProd",
        models_dir=tmp_path,
    )
    assert missing["status"] == "no_production_model"
    assert missing["production"] is None
    assert set(missing["candidate"]["metrics"].keys()) == {"mae", "rmse", "r2"}


def test_k_adapter_connects_retraining_result_without_changing_retrain():
    import inspect

    from app.retraining.retraining_service import RetrainingService

    sig = inspect.signature(RetrainingService.retrain)
    names = list(sig.parameters.keys())
    # Backward-compatible extension (Step 34): legacy params first,
    # opt-in promotion params appended with safe defaults.
    assert names[:3] == ["self", "use_case", "data_path"]
    assert "promote" in names
    assert sig.parameters["promote"].default is False

    df = generate_house_price_data(n_samples=150)
    X_train, X_test, y_train, y_test = split_house_price_data(df)
    candidate = _fit_house_price_pipeline(LinearRegression(), X_train, y_train)
    production = _fit_house_price_pipeline(
        DummyRegressor(strategy="mean"), X_train, y_train
    )

    result = evaluate_retraining_candidate(
        candidate,
        X_test,
        y_test,
        candidate_model_name="LinearRegression",
        production_model=production,
        production_model_name="DummyMean",
    )
    assert result["candidate"]["model_name"] == "LinearRegression"
    assert result["production"]["model_name"] == "DummyMean"
    assert result["status"] in {"candidate_better", "production_better"}
