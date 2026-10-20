"""Step 55 tests: explicit initial deployment through the promoter.

Uses stubs or temporary directories/MLflow stores only. Never touches
the real models/, mlruns/, or database.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from app.core.config import get_project_root
from app.core.exceptions import ModelPromotionError, RetrainingError
from app.engine.ml_engine import MLEngine
from app.engine.mlflow_tracker import MLflowTracker
from app.retraining.model_promoter import PROMOTED, ModelPromoter
from app.retraining.model_promotion import ModelPromotionDecision
from app.retraining.retraining_service import RetrainingService
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_persistence import read_production_pointer
from app.use_cases.house_price.use_case import HousePriceUseCase


def _no_production_result(**overrides):
    base = {
        "candidate": {"model_name": "c", "metrics": {"mae": 1.0, "rmse": 2.0, "r2": 0.85}},
        "production": None,
        "comparison": {
            "primary_metric": "r2",
            "candidate_score": 0.85,
            "production_score": None,
            "improvement": None,
            "candidate_better": False,
        },
        "status": "no_production_model",
    }
    base.update(overrides)
    return base


def _candidate_better_result():
    return {
        "candidate": {"model_name": "c", "metrics": {"mae": 1.0, "rmse": 2.0, "r2": 0.9}},
        "production": {"model_name": "p", "metrics": {"mae": 2.0, "rmse": 3.0, "r2": 0.7}},
        "comparison": {
            "primary_metric": "r2",
            "candidate_score": 0.9,
            "production_score": 0.7,
            "improvement": 0.2,
            "candidate_better": True,
        },
        "status": "candidate_better",
    }


def _stub_candidate():
    model = MagicMock()
    model.predict = MagicMock(return_value=np.asarray([1.0, 2.0]))
    return model


def _make_promoter(save_return=None, register_return=None):
    save_mock = MagicMock(
        return_value=save_return if save_return is not None else Path("/tmp/fake.joblib")
    )
    tracker_mock = MagicMock()
    tracker_mock.register_model = MagicMock(
        return_value=register_return
        if register_return is not None
        else {
            "registered_model_name": "HousePriceModel",
            "version": "1",
            "run_id": "run-first",
            "source": "runs:/run-first/model",
        }
    )
    promoter = ModelPromoter(
        decision=ModelPromotionDecision(),
        save_fn=save_mock,
        tracker=tracker_mock,
    )
    return promoter, save_mock, tracker_mock


def test_deploy_initial_success_reports_promoted():
    promoter, save_mock, tracker_mock = _make_promoter()
    out = promoter.deploy_initial_candidate(
        _stub_candidate(),
        _no_production_result(),
        candidate_model_name="LinearRegression",
        candidate_run_id="run-first",
    )
    assert out["decision"] == "promoted"
    assert out["decision"] == PROMOTED
    assert out["model_name"] == "LinearRegression"
    assert out["run_id"] == "run-first"
    assert out["registered_model_name"] == "HousePriceModel"
    assert out["model_version"] == "1"
    assert save_mock.call_count == 1
    assert tracker_mock.register_model.call_count == 1
    _, reg_kwargs = tracker_mock.register_model.call_args
    assert reg_kwargs["run_id"] == "run-first"


def test_deploy_initial_writes_no_pointer_itself(tmp_path: Path):
    promoter, _, _ = _make_promoter()
    promoter.deploy_initial_candidate(
        _stub_candidate(),
        _no_production_result(),
        candidate_model_name="LinearRegression",
        candidate_run_id="run-first",
        models_dir=tmp_path,
    )
    # Publication is the service's follow-up step, not part of deployment.
    assert not (tmp_path / "production.json").exists()


def test_deploy_initial_refuses_existing_production():
    promoter, save_mock, tracker_mock = _make_promoter()
    with pytest.raises(ModelPromotionError):
        promoter.deploy_initial_candidate(
            _stub_candidate(),
            _candidate_better_result(),
            candidate_model_name="LinearRegression",
            candidate_run_id="run-first",
        )
    save_mock.assert_not_called()
    tracker_mock.register_model.assert_not_called()


def test_deploy_initial_refuses_unknown_decision():
    promoter, save_mock, tracker_mock = _make_promoter()
    with pytest.raises(ModelPromotionError):
        promoter.deploy_initial_candidate(
            _stub_candidate(),
            {"status": "bogus", "comparison": {}},
            candidate_model_name="LinearRegression",
            candidate_run_id="run-first",
        )
    save_mock.assert_not_called()
    tracker_mock.register_model.assert_not_called()


def test_deploy_initial_validates_identity():
    promoter, save_mock, tracker_mock = _make_promoter()
    for name, run_id in ((None, "r"), ("", "r"), ("   ", "r"), ("m", None), ("m", ""), ("m", "  ")):
        with pytest.raises(ModelPromotionError):
            promoter.deploy_initial_candidate(
                _stub_candidate(),
                _no_production_result(),
                candidate_model_name=name,
                candidate_run_id=run_id,
            )
    save_mock.assert_not_called()
    tracker_mock.register_model.assert_not_called()


def test_deploy_initial_requires_tracker():
    promoter = ModelPromoter(decision=ModelPromotionDecision(), save_fn=MagicMock())
    with pytest.raises(ModelPromotionError):
        promoter.deploy_initial_candidate(
            _stub_candidate(),
            _no_production_result(),
            candidate_model_name="m",
            candidate_run_id="r",
        )


def test_update_production_pointer_publishes_promoted(tmp_path: Path):
    promoter, _, _ = _make_promoter()
    pointer = promoter.update_production_pointer(
        {
            "decision": "promoted",
            "model_name": "LinearRegression",
            "run_id": "run-first",
            "registered_model_name": "HousePriceModel",
            "model_version": "3",
            "reason": "Promoted.",
        },
        models_dir=tmp_path,
    )
    assert pointer["model_name"] == "LinearRegression"
    assert pointer["model_version"] == "3"
    assert pointer["run_id"] == "run-first"
    stored = read_production_pointer(models_dir=tmp_path)
    assert stored == pointer


def test_update_production_pointer_rejects_non_promoted(tmp_path: Path):
    promoter, _, _ = _make_promoter()
    for decision in ("rejected", "requires_initial_promotion", "failed"):
        with pytest.raises(ModelPromotionError):
            promoter.update_production_pointer(
                {
                    "decision": decision,
                    "model_name": "m",
                    "run_id": "r",
                    "registered_model_name": "HousePriceModel",
                    "model_version": None,
                    "reason": "x",
                },
                models_dir=tmp_path,
            )
    assert not (tmp_path / "production.json").exists()


def test_update_production_pointer_rejects_missing_version(tmp_path: Path):
    promoter, _, _ = _make_promoter()
    with pytest.raises(ModelPromotionError):
        promoter.update_production_pointer(
            {
                "decision": "promoted",
                "model_name": "m",
                "run_id": "r",
                "registered_model_name": "HousePriceModel",
                "model_version": None,
                "reason": "x",
            },
            models_dir=tmp_path,
        )
    assert not (tmp_path / "production.json").exists()


class _PromoteOnlyStub:
    """Duck-typed promoter without initial-deployment support."""

    def __init__(self, result):
        self._result = result

    def promote(self, **kwargs):
        return self._result


def test_service_initial_flag_without_supporting_promoter_fails(tmp_path: Path):
    tracker = MLflowTracker(
        tracking_uri=str(tmp_path / "mlruns"), experiment_name="house_price_retraining"
    )
    service = RetrainingService(
        engine=MLEngine(),
        tracker=tracker,
        promoter=_PromoteOnlyStub({"decision": "requires_initial_promotion"}),
    )
    df = generate_house_price_data(n_samples=60)
    with pytest.raises(RetrainingError):
        service.retrain(
            HousePriceUseCase(), df, promote=True, allow_initial_promotion=True,
            models_dir=tmp_path,
        )
    assert not (tmp_path / "production.json").exists()


def test_service_promote_true_default_still_refuses_without_side_effects(tmp_path: Path):
    tracker = MLflowTracker(
        tracking_uri=str(tmp_path / "mlruns"), experiment_name="house_price_retraining"
    )
    from app.retraining.model_promoter import create_house_price_promoter

    promoter = create_house_price_promoter(tracker=tracker, models_dir=str(tmp_path / "models"))
    service = RetrainingService(engine=MLEngine(), tracker=tracker, promoter=promoter)
    df = generate_house_price_data(n_samples=120)
    result = service.retrain(
        HousePriceUseCase(), df, promote=True, models_dir=str(tmp_path / "models")
    )
    assert result["promotion"]["decision"] == "requires_initial_promotion"
    assert not (tmp_path / "models" / "production.json").exists()


def test_service_initial_deployment_creates_serving_state(tmp_path: Path):
    models_dir = tmp_path / "models"
    tracker = MLflowTracker(
        tracking_uri=str(tmp_path / "mlruns"), experiment_name="house_price_retraining"
    )
    from app.retraining.model_promoter import create_house_price_promoter

    promoter = create_house_price_promoter(tracker=tracker, models_dir=str(models_dir))
    service = RetrainingService(engine=MLEngine(), tracker=tracker, promoter=promoter)
    df = generate_house_price_data(n_samples=120)
    result = service.retrain(
        HousePriceUseCase(),
        df,
        promote=True,
        allow_initial_promotion=True,
        models_dir=str(models_dir),
    )
    promotion = result["promotion"]
    assert promotion["decision"] == "promoted"
    assert promotion["model_version"]
    pointer = read_production_pointer(models_dir=models_dir)
    assert pointer["model_name"] == result["best_model_name"]
    assert pointer["run_id"] == result["best_run_id"]
    assert pointer["registered_model_name"] == "HousePriceModel"
    assert pointer["model_version"] == promotion["model_version"]
    assert (models_dir / f"{result['best_model_name']}.joblib").is_file()


def test_real_development_artifacts_untouched():
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass
