"""Step 66 (I-16) focused tests: scheduler forwards production overrides.

Proves RetrainingScheduler.run_once() forwards configured
production_model and models_dir to RetrainingService.retrain(),
preserves defaults when omitted, and returns the result unchanged.
Stub services only; no real retraining, database, MLflow, or models.
"""

from pathlib import Path
from unittest.mock import MagicMock

import inspect
import pytest

from app.core.exceptions import RetrainingError
from app.scheduler.retraining_scheduler import RetrainingScheduler
from app.use_cases.house_price.use_case import HousePriceUseCase


def _stub_service(result=None):
    service = MagicMock()
    service.retrain = MagicMock(return_value=result if result is not None else {"ok": True})
    return service


def test_explicit_production_model_reaches_service():
    sentinel = object()
    service = _stub_service()
    scheduler = RetrainingScheduler(
        retraining_service=service,
        use_case=HousePriceUseCase(),
        data_path="d.csv",
        production_model=sentinel,
    )
    assert scheduler.production_model is sentinel
    scheduler.run_once(promote=True)
    _, kwargs = service.retrain.call_args
    assert kwargs["production_model"] is sentinel


def test_explicit_models_dir_reaches_service(tmp_path: Path):
    service = _stub_service()
    scheduler = RetrainingScheduler(
        retraining_service=service,
        use_case=HousePriceUseCase(),
        data_path="d.csv",
        models_dir=str(tmp_path / "models"),
    )
    assert scheduler.models_dir == str(tmp_path / "models")
    scheduler.run_once(promote=False)
    _, kwargs = service.retrain.call_args
    assert kwargs["models_dir"] == str(tmp_path / "models")


def test_models_dir_path_object_forwarded():
    service = _stub_service()
    models_dir = Path("some/models")
    scheduler = RetrainingScheduler(
        retraining_service=service,
        use_case=HousePriceUseCase(),
        data_path="d.csv",
        models_dir=models_dir,
    )
    scheduler.run_once(promote=True)
    _, kwargs = service.retrain.call_args
    assert kwargs["models_dir"] is models_dir


def test_omitted_overrides_preserve_defaults():
    service = _stub_service()
    scheduler = RetrainingScheduler(
        retraining_service=service,
        use_case=HousePriceUseCase(),
        data_path="d.csv",
    )
    assert scheduler.production_model is None
    assert scheduler.models_dir is None
    scheduler.run_once(promote=True)
    _, kwargs = service.retrain.call_args
    assert kwargs["production_model"] is None
    assert kwargs["models_dir"] is None
    # Legacy keys still forwarded unchanged.
    assert kwargs["promote"] is True
    assert kwargs["data_path"] == "d.csv"


def test_run_once_returns_result_unchanged_with_overrides():
    result = {"use_case": "house_price", "marker": object()}
    service = _stub_service(result=result)
    scheduler = RetrainingScheduler(
        retraining_service=service,
        use_case=HousePriceUseCase(),
        data_path="d.csv",
        production_model=object(),
        models_dir="m",
    )
    assert scheduler.run_once(promote=False) is result


def test_invalid_models_dir_rejected():
    with pytest.raises(RetrainingError):
        RetrainingScheduler(
            retraining_service=_stub_service(),
            use_case=HousePriceUseCase(),
            data_path="d.csv",
            models_dir=12345,
        )
    with pytest.raises(RetrainingError):
        RetrainingScheduler(
            retraining_service=_stub_service(),
            use_case=HousePriceUseCase(),
            data_path="d.csv",
            models_dir="   ",
        )


def test_scheduler_default_matches_service_default_no_implicit_promote():
    # Step 67 (I-17): both layers default to promote=False (safe).
    from app.retraining.retraining_service import RetrainingService

    assert (
        inspect.signature(RetrainingScheduler.run_once).parameters["promote"].default
        is False
    )
    assert (
        inspect.signature(RetrainingService.retrain).parameters["promote"].default
        is False
    )
    service = _stub_service()
    scheduler = RetrainingScheduler(
        retraining_service=service,
        use_case=HousePriceUseCase(),
        data_path="d.csv",
    )
    scheduler.run_once()
    _, kwargs = service.retrain.call_args
    assert kwargs["promote"] is False
