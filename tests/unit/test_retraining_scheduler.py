"""Foundation tests for RetrainingScheduler (Step 37).

Uses mocks/stubs only. Never runs real retraining, never touches the
real database, the real MLflow registry, or production model artifacts.
"""

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml

from app.core.config import get_project_root, load_config
from app.core.exceptions import AdaptiveAIError, ConfigurationError, RetrainingError
from app.scheduler import RetrainingScheduler
from app.scheduler.retraining_scheduler import (
    DEFAULT_INTERVAL_MINUTES,
    parse_scheduler_config,
    resolve_scheduler_config,
)
from app.use_cases.house_price.use_case import HousePriceUseCase


def _stub_service(result=None):
    service = MagicMock()
    service.retrain = MagicMock(return_value=result if result is not None else {"ok": True})
    return service


def _scheduler(service=None, use_case=None, data_path="data/reference/house_price_reference.csv", **kwargs):
    return RetrainingScheduler(
        retraining_service=service if service is not None else _stub_service(),
        use_case=use_case if use_case is not None else HousePriceUseCase(),
        data_path=data_path,
        **kwargs,
    )


def _scheduler_source():
    return (Path(get_project_root()) / "app" / "scheduler" / "retraining_scheduler.py").read_text(
        encoding="utf-8"
    )


def _init_source():
    return (Path(get_project_root()) / "app" / "scheduler" / "__init__.py").read_text(
        encoding="utf-8"
    )


# --- 1. Constructor dependency injection ---


def test_constructor_stores_injected_dependencies():
    service = _stub_service()
    use_case = HousePriceUseCase()
    scheduler = RetrainingScheduler(
        retraining_service=service,
        use_case=use_case,
        data_path="some/path.csv",
        enabled=False,
        interval_minutes=60,
    )
    assert scheduler.retraining_service is service
    assert scheduler.use_case is use_case
    assert scheduler.data_path == "some/path.csv"
    assert scheduler.enabled is False
    assert scheduler.interval_minutes == 60


def test_constructor_rejects_missing_retrain():
    with pytest.raises(RetrainingError):
        RetrainingScheduler(
            retraining_service=object(),
            use_case=HousePriceUseCase(),
            data_path="data/x.csv",
        )
    with pytest.raises(RetrainingError):
        RetrainingScheduler(
            retraining_service=None,
            use_case=HousePriceUseCase(),
            data_path="data/x.csv",
        )


def test_constructor_rejects_invalid_use_case_and_data_path():
    with pytest.raises(RetrainingError):
        RetrainingScheduler(
            retraining_service=_stub_service(),
            use_case="not-a-use-case",
            data_path="data/x.csv",
        )
    with pytest.raises(RetrainingError):
        RetrainingScheduler(
            retraining_service=_stub_service(),
            use_case=HousePriceUseCase(),
            data_path="",
        )
    with pytest.raises(RetrainingError):
        RetrainingScheduler(
            retraining_service=_stub_service(),
            use_case=HousePriceUseCase(),
            data_path=12345,
        )


def test_package_exports_scheduler():
    import app.scheduler as scheduler_pkg

    assert scheduler_pkg.RetrainingScheduler is RetrainingScheduler
    assert "RetrainingScheduler" in scheduler_pkg.__all__
    assert "RetrainingScheduler" in _init_source()


# --- 2-6. run_once delegation ---


def test_run_once_calls_service_exactly_once():
    service = _stub_service(result={"best": 1})
    scheduler = _scheduler(service=service)
    scheduler.run_once(promote=True)
    assert service.retrain.call_count == 1


def test_run_once_passes_correct_use_case_and_data_path():
    service = _stub_service()
    use_case = HousePriceUseCase()
    scheduler = _scheduler(service=service, use_case=use_case, data_path="custom/data.csv")
    scheduler.run_once(promote=True)
    _, kwargs = service.retrain.call_args
    assert kwargs["use_case"] is use_case
    assert kwargs["data_path"] == "custom/data.csv"


def test_run_once_propagates_promote_flag():
    for flag in (True, False):
        service = _stub_service()
        scheduler = _scheduler(service=service)
        scheduler.run_once(promote=flag)
        _, kwargs = service.retrain.call_args
        assert kwargs["promote"] is flag


def test_run_once_default_promote_is_false():
    # Step 67 (I-17): the scheduler must NOT implicitly promote; the
    # default matches RetrainingService.retrain(promote=False).
    sig = inspect.signature(RetrainingScheduler.run_once)
    assert sig.parameters["promote"].default is False
    service = _stub_service()
    scheduler = _scheduler(service=service)
    scheduler.run_once()
    _, kwargs = service.retrain.call_args
    assert kwargs["promote"] is False


def test_run_once_returns_exact_service_result_without_reconstruction():
    result = {"use_case": "house_price", "best_model_name": "X", "marker": object()}
    service = _stub_service(result=result)
    scheduler = _scheduler(service=service)
    out = scheduler.run_once(promote=False)
    assert out is result


def test_run_once_rejects_non_boolean_promote():
    scheduler = _scheduler()
    with pytest.raises(RetrainingError):
        scheduler.run_once(promote="yes")
    with pytest.raises(RetrainingError):
        scheduler.run_once(promote=1)


# --- 7-8. Safety / error behavior ---


def test_retraining_error_propagates_unchanged():
    original = RetrainingError("underlying boom")
    service = MagicMock()
    service.retrain = MagicMock(side_effect=original)
    scheduler = _scheduler(service=service)
    with pytest.raises(RetrainingError) as exc_info:
        scheduler.run_once(promote=True)
    assert exc_info.value is original


def test_unexpected_exception_wrapped_in_retraining_error_with_cause():
    service = MagicMock()
    service.retrain = MagicMock(side_effect=ValueError("unexpected boom"))
    scheduler = _scheduler(service=service)
    with pytest.raises(RetrainingError, match="Scheduled retraining failed") as exc_info:
        scheduler.run_once(promote=True)
    assert isinstance(exc_info.value.__cause__, ValueError)


def test_unexpected_adaptive_ai_error_wrapped():
    service = MagicMock()
    service.retrain = MagicMock(side_effect=AdaptiveAIError("infra boom"))
    scheduler = _scheduler(service=service)
    with pytest.raises(RetrainingError):
        scheduler.run_once(promote=True)


def test_failed_run_never_reports_success():
    service = MagicMock()
    service.retrain = MagicMock(side_effect=RetrainingError("nope"))
    scheduler = _scheduler(service=service)
    with pytest.raises(RetrainingError):
        scheduler.run_once()
    # No fallback result, no swallowed error.


# --- 9-11. No direct infra calls ---


def test_no_direct_mlflow_calls():
    source = _scheduler_source()
    for forbidden in (
        "mlflow",
        "Mlflow",
        "track_candidate",
        "register_model",
        "ensure_registered_model",
        "create_model_version",
    ):
        assert forbidden not in source


def test_no_direct_database_calls():
    source = _scheduler_source()
    for forbidden in (
        "import sqlite3",
        "sqlite3",
        "SELECT ",
        "DatabaseManager",
        "PromotionRepository",
        "MonitoringRepository",
        "PredictionRepository",
        "save_promotion_result",
        "save_monitoring_result",
        "get_connection",
    ):
        assert forbidden not in source


def test_no_direct_model_persistence():
    source = _scheduler_source()
    for forbidden in (
        "joblib",
        "save_house_price_model",
        "load_house_price_model",
        "ModelPromoter",
        "model_promoter",
    ):
        assert forbidden not in source


def test_no_monitoring_drift_duplication():
    source = _scheduler_source()
    for forbidden in ("DriftDetector", "MonitoringService", "detect(", "get_monitoring_data"):
        assert forbidden not in source


# --- 12-13. No automatic background execution ---


def test_construction_does_not_trigger_retraining():
    service = _stub_service()
    _scheduler(service=service, enabled=False)
    service.retrain.assert_not_called()
    service2 = _stub_service()
    _scheduler(service=service2, enabled=True)
    service2.retrain.assert_not_called()


def test_import_does_not_trigger_retraining():
    # Importing the module must not call any service (no module-level run).
    import app.scheduler.retraining_scheduler as module

    assert not hasattr(module, "scheduler")
    assert not hasattr(module, "default_scheduler")
    assert not hasattr(module, "global_scheduler")
    for name in dir(module):
        assert "singleton" not in name.lower()


def test_enabled_false_does_not_cause_automatic_execution():
    service = _stub_service()
    scheduler = _scheduler(service=service, enabled=False)
    assert scheduler.enabled is False
    service.retrain.assert_not_called()
    # Explicit run_once still works when disabled (explicit demand only).
    result = {"explicit": True}
    service.retrain = MagicMock(return_value=result)
    out = scheduler.run_once(promote=True)
    assert out is result
    assert service.retrain.call_count == 1


def test_no_background_execution_primitives():
    source = _scheduler_source()
    for forbidden in (
        "threading",
        "Thread(",
        "sleep(",
        "while True",
        "while 1",
        "APScheduler",
        "BackgroundScheduler",
        "BlockingScheduler",
        "Celery",
        "celery",
        "daemon",
        "asyncio",
        "time.sleep",
    ):
        assert forbidden not in source


# --- 14-15. Configuration validation ---


@pytest.mark.parametrize("bad", [0, -1, -60, "60", 1.5, True, False, None, [60], {"v": 60}])
def test_interval_validation_rejects_non_positive_int(bad):
    with pytest.raises(ConfigurationError):
        _scheduler(interval_minutes=bad)


@pytest.mark.parametrize("good", [1, 5, 60, 1440])
def test_interval_validation_accepts_positive_int(good):
    scheduler = _scheduler(interval_minutes=good)
    assert scheduler.interval_minutes == good


@pytest.mark.parametrize("bad", ["false", 0, 1, None, "yes"])
def test_enabled_must_be_boolean(bad):
    with pytest.raises(ConfigurationError):
        _scheduler(enabled=bad)


def test_invalid_scheduler_configuration_rejected():
    with pytest.raises(ConfigurationError):
        parse_scheduler_config({"enabled": "yes", "interval_minutes": 60})
    with pytest.raises(ConfigurationError):
        parse_scheduler_config({"enabled": False, "interval_minutes": 0})
    with pytest.raises(ConfigurationError):
        parse_scheduler_config({"enabled": False, "interval_minutes": -5})
    with pytest.raises(ConfigurationError):
        parse_scheduler_config({"enabled": False, "interval_minutes": "60"})
    with pytest.raises(ConfigurationError):
        parse_scheduler_config({"enabled": False, "interval_minutes": True})
    with pytest.raises(ConfigurationError):
        parse_scheduler_config("not-a-dict")


def test_parse_and_resolve_defaults():
    assert parse_scheduler_config(None) == {"enabled": False, "interval_minutes": 60}
    assert parse_scheduler_config({}) == {"enabled": False, "interval_minutes": 60}
    assert resolve_scheduler_config(None) == {"enabled": False, "interval_minutes": 60}
    assert resolve_scheduler_config({}) == {"enabled": False, "interval_minutes": 60}
    assert resolve_scheduler_config({"scheduler": {"enabled": True, "interval_minutes": 15}}) == {
        "enabled": True,
        "interval_minutes": 15,
    }
    with pytest.raises(ConfigurationError):
        resolve_scheduler_config("not-a-dict")


def test_from_config_builds_scheduler():
    service = _stub_service()
    scheduler = RetrainingScheduler.from_config(
        retraining_service=service,
        use_case=HousePriceUseCase(),
        data_path="d.csv",
        scheduler_config={"enabled": True, "interval_minutes": 30},
    )
    assert scheduler.enabled is True
    assert scheduler.interval_minutes == 30
    assert scheduler.data_path == "d.csv"
    with pytest.raises(ConfigurationError):
        RetrainingScheduler.from_config(
            retraining_service=service,
            use_case=HousePriceUseCase(),
            scheduler_config={"enabled": False, "interval_minutes": 0},
        )


def test_platform_yaml_has_valid_scheduler_section():
    config = load_config()
    assert "scheduler" in config
    assert config["scheduler"]["enabled"] is False
    assert config["scheduler"]["interval_minutes"] == DEFAULT_INTERVAL_MINUTES
    parsed = resolve_scheduler_config(config)
    assert parsed == {"enabled": False, "interval_minutes": DEFAULT_INTERVAL_MINUTES}


def test_temporary_config_file_validated(tmp_path: Path):
    good = {
        "application": {"name": "AdaptiveAI", "environment": "test"},
        "random_seed": 42,
        "logging": {"level": "INFO"},
        "scheduler": {"enabled": True, "interval_minutes": 10},
    }
    good_path = tmp_path / "good.yaml"
    good_path.write_text(yaml.safe_dump(good), encoding="utf-8")
    # Absolute paths pass through os.path.join in load_config.
    loaded = load_config(str(good_path))
    assert resolve_scheduler_config(loaded) == {"enabled": True, "interval_minutes": 10}

    bad = {
        "application": {"name": "AdaptiveAI", "environment": "test"},
        "random_seed": 42,
        "logging": {"level": "INFO"},
        "scheduler": {"enabled": "yes", "interval_minutes": -3},
    }
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump(bad), encoding="utf-8")
    with pytest.raises((ValueError, ConfigurationError)):
        # Config-level validation raises ValueError; scheduler-level raises ConfigurationError.
        try:
            bad_loaded = load_config(str(bad_path))
        except ValueError:
            raise
        parse_scheduler_config(bad_loaded.get("scheduler"))


# --- 16. get_status ---


def test_get_status_shape_and_values():
    scheduler = _scheduler(enabled=False, interval_minutes=60)
    status = scheduler.get_status()
    assert status == {
        "enabled": False,
        "interval_minutes": 60,
        "running": False,
        "last_run": None,
    }


def test_get_status_is_side_effect_free():
    service = _stub_service()
    scheduler = _scheduler(service=service, enabled=True, interval_minutes=15)
    first = scheduler.get_status()
    second = scheduler.get_status()
    assert first == second
    assert first is not second
    service.retrain.assert_not_called()
    assert first["running"] is False
    assert first["last_run"] is None


def test_get_status_does_not_pretend_a_run_occurred():
    service = _stub_service(result={"x": 1})
    scheduler = _scheduler(service=service)
    assert scheduler.get_status()["last_run"] is None
    scheduler.run_once()
    # Foundation step has no run-history tracking; must still not fabricate.
    assert scheduler.get_status()["last_run"] is None
    assert scheduler.get_status()["running"] is False


# --- 17. Repeated explicit runs ---


def test_repeated_run_once_executes_independently():
    service = MagicMock()
    service.retrain = MagicMock(side_effect=[{"n": 1}, {"n": 2}])
    scheduler = _scheduler(service=service)
    first = scheduler.run_once(promote=True)
    second = scheduler.run_once(promote=False)
    assert service.retrain.call_count == 2
    assert first == {"n": 1}
    assert second == {"n": 2}
    promotes = [call.kwargs["promote"] for call in service.retrain.call_args_list]
    assert promotes == [True, False]


# --- 18. No global singleton ---


def test_no_global_singleton_scheduler():
    import app.scheduler.retraining_scheduler as module
    import app.scheduler as pkg

    for container in (module, pkg):
        for attr in dir(container):
            value = getattr(container, attr)
            if isinstance(value, RetrainingScheduler):
                raise AssertionError(f"Global scheduler instance found: {attr}")
    first = _scheduler()
    second = _scheduler()
    assert first is not second
    assert first.retraining_service is not second.retraining_service


# --- 19-21. No API/dashboard/heavy deps ---


def test_no_api_imports():
    source = _scheduler_source()
    for forbidden in ("fastapi", "FastAPI(", "APIRouter", "from app.api", "import app.api"):
        assert forbidden not in source


def test_no_dashboard_imports():
    source = _scheduler_source()
    for forbidden in ("streamlit", "Streamlit", "from dashboard", "import dashboard", "plotly", "dash"):
        assert forbidden not in source


def test_no_apscheduler_celery_dependency():
    source = _scheduler_source()
    for forbidden in ("APScheduler", "apscheduler", "Celery", "celery"):
        assert forbidden not in source
    requirements = (Path(get_project_root()) / "requirements.txt").read_text(encoding="utf-8").lower()
    assert "apscheduler" not in requirements
    assert "celery" not in requirements


def test_init_exports_do_not_pull_api_or_dashboard():
    source = _init_source()
    for forbidden in ("fastapi", "streamlit", "APScheduler", "Celery", "sqlite3", "mlflow"):
        assert forbidden not in source


# --- No real side effects ---


def test_scheduler_tests_create_no_real_artifacts(tmp_path: Path):
    service = _stub_service(result={"ok": True})
    scheduler = _scheduler(service=service)
    scheduler.run_once()
    scheduler.get_status()
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass
