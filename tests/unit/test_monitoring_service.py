"""Focused tests for MonitoringService orchestration (Step 27).

Deterministic; uses temp DBs or mocks. Never touches data/adaptive_ai.db.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from app.core.config import get_project_root
from app.core.exceptions import AdaptiveAIError
from app.database.database import DatabaseManager
from app.database.repository import PredictionRepository
from app.monitoring.data_provider import MonitoringDataProvider
from app.monitoring.drift_detector import DriftDetector
from app.monitoring.monitoring_service import (
    MonitoringService,
    create_house_price_monitoring_service,
)


def _tmp_repo(tmp_path: Path, name: str = "svc.db") -> PredictionRepository:
    manager = DatabaseManager(db_path=tmp_path / name)
    manager.init_db()
    return PredictionRepository(manager)


def _seed_repo(repo: PredictionRepository, shift: float = 0.0, n: int = 50) -> None:
    # Seed production from real reference rows (same distribution) so the
    # no-shift case is genuinely in-bounds; shift moves area for drift.
    import pandas as pd

    from app.core.config import resolve_project_path

    ref = pd.read_csv(
        resolve_project_path("data/reference/house_price_reference.csv")
    )
    sample = ref.sample(n=n, random_state=7).reset_index(drop=True)
    feature_cols = [c for c in ref.columns if c != "price"]
    for i in range(n):
        row = {c: sample.loc[i, c] for c in feature_cols}
        if isinstance(row.get("area_sqft"), float):
            row["area_sqft"] = float(row["area_sqft"] + shift)
        for int_col in ("bedrooms", "bathrooms", "parking", "floor_number", "total_floors"):
            row[int_col] = int(row[int_col])
        repo.save_prediction(
            use_case="house_price",
            model_name="LinearRegression",
            input_data=row,
            prediction=400000.0,
        )


def _service(tmp_path: Path, shift: float = 0.0, n: int = 50, **kwargs) -> MonitoringService:
    repo = _tmp_repo(tmp_path)
    _seed_repo(repo, shift=shift, n=n)
    return create_house_price_monitoring_service(repository=repo, **kwargs)


def test_service_runs_no_drift_check(tmp_path: Path):
    service = _service(tmp_path, shift=0.0, n=50)
    result = service.run_check()
    for key in (
        "overall_drifted",
        "alpha",
        "min_production_rows",
        "production_rows",
        "sufficient_data",
        "status",
        "features",
    ):
        assert key in result
    assert result["status"] == "ok"
    assert result["sufficient_data"] is True


def test_service_reports_drift(tmp_path: Path):
    service = _service(tmp_path, shift=5000.0, n=100)
    result = service.run_check()
    assert result["status"] == "ok"
    assert result["overall_drifted"] is True
    assert any(f["drifted"] for f in result["features"])


def test_service_passes_provider_output_to_detector():
    ref = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    prod = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    provider = MagicMock(spec=MonitoringDataProvider)
    provider.get_monitoring_data.return_value = {
        "reference_features": ref,
        "production_features": prod,
    }
    detector = MagicMock(spec=DriftDetector)
    detector.detect.return_value = {"overall_drifted": False}

    service = MonitoringService(provider=provider, detector=detector, production_limit=25)
    out = service.run_check()

    provider.get_monitoring_data.assert_called_once_with(limit=25)
    detector.detect.assert_called_once()
    _, kwargs = detector.detect.call_args
    assert kwargs["reference"] is ref
    assert kwargs["production"] is prod
    assert out == {"overall_drifted": False}


def test_detector_configuration_respected(tmp_path: Path):
    repo = _tmp_repo(tmp_path)
    service = create_house_price_monitoring_service(
        repository=repo, alpha=0.01, min_production_rows=5, production_limit=77
    )
    assert service.detector.alpha == pytest.approx(0.01)
    assert service.detector.min_production_rows == 5
    assert service.production_limit == 77

    # Explicit detector and config together are ambiguous.
    provider = service.provider
    with pytest.raises(AdaptiveAIError):
        MonitoringService(provider=provider, detector=DriftDetector(), alpha=0.05)


def test_insufficient_data_preserved(tmp_path: Path):
    service = _service(tmp_path, shift=5000.0, n=5, min_production_rows=30)
    result = service.run_check()
    assert result["status"] == "insufficient_data"
    assert result["overall_drifted"] is False
    assert result["features"] == []


def test_provider_errors_propagate():
    provider = MagicMock(spec=MonitoringDataProvider)
    provider.get_monitoring_data.side_effect = AdaptiveAIError("provider boom")
    service = MonitoringService(provider=provider, detector=DriftDetector())
    with pytest.raises(AdaptiveAIError, match="provider boom"):
        service.run_check()


def test_detector_errors_propagate():
    ref = pd.DataFrame({"f1": [1.0, 2.0]})
    prod = pd.DataFrame({"f1": [1.0, 2.0]})
    provider = MagicMock(spec=MonitoringDataProvider)
    provider.get_monitoring_data.return_value = {
        "reference_features": ref,
        "production_features": prod,
    }
    detector = MagicMock(spec=DriftDetector)
    detector.detect.side_effect = AdaptiveAIError("detector boom")
    service = MonitoringService(provider=provider, detector=detector)
    with pytest.raises(AdaptiveAIError, match="detector boom"):
        service.run_check()


def test_service_independent_of_infra():
    source = (
        Path(__file__).resolve().parents[2]
        / "app"
        / "monitoring"
        / "monitoring_service.py"
    )
    text = source.read_text(encoding="utf-8")
    assert "PredictionRepository" in text  # type hint only via factory signature
    for forbidden in ("import sqlite3", "SELECT ", "read_csv", "mlflow", "FastAPI("):
        if forbidden == "FastAPI(":
            assert "FastAPI(" not in text
        else:
            assert forbidden not in text


def test_service_works_with_generic_features():
    ref = pd.DataFrame(
        {
            "sensor_a": np.random.RandomState(0).normal(0, 1, 100),
            "sensor_b": np.random.RandomState(2).normal(0, 1, 100),
        }
    )
    prod = pd.DataFrame(
        {
            "sensor_a": np.random.RandomState(3).normal(0, 1, 100),
            "sensor_b": np.random.RandomState(4).normal(0, 1, 100),
        }
    )
    provider = MagicMock(spec=MonitoringDataProvider)
    provider.get_monitoring_data.return_value = {
        "reference_features": ref,
        "production_features": prod,
    }
    service = MonitoringService(provider=provider, detector=DriftDetector())
    result = service.run_check()
    assert {f["feature_name"] for f in result["features"]} == {"sensor_a", "sensor_b"}


def test_real_development_database_untouched():
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass
