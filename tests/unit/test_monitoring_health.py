"""Phase 4 tests: health-check service, persistence, and provider data.

Verifies the additive health path (numeric + categorical + quality +
severity) on isolated temporary databases: healthy and insufficient
windows, malformed-row tolerance, enriched persistence round-trip,
legacy detector rows reading back with null enrichment, and the
prediction counter. Never touches production state.
"""

from pathlib import Path

import pandas as pd
import pytest

from app.core.config import resolve_project_path
from app.database.database import DatabaseManager
from app.database.monitoring_repository import MonitoringRepository
from app.database.repository import PredictionRepository
from app.monitoring.data_provider import create_house_price_provider
from app.monitoring.monitoring_service import create_house_price_monitoring_service


def _tmp_repos(tmp_path: Path, name: str = "health.db"):
    manager = DatabaseManager(db_path=tmp_path / name)
    manager.init_db()
    return PredictionRepository(manager), MonitoringRepository(manager)


def _seed_from_reference(repo: PredictionRepository, n: int = 60, seed: int = 7) -> None:
    ref = pd.read_csv(resolve_project_path("data/reference/house_price_reference.csv"))
    cols = [c for c in ref.columns if c != "price"]
    sample = ref.sample(n=n, random_state=seed).reset_index(drop=True)
    for i in range(n):
        row = {c: sample.loc[i, c] for c in cols}
        for k in ("bedrooms", "bathrooms", "parking", "floor_number", "total_floors"):
            row[k] = int(row[k])
        repo.save_prediction(
            use_case="house_price", model_name="M", input_data=row, prediction=1e7
        )


def test_healthy_window_summary(tmp_path: Path):
    pred_repo, mon_repo = _tmp_repos(tmp_path)
    _seed_from_reference(pred_repo)
    service = create_house_price_monitoring_service(
        repository=pred_repo, monitoring_repository=mon_repo
    )
    summary = service.run_health_check(
        serving_info={"model_name": "LinearRegression", "model_version": "311"},
        total_predictions=pred_repo.count_predictions("house_price"),
    )
    assert summary["status"] == "ok"
    assert summary["overall_health"] == "healthy"
    assert summary["severity"] == "healthy"
    assert summary["data_quality_status"] == "pass"
    assert summary["overall_drifted"] is False
    assert summary["drifted_count"] == 0
    assert summary["total_monitored"] == 10
    assert summary["production_rows"] == 60
    assert summary["total_predictions"] == 60
    assert summary["model_name"] == "LinearRegression"
    assert summary["model_version"] == "311"
    assert summary["schema_version"] == 2
    assert isinstance(summary["timestamp"], str) and summary["timestamp"]
    assert len(summary["features"]) == 10
    assert {f["feature_name"] for f in summary["features"]} == {
        "area_sqft", "bedrooms", "bathrooms", "parking", "location",
        "property_age", "floor_number", "total_floors", "furnishing_status",
        "property_type",
    }
    assert "monitoring_run_id" in summary
    stored = mon_repo.get_monitoring_result_by_id(summary["monitoring_run_id"])
    assert stored is not None
    assert stored["severity"] == "healthy"
    assert stored["overall_health"] == "healthy"
    assert stored["model_name"] == "LinearRegression"
    assert stored["schema_version"] == 2
    assert stored["drifted_features"] == 0


def test_empty_window_is_unknown(tmp_path: Path):
    pred_repo, mon_repo = _tmp_repos(tmp_path, "empty.db")
    service = create_house_price_monitoring_service(
        repository=pred_repo, monitoring_repository=mon_repo
    )
    summary = service.run_health_check()
    assert summary["status"] == "insufficient_data"
    assert summary["overall_health"] == "unknown"
    assert summary["severity"] == "unknown"
    assert summary["data_quality_status"] == "unknown"
    assert summary["model_name"] is None
    assert summary["production_rows"] == 0


def test_malformed_rows_counted_not_raised(tmp_path: Path):
    pred_repo, mon_repo = _tmp_repos(tmp_path, "malformed.db")
    _seed_from_reference(pred_repo, n=35)
    manager = pred_repo._db
    conn = manager.get_connection()
    try:
        conn.execute(
            "INSERT INTO predictions (timestamp, use_case, model_name, model_version, input_data, prediction)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            ("2026-01-01T00:00:00+00:00", "house_price", "M", None, "not-json", 1.0),
        )
        conn.commit()
    finally:
        DatabaseManager.close_connection(conn)
    service = create_house_price_monitoring_service(
        repository=pred_repo, monitoring_repository=mon_repo
    )
    summary = service.run_health_check()
    assert summary["status"] == "ok"
    assert summary["window_malformed_records"] == 1
    assert summary["data_quality"]["malformed_rows"] == 1
    assert summary["data_quality_status"] == "fail"


def test_shifted_window_detects_categorical_drift(tmp_path: Path):
    pred_repo, mon_repo = _tmp_repos(tmp_path, "shifted.db")
    _seed_from_reference(pred_repo, n=60)
    # Concentrate the newest 60 rows on one locality/type.
    ref = pd.read_csv(resolve_project_path("data/reference/house_price_reference.csv"))
    cols = [c for c in ref.columns if c != "price"]
    sample = ref.sample(n=60, random_state=3).reset_index(drop=True)
    for i in range(60):
        row = {c: sample.loc[i, c] for c in cols}
        for k in ("bedrooms", "bathrooms", "parking", "floor_number", "total_floors"):
            row[k] = int(row[k])
        row["location"] = "Kothrud"
        row["property_type"] = "Villa"
        pred_repo.save_prediction(
            use_case="house_price", model_name="M", input_data=row, prediction=1e7
        )
    service = create_house_price_monitoring_service(
        repository=pred_repo, monitoring_repository=mon_repo
    )
    summary = service.run_health_check(production_limit=60)
    assert summary["overall_drifted"] is True
    loc = next(f for f in summary["features"] if f["feature_name"] == "location")
    assert loc["type"] == "categorical"
    assert loc["drifted"] is True
    assert loc["severity"] in {"warning", "critical"}
    assert summary["overall_health"] in {"warning", "critical"}


def test_legacy_detector_rows_read_with_null_enrichment(tmp_path: Path):
    pred_repo, mon_repo = _tmp_repos(tmp_path, "mixed.db")
    _seed_from_reference(pred_repo, n=35)
    service = create_house_price_monitoring_service(
        repository=pred_repo, monitoring_repository=mon_repo
    )
    legacy = service.run_check()
    assert legacy["status"] == "ok"
    rows = mon_repo.list_monitoring_results(use_case="house_price")
    assert len(rows) == 1
    assert rows[0]["severity"] is None
    assert rows[0]["overall_health"] is None
    assert rows[0]["model_name"] is None
    assert rows[0]["data_quality_status"] is None


def test_full_provider_data_accounting(tmp_path: Path):
    pred_repo, _ = _tmp_repos(tmp_path, "prov.db")
    _seed_from_reference(pred_repo, n=40)
    pred_repo.save_prediction(
        use_case="house_price",
        model_name="M",
        input_data={"area": 1000.0},
        prediction=2e5,
    )
    provider = create_house_price_provider(repository=pred_repo)
    data = provider.get_full_monitoring_data(limit=100)
    assert data["total_records"] == 41
    assert data["v2_records"] == 40
    assert data["legacy_records"] == 1
    assert data["malformed_records"] == 0
    assert list(data["reference_full"].columns) == provider.feature_names
    assert list(data["production_full"].columns) == provider.feature_names
    assert provider.schema_version == 2


def test_count_predictions(tmp_path: Path):
    pred_repo, _ = _tmp_repos(tmp_path, "count.db")
    assert pred_repo.count_predictions("house_price") == 0
    _seed_from_reference(pred_repo, n=12)
    assert pred_repo.count_predictions("house_price") == 12
    assert pred_repo.count_predictions() == 12
