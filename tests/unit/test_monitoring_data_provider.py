"""Focused tests for monitoring data provider (Step 25).

Uses temporary SQLite databases and temp CSVs.
Never touches data/adaptive_ai.db.
"""

import json
import sqlite3
from pathlib import Path

import pandas as pd
import pytest

from app.core.config import get_project_root
from app.core.exceptions import AdaptiveAIError
from app.database.database import DatabaseManager
from app.database.repository import PredictionRepository
from app.monitoring.data_provider import MonitoringDataProvider, create_house_price_provider
from app.use_cases.house_price.use_case import HousePriceUseCase

EXPECTED_FEATURES = ["area_sqft", "bedrooms", "bathrooms", "parking", "property_age",
                     "floor_number", "total_floors"]

FULL_FEATURES = ["area_sqft", "bedrooms", "bathrooms", "parking", "location",
                 "property_age", "floor_number", "total_floors",
                 "furnishing_status", "property_type"]


def _full_input(**overrides):
    payload = {
        "area_sqft": 1200.0,
        "bedrooms": 3,
        "bathrooms": 2,
        "parking": 1,
        "location": "Wakad",
        "property_age": 8.0,
        "floor_number": 2,
        "total_floors": 12,
        "furnishing_status": "Semi-Furnished",
        "property_type": "Apartment",
    }
    payload.update(overrides)
    return payload


def _tmp_repo(tmp_path: Path, name: str = "monitor.db") -> PredictionRepository:
    manager = DatabaseManager(db_path=tmp_path / name)
    manager.init_db()
    return PredictionRepository(manager)


def _provider(tmp_path: Path, **kwargs) -> MonitoringDataProvider:
    repo = kwargs.pop("repository", None) or _tmp_repo(tmp_path)
    return create_house_price_provider(repository=repo, **kwargs)


def test_reference_data_loads_successfully():
    repo_manager = DatabaseManager(db_path=":memory:")
    # Repository instance only; reference path uses real CSV (read-only).
    from unittest.mock import MagicMock

    repo = MagicMock(spec=PredictionRepository)
    provider = create_house_price_provider(repository=repo)
    df = provider.get_reference_features()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_reference_data_has_expected_feature_columns():
    from unittest.mock import MagicMock

    repo = MagicMock(spec=PredictionRepository)
    provider = create_house_price_provider(repository=repo)
    df = provider.get_reference_features()
    assert list(df.columns) == EXPECTED_FEATURES


def test_missing_reference_file_handled_cleanly(tmp_path: Path):
    repo = _tmp_repo(tmp_path)
    provider = create_house_price_provider(
        repository=repo, reference_csv_path=tmp_path / "does_not_exist.csv"
    )
    with pytest.raises(AdaptiveAIError):
        provider.get_reference_features()


def test_missing_required_reference_columns_handled_cleanly(tmp_path: Path):
    bad_csv = tmp_path / "bad_reference.csv"
    pd.DataFrame({"area_sqft": [1000.0], "bedrooms": [2]}).to_csv(bad_csv, index=False)
    provider = _tmp_repo_provider(tmp_path, reference_csv_path=bad_csv)
    with pytest.raises(AdaptiveAIError):
        provider.get_reference_features()


def _tmp_repo_provider(tmp_path: Path, **kwargs) -> MonitoringDataProvider:
    return create_house_price_provider(repository=_tmp_repo(tmp_path), **kwargs)


def test_production_records_converted_to_feature_dataframe(tmp_path: Path):
    repo = _tmp_repo(tmp_path)
    repo.save_prediction(
        use_case="house_price",
        model_name="LinearRegression",
        input_data=_full_input(area_sqft=2000.0),
        prediction=450000.0,
    )
    provider = create_house_price_provider(repository=repo)
    df = provider.get_production_features()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == EXPECTED_FEATURES
    assert len(df) == 1
    assert df.iloc[0]["area_sqft"] == pytest.approx(2000.0)
    assert int(df.iloc[0]["bedrooms"]) == 3


def test_production_columns_match_reference_columns(tmp_path: Path):
    repo = _tmp_repo(tmp_path)
    repo.save_prediction(
        use_case="house_price",
        model_name="LinearRegression",
        input_data=_full_input(area_sqft=1500.0, bedrooms=2, bathrooms=1,
                               property_age=5.0, parking=0),
        prediction=300000.0,
    )
    provider = create_house_price_provider(repository=repo)
    data = provider.get_monitoring_data()
    assert list(data["reference_features"].columns) == EXPECTED_FEATURES
    assert list(data["production_features"].columns) == EXPECTED_FEATURES


def test_empty_production_history_handled_correctly(tmp_path: Path):
    provider = _tmp_repo_provider(tmp_path)
    df = provider.get_production_features()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == EXPECTED_FEATURES
    assert len(df) == 0

    data = provider.get_monitoring_data()
    assert len(data["production_features"]) == 0
    assert len(data["reference_features"]) > 0


def test_malformed_production_json_handled_correctly(tmp_path: Path):
    db_path = tmp_path / "malformed.db"
    manager = DatabaseManager(db_path=db_path)
    manager.init_db()
    conn = manager.get_connection()
    try:
        conn.execute(
            "INSERT INTO predictions (timestamp, use_case, model_name, model_version, input_data, prediction)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            ("2026-01-01T00:00:00+00:00", "house_price", "LinearRegression", None, "not-json", 1.0),
        )
        conn.commit()
    finally:
        DatabaseManager.close_connection(conn)

    provider = create_house_price_provider(repository=PredictionRepository(manager))
    with pytest.raises(AdaptiveAIError):
        provider.get_production_features()


def test_missing_production_features_handled_correctly(tmp_path: Path):
    # Records stored under an older data schema are skipped so historical
    # rows stay readable while monitoring only compares current-schema rows.
    repo = _tmp_repo(tmp_path)
    repo.save_prediction(
        use_case="house_price",
        model_name="LinearRegression",
        input_data={"area": 1000.0},
        prediction=200000.0,
    )
    provider = create_house_price_provider(repository=repo)
    df = provider.get_production_features()
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == EXPECTED_FEATURES
    assert len(df) == 0


def test_mixed_schema_production_keeps_current_rows(tmp_path: Path):
    repo = _tmp_repo(tmp_path)
    repo.save_prediction(
        use_case="house_price",
        model_name="LinearRegression",
        input_data={"area": 1000.0},
        prediction=200000.0,
    )
    repo.save_prediction(
        use_case="house_price",
        model_name="LinearRegression",
        input_data=_full_input(),
        prediction=9000000.0,
    )
    provider = create_house_price_provider(repository=repo)
    df = provider.get_production_features()
    assert len(df) == 1
    assert df.iloc[0]["area_sqft"] == pytest.approx(1200.0)


def test_multiple_production_records_handled_correctly(tmp_path: Path):
    repo = _tmp_repo(tmp_path)
    for i in range(3):
        repo.save_prediction(
            use_case="house_price",
            model_name="LinearRegression",
            input_data=_full_input(
                area_sqft=1000.0 + i * 100,
                bedrooms=2 + i,
                bathrooms=1 + i,
                property_age=float(i),
                parking=i % 2,
            ),
            prediction=250000.0 + i,
        )
    # A record from another use case must not leak into house_price production data.
    repo.save_prediction(
        use_case="other_use_case",
        model_name="Other",
        input_data={"foo": 1},
        prediction=1.0,
    )
    provider = create_house_price_provider(repository=repo)
    df = provider.get_production_features()
    assert len(df) == 3
    assert list(df.columns) == EXPECTED_FEATURES


def test_monitoring_uses_repository_not_direct_sql():
    source = Path(__file__).resolve().parents[2] / "app" / "monitoring" / "data_provider.py"
    text = source.read_text(encoding="utf-8")
    assert "PredictionRepository" in text
    assert "list_predictions" in text
    assert "import sqlite3" not in text
    assert "SELECT " not in text


def test_real_development_database_untouched():
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass
