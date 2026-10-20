"""Step 29 tests: expose monitoring through FastAPI.

Covers POST /monitor/house-price and GET /monitor/house-price/history
using temporary databases only. Never touches data/adaptive_ai.db.
"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import app.api.main as api_main
from app.api.main import create_app
from app.core.config import get_project_root
from app.core.exceptions import AdaptiveAIError
from app.database.database import DatabaseManager
from app.database.monitoring_repository import MonitoringRepository
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_persistence import save_house_price_model
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.split import split_house_price_data

VALID_PAYLOAD = {
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


def _make_client(tmp_path: Path, db_name: str = "monitor_api.db"):
    db_manager = DatabaseManager(db_path=tmp_path / db_name)
    application = create_app(db_manager=db_manager)
    return TestClient(application), application


def _seed_predictions(repo, n: int = 50, shift: float = 0.0) -> None:
    # Seed from the synthetic population so the no-shift case matches the
    # reference distribution; shift moves area_sqft for drift cases.
    from app.use_cases.house_price.data import generate_house_price_data

    df = generate_house_price_data(n_samples=n)
    feature_cols = [c for c in df.columns if c != "price"]
    for i in range(n):
        row = {c: df.loc[i, c] for c in feature_cols}
        row["area_sqft"] = float(row["area_sqft"] + shift)
        for int_col in ("bedrooms", "bathrooms", "parking", "floor_number", "total_floors"):
            row[int_col] = int(row[int_col])
        repo.save_prediction(
            use_case="house_price",
            model_name="LinearRegression",
            input_data=row,
            prediction=400000.0,
        )


def _ok_result(**overrides):
    result = {
        "overall_drifted": False,
        "alpha": 0.05,
        "min_production_rows": 30,
        "production_rows": 50,
        "sufficient_data": True,
        "status": "ok",
        "features": [
            {"feature_name": "area_sqft", "statistic": 0.1, "p_value": 0.4, "drifted": False},
        ],
    }
    result.update(overrides)
    return result


def _insufficient_result():
    return {
        "overall_drifted": False,
        "alpha": 0.05,
        "min_production_rows": 30,
        "production_rows": 0,
        "sufficient_data": False,
        "status": "insufficient_data",
        "features": [],
    }


def _setup_model(tmp_path: Path, monkeypatch):
    df = generate_house_price_data(n_samples=200)
    X_train, _, y_train, _ = split_house_price_data(df)
    pipeline = Pipeline(
        steps=[
            ("preprocessor", get_preprocessor()),
            ("regressor", LinearRegression()),
        ]
    )
    pipeline.fit(X_train, y_train)
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_house_price_model(pipeline, "LinearRegression", models_dir=models_dir)
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODEL_NAME", "LinearRegression")
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODELS_DIR", str(models_dir))


# --- POST: success paths ---


def test_post_monitor_success_returns_expected_result(tmp_path: Path):
    client, application = _make_client(tmp_path, "success3.db")
    application.state.prediction_repo.init()
    _seed_predictions(application.state.prediction_repo, n=50)
    response = client.post("/monitor/house-price")
    assert response.status_code == 200
    body = response.json()
    for key in (
        "overall_drifted",
        "alpha",
        "min_production_rows",
        "production_rows",
        "sufficient_data",
        "status",
        "features",
    ):
        assert key in body
    assert body["status"] == "ok"
    assert body["sufficient_data"] is True
    assert body["production_rows"] == 50
    assert isinstance(body["features"], list)
    assert len(body["features"]) > 0


def test_post_monitor_creates_exactly_one_record(tmp_path: Path):
    client, application = _make_client(tmp_path, "one.db")
    application.state.prediction_repo.init()
    application.state.monitoring_repo.init()
    assert application.state.monitoring_repo.list_monitoring_results() == []
    _seed_predictions(application.state.prediction_repo, n=50)
    response = client.post("/monitor/house-price")
    assert response.status_code == 200
    stored = application.state.monitoring_repo.list_monitoring_results()
    assert len(stored) == 1


def test_post_monitor_returned_result_matches_persisted(tmp_path: Path):
    client, application = _make_client(tmp_path, "match.db")
    application.state.prediction_repo.init()
    _seed_predictions(application.state.prediction_repo, n=50)
    response = client.post("/monitor/house-price")
    assert response.status_code == 200
    body = response.json()
    stored = application.state.monitoring_repo.list_monitoring_results()
    assert len(stored) == 1
    record = stored[0]
    assert record["use_case"] == "house_price"
    for key in (
        "overall_drifted",
        "alpha",
        "min_production_rows",
        "production_rows",
        "sufficient_data",
        "status",
    ):
        assert record[key] == body[key]
    assert record["features"] == body["features"]
    assert "id" in record and "timestamp" in record


def test_post_monitor_returns_service_result_unchanged(tmp_path: Path, monkeypatch):
    client, application = _make_client(tmp_path, "passthrough.db")
    expected = _ok_result()
    fake = MagicMock()
    fake.run_check.return_value = expected
    monkeypatch.setattr(
        api_main,
        "create_house_price_monitoring_service",
        lambda **kwargs: fake,
    )
    response = client.post("/monitor/house-price")
    assert response.status_code == 200
    assert response.json() == expected
    fake.run_check.assert_called_once_with()


def test_post_monitor_insufficient_data_returns_200_and_persists(tmp_path: Path):
    client, application = _make_client(tmp_path, "insufficient.db")
    # Fresh database: no predictions seeded.
    response = client.post("/monitor/house-price")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "insufficient_data"
    assert body["sufficient_data"] is False
    assert body["overall_drifted"] is False
    assert body["features"] == []
    stored = application.state.monitoring_repo.list_monitoring_results()
    assert len(stored) == 1
    assert stored[0]["status"] == "insufficient_data"
    assert stored[0]["sufficient_data"] is False
    assert stored[0]["overall_drifted"] is False
    assert stored[0]["features"] == []


# --- POST: failure paths ---


def test_post_monitor_service_failure_returns_clean_500(tmp_path: Path, monkeypatch):
    client, application = _make_client(tmp_path, "svc_fail.db")

    def _boom_factory(**kwargs):
        fake = MagicMock()
        fake.run_check.side_effect = AdaptiveAIError(
            "SELECT * FROM predictions Traceback /tmp/secret.db boom"
        )
        return fake

    monkeypatch.setattr(
        api_main, "create_house_price_monitoring_service", _boom_factory
    )
    response = client.post("/monitor/house-price")
    assert response.status_code == 500
    body = response.json()
    assert body.get("detail") == "House Price monitoring check failed."
    text = str(body)
    assert "SELECT" not in text
    assert "Traceback" not in text
    assert "/tmp/secret.db" not in text
    assert "boom" not in text or "monitoring check failed" in text
    assert application.state.monitoring_repo.list_monitoring_results() == []


def test_post_monitor_unexpected_failure_returns_clean_500(tmp_path: Path, monkeypatch):
    client, _ = _make_client(tmp_path, "unexpected.db")

    def _boom_factory(**kwargs):
        fake = MagicMock()
        fake.run_check.side_effect = RuntimeError(
            "SELECT crash Traceback /var/data/adaptive_ai.db"
        )
        return fake

    monkeypatch.setattr(
        api_main, "create_house_price_monitoring_service", _boom_factory
    )
    response = client.post("/monitor/house-price")
    assert response.status_code == 500
    body = response.json()
    assert body.get("detail") == "House Price monitoring check failed."
    text = str(body)
    assert "SELECT" not in text
    assert "Traceback" not in text
    assert "adaptive_ai.db" not in text


def test_post_monitor_repository_failure_returns_clean_500(tmp_path: Path, monkeypatch):
    client, application = _make_client(tmp_path, "repofail.db")
    application.state.prediction_repo.init()
    _seed_predictions(application.state.prediction_repo, n=50)

    def _fail_save(*args, **kwargs):
        raise AdaptiveAIError("SELECT failed Traceback /tmp/secret.db")

    monkeypatch.setattr(
        application.state.monitoring_repo, "save_monitoring_result", _fail_save
    )
    response = client.post("/monitor/house-price")
    assert response.status_code == 500
    body = response.json()
    assert body.get("detail") == "House Price monitoring check failed."
    text = str(body)
    assert "SELECT" not in text
    assert "Traceback" not in text
    assert "secret.db" not in text


def test_post_monitor_no_internal_details_leaked(tmp_path: Path, monkeypatch):
    client, _ = _make_client(tmp_path, "leak.db")

    def _boom_factory(**kwargs):
        raise ValueError("SELECT * FROM monitoring_results at /data/adaptive_ai.db Traceback xxx")

    monkeypatch.setattr(
        api_main, "create_house_price_monitoring_service", _boom_factory
    )
    response = client.post("/monitor/house-price")
    assert response.status_code == 500
    text = str(response.json())
    assert "SELECT" not in text
    assert "Traceback" not in text
    assert "adaptive_ai.db" not in text
    assert "/data/" not in text


# --- GET: history ---


def test_get_history_empty(tmp_path: Path):
    client, _ = _make_client(tmp_path, "empty_hist.db")
    response = client.get("/monitor/house-price/history")
    assert response.status_code == 200
    assert response.json() == {"items": [], "limit": 100, "offset": 0}


def test_get_history_empty_respects_limit_offset(tmp_path: Path):
    client, _ = _make_client(tmp_path, "empty_lo.db")
    response = client.get("/monitor/house-price/history?limit=5&offset=2")
    assert response.status_code == 200
    assert response.json() == {"items": [], "limit": 5, "offset": 2}


def test_get_history_returns_persisted_results(tmp_path: Path):
    client, application = _make_client(tmp_path, "hist.db")
    repo: MonitoringRepository = application.state.monitoring_repo
    repo.init()
    repo.save_monitoring_result(use_case="house_price", result=_ok_result())
    repo.save_monitoring_result(use_case="house_price", result=_insufficient_result())
    response = client.get("/monitor/house-price/history")
    assert response.status_code == 200
    body = response.json()
    assert body["limit"] == 100
    assert body["offset"] == 0
    assert len(body["items"]) == 2
    for item in body["items"]:
        for key in (
            "id",
            "timestamp",
            "use_case",
            "overall_drifted",
            "status",
            "alpha",
            "min_production_rows",
            "production_rows",
            "sufficient_data",
            "features",
        ):
            assert key in item
        assert item["use_case"] == "house_price"
        # Phase 4 additive enrichment: detector-only rows carry the new
        # keys with null values (never fabricated).
        for key in (
            "model_name",
            "model_version",
            "schema_version",
            "severity",
            "overall_health",
            "data_quality_status",
            "drifted_features",
        ):
            assert key in item
            assert item[key] is None


def test_get_history_ordering_follows_repository(tmp_path: Path):
    client, application = _make_client(tmp_path, "order.db")
    repo = application.state.monitoring_repo
    repo.init()
    repo.save_monitoring_result(use_case="house_price", result=_ok_result())
    repo.save_monitoring_result(use_case="house_price", result=_insufficient_result())
    repo.save_monitoring_result(use_case="house_price", result=_ok_result(production_rows=60))
    expected = repo.list_monitoring_results(limit=100, offset=0, use_case="house_price")
    response = client.get("/monitor/house-price/history")
    assert response.status_code == 200
    items = response.json()["items"]
    assert [i["id"] for i in items] == [r["id"] for r in expected]
    # Step 63 (I-14): newest-first.
    assert [i["id"] for i in items] == [3, 2, 1]


def test_get_history_limit_works(tmp_path: Path):
    client, application = _make_client(tmp_path, "limit.db")
    repo = application.state.monitoring_repo
    repo.init()
    for _ in range(5):
        repo.save_monitoring_result(use_case="house_price", result=_ok_result())
    response = client.get("/monitor/house-price/history?limit=2")
    assert response.status_code == 200
    body = response.json()
    assert body["limit"] == 2
    assert body["offset"] == 0
    assert len(body["items"]) == 2
    # Step 63 (I-14): limit paginates from the newest record.
    assert [i["id"] for i in body["items"]] == [5, 4]


def test_get_history_offset_works(tmp_path: Path):
    client, application = _make_client(tmp_path, "offset.db")
    repo = application.state.monitoring_repo
    repo.init()
    for _ in range(5):
        repo.save_monitoring_result(use_case="house_price", result=_ok_result())
    response = client.get("/monitor/house-price/history?offset=2")
    assert response.status_code == 200
    body = response.json()
    assert body["offset"] == 2
    assert len(body["items"]) == 3
    # Step 63 (I-14): offset paginates from the newest record.
    assert [i["id"] for i in body["items"]] == [3, 2, 1]

    response = client.get("/monitor/house-price/history?limit=2&offset=1")
    assert response.status_code == 200
    body = response.json()
    assert body["limit"] == 2
    assert body["offset"] == 1
    assert [i["id"] for i in body["items"]] == [4, 3]


def test_get_history_invalid_limit_rejected(tmp_path: Path):
    client, _ = _make_client(tmp_path, "badlimit.db")
    assert client.get("/monitor/house-price/history?limit=0").status_code == 422
    assert client.get("/monitor/house-price/history?limit=-1").status_code == 422
    assert client.get("/monitor/house-price/history?limit=abc").status_code == 422


def test_get_history_invalid_offset_rejected(tmp_path: Path):
    client, _ = _make_client(tmp_path, "badoffset.db")
    assert client.get("/monitor/house-price/history?offset=-1").status_code == 422
    assert client.get("/monitor/house-price/history?offset=abc").status_code == 422


def test_get_history_repository_failure_returns_clean_500(tmp_path: Path, monkeypatch):
    client, application = _make_client(tmp_path, "histfail.db")

    def _fail_list(*args, **kwargs):
        raise AdaptiveAIError("SELECT failed Traceback /tmp/secret.db")

    monkeypatch.setattr(
        application.state.monitoring_repo, "list_monitoring_results", _fail_list
    )
    response = client.get("/monitor/house-price/history")
    assert response.status_code == 500
    body = response.json()
    assert body.get("detail") == "Failed to retrieve monitoring history."
    text = str(body)
    assert "SELECT" not in text
    assert "Traceback" not in text
    assert "secret.db" not in text


def test_get_history_unexpected_failure_returns_clean_500(tmp_path: Path, monkeypatch):
    client, application = _make_client(tmp_path, "histfail2.db")

    def _fail_list(*args, **kwargs):
        raise RuntimeError("SELECT crash at /data/adaptive_ai.db Traceback")

    monkeypatch.setattr(
        application.state.monitoring_repo, "list_monitoring_results", _fail_list
    )
    response = client.get("/monitor/house-price/history")
    assert response.status_code == 500
    text = str(response.json())
    assert "SELECT" not in text
    assert "Traceback" not in text
    assert "adaptive_ai.db" not in text


# --- Existing behavior unchanged ---


def test_health_behavior_unchanged(tmp_path: Path):
    client, _ = _make_client(tmp_path, "health.db")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_predict_behavior_unchanged(tmp_path: Path, monkeypatch):
    _setup_model(tmp_path, monkeypatch)
    client, application = _make_client(tmp_path, "predict.db")
    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    assert response.status_code == 200
    assert "prediction" in response.json()
    bad = dict(VALID_PAYLOAD, area_sqft=-5)
    assert client.post("/predict/house-price", json=bad).status_code == 422


def test_prediction_history_behavior_unchanged(tmp_path: Path, monkeypatch):
    _setup_model(tmp_path, monkeypatch)
    client, application = _make_client(tmp_path, "predhist.db")
    application.state.prediction_repo.init()
    application.state.monitoring_repo.init()
    assert application.state.prediction_repo.list_predictions() == []
    response = client.post("/predict/house-price", json=VALID_PAYLOAD)
    assert response.status_code == 200
    records = application.state.prediction_repo.list_predictions()
    assert len(records) == 1
    assert records[0]["use_case"] == "house_price"
    # Monitoring is manual only: prediction must not create monitoring results.
    assert application.state.monitoring_repo.list_monitoring_results() == []


def test_monitoring_post_then_history_integration(tmp_path: Path):
    client, application = _make_client(tmp_path, "integration.db")
    application.state.prediction_repo.init()
    _seed_predictions(application.state.prediction_repo, n=50)
    post = client.post("/monitor/house-price")
    assert post.status_code == 200
    history = client.get("/monitor/house-price/history")
    assert history.status_code == 200
    body = history.json()
    assert len(body["items"]) == 1
    assert body["items"][0]["overall_drifted"] == post.json()["overall_drifted"]
    assert body["items"][0]["features"] == post.json()["features"]


def test_api_layer_contains_no_sql_or_drift_logic():
    source = Path(api_main.__file__).read_text(encoding="utf-8")
    assert "SELECT " not in source
    assert "import sqlite3" not in source
    assert "ks_2samp" not in source
    assert "scipy" not in source
    # API must not decide drift per-feature; it only orchestrates services.
    assert '["drifted"]' not in source
    assert "['drifted']" not in source


def test_real_development_database_untouched():
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass
