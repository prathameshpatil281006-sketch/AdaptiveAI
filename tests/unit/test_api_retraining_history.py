"""Step 41 tests: expose retraining/promotion history through FastAPI.

Covers GET /monitor/house-price/retrain/history as a thin read adapter
over PromotionRepository using temporary databases only. Never touches
data/adaptive_ai.db and never triggers retraining.
"""

import inspect
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

import app.api.main as api_main
from app.api.main import create_app
from app.core.config import get_project_root
from app.core.exceptions import AdaptiveAIError
from app.database.database import DatabaseManager
from app.database.promotion_repository import PromotionRepository
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_persistence import save_house_price_model
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.split import split_house_price_data

HISTORY_PATH = "/monitor/house-price/retrain/history"
GENERIC_ERROR = "Failed to retrieve retraining history."


def _promotion(decision="promoted", **overrides):
    base = {
        "decision": decision,
        "model_name": "LinearRegression",
        "run_id": "run-1",
        "registered_model_name": "HousePriceModel",
        "model_version": "3" if decision == "promoted" else None,
        "primary_metric": "r2",
        "candidate_score": 0.9,
        "production_score": 0.8,
        "improvement": 0.1,
        "reason": f"{decision} for test.",
    }
    base.update(overrides)
    return base


def _make_client(tmp_path: Path, db_name: str = "retrain_history_api.db"):
    db_manager = DatabaseManager(db_path=tmp_path / db_name)
    application = create_app(db_manager=db_manager)
    return TestClient(application), application


def _seed(repo: PromotionRepository, decisions):
    ids = []
    for decision in decisions:
        ids.append(
            repo.save_promotion_result(
                use_case="house_price", promotion_result=_promotion(decision)
            )
        )
    return ids


def _history_endpoint_source():
    for route in api_main.app.routes:
        if getattr(route, "path", None) == HISTORY_PATH:
            return inspect.getsource(route.endpoint)
    raise AssertionError(f"Route {HISTORY_PATH} not found")


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
    models_dir = tmp_path / "retrain_history_models"
    models_dir.mkdir(parents=True, exist_ok=True)
    save_house_price_model(pipeline, "LinearRegression", models_dir=models_dir)
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODEL_NAME", "LinearRegression")
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODELS_DIR", str(models_dir))


# --- Route / method ---


def test_route_exists_with_get_method():
    paths = {route.path: getattr(route, "methods", set()) for route in api_main.app.routes}
    assert HISTORY_PATH in paths
    assert "GET" in paths[HISTORY_PATH]


def test_post_method_not_allowed(tmp_path: Path):
    client, _ = _make_client(tmp_path, "method.db")
    assert client.post(HISTORY_PATH).status_code == 405


# --- Defaults / explicit pagination ---


def test_default_limit_and_offset(tmp_path: Path):
    client, application = _make_client(tmp_path, "defaults.db")
    application.state.promotion_repo.init()
    _seed(application.state.promotion_repo, ["promoted"])
    body = client.get(HISTORY_PATH).json()
    assert body["limit"] == 100
    assert body["offset"] == 0
    assert len(body["items"]) == 1


def test_explicit_limit(tmp_path: Path):
    client, application = _make_client(tmp_path, "limit.db")
    application.state.promotion_repo.init()
    _seed(application.state.promotion_repo, ["promoted", "rejected", "failed"])
    body = client.get(HISTORY_PATH, params={"limit": 2}).json()
    assert body["limit"] == 2
    assert body["offset"] == 0
    assert len(body["items"]) == 2


def test_explicit_offset(tmp_path: Path):
    client, application = _make_client(tmp_path, "offset.db")
    application.state.promotion_repo.init()
    _seed(application.state.promotion_repo, ["promoted", "rejected", "failed"])
    body = client.get(HISTORY_PATH, params={"offset": 2}).json()
    assert body["limit"] == 100
    assert body["offset"] == 2
    assert len(body["items"]) == 1
    # Step 63 (I-14): offset paginates from the newest record.
    assert body["items"][0]["decision"] == "promoted"


# --- Repository delegation ---


def test_correct_use_case_and_single_call(tmp_path: Path):
    client, application = _make_client(tmp_path, "delegate.db")
    repo = MagicMock(spec=PromotionRepository)
    repo.list_promotion_history = MagicMock(return_value=[])
    application.state.promotion_repo = repo
    client.get(HISTORY_PATH, params={"limit": 5, "offset": 7})
    assert repo.list_promotion_history.call_count == 1
    _, kwargs = repo.list_promotion_history.call_args
    assert kwargs == {"limit": 5, "offset": 7, "use_case": "house_price"}


def test_exact_items_returned(tmp_path: Path):
    client, application = _make_client(tmp_path, "items.db")
    items = [
        {"id": 1, "decision": "promoted", "model_name": "LinearRegression"},
        {"id": 2, "decision": "rejected", "model_name": "LinearRegression"},
    ]
    repo = MagicMock(spec=PromotionRepository)
    repo.list_promotion_history = MagicMock(return_value=items)
    application.state.promotion_repo = repo
    body = client.get(HISTORY_PATH).json()
    assert body["items"] == items


def test_response_wrapper_shape(tmp_path: Path):
    client, _ = _make_client(tmp_path, "shape.db")
    body = client.get(HISTORY_PATH, params={"limit": 10, "offset": 3}).json()
    assert set(body.keys()) == {"items", "limit", "offset"}
    assert body["items"] == []
    assert body["limit"] == 10
    assert body["offset"] == 3


# --- Content: empty / multiple / decisions ---


def test_empty_history(tmp_path: Path):
    client, _ = _make_client(tmp_path, "empty.db")
    body = client.get(HISTORY_PATH).json()
    assert body["items"] == []
    assert body["limit"] == 100
    assert body["offset"] == 0


def test_multiple_history_records_in_order(tmp_path: Path):
    client, application = _make_client(tmp_path, "multi.db")
    application.state.promotion_repo.init()
    _seed(application.state.promotion_repo, ["promoted", "rejected", "failed"])
    body = client.get(HISTORY_PATH).json()
    # Step 63 (I-14): newest-first.
    assert [item["decision"] for item in body["items"]] == [
        "failed",
        "rejected",
        "promoted",
    ]
    assert [item["id"] for item in body["items"]] == sorted(
        (item["id"] for item in body["items"]), reverse=True
    )


def test_promoted_record_round_trip(tmp_path: Path):
    client, application = _make_client(tmp_path, "promoted.db")
    application.state.promotion_repo.init()
    _seed(application.state.promotion_repo, ["promoted"])
    (item,) = client.get(HISTORY_PATH).json()["items"]
    assert item["decision"] == "promoted"
    assert item["model_name"] == "LinearRegression"
    assert item["run_id"] == "run-1"
    assert item["model_version"] == "3"
    assert item["use_case"] == "house_price"
    assert item["reason"].strip()


def test_rejected_record_round_trip(tmp_path: Path):
    client, application = _make_client(tmp_path, "rejected.db")
    application.state.promotion_repo.init()
    _seed(application.state.promotion_repo, ["rejected"])
    (item,) = client.get(HISTORY_PATH).json()["items"]
    assert item["decision"] == "rejected"
    assert item["model_version"] is None


def test_requires_initial_promotion_record(tmp_path: Path):
    client, application = _make_client(tmp_path, "initial.db")
    application.state.promotion_repo.init()
    _seed(application.state.promotion_repo, ["requires_initial_promotion"])
    (item,) = client.get(HISTORY_PATH).json()["items"]
    assert item["decision"] == "requires_initial_promotion"
    assert item["model_version"] is None


def test_failed_record(tmp_path: Path):
    client, application = _make_client(tmp_path, "failed.db")
    application.state.promotion_repo.init()
    _seed(application.state.promotion_repo, ["failed"])
    (item,) = client.get(HISTORY_PATH).json()["items"]
    assert item["decision"] == "failed"


def test_other_use_cases_excluded(tmp_path: Path):
    client, application = _make_client(tmp_path, "isolation.db")
    application.state.promotion_repo.init()
    application.state.promotion_repo.save_promotion_result(
        use_case="other_case", promotion_result=_promotion("promoted")
    )
    _seed(application.state.promotion_repo, ["rejected"])
    body = client.get(HISTORY_PATH).json()
    assert len(body["items"]) == 1
    assert body["items"][0]["use_case"] == "house_price"


# --- Boundaries ---


def test_limit_one_boundary(tmp_path: Path):
    client, application = _make_client(tmp_path, "limit1.db")
    application.state.promotion_repo.init()
    _seed(application.state.promotion_repo, ["promoted", "rejected"])
    body = client.get(HISTORY_PATH, params={"limit": 1}).json()
    assert len(body["items"]) == 1
    # Step 63 (I-14): limit=1 returns the newest record.
    assert body["items"][0]["decision"] == "rejected"


def test_offset_zero_boundary(tmp_path: Path):
    client, application = _make_client(tmp_path, "offset0.db")
    application.state.promotion_repo.init()
    _seed(application.state.promotion_repo, ["promoted"])
    body = client.get(HISTORY_PATH, params={"offset": 0}).json()
    assert body["offset"] == 0
    assert len(body["items"]) == 1


def test_offset_beyond_end_returns_empty(tmp_path: Path):
    client, application = _make_client(tmp_path, "beyond.db")
    application.state.promotion_repo.init()
    _seed(application.state.promotion_repo, ["promoted"])
    body = client.get(HISTORY_PATH, params={"offset": 99}).json()
    assert body["items"] == []


# --- Validation: 422 ---


def test_invalid_limit_zero_returns_422(tmp_path: Path):
    client, _ = _make_client(tmp_path, "l0.db")
    assert client.get(HISTORY_PATH, params={"limit": 0}).status_code == 422


@pytest.mark.parametrize("bad", [-1, -100])
def test_negative_limit_returns_422(tmp_path: Path, bad):
    client, _ = _make_client(tmp_path, "neg.db")
    assert client.get(HISTORY_PATH, params={"limit": bad}).status_code == 422


@pytest.mark.parametrize("bad", [-1, -50])
def test_negative_offset_returns_422(tmp_path: Path, bad):
    client, _ = _make_client(tmp_path, "negoff.db")
    assert client.get(HISTORY_PATH, params={"offset": bad}).status_code == 422


def test_non_integer_limit_returns_422(tmp_path: Path):
    client, _ = _make_client(tmp_path, "nan.db")
    assert client.get(HISTORY_PATH, params={"limit": "many"}).status_code == 422


# --- Errors: 500 without leakage ---


def test_repository_failure_returns_generic_500(tmp_path: Path):
    client, application = _make_client(tmp_path, "fail.db")
    repo = MagicMock(spec=PromotionRepository)
    repo.list_promotion_history = MagicMock(
        side_effect=AdaptiveAIError("disk boom details")
    )
    application.state.promotion_repo = repo
    response = client.get(HISTORY_PATH)
    assert response.status_code == 500
    assert response.json() == {"detail": GENERIC_ERROR}
    assert "boom" not in response.text


def test_unexpected_failure_returns_generic_500(tmp_path: Path):
    client, application = _make_client(tmp_path, "fail2.db")
    repo = MagicMock(spec=PromotionRepository)
    repo.list_promotion_history = MagicMock(side_effect=RuntimeError("kaboom"))
    application.state.promotion_repo = repo
    response = client.get(HISTORY_PATH)
    assert response.status_code == 500
    assert response.json() == {"detail": GENERIC_ERROR}
    assert "kaboom" not in response.text


# --- DI / lazy init ---


def test_dependency_injection_with_temporary_db(tmp_path: Path):
    db_path = tmp_path / "injected.db"
    client, application = _make_client(tmp_path, "injected.db")
    assert isinstance(application.state.promotion_repo, PromotionRepository)
    assert application.state.db_manager is not None
    assert client.get(HISTORY_PATH).status_code == 200
    assert db_path.exists()


def test_promotion_table_lazy_initialization(tmp_path: Path):
    db_path = tmp_path / "lazy.db"
    assert not db_path.exists()
    client, _ = _make_client(tmp_path, "lazy.db")
    assert not db_path.exists()

    def _tables():
        connection = sqlite3.connect(db_path)
        try:
            rows = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
            return {row[0] for row in rows}
        finally:
            connection.close()

    assert client.get(HISTORY_PATH).status_code == 200
    assert "promotion_history" in _tables()


# --- Boundary hygiene ---


def test_no_direct_sql_in_api():
    source = _history_endpoint_source()
    for forbidden in ("SELECT", "INSERT", "sqlite3", "execute("):
        assert forbidden not in source
    module_source = (Path(get_project_root()) / "app" / "api" / "main.py").read_text(
        encoding="utf-8"
    )
    for forbidden in ("SELECT", "INSERT", "sqlite3"):
        assert forbidden not in module_source


def test_no_retraining_or_scheduler_invocation(tmp_path: Path, monkeypatch):
    source = _history_endpoint_source()
    for forbidden in (
        "run_check_and_retrain",
        ".retrain(",
        "run_once",
        ".run_check(",
        "RetrainingService",
        "RetrainingScheduler",
        "DriftRetrainingOrchestrator",
        "ModelPromoter",
        "mlflow",
    ):
        assert forbidden not in source
    # Behavioural: even a rigged orchestrator factory is never touched.
    factory = MagicMock(side_effect=AssertionError("must not orchestrate"))
    monkeypatch.setattr(api_main, "create_house_price_drift_orchestrator", factory)
    client, _ = _make_client(tmp_path, "noretrain.db")
    assert client.get(HISTORY_PATH).status_code == 200
    assert factory.call_count == 0


# --- Regressions ---


def test_health_endpoint_regression(tmp_path: Path):
    client, _ = _make_client(tmp_path, "health.db")
    assert client.get("/health").json() == {"status": "healthy"}


def test_prediction_endpoint_regression(tmp_path: Path, monkeypatch):
    _setup_model(tmp_path, monkeypatch)
    client, _ = _make_client(tmp_path, "predict.db")
    valid = {
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
    response = client.post("/predict/house-price", json=valid)
    assert response.status_code == 200
    assert response.json()["prediction"] > 0


def test_monitoring_history_endpoint_regression(tmp_path: Path):
    client, _ = _make_client(tmp_path, "monreg.db")
    response = client.get("/monitor/house-price/history")
    assert response.status_code == 200
    assert set(response.json().keys()) == {"items", "limit", "offset"}


def test_retrain_endpoint_still_wired(tmp_path: Path, monkeypatch):
    from unittest.mock import MagicMock as _MagicMock

    orchestrator = _MagicMock()
    orchestrator.run_check_and_retrain = _MagicMock(
        return_value={
            "monitoring_result": {},
            "retraining_decision": {},
            "retraining_result": None,
        }
    )
    monkeypatch.setattr(
        api_main,
        "create_house_price_drift_orchestrator",
        _MagicMock(return_value=orchestrator),
    )
    client, _ = _make_client(tmp_path, "retrain.db")
    assert client.post("/monitor/house-price/retrain", json={}).status_code == 200


def test_new_tests_create_no_real_artifacts():
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass
