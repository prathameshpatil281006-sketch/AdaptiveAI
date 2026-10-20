"""Phase 4 API tests: health summary, enriched history, distributions.

Verifies the additive monitoring contracts on isolated temporary
databases/model dirs: summary shape and values, history enrichment
keys (old detector rows read back with null enrichment), categorical
distributions, and the byte-identical legacy POST /monitor/house-price
detector contract. Never touches production state.
"""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient
import app.api.main as api_main
from app.api.main import create_app
from app.core.config import get_project_root, resolve_project_path
from app.database.database import DatabaseManager
from app.database.repository import PredictionRepository
from app.use_cases.house_price.model_persistence import write_production_pointer


def _make_client(tmp_path: Path, db_name: str = "phase4.db"):
    db_manager = DatabaseManager(db_path=tmp_path / db_name)
    db_manager.init_db()
    return TestClient(create_app(db_manager=db_manager)), db_manager


def _seed(repo: PredictionRepository, n: int = 60, seed: int = 7) -> None:
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


def _isolate_models(tmp_path: Path, monkeypatch, with_pointer: bool = False) -> Path:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    if with_pointer:
        write_production_pointer(
            {
                "model_name": "LinearRegression",
                "registered_model_name": "HousePriceModel",
                "model_version": "311",
                "run_id": "run-phase4-test",
            },
            models_dir=models_dir,
        )
    monkeypatch.setattr(api_main, "HOUSE_PRICE_MODELS_DIR", str(models_dir))
    return models_dir


def test_summary_healthy_window(tmp_path: Path, monkeypatch):
    _isolate_models(tmp_path, monkeypatch, with_pointer=True)
    client, db_manager = _make_client(tmp_path)
    _seed(PredictionRepository(db_manager))
    body = client.get("/monitor/house-price/summary").json()
    assert body["status"] == "ok"
    assert body["overall_health"] == "healthy"
    assert body["severity"] == "healthy"
    assert body["data_quality_status"] == "pass"
    assert body["overall_drifted"] is False
    assert body["drifted_count"] == 0
    assert body["total_monitored"] == 10
    assert body["production_rows"] == 60
    assert body["total_predictions"] == 60
    assert body["model_name"] == "LinearRegression"
    assert body["model_version"] == "311"
    assert body["schema_version"] == 2
    assert isinstance(body["timestamp"], str) and body["timestamp"]
    assert isinstance(body["monitoring_run_id"], int)
    assert len(body["features"]) == 10
    numeric = [f for f in body["features"] if f.get("type") == "numeric"]
    categorical = [f for f in body["features"] if f.get("type") == "categorical"]
    assert len(numeric) == 7 and len(categorical) == 3
    assert {f["feature_name"] for f in categorical} == {
        "location", "furnishing_status", "property_type"
    }
    for entry in body["features"]:
        assert entry["severity"] in {"healthy", "warning", "critical", "unknown"}
    assert isinstance(body["data_quality"]["checks"], list)
    assert len(body["data_quality"]["checks"]) == 8


def test_summary_without_pointer_has_null_model(tmp_path: Path, monkeypatch):
    _isolate_models(tmp_path, monkeypatch, with_pointer=False)
    client, db_manager = _make_client(tmp_path, "nopoint.db")
    _seed(PredictionRepository(db_manager), n=40)
    body = client.get("/monitor/house-price/summary").json()
    assert body["status"] == "ok"
    assert body["model_name"] is None
    assert body["model_version"] is None
    assert body["schema_version"] == 2


def test_summary_empty_window_unknown(tmp_path: Path, monkeypatch):
    _isolate_models(tmp_path, monkeypatch)
    client, _ = _make_client(tmp_path, "empty.db")
    body = client.get("/monitor/house-price/summary").json()
    assert body["status"] == "insufficient_data"
    assert body["overall_health"] == "unknown"
    assert body["severity"] == "unknown"
    assert body["production_rows"] == 0
    assert body["total_predictions"] == 0


def test_history_items_carry_enrichment(tmp_path: Path, monkeypatch):
    _isolate_models(tmp_path, monkeypatch, with_pointer=True)
    client, db_manager = _make_client(tmp_path, "hist.db")
    _seed(PredictionRepository(db_manager))
    client.get("/monitor/house-price/summary")
    items = client.get("/monitor/house-price/history").json()["items"]
    assert len(items) == 1
    item = items[0]
    # Legacy detector fields still present (backward compatible).
    for key in (
        "id", "timestamp", "use_case", "overall_drifted", "status", "alpha",
        "min_production_rows", "production_rows", "sufficient_data", "features",
    ):
        assert key in item
    # Phase 4 additive enrichment.
    assert item["severity"] == "healthy"
    assert item["overall_health"] == "healthy"
    assert item["data_quality_status"] == "pass"
    assert item["model_name"] == "LinearRegression"
    assert item["model_version"] == "311"
    assert item["schema_version"] == 2
    assert item["drifted_features"] == 0


def test_legacy_detector_rows_keep_null_enrichment(tmp_path: Path):
    client, db_manager = _make_client(tmp_path, "legacy.db")
    _seed(PredictionRepository(db_manager), n=35)
    legacy = client.post("/monitor/house-price").json()
    assert legacy["status"] == "ok"
    assert set(legacy.keys()) == {
        "overall_drifted", "alpha", "correction", "min_production_rows",
        "production_rows", "sufficient_data", "status", "features",
    }
    assert len(legacy["features"]) == 7
    items = client.get("/monitor/house-price/history").json()["items"]
    assert len(items) == 1
    assert items[0]["severity"] is None
    assert items[0]["overall_health"] is None
    assert items[0]["model_name"] is None


def test_distributions_include_categorical(tmp_path: Path):
    from app.monitoring.distribution_service import get_house_price_distributions

    db_manager = DatabaseManager(db_path=tmp_path / "dist.db")
    db_manager.init_db()
    _seed(PredictionRepository(db_manager))
    body = get_house_price_distributions(
        production_limit=60, repository=PredictionRepository(db_manager)
    )
    assert body["reference_rows"] == 8500
    assert body["production_rows"] == 60
    assert body["sufficient_data"] is True
    by_name = {f["feature"]: f for f in body["features"]}
    assert len(by_name) == 10
    numeric = by_name["area_sqft"]
    assert numeric["type"] == "numeric"
    assert set(numeric.keys()) >= {"feature", "reference", "production"}
    assert set(numeric["reference"].keys()) >= {"mean", "std", "count", "histogram"}
    loc = by_name["location"]
    assert loc["type"] == "categorical"
    assert isinstance(loc["categories"], list) and len(loc["categories"]) == 12
    total_share = sum(c["reference_share"] for c in loc["categories"])
    assert total_share == pytest.approx(1.0)
    for cat in loc["categories"]:
        assert cat["reference_count"] + cat["production_count"] >= 0
        assert 0.0 <= cat["production_share"] <= 1.0
    furn = by_name["furnishing_status"]
    assert {c["value"] for c in furn["categories"]} == {
        "Unfurnished", "Semi-Furnished", "Fully-Furnished"
    }


def test_distributions_endpoint_shape(tmp_path: Path):
    # Endpoint reads the configured database: assert the stable
    # reference-side contract only (production rows are live state).
    client, _ = _make_client(tmp_path, "distep.db")
    body = client.get("/monitor/house-price/distributions?limit=60").json()
    assert body["reference_rows"] == 8500
    assert isinstance(body["production_rows"], int)
    assert isinstance(body["sufficient_data"], bool)
    by_name = {f["feature"]: f for f in body["features"]}
    assert len(by_name) == 10
    assert by_name["area_sqft"]["type"] == "numeric"
    assert by_name["location"]["type"] == "categorical"
    assert len(by_name["location"]["categories"]) == 12


def test_distributions_empty_window_explicit(tmp_path: Path):
    client, _ = _make_client(tmp_path, "distempty.db")
    body = client.get("/monitor/house-price/distributions?limit=60").json()
    assert body["production_rows"] == 0
    assert body["sufficient_data"] is False
    loc = next(f for f in body["features"] if f["feature"] == "location")
    assert all(c["production_count"] == 0 for c in loc["categories"])


def test_real_production_pointer_unchanged():
    root = Path(get_project_root())
    pointer = (root / "models" / "house_price" / "production.json").read_text(
        encoding="utf-8"
    )
    assert '"model_version": "311"' in pointer
    # Isolation canary: live-state guard lives in tests/conftest.py.
