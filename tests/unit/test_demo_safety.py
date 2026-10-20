"""Phase 6 tests: production-safety proofs for Demo Mode.

Each test below maps to one mandatory safety property: demo activity
must never change production predictions, monitoring history, models,
reference data, or the production pointer — only an explicit promotion
may move the pointer. All state is temporary; production is untouched.
"""

import hashlib
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import Pipeline

from app.core.config import resolve_project_path
from app.database.database import DatabaseManager
from app.database.lifecycle_repository import (
    LifecycleEventRepository,
    PointerHistoryRepository,
    RetrainingRunRepository,
)
from app.database.monitoring_repository import MonitoringRepository
from app.database.repository import PredictionRepository
from app.demo.repository import DemoSessionRepository
from app.demo.service import DemoService
from app.engine.mlflow_tracker import MLflowTracker
from app.monitoring.data_provider import create_house_price_provider
from app.monitoring.distribution_service import get_house_price_distributions
from app.monitoring.monitoring_service import create_house_price_monitoring_service
from app.retraining.lifecycle import create_house_price_lifecycle
from app.retraining.model_promoter import create_house_price_promoter
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_persistence import (
    read_production_pointer,
    save_house_price_model,
    write_production_pointer,
)
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.split import split_house_price_data


def _seed_production(repo: PredictionRepository, n: int = 60, seed: int = 7) -> None:
    ref = pd.read_csv(resolve_project_path("data/reference/house_price_reference.csv"))
    cols = [c for c in ref.columns if c != "price"]
    sample = ref.sample(n=n, random_state=seed).reset_index(drop=True)
    for i in range(n):
        row = {c: sample.loc[i, c] for c in cols}
        for k in ("bedrooms", "bathrooms", "parking", "floor_number", "total_floors"):
            row[k] = int(row[k])
        repo.save_prediction(
            use_case="house_price",
            model_name="Prod",
            input_data=row,
            prediction=1e7,
            source="production",
        )


def _wired(tmp_path: Path, name: str = "safety.db") -> dict:
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    db_manager = DatabaseManager(db_path=tmp_path / name)
    db_manager.init_db()

    df0 = generate_house_price_data(n_samples=200)
    X_train, _, y_train, _ = split_house_price_data(df0)
    pipe = Pipeline(
        [("preprocessor", get_preprocessor()), ("model", DummyRegressor(strategy="mean"))]
    )
    pipe.fit(X_train, y_train)
    save_house_price_model(pipe, "DummyMean", models_dir=models_dir)
    write_production_pointer(
        {
            "model_name": "DummyMean",
            "registered_model_name": "HousePriceModel",
            "model_version": "1",
            "run_id": "run-base",
        },
        models_dir=models_dir,
    )

    pred_repo = PredictionRepository(db_manager)
    provider = create_house_price_provider(repository=pred_repo, source="production")
    tracker = MLflowTracker(
        tracking_uri=str(tmp_path / "mlruns"), experiment_name="house_price_retraining"
    )
    promoter = create_house_price_promoter(tracker=tracker, models_dir=str(models_dir))
    lifecycle = create_house_price_lifecycle(
        tracker=tracker,
        promoter=promoter,
        run_repository=RetrainingRunRepository(db_manager),
        event_repository=LifecycleEventRepository(db_manager),
        pointer_history_repository=PointerHistoryRepository(db_manager),
        monitoring_provider=provider,
        models_dir=str(models_dir),
    )
    demo = DemoService(
        prediction_repository=pred_repo,
        session_repository=DemoSessionRepository(db_manager),
        event_repository=LifecycleEventRepository(db_manager),
        models_dir=str(models_dir),
    )
    return {
        "db": db_manager,
        "models_dir": models_dir,
        "pred_repo": pred_repo,
        "lifecycle": lifecycle,
        "demo": demo,
    }


def _pointer_hash(models_dir: Path) -> str:
    return hashlib.sha256((models_dir / "production.json").read_bytes()).hexdigest()


def _artifact_hash(models_dir: Path) -> str:
    return hashlib.sha256((models_dir / "DummyMean.joblib").read_bytes()).hexdigest()


def _reference_hash() -> str:
    return hashlib.sha256(
        resolve_project_path("data/reference/house_price_reference.csv").read_bytes()
    ).hexdigest()


def test_1_generate_leaves_production_count_unchanged(tmp_path: Path):
    env = _wired(tmp_path, "s1.db")
    _seed_production(env["pred_repo"])
    before = env["pred_repo"].count_predictions("house_price", source="production")
    assert before == 60
    session = env["demo"].create_session("area_shift", sample_count=100, seed=42)
    env["demo"].generate_samples(session["session_id"])
    assert env["pred_repo"].count_predictions("house_price", source="production") == before
    assert env["pred_repo"].count_predictions("house_price", source="demo") == 100


def test_2_demo_monitoring_leaves_monitoring_history_unchanged(tmp_path: Path):
    env = _wired(tmp_path, "s2.db")
    _seed_production(env["pred_repo"])
    mon_repo = MonitoringRepository(env["db"])
    service = create_house_price_monitoring_service(
        repository=env["pred_repo"], monitoring_repository=mon_repo, source="production"
    )
    service.run_health_check()
    before = mon_repo.list_monitoring_results(limit=100, use_case="house_price")
    assert len(before) == 1
    session = env["demo"].create_session("locality_shift", sample_count=100, seed=42)
    env["demo"].generate_samples(session["session_id"])
    env["demo"].monitor_session(session["session_id"])
    after = mon_repo.list_monitoring_results(limit=100, use_case="house_price")
    assert len(after) == len(before) == 1
    assert after[0]["id"] == before[0]["id"]


def test_3_reset_leaves_production_count_unchanged(tmp_path: Path):
    env = _wired(tmp_path, "s3.db")
    _seed_production(env["pred_repo"])
    session = env["demo"].create_session("config_shift", sample_count=40, seed=3)
    env["demo"].generate_samples(session["session_id"])
    env["demo"].reset_session(session["session_id"])
    assert env["pred_repo"].count_predictions("house_price", source="production") == 60
    assert env["pred_repo"].count_predictions("house_price", source="demo") == 0


def test_456_only_explicit_promote_moves_pointer(tmp_path: Path):
    env = _wired(tmp_path, "s456.db")
    _seed_production(env["pred_repo"])
    session = env["demo"].create_session("area_shift", sample_count=100, seed=42)
    env["demo"].generate_samples(session["session_id"])
    env["demo"].monitor_session(session["session_id"])
    before = _pointer_hash(env["models_dir"])
    train_df = generate_house_price_data(n_samples=300)

    run = env["demo"].start_retraining(
        session["session_id"], env["lifecycle"], data_path=train_df
    )
    assert run["status"] == "completed"
    assert run["trigger"].startswith("demo:area_shift:")
    assert _pointer_hash(env["models_dir"]) == before

    checked = env["lifecycle"].promotion_check(run["run_id"])
    assert checked is not None and checked["eligible"] is True
    assert _pointer_hash(env["models_dir"]) == before

    outcome = env["lifecycle"].promote(run["run_id"])
    assert outcome is not None and outcome["promoted"] is True
    assert _pointer_hash(env["models_dir"]) != before
    pointer = read_production_pointer(models_dir=env["models_dir"])
    assert pointer["model_name"] == run["best_model"]

    events = LifecycleEventRepository(env["db"]).list_events(use_case="house_price")
    kinds = [e["event"] for e in events]
    assert "demo_retraining_started" in kinds
    assert "model_promoted" in kinds


def test_7_reset_preserves_models_reference_pointer(tmp_path: Path):
    env = _wired(tmp_path, "s7.db")
    _seed_production(env["pred_repo"])
    pointer_before = _pointer_hash(env["models_dir"])
    artifact_before = _artifact_hash(env["models_dir"])
    reference_before = _reference_hash()
    session = env["demo"].create_session("data_quality", sample_count=100, seed=8)
    env["demo"].generate_samples(session["session_id"])
    env["demo"].monitor_session(session["session_id"])
    env["demo"].reset_session(session["session_id"])
    assert _pointer_hash(env["models_dir"]) == pointer_before
    assert _artifact_hash(env["models_dir"]) == artifact_before
    assert _reference_hash() == reference_before


def test_8_demo_rows_absent_from_production_monitoring(tmp_path: Path):
    env = _wired(tmp_path, "s8.db")
    _seed_production(env["pred_repo"])
    session = env["demo"].create_session("area_shift", sample_count=100, seed=42)
    env["demo"].generate_samples(session["session_id"])
    service = create_house_price_monitoring_service(
        repository=env["pred_repo"], monitoring_repository=None, source="production"
    )
    summary = service.run_health_check()
    assert summary["production_rows"] == 60
    items = env["pred_repo"].list_predictions(
        limit=500, use_case="house_price", source="production"
    )
    assert len(items) == 60
    dists = get_house_price_distributions(repository=env["pred_repo"])
    assert dists["production_rows"] == 60


def test_9_production_rows_absent_from_demo_monitoring(tmp_path: Path):
    env = _wired(tmp_path, "s9.db")
    _seed_production(env["pred_repo"])
    session = env["demo"].create_session("healthy_baseline", sample_count=30, seed=42)
    env["demo"].generate_samples(session["session_id"])
    out = env["demo"].monitor_session(session["session_id"])
    assert out["monitoring"]["production_rows"] == 30
    assert out["monitoring"]["window_records"] == 30


def test_10_same_seed_scenario_reproducible(tmp_path: Path):
    env = _wired(tmp_path, "s10.db")
    first = env["demo"].create_session("locality_shift", sample_count=40, seed=2024)
    second = env["demo"].create_session("locality_shift", sample_count=40, seed=2024)
    env["demo"].generate_samples(first["session_id"])
    env["demo"].generate_samples(second["session_id"])
    repo = env["pred_repo"]
    rows_a = repo.list_predictions(
        limit=40, use_case="house_price", source="demo",
        demo_session_id=first["session_id"],
    )
    rows_b = repo.list_predictions(
        limit=40, use_case="house_price", source="demo",
        demo_session_id=second["session_id"],
    )
    assert [str(r["input_data"]) for r in rows_a] != []
    assert sorted(str(r["input_data"]) for r in rows_a) == sorted(
        str(r["input_data"]) for r in rows_b
    )
