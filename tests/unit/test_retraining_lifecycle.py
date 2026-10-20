"""Phase 5 tests: lifecycle service (runs, gate, promote, rollback).

End-to-end lifecycle behavior on isolated temporary state (database,
models, MLflow): successful and failed runs, reproducibility, no
automatic promotion, gate-blocked and failed promotions leaving the
pointer untouched, explicit promotion moving it only after gates pass,
and validated rollback. Never touches production state.
"""

import hashlib
from pathlib import Path

import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from app.core.config import resolve_project_path
from app.database.database import DatabaseManager
from app.database.lifecycle_repository import (
    LifecycleEventRepository,
    PointerHistoryRepository,
    RetrainingRunRepository,
)
from app.database.repository import PredictionRepository
from app.engine.mlflow_tracker import MLflowTracker
from app.monitoring.data_provider import create_house_price_provider
from app.retraining.lifecycle import build_baseline_comparison, create_house_price_lifecycle
from app.retraining.model_promoter import create_house_price_promoter
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_persistence import (
    read_production_pointer,
    save_house_price_model,
    write_production_pointer,
)
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.split import split_house_price_data


def _wired(tmp_path: Path, name: str = "lc.db", weak_production: bool = True):
    """Builds an isolated lifecycle service with a weak v2 production model."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    db_manager = DatabaseManager(db_path=tmp_path / name)
    db_manager.init_db()

    df0 = generate_house_price_data(n_samples=200)
    X_train, _, y_train, _ = split_house_price_data(df0)
    base = DummyRegressor(strategy="mean") if weak_production else LinearRegression()
    pipe = Pipeline([("preprocessor", get_preprocessor()), ("model", base)])
    pipe.fit(X_train, y_train)
    prod_name = "DummyMean" if weak_production else "LinearRegression"
    save_house_price_model(pipe, prod_name, models_dir=models_dir)
    write_production_pointer(
        {
            "model_name": prod_name,
            "registered_model_name": "HousePriceModel",
            "model_version": "1",
            "run_id": "run-base",
        },
        models_dir=models_dir,
    )

    pred_repo = PredictionRepository(db_manager)
    provider = create_house_price_provider(repository=pred_repo)
    tracker = MLflowTracker(
        tracking_uri=str(tmp_path / "mlruns"), experiment_name="house_price_retraining"
    )
    promoter = create_house_price_promoter(
        tracker=tracker, models_dir=str(models_dir)
    )
    service = create_house_price_lifecycle(
        tracker=tracker,
        promoter=promoter,
        run_repository=RetrainingRunRepository(db_manager),
        event_repository=LifecycleEventRepository(db_manager),
        pointer_history_repository=PointerHistoryRepository(db_manager),
        monitoring_provider=provider,
        models_dir=str(models_dir),
    )
    return {
        "service": service,
        "db_manager": db_manager,
        "models_dir": models_dir,
        "pred_repo": pred_repo,
        "tmp": tmp_path,
    }


def _seed_predictions(pred_repo: PredictionRepository, n: int = 60, seed: int = 5) -> None:
    ref = pd.read_csv(resolve_project_path("data/reference/house_price_reference.csv"))
    cols = [c for c in ref.columns if c != "price"]
    sample = ref.sample(n=n, random_state=seed).reset_index(drop=True)
    for i in range(n):
        row = {c: sample.loc[i, c] for c in cols}
        for k in ("bedrooms", "bathrooms", "parking", "floor_number", "total_floors"):
            row[k] = int(row[k])
        pred_repo.save_prediction(
            use_case="house_price", model_name="M", input_data=row, prediction=1e7
        )


def _pointer_hash(models_dir: Path) -> str:
    raw = (models_dir / "production.json").read_bytes()
    return hashlib.sha256(raw).hexdigest()


def test_build_baseline_comparison_math():
    out = build_baseline_comparison(
        {"mae": 10.0, "rmse": 15.0, "r2": 0.9},
        {"mae": 20.0, "rmse": 30.0, "r2": 0.8},
    )
    assert out["candidate"]["mae"] == 10.0
    assert out["production"]["r2"] == 0.8
    assert out["deltas"]["mae_delta"] == pytest.approx(10.0)
    assert out["deltas"]["rmse_delta"] == pytest.approx(15.0)
    assert out["deltas"]["r2_improvement"] == pytest.approx(0.1)
    assert out["deltas"]["mae_reduction_pct"] == pytest.approx(50.0)
    assert out["deltas"]["r2_improvement_pct"] == pytest.approx(12.5)


def test_build_baseline_comparison_no_baseline():
    out = build_baseline_comparison({"mae": 1.0, "rmse": 2.0, "r2": 0.5}, None)
    assert out["production"] is None
    assert out["deltas"] is None
    assert out["candidate"] == {"mae": 1.0, "rmse": 2.0, "r2": 0.5}


def test_start_run_success_records_everything(tmp_path: Path):
    env = _wired(tmp_path)
    service = env["service"]
    before = _pointer_hash(env["models_dir"])
    train_df = generate_house_price_data(n_samples=300)

    run = service.start_run(production_limit=100, trigger="manual", data_path=train_df)

    assert run["status"] == "completed"
    assert run["run_id"].startswith("rtr_")
    assert run["trigger"] == "manual"
    assert run["dataset_rows"] == 300
    assert run["schema_version"] == 2
    assert run["train_rows"] == 240 and run["test_rows"] == 60
    assert run["random_seed"] == 42
    assert isinstance(run["started_at"], str) and isinstance(run["completed_at"], str)
    assert run["production_model"] == "DummyMean"
    assert run["production_version"] == "1"
    assert run["best_model"] in {
        "LinearRegression", "RandomForestRegressor", "GradientBoostingRegressor",
    }
    assert set(run["best_metrics"]) >= {"mae", "rmse", "r2"}
    assert run["candidate_run_id"]
    assert (env["models_dir"] / f".staging/{run['run_id']}.joblib").is_file()
    comparison = run["comparison"]
    assert comparison["candidate"]["metrics"]["r2"] > 0.5
    assert comparison["production"]["metrics"]["r2"] < 0.1
    assert comparison["deltas"]["r2_improvement"] > 0.5
    # No automatic promotion: pointer untouched, no decision recorded.
    assert _pointer_hash(env["models_dir"]) == before
    assert run["promotion_decision"] is None
    events = LifecycleEventRepository(env["db_manager"]).list_events()
    kinds = [e["event"] for e in events]
    assert "retraining_started" in kinds and "retraining_completed" in kinds


def test_start_run_failure_records_failed_run(tmp_path: Path):
    env = _wired(tmp_path, name="fail.db")
    service = env["service"]
    before = _pointer_hash(env["models_dir"])
    bad = pd.DataFrame({"area_sqft": [1.0]})  # fails validation/split

    run = service.start_run(trigger="manual", data_path=bad)

    assert run["status"] == "failed"
    assert isinstance(run["failure_reason"], str) and run["failure_reason"]
    assert run["completed_at"] is not None
    assert _pointer_hash(env["models_dir"]) == before
    events = LifecycleEventRepository(env["db_manager"]).list_events()
    assert any(e["event"] == "retraining_failed" for e in events)


def test_runs_are_reproducible(tmp_path: Path):
    env = _wired(tmp_path, name="repro.db")
    service = env["service"]
    train_df = generate_house_price_data(n_samples=300)
    first = service.start_run(trigger="manual", data_path=train_df)
    second = service.start_run(trigger="manual", data_path=train_df)
    assert first["status"] == "completed" and second["status"] == "completed"
    assert first["run_id"] != second["run_id"]
    assert first["best_model"] == second["best_model"]
    assert first["best_metrics"]["r2"] == pytest.approx(second["best_metrics"]["r2"])
    assert first["comparison"]["deltas"] == second["comparison"]["deltas"]
    runs = service.list_runs(use_case="house_price")
    assert [r["run_id"] for r in runs] == [second["run_id"], first["run_id"]]


def test_promote_blocked_when_candidate_worse(tmp_path: Path):
    # Strong production (full-data LR) beats a small-data candidate run.
    env = _wired(tmp_path, name="blocked.db", weak_production=False)
    service = env["service"]
    before = _pointer_hash(env["models_dir"])
    train_df = generate_house_price_data(n_samples=120)
    run = service.start_run(trigger="manual", data_path=train_df)
    assert run["status"] == "completed"

    checked = service.promotion_check(run["run_id"])
    assert checked is not None
    outcome = service.promote(run["run_id"])
    assert outcome is not None
    assert outcome["promoted"] is False
    assert _pointer_hash(env["models_dir"]) == before
    stored = service.get_run(run["run_id"])
    assert stored is not None and stored["promotion_decision"] == "blocked"


def test_explicit_promote_moves_pointer_then_rollback(tmp_path: Path):
    env = _wired(tmp_path, name="happy.db")
    service = env["service"]
    _seed_predictions(env["pred_repo"])
    train_df = generate_house_price_data(n_samples=300)
    run = service.start_run(trigger="manual", data_path=train_df)
    assert run["status"] == "completed"

    checked = service.promotion_check(run["run_id"])
    assert checked is not None and checked["eligible"] is True
    before = _pointer_hash(env["models_dir"])

    outcome = service.promote(run["run_id"])
    assert outcome is not None and outcome["promoted"] is True
    assert outcome["previous_production_version"] == "1"
    assert _pointer_hash(env["models_dir"]) != before
    pointer = read_production_pointer(models_dir=env["models_dir"])
    assert pointer["model_name"] == run["best_model"]
    stored = service.get_run(run["run_id"])
    assert stored is not None and stored["promotion_decision"] == "promoted"
    assert stored["candidate_version"] == pointer["model_version"]
    history = PointerHistoryRepository(env["db_manager"]).list_transitions()
    assert len(history) == 1
    assert history[0]["run_id"] == run["run_id"]

    rolled = service.rollback(history[0]["id"])
    assert rolled is not None and rolled["rolled_back"] is True
    restored = read_production_pointer(models_dir=env["models_dir"])
    assert restored["model_name"] == "DummyMean"
    assert restored["model_version"] == "1"
    history = PointerHistoryRepository(env["db_manager"]).list_transitions()
    assert len(history) == 2
    events = LifecycleEventRepository(env["db_manager"]).list_events()
    kinds = [e["event"] for e in events]
    assert "model_promoted" in kinds and "model_rollback" in kinds


def test_failed_promoter_leaves_pointer_unchanged(tmp_path: Path):
    import app.retraining.lifecycle as lifecycle_module

    env = _wired(tmp_path, name="promofail.db")
    _seed_predictions(env["pred_repo"])
    train_df = generate_house_price_data(n_samples=300)
    run = env["service"].start_run(trigger="manual", data_path=train_df)
    assert env["service"].promotion_check(run["run_id"])["eligible"] is True
    before = _pointer_hash(env["models_dir"])

    class _BoomPromoter:
        def promote(self, **kwargs):
            raise RuntimeError("registry exploded")

        def update_production_pointer(self, *args, **kwargs):
            raise AssertionError("must not publish after promoter failure")

    env["service"]._promoter = _BoomPromoter()
    with pytest.raises(Exception):
        env["service"].promote(run["run_id"])
    assert _pointer_hash(env["models_dir"]) == before
    assert lifecycle_module.PROMOTION_DECIDED_PROMOTED != (
        env["service"].get_run(run["run_id"]) or {}
    ).get("promotion_decision")


def test_incompatible_candidate_cannot_promote(tmp_path: Path):
    env = _wired(tmp_path, name="incompat.db")
    service = env["service"]
    # Stage a schema-v1 style artifact under a fake run identity.
    legacy = pd.DataFrame(
        {
            "area": [1000.0, 1500.0, 2000.0, 2500.0],
            "bedrooms": [2, 3, 3, 4],
            "bathrooms": [1, 2, 2, 3],
            "property_age": [5.0, 10.0, 2.0, 20.0],
            "parking": [1, 1, 2, 2],
            "house_price": [5e6, 7e6, 9e6, 12e6],
        }
    )
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline(
        steps=[("preprocessor", StandardScaler()), ("model", LinearRegression())]
    )
    pipe.fit(
        legacy[["area", "bedrooms", "bathrooms", "property_age", "parking"]],
        legacy["house_price"],
    )
    staging = env["models_dir"] / ".staging"
    staging.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(pipe, staging / "rtr_legacy.joblib")
    run_repo = RetrainingRunRepository(env["db_manager"])
    run_repo.create_run(run_id="rtr_legacy", use_case="house_price", trigger="manual")
    run_repo.update_run(
        "rtr_legacy",
        status="completed",
        best_model="LinearRegression",
        best_metrics={"mae": 1.0, "rmse": 2.0, "r2": 0.5},
        candidate_artifact=".staging/rtr_legacy",
        candidate_run_id="run-legacy",
    )
    before = _pointer_hash(env["models_dir"])
    checked = service.promotion_check("rtr_legacy")
    assert checked is not None and checked["eligible"] is False
    outcome = service.promote("rtr_legacy")
    assert outcome is not None and outcome["promoted"] is False
    assert _pointer_hash(env["models_dir"]) == before


def test_unknown_run_ids_return_none(tmp_path: Path):
    env = _wired(tmp_path, name="unknown.db")
    service = env["service"]
    assert service.get_run("rtr_nope") is None
    assert service.promotion_check("rtr_nope") is None
    assert service.promote("rtr_nope") is None
    assert service.rollback(999999) is None


def test_rollback_refusals_leave_pointer_unchanged(tmp_path: Path):
    env = _wired(tmp_path, name="rbrefuse.db")
    service = env["service"]
    before = _pointer_hash(env["models_dir"])
    repo = PointerHistoryRepository(env["db_manager"])
    # Entry with no previous state.
    rid = repo.record_transition(
        use_case="house_price",
        new_pointer={"model_name": "X", "model_version": "1"},
        previous_pointer=None,
        reason="first deployment",
    )
    import app.core.exceptions as exc

    with pytest.raises(exc.RetrainingError):
        service.rollback(rid)
    assert _pointer_hash(env["models_dir"]) == before


def test_real_production_pointer_unchanged():
    from app.core.config import get_project_root

    root = Path(get_project_root())
    pointer = (root / "models" / "house_price" / "production.json").read_text(
        encoding="utf-8"
    )
    assert '"model_version": "311"' in pointer
    # Isolation canary: live-state guard lives in tests/conftest.py.
