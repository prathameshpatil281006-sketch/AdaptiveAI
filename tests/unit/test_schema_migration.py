"""Phase 5 tests: schema-migration deployment path.

Covers the controlled migration flow for a schema-incompatible
production baseline: detection (incompatible_baseline, never a crash),
absolute migration gating (never the relative regression gate),
explicit-confirmation promotion, audit/pointer-history recording, and
post-migration rollback. The normal compatible-upgrade path is asserted
unchanged. All state is isolated under tmp_path; production is never
touched.
"""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.exceptions import ModelPromotionError, RetrainingError
from app.database.database import DatabaseManager
from app.database.lifecycle_repository import (
    LifecycleEventRepository,
    PointerHistoryRepository,
    RetrainingRunRepository,
)
from app.database.repository import PredictionRepository
from app.engine.mlflow_tracker import MLflowTracker
from app.monitoring.data_provider import create_house_price_provider
from app.retraining.lifecycle import create_house_price_lifecycle
from app.retraining.model_promoter import create_house_price_promoter
from app.retraining.promotion_gate import (
    MIGRATION_REQUIRED_MESSAGE,
    check_migration_eligibility,
    check_promotion_eligibility,
)
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_persistence import (
    read_production_pointer,
    save_house_price_model,
    write_production_pointer,
)

LEGACY_FEATURES = ["area", "bedrooms", "bathrooms", "property_age", "parking"]


def _legacy_production_pipeline() -> Pipeline:
    """Builds a v1-style pipeline expecting legacy feature names."""
    numeric = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    preprocessor = ColumnTransformer(
        transformers=[("num", numeric, list(LEGACY_FEATURES))], remainder="drop"
    )
    return Pipeline([("preprocessor", preprocessor), ("model", LinearRegression())])


def _wired_migration(tmp_path: Path):
    """Isolated lifecycle service with a legacy-v1 production artifact."""
    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    db_manager = DatabaseManager(db_path=tmp_path / "mig.db")
    db_manager.init_db()

    rng = np.random.RandomState(7)
    n = 60
    legacy = pd.DataFrame(
        {
            "area": rng.uniform(400, 2000, n),
            "bedrooms": rng.randint(1, 5, n),
            "bathrooms": rng.randint(1, 4, n),
            "property_age": rng.uniform(0, 30, n),
            "parking": rng.randint(0, 3, n),
            "price": rng.uniform(3e6, 2e7, n),
        }
    )
    pipe = _legacy_production_pipeline()
    pipe.fit(legacy[LEGACY_FEATURES], legacy["price"])
    save_house_price_model(pipe, "LinearRegression", models_dir=models_dir)
    write_production_pointer(
        {
            "model_name": "LinearRegression",
            "registered_model_name": "HousePriceModel",
            "model_version": "311",
            "run_id": "run-legacy-v1",
        },
        models_dir=models_dir,
    )

    pred_repo = PredictionRepository(db_manager)
    ref = generate_house_price_data(n_samples=200)
    cols = [c for c in ref.columns if c != "price"]
    sample = ref.sample(n=60, random_state=5).reset_index(drop=True)
    for i in range(60):
        row = {c: sample.loc[i, c] for c in cols}
        for k in ("bedrooms", "bathrooms", "parking", "floor_number", "total_floors"):
            row[k] = int(row[k])
        pred_repo.save_prediction(
            use_case="house_price", model_name="M", input_data=row, prediction=1e7
        )

    provider = create_house_price_provider(repository=pred_repo)
    tracker = MLflowTracker(
        tracking_uri=str(tmp_path / "mlruns"), experiment_name="house_price_retraining"
    )
    promoter = create_house_price_promoter(tracker=tracker, models_dir=str(models_dir))
    service = create_house_price_lifecycle(
        tracker=tracker,
        promoter=promoter,
        run_repository=RetrainingRunRepository(db_manager),
        event_repository=LifecycleEventRepository(db_manager),
        pointer_history_repository=PointerHistoryRepository(db_manager),
        monitoring_provider=provider,
        models_dir=str(models_dir),
    )
    return {"service": service, "db_manager": db_manager, "models_dir": models_dir}


def _pointer_bytes(models_dir: Path) -> bytes:
    return (models_dir / "production.json").read_bytes()


# A. schema-compatible upgrade still uses the normal regression gate.
def test_a_compatible_upgrade_uses_normal_gate():
    result = check_promotion_eligibility(
        run_completed=True,
        artifact_available=True,
        artifact_loadable=True,
        schema_compatible=True,
        candidate_metrics={"mae": 1.0, "rmse": 2.0, "r2": 0.9},
        production_metrics={"mae": 2.0, "rmse": 3.0, "r2": 0.8},
        improvement=0.1,
        data_quality_status="pass",
    )
    assert result["eligible"] is True
    assert [c["name"] for c in result["checks"]] == [
        "run_completed",
        "artifact_available",
        "artifact_loadable",
        "schema_compatible",
        "metrics_available",
        "production_baseline",
        "regression_tolerance",
        "data_quality",
    ]
    assert "migration" not in result


# B. schema-incompatible production does not crash retraining.
def test_b_incompatible_production_completes_run(tmp_path: Path):
    env = _wired_migration(tmp_path)
    train_df = generate_house_price_data(n_samples=150)
    run = env["service"].start_run(
        production_limit=100, trigger="manual", data_path=train_df
    )
    assert run["status"] == "completed"
    assert run["best_model"] in {
        "LinearRegression",
        "RandomForestRegressor",
        "GradientBoostingRegressor",
    }
    assert isinstance(run["candidate_artifact"], str) and run["candidate_artifact"]
    assert run["failure_reason"] is None
    metrics = run["best_metrics"]
    for key in ("mae", "rmse", "r2"):
        assert np.isfinite(metrics[key])


# C. schema-incompatible production enters migration mode.
def test_c_incompatible_baseline_marks_migration(tmp_path: Path):
    env = _wired_migration(tmp_path)
    train_df = generate_house_price_data(n_samples=150)
    run = env["service"].start_run(
        production_limit=100, trigger="manual", data_path=train_df
    )
    comparison = run["comparison"]
    assert comparison["status"] == "incompatible_baseline"
    assert MIGRATION_REQUIRED_MESSAGE in comparison["reason"]
    assert comparison["migration"]["required"] is True
    assert comparison["candidate_schema_version"] == 2
    assert comparison["current_schema_version"] == 2
    assert comparison["production"]["metrics"] is None
    assert comparison["production"]["model_name"] == "LinearRegression"

    checked = env["service"].promotion_check(run["run_id"])
    assert checked["eligible"] is True
    assert checked["migration"]["required"] is True
    assert checked["migration"]["message"] == MIGRATION_REQUIRED_MESSAGE


# D. migration candidate must satisfy absolute validation.
def test_d_migration_gate_marks_baseline_and_tolerance(tmp_path: Path):
    env = _wired_migration(tmp_path)
    train_df = generate_house_price_data(n_samples=150)
    run = env["service"].start_run(
        production_limit=100, trigger="manual", data_path=train_df
    )
    checked = env["service"].promotion_check(run["run_id"])
    by_name = {c["name"]: c for c in checked["checks"]}
    assert by_name["production_baseline"]["passed"] is True
    assert "PASS_WITH_INCOMPATIBLE_BASELINE" in by_name["production_baseline"]["message"]
    assert by_name["regression_tolerance"]["passed"] is True
    assert "NOT_APPLICABLE_SCHEMA_MIGRATION" in by_name["regression_tolerance"]["message"]
    assert by_name["migration_validation"]["passed"] is True
    assert by_name["schema_compatible"]["passed"] is True
    assert checked["eligible"] is True


# E. migration candidate with bad metrics is rejected.
@pytest.mark.parametrize("metrics", [None, {"mae": None, "rmse": 2.0, "r2": 0.5},
                                     {"mae": 1.0, "rmse": 2.0, "r2": -0.2},
                                     {"mae": 1.0, "rmse": float("nan"), "r2": 0.5}])
def test_e_bad_metrics_rejected(metrics):
    result = check_migration_eligibility(
        run_completed=True,
        artifact_available=True,
        artifact_loadable=True,
        schema_compatible=True,
        candidate_metrics=metrics,
        candidate_schema_version=2,
        current_schema_version=2,
        data_quality_status="pass",
    )
    assert result["eligible"] is False
    by_name = {c["name"]: c for c in result["checks"]}
    assert by_name["migration_validation"]["passed"] is False


# F. migration candidate with incompatible schema is rejected.
def test_f_incompatible_candidate_schema_rejected():
    result = check_migration_eligibility(
        run_completed=True,
        artifact_available=True,
        artifact_loadable=True,
        schema_compatible=False,
        candidate_metrics={"mae": 1.0, "rmse": 2.0, "r2": 0.9},
        candidate_schema_version=1,
        current_schema_version=2,
        data_quality_status="pass",
    )
    assert result["eligible"] is False
    by_name = {c["name"]: c for c in result["checks"]}
    assert by_name["schema_compatible"]["passed"] is False
    assert by_name["migration_validation"]["passed"] is False


# G. migration promotion requires explicit confirmation.
def test_g_promotion_requires_explicit_confirmation(tmp_path: Path):
    env = _wired_migration(tmp_path)
    train_df = generate_house_price_data(n_samples=150)
    run = env["service"].start_run(
        production_limit=100, trigger="manual", data_path=train_df
    )
    before = _pointer_bytes(env["models_dir"])
    for kwargs in ({}, {"allow_migration": False}):
        outcome = env["service"].promote(run["run_id"], **kwargs)
        assert outcome["promoted"] is False
        assert "allow_migration" in outcome["reason"]
        assert MIGRATION_REQUIRED_MESSAGE in outcome["reason"]
    assert _pointer_bytes(env["models_dir"]) == before
    pointer = read_production_pointer(models_dir=str(env["models_dir"]))
    assert pointer["model_version"] == "311"


# H. failed migration promotion leaves the production pointer unchanged.
def test_h_failed_migration_leaves_pointer_unchanged(tmp_path: Path):
    env = _wired_migration(tmp_path)
    train_df = generate_house_price_data(n_samples=150)
    run = env["service"].start_run(
        production_limit=100, trigger="manual", data_path=train_df
    )
    staged = env["models_dir"] / f"{run['candidate_artifact']}.joblib"
    assert staged.is_file()
    staged.unlink()
    before = _pointer_bytes(env["models_dir"])
    outcome = env["service"].promote(run["run_id"], allow_migration=True)
    assert outcome["promoted"] is False
    assert outcome["eligible"] is False
    assert _pointer_bytes(env["models_dir"]) == before


# I. successful migration promotion records pointer history and audit event.
def test_i_successful_migration_records_history_and_audit(tmp_path: Path):
    env = _wired_migration(tmp_path)
    train_df = generate_house_price_data(n_samples=150)
    run = env["service"].start_run(
        production_limit=100, trigger="manual", data_path=train_df
    )
    outcome = env["service"].promote(run["run_id"], allow_migration=True)
    assert outcome["promoted"] is True
    assert outcome["migration"]["required"] is True
    assert outcome["previous_production_version"] == "311"

    pointer = read_production_pointer(models_dir=str(env["models_dir"]))
    assert pointer["model_name"] == run["best_model"]
    assert pointer["model_version"] != "311"

    history = PointerHistoryRepository(env["db_manager"]).list_transitions()
    assert len(history) == 1
    assert history[0]["previous_pointer"]["model_version"] == "311"
    assert history[0]["new_pointer"]["model_version"] == pointer["model_version"]

    events = LifecycleEventRepository(env["db_manager"]).list_events()
    kinds = [e["event"] for e in events]
    assert "model_promoted" in kinds


# J. rollback still works after migration.
def test_j_rollback_after_migration(tmp_path: Path):
    env = _wired_migration(tmp_path)
    train_df = generate_house_price_data(n_samples=150)
    run = env["service"].start_run(
        production_limit=100, trigger="manual", data_path=train_df
    )
    outcome = env["service"].promote(run["run_id"], allow_migration=True)
    assert outcome["promoted"] is True
    migrated_pointer = read_production_pointer(models_dir=str(env["models_dir"]))

    history = PointerHistoryRepository(env["db_manager"]).list_transitions()
    assert len(history) == 1
    assert env["service"].rollback(9999) is None

    # Rollback to the recorded v311 previous pointer exercises the
    # existing rollback validation. Two correct outcomes exist because
    # the pre-existing promoter saves under the candidate model name:
    # a different-name winner preserves the legacy bytes, so rollback
    # to v311 fails closed on schema validation; a same-name
    # (LinearRegression) winner overwrites the file, so rollback
    # restores the v311 reference over v2 content and succeeds.
    pointer_before_rb = read_production_pointer(models_dir=str(env["models_dir"]))
    if run["best_model"] == "LinearRegression":
        rolled_back = env["service"].rollback(history[0]["id"])
        assert rolled_back["rolled_back"] is True
        assert (
            read_production_pointer(models_dir=str(env["models_dir"]))["model_version"]
            == "311"
        )
    else:
        with pytest.raises(RetrainingError):
            env["service"].rollback(history[0]["id"])
        assert read_production_pointer(models_dir=str(env["models_dir"])) == pointer_before_rb

    # A second lifecycle run now follows the normal compatible path, and a
    # subsequent compatible upgrade restores validated rollback.
    run2 = env["service"].start_run(
        production_limit=100, trigger="manual",
        data_path=generate_house_price_data(n_samples=3000),
    )
    assert run2["status"] == "completed"
    assert run2["comparison"]["status"] != "incompatible_baseline"
    checked2 = env["service"].promotion_check(run2["run_id"])
    assert "migration" not in checked2
    assert [c["name"] for c in checked2["checks"]] == [
        "run_completed",
        "artifact_available",
        "artifact_loadable",
        "schema_compatible",
        "metrics_available",
        "production_baseline",
        "regression_tolerance",
        "data_quality",
    ]
    outcome2 = env["service"].promote(run2["run_id"])
    assert outcome2["promoted"] is True
    history2 = PointerHistoryRepository(env["db_manager"]).list_transitions()
    # migration + (successful-rollback transition, if the fixture took the
    # same-name branch) + second upgrade.
    assert len(history2) in (2, 3)
    rolled2 = env["service"].rollback(history2[0]["id"])
    assert rolled2["rolled_back"] is True
    restored = read_production_pointer(models_dir=str(env["models_dir"]))
    assert restored["model_name"] == history2[0]["previous_pointer"]["model_name"]
    assert restored["model_version"] == history2[0]["previous_pointer"]["model_version"]
    events = LifecycleEventRepository(env["db_manager"]).list_events()
    assert "model_rollback" in [e["event"] for e in events]


def test_promoter_rejects_non_passing_migration_assessment(tmp_path: Path):
    tracker = MLflowTracker(
        tracking_uri=str(tmp_path / "mlruns"), experiment_name="house_price_retraining"
    )
    promoter = create_house_price_promoter(tracker=tracker, models_dir=str(tmp_path))
    with pytest.raises(ModelPromotionError):
        promoter.promote_schema_migration(
            candidate_model=object(),
            migration_assessment={"eligible": False, "checks": [], "migration": {"required": True}},
            candidate_model_name="M",
            candidate_run_id="run-x",
        )


def test_migration_sha_state_guard(tmp_path: Path):
    """Pointer file is untouched by run/check/confirmed-blocked flows."""
    env = _wired_migration(tmp_path)
    digest_before = hashlib.sha256(_pointer_bytes(env["models_dir"])).hexdigest()
    train_df = generate_house_price_data(n_samples=150)
    run = env["service"].start_run(
        production_limit=100, trigger="manual", data_path=train_df
    )
    env["service"].promotion_check(run["run_id"])
    env["service"].promote(run["run_id"])
    assert hashlib.sha256(_pointer_bytes(env["models_dir"])).hexdigest() == digest_before
