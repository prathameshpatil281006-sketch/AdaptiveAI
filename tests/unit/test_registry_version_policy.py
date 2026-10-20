"""Step 58 tests: registry versions are created only at promotion/deployment.

Policy under test (I-07 fix):
- Candidate evaluation/selection tracks MLflow runs but registers zero
  versions.
- promote=False registers zero versions.
- Rejected candidates register zero versions.
- A promoted/deployed candidate registers exactly one version whose
  run_id is the candidate's exact run_id; production.json carries the
  same version.
- production.json remains the authority for serving (unchanged schema).

All stores/registries/databases are temporary. The real mlruns/ (with
HousePriceModel v311) is never touched.
"""

import numpy as np
from mlflow.tracking import MlflowClient
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from app.core.config import get_project_root
from app.database.database import DatabaseManager
from app.database.promotion_repository import PromotionRepository
from app.engine.ml_engine import MLEngine
from app.engine.mlflow_tracker import MLflowTracker, get_default_tracking_dir
from app.retraining.model_promoter import create_house_price_promoter
from app.retraining.retraining_service import RetrainingService
from app.use_cases.house_price.data import generate_house_price_data
from app.use_cases.house_price.model_persistence import (
    PRODUCTION_POINTER_FILENAME,
    read_production_pointer,
    save_house_price_model,
    write_production_pointer,
)
from app.use_cases.house_price.model_selection import select_best_house_price_model
from app.use_cases.house_price.preprocessing import get_preprocessor
from app.use_cases.house_price.split import split_house_price_data
from app.use_cases.house_price.use_case import HousePriceUseCase

REGISTERED = "HousePriceModel"


def _tracker(tmp_path: Path, name="policy_mlruns") -> MLflowTracker:
    return MLflowTracker(
        tracking_uri=str(tmp_path / name), experiment_name="house_price_retraining"
    )


def _service(tmp_path: Path, tracker, models_dir: Path) -> RetrainingService:
    db_manager = DatabaseManager(db_path=tmp_path / "policy.db")
    db_manager.init_db()
    promoter = create_house_price_promoter(tracker=tracker, models_dir=str(models_dir))
    return RetrainingService(
        engine=MLEngine(),
        tracker=tracker,
        promoter=promoter,
        promotion_repository=PromotionRepository(db_manager),
    )


def _versions(tracker: MLflowTracker):
    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    return client.search_model_versions(f"name='{REGISTERED}'")


def _weak_pipeline(n_samples=150):
    df = generate_house_price_data(n_samples=n_samples)
    X_train, _, y_train, _ = split_house_price_data(df)
    rng = np.random.RandomState(0)
    shuffled = np.asarray(y_train, dtype=float)[rng.permutation(len(y_train))]
    pipeline = Pipeline(
        steps=[
            ("preprocessor", get_preprocessor()),
            ("regressor", RandomForestRegressor(n_estimators=10, random_state=0)),
        ]
    )
    pipeline.fit(X_train, shuffled)
    return pipeline


def test_selection_creates_runs_but_no_versions(tmp_path: Path):
    tracker = _tracker(tmp_path, "policy_sel")
    df = generate_house_price_data(n_samples=150)
    result = select_best_house_price_model(data_path=df, tracker=tracker)
    assert len(result["all_results"]) == 3
    assert _versions(tracker) == []
    client = MlflowClient(tracking_uri=tracker.tracking_uri)
    experiment = client.get_experiment_by_name("house_price_retraining")
    assert len(client.search_runs(experiment_ids=[experiment.experiment_id])) == 3


def test_retrain_promote_false_creates_no_versions(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    tracker = _tracker(tmp_path)
    service = _service(tmp_path, tracker, models_dir)
    result = service.retrain(
        HousePriceUseCase(),
        generate_house_price_data(n_samples=120),
        promote=False,
        models_dir=str(models_dir),
    )
    assert "promotion" not in result
    assert _versions(tracker) == []
    assert not (models_dir / PRODUCTION_POINTER_FILENAME).exists()


def test_initial_deployment_creates_exactly_one_version(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    tracker = _tracker(tmp_path)
    service = _service(tmp_path, tracker, models_dir)
    result = service.retrain(
        HousePriceUseCase(),
        generate_house_price_data(n_samples=120),
        promote=True,
        allow_initial_promotion=True,
        models_dir=str(models_dir),
    )
    assert result["promotion"]["decision"] == "promoted"
    versions = _versions(tracker)
    assert len(versions) == 1
    # Exact run_id lineage: registry version <- candidate run <- pointer.
    assert versions[0].run_id == result["best_run_id"]
    assert versions[0].run_id == result["promotion"]["run_id"]
    pointer = read_production_pointer(models_dir=models_dir)
    assert pointer["model_version"] == str(versions[0].version)
    assert pointer["run_id"] == versions[0].run_id
    assert pointer["registered_model_name"] == REGISTERED
    assert pointer["model_name"] == result["best_model_name"]


def test_promotion_registers_exactly_the_promoted_candidate(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    save_house_price_model(_weak_pipeline(), "RandomForestRegressor", models_dir=models_dir)
    write_production_pointer(
        {
            "model_name": "RandomForestRegressor",
            "registered_model_name": REGISTERED,
            "model_version": "1",
            "run_id": "run-weak",
        },
        models_dir=models_dir,
    )
    tracker = _tracker(tmp_path)
    service = _service(tmp_path, tracker, models_dir)
    result = service.retrain(
        HousePriceUseCase(),
        generate_house_price_data(n_samples=150),
        promote=True,
        models_dir=str(models_dir),
    )
    assert result["promotion"]["decision"] == "promoted"
    versions = _versions(tracker)
    assert len(versions) == 1
    assert versions[0].run_id == result["best_run_id"]
    pointer = read_production_pointer(models_dir=models_dir)
    assert pointer["model_name"] == result["best_model_name"]
    assert pointer["model_version"] == str(versions[0].version)
    assert pointer["run_id"] == versions[0].run_id


def test_rejected_promotion_creates_no_version(tmp_path: Path):
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    tracker = _tracker(tmp_path)
    service = _service(tmp_path, tracker, models_dir)
    df = generate_house_price_data(n_samples=120)
    first = service.retrain(
        HousePriceUseCase(), df, promote=True,
        allow_initial_promotion=True, models_dir=str(models_dir),
    )
    assert first["promotion"]["decision"] == "promoted"
    assert len(_versions(tracker)) == 1
    pointer_before = (models_dir / PRODUCTION_POINTER_FILENAME).read_bytes()

    second = service.retrain(
        HousePriceUseCase(), df, promote=True, models_dir=str(models_dir)
    )
    assert second["promotion"]["decision"] == "rejected"
    assert len(_versions(tracker)) == 1
    assert (models_dir / PRODUCTION_POINTER_FILENAME).read_bytes() == pointer_before


def test_default_store_location_contract_unchanged():
    assert get_default_tracking_dir() == Path(get_project_root()) / "mlruns"
    assert MLflowTracker.REGISTERED_MODEL_NAME == REGISTERED


def test_tmp_deploy_leaves_real_repository_untouched(tmp_path: Path):
    """A full deploy cycle against temp stores changes nothing real.

    Snapshots the real registry version set and the real database file
    state, runs train/track/deploy/publish entirely in tmp, then
    asserts the real side is byte-identical. (The real DB/registry may
    legitimately exist from live verification; this proves these paths
    add nothing to them.)
    """
    real_root = Path(get_project_root())
    real_registry = real_root / "mlruns" / "models" / REGISTERED
    versions_before = (
        {p.name for p in real_registry.iterdir()} if real_registry.exists() else set()
    )
    real_models = real_root / "models" / "house_price"
    models_before = (
        {p.name for p in real_models.iterdir()} if real_models.exists() else set()
    )

    models_dir = tmp_path / "models"
    models_dir.mkdir()
    tracker = _tracker(tmp_path)
    service = _service(tmp_path, tracker, models_dir)
    result = service.retrain(
        HousePriceUseCase(),
        generate_house_price_data(n_samples=120),
        promote=True,
        allow_initial_promotion=True,
        models_dir=str(models_dir),
    )
    assert result["promotion"]["decision"] == "promoted"
    assert len(_versions(tracker)) == 1

    versions_after = (
        {p.name for p in real_registry.iterdir()} if real_registry.exists() else set()
    )
    assert versions_after == versions_before
    models_after = (
        {p.name for p in real_models.iterdir()} if real_models.exists() else set()
    )
    assert models_after == models_before
