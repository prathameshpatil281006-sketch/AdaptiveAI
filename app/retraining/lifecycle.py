"""Explicit retraining lifecycle for AdaptiveAI (Phase 5).

Safe, explainable, reproducible model lifecycle built ONLY on the
existing architecture (shared training path, evaluator, decision,
promoter, pointer authority, MLflow registry ``HousePriceModel``):

    monitoring summary
        |
        v
    recommend() -- pure advice, never trains/promotes
        |
        v
    start_run() -- trains via RetrainingService (promote=False),
                   stages the best pipeline, evaluates vs production,
                   gates, records everything. NEVER promotes.
        |
        v
    promotion_check() -- fresh gate vs CURRENT production. Records
                         a promotion_check event. NEVER promotes.
        |
        v
    promote() -- explicit only: re-gates, promotes through the
                 existing ModelPromoter, moves the pointer atomically,
                 records history + audit. Any gate/validation failure
                 leaves the pointer untouched. Schema-migration runs
                 additionally require allow_migration=True confirmation.
         |
         v
     rollback() -- explicit only: restores a history-recorded previous
                   pointer after artifact + schema validation.

Schema-migration mode: when the serving production artifact loads but
cannot predict on the current feature schema, the baseline comparison
is recorded as ``incompatible_baseline`` (never a regression
comparison, never a crash): the run still completes with the staged
schema-v2 candidate, and promotion proceeds only through the absolute
migration gate plus explicit human confirmation. Normal compatible
upgrades always use the relative 8-gate path unchanged.

Reproducibility: fixed config seed, versioned reference dataset
(hash recorded per run), deterministic split, pinned candidate
hyperparameters. Two runs on the same data yield identical metrics
(run IDs and timestamps differ).

No second registry, no background work, no automatic promotion, no
SQLite work except through injected repositories, no pipeline objects
ever returned to callers outside this process (API layer only sees
JSON-safe dicts).
"""

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from app.core.config import resolve_project_path
from app.core.exceptions import AdaptiveAIError, ModelPromotionError, RetrainingError
from app.database.lifecycle_repository import (
    LifecycleEventRepository,
    PointerHistoryRepository,
    RetrainingRunRepository,
)
from app.engine.ml_engine import MLEngine
from app.engine.mlflow_tracker import MLflowTracker
from app.retraining.decision import recommend_retraining
from app.retraining.model_evaluator import ModelEvaluator
from app.retraining.promotion_gate import (
    MIGRATION_REQUIRED_MESSAGE,
    REGRESSION_TOLERANCE_R2,
    check_migration_eligibility,
    check_promotion_eligibility,
)
from app.retraining.model_promoter import ModelPromoter
from app.retraining.retraining_service import RetrainingService
from app.use_cases.base import BaseUseCase

logger = logging.getLogger(__name__)

STAGING_SUBDIR = ".staging"

PROMOTION_DECIDED_PROMOTED = "promoted"
PROMOTION_DECIDED_BLOCKED = "blocked"


def _utc_now_iso() -> str:
    """Returns the current UTC time as a timezone-aware ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _new_run_id() -> str:
    """Generates a backend-side run identifier (unique, unguessable)."""
    return f"rtr_{uuid.uuid4().hex[:12]}"


def _finite_or_none(value: Any) -> Optional[float]:
    """Converts a finite number to float, else None (never raises)."""
    import math

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    as_float = float(value)
    return as_float if math.isfinite(as_float) else None


def build_baseline_comparison(
    candidate_metrics: Optional[Dict[str, Any]],
    production_metrics: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Builds the production-vs-candidate comparison from real metrics.

    Deltas (documented formulas, positive favors the candidate for
    errors, positive favors the candidate for R2):

    - ``mae_delta = production_mae - candidate_mae`` (same for RMSE),
    - ``mae_reduction_pct = 100 * mae_delta / production_mae``
      (None when production MAE is not positive),
    - ``r2_improvement = candidate_r2 - production_r2``,
    - ``r2_improvement_pct = 100 * r2_improvement / |production_r2|``
      (None when production R2 is ~0).

    No subjective verdict is attached — the gate decides eligibility
    separately. Production None (no baseline) yields null production
    and null deltas without inventing numbers.
    """
    candidate = dict(candidate_metrics) if isinstance(candidate_metrics, dict) else None
    production = (
        dict(production_metrics) if isinstance(production_metrics, dict) else None
    )
    deltas: Optional[Dict[str, Any]] = None
    if candidate is not None and production is not None:
        cand_mae = _finite_or_none(candidate.get("mae"))
        prod_mae = _finite_or_none(production.get("mae"))
        cand_rmse = _finite_or_none(candidate.get("rmse"))
        prod_rmse = _finite_or_none(production.get("rmse"))
        cand_r2 = _finite_or_none(candidate.get("r2"))
        prod_r2 = _finite_or_none(production.get("r2"))
        if None not in (cand_mae, prod_mae, cand_rmse, prod_rmse, cand_r2, prod_r2):
            assert cand_mae is not None and prod_mae is not None
            assert cand_rmse is not None and prod_rmse is not None
            assert cand_r2 is not None and prod_r2 is not None
            mae_delta = prod_mae - cand_mae
            rmse_delta = prod_rmse - cand_rmse
            r2_improvement = cand_r2 - prod_r2
            deltas = {
                "mae_delta": float(mae_delta),
                "rmse_delta": float(rmse_delta),
                "r2_improvement": float(r2_improvement),
                "mae_reduction_pct": (
                    float(100.0 * mae_delta / prod_mae) if prod_mae > 0 else None
                ),
                "rmse_reduction_pct": (
                    float(100.0 * rmse_delta / prod_rmse) if prod_rmse > 0 else None
                ),
                "r2_improvement_pct": (
                    float(100.0 * r2_improvement / abs(prod_r2))
                    if abs(prod_r2) > 1e-12
                    else None
                ),
            }
    return {
        "production": production,
        "candidate": candidate,
        "deltas": deltas,
    }


class RetrainingLifecycleService:
    """Explicit, audited retraining lifecycle for House Price."""

    def __init__(
        self,
        engine: Optional[MLEngine] = None,
        tracker: Optional[MLflowTracker] = None,
        promoter: Optional[ModelPromoter] = None,
        run_repository: Optional[RetrainingRunRepository] = None,
        event_repository: Optional[LifecycleEventRepository] = None,
        pointer_history_repository: Optional[PointerHistoryRepository] = None,
        monitoring_provider: Optional[Any] = None,
        models_dir: Optional[Union[str, Path]] = None,
        staging_subdir: str = STAGING_SUBDIR,
        data_path: Optional[Union[str, Path, pd.DataFrame]] = None,
        regression_tolerance: float = REGRESSION_TOLERANCE_R2,
    ) -> None:
        """Wires lifecycle dependencies (all injectable for tests)."""
        if engine is not None and not isinstance(engine, MLEngine):
            raise RetrainingError("engine must be an MLEngine instance or None.")
        if tracker is not None and not isinstance(tracker, MLflowTracker):
            raise RetrainingError("tracker must be an MLflowTracker instance or None.")
        if promoter is not None and not isinstance(promoter, ModelPromoter):
            raise RetrainingError("promoter must be a ModelPromoter instance or None.")
        for label, repo, cls in (
            ("run_repository", run_repository, RetrainingRunRepository),
            ("event_repository", event_repository, LifecycleEventRepository),
            (
                "pointer_history_repository",
                pointer_history_repository,
                PointerHistoryRepository,
            ),
        ):
            if repo is not None and not isinstance(repo, cls):
                raise RetrainingError(f"{label} must be a {cls.__name__} instance or None.")
        if models_dir is not None and not isinstance(models_dir, (str, Path)):
            raise RetrainingError("models_dir must be a path string, Path, or None.")
        if not isinstance(staging_subdir, str) or not staging_subdir.strip():
            raise RetrainingError("staging_subdir must be a non-empty string.")
        tolerance = _finite_or_none(regression_tolerance)
        if tolerance is None:
            raise RetrainingError("regression_tolerance must be a finite number.")
        self._engine = engine if engine is not None else MLEngine()
        self._tracker = tracker
        self._promoter = promoter
        self._run_repo = run_repository
        self._event_repo = event_repository
        self._pointer_history_repo = pointer_history_repository
        self._monitoring_provider = monitoring_provider
        self._models_dir = models_dir
        self._staging_subdir = staging_subdir.strip()
        self._data_path = data_path
        self._regression_tolerance = float(tolerance)

    # -- repositories ----------------------------------------------------

    def _require_run_repo(self) -> RetrainingRunRepository:
        if self._run_repo is None:
            raise RetrainingError("A run repository is required for lifecycle runs.")
        return self._run_repo

    def _require_event_repo(self) -> LifecycleEventRepository:
        if self._event_repo is None:
            raise AdaptiveAIError("An event repository is required for audit.")
        return self._event_repo

    # -- decision ----------------------------------------------------------

    def recommend(
        self,
        summary: Optional[Dict[str, Any]],
        production_model: Optional[str] = None,
        production_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Returns a retraining recommendation (advice only, no side effects)."""
        return recommend_retraining(
            summary,
            production_model=production_model,
            production_version=production_version,
        )

    # -- runs ---------------------------------------------------------------

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Returns a recorded run, or None when unknown."""
        return self._require_run_repo().get_run(run_id)

    def list_runs(
        self, limit: int = 100, offset: int = 0, use_case: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Lists recorded runs newest-first."""
        return self._require_run_repo().list_runs(
            limit=limit, offset=offset, use_case=use_case
        )

    def start_run(
        self,
        production_limit: int = 1000,
        trigger: str = "manual",
        data_path: Optional[Union[str, Path, pd.DataFrame]] = None,
    ) -> Dict[str, Any]:
        """Trains, stages, evaluates, and gates a candidate (never promotes).

        A failed training/evaluation records a ``failed`` run (with
        ``failure_reason``) and returns it; only ledger infrastructure
        failures raise. The production pointer is never touched here.
        """
        from app.use_cases.house_price.training import (
            load_house_price_dataframe,
            prepare_house_price_splits,
        )
        from app.use_cases.house_price.use_case import HousePriceUseCase

        if isinstance(production_limit, bool) or not isinstance(production_limit, int):
            raise RetrainingError("production_limit must be a positive integer.")
        if production_limit < 1:
            raise RetrainingError("production_limit must be a positive integer.")
        if not isinstance(trigger, str) or not trigger.strip():
            raise RetrainingError("trigger must be a non-empty string.")
        run_repo = self._require_run_repo()
        event_repo = self._require_event_repo()
        use_case: BaseUseCase = HousePriceUseCase()
        resolved_data = data_path if data_path is not None else self._data_path

        try:
            from app.core.config import load_config

            seed = load_config().get("random_seed", 42)
            seed = int(seed) if isinstance(seed, int) and not isinstance(seed, bool) else 42
        except Exception:
            seed = 42

        try:
            df = load_house_price_dataframe(
                resolved_data
                if resolved_data is not None
                else "data/reference/house_price_reference.csv"
            )
        except Exception as e:
            raise RetrainingError(f"Failed to load training data: {e}") from e

        dataset_rows = int(len(df))
        dataset_ref = self._dataset_reference(resolved_data, dataset_rows)
        pointer = self._read_pointer()
        prod_model = pointer.get("model_name") if pointer else None
        prod_version = pointer.get("model_version") if pointer else None
        schema_version = (
            int(use_case.schema_version)
            if isinstance(getattr(use_case, "schema_version", None), int)
            else None
        )
        feature_schema = list(use_case.get_feature_names())
        candidates_config = [
            {
                "model_type": c.get("model_type"),
                "hyperparameters": dict(c.get("hyperparameters", {}) or {}),
            }
            for c in use_case.get_model_candidates()
        ]
        run_id = _new_run_id()
        try:
            X_train, X_test, y_train, y_test = prepare_house_price_splits(df)
        except Exception as split_error:
            # Preparation failures still open a failed run (dataset facts
            # are known); only unloadable data raises with no run at all.
            failed = run_repo.create_run(
                run_id=run_id,
                use_case=use_case.name,
                trigger=str(trigger).strip(),
                dataset_ref=dataset_ref,
                dataset_rows=dataset_rows,
                schema_version=schema_version,
                random_seed=int(seed),
                feature_schema=feature_schema,
                candidates=candidates_config,
                production_model=prod_model,
                production_version=prod_version,
            )
            failed = self._mark_run_failed(
                run_repo,
                event_repo,
                failed,
                use_case.name,
                f"Failed to prepare training data: {split_error}",
            )
            return failed
        run = run_repo.create_run(
            run_id=run_id,
            use_case=use_case.name,
            trigger=str(trigger).strip(),
            dataset_ref=dataset_ref,
            dataset_rows=dataset_rows,
            schema_version=schema_version,
            train_rows=int(len(X_train)),
            test_rows=int(len(X_test)),
            random_seed=int(seed),
            feature_schema=feature_schema,
            candidates=candidates_config,
            production_model=prod_model,
            production_version=prod_version,
        )
        event_repo.record_event(
            use_case=use_case.name,
            event="retraining_started",
            run_id=run_id,
            reason=f"Retraining started (trigger: {str(trigger).strip()}).",
        )
        try:
            training_service = RetrainingService(
                engine=self._engine, tracker=self._tracker
            )
            trained = training_service.retrain(
                use_case, df, promote=False
            )
            best_name = trained["best_model_name"]
            best_metrics = {
                "mae": float(trained["best_metrics"]["mae"]),
                "rmse": float(trained["best_metrics"]["rmse"]),
                "r2": float(trained["best_metrics"]["r2"]),
            }
            best_pipeline = trained["best_pipeline"]
            candidate_run_id = str(trained["best_run_id"])
            all_results = [
                {
                    "model_name": entry["model_name"],
                    "metrics": {
                        "mae": float(entry["metrics"]["mae"]),
                        "rmse": float(entry["metrics"]["rmse"]),
                        "r2": float(entry["metrics"]["r2"]),
                    },
                    "run_id": str(entry["run_id"]),
                }
                for entry in trained.get("all_results", [])
            ]

            artifact_name = f"{self._staging_subdir}/{run_id}"
            self._save_staged(best_pipeline, artifact_name)
            # Baseline comparison always runs on the canonical reference
            # holdout (shared evaluation set), so comparisons stay
            # comparable across runs regardless of training data.
            comparison = self._compare_against_production(
                best_pipeline, best_name
            )
            if comparison.get("status") == "incompatible_baseline":
                # Schema migration: production cannot serve the current
                # schema, so no regression comparison exists. The run
                # still completes (candidate staged, metrics recorded)
                # and promotion proceeds only through the explicit
                # migration gate with human confirmation.
                migration_gate = self._evaluate_migration_gate(
                    run_completed=True,
                    artifact_name=artifact_name,
                    candidate_metrics=best_metrics,
                    candidate_schema_version=schema_version,
                    window_limit=production_limit,
                )
                migrated = run_repo.update_run(
                    run_id,
                    status="completed",
                    completed_at=_utc_now_iso(),
                    best_model=str(best_name),
                    best_metrics={**best_metrics, "all_results": all_results},
                    comparison=comparison,
                    candidate_artifact=artifact_name,
                    candidate_run_id=candidate_run_id,
                    promotion_eligible=bool(migration_gate["eligible"]),
                    gate_checks=migration_gate["checks"],
                )
                event_repo.record_event(
                    use_case=use_case.name,
                    event="retraining_completed",
                    run_id=run_id,
                    model_name=str(best_name),
                    reason=(
                        f"Retraining completed: best {best_name} "
                        f"R2 {best_metrics['r2']:.4f}; "
                        f"{MIGRATION_REQUIRED_MESSAGE} "
                        f"Migration eligibility: {migration_gate['eligible']}."
                    ),
                    details={"best_metrics": best_metrics},
                )
                return migrated
            gate = self._evaluate_gate(
                run_completed=True,
                artifact_name=artifact_name,
                candidate_metrics=best_metrics,
                comparison=comparison,
                window_limit=production_limit,
            )
            completed = run_repo.update_run(
                run_id,
                status="completed",
                completed_at=_utc_now_iso(),
                best_model=str(best_name),
                best_metrics={**best_metrics, "all_results": all_results},
                comparison=comparison,
                candidate_artifact=artifact_name,
                candidate_run_id=candidate_run_id,
                promotion_eligible=bool(gate["eligible"]),
                gate_checks=gate["checks"],
            )
            event_repo.record_event(
                use_case=use_case.name,
                event="retraining_completed",
                run_id=run_id,
                model_name=str(best_name),
                reason=(
                    f"Retraining completed: best {best_name} "
                    f"R2 {best_metrics['r2']:.4f}; "
                    f"promotion eligibility: {gate['eligible']}."
                ),
                details={"best_metrics": best_metrics},
            )
            return completed
        except Exception as e:
            # Any training/evaluation/staging failure records a failed
            # run (with reason) and returns it; only ledger
            # infrastructure failures inside the helper raise instead.
            failed_reason = (
                str(e)
                if isinstance(e, (RetrainingError, AdaptiveAIError))
                else f"Retraining failed: {e}"
            )
            return self._mark_run_failed(
                run_repo, event_repo, run, use_case.name, failed_reason
            )

    @staticmethod
    def _mark_run_failed(
        run_repo: Any,
        event_repo: Any,
        run: Dict[str, Any],
        use_case_name: str,
        reason: str,
    ) -> Dict[str, Any]:
        """Marks a created run failed (best-effort audit) and returns it."""
        run_id = run.get("run_id", "unknown")
        try:
            failed = run_repo.update_run(
                str(run_id),
                status="failed",
                completed_at=_utc_now_iso(),
                failure_reason=reason[:2000],
            )
        except Exception as ledger_error:
            raise RetrainingError(
                "Retraining failed and the failure could not be recorded: "
                f"{ledger_error}"
            ) from ledger_error
        try:
            event_repo.record_event(
                use_case=use_case_name,
                event="retraining_failed",
                run_id=str(run_id),
                reason=reason[:2000],
            )
        except Exception:
            pass
        logger.error(f"Retraining run {run_id} failed: {reason}")
        return failed

    # -- promotion -----------------------------------------------------------

    def promotion_check(
        self, run_id: str, window_limit: int = 1000
    ) -> Optional[Dict[str, Any]]:
        """Re-evaluates the promotion gate vs CURRENT production.

        Records a ``promotion_check`` audit event. Never promotes.
        Runs in schema-migration mode (absolute validation, explicit
        migration notice) when the recorded comparison carries status
        ``incompatible_baseline``. Returns None for unknown runs
        (callers map to 404).
        """
        run_repo = self._require_run_repo()
        event_repo = self._require_event_repo()
        run = run_repo.get_run(run_id)
        if run is None:
            return None
        if not isinstance(window_limit, int) or isinstance(window_limit, bool) or window_limit < 1:
            raise RetrainingError("window_limit must be a positive integer.")

        stored_comparison = run.get("comparison")
        if isinstance(stored_comparison, dict) and stored_comparison.get("status") == "incompatible_baseline":
            return self._migration_promotion_check(run_id, run, window_limit)

        artifact_name = run.get("candidate_artifact")
        candidate_metrics = run.get("best_metrics") or {}
        if isinstance(candidate_metrics, dict):
            candidate_metrics = {
                k: candidate_metrics.get(k) for k in ("mae", "rmse", "r2")
            }
        artifact_available = self._staged_artifact_exists(artifact_name)
        pipeline = self._load_staged(artifact_name) if artifact_available else None
        schema_compatible = self._trial_predict(pipeline) if pipeline is not None else False
        if pipeline is not None and schema_compatible:
            comparison = self._compare_against_production(
                pipeline, run.get("best_model") or "candidate"
            )
        else:
            # Incompatible or missing candidate: no fresh evaluation is
            # possible, so the gate judges stored metrics (or their
            # absence) and fails closed on schema/artifact checks.
            current_pointer = self._read_pointer()
            comparison = {
                "candidate": {
                    "model_name": run.get("best_model"),
                    "metrics": candidate_metrics
                    if isinstance(candidate_metrics, dict)
                    else None,
                },
                "production": None
                if current_pointer is None
                else {
                    "model_name": current_pointer.get("model_name"),
                    "metrics": None,
                },
                "deltas": None,
                "status": "unevaluable",
            }
        gate = self._evaluate_gate(
            run_completed=run.get("status") == "completed",
            artifact_name=artifact_name,
            artifact_available=artifact_available,
            artifact_loadable=pipeline is not None,
            schema_compatible=schema_compatible,
            candidate_metrics=candidate_metrics if isinstance(candidate_metrics, dict) else None,
            comparison=comparison,
            window_limit=window_limit,
        )
        run_repo.update_run(
            run_id,
            promotion_eligible=bool(gate["eligible"]),
            gate_checks=gate["checks"],
        )
        current = self._read_pointer()
        event_repo.record_event(
            use_case=run.get("use_case", "house_price"),
            event="promotion_check",
            run_id=run_id,
            model_name=run.get("best_model"),
            reason=(
                f"Promotion gate {'eligible' if gate['eligible'] else 'not eligible'} "
                f"for candidate '{run.get('best_model')}'."
            ),
            details={"eligible": gate["eligible"], "checks": gate["checks"]},
        )
        return {
            "run_id": run_id,
            "eligible": bool(gate["eligible"]),
            "checks": gate["checks"],
            "comparison": comparison,
            "production": {
                "model_name": current.get("model_name") if current else None,
                "model_version": current.get("model_version") if current else None,
            },
        }

    def _migration_promotion_check(
        self, run_id: str, run: Dict[str, Any], window_limit: int
    ) -> Dict[str, Any]:
        """Re-evaluates the migration gate vs CURRENT production.

        Absolute candidate validation only: with an incompatible
        production baseline there is no regression comparison to
        refresh. Never promotes.
        """
        from app.use_cases.house_price.use_case import HousePriceUseCase

        run_repo = self._require_run_repo()
        event_repo = self._require_event_repo()
        artifact_name = run.get("candidate_artifact")
        candidate_metrics = run.get("best_metrics") or {}
        if isinstance(candidate_metrics, dict):
            candidate_metrics = {
                k: candidate_metrics.get(k) for k in ("mae", "rmse", "r2")
            }
        try:
            candidate_schema_version = run.get("schema_version")
            if not isinstance(candidate_schema_version, int) or isinstance(
                candidate_schema_version, bool
            ):
                candidate_schema_version = int(HousePriceUseCase().schema_version)
        except Exception:
            candidate_schema_version = run.get("schema_version")
        gate = self._evaluate_migration_gate(
            run_completed=run.get("status") == "completed",
            artifact_name=artifact_name,
            candidate_metrics=candidate_metrics
            if isinstance(candidate_metrics, dict)
            else None,
            candidate_schema_version=candidate_schema_version
            if isinstance(candidate_schema_version, int)
            and not isinstance(candidate_schema_version, bool)
            else None,
            window_limit=window_limit,
        )
        migration_block = gate.get("migration") if isinstance(gate, dict) else None
        run_repo.update_run(
            run_id,
            promotion_eligible=bool(gate["eligible"]),
            gate_checks=gate["checks"],
        )
        current = self._read_pointer()
        event_repo.record_event(
            use_case=run.get("use_case", "house_price"),
            event="promotion_check",
            run_id=run_id,
            model_name=run.get("best_model"),
            reason=(
                f"{MIGRATION_REQUIRED_MESSAGE} Migration gate "
                f"{'eligible' if gate['eligible'] else 'not eligible'} "
                f"for candidate '{run.get('best_model')}'."
            ),
            details={"eligible": gate["eligible"], "checks": gate["checks"]},
        )
        stored_comparison = run.get("comparison")
        return {
            "run_id": run_id,
            "eligible": bool(gate["eligible"]),
            "checks": gate["checks"],
            "comparison": stored_comparison,
            "production": {
                "model_name": current.get("model_name") if current else None,
                "model_version": current.get("model_version") if current else None,
            },
            "migration": migration_block,
        }

    def promote(self, run_id: str, allow_migration: bool = False) -> Optional[Dict[str, Any]]:
        """Explicitly promotes a gated candidate (pointer moves only here).

        Re-runs the promotion gate first; any failure (or a missing
        run) leaves the production pointer untouched. Returns None for
        unknown runs.

        Schema-migration runs (comparison status
        ``incompatible_baseline``) additionally require
        ``allow_migration=True`` — an explicit human confirmation.
        Without it the pointer is left untouched and the response
        explains that confirmation is required.
        """
        from app.use_cases.house_price.model_persistence import read_production_pointer

        run_repo = self._require_run_repo()
        event_repo = self._require_event_repo()
        if self._promoter is None:
            raise RetrainingError(
                "A promoter is required for explicit promotion. "
                "Inject one via RetrainingLifecycleService(promoter=...)."
            )
        run = run_repo.get_run(run_id)
        if run is None:
            return None
        use_case_name = run.get("use_case", "house_price")

        checked = self.promotion_check(run_id)
        assert checked is not None
        if isinstance(checked.get("migration"), dict) and checked["migration"].get(
            "required"
        ) is True:
            return self._promote_migration_decision(
                run_id, run, checked, use_case_name, allow_migration
            )
        if not checked["eligible"]:
            run_repo.update_run(run_id, promotion_decision=PROMOTION_DECIDED_BLOCKED)
            return {
                "run_id": run_id,
                "promoted": False,
                "eligible": False,
                "checks": checked["checks"],
                "comparison": checked["comparison"],
                "production": checked["production"],
                "reason": "Promotion gate not satisfied; production pointer unchanged.",
            }

        try:
            previous_pointer = read_production_pointer(models_dir=self._models_dir)
        except Exception as e:
            raise RetrainingError(
                f"Cannot promote without a readable production reference: {e}"
            ) from e
        artifact_name = run.get("candidate_artifact")
        pipeline = self._load_staged(artifact_name)
        if pipeline is None:
            raise RetrainingError("Staged candidate is unavailable; promotion refused.")
        # Fresh evaluation for the promoter (deterministic re-run of the
        # same comparison the gate just approved).
        fresh = self._compare_against_production(
            pipeline,
            str(run.get("best_model")),
            include_evaluation=True,
        )
        evaluation = fresh.pop("evaluation_result", None)
        if not isinstance(evaluation, dict):
            raise RetrainingError("Promotion evaluation is missing; promotion refused.")
        try:
            outcome = self._promoter.promote(
                candidate_model=pipeline,
                evaluation_result=evaluation,
                candidate_model_name=str(run.get("best_model")),
                candidate_run_id=str(run.get("candidate_run_id")),
                production_model_name=str(previous_pointer.get("model_name")),
                models_dir=self._models_dir,
            )
        except (ModelPromotionError, RetrainingError):
            raise
        except Exception as e:
            raise RetrainingError(f"Promotion failed: {e}") from e
        if not isinstance(outcome, dict) or outcome.get("decision") != "promoted":
            raise RetrainingError(
                "Promoter declined promotion; production pointer unchanged."
            )
        try:
            published = self._promoter.update_production_pointer(
                outcome, models_dir=self._models_dir, use_case=use_case_name
            )
        except (ModelPromotionError, RetrainingError):
            raise
        except Exception as e:
            raise RetrainingError(f"Production publication failed: {e}") from e

        new_version = published.get("model_version")
        run_repo.update_run(
            run_id,
            promotion_decision=PROMOTION_DECIDED_PROMOTED,
            candidate_version=str(new_version) if new_version else None,
        )
        if self._pointer_history_repo is not None:
            try:
                self._pointer_history_repo.record_transition(
                    use_case=use_case_name,
                    new_pointer=dict(published),
                    previous_pointer=dict(previous_pointer),
                    reason=f"Explicit promotion of run {run_id}.",
                    run_id=run_id,
                )
            except AdaptiveAIError as e:
                raise RetrainingError(
                    f"Promotion succeeded but pointer-history audit failed: {e}"
                ) from e
        event_repo.record_event(
            use_case=use_case_name,
            event="model_promoted",
            run_id=run_id,
            model_name=str(run.get("best_model")),
            model_version=str(new_version) if new_version else None,
            previous_production_version=str(previous_pointer.get("model_version"))
            if previous_pointer.get("model_version")
            else None,
            new_production_version=str(new_version) if new_version else None,
            reason=f"Explicit promotion of run {run_id} to v{new_version}.",
            details={"candidate_metrics": run.get("best_metrics")},
        )
        return {
            "run_id": run_id,
            "promoted": True,
            "eligible": True,
            "checks": checked["checks"],
            "comparison": fresh,
            "production": {
                "model_name": published.get("model_name"),
                "model_version": published.get("model_version"),
                "registered_model_name": published.get("registered_model_name"),
                "run_id": published.get("run_id"),
            },
            "previous_production_version": previous_pointer.get("model_version"),
            "reason": f"Run {run_id} promoted; production is now v{new_version}.",
        }

    def _promote_migration_decision(
        self,
        run_id: str,
        run: Dict[str, Any],
        checked: Dict[str, Any],
        use_case_name: str,
        allow_migration: bool,
    ) -> Dict[str, Any]:
        """Decides and optionally executes a schema-migration promotion.

        Migration promotion requires BOTH a passing migration gate AND
        explicit ``allow_migration=True`` confirmation. Any other case
        records a blocked decision and leaves the production pointer
        untouched. On the confirmed path the candidate is re-validated
        (artifact loads, trial-predicts on the current schema) and
        promoted through the existing promoter, pointer publication,
        pointer history, and audit event — the same publication
        machinery as normal upgrades.
        """
        from app.core.exceptions import ModelPromotionError
        from app.use_cases.house_price.model_persistence import read_production_pointer

        run_repo = self._require_run_repo()
        event_repo = self._require_event_repo()
        migration_block = checked.get("migration")

        def _blocked(reason: str) -> Dict[str, Any]:
            run_repo.update_run(run_id, promotion_decision=PROMOTION_DECIDED_BLOCKED)
            return {
                "run_id": run_id,
                "promoted": False,
                "eligible": bool(checked["eligible"]),
                "checks": checked["checks"],
                "comparison": checked["comparison"],
                "production": checked["production"],
                "migration": migration_block,
                "reason": reason,
            }

        if not checked["eligible"]:
            return _blocked(
                f"{MIGRATION_REQUIRED_MESSAGE} Migration gate not satisfied; "
                "production pointer unchanged."
            )
        if allow_migration is not True:
            return _blocked(
                f"{MIGRATION_REQUIRED_MESSAGE} Explicit confirmation is "
                "required (allow_migration=True); production pointer unchanged."
            )
        if self._promoter is None:
            raise RetrainingError(
                "A promoter is required for explicit promotion. "
                "Inject one via RetrainingLifecycleService(promoter=...)."
            )
        try:
            previous_pointer = read_production_pointer(models_dir=self._models_dir)
        except Exception as e:
            raise RetrainingError(
                f"Cannot promote without a readable production reference: {e}"
            ) from e
        artifact_name = run.get("candidate_artifact")
        pipeline = self._load_staged(artifact_name)
        if pipeline is None:
            raise RetrainingError("Staged candidate is unavailable; promotion refused.")
        if not self._trial_predict(pipeline):
            raise RetrainingError(
                "Staged candidate cannot serve the current schema; promotion refused."
            )
        candidate_run_id = run.get("candidate_run_id")
        if not isinstance(candidate_run_id, str) or not candidate_run_id.strip():
            raise RetrainingError(
                "Candidate is not traceable to a retraining run; promotion refused."
            )
        try:
            outcome = self._promoter.promote_schema_migration(
                candidate_model=pipeline,
                migration_assessment={
                    "eligible": bool(checked["eligible"]),
                    "checks": checked["checks"],
                    "migration": migration_block,
                },
                candidate_model_name=str(run.get("best_model")),
                candidate_run_id=str(candidate_run_id),
                production_model_name=str(previous_pointer.get("model_name")),
                models_dir=self._models_dir,
            )
        except (ModelPromotionError, RetrainingError):
            raise
        except Exception as e:
            raise RetrainingError(f"Migration promotion failed: {e}") from e
        if not isinstance(outcome, dict) or outcome.get("decision") != "promoted":
            raise RetrainingError(
                "Promoter declined migration promotion; production pointer unchanged."
            )
        try:
            published = self._promoter.update_production_pointer(
                outcome, models_dir=self._models_dir, use_case=use_case_name
            )
        except (ModelPromotionError, RetrainingError):
            raise
        except Exception as e:
            raise RetrainingError(f"Production publication failed: {e}") from e

        new_version = published.get("model_version")
        run_repo.update_run(
            run_id,
            promotion_decision=PROMOTION_DECIDED_PROMOTED,
            candidate_version=str(new_version) if new_version else None,
        )
        if self._pointer_history_repo is not None:
            try:
                self._pointer_history_repo.record_transition(
                    use_case=use_case_name,
                    new_pointer=dict(published),
                    previous_pointer=dict(previous_pointer),
                    reason=(
                        f"Explicit schema-migration promotion of run {run_id} "
                        f"from {previous_pointer.get('model_name')} "
                        f"v{previous_pointer.get('model_version')}."
                    ),
                    run_id=run_id,
                )
            except AdaptiveAIError as e:
                raise RetrainingError(
                    f"Promotion succeeded but pointer-history audit failed: {e}"
                ) from e
        event_repo.record_event(
            use_case=use_case_name,
            event="model_promoted",
            run_id=run_id,
            model_name=str(run.get("best_model")),
            model_version=str(new_version) if new_version else None,
            previous_production_version=str(previous_pointer.get("model_version"))
            if previous_pointer.get("model_version")
            else None,
            new_production_version=str(new_version) if new_version else None,
            reason=(
                f"Explicit schema-migration promotion of run {run_id} "
                f"to v{new_version}."
            ),
            details={"candidate_metrics": run.get("best_metrics")},
        )
        return {
            "run_id": run_id,
            "promoted": True,
            "eligible": True,
            "checks": checked["checks"],
            "comparison": checked["comparison"],
            "production": {
                "model_name": published.get("model_name"),
                "model_version": published.get("model_version"),
                "registered_model_name": published.get("registered_model_name"),
                "run_id": published.get("run_id"),
            },
            "previous_production_version": previous_pointer.get("model_version"),
            "migration": migration_block,
            "reason": (
                f"Run {run_id} promoted by explicit schema migration; "
                f"production is now v{new_version}."
            ),
        }

    def rollback(self, pointer_history_id: int) -> Optional[Dict[str, Any]]:
        """Restores a history-recorded previous pointer (explicit only).

        Validates the target artifact (exists, loads, predicts on the
        current schema) and refuses no-op rollbacks. On success the
        pointer moves atomically, history is appended, and an audit
        event is recorded. Returns None for unknown history ids.
        """
        from app.use_cases.house_price.model_persistence import (
            load_house_price_model,
            read_production_pointer,
            write_production_pointer,
        )

        if self._pointer_history_repo is None:
            raise RetrainingError("A pointer-history repository is required for rollback.")
        event_repo = self._require_event_repo()
        entry = self._pointer_history_repo.get_transition(pointer_history_id)
        if entry is None:
            return None
        target = entry.get("previous_pointer")
        if not isinstance(target, dict) or not target:
            raise RetrainingError(
                "This history entry records no previous production state; rollback refused."
            )
        target_name = target.get("model_name")
        if not isinstance(target_name, str) or not target_name.strip():
            raise RetrainingError("Rollback target has no model identity; rollback refused.")
        try:
            current = read_production_pointer(models_dir=self._models_dir)
        except Exception as e:
            raise RetrainingError(
                f"Cannot roll back without a readable production reference: {e}"
            ) from e
        if (
            target.get("model_name") == current.get("model_name")
            and target.get("model_version") == current.get("model_version")
            and target.get("run_id") == current.get("run_id")
        ):
            raise RetrainingError("Rollback target already serves production; refusing no-op.")
        try:
            pipeline = load_house_price_model(
                str(target_name).strip(), models_dir=self._models_dir
            )
        except Exception as e:
            raise RetrainingError(
                f"Rollback target artifact is unavailable ({e}); pointer unchanged."
            ) from e
        if not self._trial_predict(pipeline):
            raise RetrainingError(
                "Rollback target cannot serve the current schema; pointer unchanged."
            )
        try:
            published_path = write_production_pointer(
                {
                    "model_name": str(target.get("model_name")).strip(),
                    "registered_model_name": str(
                        target.get("registered_model_name", "HousePriceModel")
                    ).strip(),
                    "model_version": str(target.get("model_version")).strip(),
                    "run_id": str(target.get("run_id")).strip(),
                    "use_case": "house_price",
                },
                models_dir=self._models_dir,
            )
            del published_path
            published = read_production_pointer(models_dir=self._models_dir)
        except AdaptiveAIError:
            raise
        except Exception as e:
            raise RetrainingError(f"Rollback publication failed: {e}") from e
        try:
            self._pointer_history_repo.record_transition(
                use_case=entry.get("use_case", "house_price"),
                new_pointer=dict(published),
                previous_pointer=dict(current),
                reason=f"Explicit rollback to v{target.get('model_version')} (history #{pointer_history_id}).",
                run_id=entry.get("run_id"),
            )
        except AdaptiveAIError as e:
            raise RetrainingError(
                f"Rollback succeeded but pointer-history audit failed: {e}"
            ) from e
        event_repo.record_event(
            use_case=entry.get("use_case", "house_price"),
            event="model_rollback",
            run_id=entry.get("run_id"),
            model_name=str(target.get("model_name")),
            model_version=str(target.get("model_version"))
            if target.get("model_version")
            else None,
            previous_production_version=str(current.get("model_version"))
            if current.get("model_version")
            else None,
            new_production_version=str(published.get("model_version"))
            if published.get("model_version")
            else None,
            reason=f"Explicit rollback to v{target.get('model_version')} (history #{pointer_history_id}).",
        )
        return {
            "pointer_history_id": int(pointer_history_id),
            "rolled_back": True,
            "production": {
                "model_name": published.get("model_name"),
                "model_version": published.get("model_version"),
                "registered_model_name": published.get("registered_model_name"),
                "run_id": published.get("run_id"),
            },
            "previous_production_version": current.get("model_version"),
            "reason": f"Production rolled back to v{published.get('model_version')}.",
        }

    # -- internals ------------------------------------------------------------

    def _read_pointer(self) -> Optional[Dict[str, Any]]:
        """Reads the production pointer (None when none deployed)."""
        from app.core.exceptions import ModelNotFoundError
        from app.use_cases.house_price.model_persistence import read_production_pointer

        try:
            return read_production_pointer(models_dir=self._models_dir)
        except ModelNotFoundError:
            return None
        except Exception as e:
            raise RetrainingError(f"Production reference is unreadable: {e}") from e

    def _dataset_reference(
        self, data_path: Any, dataset_rows: int
    ) -> Optional[str]:
        """Builds a dataset identity string (path + content hash for CSVs)."""
        import hashlib

        if data_path is None:
            data_path = "data/reference/house_price_reference.csv"
        if isinstance(data_path, pd.DataFrame):
            return f"inline-dataframe:{dataset_rows}-rows"
        try:
            resolved = resolve_project_path(data_path)
        except Exception:
            return str(data_path)
        try:
            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()[:10]
            return f"{resolved.name}#sha256:{digest}"
        except Exception:
            return str(data_path)

    def _staged_path(self, artifact_name: Optional[str]) -> Optional[Path]:
        """Resolves a staging artifact name to a filesystem path."""
        if not isinstance(artifact_name, str) or not artifact_name.strip():
            return None
        from app.use_cases.house_price.model_persistence import _get_default_models_dir

        base = (
            Path(self._models_dir)
            if self._models_dir is not None
            else Path(_get_default_models_dir())
        )
        return base / f"{artifact_name.strip()}.joblib"

    def _save_staged(self, pipeline: Any, artifact_name: str) -> Path:
        """Stages a trained pipeline outside the production location."""
        from app.use_cases.house_price.model_persistence import save_house_price_model

        staged = self._staged_path(artifact_name)
        if staged is not None:
            try:
                staged.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                raise RetrainingError(
                    f"Failed to prepare staging directory: {e}"
                ) from e
        try:
            return save_house_price_model(
                pipeline, artifact_name, models_dir=self._models_dir
            )
        except Exception as e:
            raise RetrainingError(f"Failed to stage candidate artifact: {e}") from e

    def _staged_artifact_exists(self, artifact_name: Optional[str]) -> bool:
        path = self._staged_path(artifact_name)
        try:
            return path is not None and path.is_file()
        except Exception:
            return False

    def _load_staged(self, artifact_name: Optional[str]) -> Any:
        """Loads a staged pipeline (None when missing/unreadable)."""
        from app.use_cases.house_price.model_persistence import load_house_price_model

        if not self._staged_artifact_exists(artifact_name):
            return None
        try:
            assert isinstance(artifact_name, str)
            return load_house_price_model(
                artifact_name.strip(), models_dir=self._models_dir
            )
        except Exception:
            return None

    def _trial_predict(self, pipeline: Any) -> bool:
        """Trial prediction on current-schema inputs (schema compatibility)."""
        if pipeline is None or not hasattr(pipeline, "predict"):
            return False
        try:
            from app.use_cases.house_price.training import prepare_house_price_data

            _, X_test, _, _ = prepare_house_price_data()
            pipeline.predict(X_test.head(5))
            return True
        except Exception:
            return False

    @staticmethod
    def _serves_current_schema(pipeline: Any, X_test: Any) -> bool:
        """Reports whether a pipeline predicts on current-schema features."""
        if pipeline is None or not hasattr(pipeline, "predict"):
            return False
        try:
            sample = X_test.head(5) if hasattr(X_test, "head") else X_test
            pipeline.predict(sample)
            return True
        except Exception:
            return False

    @staticmethod
    def _expected_features(pipeline: Any) -> Optional[List[str]]:
        """Returns a pipeline's expected input features, or None if unknown."""
        try:
            names = getattr(pipeline, "feature_names_in_", None)
            if names is None and hasattr(pipeline, "steps") and pipeline.steps:
                names = getattr(pipeline.steps[0][1], "feature_names_in_", None)
            if names is None:
                return None
            return [str(c) for c in list(names)]
        except Exception:
            return None

    def _compare_against_production(
        self,
        candidate_pipeline: Any,
        candidate_name: str,
        X_test: Any = None,
        y_test: Any = None,
        include_evaluation: bool = False,
    ) -> Dict[str, Any]:
        """Evaluates the candidate against current production (or no baseline).

        Schema-migration case: when a production artifact loads but
        cannot predict on the current-schema holdout while the
        candidate can, the baseline is explicitly unmeasurable. The
        comparison is then recorded with ``status ==
        "incompatible_baseline"`` (never a regression comparison, never
        an exception): the run completes, the candidate is staged, and
        promotion proceeds only through the explicit migration gate
        with human confirmation. A candidate that itself cannot predict
        still fails the run as before.
        """
        from app.core.exceptions import ModelNotFoundError
        from app.use_cases.house_price.model_persistence import load_house_price_model
        from app.use_cases.house_price.use_case import HousePriceUseCase

        if candidate_pipeline is None:
            raise RetrainingError("Candidate pipeline is missing; cannot evaluate.")
        if X_test is None or y_test is None:
            from app.use_cases.house_price.training import prepare_house_price_data

            _, X_test, _, y_test = prepare_house_price_data()
        current_schema_version = int(HousePriceUseCase().schema_version)
        pointer = self._read_pointer()
        production_model = None
        production_name = None
        if pointer is not None:
            production_name = pointer.get("model_name")
            try:
                production_model = load_house_price_model(
                    str(production_name), models_dir=self._models_dir
                )
            except ModelNotFoundError:
                production_model = None
                production_name = None
            except Exception as e:
                raise RetrainingError(
                    f"Production model failed to load for comparison: {e}"
                ) from e
        evaluator = ModelEvaluator()
        if production_model is not None and not self._serves_current_schema(
            production_model, X_test
        ):
            # Loaded production cannot serve the current schema: explicit
            # incompatible baseline (not a crash, not a comparison).
            if not self._serves_current_schema(candidate_pipeline, X_test):
                raise RetrainingError(
                    "Candidate evaluation failed during predict on the "
                    "current schema; run cannot complete."
                )
            try:
                candidate_only = evaluator.evaluate_candidate(
                    candidate_model=candidate_pipeline,
                    production_model=None,
                    X_test=X_test,
                    y_test=y_test,
                    candidate_model_name=str(candidate_name),
                    production_model_name=None,
                    models_dir=self._models_dir,
                )
            except RetrainingError:
                raise
            except Exception as e:
                raise RetrainingError(f"Candidate evaluation failed: {e}") from e
            candidate_metrics = dict(candidate_only["candidate"]["metrics"])
            expected = self._expected_features(production_model)
            reason = (
                f"{MIGRATION_REQUIRED_MESSAGE} Production "
                f"'{production_name}' expects "
                f"{expected if expected is not None else 'legacy features'} "
                f"and cannot predict on current schema v{current_schema_version}."
            )
            comparison: Dict[str, Any] = {
                "candidate": {
                    "model_name": candidate_only["candidate"]["model_name"],
                    "metrics": dict(candidate_metrics),
                },
                "production": {
                    "model_name": str(production_name),
                    "metrics": None,
                    "expected_features": expected,
                },
                "deltas": None,
                "status": "incompatible_baseline",
                "reason": reason,
                "candidate_schema_version": int(current_schema_version),
                "current_schema_version": int(current_schema_version),
                "production_expected_features": expected,
                "migration": {
                    "required": True,
                    "message": MIGRATION_REQUIRED_MESSAGE,
                    "candidate_schema_version": int(current_schema_version),
                    "current_schema_version": int(current_schema_version),
                },
            }
            comparison["candidate_model_name"] = candidate_only["candidate"]["model_name"]
            comparison["production_model_name"] = str(production_name)
            if include_evaluation:
                comparison["evaluation_result"] = candidate_only
            return comparison
        try:
            result = evaluator.evaluate_candidate(
                candidate_model=candidate_pipeline,
                production_model=production_model,
                X_test=X_test,
                y_test=y_test,
                candidate_model_name=str(candidate_name),
                production_model_name=str(production_name)
                if production_name
                else None,
                models_dir=self._models_dir,
            )
        except RetrainingError:
            raise
        except Exception as e:
            raise RetrainingError(f"Candidate evaluation failed: {e}") from e
        candidate_metrics = dict(result["candidate"]["metrics"])
        production_block = result.get("production")
        production_metrics = (
            dict(production_block["metrics"]) if isinstance(production_block, dict) else None
        )
        base = build_baseline_comparison(candidate_metrics, production_metrics)
        comparison = {
            "candidate": {
                "model_name": result["candidate"]["model_name"],
                "metrics": base["candidate"],
            },
            "production": None
            if base["production"] is None
            else {
                "model_name": production_block.get("model_name")
                if isinstance(production_block, dict)
                else None,
                "metrics": base["production"],
            },
            "deltas": base["deltas"],
        }
        comparison["status"] = result.get("status")
        comparison["candidate_model_name"] = result["candidate"]["model_name"]
        comparison["production_model_name"] = (
            production_block.get("model_name") if isinstance(production_block, dict) else None
        )
        if include_evaluation:
            comparison["evaluation_result"] = result
        return comparison

    def _evaluate_gate(
        self,
        run_completed: bool,
        artifact_name: Optional[str],
        candidate_metrics: Optional[Dict[str, Any]] = None,
        comparison: Optional[Dict[str, Any]] = None,
        window_limit: int = 1000,
        artifact_available: Optional[bool] = None,
        artifact_loadable: Optional[bool] = None,
        schema_compatible: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Gathers gate evidence (I/O) and judges it (pure gate)."""
        from app.monitoring.data_quality import run_house_price_data_quality

        if artifact_available is None:
            artifact_available = self._staged_artifact_exists(artifact_name)
        pipeline = None
        if artifact_loadable is None:
            pipeline = self._load_staged(artifact_name) if artifact_available else None
            artifact_loadable = pipeline is not None
        if schema_compatible is None:
            if pipeline is None and artifact_loadable:
                pipeline = self._load_staged(artifact_name)
            schema_compatible = self._trial_predict(pipeline)
        comparison = comparison if isinstance(comparison, dict) else {}
        deltas = comparison.get("deltas") if isinstance(comparison.get("deltas"), dict) else {}
        production_block = comparison.get("production")
        production_metrics = (
            production_block.get("metrics")
            if isinstance(production_block, dict)
            and isinstance(production_block.get("metrics"), dict)
            else None
        )
        dq_status: Optional[str] = None
        if self._monitoring_provider is not None:
            try:
                window = self._monitoring_provider.get_full_monitoring_data(
                    limit=window_limit
                )
                dq_status = run_house_price_data_quality(window["raw_rows"]).get("status")
            except Exception as e:
                logger.warning(f"Gate data-quality assessment failed: {e}")
                dq_status = None
        return check_promotion_eligibility(
            run_completed=run_completed is True,
            artifact_available=artifact_available is True,
            artifact_loadable=artifact_loadable is True,
            schema_compatible=schema_compatible is True,
            candidate_metrics=candidate_metrics
            if isinstance(candidate_metrics, dict)
            else None,
            production_metrics=production_metrics
            if isinstance(production_metrics, dict)
            else None,
            improvement=deltas.get("r2_improvement"),
            data_quality_status=dq_status,
            regression_tolerance=self._regression_tolerance,
        )

    def _evaluate_migration_gate(
        self,
        run_completed: bool,
        artifact_name: Optional[str],
        candidate_metrics: Optional[Dict[str, Any]] = None,
        candidate_schema_version: Optional[int] = None,
        window_limit: int = 1000,
        artifact_available: Optional[bool] = None,
        artifact_loadable: Optional[bool] = None,
        schema_compatible: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Gathers migration-gate evidence (I/O) and judges it (pure gate)."""
        from app.monitoring.data_quality import run_house_price_data_quality
        from app.use_cases.house_price.use_case import HousePriceUseCase

        if artifact_available is None:
            artifact_available = self._staged_artifact_exists(artifact_name)
        pipeline = None
        if artifact_loadable is None:
            pipeline = self._load_staged(artifact_name) if artifact_available else None
            artifact_loadable = pipeline is not None
        if schema_compatible is None:
            if pipeline is None and artifact_loadable:
                pipeline = self._load_staged(artifact_name)
            schema_compatible = self._trial_predict(pipeline)
        try:
            current_schema_version = int(HousePriceUseCase().schema_version)
        except Exception:
            current_schema_version = candidate_schema_version
        dq_status: Optional[str] = None
        if self._monitoring_provider is not None:
            try:
                window = self._monitoring_provider.get_full_monitoring_data(
                    limit=window_limit
                )
                dq_status = run_house_price_data_quality(window["raw_rows"]).get("status")
            except Exception as e:
                logger.warning(f"Migration gate data-quality assessment failed: {e}")
                dq_status = None
        return check_migration_eligibility(
            run_completed=run_completed is True,
            artifact_available=artifact_available is True,
            artifact_loadable=artifact_loadable is True,
            schema_compatible=schema_compatible is True,
            candidate_metrics=candidate_metrics
            if isinstance(candidate_metrics, dict)
            else None,
            candidate_schema_version=candidate_schema_version
            if isinstance(candidate_schema_version, int)
            and not isinstance(candidate_schema_version, bool)
            else None,
            current_schema_version=current_schema_version
            if isinstance(current_schema_version, int)
            and not isinstance(current_schema_version, bool)
            else None,
            data_quality_status=dq_status,
        )


def create_house_price_lifecycle(
    engine: Optional[MLEngine] = None,
    tracker: Optional[MLflowTracker] = None,
    promoter: Optional[ModelPromoter] = None,
    run_repository: Optional[RetrainingRunRepository] = None,
    event_repository: Optional[LifecycleEventRepository] = None,
    pointer_history_repository: Optional[PointerHistoryRepository] = None,
    monitoring_provider: Optional[Any] = None,
    models_dir: Optional[Union[str, Path]] = None,
    data_path: Optional[Union[str, Path, pd.DataFrame]] = None,
    regression_tolerance: float = REGRESSION_TOLERANCE_R2,
) -> RetrainingLifecycleService:
    """Small factory wiring the lifecycle service for House Price."""
    return RetrainingLifecycleService(
        engine=engine,
        tracker=tracker,
        promoter=promoter,
        run_repository=run_repository,
        event_repository=event_repository,
        pointer_history_repository=pointer_history_repository,
        monitoring_provider=monitoring_provider,
        models_dir=models_dir,
        data_path=data_path,
        regression_tolerance=regression_tolerance,
    )
