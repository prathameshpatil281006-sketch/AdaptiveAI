"""Demo-mode service for AdaptiveAI (Phase 6 Scenario Simulator).

Orchestrates the existing Phase 4 / Phase 5 machinery over ISOLATED
simulated traffic — no second pipeline, no duplicated monitoring or
training logic:

    scenario samples (deterministic)
        |
        v
    predictions rows with source='demo' + session id
        |
        v
    MonitoringService over a demo-scoped provider
    (monitoring_repository=None: demo results live on the session
    record and NEVER enter ``monitoring_results``)
        |
        v
    recommend_retraining() (pure advice, unchanged)
        |
        v
    RetrainingLifecycleService.start_run() with a ``demo:`` trigger
    (existing training path, never promotes)

Production safety properties (each covered by tests):

- Generating demo samples never changes the production prediction
  count (filtered reads exclude ``source='demo'``).
- Demo monitoring never writes to ``monitoring_results``.
- Reset deletes ONLY ``source='demo'`` rows of the session; it never
  touches production predictions, models, the reference dataset, the
  production pointer, or audit history.
- Retraining and promotion checks never move the production pointer;
  only the existing explicit ``promote()`` can, and the demo UI warns
  about it explicitly.
"""

import logging
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from app.core.exceptions import AdaptiveAIError
from app.database.lifecycle_repository import LifecycleEventRepository
from app.database.repository import PredictionRepository, utc_now_iso
from app.demo.repository import DemoSessionRepository
from app.demo.scenarios import (
    SUPPORTED_SCENARIOS,
    generate_scenario_samples,
)

logger = logging.getLogger(__name__)

USE_CASE = "house_price"
DEMO_SOURCE = "demo"
DEMO_MONITOR_WINDOW = 10000


def _new_session_id() -> str:
    """Generates a backend-side demo session identifier."""
    return f"demo-{uuid.uuid4().hex[:8]}"


class DemoService:
    """Isolated scenario-simulator orchestration for House Price."""

    def __init__(
        self,
        prediction_repository: PredictionRepository,
        session_repository: DemoSessionRepository,
        event_repository: Optional[LifecycleEventRepository] = None,
        models_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Wires demo dependencies (all injectable for tests)."""
        if not isinstance(prediction_repository, PredictionRepository):
            raise AdaptiveAIError(
                "prediction_repository must be a PredictionRepository instance."
            )
        if not isinstance(session_repository, DemoSessionRepository):
            raise AdaptiveAIError(
                "session_repository must be a DemoSessionRepository instance."
            )
        if event_repository is not None and not isinstance(
            event_repository, LifecycleEventRepository
        ):
            raise AdaptiveAIError(
                "event_repository must be a LifecycleEventRepository instance or None."
            )
        self._predictions = prediction_repository
        self._sessions = session_repository
        self._events = event_repository
        self._models_dir = models_dir

    # -- sessions ---------------------------------------------------------

    def create_session(
        self,
        scenario: str,
        sample_count: int = 100,
        seed: int = 42,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Creates a demo session (validates the scenario eagerly).

        Records a ``demo_session_created`` audit event. Creates no
        samples yet — call ``generate_samples`` next.
        """
        if scenario not in SUPPORTED_SCENARIOS:
            raise AdaptiveAIError(
                "Unknown demo scenario "
                f"'{scenario}'. Supported: {', '.join(SUPPORTED_SCENARIOS)}."
            )
        if isinstance(sample_count, bool) or not isinstance(sample_count, int):
            raise AdaptiveAIError("sample_count must be an integer.")
        if sample_count < 1 or sample_count > 1000:
            raise AdaptiveAIError("sample_count must be between 1 and 1000.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise AdaptiveAIError("seed must be a non-negative integer.")
        if params is not None and not isinstance(params, dict):
            raise AdaptiveAIError("params must be a mapping or None.")
        # Eager validation so a bad scenario/param fails before any row
        # or audit entry exists (generate_scenario_samples is pure).
        generate_scenario_samples(scenario, 1, int(seed), params)
        self._predictions.init()
        self._sessions.init()
        session = self._sessions.create_session(
            session_id=_new_session_id(),
            use_case=USE_CASE,
            scenario=scenario,
            sample_count=int(sample_count),
            seed=int(seed),
            params=dict(params) if params else None,
        )
        self._record_event(
            "demo_session_created",
            reason=(
                f"Demo session {session['session_id']} created "
                f"(scenario: {scenario}, samples: {sample_count}, seed: {seed})."
            ),
            details={"session_id": session["session_id"], "scenario": scenario},
        )
        logger.info(
            "Demo session created: %s scenario=%s", session["session_id"], scenario
        )
        return session

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Returns a session, or None when unknown."""
        return self._sessions.get_session(session_id)

    def list_sessions(
        self, limit: int = 100, offset: int = 0
    ) -> List[Dict[str, Any]]:
        """Lists demo sessions newest-first."""
        return self._sessions.list_sessions(
            limit=limit, offset=offset, use_case=USE_CASE
        )

    def _require_session(self, session_id: str) -> Dict[str, Any]:
        session = self._sessions.get_session(session_id)
        if session is None:
            raise AdaptiveAIError(f"Demo session not found: {session_id}.")
        return session

    # -- generation --------------------------------------------------------

    def generate_samples(self, session_id: str) -> Dict[str, Any]:
        """Generates and persists isolated demo samples for a session.

        Deterministic: same scenario + sample_count + seed (+ params)
        always yields identical samples. Regenerating replaces the
        session's previous demo rows (scoped delete first) so samples
        never accumulate across clicks. Production prediction history
        is untouched (separate ``source`` class).
        """
        session = self._require_session(session_id)
        if session.get("status") not in ("active", "reset"):
            raise AdaptiveAIError(
                f"Demo session {session_id} is '{session.get('status')}' "
                "and cannot generate samples."
            )
        samples = generate_scenario_samples(
            session["scenario"],
            session["sample_count"],
            session["seed"],
            session.get("params"),
        )
        removed = self._predictions.delete_demo_session_predictions(session_id)
        if removed:
            logger.info(
                "Demo regenerate replaced %d stale row(s) for %s.",
                removed,
                session_id,
            )
        inserted = 0
        estimator = self._resolve_serving_estimator(samples[0])
        for input_data in samples:
            prediction, model_name, model_version = self._row_prediction(
                estimator, input_data
            )
            self._predictions.save_prediction(
                use_case=USE_CASE,
                model_name=model_name,
                input_data=input_data,
                prediction=prediction,
                model_version=model_version,
                source=DEMO_SOURCE,
                demo_session_id=session_id,
            )
            inserted += 1
        updated = self._sessions.update_session(
            session_id,
            status="active",
            samples_generated=inserted,
            started_at=utc_now_iso(),
        )
        self._record_event(
            "demo_scenario_generated",
            reason=(
                f"Demo scenario '{session['scenario']}' generated "
                f"{inserted} isolated sample(s) for {session_id}."
            ),
            details={
                "session_id": session_id,
                "scenario": session["scenario"],
                "samples": inserted,
                "seed": session["seed"],
            },
        )
        return {"session": updated, "samples_generated": inserted, "scenario": session["scenario"]}

    # -- monitoring ---------------------------------------------------------

    def monitor_session(self, session_id: str) -> Dict[str, Any]:
        """Runs the REAL Phase 4 health check over demo traffic only.

        Uses the same ``MonitoringService`` / KS + chi-square detectors
        / data-quality / severity path as production monitoring, but
        scoped to this session's rows and with NO persistence repository
        — the result is stored on the session record, so the production
        ``monitoring_results`` history is unchanged. Also computes the
        existing Phase 5 retraining recommendation (pure advice).
        """
        from app.monitoring.monitoring_service import MonitoringService
        from app.monitoring.data_provider import create_house_price_provider
        from app.retraining.decision import recommend_retraining

        session = self._require_session(session_id)
        if int(session.get("samples_generated") or 0) < 1:
            raise AdaptiveAIError(
                f"Demo session {session_id} has no samples yet; generate first."
            )
        provider = create_house_price_provider(
            repository=self._predictions,
            source=DEMO_SOURCE,
            demo_session_id=session_id,
        )
        service = MonitoringService(provider=provider, monitoring_repository=None)
        summary = service.run_health_check(
            production_limit=DEMO_MONITOR_WINDOW,
            serving_info=self._serving_info(),
            total_predictions=int(session.get("samples_generated") or 0),
        )
        summary["demo_session_id"] = session_id
        summary["demo_scenario"] = session["scenario"]
        serving = self._serving_info()
        decision = recommend_retraining(
            summary,
            production_model=serving.get("model_name"),
            production_version=serving.get("model_version"),
        )
        monitored_at = utc_now_iso()
        updated = self._sessions.update_session(
            session_id,
            status="completed",
            completed_at=monitored_at,
            last_monitoring=summary,
            last_decision=decision,
            last_monitored_at=monitored_at,
        )
        self._record_event(
            "demo_monitoring_run",
            reason=(
                f"Demo monitoring for {session_id} "
                f"(scenario: {session['scenario']}): "
                f"health={summary.get('overall_health')}, "
                f"severity={summary.get('severity')}, "
                f"drifted={summary.get('drifted_count')}/"
                f"{summary.get('total_monitored')}."
            ),
            details={
                "session_id": session_id,
                "scenario": session["scenario"],
                "overall_health": summary.get("overall_health"),
                "severity": summary.get("severity"),
            },
        )
        return {"session": updated, "monitoring": summary, "decision": decision}

    # -- reset ---------------------------------------------------------------

    def reset_session(self, session_id: str) -> Dict[str, Any]:
        """Resets a demo session WITHOUT touching production state.

        Deletes ONLY ``source='demo'`` rows of this session, clears the
        stored monitoring snapshot, and marks the session ``reset``.
        Never removes production predictions, models, reference data,
        the production pointer, or audit history (the reset itself is
        audited as ``demo_session_reset``).
        """
        session = self._require_session(session_id)
        removed = self._predictions.delete_demo_session_predictions(session_id)
        reset_at = utc_now_iso()
        updated = self._sessions.update_session(
            session_id,
            status="reset",
            samples_generated=0,
            reset_at=reset_at,
            last_monitoring=None,
            last_decision=None,
            last_monitored_at=None,
        )
        self._record_event(
            "demo_session_reset",
            reason=(
                f"Demo session {session_id} reset: "
                f"{removed} isolated demo row(s) removed; "
                "production state untouched."
            ),
            details={"session_id": session_id, "removed_demo_rows": removed},
        )
        logger.info("Demo session reset: %s removed=%d", session_id, removed)
        return {"session": updated, "removed_demo_rows": removed}

    # -- retraining (delegates to the existing Phase 5 lifecycle) ------------

    def start_retraining(
        self,
        session_id: str,
        lifecycle: Any,
        production_limit: int = 1000,
        data_path: Any = None,
    ) -> Dict[str, Any]:
        """Starts an explicit lifecycle run tagged as demo-triggered.

        Training reuses the existing lifecycle service (shared training
        path, evaluation, gate); the run trigger is stamped
        ``demo:<scenario>:<session_id>`` so demo-triggered retraining is
        distinguishable from production retraining. Never promotes —
        promotion stays an explicit, separately confirmed call.

        ``data_path`` is a tests-only override forwarded to the
        lifecycle run (production callers always train on the
        versioned reference dataset).
        """
        session = self._require_session(session_id)
        trigger = f"demo:{session['scenario']}:{session_id}"
        run = lifecycle.start_run(
            production_limit=production_limit, trigger=trigger, data_path=data_path
        )
        self._record_event(
            "demo_retraining_started",
            reason=(
                f"Demo scenario '{session['scenario']}' ({session_id}) "
                f"triggered retraining run {run.get('run_id')}."
            ),
            details={
                "session_id": session_id,
                "scenario": session["scenario"],
                "run_id": run.get("run_id"),
                "trigger": trigger,
            },
        )
        return run

    # -- internals ------------------------------------------------------------

    def _record_event(
        self, event: str, reason: str, details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Appends a demo audit event (best-effort; never breaks flow)."""
        if self._events is None:
            return
        try:
            self._events.init()
            self._events.record_event(
                use_case=USE_CASE, event=event, reason=reason, details=details
            )
        except Exception as e:
            logger.warning("Demo audit event '%s' not recorded: %s", event, e)

    def _serving_info(self) -> Dict[str, Optional[str]]:
        """Reads the production pointer read-only (for monitor labels)."""
        try:
            from app.use_cases.house_price.model_persistence import read_production_pointer

            pointer = read_production_pointer(models_dir=self._models_dir)
            return {
                "model_name": pointer.get("model_name"),
                "model_version": pointer.get("model_version"),
            }
        except Exception:
            return {"model_name": None, "model_version": None}

    def _resolve_serving_estimator(
        self, probe_input: Dict[str, Any]
    ) -> Optional[tuple]:
        """Resolves the serving pipeline once per generation batch.

        Returns ``(model, model_name, model_version)`` when the current
        production artifact loads AND serves the v2 schema (verified
        with one trial prediction on a real demo row), else None so
        the whole batch falls back to benchmark estimates. Resolving
        once keeps generation quiet and fast instead of failing row by
        row (e.g. the legacy schema-v1 production artifact cannot
        serve v2 inputs and fails closed).
        """
        try:
            import pandas as pd

            from app.use_cases.house_price.model_persistence import (
                load_house_price_model,
                read_production_pointer,
            )
            from app.use_cases.house_price.predict import predict_house_price

            pointer = read_production_pointer(models_dir=self._models_dir)
            model = load_house_price_model(
                str(pointer["model_name"]), models_dir=self._models_dir
            )
            predict_house_price(model, pd.DataFrame([probe_input]))
            version = pointer.get("model_version")
            return model, str(pointer["model_name"]), str(version) if version else None
        except Exception as e:
            logger.info(
                "Demo generation uses benchmark estimates "
                "(no compatible production model: %s).",
                e,
            )
            return None

    def _row_prediction(
        self, estimator: Optional[tuple], input_data: Dict[str, Any]
    ) -> tuple:
        """Estimates a stored prediction value for a demo row.

        Uses the resolved serving pipeline when available (realistic
        traffic); otherwise the locality benchmark rate × area. The
        value is never consumed by drift detection (input features
        only) and is documented as simulated.
        """
        if estimator is not None:
            try:
                import pandas as pd

                from app.use_cases.house_price.predict import predict_house_price

                model, name, version = estimator
                value = float(
                    predict_house_price(model, pd.DataFrame([input_data]))[0]
                )
                return value, name, version
            except Exception as e:
                logger.debug("Demo row prediction failed, using benchmark: %s", e)
        try:
            from app.use_cases.house_price.benchmarks import get_locality_benchmark

            benchmark = get_locality_benchmark(str(input_data.get("location", "")))
            value = float(benchmark["benchmark_price_per_sqft"]) * float(
                input_data.get("area_sqft", 0.0)
            )
            return value, "demo-simulator", None
        except Exception:
            return 0.0, "demo-simulator", None


def create_house_price_demo_service(
    db_manager: Any = None,
    models_dir: Optional[Union[str, Path]] = None,
) -> DemoService:
    """Small factory wiring the demo service for House Price."""
    from app.database.database import DatabaseManager
    from app.database.lifecycle_repository import LifecycleEventRepository
    from app.demo.repository import DemoSessionRepository

    manager = db_manager if db_manager is not None else DatabaseManager()
    return DemoService(
        prediction_repository=PredictionRepository(manager),
        session_repository=DemoSessionRepository(manager),
        event_repository=LifecycleEventRepository(manager),
        models_dir=models_dir,
    )
