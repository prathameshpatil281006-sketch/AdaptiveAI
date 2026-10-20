"""Safe model promotion execution for AdaptiveAI (Step 33).

Execution layer that performs promotion ONLY when the decision layer
explicitly returns ``"promote"``::

    RetrainingService
        |
        v
    ModelEvaluator (candidate vs production -> evaluation_result)
        |
        v
    ModelPromotionDecision.decide(evaluation_result) -> decision
        |
        v
    ModelPromoter.promote(...) -- this module
        |-- decision == "reject" -> return rejected, zero side effects
        |-- decision == "requires_initial_promotion" -> return without
        |   auto-promoting (explicit first deployment required)
        |-- decision == "promote" -> persist + register, return promoted
        |-- side-effect failure -> raise ModelPromotionError (never report
            success falsely)

Promotion reuses existing APIs only:

- persistence via :func:`save_house_price_model`
  (``app.use_cases.house_price.model_persistence``)
- registry via :meth:`MLflowTracker.register_model`
  (``app.engine.mlflow_tracker``), which preserves the existing
  ``HousePriceModel`` naming convention. No registry logic is duplicated
  and no deprecated MLflow stage APIs are introduced (the tracker
  internally handles registration on MLflow 3.x).

Atomicity limitation (documented, not silently fixed):
the operation consists of two independent side effects (filesystem
persist, then registry version). The existing persistence/registry APIs
offer no transaction. If persistence succeeds and registry then fails,
the filesystem file remains and NO rollback is attempted (deleting or
restoring files could destroy the previous production artifact and
cannot be done safely with the current APIs). The failure is raised as
:class:`ModelPromotionError` with explicit partial-failure context so
the caller never mistakes it for success.

This module never touches the API, scheduler, dashboard, or database,
never invents run IDs or versions, and never mutates caller-owned
dictionaries.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

from app.core.exceptions import AdaptiveAIError, ModelPromotionError
from app.engine.mlflow_tracker import MLflowTracker
from app.retraining.model_evaluator import CANDIDATE_BETTER, NO_PRODUCTION_MODEL
from app.retraining.model_promotion import (
    DECISION_PROMOTE,
    DECISION_REJECT,
    DECISION_REQUIRES_INITIAL_PROMOTION,
    ModelPromotionDecision,
)
from app.use_cases.house_price.model_persistence import (
    read_production_pointer,
    save_house_price_model,
    write_production_pointer,
)

logger = logging.getLogger(__name__)

# Promoter result vocabulary (stable contract). Note the past tense:
# the decision layer says "promote"/"reject" (permission), the promoter
# reports "promoted"/"rejected" (outcome).
PROMOTED = "promoted"
REJECTED = "rejected"
REQUIRES_INITIAL_PROMOTION = "requires_initial_promotion"
FAILED = "failed"


def _is_non_empty_string(value: Any) -> bool:
    """Returns True only for non-empty, non-whitespace strings."""
    return isinstance(value, str) and bool(value.strip())


class ModelPromoter:
    """Executes promotion gated strictly by :class:`ModelPromotionDecision`.

    Dependency injection keeps the promoter testable:

    Args:
        decision: Decision layer instance. Defaults to a fresh
            :class:`ModelPromotionDecision`. Any object with a callable
            ``decide(evaluation_result) -> dict`` is accepted so tests can
            inject doubles.
        save_fn: Persistence callable
            ``(model, model_name, models_dir=None) -> Path``. Defaults to
            :func:`save_house_price_model`. Tests inject a stub.
        tracker: Object exposing ``register_model(run_id,
            registered_model_name, artifact_path) -> dict`` (e.g. a real
            :class:`MLflowTracker`). Defaults to ``None`` so unit tests never
            accidentally touch the real registry; promotion without a tracker
            raises :class:`ModelPromotionError`.
        registered_model_name: Default registry target. Defaults to
            ``MLflowTracker.REGISTERED_MODEL_NAME`` (``"HousePriceModel"``),
            preserving the existing convention and avoiding unrelated models.
        models_dir: Default filesystem directory for persistence. ``None``
            uses the persistence default (``models/house_price/``). Tests
            should pass a ``tmp_path`` per call instead.
        artifact_path: Default MLflow artifact path. Defaults to ``"model"``
            (matches :meth:`MLflowTracker.register_model`).
    """

    PROMOTED = PROMOTED
    REJECTED = REJECTED
    REQUIRES_INITIAL_PROMOTION = REQUIRES_INITIAL_PROMOTION
    FAILED = FAILED

    def __init__(
        self,
        decision: Optional[Any] = None,
        save_fn: Optional[Callable[..., Any]] = None,
        tracker: Optional[Any] = None,
        registered_model_name: Optional[str] = None,
        models_dir: Optional[Union[str, Path]] = None,
        artifact_path: str = "model",
    ) -> None:
        if decision is None:
            decision = ModelPromotionDecision()
        if not hasattr(decision, "decide") or not callable(decision.decide):
            raise ModelPromotionError(
                "decision must expose a callable decide(evaluation_result)."
            )
        if save_fn is None:
            save_fn = save_house_price_model
        if not callable(save_fn):
            raise ModelPromotionError("save_fn must be callable or None.")
        if tracker is not None:
            register = getattr(tracker, "register_model", None)
            if not callable(register):
                raise ModelPromotionError(
                    "tracker must expose a callable register_model() or be None."
                )
        if registered_model_name is None:
            registered_model_name = MLflowTracker.REGISTERED_MODEL_NAME
        if not _is_non_empty_string(registered_model_name):
            raise ModelPromotionError(
                "registered_model_name must be a non-empty string or None."
            )
        if models_dir is not None and not isinstance(models_dir, (str, Path)):
            raise ModelPromotionError("models_dir must be a path string, Path, or None.")
        if not _is_non_empty_string(artifact_path):
            raise ModelPromotionError("artifact_path must be a non-empty string.")

        self._decision = decision
        self._save_fn = save_fn
        self._tracker = tracker
        self._registered_model_name = str(registered_model_name)
        self._models_dir = models_dir
        self._artifact_path = str(artifact_path)

    @property
    def decision(self) -> Any:
        """Returns the injected decision layer."""
        return self._decision

    @property
    def tracker(self) -> Optional[Any]:
        """Returns the injected tracker (may be None until promotion)."""
        return self._tracker

    def promote(
        self,
        candidate_model: Any,
        evaluation_result: Dict[str, Any],
        candidate_model_name: str,
        candidate_run_id: str,
        production_model_name: Optional[str] = None,
        models_dir: Optional[Union[str, Path]] = None,
        registered_model_name: Optional[str] = None,
        artifact_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Promotes the candidate ONLY on an explicit ``"promote"`` decision.

        Args:
            candidate_model: Trained candidate pipeline (e.g. best_pipeline
                from :class:`RetrainingService`). Validated only on the
                promote path; gating paths return without inspecting it.
            evaluation_result: Standard :class:`ModelEvaluator` output dict.
                Never mutated. Its ``status`` must be ``"candidate_better"``
                for promotion; anything else never promotes.
            candidate_model_name: Filename identity for persistence
                (``<name>.joblib``). Must be a non-empty string on the
                promote path.
            candidate_run_id: Exact MLflow run ID holding the candidate
                artifact. Must be a non-empty string on the promote path;
                it is passed through verbatim -- never invented.
            production_model_name: Informational only in this step (which
                production artifact was compared). Not used for filesystem
                writes; the file written is ``candidate_model_name``.
            models_dir: Filesystem directory override for this call. ``None``
                falls back to the constructor default (itself ``None`` =>
                persistence default).
            registered_model_name: Registry target override for this call.
                ``None`` falls back to the constructor default
                (``"HousePriceModel"``).
            artifact_path: MLflow artifact path override for this call.
                ``None`` falls back to the constructor default (``"model"``).

        Returns:
            Dict with ``decision``, ``model_name``, ``run_id``,
            ``registered_model_name``, ``model_version``, ``reason`` where
            ``decision`` is ``"promoted"``, ``"rejected"``, or
            ``"requires_initial_promotion"``. Registry names/versions on
            success come verbatim from the registry response -- never
            fabricated.

        Raises:
            ModelPromotionError: On malformed evaluation results (propagated
                from the decision layer), validation failures (missing
                candidate/run_id/names/tracker), malformed registry
                responses, or any persistence/registry side-effect failure
                (with partial-failure context; never falsely ``"promoted"``).
        """
        resolved_models_dir = models_dir if models_dir is not None else self._models_dir
        resolved_registered = (
            registered_model_name
            if registered_model_name is not None
            else self._registered_model_name
        )
        resolved_artifact = artifact_path if artifact_path is not None else self._artifact_path

        # 1. Gate strictly on the decision layer. No validation or side
        #    effects happen before this call, so rejected candidates can
        #    never trigger persistence/registry.
        decided = self._decision.decide(evaluation_result)
        decision_value = decided.get("decision") if isinstance(decided, dict) else None
        decision_reason = ""
        if isinstance(decided, dict):
            raw_reason = decided.get("reason")
            if isinstance(raw_reason, str):
                decision_reason = raw_reason

        if decision_value == DECISION_REJECT:
            logger.info("Promotion rejected by decision layer; no side effects.")
            return {
                "decision": REJECTED,
                "model_name": candidate_model_name,
                "run_id": candidate_run_id,
                "registered_model_name": resolved_registered
                if _is_non_empty_string(resolved_registered)
                else None,
                "model_version": None,
                "reason": f"Rejected: {decision_reason}".strip()
                if decision_reason
                else "Rejected by promotion decision; production kept.",
            }
        if decision_value == DECISION_REQUIRES_INITIAL_PROMOTION:
            logger.info(
                "No production model; initial promotion requires explicit "
                "handling, not auto-promotion."
            )
            return {
                "decision": REQUIRES_INITIAL_PROMOTION,
                "model_name": candidate_model_name,
                "run_id": candidate_run_id,
                "registered_model_name": resolved_registered
                if _is_non_empty_string(resolved_registered)
                else None,
                "model_version": None,
                "reason": (
                    "Requires explicit initial deployment; "
                    "not auto-promoted. "
                    f"{decision_reason}".strip()
                ),
            }
        if decision_value != DECISION_PROMOTE:
            raise ModelPromotionError(
                f"Unknown promotion decision: {decision_value!r}. "
                "Expected 'promote', 'reject', or 'requires_initial_promotion'."
            )

        # 2. Promote path only: strict validation BEFORE any side effect.
        if not isinstance(evaluation_result, dict):
            raise ModelPromotionError("evaluation_result must be a dict.")
        if evaluation_result.get("status") != CANDIDATE_BETTER:
            raise ModelPromotionError(
                "Promotion requires evaluation status 'candidate_better'; "
                f"got {evaluation_result.get('status')!r}. Refusing to promote."
            )
        if candidate_model is None:
            raise ModelPromotionError(
                "candidate_model must not be None for promotion."
            )
        if not _is_non_empty_string(candidate_model_name):
            raise ModelPromotionError(
                "candidate_model_name must be a non-empty string for promotion."
            )
        if not _is_non_empty_string(candidate_run_id):
            raise ModelPromotionError(
                "candidate_run_id must be a non-empty string for promotion; "
                "refusing to promote without the candidate's actual run ID."
            )
        if not _is_non_empty_string(resolved_registered):
            raise ModelPromotionError(
                "registered_model_name must be a non-empty string for promotion."
            )
        if not _is_non_empty_string(resolved_artifact):
            raise ModelPromotionError(
                "artifact_path must be a non-empty string for promotion."
            )
        if self._tracker is None:
            raise ModelPromotionError(
                "No MLflow tracker configured; refusing to promote."
            )

        candidate_name_str = str(candidate_model_name).strip()
        run_id_str = str(candidate_run_id).strip()
        registered_str = str(resolved_registered).strip()
        artifact_str = str(resolved_artifact).strip()

        # 3. Side effects: persistence first, then registry. As atomic as
        #    practical with the existing APIs (no transaction support).
        try:
            saved_path = self._save_fn(
                candidate_model,
                candidate_name_str,
                models_dir=resolved_models_dir,
            )
        except ModelPromotionError:
            raise
        except AdaptiveAIError as e:
            raise ModelPromotionError(
                f"Promotion persistence failed for '{candidate_name_str}': {e}"
            ) from e
        except Exception as e:
            raise ModelPromotionError(
                f"Promotion persistence failed for '{candidate_name_str}': {e}"
            ) from e
        logger.info(
            "Promotion persistence succeeded for '%s' at %s; registering run '%s'.",
            candidate_name_str,
            saved_path,
            run_id_str,
        )

        try:
            registry_info = self._tracker.register_model(
                run_id=run_id_str,
                registered_model_name=registered_str,
                artifact_path=artifact_str,
            )
        except ModelPromotionError as e:
            # Partial failure: filesystem write already happened, registry
            # did not. No rollback attempted (see module docstring).
            raise ModelPromotionError(
                f"Promotion registry failed for run '{run_id_str}' after "
                f"persistence succeeded at '{saved_path}'; no rollback "
                f"performed (partial promotion): {e}"
            ) from e
        except Exception as e:
            raise ModelPromotionError(
                f"Promotion registry failed for run '{run_id_str}' after "
                f"persistence succeeded at '{saved_path}'; no rollback "
                f"performed (partial promotion): {e}"
            ) from e

        # 4. Propagate the ACTUAL registry response; never fabricate.
        if not isinstance(registry_info, dict):
            raise ModelPromotionError(
                "Promotion registry returned an invalid response "
                f"(expected dict, got {type(registry_info).__name__}); "
                "refusing to report success."
            )
        actual_name = registry_info.get("registered_model_name")
        actual_version = registry_info.get("version")
        if not _is_non_empty_string(actual_name):
            raise ModelPromotionError(
                "Promotion registry response missing 'registered_model_name'; "
                "refusing to report success."
            )
        if not _is_non_empty_string(actual_version):
            raise ModelPromotionError(
                "Promotion registry response missing 'version'; "
                "refusing to fabricate a version."
            )

        logger.info(
            "Promotion completed: model '%s' run '%s' registered as '%s' v%s.",
            candidate_name_str,
            run_id_str,
            str(actual_name).strip(),
            str(actual_version).strip(),
        )
        return {
            "decision": PROMOTED,
            "model_name": candidate_name_str,
            "run_id": run_id_str,
            "registered_model_name": str(actual_name).strip(),
            "model_version": str(actual_version).strip(),
            "reason": (
                f"Promoted '{candidate_name_str}' "
                f"(run {run_id_str}) as '{str(actual_name).strip()}' "
                f"v{str(actual_version).strip()}. {decision_reason}".strip()
            ),
        }


    def promote_schema_migration(
        self,
        candidate_model: Any,
        migration_assessment: Dict[str, Any],
        candidate_model_name: str,
        candidate_run_id: str,
        production_model_name: Optional[str] = None,
        models_dir: Optional[Union[str, Path]] = None,
        registered_model_name: Optional[str] = None,
        artifact_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Explicitly promotes a migration-gated candidate over an
        incompatible production baseline.

        Schema-migration path. The caller (the lifecycle service) must
        already hold BOTH a passing migration gate AND explicit human
        confirmation; this method re-validates the assessment itself and
        never triggers implicitly. The production artifact being
        replaced is informational only (which incompatible model was
        serving); the file written is always the candidate.

        Gates strictly on the migration assessment: it must be a dict
        with ``eligible is True``, every ``checks`` entry passed, and
        ``migration.required is True``. Anything else raises instead of
        promoting, so an unvalidated candidate can never overwrite a
        live production model.

        Side effects (persist, then register) mirror the promote path;
        the caller publishes the production serving reference afterwards
        via :meth:`update_production_pointer`. Partial-failure semantics
        match :meth:`promote` (no rollback; failure raised, never
        reported as success).

        Returns:
            Promoted-outcome dict (decision ``"promoted"``) with the
            actual registry name/version, suitable for audit history
            and pointer publication.

        Raises:
            ModelPromotionError: If the assessment is not a passing
                migration assessment, inputs are invalid, or any side
                effect fails.
        """
        resolved_models_dir = models_dir if models_dir is not None else self._models_dir
        resolved_registered = (
            registered_model_name
            if registered_model_name is not None
            else self._registered_model_name
        )
        resolved_artifact = artifact_path if artifact_path is not None else self._artifact_path

        # Informational only (which incompatible artifact was serving).
        _ = production_model_name

        if not isinstance(migration_assessment, dict):
            raise ModelPromotionError("migration_assessment must be a dict.")
        if migration_assessment.get("eligible") is not True:
            raise ModelPromotionError(
                "Schema migration requires a passing migration gate; "
                "refusing to promote."
            )
        assessment_checks = migration_assessment.get("checks")
        if not isinstance(assessment_checks, list) or not assessment_checks:
            raise ModelPromotionError(
                "Schema migration requires a non-empty migration gate record; "
                "refusing to promote."
            )
        for entry in assessment_checks:
            if not isinstance(entry, dict) or entry.get("passed") is not True:
                raise ModelPromotionError(
                    "Schema migration requires every migration check to pass; "
                    f"refusing to promote (failing check: {entry!r})."
                )
        migration_block = migration_assessment.get("migration")
        if not isinstance(migration_block, dict) or migration_block.get("required") is not True:
            raise ModelPromotionError(
                "Schema migration requires an explicit migration requirement; "
                "refusing to promote."
            )

        if candidate_model is None:
            raise ModelPromotionError(
                "candidate_model must not be None for schema migration."
            )
        if not _is_non_empty_string(candidate_model_name):
            raise ModelPromotionError(
                "candidate_model_name must be a non-empty string for schema migration."
            )
        if not _is_non_empty_string(candidate_run_id):
            raise ModelPromotionError(
                "candidate_run_id must be a non-empty string for schema "
                "migration; refusing to promote without the candidate's "
                "actual run ID."
            )
        if not _is_non_empty_string(resolved_registered):
            raise ModelPromotionError(
                "registered_model_name must be a non-empty string for schema migration."
            )
        if not _is_non_empty_string(resolved_artifact):
            raise ModelPromotionError(
                "artifact_path must be a non-empty string for schema migration."
            )
        if self._tracker is None:
            raise ModelPromotionError(
                "No MLflow tracker configured; refusing schema migration."
            )

        candidate_name_str = str(candidate_model_name).strip()
        run_id_str = str(candidate_run_id).strip()
        registered_str = str(resolved_registered).strip()
        artifact_str = str(resolved_artifact).strip()

        try:
            saved_path = self._save_fn(
                candidate_model,
                candidate_name_str,
                models_dir=resolved_models_dir,
            )
        except ModelPromotionError:
            raise
        except AdaptiveAIError as e:
            raise ModelPromotionError(
                f"Schema-migration persistence failed for '{candidate_name_str}': {e}"
            ) from e
        except Exception as e:
            raise ModelPromotionError(
                f"Schema-migration persistence failed for '{candidate_name_str}': {e}"
            ) from e
        logger.info(
            "Schema-migration persistence succeeded for '%s' at %s; registering run '%s'.",
            candidate_name_str,
            saved_path,
            run_id_str,
        )

        try:
            registry_info = self._tracker.register_model(
                run_id=run_id_str,
                registered_model_name=registered_str,
                artifact_path=artifact_str,
            )
        except ModelPromotionError as e:
            raise ModelPromotionError(
                f"Schema-migration registry failed for run '{run_id_str}' after "
                f"persistence succeeded at '{saved_path}'; no rollback "
                f"performed (partial migration): {e}"
            ) from e
        except Exception as e:
            raise ModelPromotionError(
                f"Schema-migration registry failed for run '{run_id_str}' after "
                f"persistence succeeded at '{saved_path}'; no rollback "
                f"performed (partial migration): {e}"
            ) from e

        if not isinstance(registry_info, dict):
            raise ModelPromotionError(
                "Schema-migration registry returned an invalid response "
                f"(expected dict, got {type(registry_info).__name__}); "
                "refusing to report success."
            )
        actual_name = registry_info.get("registered_model_name")
        actual_version = registry_info.get("version")
        if not _is_non_empty_string(actual_name):
            raise ModelPromotionError(
                "Schema-migration registry response missing "
                "'registered_model_name'; refusing to report success."
            )
        if not _is_non_empty_string(actual_version):
            raise ModelPromotionError(
                "Schema-migration registry response missing 'version'; "
                "refusing to fabricate a version."
            )

        logger.info(
            "Schema migration completed: model '%s' run '%s' registered as '%s' v%s.",
            candidate_name_str,
            run_id_str,
            str(actual_name).strip(),
            str(actual_version).strip(),
        )
        return {
            "decision": PROMOTED,
            "model_name": candidate_name_str,
            "run_id": run_id_str,
            "registered_model_name": str(actual_name).strip(),
            "model_version": str(actual_version).strip(),
            "reason": (
                f"Schema migration of '{candidate_name_str}' "
                f"(run {run_id_str}) as '{str(actual_name).strip()}' "
                f"v{str(actual_version).strip()} over an incompatible "
                "production baseline."
            ),
        }

    def deploy_initial_candidate(
        self,
        candidate_model: Any,
        evaluation_result: Dict[str, Any],
        candidate_model_name: str,
        candidate_run_id: str,
        production_model_name: Optional[str] = None,
        models_dir: Optional[Union[str, Path]] = None,
        registered_model_name: Optional[str] = None,
        artifact_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Explicitly deploys the best candidate as the first production model.

        Step 55 first-deployment path. The caller must opt in explicitly
        (the retraining service only routes here with
        ``allow_initial_promotion=True``); this method never triggers
        implicitly from :meth:`promote`, which keeps refusing automatic
        first deployment with ``requires_initial_promotion``.

        Gates strictly on the decision layer: the evaluation must carry
        status ``no_production_model`` and the decision must be
        ``requires_initial_promotion``. Anything else (including an
        existing production model) raises instead of deploying, so this
        path can never overwrite a live production model or promote an
        unevaluated candidate.

        Side effects (persist, then register) mirror the promote path;
        the caller publishes the production serving reference afterwards
        via :meth:`update_production_pointer`. Partial-failure semantics
        match :meth:`promote` (no rollback; failure raised, never
        reported as success).

        Returns:
            Promoted-outcome dict (decision ``"promoted"``) with the
            actual registry name/version, suitable for audit history.

        Raises:
            ModelPromotionError: If no production model actually is
                missing, the decision is not initial-promotion, inputs
                are invalid, or any side effect fails.
        """
        resolved_models_dir = models_dir if models_dir is not None else self._models_dir
        resolved_registered = (
            registered_model_name
            if registered_model_name is not None
            else self._registered_model_name
        )
        resolved_artifact = artifact_path if artifact_path is not None else self._artifact_path

        # Informational only (which artifact was compared, if any). The
        # file written is always the candidate; this path requires that
        # no production model exists at all.
        _ = production_model_name

        if not isinstance(evaluation_result, dict):
            raise ModelPromotionError("evaluation_result must be a dict.")
        if evaluation_result.get("status") != NO_PRODUCTION_MODEL:
            raise ModelPromotionError(
                "Initial deployment requires evaluation status "
                f"'{NO_PRODUCTION_MODEL}'; got "
                f"{evaluation_result.get('status')!r}. Refusing to deploy "
                "over an existing production model."
            )
        decided = self._decision.decide(evaluation_result)
        decision_value = decided.get("decision") if isinstance(decided, dict) else None
        if decision_value != DECISION_REQUIRES_INITIAL_PROMOTION:
            raise ModelPromotionError(
                f"Initial deployment requires decision "
                f"'{DECISION_REQUIRES_INITIAL_PROMOTION}'; got "
                f"{decision_value!r}. Refusing to deploy."
            )
        decision_reason = ""
        if isinstance(decided, dict):
            raw_reason = decided.get("reason")
            if isinstance(raw_reason, str):
                decision_reason = raw_reason

        if candidate_model is None:
            raise ModelPromotionError(
                "candidate_model must not be None for initial deployment."
            )
        if not _is_non_empty_string(candidate_model_name):
            raise ModelPromotionError(
                "candidate_model_name must be a non-empty string for initial deployment."
            )
        if not _is_non_empty_string(candidate_run_id):
            raise ModelPromotionError(
                "candidate_run_id must be a non-empty string for initial "
                "deployment; refusing to deploy without the candidate's "
                "actual run ID."
            )
        if not _is_non_empty_string(resolved_registered):
            raise ModelPromotionError(
                "registered_model_name must be a non-empty string for initial deployment."
            )
        if not _is_non_empty_string(resolved_artifact):
            raise ModelPromotionError(
                "artifact_path must be a non-empty string for initial deployment."
            )
        if self._tracker is None:
            raise ModelPromotionError(
                "No MLflow tracker configured; refusing initial deployment."
            )

        candidate_name_str = str(candidate_model_name).strip()
        run_id_str = str(candidate_run_id).strip()
        registered_str = str(resolved_registered).strip()
        artifact_str = str(resolved_artifact).strip()

        try:
            saved_path = self._save_fn(
                candidate_model,
                candidate_name_str,
                models_dir=resolved_models_dir,
            )
        except ModelPromotionError:
            raise
        except AdaptiveAIError as e:
            raise ModelPromotionError(
                f"Initial deployment persistence failed for '{candidate_name_str}': {e}"
            ) from e
        except Exception as e:
            raise ModelPromotionError(
                f"Initial deployment persistence failed for '{candidate_name_str}': {e}"
            ) from e
        logger.info(
            "Initial deployment persistence succeeded for '%s' at %s; registering run '%s'.",
            candidate_name_str,
            saved_path,
            run_id_str,
        )

        try:
            registry_info = self._tracker.register_model(
                run_id=run_id_str,
                registered_model_name=registered_str,
                artifact_path=artifact_str,
            )
        except ModelPromotionError as e:
            raise ModelPromotionError(
                f"Initial deployment registry failed for run '{run_id_str}' after "
                f"persistence succeeded at '{saved_path}'; no rollback "
                f"performed (partial deployment): {e}"
            ) from e
        except Exception as e:
            raise ModelPromotionError(
                f"Initial deployment registry failed for run '{run_id_str}' after "
                f"persistence succeeded at '{saved_path}'; no rollback "
                f"performed (partial deployment): {e}"
            ) from e

        if not isinstance(registry_info, dict):
            raise ModelPromotionError(
                "Initial deployment registry returned an invalid response "
                f"(expected dict, got {type(registry_info).__name__}); "
                "refusing to report success."
            )
        actual_name = registry_info.get("registered_model_name")
        actual_version = registry_info.get("version")
        if not _is_non_empty_string(actual_name):
            raise ModelPromotionError(
                "Initial deployment registry response missing "
                "'registered_model_name'; refusing to report success."
            )
        if not _is_non_empty_string(actual_version):
            raise ModelPromotionError(
                "Initial deployment registry response missing 'version'; "
                "refusing to fabricate a version."
            )

        logger.info(
            "Initial deployment completed: model '%s' run '%s' registered as '%s' v%s.",
            candidate_name_str,
            run_id_str,
            str(actual_name).strip(),
            str(actual_version).strip(),
        )
        return {
            "decision": PROMOTED,
            "model_name": candidate_name_str,
            "run_id": run_id_str,
            "registered_model_name": str(actual_name).strip(),
            "model_version": str(actual_version).strip(),
            "reason": (
                f"Initial deployment of '{candidate_name_str}' "
                f"(run {run_id_str}) as '{str(actual_name).strip()}' "
                f"v{str(actual_version).strip()}. {decision_reason}".strip()
            ),
        }

    def update_production_pointer(
        self,
        promotion_result: Dict[str, Any],
        models_dir: Optional[Union[str, Path]] = None,
        use_case: str = "house_price",
    ) -> Dict[str, Any]:
        """Publishes the production serving reference for a promoted outcome.

        Step 55 serving contract: after :meth:`promote` (or
        :meth:`deploy_initial_candidate`) reports ``"promoted"``, the
        caller routes the outcome here so prediction serving resolves
        the same model+version the promotion designated. Only a
        ``"promoted"`` outcome is publishable; rejected, initial-gate,
        and failed outcomes raise instead of moving the reference, so
        serving can never silently follow a non-promoted candidate.
        The write is atomic as practical (temporary file + rename), so
        a failed publish leaves the previous reference untouched.

        Args:
            promotion_result: Promoter outcome dict with ``decision``,
                ``model_name``, ``run_id``, ``registered_model_name``,
                ``model_version``. Never mutated.
            models_dir: Filesystem directory override for this call.
                ``None`` falls back to the constructor default (itself
                ``None`` => persistence default).
            use_case: Use-case label recorded in the reference.

        Returns:
            The published production pointer mapping.

        Raises:
            ModelPromotionError: If the outcome is not promotable, is
                malformed, or the reference write fails.
        """
        if not isinstance(promotion_result, dict):
            raise ModelPromotionError(
                "promotion_result must be a dict for production publication."
            )
        if promotion_result.get("decision") != PROMOTED:
            raise ModelPromotionError(
                "Only a 'promoted' outcome can become production; got "
                f"{promotion_result.get('decision')!r}. Production reference unchanged."
            )
        model_name = promotion_result.get("model_name")
        run_id = promotion_result.get("run_id")
        registered = promotion_result.get("registered_model_name")
        version = promotion_result.get("model_version")
        if not _is_non_empty_string(model_name):
            raise ModelPromotionError(
                "promotion_result is missing model_name; refusing to publish production."
            )
        if not _is_non_empty_string(run_id):
            raise ModelPromotionError(
                "promotion_result is missing run_id; refusing to publish production."
            )
        if not _is_non_empty_string(registered):
            raise ModelPromotionError(
                "promotion_result is missing registered_model_name; "
                "refusing to publish production."
            )
        if not _is_non_empty_string(version):
            raise ModelPromotionError(
                "promotion_result is missing model_version; refusing to "
                "fabricate a production version."
            )
        resolved_models_dir = models_dir if models_dir is not None else self._models_dir
        pointer = {
            "model_name": str(model_name).strip(),
            "registered_model_name": str(registered).strip(),
            "model_version": str(version).strip(),
            "run_id": str(run_id).strip(),
            "use_case": str(use_case) if isinstance(use_case, str) and use_case.strip() else "house_price",
        }
        try:
            write_production_pointer(pointer, models_dir=resolved_models_dir)
            stored = read_production_pointer(models_dir=resolved_models_dir)
        except AdaptiveAIError as e:
            raise ModelPromotionError(
                f"Production publication failed for '{pointer['model_name']}' "
                f"(run {pointer['run_id']}); serving reference unchanged: {e}"
            ) from e
        except Exception as e:
            raise ModelPromotionError(
                f"Production publication failed for '{pointer['model_name']}' "
                f"(run {pointer['run_id']}); serving reference unchanged: {e}"
            ) from e
        logger.info(
            "Production reference published: '%s' (%s v%s, run %s).",
            stored["model_name"],
            stored["registered_model_name"],
            stored["model_version"],
            stored["run_id"],
        )
        return stored


def create_house_price_promoter(
    tracker: Optional[Any] = None,
    models_dir: Optional[Union[str, Path]] = None,
) -> ModelPromoter:
    """Builds the default house-price promoter with shared tracking.

    Reuses the existing promoter defaults (decision gate, save
    function, registered name, artifact path); only tracking and the
    filesystem target are injectable for tests. The caller shares the
    returned promoter's tracker with the retraining service so
    registration uses the same backend that tracked the candidate.
    """
    tracker = tracker or MLflowTracker(experiment_name="house_price_retraining")
    return ModelPromoter(tracker=tracker, models_dir=models_dir)
