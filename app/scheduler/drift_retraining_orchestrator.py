"""Explicit drift-to-retraining orchestration (Step 39).

Connects one monitoring check to one drift advice to at most one
scheduled retraining cycle, only on explicit demand::

    monitoring service
        |
        v
    monitoring result
        |
        v
    trigger advice
        |
        v
    retraining cycle (only when advice requires it)

Design notes:

- Dependency injection only. No globals, no singletons, no
  import-time side effects.
- Exactly one explicit operation. No loops, no waits, no workers,
  no automatic execution, and no external scheduling libraries.
- No drift logic and no training logic here. The advice layer owns
  the decision; the cycle layer owns the execution. Results are
  returned as-is, never rebuilt and never mutated.
- Failures from any stage propagate unchanged so a failed run is
  never reported as success.
"""

import logging
from typing import Any, Dict

from app.core.exceptions import AdaptiveAIError, RetrainingError

logger = logging.getLogger(__name__)


class DriftRetrainingOrchestrator:
    """Manual bridge from a monitoring check to a retraining cycle."""

    def __init__(
        self,
        monitoring_service: Any,
        trigger_decision: Any,
        retraining_scheduler: Any,
    ) -> None:
        """Wires the three collaborators with injection for testing.

        Args:
            monitoring_service: Object exposing ``run_check()``.
            trigger_decision: Object exposing ``decide(result)``.
            retraining_scheduler: Object exposing
                ``run_once(promote=...)``.

        Raises:
            AdaptiveAIError: If any collaborator misses its entry point.
        """
        if not callable(getattr(monitoring_service, "run_check", None)):
            raise AdaptiveAIError(
                "monitoring_service must expose a callable run_check()."
            )
        if not callable(getattr(trigger_decision, "decide", None)):
            raise AdaptiveAIError(
                "trigger_decision must expose a callable decide()."
            )
        if not callable(getattr(retraining_scheduler, "run_once", None)):
            raise AdaptiveAIError(
                "retraining_scheduler must expose a callable run_once()."
            )
        self._monitoring_service = monitoring_service
        self._trigger_decision = trigger_decision
        self._retraining_scheduler = retraining_scheduler

    @property
    def monitoring_service(self) -> Any:
        """Returns the injected monitoring service."""
        return self._monitoring_service

    @property
    def trigger_decision(self) -> Any:
        """Returns the injected trigger advice layer."""
        return self._trigger_decision

    @property
    def retraining_scheduler(self) -> Any:
        """Returns the injected retraining cycle layer."""
        return self._retraining_scheduler

    def run_check_and_retrain(self, promote: bool = True) -> Dict[str, Any]:
        """Runs one check, one advice step, and at most one cycle.

        Args:
            promote: Explicit flag forwarded unchanged to the cycle
                layer. Must be a real boolean. Defaults to True.

        Returns:
            Dict with the exact ``monitoring_result``, the exact
            ``retraining_decision``, and ``retraining_result`` (the
            exact cycle result, or None when advice is negative).

        Raises:
            RetrainingError: If ``promote`` is not a boolean.
            AdaptiveAIError: If the advice output is malformed.
            Exception: Stage failures propagate unchanged.
        """
        if not isinstance(promote, bool):
            raise RetrainingError("promote must be a boolean.")

        monitoring_result = self._monitoring_service.run_check()
        retraining_decision = self._trigger_decision.decide(monitoring_result)

        if not isinstance(retraining_decision, dict):
            raise AdaptiveAIError(
                "Trigger advice must return a dict, "
                f"got {type(retraining_decision).__name__}."
            )
        decision = retraining_decision.get("decision")
        if decision not in ("retraining_required", "no_retraining"):
            raise AdaptiveAIError(
                f"Unsupported trigger advice: {decision!r}. "
                "Expected 'retraining_required' or 'no_retraining'."
            )

        if decision != "retraining_required":
            logger.info(
                "Drift orchestration completed without retraining: decision=%s.",
                decision,
            )
            return {
                "monitoring_result": monitoring_result,
                "retraining_decision": retraining_decision,
                "retraining_result": None,
            }

        logger.info("Drift orchestration triggered a retraining cycle.")
        retraining_result = self._retraining_scheduler.run_once(promote=promote)
        return {
            "monitoring_result": monitoring_result,
            "retraining_decision": retraining_decision,
            "retraining_result": retraining_result,
        }
