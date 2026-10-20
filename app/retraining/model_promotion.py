"""Safe model promotion decision layer for AdaptiveAI (Step 32).

Decision-only component that maps a :class:`ModelEvaluator`
candidate-vs-production comparison result to an explicit promotion
decision::

    RetrainingService
        |
        v
    ModelEvaluator (candidate vs production -> evaluation_result)
        |
        v
    ModelPromotionDecision (evaluation_result -> decision, no side effects)

Decision vocabulary (stable contract):

    - ``"promote"``: candidate explicitly better; caller MAY promote later.
    - ``"reject"``: keep production; candidate not better or invalid.
    - ``"requires_initial_promotion"``: no production model exists; the
      caller must explicitly handle first deployment (never auto-promote
      here).

The layer is deterministic and side-effect free: it never saves,
overwrites, registers, or promotes models, and never touches SQLite,
the MLflow registry, the filesystem, the API, the scheduler, or the
dashboard. It never mutates its input. Promotion requires the
evaluator's explicit comparison result -- candidate existence alone
never promotes.
"""

import logging
import math
import numbers
from typing import Any, Dict, Optional

from app.core.exceptions import ModelPromotionError
from app.retraining.model_evaluator import (
    CANDIDATE_BETTER,
    NO_PRODUCTION_MODEL,
    PRODUCTION_BETTER,
)

logger = logging.getLogger(__name__)

# Stable decision vocabulary. Callers should compare against these
# constants rather than hard-coding strings.
DECISION_PROMOTE = "promote"
DECISION_REJECT = "reject"
DECISION_REQUIRES_INITIAL_PROMOTION = "requires_initial_promotion"


def _is_finite_number(value: Any) -> bool:
    """Returns True only for real, finite, non-bool numbers."""
    if isinstance(value, bool):
        return False
    if not isinstance(value, numbers.Real):
        return False
    try:
        as_float = float(value)
    except (TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(as_float)


def _safe_float(value: Any) -> Optional[float]:
    """Converts a finite number to float, else None (never raises)."""
    if _is_finite_number(value):
        return float(value)
    return None


def _safe_primary_metric(value: Any) -> Optional[str]:
    """Returns a stripped non-empty string metric name, else None."""
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


class ModelPromotionDecision:
    """Maps a ModelEvaluator result to a promotion decision.

    The input must be the standard evaluator result dict with ``status``
    and ``comparison`` keys (see :meth:`ModelEvaluator.evaluate_candidate`).
    The input dictionary is never mutated; a fresh decision dict is
    returned on every call and identical inputs always yield identical
    outputs.
    """

    DECISION_PROMOTE = DECISION_PROMOTE
    DECISION_REJECT = DECISION_REJECT
    DECISION_REQUIRES_INITIAL_PROMOTION = DECISION_REQUIRES_INITIAL_PROMOTION

    def decide(self, evaluation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Decides promotion from an evaluator comparison result.

        Args:
            evaluation_result: Evaluator output dict with ``status`` in
                ``{"candidate_better", "production_better",
                "no_production_model"}`` and a ``comparison`` block.

        Returns:
            Dict with ``decision``, ``reason``, ``primary_metric``,
            ``candidate_score``, ``production_score``, ``improvement``::

                {
                    "decision": "promote" | "reject"
                                | "requires_initial_promotion",
                    "reason": str,
                    "primary_metric": str | None,
                    "candidate_score": float | None,
                    "production_score": float | None,
                    "improvement": float | None,
                }

        Raises:
            ModelPromotionError: If the envelope is invalid (non-dict,
                missing/unknown status) or a ``candidate_better`` claim is
                structurally invalid (missing comparison, missing/invalid
                primary metric, non-numeric scores). ``ModelPromotionError``
                subclasses ``AdaptiveAIError``, so ``pytest.raises(
                AdaptiveAIError)`` also catches it.
        """
        if not isinstance(evaluation_result, dict):
            raise ModelPromotionError(
                "evaluation_result must be a dict with 'status' and 'comparison'."
            )
        status = evaluation_result.get("status")
        if status == CANDIDATE_BETTER:
            return self._decide_candidate_better(evaluation_result)
        if status == PRODUCTION_BETTER:
            return self._decide_production_better(evaluation_result)
        if status == NO_PRODUCTION_MODEL:
            return self._decide_no_production(evaluation_result)
        raise ModelPromotionError(
            f"Unsupported evaluation status: {status!r}. "
            f"Expected one of: {CANDIDATE_BETTER!r}, {PRODUCTION_BETTER!r}, "
            f"{NO_PRODUCTION_MODEL!r}."
        )

    def _decide_candidate_better(
        self, evaluation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validates a candidate_better claim; promotes only if fully valid."""
        comparison = evaluation_result.get("comparison")
        if not isinstance(comparison, dict):
            raise ModelPromotionError(
                "Invalid evaluation result for status 'candidate_better': "
                "'comparison' must be a dict."
            )
        primary_metric = comparison.get("primary_metric")
        if not isinstance(primary_metric, str) or not primary_metric.strip():
            raise ModelPromotionError(
                "Invalid evaluation result for status 'candidate_better': "
                "'comparison.primary_metric' must be a non-empty string."
            )
        primary_metric = primary_metric.strip()

        candidate_score = comparison.get("candidate_score")
        production_score = comparison.get("production_score")
        improvement = comparison.get("improvement")
        candidate_better_flag = comparison.get("candidate_better")

        if not _is_finite_number(candidate_score):
            raise ModelPromotionError(
                "Invalid evaluation result for status 'candidate_better': "
                "'comparison.candidate_score' must be a finite number."
            )
        if not _is_finite_number(production_score):
            raise ModelPromotionError(
                "Invalid evaluation result for status 'candidate_better': "
                "'comparison.production_score' must be a finite number."
            )
        if not _is_finite_number(improvement):
            raise ModelPromotionError(
                "Invalid evaluation result for status 'candidate_better': "
                "'comparison.improvement' must be a finite number."
            )

        candidate_f = float(candidate_score)
        production_f = float(production_score)
        improvement_f = float(improvement)

        # Logical consistency checks fail closed with "reject" (structure
        # is valid, but the evidence does not support promotion).
        if candidate_better_flag is not True:
            logger.info(
                "Promotion rejected: status is candidate_better but "
                "comparison.candidate_better is not True (%r).",
                candidate_better_flag,
            )
            return {
                "decision": DECISION_REJECT,
                "reason": (
                    "Rejected: evaluator status is 'candidate_better' but "
                    "comparison.candidate_better is not exactly True."
                ),
                "primary_metric": primary_metric,
                "candidate_score": candidate_f,
                "production_score": production_f,
                "improvement": improvement_f,
            }
        if not (improvement_f > 0):
            logger.info(
                "Promotion rejected: non-positive improvement (%r).", improvement_f
            )
            return {
                "decision": DECISION_REJECT,
                "reason": (
                    "Rejected: improvement must be > 0 for promotion, "
                    f"got {improvement_f}."
                ),
                "primary_metric": primary_metric,
                "candidate_score": candidate_f,
                "production_score": production_f,
                "improvement": improvement_f,
            }

        logger.info(
            "Promotion approved: %s candidate=%.4f production=%.4f improvement=%+.4f",
            primary_metric,
            candidate_f,
            production_f,
            improvement_f,
        )
        return {
            "decision": DECISION_PROMOTE,
            "reason": (
                f"Candidate is better on '{primary_metric}' "
                f"(improvement {improvement_f:+.4f})."
            ),
            "primary_metric": primary_metric,
            "candidate_score": candidate_f,
            "production_score": production_f,
            "improvement": improvement_f,
        }

    def _decide_production_better(
        self, evaluation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Always rejects; extracts scores safely without raising on shape."""
        comparison = evaluation_result.get("comparison")
        if not isinstance(comparison, dict):
            logger.info("Promotion rejected: missing/invalid comparison block.")
            return {
                "decision": DECISION_REJECT,
                "reason": (
                    "Rejected: production is better or equal; "
                    "comparison block is missing/invalid."
                ),
                "primary_metric": None,
                "candidate_score": None,
                "production_score": None,
                "improvement": None,
            }

        primary_metric = _safe_primary_metric(comparison.get("primary_metric"))
        candidate_f = _safe_float(comparison.get("candidate_score"))
        production_f = _safe_float(comparison.get("production_score"))
        improvement_f = _safe_float(comparison.get("improvement"))
        candidate_better_flag = comparison.get("candidate_better")

        # Inconsistencies (flag True, positive improvement) still reject --
        # status says production is better, so fail safe with reject.
        if candidate_better_flag is True:
            reason = (
                "Rejected: status is 'production_better' but "
                "comparison.candidate_better is True (inconsistent); "
                "keeping production."
            )
        elif improvement_f is not None and improvement_f > 0:
            reason = (
                "Rejected: status is 'production_better' but improvement "
                f"is positive ({improvement_f:+.4f}, inconsistent); "
                "keeping production."
            )
        else:
            reason = "Production is better or equal; keeping production."
        logger.info("Promotion rejected: status=production_better. %s", reason)
        return {
            "decision": DECISION_REJECT,
            "reason": reason,
            "primary_metric": primary_metric,
            "candidate_score": candidate_f,
            "production_score": production_f,
            "improvement": improvement_f,
        }

    def _decide_no_production(
        self, evaluation_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Never auto-promotes; routes first deployment to explicit handling."""
        comparison = evaluation_result.get("comparison")
        if not isinstance(comparison, dict):
            comparison = {}
        primary_metric = _safe_primary_metric(comparison.get("primary_metric"))
        candidate_f = _safe_float(comparison.get("candidate_score"))
        # Production/improvement are expected to be None here; extract
        # leniently in case the evaluator shape evolves.
        production_f = _safe_float(comparison.get("production_score"))
        improvement_f = _safe_float(comparison.get("improvement"))
        logger.info(
            "No production model: requires explicit initial promotion handling."
        )
        return {
            "decision": DECISION_REQUIRES_INITIAL_PROMOTION,
            "reason": (
                "No production model exists; initial promotion requires "
                "explicit handling (not auto-approved)."
            ),
            "primary_metric": primary_metric,
            "candidate_score": candidate_f,
            "production_score": production_f,
            "improvement": improvement_f,
        }
