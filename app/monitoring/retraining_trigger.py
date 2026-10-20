"""Deterministic decision layer mapping drift results to advice (Step 38).

Converts the existing monitoring result contract into a retraining
decision without triggering any work::

    monitoring result
        |
        v
    RetrainingTriggerDecision.decide(...)
        |
        v
    {"decision": "retraining_required" | "no_retraining", ...}

Design notes:

- Pure function of the input mapping. Deterministic, side-effect free.
- Reads only the existing result fields. Never recomputes drift,
  never invents statistics, never mutates the input.
- Insufficient production data never yields a positive decision.
- Malformed or contradictory results raise instead of being
  reinterpreted.
- This module triggers nothing, persists nothing, and talks to no
  filesystem, database, registry, model store, service, or UI layer.
"""

import logging
import math
from typing import Any, Dict, List

from app.core.exceptions import DriftDetectionError

logger = logging.getLogger(__name__)

DECISION_REQUIRED = "retraining_required"
DECISION_NONE = "no_retraining"

SUPPORTED_STATUSES = frozenset({"ok", "insufficient_data"})


def _require_bool(value: Any, field: str) -> bool:
    """Validates a strict boolean (rejects 0/1/strings)."""
    if not isinstance(value, bool):
        raise DriftDetectionError(f"'{field}' must be a boolean, got {type(value).__name__}.")
    return value


def _require_alpha(value: Any) -> float:
    """Validates alpha as a finite number strictly inside (0, 1)."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DriftDetectionError("'alpha' must be a finite number in (0, 1).")
    as_float = float(value)
    if not math.isfinite(as_float) or not 0.0 < as_float < 1.0:
        raise DriftDetectionError("'alpha' must be a finite number in (0, 1).")
    return as_float


def _require_min_production_rows(value: Any) -> int:
    """Validates min_production_rows as a positive integer (not bool)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise DriftDetectionError("'min_production_rows' must be a positive integer.")
    if value < 1:
        raise DriftDetectionError("'min_production_rows' must be a positive integer.")
    return int(value)


def _require_production_rows(value: Any) -> int:
    """Validates production_rows as a non-negative integer (not bool)."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise DriftDetectionError("'production_rows' must be a non-negative integer.")
    if value < 0:
        raise DriftDetectionError("'production_rows' must be a non-negative integer.")
    return int(value)


def _extract_drifted_features(features: Any) -> List[str]:
    """Returns names whose existing 'drifted' flag is True.

    Validates record shapes strictly but never recomputes anything.
    """
    if not isinstance(features, list):
        raise DriftDetectionError("'features' must be a list.")
    drifted: List[str] = []
    for entry in features:
        if not isinstance(entry, dict):
            raise DriftDetectionError("Each entry in 'features' must be a dict.")
        name = entry.get("feature_name")
        if not isinstance(name, str) or not name.strip():
            raise DriftDetectionError("Each feature entry needs a non-empty 'feature_name'.")
        flag = entry.get("drifted")
        if not isinstance(flag, bool):
            raise DriftDetectionError(
                f"Feature '{name}' has an invalid 'drifted' flag: "
                f"expected a boolean, got {type(flag).__name__}."
            )
        if flag:
            drifted.append(str(name))
    return drifted


class RetrainingTriggerDecision:
    """Stateless mapping from monitoring results to retraining advice."""

    def decide(self, monitoring_result: Dict[str, Any]) -> Dict[str, Any]:
        """Maps one monitoring result to a retraining decision.

        Args:
            monitoring_result: Existing monitoring result mapping with
                overall_drifted, alpha, min_production_rows,
                production_rows, sufficient_data, status, and features.

        Returns:
            New dict with decision, reason, status, overall_drifted,
            production_rows, and drifted_features. The input is never
            mutated.

        Raises:
            DriftDetectionError: If the input is missing, not a dict,
                or contradicts the monitoring result contract.
        """
        if not isinstance(monitoring_result, dict):
            raise DriftDetectionError(
                "monitoring_result must be a dict, "
                f"got {type(monitoring_result).__name__}."
            )
        for key in (
            "overall_drifted",
            "alpha",
            "min_production_rows",
            "production_rows",
            "sufficient_data",
            "status",
            "features",
        ):
            if key not in monitoring_result:
                raise DriftDetectionError(f"monitoring_result is missing '{key}'.")

        status = monitoring_result["status"]
        if status not in SUPPORTED_STATUSES:
            raise DriftDetectionError(
                f"Unsupported monitoring status: {status!r}. "
                "Expected 'ok' or 'insufficient_data'."
            )
        overall_drifted = _require_bool(monitoring_result["overall_drifted"], "overall_drifted")
        sufficient_data = _require_bool(monitoring_result["sufficient_data"], "sufficient_data")
        alpha = _require_alpha(monitoring_result["alpha"])
        min_production_rows = _require_min_production_rows(
            monitoring_result["min_production_rows"]
        )
        production_rows = _require_production_rows(monitoring_result["production_rows"])

        if status == "ok" and sufficient_data is not True:
            raise DriftDetectionError(
                "Contradictory monitoring result: status 'ok' requires sufficient_data=True."
            )
        if status == "insufficient_data" and sufficient_data is not False:
            raise DriftDetectionError(
                "Contradictory monitoring result: status 'insufficient_data' "
                "requires sufficient_data=False."
            )

        drifted_features = _extract_drifted_features(monitoring_result["features"])

        if status == "insufficient_data":
            logger.info(
                "Retraining advice: no_retraining (insufficient_data, rows=%d).",
                production_rows,
            )
            return {
                "decision": DECISION_NONE,
                "reason": (
                    "Insufficient production data "
                    f"({production_rows} rows, minimum {min_production_rows}); "
                    "retraining is not triggered."
                ),
                "status": str(status),
                "overall_drifted": bool(overall_drifted),
                "production_rows": int(production_rows),
                "drifted_features": [],
            }

        if overall_drifted:
            logger.info(
                "Retraining advice: retraining_required (%d drifted feature(s)).",
                len(drifted_features),
            )
            detail = (
                f"drift detected in {len(drifted_features)} feature(s): "
                + ", ".join(drifted_features)
                if drifted_features
                else "drift detected (alpha=%.4f)" % alpha
            )
            return {
                "decision": DECISION_REQUIRED,
                "reason": f"Drift detected; retraining is required ({detail}).",
                "status": str(status),
                "overall_drifted": True,
                "production_rows": int(production_rows),
                "drifted_features": list(drifted_features),
            }

        logger.info("Retraining advice: no_retraining (no drift).")
        return {
            "decision": DECISION_NONE,
            "reason": "No drift detected; retraining is not required.",
            "status": str(status),
            "overall_drifted": False,
            "production_rows": int(production_rows),
            "drifted_features": [],
        }
