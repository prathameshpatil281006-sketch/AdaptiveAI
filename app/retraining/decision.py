"""Retraining recommendation service for AdaptiveAI (Phase 5).

Pure, deterministic mapping from a Phase 4 monitoring summary (plus
the serving production identity) to an explicit retraining
recommendation. No I/O, no training, no promotion, no database.

Recommendation rules (explicit constants, all testable):

- ``MIN_WINDOW_ROWS`` (30): windows below the detector floor can
  never yield a confident recommendation (``blocked_by`` gets
  ``insufficient-data``).
- Data quality ``fail`` blocks retraining (``blocked_by`` gets
  ``data-quality-failed``). A ``warn`` is advisory only: it is
  reported as a trigger, never a blocker.
- ``severity == critical`` always recommends (large or broad shift).
- Otherwise retraining is recommended when at least
  ``ACTION_DRIFT_COUNT`` (2) features drifted. A single small-effect
  drift is reported but does NOT recommend retraining — drift alone
  never equals an automatic retraining call.
- Categorical drift is reported with a ``categorical-drift:`` prefix
  so it is visible without changing the numeric production trigger
  semantics (the drift → retraining orchestrator still consumes only
  the numeric detector result).

Returned recommendation::

    {
        "recommended": bool,
        "reason": str,
        "triggers": [str, ...],
        "blocked_by": [str, ...],
        "production_model": str | None,
        "production_version": str | None,
        "monitored_prediction_count": int | None,
    }
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Minimum production window for a confident recommendation (mirrors
# the drift detector floor so a recommendation can never rest on
# data the detector itself calls insufficient).
MIN_WINDOW_ROWS = 30

# Drifted-feature count that recommends retraining even without a
# critical severity (broad shift). Below this, a lone small drift is
# reported but not acted on.
ACTION_DRIFT_COUNT = 2


def _triggers_from_summary(summary: Dict[str, Any]) -> List[str]:
    """Builds trigger labels from drifted features and quality state."""
    triggers: List[str] = []
    features = summary.get("features", [])
    if isinstance(features, list):
        for entry in features:
            if not isinstance(entry, dict) or entry.get("drifted") is not True:
                continue
            name = entry.get("feature_name")
            if not isinstance(name, str) or not name.strip():
                continue
            prefix = (
                "categorical-drift:"
                if entry.get("type") == "categorical"
                else "numeric-drift:"
            )
            triggers.append(f"{prefix}{name}")
    dq_status = summary.get("data_quality_status")
    if dq_status == "warn":
        triggers.append("data-quality-warning")
    return triggers


def recommend_retraining(
    summary: Optional[Dict[str, Any]],
    production_model: Optional[str] = None,
    production_version: Optional[str] = None,
) -> Dict[str, Any]:
    """Recommends (or declines) retraining from a monitoring summary.

    Args:
        summary: Phase 4 health summary mapping (status,
            sufficient_data, production_rows, severity,
            drifted_count, data_quality_status, features). None (or
            a non-dict) means no assessment is possible.
        production_model: Actually-serving model name (informational).
        production_version: Actually-serving version (informational).

    Returns:
        Recommendation dict (see module docstring). Never raises on
        malformed summaries — they yield a blocked, not-recommended
        decision with an explicit reason.
    """
    base = {
        "recommended": False,
        "reason": "",
        "triggers": [],
        "blocked_by": [],
        "production_model": production_model,
        "production_version": production_version,
        "monitored_prediction_count": None,
    }
    if not isinstance(summary, dict):
        base["reason"] = "No monitoring summary available; retraining not recommended."
        base["blocked_by"] = ["insufficient-data"]
        return base

    production_rows = summary.get("production_rows")
    if isinstance(production_rows, bool) or not isinstance(production_rows, int):
        production_rows = None
    base["monitored_prediction_count"] = production_rows

    sufficient = summary.get("sufficient_data") is True
    status_ok = summary.get("status") == "ok"
    rows_ok = (
        isinstance(production_rows, int) and production_rows >= MIN_WINDOW_ROWS
    )
    if not (sufficient and status_ok and rows_ok):
        base["reason"] = (
            "Insufficient monitoring data for a confident recommendation "
            f"(rows={production_rows}, minimum {MIN_WINDOW_ROWS})."
        )
        base["blocked_by"] = ["insufficient-data"]
        return base

    dq_status = summary.get("data_quality_status")
    if dq_status == "fail":
        base["reason"] = (
            "Retraining blocked: data quality failed on the current window. "
            "Fix the data pipeline before retraining."
        )
        base["blocked_by"] = ["data-quality-failed"]
        base["triggers"] = _triggers_from_summary(summary)
        return base
    if dq_status not in ("pass", "warn"):
        base["reason"] = (
            "Retraining blocked: data-quality state is unknown for the window."
        )
        base["blocked_by"] = ["data-quality-unknown"]
        return base

    triggers = _triggers_from_summary(summary)
    base["triggers"] = triggers
    severity = summary.get("severity")
    drifted = summary.get("drifted_features", [])
    drifted_count = summary.get("drifted_count")
    if not isinstance(drifted_count, int) or isinstance(drifted_count, bool):
        drifted_count = len(drifted) if isinstance(drifted, list) else 0

    if severity == "critical":
        base["recommended"] = True
        base["reason"] = (
            f"Retraining recommended: critical drift severity with "
            f"{drifted_count} drifted feature(s)."
        )
        return base
    if drifted_count >= ACTION_DRIFT_COUNT:
        base["recommended"] = True
        base["reason"] = (
            f"Retraining recommended: broad shift across {drifted_count} "
            f"features (action threshold {ACTION_DRIFT_COUNT})."
        )
        return base
    if drifted_count > 0:
        base["reason"] = (
            f"Retraining not recommended: a single small-effect drift "
            f"({drifted_count} feature(s)) is below the action threshold "
            f"({ACTION_DRIFT_COUNT}); monitor and re-check."
        )
        return base
    base["reason"] = "Retraining not recommended: no drift and data quality passes."
    return base
