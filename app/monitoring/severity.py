"""Drift/data-quality severity classification for AdaptiveAI (Phase 4).

Deterministic mapping from statistical evidence to one of:

- ``healthy``: sufficient data, nothing drifted, data quality passes.
- ``warning``: at least one drifted feature with a small/moderate
  effect, or data quality raised warnings — attention, no emergency.
- ``critical``: at least one drifted feature with a large effect, or
  drift spread across 3+ features, or data quality failed.
- ``unknown``: insufficient production data to say anything honest.

Documented thresholds (project conventions, NOT universal claims and
NOT a model-quality score — severity describes distribution shift
and sample adequacy only):

- Numeric effect size = KS D statistic:
  large when D >= ``KS_LARGE_EFFECT`` (0.30).
- Categorical effect size = Population Stability Index (PSI):
  large when PSI >= ``PSI_LARGE_EFFECT`` (0.25, the conventional
  large-shift band; 0.10-0.25 reads as moderate).
- Broad shift = ``BROAD_DRIFT_FEATURE_COUNT`` (3) or more drifted
  features across both families.

Overall health combines drift severity with data-quality status
(``pass``/``warn``/``fail``/``unknown`` from the data-quality
service): critical wins over warning wins over healthy; unknown only
when the drift side itself is unknown.

No I/O, no database, no model access. Pure functions of plain data.
"""

from typing import Any, Dict, List, Optional

SEVERITY_HEALTHY = "healthy"
SEVERITY_WARNING = "warning"
SEVERITY_CRITICAL = "critical"
SEVERITY_UNKNOWN = "unknown"

# Effect-size cut points (documented above).
KS_LARGE_EFFECT = 0.30
PSI_LARGE_EFFECT = 0.25

# Number of drifted features that counts as a broad shift.
BROAD_DRIFT_FEATURE_COUNT = 3


def _effect_of(entry: Dict[str, Any]) -> Optional[float]:
    """Extracts the comparable effect size for one drifted feature."""
    kind = entry.get("type", "numeric")
    if kind == "categorical":
        value = entry.get("effect_size")
    else:
        value = entry.get("statistic")
    try:
        as_float = float(value)
    except (TypeError, ValueError):
        return None
    import math

    if not math.isfinite(as_float) or as_float < 0:
        return None
    return as_float


def classify_drift_severity(
    numeric_features: Optional[List[Dict[str, Any]]] = None,
    categorical_features: Optional[List[Dict[str, Any]]] = None,
    sufficient_data: bool = True,
) -> Dict[str, Any]:
    """Classifies drift severity from per-feature drift evidence.

    Args:
        numeric_features: Numeric drift entries (each with
            ``drifted`` bool and KS ``statistic``).
        categorical_features: Categorical drift entries (each with
            ``drifted`` bool and PSI ``effect_size``).
        sufficient_data: False forces ``unknown`` regardless.

    Returns:
        Dict with ``severity``, ``drifted_features`` (names),
        ``drifted_count``, ``large_effect_features`` (names), and a
        human-readable ``reason``.
    """
    numeric_features = list(numeric_features or [])
    categorical_features = list(categorical_features or [])
    if not sufficient_data:
        return {
            "severity": SEVERITY_UNKNOWN,
            "drifted_features": [],
            "drifted_count": 0,
            "large_effect_features": [],
            "reason": "Insufficient production data for drift assessment.",
        }
    drifted = [
        str(e.get("feature_name"))
        for e in (numeric_features + categorical_features)
        if isinstance(e, dict) and e.get("drifted") is True
    ]
    large: List[str] = []
    for entry in numeric_features + categorical_features:
        if not isinstance(entry, dict) or entry.get("drifted") is not True:
            continue
        effect = _effect_of(entry)
        if effect is None:
            continue
        threshold = (
            PSI_LARGE_EFFECT
            if entry.get("type") == "categorical"
            else KS_LARGE_EFFECT
        )
        if effect >= threshold:
            large.append(str(entry.get("feature_name")))
    if large or len(drifted) >= BROAD_DRIFT_FEATURE_COUNT:
        return {
            "severity": SEVERITY_CRITICAL,
            "drifted_features": drifted,
            "drifted_count": len(drifted),
            "large_effect_features": large,
            "reason": (
                f"{len(drifted)} drifted feature(s) with large/broad shift: "
                + (", ".join(large or drifted))
            ),
        }
    if drifted:
        return {
            "severity": SEVERITY_WARNING,
            "drifted_features": drifted,
            "drifted_count": len(drifted),
            "large_effect_features": [],
            "reason": f"{len(drifted)} drifted feature(s) with small/moderate effect: "
            + ", ".join(drifted),
        }
    return {
        "severity": SEVERITY_HEALTHY,
        "drifted_features": [],
        "drifted_count": 0,
        "large_effect_features": [],
        "reason": "No drifted features with sufficient data.",
    }


def classify_overall_health(
    drift_severity: str, data_quality_status: Optional[str]
) -> Dict[str, str]:
    """Combines drift severity with data-quality status.

    Args:
        drift_severity: One of healthy/warning/critical/unknown.
        data_quality_status: One of pass/warn/fail/unknown (or None,
            treated as unknown only when drift is also unknown;
            otherwise ignored as "not assessed").

    Returns:
        Dict with ``overall_health`` and ``reason``.
    """
    valid_drift = {
        SEVERITY_HEALTHY,
        SEVERITY_WARNING,
        SEVERITY_CRITICAL,
        SEVERITY_UNKNOWN,
    }
    if drift_severity not in valid_drift:
        drift_severity = SEVERITY_UNKNOWN
    dq = (data_quality_status or "").strip().lower()
    if dq not in {"pass", "warn", "fail", "unknown", ""}:
        dq = "unknown"
    if drift_severity == SEVERITY_UNKNOWN and dq in {"unknown", ""}:
        return {
            "overall_health": SEVERITY_UNKNOWN,
            "reason": "Insufficient production data for health assessment.",
        }
    if drift_severity == SEVERITY_CRITICAL or dq == "fail":
        motive = []
        if drift_severity == SEVERITY_CRITICAL:
            motive.append("drift is critical")
        if dq == "fail":
            motive.append("data quality failed")
        return {
            "overall_health": SEVERITY_CRITICAL,
            "reason": "Critical health: " + " and ".join(motive) + ".",
        }
    if drift_severity == SEVERITY_WARNING or dq == "warn":
        motive = []
        if drift_severity == SEVERITY_WARNING:
            motive.append("drift warning")
        if dq == "warn":
            motive.append("data-quality warning")
        return {
            "overall_health": SEVERITY_WARNING,
            "reason": "Health warning: " + " and ".join(motive) + ".",
        }
    if drift_severity == SEVERITY_UNKNOWN:
        return {
            "overall_health": SEVERITY_UNKNOWN,
            "reason": "Drift assessment unknown; data quality did not fail.",
        }
    return {
        "overall_health": SEVERITY_HEALTHY,
        "reason": "No drift and data quality passes with sufficient data.",
    }
