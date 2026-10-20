"""Promotion eligibility gate for AdaptiveAI (Phase 5).

Pure, deterministic evaluation of whether a trained candidate may be
promoted. Eligibility is NOT promotion: an eligible candidate still
requires an explicit promotion call, and an explicit call re-runs
this gate first.

Documented checks (each returns ``{name, passed, message}``):

1. ``run_completed`` — the lifecycle run finished training.
2. ``artifact_available`` — the staged candidate file exists.
3. ``artifact_loadable`` — the staged file deserializes and predicts.
4. ``schema_compatible`` — a trial prediction on current-schema
   inputs succeeds (an incompatible candidate can never pass).
5. ``metrics_available`` — candidate MAE/RMSE/R2 are present and finite.
6. ``production_baseline`` — current production metrics exist, so an
   improvement can be measured. With no production model the gate
   fails closed with "initial deployment required" (first deployment
   stays on the explicit deploy-initial path, never the upgrade path).
7. ``regression_tolerance`` — ``candidate_r2 - production_r2`` strictly
   exceeds ``REGRESSION_TOLERANCE_R2`` (default 0.0: any measured
   regression blocks; ties block too).
8. ``data_quality`` — the current prediction window must not fail
   data quality (``fail`` or ``unknown`` block; ``warn`` passes with a
   note so advisories never silently promote into an incident).

``eligible`` is True only when every check passes. No I/O: callers
supply the evidence (loaded pipeline trial results, comparison,
quality status); this module only judges it.
"""

import logging
import math
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default regression tolerance on R2 improvement. Any measured
# regression — including a tie — blocks promotion.
REGRESSION_TOLERANCE_R2 = 0.0

# Absolute candidate-quality floor for schema-migration promotion.
# No absolute quality threshold exists anywhere else in the project
# (all existing bars are relative: REGRESSION_TOLERANCE_R2 compares
# candidate R2 against production R2, which is unmeasurable when the
# production baseline is schema-incompatible). This floor reuses the
# documented tolerance constant as an absolute bar: a migration
# candidate must explain strictly more variance than the training
# mean (R2 > 0). It is migration-specific, reported explicitly in the
# migration_validation message, and never applied to normal upgrades.
MIGRATION_MIN_R2 = 0.0

# Human-facing migration notice. Surfaced in promotion-check responses
# whenever the production baseline cannot be measured because the
# serving artifact cannot predict on the current feature schema.
MIGRATION_REQUIRED_MESSAGE = (
    "Schema migration required: production baseline is incompatible "
    "with candidate schema."
)


def _check(name: str, passed: bool, message: str) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "message": str(message)}


def _finite_number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    as_float = float(value)
    return as_float if math.isfinite(as_float) else None


def check_promotion_eligibility(
    run_completed: bool = False,
    artifact_available: bool = False,
    artifact_loadable: bool = False,
    schema_compatible: bool = False,
    candidate_metrics: Optional[Dict[str, Any]] = None,
    production_metrics: Optional[Dict[str, Any]] = None,
    improvement: Optional[float] = None,
    data_quality_status: Optional[str] = None,
    regression_tolerance: float = REGRESSION_TOLERANCE_R2,
) -> Dict[str, Any]:
    """Evaluates promotion eligibility from explicit evidence.

    Args:
        run_completed: Lifecycle run reached ``completed``.
        artifact_available: Staged candidate file exists.
        artifact_loadable: Staged file deserializes and predicts.
        schema_compatible: Trial prediction on current-schema inputs.
        candidate_metrics: Candidate ``{mae, rmse, r2}`` (finite).
        production_metrics: Current production ``{mae, rmse, r2}``
            (None when no production model exists).
        improvement: ``candidate_r2 - production_r2`` (None when no
            baseline exists).
        data_quality_status: Current window ``pass``/``warn``/``fail``/
            ``unknown`` (None treated as unknown).
        regression_tolerance: Minimum required improvement.

    Returns:
        Dict with ``eligible`` bool and ``checks`` list.
    """
    checks: List[Dict[str, Any]] = []
    checks.append(
        _check(
            "run_completed",
            run_completed is True,
            "Training run completed."
            if run_completed is True
            else "Training run did not complete; promotion refused.",
        )
    )
    checks.append(
        _check(
            "artifact_available",
            artifact_available is True,
            "Staged candidate artifact exists."
            if artifact_available is True
            else "Staged candidate artifact is missing; promotion refused.",
        )
    )
    checks.append(
        _check(
            "artifact_loadable",
            artifact_loadable is True,
            "Staged candidate loads and predicts."
            if artifact_loadable is True
            else "Staged candidate cannot be loaded; promotion refused.",
        )
    )
    checks.append(
        _check(
            "schema_compatible",
            schema_compatible is True,
            "Candidate predicts on the current feature schema."
            if schema_compatible is True
            else "Candidate is schema-incompatible; promotion refused.",
        )
    )

    metrics_ok = False
    if isinstance(candidate_metrics, dict):
        metrics_ok = all(
            _finite_number(candidate_metrics.get(key)) is not None
            for key in ("mae", "rmse", "r2")
        )
    checks.append(
        _check(
            "metrics_available",
            metrics_ok,
            "Candidate MAE/RMSE/R2 are present and finite."
            if metrics_ok
            else "Candidate metrics are missing or non-finite; promotion refused.",
        )
    )

    baseline_ok = False
    if isinstance(production_metrics, dict):
        baseline_ok = all(
            _finite_number(production_metrics.get(key)) is not None
            for key in ("mae", "rmse", "r2")
        )
    checks.append(
        _check(
            "production_baseline",
            baseline_ok,
            "Current production baseline is measurable."
            if baseline_ok
            else "No production baseline; initial deployment is required "
            "(explicit deploy-initial path, never this upgrade path).",
        )
    )

    tolerance = _finite_number(regression_tolerance)
    if tolerance is None:
        tolerance = REGRESSION_TOLERANCE_R2
    gain = _finite_number(improvement)
    tolerance_ok = (
        baseline_ok and gain is not None and gain > tolerance
    )
    checks.append(
        _check(
            "regression_tolerance",
            bool(tolerance_ok),
            f"R2 improvement {gain:+.4f} exceeds tolerance {tolerance:+.4f}."
            if tolerance_ok
            else (
                "Candidate does not beat production "
                f"(improvement {gain} vs tolerance {tolerance}); promotion refused."
            ),
        )
    )

    dq = (data_quality_status or "").strip().lower()
    if dq == "pass" or dq == "warn":
        dq_ok = True
        dq_message = (
            "Current window data quality passes."
            if dq == "pass"
            else "Current window data quality warns; noted, not blocking."
        )
    else:
        dq_ok = False
        dq_message = (
            "Current window data quality failed; promotion refused."
            if dq == "fail"
            else "Current window data quality is unknown; promotion refused."
        )
    checks.append(_check("data_quality", dq_ok, dq_message))

    eligible = all(check["passed"] for check in checks)
    logger.info(
        "Promotion gate evaluated: eligible=%s (%d/%d checks passed).",
        eligible,
        sum(1 for check in checks if check["passed"]),
        len(checks),
    )
    return {"eligible": eligible, "checks": checks}


def check_migration_eligibility(
    run_completed: bool = False,
    artifact_available: bool = False,
    artifact_loadable: bool = False,
    schema_compatible: bool = False,
    candidate_metrics: Optional[Dict[str, Any]] = None,
    candidate_schema_version: Optional[int] = None,
    current_schema_version: Optional[int] = None,
    data_quality_status: Optional[str] = None,
    min_r2: float = MIGRATION_MIN_R2,
) -> Dict[str, Any]:
    """Evaluates schema-migration promotion eligibility from evidence.

    Migration mode applies ONLY when the serving production artifact
    cannot predict on the current feature schema, so no
    production-relative comparison exists. The normal upgrade gate
    (:func:`check_promotion_eligibility`) is untouched and still
    governs every schema-compatible upgrade; this function never
    weakens it.

    Documented checks (each returns ``{name, passed, message}``):

    1. ``run_completed`` — the lifecycle run finished training.
    2. ``artifact_available`` — the staged candidate file exists and
       is traceable to the run (callers only set this when the
       recorded ``candidate_artifact`` resolves to a real file).
    3. ``artifact_loadable`` — the staged file deserializes and predicts.
    4. ``schema_compatible`` — trial prediction on current-schema inputs.
    5. ``metrics_available`` — candidate MAE/RMSE/R2 present and finite.
    6. ``production_baseline`` — always reports
       ``PASS_WITH_INCOMPATIBLE_BASELINE``: the baseline is explicitly
       unmeasurable (production cannot serve the current schema), which
       is the reason migration mode exists. Never a comparison.
    7. ``regression_tolerance`` — always reports
       ``NOT_APPLICABLE_SCHEMA_MIGRATION``: with no baseline there is
       no improvement to compare against a tolerance.
    8. ``migration_validation`` — absolute candidate validation: R2
       strictly exceeds ``min_r2`` (default ``MIGRATION_MIN_R2``) and
       the candidate schema equals the current schema while differing
       from nothing measurable on the production side (production is
       incompatible by construction of this mode).
    9. ``data_quality`` — same semantics as the normal gate
       (``pass``/``warn`` ok; ``fail``/``unknown`` block).

    ``eligible`` is True only when every check passes. No I/O and no
    promotion: eligibility still requires an explicit, separately
    confirmed promotion call.
    """
    checks: List[Dict[str, Any]] = []
    checks.append(
        _check(
            "run_completed",
            run_completed is True,
            "Training run completed."
            if run_completed is True
            else "Training run did not complete; migration promotion refused.",
        )
    )
    checks.append(
        _check(
            "artifact_available",
            artifact_available is True,
            "Staged candidate artifact exists and is traceable to the run."
            if artifact_available is True
            else "Staged candidate artifact is missing; migration promotion refused.",
        )
    )
    checks.append(
        _check(
            "artifact_loadable",
            artifact_loadable is True,
            "Staged candidate loads and predicts."
            if artifact_loadable is True
            else "Staged candidate cannot be loaded; migration promotion refused.",
        )
    )
    checks.append(
        _check(
            "schema_compatible",
            schema_compatible is True,
            "Candidate predicts on the current feature schema."
            if schema_compatible is True
            else "Candidate is schema-incompatible; migration promotion refused.",
        )
    )

    metrics_ok = False
    if isinstance(candidate_metrics, dict):
        metrics_ok = all(
            _finite_number(candidate_metrics.get(key)) is not None
            for key in ("mae", "rmse", "r2")
        )
    checks.append(
        _check(
            "metrics_available",
            metrics_ok,
            "Candidate MAE/RMSE/R2 are present and finite."
            if metrics_ok
            else "Candidate metrics are missing or non-finite; migration promotion refused.",
        )
    )

    checks.append(
        _check(
            "production_baseline",
            True,
            "PASS_WITH_INCOMPATIBLE_BASELINE: production cannot serve the "
            "current schema, so no production baseline is measurable; "
            "migration proceeds on absolute validation only.",
        )
    )
    checks.append(
        _check(
            "regression_tolerance",
            True,
            "NOT_APPLICABLE_SCHEMA_MIGRATION: with no measurable baseline "
            "there is no R2 improvement to compare against a tolerance.",
        )
    )

    floor = _finite_number(min_r2)
    if floor is None:
        floor = MIGRATION_MIN_R2
    candidate_r2 = (
        _finite_number(candidate_metrics.get("r2"))
        if isinstance(candidate_metrics, dict)
        else None
    )
    schema_ok = (
        isinstance(candidate_schema_version, int)
        and not isinstance(candidate_schema_version, bool)
        and isinstance(current_schema_version, int)
        and not isinstance(current_schema_version, bool)
        and candidate_schema_version == current_schema_version
    )
    absolute_ok = (
        metrics_ok
        and candidate_r2 is not None
        and candidate_r2 > floor
        and schema_ok
    )
    if absolute_ok:
        migration_message = (
            f"Migration validation passed: candidate R2 {candidate_r2:+.4f} "
            f"exceeds absolute floor {floor:+.4f} on current schema "
            f"v{current_schema_version}."
        )
    else:
        migration_message = (
            "Migration validation failed: candidate must carry finite "
            f"metrics with R2 > {floor} on current schema "
            f"v{current_schema_version} "
            f"(candidate schema v{candidate_schema_version}, "
            f"R2 {candidate_r2}); migration promotion refused."
        )
    checks.append(_check("migration_validation", bool(absolute_ok), migration_message))

    dq = (data_quality_status or "").strip().lower()
    if dq == "pass" or dq == "warn":
        dq_ok = True
        dq_message = (
            "Current window data quality passes."
            if dq == "pass"
            else "Current window data quality warns; noted, not blocking."
        )
    else:
        dq_ok = False
        dq_message = (
            "Current window data quality failed; migration promotion refused."
            if dq == "fail"
            else "Current window data quality is unknown; migration promotion refused."
        )
    checks.append(_check("data_quality", dq_ok, dq_message))

    eligible = all(check["passed"] for check in checks)
    logger.info(
        "Migration gate evaluated: eligible=%s (%d/%d checks passed).",
        eligible,
        sum(1 for check in checks if check["passed"]),
        len(checks),
    )
    return {
        "eligible": eligible,
        "checks": checks,
        "migration": {
            "required": True,
            "message": MIGRATION_REQUIRED_MESSAGE,
            "candidate_schema_version": candidate_schema_version,
            "current_schema_version": current_schema_version,
        },
    }
