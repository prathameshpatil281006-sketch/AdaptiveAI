"""Monitoring service for AdaptiveAI (Steps 27-28).

Orchestrates the existing MonitoringDataProvider (Step 25) and
DriftDetector (Step 26), optionally persisting each successful
check via MonitoringRepository (Step 28):

    MonitoringService
          |
          v
    MonitoringDataProvider -> reference_features / production_features
          |
          v
    DriftDetector -> structured drift result
          |
          v
    MonitoringRepository.save_monitoring_result(...) (optional)

The service performs no CSV/SQL/training work itself.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from app.core.exceptions import AdaptiveAIError
from app.database.monitoring_repository import MonitoringRepository
from app.database.repository import PredictionRepository
from app.monitoring.categorical_drift import CategoricalDriftDetector
from app.monitoring.data_provider import MonitoringDataProvider
from app.monitoring.data_quality import run_house_price_data_quality
from app.monitoring.drift_detector import (
    DEFAULT_ALPHA,
    DEFAULT_MIN_PRODUCTION_ROWS,
    DriftDetector,
)
from app.monitoring.severity import (
    KS_LARGE_EFFECT,
    PSI_LARGE_EFFECT,
    SEVERITY_CRITICAL,
    SEVERITY_HEALTHY,
    SEVERITY_UNKNOWN,
    SEVERITY_WARNING,
    classify_drift_severity,
    classify_overall_health,
)

logger = logging.getLogger(__name__)

DEFAULT_PRODUCTION_LIMIT = 1000


class MonitoringService:
    """Thin orchestration layer for monitoring checks."""

    def __init__(
        self,
        provider: MonitoringDataProvider,
        detector: Optional[DriftDetector] = None,
        production_limit: int = DEFAULT_PRODUCTION_LIMIT,
        alpha: Optional[float] = None,
        min_production_rows: Optional[int] = None,
        monitoring_repository: Optional[MonitoringRepository] = None,
        categorical_detector: Optional[CategoricalDriftDetector] = None,
    ) -> None:
        """Wires provider, detector, and optional result persistence.

        Args:
            provider: Configured MonitoringDataProvider (any use case).
            detector: Optional preconfigured DriftDetector. When None, one
                is built from alpha/min_production_rows (or their defaults).
            production_limit: Default production history limit per check.
            alpha: Significance level used only when building the default
                detector. Must be None when a detector is provided.
            min_production_rows: Minimum production rows used only when
                building the default detector. Must be None when provided.
            monitoring_repository: Optional MonitoringRepository used to
                persist each successful check. When None, no persistence
                occurs and behavior matches Step 27 exactly.
        """
        if not isinstance(provider, MonitoringDataProvider):
            raise AdaptiveAIError("provider must be a MonitoringDataProvider instance.")
        if detector is not None and not isinstance(detector, DriftDetector):
            raise AdaptiveAIError("detector must be a DriftDetector instance or None.")
        if detector is not None and (
            alpha is not None or min_production_rows is not None
        ):
            raise AdaptiveAIError(
                "Pass either a configured detector or alpha/min_production_rows, not both."
            )
        if isinstance(production_limit, bool) or not isinstance(
            production_limit, int
        ):
            raise AdaptiveAIError("production_limit must be a positive integer.")
        if production_limit < 1:
            raise AdaptiveAIError("production_limit must be a positive integer.")
        if monitoring_repository is not None and not isinstance(
            monitoring_repository, MonitoringRepository
        ):
            raise AdaptiveAIError(
                "monitoring_repository must be a MonitoringRepository instance or None."
            )
        if categorical_detector is not None and not isinstance(
            categorical_detector, CategoricalDriftDetector
        ):
            raise AdaptiveAIError(
                "categorical_detector must be a CategoricalDriftDetector instance or None."
            )

        if detector is None:
            detector = DriftDetector(
                alpha=DEFAULT_ALPHA if alpha is None else alpha,
                min_production_rows=DEFAULT_MIN_PRODUCTION_ROWS
                if min_production_rows is None
                else min_production_rows,
            )
        if categorical_detector is None:
            # Inherit the numeric detector's configuration so both
            # families test at the same alpha/sample floor. Mock
            # detectors in unit tests expose non-numeric attributes;
            # fall back to defaults rather than failing construction.
            alpha_value = getattr(detector, "alpha", None)
            if isinstance(alpha_value, bool) or not isinstance(
                alpha_value, (int, float)
            ):
                alpha_value = DEFAULT_ALPHA
            min_rows_value = getattr(detector, "min_production_rows", None)
            if (
                isinstance(min_rows_value, bool)
                or not isinstance(min_rows_value, int)
                or min_rows_value < 1
            ):
                min_rows_value = DEFAULT_MIN_PRODUCTION_ROWS
            categorical_detector = CategoricalDriftDetector(
                alpha=float(alpha_value),
                min_production_rows=int(min_rows_value),
            )
        self._provider = provider
        self._detector = detector
        self._categorical_detector = categorical_detector
        self._production_limit = int(production_limit)
        self._monitoring_repository = monitoring_repository

    @property
    def provider(self) -> MonitoringDataProvider:
        """Returns the injected provider."""
        return self._provider

    @property
    def detector(self) -> DriftDetector:
        """Returns the injected/built detector."""
        return self._detector

    @property
    def categorical_detector(self) -> CategoricalDriftDetector:
        """Returns the injected/built categorical detector."""
        return self._categorical_detector

    @property
    def production_limit(self) -> int:
        """Returns the default production history limit."""
        return self._production_limit

    @property
    def monitoring_repository(self) -> Optional[MonitoringRepository]:
        """Returns the injected monitoring repository, if any."""
        return self._monitoring_repository

    def run_check(self, production_limit: Optional[int] = None) -> Dict[str, Any]:
        """Runs one monitoring check, persists it, and returns the result.

        Provider and detector errors propagate unchanged and never
        create a monitoring record. Persistence failures also propagate
        as AdaptiveAI errors without altering the detector result.

        The returned drift result is the detector result unchanged.
        """
        limit = self._production_limit if production_limit is None else production_limit
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise AdaptiveAIError("production_limit must be a positive integer.")
        data = self._provider.get_monitoring_data(limit=limit)
        result = self._detector.detect(
            reference=data["reference_features"],
            production=data["production_features"],
        )
        if self._monitoring_repository is not None:
            try:
                self._monitoring_repository.init()
                self._monitoring_repository.save_monitoring_result(
                    use_case=self._provider.use_case_name,
                    result=result,
                )
            except AdaptiveAIError:
                raise
            except Exception as e:
                logger.error(f"Failed to persist monitoring result: {e}")
                raise AdaptiveAIError(f"Failed to persist monitoring result: {e}") from e
        logger.info(
            "Monitoring check completed: status=%s overall_drifted=%s",
            result.get("status"),
            result.get("overall_drifted"),
        )
        return result


    def run_health_check(
        self,
        production_limit: Optional[int] = None,
        serving_info: Optional[Dict[str, Any]] = None,
        total_predictions: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Runs a full Phase 4 health check, persists it, returns the summary.

        Combines (all on the same monitoring window):
        - numeric drift (existing KS detector, unchanged),
        - categorical drift (chi-square detector),
        - data-quality checks over the raw window rows,
        - deterministic severity + overall health.

        The legacy ``run_check`` (detector-only, trigger path) is
        untouched; this is the additive health path.

        Args:
            production_limit: Window size (defaults to the configured
                production_limit). Must be a positive integer.
            serving_info: Optional mapping with the actually-serving
                ``model_name``/``model_version`` (resolved by the
                caller, e.g. from the production pointer). Never
                inferred here.
            total_predictions: Optional whole-table prediction count
                for the summary (caller-supplied to keep this layer
                free of counting queries).

        Returns:
            JSON-safe summary with timestamp, use-case/model/schema
            metadata, counts, numeric + categorical drift blocks (each
            feature annotated with type/effect size/severity),
            data-quality block, drift severity, overall health, and
            status ("ok" or "insufficient_data").
        """
        from datetime import datetime, timezone

        limit = self._production_limit if production_limit is None else production_limit
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise AdaptiveAIError("production_limit must be a positive integer.")

        data = self._provider.get_full_monitoring_data(limit=limit)
        reference_full = data["reference_full"]
        production_full = data["production_full"]
        raw_rows = data["raw_rows"]

        numeric_cols = list(self._provider.monitored_feature_names)
        categorical_cols = [c for c in list(reference_full.columns) if c not in numeric_cols]

        numeric_result = self._detector.detect(
            reference=reference_full[numeric_cols],
            production=production_full[numeric_cols],
        )
        categorical_result = self._categorical_detector.detect(
            reference=reference_full[categorical_cols],
            production=production_full[categorical_cols],
            features=categorical_cols,
        )
        quality = run_house_price_data_quality(raw_rows)

        sufficient = bool(numeric_result.get("sufficient_data", False))
        annotated_numeric = []
        for entry in numeric_result.get("features", []):
            annotated_numeric.append(
                _annotate_feature(
                    dict(entry),
                    kind="numeric",
                    large_threshold=KS_LARGE_EFFECT,
                    sufficient=sufficient,
                )
            )
        annotated_categorical = []
        for entry in categorical_result.get("features", []):
            testable = entry.get("status", "ok") == "ok"
            annotated_categorical.append(
                _annotate_feature(
                    dict(entry),
                    kind="categorical",
                    large_threshold=PSI_LARGE_EFFECT,
                    sufficient=sufficient and testable,
                )
            )

        drift = classify_drift_severity(
            numeric_features=annotated_numeric,
            categorical_features=annotated_categorical,
            sufficient_data=sufficient,
        )
        health = classify_overall_health(drift["severity"], quality.get("status"))
        overall_drifted = bool(
            numeric_result.get("overall_drifted", False)
            or categorical_result.get("overall_drifted", False)
        )

        use_case_name = self._provider.use_case_name
        serving = serving_info if isinstance(serving_info, dict) else {}
        model_name = serving.get("model_name")
        model_name = str(model_name).strip() if isinstance(model_name, str) and model_name.strip() else None
        model_version = serving.get("model_version")
        model_version = (
            str(model_version).strip()
            if isinstance(model_version, str) and model_version.strip()
            else None
        )
        if isinstance(total_predictions, bool) or (
            total_predictions is not None and not isinstance(total_predictions, int)
        ):
            raise AdaptiveAIError("total_predictions must be an integer or None.")
        if total_predictions is not None and total_predictions < 0:
            raise AdaptiveAIError("total_predictions must be non-negative.")

        summary: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "use_case": use_case_name,
            "model_name": model_name,
            "model_version": model_version,
            "schema_version": self._provider.schema_version,
            "total_predictions": total_predictions,
            "window_limit": int(limit),
            "window_records": int(data["total_records"]),
            "window_v2_records": int(data["v2_records"]),
            "window_legacy_records": int(data["legacy_records"]),
            "window_malformed_records": int(data["malformed_records"]),
            "status": "ok" if sufficient else "insufficient_data",
            "overall_drifted": overall_drifted,
            "overall_health": health["overall_health"],
            "health_reason": health["reason"],
            "severity": drift["severity"],
            "severity_reason": drift["reason"],
            "drifted_features": list(drift["drifted_features"]),
            "drifted_count": int(drift["drifted_count"]),
            "total_monitored": int(len(numeric_cols) + len(categorical_cols)),
            "alpha": float(self._detector.alpha),
            "min_production_rows": int(self._detector.min_production_rows),
            "production_rows": int(data["v2_records"]),
            "sufficient_data": sufficient,
            "reference_rows": int(len(reference_full)),
            "numeric_drift": {
                "overall_drifted": bool(numeric_result.get("overall_drifted", False)),
                "method": "ks",
                "correction": numeric_result.get("correction", "holm-bonferroni"),
                "features": annotated_numeric,
            },
            "categorical_drift": {
                "overall_drifted": bool(categorical_result.get("overall_drifted", False)),
                "method": categorical_result.get("method", "chi-square"),
                "correction": categorical_result.get("correction", "holm-bonferroni"),
                "features": annotated_categorical,
            },
            "features": annotated_numeric + annotated_categorical,
            "data_quality": quality,
            "data_quality_status": str(quality.get("status")),
        }

        if self._monitoring_repository is not None:
            try:
                self._monitoring_repository.init()
                record_id = self._monitoring_repository.save_health_summary(
                    use_case=use_case_name,
                    summary={
                        "overall_drifted": summary["overall_drifted"],
                        "status": summary["status"],
                        "alpha": summary["alpha"],
                        "min_production_rows": summary["min_production_rows"],
                        "production_rows": summary["production_rows"],
                        "sufficient_data": summary["sufficient_data"],
                        "severity": summary["severity"],
                        "overall_health": summary["overall_health"],
                        "data_quality_status": summary["data_quality_status"],
                        "drifted_count": summary["drifted_count"],
                        "total_monitored": summary["total_monitored"],
                        "model_name": summary["model_name"],
                        "model_version": summary["model_version"],
                        "schema_version": summary["schema_version"],
                        "features": summary["features"],
                    },
                    timestamp=summary["timestamp"],
                )
                summary["monitoring_run_id"] = int(record_id)
            except AdaptiveAIError:
                raise
            except Exception as e:
                logger.error(f"Failed to persist health summary: {e}")
                raise AdaptiveAIError(f"Failed to persist health summary: {e}") from e
        logger.info(
            "Health check completed: status=%s overall_health=%s drifted=%d/%d dq=%s",
            summary["status"],
            summary["overall_health"],
            summary["drifted_count"],
            summary["total_monitored"],
            summary["data_quality_status"],
        )
        return summary


def _annotate_feature(
    entry: Dict[str, Any],
    kind: str,
    large_threshold: float,
    sufficient: bool,
) -> Dict[str, Any]:
    """Adds type/effect-size/severity annotations to a drift entry."""
    entry["type"] = kind
    if kind == "categorical":
        effect = entry.get("effect_size")
    else:
        effect = entry.get("statistic")
    try:
        effect_value = float(effect)
        import math

        if not math.isfinite(effect_value) or effect_value < 0:
            effect_value = None
    except (TypeError, ValueError):
        effect_value = None
    entry["effect_size"] = effect_value
    if not sufficient:
        entry["severity"] = SEVERITY_UNKNOWN
    elif entry.get("status", "ok") != "ok":
        entry["severity"] = SEVERITY_UNKNOWN
    elif entry.get("drifted") is True:
        entry["severity"] = (
            SEVERITY_CRITICAL
            if effect_value is not None and effect_value >= large_threshold
            else SEVERITY_WARNING
        )
    else:
        entry["severity"] = SEVERITY_HEALTHY
    return entry


def create_house_price_monitoring_service(
    repository: PredictionRepository,
    reference_csv_path: Optional[Union[str, Path]] = None,
    alpha: float = DEFAULT_ALPHA,
    min_production_rows: int = DEFAULT_MIN_PRODUCTION_ROWS,
    production_limit: int = DEFAULT_PRODUCTION_LIMIT,
    monitoring_repository: Optional[MonitoringRepository] = None,
    source: Optional[str] = None,
    demo_session_id: Optional[str] = None,
) -> MonitoringService:
    """Small factory wiring provider + detector for House Price.

    Phase 6: ``source``/``demo_session_id`` select the observed
    traffic class (None = all rows, legacy default). Production
    callers pass ``source="production"``; the demo simulator passes
    ``source="demo"`` with its session id and no persistence
    repository (demo results live on the session record, never in
    ``monitoring_results``).
    """
    from app.monitoring.data_provider import create_house_price_provider

    provider = create_house_price_provider(
        repository=repository,
        reference_csv_path=reference_csv_path,
        source=source,
        demo_session_id=demo_session_id,
    )
    detector = DriftDetector(alpha=alpha, min_production_rows=min_production_rows)
    return MonitoringService(
        provider=provider,
        detector=detector,
        production_limit=production_limit,
        monitoring_repository=monitoring_repository,
    )
