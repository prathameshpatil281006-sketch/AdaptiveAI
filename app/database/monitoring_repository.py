"""Monitoring-result repository for AdaptiveAI (Step 28).

Dedicated data-access layer over the ``monitoring_results`` table,
kept separate from prediction history. Stores the detector's
``features`` list as deterministic JSON text.
Timestamps follow the same UTC ISO-8601 convention as PredictionRepository.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from app.core.exceptions import AdaptiveAIError
from app.database.database import DatabaseManager

logger = logging.getLogger(__name__)


def _normalize_timestamp(timestamp: Optional[Union[str, datetime]]) -> str:
    """Normalizes timestamps to UTC ISO-8601 strings."""
    if timestamp is None:
        return datetime.now(timezone.utc).isoformat()
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc).isoformat()
    if isinstance(timestamp, str):
        if not timestamp.strip():
            raise AdaptiveAIError("timestamp must be a non-empty string.")
        return timestamp
    raise AdaptiveAIError("timestamp must be a datetime, ISO-8601 string, or None.")


def _validate_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validates a drift result dict and returns normalized scalar fields."""
    if not isinstance(result, dict):
        raise AdaptiveAIError("Monitoring result must be a dict.")

    for field in (
        "overall_drifted",
        "alpha",
        "min_production_rows",
        "production_rows",
        "sufficient_data",
        "status",
        "features",
    ):
        if field not in result:
            raise AdaptiveAIError(f"Monitoring result is missing required field: '{field}'.")

    overall = result["overall_drifted"]
    sufficient = result["sufficient_data"]
    if not isinstance(overall, bool) or not isinstance(sufficient, bool):
        raise AdaptiveAIError("overall_drifted and sufficient_data must be booleans.")

    alpha = result["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise AdaptiveAIError("alpha must be a number in (0, 1).")
    if not 0.0 < float(alpha) < 1.0:
        raise AdaptiveAIError("alpha must be in (0, 1).")

    min_rows = result["min_production_rows"]
    prod_rows = result["production_rows"]
    if isinstance(min_rows, bool) or not isinstance(min_rows, int) or min_rows < 1:
        raise AdaptiveAIError("min_production_rows must be a positive integer.")
    if isinstance(prod_rows, bool) or not isinstance(prod_rows, int) or prod_rows < 0:
        raise AdaptiveAIError("production_rows must be a non-negative integer.")

    status = result["status"]
    if not isinstance(status, str) or not status.strip():
        raise AdaptiveAIError("status must be a non-empty string.")

    features = result["features"]
    if not isinstance(features, list):
        raise AdaptiveAIError("features must be a list.")
    for item in features:
        if not isinstance(item, dict):
            raise AdaptiveAIError("Each feature result must be a dict.")
        for field in ("feature_name", "statistic", "p_value", "drifted"):
            if field not in item:
                raise AdaptiveAIError(
                    f"Feature result is missing required field: '{field}'."
                )
        if not isinstance(item["feature_name"], str) or not item["feature_name"].strip():
            raise AdaptiveAIError("feature_name must be a non-empty string.")
        for numeric_field in ("statistic", "p_value"):
            value = item[numeric_field]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AdaptiveAIError(f"{numeric_field} must be a number.")
        if not isinstance(item["drifted"], bool):
            raise AdaptiveAIError("drifted must be a boolean.")

    try:
        payload = json.dumps(features, sort_keys=True)
        json.loads(payload)
    except (TypeError, ValueError) as e:
        raise AdaptiveAIError(f"Feature results are not JSON serializable: {e}") from e

    return {
        "overall_drifted": overall,
        "alpha": float(alpha),
        "min_production_rows": int(min_rows),
        "production_rows": int(prod_rows),
        "sufficient_data": sufficient,
        "status": str(status),
        "feature_payload": payload,
    }


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Converts a monitoring_results row to a dict with decoded features.

    Phase 4 additive columns are read defensively (``row.keys()``
    membership) so rows written before the migration — or databases
    never migrated — still decode with ``None`` enrichment instead of
    raising.
    """
    try:
        features = json.loads(str(row["feature_results"]))
    except (TypeError, ValueError) as e:
        raise AdaptiveAIError(f"Stored feature_results is not valid JSON: {e}") from e
    if not isinstance(features, list):
        raise AdaptiveAIError("Stored feature_results must decode to a list.")
    columns = set(row.keys())
    record: Dict[str, Any] = {
        "id": int(row["id"]),
        "timestamp": str(row["timestamp"]),
        "use_case": str(row["use_case"]),
        "overall_drifted": bool(row["overall_drifted"]),
        "status": str(row["status"]),
        "alpha": float(row["alpha"]),
        "min_production_rows": int(row["min_production_rows"]),
        "production_rows": int(row["production_rows"]),
        "sufficient_data": bool(row["sufficient_data"]),
        "features": features,
    }
    record["model_name"] = (
        str(row["model_name"]) if "model_name" in columns and row["model_name"] is not None else None
    )
    record["model_version"] = (
        str(row["model_version"]) if "model_version" in columns and row["model_version"] is not None else None
    )
    record["schema_version"] = (
        int(row["schema_version"]) if "schema_version" in columns and row["schema_version"] is not None else None
    )
    record["severity"] = (
        str(row["severity"]) if "severity" in columns and row["severity"] is not None else None
    )
    record["overall_health"] = (
        str(row["overall_health"])
        if "overall_health" in columns and row["overall_health"] is not None
        else None
    )
    record["data_quality_status"] = (
        str(row["data_quality_status"])
        if "data_quality_status" in columns and row["data_quality_status"] is not None
        else None
    )
    record["drifted_features"] = (
        int(row["drifted_features"])
        if "drifted_features" in columns and row["drifted_features"] is not None
        else None
    )
    return record


def _validate_health_summary(summary: Dict[str, Any]) -> Dict[str, Any]:
    """Validates a Phase 4 health summary and normalizes its columns."""
    if not isinstance(summary, dict):
        raise AdaptiveAIError("Health summary must be a dict.")
    for field in (
        "overall_drifted",
        "status",
        "alpha",
        "min_production_rows",
        "production_rows",
        "sufficient_data",
        "severity",
        "overall_health",
        "data_quality_status",
        "drifted_count",
        "total_monitored",
    ):
        if field not in summary:
            raise AdaptiveAIError(f"Health summary is missing required field: '{field}'.")

    def _strict_bool(value: Any, field: str) -> bool:
        if not isinstance(value, bool):
            raise AdaptiveAIError(f"Health summary field '{field}' must be a boolean.")
        return value

    def _optional_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def _optional_int(value: Any, field: str) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool) or not isinstance(value, int):
            raise AdaptiveAIError(f"Health summary field '{field}' must be an integer or null.")
        return int(value)

    alpha = summary["alpha"]
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise AdaptiveAIError("Health summary field 'alpha' must be a number in (0, 1).")
    if not 0.0 < float(alpha) < 1.0:
        raise AdaptiveAIError("Health summary field 'alpha' must be in (0, 1).")
    min_rows = summary["min_production_rows"]
    prod_rows = summary["production_rows"]
    if isinstance(min_rows, bool) or not isinstance(min_rows, int) or min_rows < 1:
        raise AdaptiveAIError("Health summary field 'min_production_rows' must be a positive integer.")
    if isinstance(prod_rows, bool) or not isinstance(prod_rows, int) or prod_rows < 0:
        raise AdaptiveAIError("Health summary field 'production_rows' must be a non-negative integer.")

    features = summary.get("features", [])
    if not isinstance(features, list):
        raise AdaptiveAIError("Health summary field 'features' must be a list.")
    try:
        payload = json.dumps(features, sort_keys=True, default=str)
        json.loads(payload)
    except (TypeError, ValueError) as e:
        raise AdaptiveAIError(f"Health summary features are not JSON serializable: {e}") from e

    status = summary["status"]
    severity = summary["severity"]
    overall_health = summary["overall_health"]
    dq_status = summary["data_quality_status"]
    for label, value in (
        ("status", status),
        ("severity", severity),
        ("overall_health", overall_health),
        ("data_quality_status", dq_status),
    ):
        if not isinstance(value, str) or not value.strip():
            raise AdaptiveAIError(f"Health summary field '{label}' must be a non-empty string.")

    drifted_count = summary["drifted_count"]
    total_monitored = summary["total_monitored"]
    for label, value in (("drifted_count", drifted_count), ("total_monitored", total_monitored)):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise AdaptiveAIError(f"Health summary field '{label}' must be a non-negative integer.")

    return {
        "overall_drifted": _strict_bool(summary["overall_drifted"], "overall_drifted"),
        "status": str(status),
        "alpha": float(alpha),
        "min_production_rows": int(min_rows),
        "production_rows": int(prod_rows),
        "sufficient_data": _strict_bool(summary["sufficient_data"], "sufficient_data"),
        "feature_payload": payload,
        "model_name": _optional_str(summary.get("model_name")),
        "model_version": _optional_str(summary.get("model_version")),
        "schema_version": _optional_int(summary.get("schema_version"), "schema_version"),
        "severity": str(severity),
        "overall_health": str(overall_health),
        "data_quality_status": str(dq_status),
        "drifted_count": int(drifted_count),
        "total_monitored": int(total_monitored),
    }


class MonitoringRepository:
    """Focused CRUD operations for the monitoring_results table."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initializes the repository with a DatabaseManager."""
        if not isinstance(db_manager, DatabaseManager):
            raise AdaptiveAIError("db_manager must be a DatabaseManager instance.")
        self._db = db_manager

    def init(self) -> None:
        """Creates required tables (idempotent)."""
        self._db.init_db()

    def save_monitoring_result(
        self,
        use_case: str,
        result: Dict[str, Any],
        timestamp: Optional[Union[str, datetime]] = None,
    ) -> int:
        """Validates and saves a drift result, returning its id."""
        if not use_case or not str(use_case).strip():
            raise AdaptiveAIError("use_case must be a non-empty string.")
        fields = _validate_result(result)
        ts = _normalize_timestamp(timestamp)

        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            cursor = conn.execute(
                "INSERT INTO monitoring_results "
                "(timestamp, use_case, overall_drifted, status, alpha, "
                "min_production_rows, production_rows, sufficient_data, feature_results) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ts,
                    str(use_case),
                    int(fields["overall_drifted"]),
                    fields["status"],
                    fields["alpha"],
                    fields["min_production_rows"],
                    fields["production_rows"],
                    int(fields["sufficient_data"]),
                    fields["feature_payload"],
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to save monitoring result: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)

    def save_health_summary(
        self,
        use_case: str,
        summary: Dict[str, Any],
        timestamp: Optional[Union[str, datetime]] = None,
    ) -> int:
        """Persists a Phase 4 health summary, returning its id.

        Stores the combined numeric + categorical feature list as
        ``feature_results`` plus the additive enrichment columns
        (model, schema version, severity, overall health, data-quality
        status, drifted count). Plain detector results written by
        ``save_monitoring_result`` keep NULL enrichment. Never deletes
        or rewrites existing rows.
        """
        if not use_case or not str(use_case).strip():
            raise AdaptiveAIError("use_case must be a non-empty string.")
        fields = _validate_health_summary(summary)
        ts = _normalize_timestamp(timestamp)

        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            cursor = conn.execute(
                "INSERT INTO monitoring_results "
                "(timestamp, use_case, overall_drifted, status, alpha, "
                "min_production_rows, production_rows, sufficient_data, feature_results, "
                "model_name, model_version, schema_version, severity, overall_health, "
                "data_quality_status, drifted_features) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ts,
                    str(use_case),
                    int(fields["overall_drifted"]),
                    fields["status"],
                    fields["alpha"],
                    fields["min_production_rows"],
                    fields["production_rows"],
                    int(fields["sufficient_data"]),
                    fields["feature_payload"],
                    fields["model_name"],
                    fields["model_version"],
                    fields["schema_version"],
                    fields["severity"],
                    fields["overall_health"],
                    fields["data_quality_status"],
                    fields["drifted_count"],
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to save health summary: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)

    def get_monitoring_result_by_id(self, result_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a monitoring result by id, or None if not found."""
        if isinstance(result_id, bool) or not isinstance(result_id, int):
            raise AdaptiveAIError("result_id must be an integer.")
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            row = conn.execute(
                "SELECT * FROM monitoring_results WHERE id = ?",
                (result_id,),
            ).fetchone()
            if row is None:
                return None
            return _row_to_dict(row)
        except sqlite3.Error as e:
            raise AdaptiveAIError(
                f"Failed to retrieve monitoring result {result_id}: {e}"
            ) from e
        finally:
            DatabaseManager.close_connection(conn)

    def list_monitoring_results(
        self,
        limit: int = 100,
        offset: int = 0,
        use_case: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Lists monitoring results newest-first (ordered by id descending).

        Step 63 (I-14): history consumers expect latest-first so the
        newest record is items[0] and limit/offset paginate from the
        latest backwards. IDs are monotonic, so id DESC is a
        deterministic newest-first ordering.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise AdaptiveAIError("limit must be a positive integer.")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise AdaptiveAIError("offset must be a non-negative integer.")

        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            if use_case is None:
                rows = conn.execute(
                    "SELECT * FROM monitoring_results ORDER BY id DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            else:
                if not str(use_case).strip():
                    raise AdaptiveAIError(
                        "use_case must be a non-empty string when provided."
                    )
                rows = conn.execute(
                    "SELECT * FROM monitoring_results WHERE use_case = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                    (str(use_case), limit, offset),
                ).fetchall()
            return [_row_to_dict(r) for r in rows]
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to list monitoring results: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)
