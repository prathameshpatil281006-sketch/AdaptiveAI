"""Promotion-history repository for AdaptiveAI (Step 35).

Dedicated data-access layer over the ``promotion_history`` table.
Persists structured :class:`ModelPromoter` results (never performs
promotion itself). Timestamps follow the same UTC ISO-8601 convention
as the prediction and monitoring repositories.
"""

import logging
import math
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from app.core.exceptions import AdaptiveAIError
from app.database.database import DatabaseManager

logger = logging.getLogger(__name__)

# Decision vocabulary corresponds exactly to ModelPromoter output.
# No additional values are invented here.
SUPPORTED_DECISIONS = frozenset(
    {
        "promoted",
        "rejected",
        "requires_initial_promotion",
        "failed",
    }
)

_COLUMNS = (
    "id, timestamp, use_case, decision, model_name, run_id, "
    "registered_model_name, model_version, primary_metric, "
    "candidate_score, production_score, improvement, reason"
)


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


def _validate_score(value: Any, field: str) -> Optional[float]:
    """Validates an optional numeric metric (finite, non-bool) or None."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AdaptiveAIError(f"{field} must be a finite number or None.")
    as_float = float(value)
    if not math.isfinite(as_float):
        raise AdaptiveAIError(f"{field} must be a finite number or None.")
    return as_float


def _validate_optional_text(value: Any, field: str) -> Optional[str]:
    """Validates an optional text field (None or non-empty string)."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AdaptiveAIError(f"{field} must be a non-empty string or None.")
    return str(value)


def _validate_promotion_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Validates a promoter result dict without mutating it."""
    if not isinstance(result, dict):
        raise AdaptiveAIError("promotion_result must be a dict.")

    decision = result.get("decision")
    if decision not in SUPPORTED_DECISIONS:
        raise AdaptiveAIError(
            "decision must be one of: promoted, rejected, "
            "requires_initial_promotion, failed."
        )

    model_name = result.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise AdaptiveAIError("model_name must be a non-empty string.")

    reason = result.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise AdaptiveAIError("reason must be a non-empty string.")

    return {
        "decision": str(decision),
        "model_name": str(model_name),
        "run_id": _validate_optional_text(result.get("run_id"), "run_id"),
        "registered_model_name": _validate_optional_text(
            result.get("registered_model_name"), "registered_model_name"
        ),
        "model_version": _validate_optional_text(
            result.get("model_version"), "model_version"
        ),
        "primary_metric": _validate_optional_text(
            result.get("primary_metric"), "primary_metric"
        ),
        "candidate_score": _validate_score(
            result.get("candidate_score"), "candidate_score"
        ),
        "production_score": _validate_score(
            result.get("production_score"), "production_score"
        ),
        "improvement": _validate_score(result.get("improvement"), "improvement"),
        "reason": str(reason),
    }


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Converts a promotion_history row to a dict with Python types."""
    return {
        "id": int(row["id"]),
        "timestamp": str(row["timestamp"]),
        "use_case": str(row["use_case"]),
        "decision": str(row["decision"]),
        "model_name": str(row["model_name"]),
        "run_id": None if row["run_id"] is None else str(row["run_id"]),
        "registered_model_name": None
        if row["registered_model_name"] is None
        else str(row["registered_model_name"]),
        "model_version": None
        if row["model_version"] is None
        else str(row["model_version"]),
        "primary_metric": None
        if row["primary_metric"] is None
        else str(row["primary_metric"]),
        "candidate_score": None
        if row["candidate_score"] is None
        else float(row["candidate_score"]),
        "production_score": None
        if row["production_score"] is None
        else float(row["production_score"]),
        "improvement": None
        if row["improvement"] is None
        else float(row["improvement"]),
        "reason": str(row["reason"]),
    }


class PromotionRepository:
    """Focused CRUD operations for the promotion_history table.

    The repository only persists explicitly provided, structured promoter
    results. It never performs promotion, never infers failure
    information, and never fabricates run IDs, versions, or scores:
    absent nullable fields are stored as NULL.
    """

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initializes the repository with a DatabaseManager."""
        if not isinstance(db_manager, DatabaseManager):
            raise AdaptiveAIError("db_manager must be a DatabaseManager instance.")
        self._db = db_manager

    def init(self) -> None:
        """Creates required tables (idempotent)."""
        self._db.init_db()

    def save_promotion_result(
        self,
        use_case: str,
        promotion_result: Dict[str, Any],
        timestamp: Optional[Union[str, datetime]] = None,
    ) -> int:
        """Validates and saves a promotion result, returning its id."""
        if not isinstance(use_case, str) or not use_case.strip():
            raise AdaptiveAIError("use_case must be a non-empty string.")
        fields = _validate_promotion_result(promotion_result)
        ts = _normalize_timestamp(timestamp)

        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            cursor = conn.execute(
                "INSERT INTO promotion_history "
                "(timestamp, use_case, decision, model_name, run_id, "
                "registered_model_name, model_version, primary_metric, "
                "candidate_score, production_score, improvement, reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ts,
                    str(use_case),
                    fields["decision"],
                    fields["model_name"],
                    fields["run_id"],
                    fields["registered_model_name"],
                    fields["model_version"],
                    fields["primary_metric"],
                    fields["candidate_score"],
                    fields["production_score"],
                    fields["improvement"],
                    fields["reason"],
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to save promotion result: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)

    def get_promotion_result_by_id(self, result_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a promotion record by id, or None if not found."""
        if isinstance(result_id, bool) or not isinstance(result_id, int):
            raise AdaptiveAIError("result_id must be an integer.")
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM promotion_history WHERE id = ?",
                (result_id,),
            ).fetchone()
            if row is None:
                return None
            return _row_to_dict(row)
        except sqlite3.Error as e:
            raise AdaptiveAIError(
                f"Failed to retrieve promotion result {result_id}: {e}"
            ) from e
        finally:
            DatabaseManager.close_connection(conn)

    def list_promotion_history(
        self,
        limit: int = 100,
        offset: int = 0,
        use_case: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Lists promotion records newest-first (ordered by id descending).

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
                    f"SELECT {_COLUMNS} FROM promotion_history "
                    "ORDER BY id DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            else:
                if not isinstance(use_case, str) or not use_case.strip():
                    raise AdaptiveAIError(
                        "use_case must be a non-empty string when provided."
                    )
                rows = conn.execute(
                    f"SELECT {_COLUMNS} FROM promotion_history "
                    "WHERE use_case = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                    (str(use_case), limit, offset),
                ).fetchall()
            return [_row_to_dict(r) for r in rows]
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to list promotion history: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)
