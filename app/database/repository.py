"""Prediction-history repository for AdaptiveAI (Step 23).

Generic data-access layer over the ``predictions`` table.
Stores ``input_data`` as JSON text so it works across future use cases.
Timestamps are UTC ISO-8601 strings (timezone-aware).
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from app.core.exceptions import AdaptiveAIError
from app.database.database import DatabaseManager

logger = logging.getLogger(__name__)


def utc_now_iso() -> str:
    """Returns the current UTC time as a timezone-aware ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _serialize_input_data(input_data: Dict[str, Any]) -> str:
    """Serializes input_data dict to JSON text consistently."""
    if not isinstance(input_data, dict):
        raise AdaptiveAIError("input_data must be a dict.")
    try:
        return json.dumps(input_data, sort_keys=True, default=str)
    except (TypeError, ValueError) as e:
        raise AdaptiveAIError(f"input_data is not JSON serializable: {e}") from e


def _deserialize_input_data(raw: str) -> Dict[str, Any]:
    """Deserializes JSON text back to a dict."""
    try:
        value = json.loads(raw)
    except (TypeError, ValueError) as e:
        raise AdaptiveAIError(f"Stored input_data is not valid JSON: {e}") from e
    if not isinstance(value, dict):
        raise AdaptiveAIError("Stored input_data must decode to a dict.")
    return value


def _normalize_timestamp(timestamp: Optional[Union[str, datetime]]) -> str:
    """Normalizes timestamps to UTC ISO-8601 strings."""
    if timestamp is None:
        return utc_now_iso()
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc).isoformat()
    if isinstance(timestamp, str):
        if not timestamp.strip():
            raise AdaptiveAIError("timestamp must be a non-empty string.")
        return timestamp
    raise AdaptiveAIError("timestamp must be a datetime, ISO-8601 string, or None.")


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Converts a predictions row to a dict with deserialized input_data.

    Phase 6 demo-isolation columns (``source`` / ``demo_session_id``)
    are read defensively so rows selected without the additive columns
    still decode: a NULL/missing source is interpreted as production
    traffic ("existing production rows are production").
    """
    return {
        "id": int(row["id"]),
        "timestamp": str(row["timestamp"]),
        "use_case": str(row["use_case"]),
        "model_name": str(row["model_name"]),
        "model_version": row["model_version"] if row["model_version"] is None else str(row["model_version"]),
        "input_data": _deserialize_input_data(str(row["input_data"])),
        "prediction": float(row["prediction"]),
        "source": _opt_column(row, "source") or "production",
        "demo_session_id": _opt_column(row, "demo_session_id"),
    }


def _opt_column(row: sqlite3.Row, column: str) -> Optional[str]:
    """Reads an optional (possibly pre-migration) column, None when absent."""
    try:
        value = row[column]
    except (KeyError, IndexError):
        return None
    return None if value is None else str(value)


# Traffic-source labels for Phase 6 demo/production isolation.
SOURCE_PRODUCTION = "production"
SOURCE_DEMO = "demo"

_PREDICTION_COLUMNS = (
    "id, timestamp, use_case, model_name, model_version, "
    "input_data, prediction, source, demo_session_id"
)


def _validate_source_filter(
    source: Optional[str], demo_session_id: Optional[str]
) -> None:
    """Validates traffic-source filter arguments (read paths).

    ``source`` None means "all traffic" (legacy default, unchanged
    behavior). ``"production"`` means real serving traffic only
    (legacy NULL rows included); ``"demo"`` means simulated traffic
    only, optionally narrowed to one session. A session filter without
    ``source="demo"`` is rejected so production reads can never
    accidentally narrow to (or leak) demo rows.
    """
    if source is None:
        if demo_session_id is not None:
            raise AdaptiveAIError(
                "demo_session_id requires source='demo'."
            )
        return
    if source not in (SOURCE_PRODUCTION, SOURCE_DEMO):
        raise AdaptiveAIError("source must be 'production', 'demo', or None.")
    if source == SOURCE_PRODUCTION and demo_session_id is not None:
        raise AdaptiveAIError(
            "demo_session_id requires source='demo'."
        )
    if demo_session_id is not None and (
        not isinstance(demo_session_id, str) or not demo_session_id.strip()
    ):
        raise AdaptiveAIError("demo_session_id must be a non-empty string.")


def _source_condition(
    source: Optional[str], demo_session_id: Optional[str]
) -> tuple:
    """Builds the (SQL fragment, params) traffic-source filter."""
    if source is None:
        return "", []
    if source == SOURCE_PRODUCTION:
        return "AND (source IS NULL OR source = ?)", [SOURCE_PRODUCTION]
    if demo_session_id is not None:
        return "AND source = ? AND demo_session_id = ?", [SOURCE_DEMO, demo_session_id.strip()]
    return "AND source = ?", [SOURCE_DEMO]


class PredictionRepository:
    """Focused CRUD operations for the predictions table."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initializes the repository with a DatabaseManager."""
        if not isinstance(db_manager, DatabaseManager):
            raise AdaptiveAIError("db_manager must be a DatabaseManager instance.")
        self._db = db_manager

    def init(self) -> None:
        """Creates required tables (idempotent)."""
        self._db.init_db()

    def save_prediction(
        self,
        use_case: str,
        model_name: str,
        input_data: Dict[str, Any],
        prediction: Union[int, float],
        model_version: Optional[str] = None,
        timestamp: Optional[Union[str, datetime]] = None,
        source: Optional[str] = None,
        demo_session_id: Optional[str] = None,
    ) -> int:
        """Saves a prediction record and returns its id.

        Phase 6 traffic isolation: ``source`` is None (legacy callers,
        stored NULL and interpreted as production), ``"production"``
        (real serving traffic), or ``"demo"`` (simulated traffic, which
        requires a non-empty ``demo_session_id``). Production callers
        must pass ``source="production"`` explicitly so demo rows stay
        out of every production read.
        """
        if not use_case or not str(use_case).strip():
            raise AdaptiveAIError("use_case must be a non-empty string.")
        if not model_name or not str(model_name).strip():
            raise AdaptiveAIError("model_name must be a non-empty string.")
        if isinstance(prediction, bool) or not isinstance(prediction, (int, float)):
            raise AdaptiveAIError("prediction must be a numeric value.")
        if model_version is not None and not isinstance(model_version, str):
            raise AdaptiveAIError("model_version must be a string or None.")
        if source is not None and source not in (SOURCE_PRODUCTION, SOURCE_DEMO):
            raise AdaptiveAIError("source must be 'production', 'demo', or None.")
        if demo_session_id is not None and (
            not isinstance(demo_session_id, str) or not demo_session_id.strip()
        ):
            raise AdaptiveAIError("demo_session_id must be a non-empty string or None.")
        if source == SOURCE_DEMO and not demo_session_id:
            raise AdaptiveAIError("source='demo' requires a demo_session_id.")
        if source != SOURCE_DEMO and demo_session_id is not None:
            raise AdaptiveAIError("demo_session_id requires source='demo'.")

        payload = _serialize_input_data(input_data)
        ts = _normalize_timestamp(timestamp)

        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            cursor = conn.execute(
                "INSERT INTO predictions "
                "(timestamp, use_case, model_name, model_version, input_data, prediction, "
                "source, demo_session_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ts,
                    str(use_case),
                    str(model_name),
                    model_version,
                    payload,
                    float(prediction),
                    source,
                    demo_session_id.strip() if isinstance(demo_session_id, str) else None,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to save prediction: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)

    def count_predictions(
        self, use_case: Optional[str] = None,
        source: Optional[str] = None,
        demo_session_id: Optional[str] = None,
    ) -> int:
        """Counts prediction rows (optionally filtered by use case).

        Phase 6: ``source="production"`` counts real traffic only
        (legacy NULL rows included, demo rows excluded);
        ``source="demo"`` counts simulated traffic only (optionally
        one session). ``source=None`` keeps the legacy count-everything
        behavior for backward compatibility.

        Lightweight ``SELECT COUNT(*)`` read; used by monitoring
        summaries without loading full history.
        """
        _validate_source_filter(source, demo_session_id)
        fragment, params = _source_condition(source, demo_session_id)
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            if use_case is None:
                row = conn.execute(
                    f"SELECT COUNT(*) AS n FROM predictions WHERE 1 = 1 {fragment}",
                    params,
                ).fetchone()
            else:
                if not str(use_case).strip():
                    raise AdaptiveAIError("use_case must be a non-empty string when provided.")
                row = conn.execute(
                    f"SELECT COUNT(*) AS n FROM predictions WHERE use_case = ? {fragment}",
                    [str(use_case), *params],
                ).fetchone()
            return int(row["n"]) if row is not None else 0
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to count predictions: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)

    def list_raw_inputs(
        self,
        limit: int = 100,
        offset: int = 0,
        use_case: Optional[str] = None,
        source: Optional[str] = None,
        demo_session_id: Optional[str] = None,
    ) -> List[str]:
        """Lists raw stored ``input_data`` payloads newest-first.

        Returns the JSON text exactly as stored WITHOUT deserializing,
        so monitoring and data-quality layers can count malformed rows
        instead of failing the whole read (``list_predictions`` keeps
        its strict fail-closed contract for serving paths).

        Phase 6: ``source``/``demo_session_id`` narrow the window to
        one traffic class (see ``count_predictions``).
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise AdaptiveAIError("limit must be a positive integer.")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise AdaptiveAIError("offset must be a non-negative integer.")
        _validate_source_filter(source, demo_session_id)
        fragment, params = _source_condition(source, demo_session_id)

        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            if use_case is None:
                rows = conn.execute(
                    f"SELECT input_data FROM predictions WHERE 1 = 1 {fragment} "
                    "ORDER BY id DESC LIMIT ? OFFSET ?",
                    (*params, limit, offset),
                ).fetchall()
            else:
                if not str(use_case).strip():
                    raise AdaptiveAIError("use_case must be a non-empty string when provided.")
                rows = conn.execute(
                    f"SELECT input_data FROM predictions WHERE use_case = ? {fragment} "
                    "ORDER BY id DESC LIMIT ? OFFSET ?",
                    (str(use_case), *params, limit, offset),
                ).fetchall()
            return [str(r["input_data"]) for r in rows]
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to list prediction inputs: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)

    def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a prediction by id, or None if not found."""
        if isinstance(prediction_id, bool) or not isinstance(prediction_id, int):
            raise AdaptiveAIError("prediction_id must be an integer.")
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            row = conn.execute(
                f"SELECT {_PREDICTION_COLUMNS} FROM predictions WHERE id = ?",
                (prediction_id,),
            ).fetchone()
            if row is None:
                return None
            return _row_to_dict(row)
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to retrieve prediction {prediction_id}: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)

    def list_predictions(
        self,
        limit: int = 100,
        offset: int = 0,
        use_case: Optional[str] = None,
        source: Optional[str] = None,
        demo_session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Lists predictions newest-first (ordered by id descending).

        Step 63 (I-14): history consumers expect latest-first so the
        newest record is items[0] and limit/offset paginate from the
        latest backwards. IDs are monotonic, so id DESC is a
        deterministic newest-first ordering.

        Phase 6: ``source``/``demo_session_id`` narrow the window to
        one traffic class (see ``count_predictions``).
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise AdaptiveAIError("limit must be a positive integer.")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise AdaptiveAIError("offset must be a non-negative integer.")
        _validate_source_filter(source, demo_session_id)
        fragment, params = _source_condition(source, demo_session_id)

        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            if use_case is None:
                rows = conn.execute(
                    f"SELECT {_PREDICTION_COLUMNS} FROM predictions "
                    f"WHERE 1 = 1 {fragment} ORDER BY id DESC LIMIT ? OFFSET ?",
                    (*params, limit, offset),
                ).fetchall()
            else:
                if not str(use_case).strip():
                    raise AdaptiveAIError("use_case must be a non-empty string when provided.")
                rows = conn.execute(
                    f"SELECT {_PREDICTION_COLUMNS} FROM predictions "
                    f"WHERE use_case = ? {fragment} ORDER BY id DESC LIMIT ? OFFSET ?",
                    (str(use_case), *params, limit, offset),
                ).fetchall()
            return [_row_to_dict(r) for r in rows]
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to list predictions: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)

    def delete_demo_session_predictions(self, demo_session_id: str) -> int:
        """Deletes ONLY simulated rows of one demo session (reset path).

        Scoped by both ``source='demo'`` and the session id, so
        production rows (including legacy NULL-source rows) can never
        match. Returns the removed row count. Never touches reference
        data, models, pointers, or audit history.
        """
        if not isinstance(demo_session_id, str) or not demo_session_id.strip():
            raise AdaptiveAIError("demo_session_id must be a non-empty string.")
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            cursor = conn.execute(
                "DELETE FROM predictions WHERE source = ? AND demo_session_id = ?",
                (SOURCE_DEMO, demo_session_id.strip()),
            )
            conn.commit()
            return int(cursor.rowcount)
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to reset demo session predictions: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)
