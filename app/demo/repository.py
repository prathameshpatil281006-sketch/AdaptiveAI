"""Demo-session repository for AdaptiveAI (Phase 6).

Dedicated data-access layer (no business logic) over the additive
``demo_sessions`` table. One row per simulator session; generated
samples live in ``predictions`` (``source='demo'`` + session id) and
demo monitoring snapshots live on the session record itself — this
table is the ONLY new table, and no production history is ever
rewritten or deleted through it.

Timestamps follow the same UTC ISO-8601 convention as the other
repositories. JSON columns decode to Python objects on read.
"""

import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from app.core.exceptions import AdaptiveAIError
from app.database.database import DatabaseManager

logger = logging.getLogger(__name__)

SESSION_STATUSES = frozenset({"active", "completed", "reset"})

_SESSION_COLUMNS = (
    "id, session_id, use_case, scenario, status, sample_count, seed, "
    "params, samples_generated, created_at, started_at, completed_at, "
    "reset_at, last_monitoring, last_decision, last_monitored_at"
)

_JSON_COLUMNS = frozenset({"params", "last_monitoring", "last_decision"})


def _utc_now_iso() -> str:
    """Returns the current UTC time as a timezone-aware ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _normalize_timestamp(timestamp: Optional[Union[str, datetime]]) -> str:
    """Normalizes timestamps to UTC ISO-8601 strings."""
    if timestamp is None:
        return _utc_now_iso()
    if isinstance(timestamp, datetime):
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc).isoformat()
    if isinstance(timestamp, str):
        if not timestamp.strip():
            raise AdaptiveAIError("timestamp must be a non-empty string.")
        return timestamp
    raise AdaptiveAIError("timestamp must be a datetime, ISO-8601 string, or None.")


def _decode_json(raw: Any, field: str) -> Any:
    """Decodes a nullable JSON column (None stays None)."""
    if raw is None:
        return None
    try:
        return json.loads(str(raw))
    except (TypeError, ValueError) as e:
        raise AdaptiveAIError(f"Stored {field} is not valid JSON: {e}") from e


def _encode_json(value: Any, field: str) -> Optional[str]:
    """Encodes a JSON column (None stays None, never fabricates)."""
    if value is None:
        return None
    try:
        return json.dumps(value, sort_keys=True, default=str)
    except (TypeError, ValueError) as e:
        raise AdaptiveAIError(f"Field '{field}' is not JSON serializable: {e}") from e


def _session_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Converts a demo_sessions row to a dict with decoded JSON."""
    record: Dict[str, Any] = {
        "id": int(row["id"]),
        "session_id": str(row["session_id"]),
        "use_case": str(row["use_case"]),
        "scenario": str(row["scenario"]),
        "status": str(row["status"]),
        "sample_count": int(row["sample_count"]),
        "seed": int(row["seed"]),
        "samples_generated": int(row["samples_generated"]),
        "created_at": str(row["created_at"]),
        "started_at": None if row["started_at"] is None else str(row["started_at"]),
        "completed_at": None if row["completed_at"] is None else str(row["completed_at"]),
        "reset_at": None if row["reset_at"] is None else str(row["reset_at"]),
        "last_monitored_at": (
            None if row["last_monitored_at"] is None else str(row["last_monitored_at"])
        ),
    }
    for column in ("params", "last_monitoring", "last_decision"):
        record[column] = _decode_json(row[column], column)
    return record


class DemoSessionRepository:
    """CRUD operations for the demo_sessions ledger."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initializes the repository with a DatabaseManager."""
        if not isinstance(db_manager, DatabaseManager):
            raise AdaptiveAIError("db_manager must be a DatabaseManager instance.")
        self._db = db_manager

    def init(self) -> None:
        """Creates required tables (idempotent)."""
        self._db.init_db()

    def create_session(
        self,
        session_id: str,
        use_case: str,
        scenario: str,
        sample_count: int,
        seed: int,
        params: Optional[Any] = None,
        created_at: Optional[Union[str, datetime]] = None,
    ) -> Dict[str, Any]:
        """Inserts a session in ``active`` status and returns it."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise AdaptiveAIError("session_id must be a non-empty string.")
        if not isinstance(use_case, str) or not use_case.strip():
            raise AdaptiveAIError("use_case must be a non-empty string.")
        if not isinstance(scenario, str) or not scenario.strip():
            raise AdaptiveAIError("scenario must be a non-empty string.")
        if isinstance(sample_count, bool) or not isinstance(sample_count, int):
            raise AdaptiveAIError("sample_count must be an integer.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise AdaptiveAIError("seed must be an integer.")
        stamp = _normalize_timestamp(created_at)
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            cursor = conn.execute(
                "INSERT INTO demo_sessions "
                "(session_id, use_case, scenario, status, sample_count, seed, "
                "params, samples_generated, created_at, started_at) "
                "VALUES (?, ?, ?, 'active', ?, ?, ?, 0, ?, ?)",
                (
                    session_id.strip(),
                    str(use_case),
                    str(scenario),
                    int(sample_count),
                    int(seed),
                    _encode_json(params, "params"),
                    stamp,
                    stamp,
                ),
            )
            conn.commit()
            created = self.get_session_by_id(int(cursor.lastrowid))
            assert created is not None
            return created
        except sqlite3.IntegrityError as e:
            raise AdaptiveAIError(f"Demo session already exists: {e}") from e
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to create demo session: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)

    def update_session(self, session_id: str, **fields: Any) -> Dict[str, Any]:
        """Updates mutable session columns and returns the session.

        Updatable: status, samples_generated, started_at,
        completed_at, reset_at, last_monitoring, last_decision,
        last_monitored_at. Unknown columns raise; the session identity
        and creation facts are immutable.
        """
        def _status(value: Any) -> str:
            if value not in SESSION_STATUSES:
                raise AdaptiveAIError(
                    "status must be one of: " + ", ".join(sorted(SESSION_STATUSES)) + "."
                )
            return str(value)

        def _optional_ts(value: Any) -> Optional[str]:
            if value is None:
                return None
            return _normalize_timestamp(value)

        def _optional_int(value: Any, field: str) -> Optional[int]:
            if value is None:
                return None
            if isinstance(value, bool) or not isinstance(value, int):
                raise AdaptiveAIError(f"Field '{field}' must be an integer or None.")
            return int(value)

        allowed = {
            "status": _status,
            "samples_generated": lambda v: _optional_int(v, "samples_generated"),
            "started_at": _optional_ts,
            "completed_at": _optional_ts,
            "reset_at": _optional_ts,
            "last_monitored_at": _optional_ts,
            "last_monitoring": lambda v: _encode_json(v, "last_monitoring"),
            "last_decision": lambda v: _encode_json(v, "last_decision"),
        }
        unknown = sorted(set(fields) - set(allowed))
        if unknown:
            raise AdaptiveAIError(
                f"Session columns are not updatable: {', '.join(unknown)}."
            )
        if not isinstance(session_id, str) or not session_id.strip():
            raise AdaptiveAIError("session_id must be a non-empty string.")
        if not fields:
            existing = self.get_session(session_id)
            if existing is None:
                raise AdaptiveAIError(f"Demo session not found: {session_id}.")
            return existing
        assignments = ", ".join(f"{column} = ?" for column in fields)
        values = [allowed[column](fields[column]) for column in fields]
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            cursor = conn.execute(
                f"UPDATE demo_sessions SET {assignments} WHERE session_id = ?",
                (*values, session_id.strip()),
            )
            if cursor.rowcount == 0:
                raise AdaptiveAIError(f"Demo session not found: {session_id}.")
            conn.commit()
            updated = self.get_session(session_id)
            assert updated is not None
            return updated
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to update demo session: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a session by its session_id, or None."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise AdaptiveAIError("session_id must be a non-empty string.")
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            row = conn.execute(
                f"SELECT {_SESSION_COLUMNS} FROM demo_sessions WHERE session_id = ?",
                (session_id.strip(),),
            ).fetchone()
            if row is None:
                return None
            return _session_row_to_dict(row)
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to retrieve demo session {session_id}: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)

    def get_session_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a session by integer id, or None if not found."""
        if isinstance(record_id, bool) or not isinstance(record_id, int):
            raise AdaptiveAIError("record_id must be an integer.")
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            row = conn.execute(
                f"SELECT {_SESSION_COLUMNS} FROM demo_sessions WHERE id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                return None
            return _session_row_to_dict(row)
        except sqlite3.Error as e:
            raise AdaptiveAIError(
                f"Failed to retrieve demo session {record_id}: {e}"
            ) from e
        finally:
            DatabaseManager.close_connection(conn)

    def list_sessions(
        self,
        limit: int = 100,
        offset: int = 0,
        use_case: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Lists sessions newest-first (ordered by id descending)."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise AdaptiveAIError("limit must be a positive integer.")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise AdaptiveAIError("offset must be a non-negative integer.")
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            if use_case is None:
                rows = conn.execute(
                    f"SELECT {_SESSION_COLUMNS} FROM demo_sessions "
                    "ORDER BY id DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            else:
                if not isinstance(use_case, str) or not use_case.strip():
                    raise AdaptiveAIError("use_case must be a non-empty string when provided.")
                rows = conn.execute(
                    f"SELECT {_SESSION_COLUMNS} FROM demo_sessions "
                    "WHERE use_case = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                    (str(use_case), limit, offset),
                ).fetchall()
            return [_session_row_to_dict(r) for r in rows]
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to list demo sessions: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)

    def count_sessions(self, use_case: Optional[str] = None) -> int:
        """Counts session rows (optionally filtered by use case)."""
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            if use_case is None:
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM demo_sessions"
                ).fetchone()
            else:
                if not isinstance(use_case, str) or not use_case.strip():
                    raise AdaptiveAIError("use_case must be a non-empty string when provided.")
                row = conn.execute(
                    "SELECT COUNT(*) AS n FROM demo_sessions WHERE use_case = ?",
                    (str(use_case),),
                ).fetchone()
            return int(row["n"]) if row is not None else 0
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to count demo sessions: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)
