"""Phase 5 lifecycle repositories for AdaptiveAI.

Dedicated data-access layers (no business logic) over three additive
tables:

- ``retraining_runs``: one row per explicit retraining run, from
  ``started`` through ``completed``/``failed``. Pipelines are never
  stored here — only the staging artifact path plus metrics,
  comparison, gate, and promotion outcome.
- ``lifecycle_events``: append-only audit trail
  (``retraining_started`` / ``retraining_completed`` /
  ``retraining_failed`` / ``promotion_check`` / ``model_promoted`` /
  ``model_rollback``). Existing promotion/monitoring history is
  untouched and stays readable.
- ``production_pointer_history``: previous → new serving-reference
  snapshots for every lifecycle pointer move (promotion or
  rollback), enabling validated rollback. Legacy promotions that
  predate this ledger simply have no entry (documented, not
  backfilled with invented data).

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

RUN_STATUSES = frozenset({"started", "completed", "failed"})

LIFECYCLE_EVENTS = frozenset(
    {
        "retraining_started",
        "retraining_completed",
        "retraining_failed",
        "promotion_check",
        "model_promoted",
        "model_rollback",
        # Phase 6 demo-mode events (Scenario Simulator). Demo lifecycle
        # actions are recorded with these prefixed events so they are
        # always distinguishable from production lifecycle actions. A
        # demo flow that reaches an actual production promotion still
        # records the authoritative ``model_promoted`` event unchanged.
        "demo_session_created",
        "demo_scenario_generated",
        "demo_monitoring_run",
        "demo_retraining_started",
        "demo_session_reset",
    }
)

_RUN_COLUMNS = (
    "id, run_id, use_case, status, trigger, started_at, completed_at, "
    "dataset_ref, dataset_rows, schema_version, train_rows, test_rows, "
    "random_seed, feature_schema, candidates, production_model, "
    "production_version, best_model, best_metrics, comparison, "
    "candidate_artifact, candidate_run_id, candidate_version, "
    "promotion_eligible, promotion_decision, gate_checks, failure_reason"
)

_JSON_COLUMNS = frozenset(
    {"feature_schema", "candidates", "best_metrics", "comparison", "gate_checks"}
)


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


def _optional_str(value: Any, field: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise AdaptiveAIError(f"Field '{field}' must be a non-empty string or None.")
    return value


def _optional_int(value: Any, field: str) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise AdaptiveAIError(f"Field '{field}' must be an integer or None.")
    return int(value)


def _run_row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """Converts a retraining_runs row to a dict with decoded JSON."""
    record: Dict[str, Any] = {
        "id": int(row["id"]),
        "run_id": str(row["run_id"]),
        "use_case": str(row["use_case"]),
        "status": str(row["status"]),
        "trigger": str(row["trigger"]),
        "started_at": str(row["started_at"]),
        "completed_at": None if row["completed_at"] is None else str(row["completed_at"]),
        "dataset_ref": None if row["dataset_ref"] is None else str(row["dataset_ref"]),
        "dataset_rows": None if row["dataset_rows"] is None else int(row["dataset_rows"]),
        "schema_version": None if row["schema_version"] is None else int(row["schema_version"]),
        "train_rows": None if row["train_rows"] is None else int(row["train_rows"]),
        "test_rows": None if row["test_rows"] is None else int(row["test_rows"]),
        "random_seed": None if row["random_seed"] is None else int(row["random_seed"]),
        "production_model": None if row["production_model"] is None else str(row["production_model"]),
        "production_version": None
        if row["production_version"] is None
        else str(row["production_version"]),
        "best_model": None if row["best_model"] is None else str(row["best_model"]),
        "candidate_artifact": None
        if row["candidate_artifact"] is None
        else str(row["candidate_artifact"]),
        "candidate_run_id": None
        if row["candidate_run_id"] is None
        else str(row["candidate_run_id"]),
        "candidate_version": None
        if row["candidate_version"] is None
        else str(row["candidate_version"]),
        "promotion_eligible": None
        if row["promotion_eligible"] is None
        else bool(row["promotion_eligible"]),
        "promotion_decision": None
        if row["promotion_decision"] is None
        else str(row["promotion_decision"]),
        "failure_reason": None
        if row["failure_reason"] is None
        else str(row["failure_reason"]),
    }
    for column in ("feature_schema", "candidates", "best_metrics", "comparison", "gate_checks"):
        record[column] = _decode_json(row[column], column)
    return record


class RetrainingRunRepository:
    """CRUD operations for the retraining_runs ledger."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initializes the repository with a DatabaseManager."""
        if not isinstance(db_manager, DatabaseManager):
            raise AdaptiveAIError("db_manager must be a DatabaseManager instance.")
        self._db = db_manager

    def init(self) -> None:
        """Creates required tables (idempotent)."""
        self._db.init_db()

    def create_run(
        self,
        run_id: str,
        use_case: str,
        trigger: str,
        started_at: Optional[Union[str, datetime]] = None,
        dataset_ref: Optional[str] = None,
        dataset_rows: Optional[int] = None,
        schema_version: Optional[int] = None,
        train_rows: Optional[int] = None,
        test_rows: Optional[int] = None,
        random_seed: Optional[int] = None,
        feature_schema: Optional[Any] = None,
        candidates: Optional[Any] = None,
        production_model: Optional[str] = None,
        production_version: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Inserts a run in ``started`` status and returns it."""
        if not isinstance(run_id, str) or not run_id.strip():
            raise AdaptiveAIError("run_id must be a non-empty string.")
        if not isinstance(use_case, str) or not use_case.strip():
            raise AdaptiveAIError("use_case must be a non-empty string.")
        if not isinstance(trigger, str) or not trigger.strip():
            raise AdaptiveAIError("trigger must be a non-empty string.")
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            cursor = conn.execute(
                "INSERT INTO retraining_runs "
                "(run_id, use_case, status, trigger, started_at, completed_at, "
                "dataset_ref, dataset_rows, schema_version, train_rows, test_rows, "
                "random_seed, feature_schema, candidates, production_model, "
                "production_version) "
                "VALUES (?, ?, 'started', ?, ?, NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id.strip(),
                    str(use_case),
                    str(trigger),
                    _normalize_timestamp(started_at),
                    _optional_str(dataset_ref, "dataset_ref"),
                    _optional_int(dataset_rows, "dataset_rows"),
                    _optional_int(schema_version, "schema_version"),
                    _optional_int(train_rows, "train_rows"),
                    _optional_int(test_rows, "test_rows"),
                    _optional_int(random_seed, "random_seed"),
                    _encode_json(feature_schema, "feature_schema"),
                    _encode_json(candidates, "candidates"),
                    _optional_str(production_model, "production_model"),
                    _optional_str(production_version, "production_version"),
                ),
            )
            conn.commit()
            created = self.get_run_by_id(int(cursor.lastrowid))
            assert created is not None
            return created
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to create retraining run: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)

    def update_run(self, run_id: str, **fields: Any) -> Dict[str, Any]:
        """Updates mutable run columns of an existing run and returns it.

        Updatable: status, completed_at, best_model, best_metrics,
        comparison, candidate_artifact, candidate_run_id,
        candidate_version, promotion_eligible, promotion_decision,
        gate_checks, failure_reason. Unknown columns raise; the run
        identity and creation facts are immutable.
        """
        allowed = {
            "status": lambda v: str(v) if v in RUN_STATUSES else (_ for _ in ()).throw(
                AdaptiveAIError("status must be one of: started, completed, failed.")
            ),
            "completed_at": lambda v: _normalize_timestamp(v),
            "best_model": lambda v: _optional_str(v, "best_model"),
            "best_metrics": lambda v: _encode_json(v, "best_metrics"),
            "comparison": lambda v: _encode_json(v, "comparison"),
            "candidate_artifact": lambda v: _optional_str(v, "candidate_artifact"),
            "candidate_run_id": lambda v: _optional_str(v, "candidate_run_id"),
            "candidate_version": lambda v: _optional_str(v, "candidate_version"),
            "promotion_eligible": lambda v: (
                None
                if v is None
                else (True if v is True else False if v is False else (_ for _ in ()).throw(
                    AdaptiveAIError("promotion_eligible must be a boolean or None.")
                ))
            ),
            "promotion_decision": lambda v: _optional_str(v, "promotion_decision"),
            "gate_checks": lambda v: _encode_json(v, "gate_checks"),
            "failure_reason": lambda v: _optional_str(v, "failure_reason"),
        }
        unknown = sorted(set(fields) - set(allowed))
        if unknown:
            raise AdaptiveAIError(f"Run columns are not updatable: {', '.join(unknown)}.")
        if not isinstance(run_id, str) or not run_id.strip():
            raise AdaptiveAIError("run_id must be a non-empty string.")
        if not fields:
            existing = self.get_run(run_id)
            if existing is None:
                raise AdaptiveAIError(f"Retraining run not found: {run_id}.")
            return existing
        assignments = ", ".join(f"{column} = ?" for column in fields)
        values = [allowed[column](fields[column]) for column in fields]
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            cursor = conn.execute(
                f"UPDATE retraining_runs SET {assignments} WHERE run_id = ?",
                (*values, run_id.strip()),
            )
            if cursor.rowcount == 0:
                raise AdaptiveAIError(f"Retraining run not found: {run_id}.")
            conn.commit()
            updated = self.get_run(run_id)
            assert updated is not None
            return updated
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to update retraining run: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a run by its backend-generated run_id, or None."""
        if not isinstance(run_id, str) or not run_id.strip():
            raise AdaptiveAIError("run_id must be a non-empty string.")
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            row = conn.execute(
                f"SELECT {_RUN_COLUMNS} FROM retraining_runs WHERE run_id = ?",
                (run_id.strip(),),
            ).fetchone()
            if row is None:
                return None
            return _run_row_to_dict(row)
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to retrieve retraining run {run_id}: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)

    def get_run_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a run by integer id, or None if not found."""
        if isinstance(record_id, bool) or not isinstance(record_id, int):
            raise AdaptiveAIError("record_id must be an integer.")
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            row = conn.execute(
                f"SELECT {_RUN_COLUMNS} FROM retraining_runs WHERE id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                return None
            return _run_row_to_dict(row)
        except sqlite3.Error as e:
            raise AdaptiveAIError(
                f"Failed to retrieve retraining run {record_id}: {e}"
            ) from e
        finally:
            DatabaseManager.close_connection(conn)

    def list_runs(
        self,
        limit: int = 100,
        offset: int = 0,
        use_case: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Lists runs newest-first (ordered by id descending)."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise AdaptiveAIError("limit must be a positive integer.")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise AdaptiveAIError("offset must be a non-negative integer.")
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            if use_case is None:
                rows = conn.execute(
                    f"SELECT {_RUN_COLUMNS} FROM retraining_runs "
                    "ORDER BY id DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            else:
                if not isinstance(use_case, str) or not use_case.strip():
                    raise AdaptiveAIError("use_case must be a non-empty string when provided.")
                rows = conn.execute(
                    f"SELECT {_RUN_COLUMNS} FROM retraining_runs "
                    "WHERE use_case = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                    (str(use_case), limit, offset),
                ).fetchall()
            return [_run_row_to_dict(r) for r in rows]
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to list retraining runs: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)


class LifecycleEventRepository:
    """Append-only audit events for the model lifecycle."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initializes the repository with a DatabaseManager."""
        if not isinstance(db_manager, DatabaseManager):
            raise AdaptiveAIError("db_manager must be a DatabaseManager instance.")
        self._db = db_manager

    def init(self) -> None:
        """Creates required tables (idempotent)."""
        self._db.init_db()

    def record_event(
        self,
        use_case: str,
        event: str,
        run_id: Optional[str] = None,
        model_name: Optional[str] = None,
        model_version: Optional[str] = None,
        previous_production_version: Optional[str] = None,
        new_production_version: Optional[str] = None,
        reason: Optional[str] = None,
        details: Optional[Any] = None,
        timestamp: Optional[Union[str, datetime]] = None,
    ) -> int:
        """Appends one lifecycle event and returns its id."""
        if not isinstance(use_case, str) or not use_case.strip():
            raise AdaptiveAIError("use_case must be a non-empty string.")
        if event not in LIFECYCLE_EVENTS:
            raise AdaptiveAIError(
                "event must be one of: " + ", ".join(sorted(LIFECYCLE_EVENTS)) + "."
            )
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            cursor = conn.execute(
                "INSERT INTO lifecycle_events "
                "(timestamp, use_case, event, run_id, model_name, model_version, "
                "previous_production_version, new_production_version, reason, details) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    _normalize_timestamp(timestamp),
                    str(use_case),
                    str(event),
                    _optional_str(run_id, "run_id"),
                    _optional_str(model_name, "model_name"),
                    _optional_str(model_version, "model_version"),
                    _optional_str(previous_production_version, "previous_production_version"),
                    _optional_str(new_production_version, "new_production_version"),
                    _optional_str(reason, "reason"),
                    _encode_json(details, "details"),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to record lifecycle event: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)

    def list_events(
        self,
        limit: int = 100,
        offset: int = 0,
        use_case: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Lists lifecycle events newest-first (ordered by id descending)."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise AdaptiveAIError("limit must be a positive integer.")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise AdaptiveAIError("offset must be a non-negative integer.")
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            if use_case is None:
                rows = conn.execute(
                    "SELECT id, timestamp, use_case, event, run_id, model_name, "
                    "model_version, previous_production_version, "
                    "new_production_version, reason, details "
                    "FROM lifecycle_events ORDER BY id DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            else:
                if not isinstance(use_case, str) or not use_case.strip():
                    raise AdaptiveAIError("use_case must be a non-empty string when provided.")
                rows = conn.execute(
                    "SELECT id, timestamp, use_case, event, run_id, model_name, "
                    "model_version, previous_production_version, "
                    "new_production_version, reason, details "
                    "FROM lifecycle_events WHERE use_case = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                    (str(use_case), limit, offset),
                ).fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "timestamp": str(r["timestamp"]),
                    "use_case": str(r["use_case"]),
                    "event": str(r["event"]),
                    "run_id": None if r["run_id"] is None else str(r["run_id"]),
                    "model_name": None if r["model_name"] is None else str(r["model_name"]),
                    "model_version": None if r["model_version"] is None else str(r["model_version"]),
                    "previous_production_version": None
                    if r["previous_production_version"] is None
                    else str(r["previous_production_version"]),
                    "new_production_version": None
                    if r["new_production_version"] is None
                    else str(r["new_production_version"]),
                    "reason": None if r["reason"] is None else str(r["reason"]),
                    "details": _decode_json(r["details"], "details"),
                }
                for r in rows
            ]
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to list lifecycle events: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)


class PointerHistoryRepository:
    """Append-only serving-reference transitions for validated rollback."""

    def __init__(self, db_manager: DatabaseManager) -> None:
        """Initializes the repository with a DatabaseManager."""
        if not isinstance(db_manager, DatabaseManager):
            raise AdaptiveAIError("db_manager must be a DatabaseManager instance.")
        self._db = db_manager

    def init(self) -> None:
        """Creates required tables (idempotent)."""
        self._db.init_db()

    def record_transition(
        self,
        use_case: str,
        new_pointer: Dict[str, Any],
        previous_pointer: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        run_id: Optional[str] = None,
        timestamp: Optional[Union[str, datetime]] = None,
    ) -> int:
        """Appends one pointer transition and returns its id."""
        if not isinstance(use_case, str) or not use_case.strip():
            raise AdaptiveAIError("use_case must be a non-empty string.")
        if not isinstance(new_pointer, dict) or not new_pointer:
            raise AdaptiveAIError("new_pointer must be a non-empty dict.")
        if previous_pointer is not None and not isinstance(previous_pointer, dict):
            raise AdaptiveAIError("previous_pointer must be a dict or None.")
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            cursor = conn.execute(
                "INSERT INTO production_pointer_history "
                "(timestamp, use_case, previous_pointer, new_pointer, reason, run_id) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    _normalize_timestamp(timestamp),
                    str(use_case),
                    _encode_json(previous_pointer, "previous_pointer"),
                    _encode_json(new_pointer, "new_pointer"),
                    _optional_str(reason, "reason"),
                    _optional_str(run_id, "run_id"),
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to record pointer transition: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)

    def get_transition(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a pointer transition by id, or None if not found."""
        if isinstance(record_id, bool) or not isinstance(record_id, int):
            raise AdaptiveAIError("record_id must be an integer.")
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            row = conn.execute(
                "SELECT id, timestamp, use_case, previous_pointer, new_pointer, reason, run_id "
                "FROM production_pointer_history WHERE id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                return None
            return {
                "id": int(row["id"]),
                "timestamp": str(row["timestamp"]),
                "use_case": str(row["use_case"]),
                "previous_pointer": _decode_json(row["previous_pointer"], "previous_pointer"),
                "new_pointer": _decode_json(row["new_pointer"], "new_pointer"),
                "reason": None if row["reason"] is None else str(row["reason"]),
                "run_id": None if row["run_id"] is None else str(row["run_id"]),
            }
        except sqlite3.Error as e:
            raise AdaptiveAIError(
                f"Failed to retrieve pointer transition {record_id}: {e}"
            ) from e
        finally:
            DatabaseManager.close_connection(conn)

    def list_transitions(
        self,
        limit: int = 100,
        offset: int = 0,
        use_case: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Lists pointer transitions newest-first (ordered by id descending)."""
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise AdaptiveAIError("limit must be a positive integer.")
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise AdaptiveAIError("offset must be a non-negative integer.")
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self._db.get_connection()
            if use_case is None:
                rows = conn.execute(
                    "SELECT id, timestamp, use_case, previous_pointer, new_pointer, reason, run_id "
                    "FROM production_pointer_history ORDER BY id DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            else:
                if not isinstance(use_case, str) or not use_case.strip():
                    raise AdaptiveAIError("use_case must be a non-empty string when provided.")
                rows = conn.execute(
                    "SELECT id, timestamp, use_case, previous_pointer, new_pointer, reason, run_id "
                    "FROM production_pointer_history WHERE use_case = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                    (str(use_case), limit, offset),
                ).fetchall()
            return [
                {
                    "id": int(r["id"]),
                    "timestamp": str(r["timestamp"]),
                    "use_case": str(r["use_case"]),
                    "previous_pointer": _decode_json(r["previous_pointer"], "previous_pointer"),
                    "new_pointer": _decode_json(r["new_pointer"], "new_pointer"),
                    "reason": None if r["reason"] is None else str(r["reason"]),
                    "run_id": None if r["run_id"] is None else str(r["run_id"]),
                }
                for r in rows
            ]
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to list pointer transitions: {e}") from e
        finally:
            DatabaseManager.close_connection(conn)
