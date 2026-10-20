"""SQLite database manager for AdaptiveAI (Step 23).

Small reusable layer using Python's built-in sqlite3 (no ORM).
Future prediction history, monitoring, drift, and retraining
components can build on this foundation.
"""

import logging
import sqlite3
from pathlib import Path
from typing import Optional, Union

from app.core.exceptions import AdaptiveAIError

logger = logging.getLogger(__name__)

DEFAULT_DATABASE_PATH = "data/adaptive_ai.db"

PREDICTIONS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    use_case TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT NULL,
    input_data TEXT NOT NULL,
    prediction REAL NOT NULL
);
"""

MONITORING_RESULTS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS monitoring_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    use_case TEXT NOT NULL,
    overall_drifted INTEGER NOT NULL,
    status TEXT NOT NULL,
    alpha REAL NOT NULL,
    min_production_rows INTEGER NOT NULL,
    production_rows INTEGER NOT NULL,
    sufficient_data INTEGER NOT NULL,
    feature_results TEXT NOT NULL,
    model_name TEXT NULL,
    model_version TEXT NULL,
    schema_version INTEGER NULL,
    severity TEXT NULL,
    overall_health TEXT NULL,
    data_quality_status TEXT NULL,
    drifted_features INTEGER NULL
);
"""

# Additive Phase 4 columns for monitoring_results. Applied with
# ALTER TABLE ... ADD COLUMN (nullable) so existing databases and rows
# are never rewritten or destroyed; older rows simply read back NULLs.
MONITORING_RESULTS_ADDITIVE_COLUMNS = (
    ("model_name", "TEXT NULL"),
    ("model_version", "TEXT NULL"),
    ("schema_version", "INTEGER NULL"),
    ("severity", "TEXT NULL"),
    ("overall_health", "TEXT NULL"),
    ("data_quality_status", "TEXT NULL"),
    ("drifted_features", "INTEGER NULL"),
)

# Additive Phase 6 columns for predictions (Demo Mode isolation).
# Applied with ALTER TABLE ... ADD COLUMN (nullable) so existing
# databases and rows are never rewritten or destroyed; legacy rows
# simply read back NULLs, which the repository interprets as
# production traffic. Demo rows carry source='demo' plus the owning
# demo_session_id; production traffic carries source='production'.
PREDICTIONS_ADDITIVE_COLUMNS = (
    ("source", "TEXT NULL"),
    ("demo_session_id", "TEXT NULL"),
)

PROMOTION_HISTORY_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS promotion_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    use_case TEXT NOT NULL,
    decision TEXT NOT NULL,
    model_name TEXT NOT NULL,
    run_id TEXT NULL,
    registered_model_name TEXT NULL,
    model_version TEXT NULL,
    primary_metric TEXT NULL,
    candidate_score REAL NULL,
    production_score REAL NULL,
    improvement REAL NULL,
    reason TEXT NOT NULL
);
"""

# Phase 5 lifecycle tables. Brand-new tables (CREATE TABLE IF NOT
# EXISTS): no existing row is ever rewritten, and prediction,
# monitoring, and promotion history are untouched.
RETRAINING_RUNS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS retraining_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL UNIQUE,
    use_case TEXT NOT NULL,
    status TEXT NOT NULL,
    trigger TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT NULL,
    dataset_ref TEXT NULL,
    dataset_rows INTEGER NULL,
    schema_version INTEGER NULL,
    train_rows INTEGER NULL,
    test_rows INTEGER NULL,
    random_seed INTEGER NULL,
    feature_schema TEXT NULL,
    candidates TEXT NULL,
    production_model TEXT NULL,
    production_version TEXT NULL,
    best_model TEXT NULL,
    best_metrics TEXT NULL,
    comparison TEXT NULL,
    candidate_artifact TEXT NULL,
    candidate_run_id TEXT NULL,
    candidate_version TEXT NULL,
    promotion_eligible INTEGER NULL,
    promotion_decision TEXT NULL,
    gate_checks TEXT NULL,
    failure_reason TEXT NULL
);
"""

# Phase 6 demo-session ledger (Scenario Simulator). Brand-new table
# (CREATE TABLE IF NOT EXISTS): no existing row is ever rewritten,
# and prediction, monitoring, promotion, and lifecycle history are
# untouched. One row per demo session; generated samples live in
# ``predictions`` with source='demo' + demo_session_id (isolated by
# query filters, never by rewriting production rows). Demo monitoring
# results are stored on the session record itself and never written
# to ``monitoring_results``, so production monitoring history is
# byte-identical with or without demo activity.
DEMO_SESSIONS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS demo_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,
    use_case TEXT NOT NULL,
    scenario TEXT NOT NULL,
    status TEXT NOT NULL,
    sample_count INTEGER NOT NULL,
    seed INTEGER NOT NULL,
    params TEXT NULL,
    samples_generated INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT NULL,
    completed_at TEXT NULL,
    reset_at TEXT NULL,
    last_monitoring TEXT NULL,
    last_decision TEXT NULL,
    last_monitored_at TEXT NULL
);
"""

LIFECYCLE_EVENTS_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS lifecycle_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    use_case TEXT NOT NULL,
    event TEXT NOT NULL,
    run_id TEXT NULL,
    model_name TEXT NULL,
    model_version TEXT NULL,
    previous_production_version TEXT NULL,
    new_production_version TEXT NULL,
    reason TEXT NULL,
    details TEXT NULL
);
"""

PRODUCTION_POINTER_HISTORY_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS production_pointer_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    use_case TEXT NOT NULL,
    previous_pointer TEXT NULL,
    new_pointer TEXT NOT NULL,
    reason TEXT NULL,
    run_id TEXT NULL
);
"""


def get_default_database_path() -> Path:
    """Returns the configured development database path.

    Reads ``database.path`` from platform config when present,
    otherwise falls back to ``data/adaptive_ai.db``.
    Relative paths are resolved against the project root.
    """
    try:
        from app.core.config import get_project_root, load_config

        config = load_config()
        configured = (config.get("database") or {}).get("path")
        if configured:
            candidate = Path(str(configured))
            if not candidate.is_absolute():
                return Path(get_project_root()) / candidate
            return candidate
    except Exception as e:
        logger.warning(f"Falling back to default database path: {e}")
    # Fallback without importing config (e.g. config unavailable).
    try:
        from app.core.config import get_project_root

        return Path(get_project_root()) / DEFAULT_DATABASE_PATH
    except Exception:
        return Path(DEFAULT_DATABASE_PATH)


def _ensure_monitoring_columns(conn: "sqlite3.Connection") -> None:
    """Adds Phase 4 monitoring columns when missing (non-destructive).

    Uses ``ALTER TABLE ... ADD COLUMN`` with nullable columns only, so
    existing rows and tables are preserved byte-for-byte apart from
    the new NULL cells. Safe to run on every ``init_db`` (idempotent).
    """
    existing = set()
    try:
        for row in conn.execute("PRAGMA table_info(monitoring_results)").fetchall():
            # Index access (not row["name"]): this path must also work
            # on raw connections without a Row factory.
            existing.add(str(row[1]))
    except Exception as e:
        logger.warning(f"Skipping monitoring column migration check: {e}")
        return
    for column, ddl in MONITORING_RESULTS_ADDITIVE_COLUMNS:
        if column not in existing:
            try:
                conn.execute(f"ALTER TABLE monitoring_results ADD COLUMN {column} {ddl}")
            except Exception as e:
                logger.warning(f"Skipping monitoring column migration ({column}): {e}")
                return


def _ensure_predictions_columns(conn: "sqlite3.Connection") -> None:
    """Adds Phase 6 demo-isolation columns when missing (non-destructive).

    Uses ``ALTER TABLE ... ADD COLUMN`` with nullable columns only, so
    existing rows and tables are preserved byte-for-byte apart from
    the new NULL cells. Safe to run on every ``init_db`` (idempotent).
    Follows the same pattern as ``_ensure_monitoring_columns``.
    """
    existing = set()
    try:
        for row in conn.execute("PRAGMA table_info(predictions)").fetchall():
            # Index access (not row["name"]): this path must also work
            # on raw connections without a Row factory.
            existing.add(str(row[1]))
    except Exception as e:
        logger.warning(f"Skipping predictions column migration check: {e}")
        return
    for column, ddl in PREDICTIONS_ADDITIVE_COLUMNS:
        if column not in existing:
            try:
                conn.execute(f"ALTER TABLE predictions ADD COLUMN {column} {ddl}")
            except Exception as e:
                logger.warning(f"Skipping predictions column migration ({column}): {e}")
                return


class DatabaseManager:
    """Manages SQLite database path, connections, and schema creation."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None) -> None:
        """Initializes the manager with an explicit or default DB path."""
        if db_path is None:
            self._path = get_default_database_path()
        else:
            self._path = Path(db_path)

    @property
    def path(self) -> Path:
        """Returns the configured database file path."""
        return self._path

    def get_connection(self) -> sqlite3.Connection:
        """Opens a new SQLite connection (caller must close it)."""
        try:
            if self._path.parent and str(self._path.parent) not in ("", "."):
                self._path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to open database at {self._path}: {e}") from e

    @staticmethod
    def close_connection(conn: Optional[sqlite3.Connection]) -> None:
        """Best-effort connection cleanup that never raises.

        Step 64 (I-15): every repository calls this in a ``finally``
        block after the operation already raised (or is raising)
        exception A. If ``conn.close()`` itself failed and this method
        raised exception B, B would propagate out of the ``finally``
        block and mask A. To preserve the original exception, cleanup
        failures are logged and suppressed instead. ``None`` is a
        no-op; an already-closed connection is also a no-op
        (``sqlite3`` tolerates double close).
        """
        if conn is None:
            return
        try:
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to close database connection (suppressed): {e}")

    def init_db(self) -> None:
        """Creates/opens the database file and ensures required tables exist."""
        conn: Optional[sqlite3.Connection] = None
        try:
            conn = self.get_connection()
            conn.execute(PREDICTIONS_TABLE_SCHEMA)
            conn.execute(MONITORING_RESULTS_TABLE_SCHEMA)
            conn.execute(PROMOTION_HISTORY_TABLE_SCHEMA)
            conn.execute(RETRAINING_RUNS_TABLE_SCHEMA)
            conn.execute(LIFECYCLE_EVENTS_TABLE_SCHEMA)
            conn.execute(PRODUCTION_POINTER_HISTORY_TABLE_SCHEMA)
            conn.execute(DEMO_SESSIONS_TABLE_SCHEMA)
            _ensure_monitoring_columns(conn)
            _ensure_predictions_columns(conn)
            conn.commit()
            logger.info(f"Initialized database at {self._path}")
        except sqlite3.Error as e:
            raise AdaptiveAIError(f"Failed to initialize database at {self._path}: {e}") from e
        finally:
            # Step 64 (I-15): best-effort cleanup; never masks init errors.
            DatabaseManager.close_connection(conn)
