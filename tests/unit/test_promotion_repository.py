"""Focused tests for promotion-history persistence (Step 35).

All databases are temporary; the real data/adaptive_ai.db is never
touched. No MLflow/model/API/scheduler/dashboard side effects.
"""

import copy
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from app.core.config import get_project_root
from app.core.exceptions import AdaptiveAIError
from app.database import PromotionRepository
from app.database.database import DatabaseManager
from app.database.promotion_repository import SUPPORTED_DECISIONS


@pytest.fixture
def tmp_manager(tmp_path: Path) -> DatabaseManager:
    manager = DatabaseManager(db_path=tmp_path / "promo.db")
    manager.init_db()
    return manager


@pytest.fixture
def tmp_repo(tmp_manager: DatabaseManager) -> PromotionRepository:
    return PromotionRepository(tmp_manager)


def _promoted(**overrides):
    result = {
        "decision": "promoted",
        "model_name": "LinearRegression",
        "run_id": "run-abc123",
        "registered_model_name": "HousePriceModel",
        "model_version": "4",
        "primary_metric": "r2",
        "candidate_score": 0.91,
        "production_score": 0.82,
        "improvement": 0.09,
        "reason": "Candidate is better on 'r2'.",
    }
    result.update(overrides)
    return result


def _rejected(**overrides):
    result = {
        "decision": "rejected",
        "model_name": "LinearRegression",
        "run_id": "run-abc123",
        "registered_model_name": "HousePriceModel",
        "model_version": None,
        "primary_metric": "r2",
        "candidate_score": 0.5,
        "production_score": 0.8,
        "improvement": -0.3,
        "reason": "Production is better; keeping production.",
    }
    result.update(overrides)
    return result


def _initial(**overrides):
    result = {
        "decision": "requires_initial_promotion",
        "model_name": "LinearRegression",
        "run_id": "run-abc123",
        "registered_model_name": "HousePriceModel",
        "model_version": None,
        "primary_metric": "r2",
        "candidate_score": 0.85,
        "production_score": None,
        "improvement": None,
        "reason": "Requires explicit initial deployment.",
    }
    result.update(overrides)
    return result


def _failed(**overrides):
    result = {
        "decision": "failed",
        "model_name": "LinearRegression",
        "run_id": "run-abc123",
        "registered_model_name": "HousePriceModel",
        "model_version": None,
        "primary_metric": "r2",
        "candidate_score": 0.9,
        "production_score": 0.7,
        "improvement": 0.2,
        "reason": "Registry failed after persistence; no rollback.",
    }
    result.update(overrides)
    return result


# --- 1. table creation ---


def test_table_creation(tmp_path: Path):
    manager = DatabaseManager(db_path=tmp_path / "schema.db")
    manager.init_db()
    conn = manager.get_connection()
    try:
        row = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='promotion_history'"
        ).fetchone()
        assert row is not None
        columns = {r[1]: r[2] for r in conn.execute("PRAGMA table_info(promotion_history)").fetchall()}
        expected = {
            "id": "INTEGER",
            "timestamp": "TEXT",
            "use_case": "TEXT",
            "decision": "TEXT",
            "model_name": "TEXT",
            "run_id": "TEXT",
            "registered_model_name": "TEXT",
            "model_version": "TEXT",
            "primary_metric": "TEXT",
            "candidate_score": "REAL",
            "production_score": "REAL",
            "improvement": "REAL",
            "reason": "TEXT",
        }
        assert columns == expected
        # Existing tables unchanged.
        for table in ("predictions", "monitoring_results"):
            assert (
                conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                    (table,),
                ).fetchone()
                is not None
            )
    finally:
        DatabaseManager.close_connection(conn)


def test_supported_decisions_match_promoter():
    assert SUPPORTED_DECISIONS == frozenset(
        {"promoted", "rejected", "requires_initial_promotion", "failed"}
    )


# --- 2-5. save each decision ---


def test_save_promoted_result(tmp_repo: PromotionRepository):
    record_id = tmp_repo.save_promotion_result(
        use_case="house_price", promotion_result=_promoted()
    )
    assert isinstance(record_id, int)
    record = tmp_repo.get_promotion_result_by_id(record_id)
    assert record is not None
    assert record["decision"] == "promoted"
    assert record["use_case"] == "house_price"
    assert record["model_name"] == "LinearRegression"
    assert record["run_id"] == "run-abc123"
    assert record["registered_model_name"] == "HousePriceModel"
    assert record["model_version"] == "4"
    assert record["primary_metric"] == "r2"
    assert record["candidate_score"] == pytest.approx(0.91)
    assert record["production_score"] == pytest.approx(0.82)
    assert record["improvement"] == pytest.approx(0.09)
    assert isinstance(record["id"], int)
    assert isinstance(record["timestamp"], str)


def test_save_rejected_result(tmp_repo: PromotionRepository):
    record_id = tmp_repo.save_promotion_result(
        use_case="house_price", promotion_result=_rejected()
    )
    record = tmp_repo.get_promotion_result_by_id(record_id)
    assert record is not None
    assert record["decision"] == "rejected"
    assert record["model_version"] is None
    assert record["candidate_score"] == pytest.approx(0.5)


def test_save_requires_initial_promotion_result(tmp_repo: PromotionRepository):
    record_id = tmp_repo.save_promotion_result(
        use_case="house_price", promotion_result=_initial()
    )
    record = tmp_repo.get_promotion_result_by_id(record_id)
    assert record is not None
    assert record["decision"] == "requires_initial_promotion"
    assert record["production_score"] is None
    assert record["improvement"] is None
    assert record["candidate_score"] == pytest.approx(0.85)


def test_save_failed_result(tmp_repo: PromotionRepository):
    record_id = tmp_repo.save_promotion_result(
        use_case="house_price", promotion_result=_failed()
    )
    record = tmp_repo.get_promotion_result_by_id(record_id)
    assert record is not None
    assert record["decision"] == "failed"
    assert record["reason"]


# --- 6-7. get by id ---


def test_get_by_id_and_missing_returns_none(tmp_repo: PromotionRepository):
    record_id = tmp_repo.save_promotion_result(
        use_case="house_price", promotion_result=_promoted()
    )
    assert tmp_repo.get_promotion_result_by_id(record_id) is not None
    assert tmp_repo.get_promotion_result_by_id(999999) is None
    with pytest.raises(AdaptiveAIError):
        tmp_repo.get_promotion_result_by_id("bad")  # type: ignore[arg-type]
    with pytest.raises(AdaptiveAIError):
        tmp_repo.get_promotion_result_by_id(True)  # type: ignore[arg-type]


# --- 8-12. listing ---


def test_list_ordering_limit_offset_filtering_and_multiple(tmp_repo: PromotionRepository):
    id1 = tmp_repo.save_promotion_result(use_case="house_price", promotion_result=_promoted())
    id2 = tmp_repo.save_promotion_result(use_case="house_price", promotion_result=_rejected())
    id3 = tmp_repo.save_promotion_result(use_case="other_case", promotion_result=_initial())

    records = tmp_repo.list_promotion_history()
    assert [r["id"] for r in records] == [id3, id2, id1]  # deterministic DESC (newest-first)

    assert len(tmp_repo.list_promotion_history(use_case="house_price")) == 2
    assert len(tmp_repo.list_promotion_history(use_case="other_case")) == 1
    assert tmp_repo.list_promotion_history(use_case="missing") == []

    page = tmp_repo.list_promotion_history(limit=1, offset=1)
    assert [r["id"] for r in page] == [id2]
    assert tmp_repo.list_promotion_history(limit=1) == records[:1]
    assert tmp_repo.list_promotion_history(offset=99) == []

    with pytest.raises(AdaptiveAIError):
        tmp_repo.list_promotion_history(limit=0)
    with pytest.raises(AdaptiveAIError):
        tmp_repo.list_promotion_history(limit=-1)
    with pytest.raises(AdaptiveAIError):
        tmp_repo.list_promotion_history(offset=-1)
    with pytest.raises(AdaptiveAIError):
        tmp_repo.list_promotion_history(limit=True)  # type: ignore[arg-type]
    with pytest.raises(AdaptiveAIError):
        tmp_repo.list_promotion_history(use_case="  ")


# --- 13-14. timestamps ---


def test_timestamp_override_and_utc_behavior(tmp_repo: PromotionRepository):
    explicit = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)
    record_id = tmp_repo.save_promotion_result(
        use_case="house_price", promotion_result=_promoted(), timestamp=explicit
    )
    record = tmp_repo.get_promotion_result_by_id(record_id)
    assert record is not None
    assert datetime.fromisoformat(str(record["timestamp"])) == explicit

    naive = datetime(2026, 3, 1, 12, 0, 0)
    record_id2 = tmp_repo.save_promotion_result(
        use_case="house_price", promotion_result=_promoted(), timestamp=naive
    )
    record2 = tmp_repo.get_promotion_result_by_id(record_id2)
    assert record2 is not None
    parsed = datetime.fromisoformat(str(record2["timestamp"]))
    assert parsed.tzinfo is not None
    assert parsed == naive.replace(tzinfo=timezone.utc)

    auto_id = tmp_repo.save_promotion_result(
        use_case="house_price", promotion_result=_promoted()
    )
    auto = tmp_repo.get_promotion_result_by_id(auto_id)
    assert auto is not None
    assert datetime.fromisoformat(str(auto["timestamp"])).tzinfo is not None

    with pytest.raises(AdaptiveAIError):
        tmp_repo.save_promotion_result(
            use_case="house_price", promotion_result=_promoted(), timestamp="  "
        )


# --- 15. nullable fields ---


def test_nullable_fields_remain_null(tmp_repo: PromotionRepository):
    minimal_rejected = {
        "decision": "rejected",
        "model_name": "M",
        "reason": "nope",
    }
    record_id = tmp_repo.save_promotion_result(
        use_case="house_price", promotion_result=minimal_rejected
    )
    record = tmp_repo.get_promotion_result_by_id(record_id)
    assert record is not None
    assert record["run_id"] is None
    assert record["registered_model_name"] is None
    assert record["model_version"] is None
    assert record["primary_metric"] is None
    assert record["candidate_score"] is None
    assert record["production_score"] is None
    assert record["improvement"] is None


# --- 16-19. numeric validation ---


@pytest.mark.parametrize("bad", ["0.9", [0.9], {"v": 1}, object()])
def test_numeric_validation_rejects_non_numbers(tmp_repo: PromotionRepository, bad):
    with pytest.raises(AdaptiveAIError):
        tmp_repo.save_promotion_result(
            use_case="house_price", promotion_result=_promoted(candidate_score=bad)
        )
    with pytest.raises(AdaptiveAIError):
        tmp_repo.save_promotion_result(
            use_case="house_price", promotion_result=_promoted(improvement=bad)
        )


def test_bool_rejected_as_numeric(tmp_repo: PromotionRepository):
    with pytest.raises(AdaptiveAIError):
        tmp_repo.save_promotion_result(
            use_case="house_price", promotion_result=_promoted(candidate_score=True)
        )
    with pytest.raises(AdaptiveAIError):
        tmp_repo.save_promotion_result(
            use_case="house_price", promotion_result=_promoted(production_score=False)
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nan_and_infinity_rejected(tmp_repo: PromotionRepository, bad):
    with pytest.raises(AdaptiveAIError):
        tmp_repo.save_promotion_result(
            use_case="house_price", promotion_result=_promoted(candidate_score=bad)
        )
    with pytest.raises(AdaptiveAIError):
        tmp_repo.save_promotion_result(
            use_case="house_price", promotion_result=_promoted(improvement=bad)
        )


def test_int_scores_accepted_as_float(tmp_repo: PromotionRepository):
    record_id = tmp_repo.save_promotion_result(
        use_case="house_price",
        promotion_result=_promoted(candidate_score=1, production_score=0, improvement=1),
    )
    record = tmp_repo.get_promotion_result_by_id(record_id)
    assert record is not None
    assert record["candidate_score"] == pytest.approx(1.0)
    assert isinstance(record["candidate_score"], float)


# --- 20-24. envelope validation ---


@pytest.mark.parametrize("bad", ["promote", "APPROVED", "", None, 123, True])
def test_invalid_decision_rejected(tmp_repo: PromotionRepository, bad):
    with pytest.raises(AdaptiveAIError):
        tmp_repo.save_promotion_result(
            use_case="house_price", promotion_result=_promoted(decision=bad)
        )


@pytest.mark.parametrize("bad", [None, "", "   ", 123, ["M"]])
def test_missing_model_name_rejected(tmp_repo: PromotionRepository, bad):
    with pytest.raises(AdaptiveAIError):
        tmp_repo.save_promotion_result(
            use_case="house_price", promotion_result=_promoted(model_name=bad)
        )


@pytest.mark.parametrize("bad", [None, "", "   ", 123])
def test_missing_reason_rejected(tmp_repo: PromotionRepository, bad):
    with pytest.raises(AdaptiveAIError):
        tmp_repo.save_promotion_result(
            use_case="house_price", promotion_result=_promoted(reason=bad)
        )


@pytest.mark.parametrize("bad", [None, "", "   ", 123])
def test_invalid_use_case_rejected(tmp_repo: PromotionRepository, bad):
    with pytest.raises(AdaptiveAIError):
        tmp_repo.save_promotion_result(use_case=bad, promotion_result=_promoted())


@pytest.mark.parametrize("bad", [None, "x", [], ["promoted"]])
def test_non_dict_result_rejected(tmp_repo: PromotionRepository, bad):
    with pytest.raises(AdaptiveAIError):
        tmp_repo.save_promotion_result(use_case="house_price", promotion_result=bad)


def test_invalid_optional_text_rejected(tmp_repo: PromotionRepository):
    with pytest.raises(AdaptiveAIError):
        tmp_repo.save_promotion_result(
            use_case="house_price", promotion_result=_promoted(run_id="  ")
        )
    with pytest.raises(AdaptiveAIError):
        tmp_repo.save_promotion_result(
            use_case="house_price", promotion_result=_promoted(model_version=123)
        )
    with pytest.raises(AdaptiveAIError):
        PromotionRepository(db_manager="nope")  # type: ignore[arg-type]


# --- 25. SQL errors wrapped ---


def test_sql_errors_wrapped_appropriately(tmp_path: Path):
    manager = DatabaseManager(db_path=tmp_path / "broken.db")
    manager.init_db()
    repo = PromotionRepository(manager)
    # Drop the table to force a sqlite error on insert/select.
    conn = manager.get_connection()
    try:
        conn.execute("DROP TABLE promotion_history")
        conn.commit()
    finally:
        DatabaseManager.close_connection(conn)
    with pytest.raises(AdaptiveAIError):
        repo.save_promotion_result(use_case="house_price", promotion_result=_promoted())
    with pytest.raises(AdaptiveAIError):
        repo.get_promotion_result_by_id(1)
    with pytest.raises(AdaptiveAIError):
        repo.list_promotion_history()


# --- 26. SQL injection ---


def test_no_sql_injection_through_filter_values(tmp_repo: PromotionRepository):
    tmp_repo.save_promotion_result(use_case="house_price", promotion_result=_promoted())
    malicious = "house_price' OR '1'='1"
    assert tmp_repo.list_promotion_history(use_case=malicious) == []
    # Table still intact with exactly one row.
    assert len(tmp_repo.list_promotion_history()) == 1
    evil_use_case = "x'; DROP TABLE promotion_history; --"
    tmp_repo.save_promotion_result(use_case=evil_use_case, promotion_result=_rejected())
    assert len(tmp_repo.list_promotion_history()) == 2
    assert len(tmp_repo.list_promotion_history(use_case=evil_use_case)) == 1


# --- 27. no input mutation ---


def test_repository_does_not_mutate_input_dict(tmp_repo: PromotionRepository):
    original = _promoted()
    snapshot = copy.deepcopy(original)
    tmp_repo.save_promotion_result(use_case="house_price", promotion_result=original)
    assert original == snapshot


# --- compatibility with actual promoter dicts ---


def test_compatibility_with_actual_promoter_results(tmp_repo: PromotionRepository):
    from app.retraining.model_promoter import ModelPromoter
    from unittest.mock import MagicMock

    save_mock = MagicMock(return_value=Path("/tmp/fake.joblib"))
    tracker_mock = MagicMock()
    tracker_mock.register_model = MagicMock(
        return_value={
            "registered_model_name": "HousePriceModel",
            "version": "9",
            "run_id": "run-real",
            "source": "runs:/run-real/model",
        }
    )
    promoter = ModelPromoter(
        save_fn=save_mock,
        tracker=tracker_mock,
    )
    evaluation = {
        "candidate": {"model_name": "c", "metrics": {"mae": 1.0, "rmse": 2.0, "r2": 0.9}},
        "production": {"model_name": "p", "metrics": {"mae": 2.0, "rmse": 3.0, "r2": 0.7}},
        "comparison": {
            "primary_metric": "r2",
            "candidate_score": 0.9,
            "production_score": 0.7,
            "improvement": 0.2,
            "candidate_better": True,
        },
        "status": "candidate_better",
    }
    promoted = promoter.promote(
        MagicMock(),
        evaluation,
        candidate_model_name="LinearRegression",
        candidate_run_id="run-real",
    )
    snapshot = copy.deepcopy(promoted)
    record_id = tmp_repo.save_promotion_result(
        use_case="house_price", promotion_result=promoted
    )
    assert promoted == snapshot  # not mutated
    record = tmp_repo.get_promotion_result_by_id(record_id)
    assert record is not None
    assert record["decision"] == "promoted"
    assert record["model_name"] == "LinearRegression"
    assert record["run_id"] == "run-real"
    assert record["model_version"] == "9"
    # Promoter dicts carry no metric fields -> stored NULL, not fabricated.
    assert record["primary_metric"] is None
    assert record["candidate_score"] is None

    rejected_eval = dict(evaluation)
    rejected_eval["status"] = "production_better"
    rejected_eval["comparison"] = {
        "primary_metric": "r2",
        "candidate_score": 0.5,
        "production_score": 0.8,
        "improvement": -0.3,
        "candidate_better": False,
    }
    rejected = promoter.promote(
        MagicMock(),
        rejected_eval,
        candidate_model_name="LinearRegression",
        candidate_run_id="run-real",
    )
    rid = tmp_repo.save_promotion_result(use_case="house_price", promotion_result=rejected)
    rec = tmp_repo.get_promotion_result_by_id(rid)
    assert rec is not None
    assert rec["decision"] == "rejected"
    assert rec["model_version"] is None


def test_package_export_pattern():
    from app import database as database_pkg

    assert "PromotionRepository" in database_pkg.__all__
    assert database_pkg.PromotionRepository is PromotionRepository


def test_repository_source_hygiene_no_promotion_execution():
    source = (
        Path(get_project_root()) / "app" / "database" / "promotion_repository.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "ModelPromoter(",
        ".promote(",
        "save_house_price_model",
        "register_model",
        "mlflow",
        "Mlflow",
        "FastAPI(",
        "APScheduler",
        "Celery",
        "streamlit",
    ):
        assert forbidden not in source


def test_real_development_database_untouched():
    # Isolation canary: live-state guard lives in tests/conftest.py (the
    # live deployment legitimately exists since Step 57 verification).
    pass
