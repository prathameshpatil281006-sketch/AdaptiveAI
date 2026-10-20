"""Data-quality checks for the House Price monitoring window (Phase 4).

Row-level validation of the current production prediction window
(raw ``input_data`` mappings, newest-first). Read-only: rows are
counted and reported, never deleted or mutated — a failing check
never touches production data.

Checks (each returns status pass/warn/fail, rows checked/affected,
and a human-readable message):

- ``schema`` (fail): rows that are not dicts, or dicts matching
  neither the v2 schema (all 10 features) nor the legacy v1 schema.
- ``legacy_schema`` (warn): rows stored under the legacy v1 schema.
  They are preserved history, excluded from drift, and only surfaced
  here so operators know the window mixes schemas.
- ``missing_values`` (fail): null/NaN cells in v2 rows.
- ``duplicates`` (warn): fully duplicated v2 rows. Repeats are
  legitimate traffic (e.g. controlled drift samples posted
  repeatedly), so duplicates warn rather than fail.
- ``numeric_ranges`` (fail): area_sqft > 0, bedrooms/bathrooms
  int-like >= 1, parking int-like 0-3, property_age >= 0,
  floor_number int-like >= 0, total_floors int-like >= 1
  (mirrors the serving validation contract).
- ``categorical_values`` (fail): location / furnishing_status /
  property_type membership in the configured value sets.
- ``relationships`` (fail): floor_number <= total_floors and
  bathrooms <= bedrooms + 2.
- ``outliers`` (warn): area outside the documented BHK carpet-area
  band (``BHK_AREA_BANDS``) for bedrooms 1-5. A heuristic flag for
  review, not a rejection.

Overall status: ``fail`` if any fail-check is affected, else
``warn`` if any warn-check is affected, else ``pass``. An empty
window yields ``unknown`` (no claim without data).

No I/O, no database writes, no model access.
"""

import logging
import math
import numbers
from typing import Any, Dict, List

from app.use_cases.house_price.market_config import (
    BHK_AREA_BANDS,
    FEATURE_COLUMNS,
    FURNISHING_VALUES,
    LOCALITY_RATES,
    NUMERIC_FEATURES,
    PROPERTY_TYPE_VALUES,
)

logger = logging.getLogger(__name__)

STATUS_PASS = "pass"
STATUS_WARN = "warn"
STATUS_FAIL = "fail"
STATUS_UNKNOWN = "unknown"

# Legacy v1 schema columns (pre-Pune prediction inputs).
LEGACY_V1_COLUMNS = {"area", "bedrooms", "bathrooms", "property_age", "parking"}


def _is_int_like(value: Any) -> bool:
    """Accepts ints and integral floats (JSON-friendly), rejects bools."""
    if isinstance(value, bool):
        return False
    if isinstance(value, numbers.Integral):
        return True
    if isinstance(value, float) and math.isfinite(value) and value.is_integer():
        return True
    return False


def _is_missing(value: Any) -> bool:
    """Null/NaN/empty-string cells count as missing."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return False


def _check(
    name: str, status: str, checked: int, affected: int, message: str
) -> Dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "checked": int(checked),
        "affected": int(affected),
        "message": message,
    }


def run_house_price_data_quality(rows: List[Any]) -> Dict[str, Any]:
    """Runs all data-quality checks over a window of raw input rows.

    Args:
        rows: Raw ``input_data`` mappings (newest-first window). May
            contain legacy-schema or malformed entries; they are
            counted, never raised on.

    Returns:
        Dict with ``status``, ``total_rows``, ``v2_rows``,
        ``legacy_rows``, ``malformed_rows``, and ``checks``.
    """
    if not isinstance(rows, list):
        rows = []
    total = len(rows)
    v2_rows: List[Dict[str, Any]] = []
    legacy_rows: List[Dict[str, Any]] = []
    malformed = 0
    schema_bad = 0
    v2_required = set(FEATURE_COLUMNS)
    for entry in rows:
        if not isinstance(entry, dict):
            malformed += 1
            continue
        if v2_required.issubset(set(entry.keys())):
            v2_rows.append(entry)
        elif LEGACY_V1_COLUMNS.issubset(set(entry.keys())):
            legacy_rows.append(entry)
        else:
            schema_bad += 1

    checks: List[Dict[str, Any]] = []
    if total == 0:
        checks.append(
            _check(
                "window",
                STATUS_UNKNOWN,
                0,
                0,
                "No production rows in the monitoring window; nothing to assess.",
            )
        )
        return {
            "status": STATUS_UNKNOWN,
            "total_rows": 0,
            "v2_rows": 0,
            "legacy_rows": 0,
            "malformed_rows": 0,
            "checks": checks,
        }

    # 1. Schema (fail on malformed or neither-schema rows).
    bad_schema = malformed + schema_bad
    checks.append(
        _check(
            "schema",
            STATUS_FAIL if bad_schema else STATUS_PASS,
            total,
            bad_schema,
            (
                f"{bad_schema} row(s) are malformed or match neither the v2 "
                "nor the legacy v1 schema."
                if bad_schema
                else "All rows match a known schema (v2 or legacy v1)."
            ),
        )
    )

    # 2. Legacy schema rows (warn: preserved, excluded from drift).
    checks.append(
        _check(
            "legacy_schema",
            STATUS_WARN if legacy_rows else STATUS_PASS,
            total,
            len(legacy_rows),
            (
                f"{len(legacy_rows)} legacy v1 row(s) in the window; preserved "
                "in history and excluded from drift comparison."
                if legacy_rows
                else "No legacy-schema rows in the window."
            ),
        )
    )

    # 3. Missing values in v2 rows (fail).
    missing_rows = sum(
        1 for r in v2_rows if any(_is_missing(r.get(c)) for c in FEATURE_COLUMNS)
    )
    checks.append(
        _check(
            "missing_values",
            STATUS_FAIL if missing_rows else STATUS_PASS,
            len(v2_rows),
            missing_rows,
            (
                f"{missing_rows} v2 row(s) contain missing values."
                if missing_rows
                else "No missing values in v2 rows."
            ),
        )
    )

    # 4. Duplicates among v2 rows (warn: repeats are legitimate traffic).
    try:
        import pandas as pd

        dup_rows = (
            int(pd.DataFrame(v2_rows).duplicated().sum()) if v2_rows else 0
        )
    except Exception:
        seen = set()
        dup_rows = 0
        for r in v2_rows:
            key = tuple(sorted((k, repr(r.get(k))) for k in FEATURE_COLUMNS))
            if key in seen:
                dup_rows += 1
            else:
                seen.add(key)
    checks.append(
        _check(
            "duplicates",
            STATUS_WARN if dup_rows else STATUS_PASS,
            len(v2_rows),
            dup_rows,
            (
                f"{dup_rows} duplicated v2 row(s); repeats can be legitimate "
                "(e.g. controlled samples) so this warns only."
                if dup_rows
                else "No duplicated v2 rows."
            ),
        )
    )

    # 5. Numeric ranges on v2 rows (fail).
    def _numeric_ok(r: Dict[str, Any]) -> bool:
        try:
            area = r.get("area_sqft")
            if not isinstance(area, (int, float)) or isinstance(area, bool):
                return False
            if not math.isfinite(float(area)) or float(area) <= 0:
                return False
            if not _is_int_like(r.get("bedrooms")) or int(r["bedrooms"]) < 1:
                return False
            if not _is_int_like(r.get("bathrooms")) or int(r["bathrooms"]) < 1:
                return False
            if not _is_int_like(r.get("parking")) or not 0 <= int(r["parking"]) <= 3:
                return False
            age = r.get("property_age")
            if not isinstance(age, (int, float)) or isinstance(age, bool):
                return False
            if not math.isfinite(float(age)) or float(age) < 0:
                return False
            if not _is_int_like(r.get("floor_number")) or int(r["floor_number"]) < 0:
                return False
            if not _is_int_like(r.get("total_floors")) or int(r["total_floors"]) < 1:
                return False
            return True
        except (TypeError, ValueError):
            return False

    bad_numeric = sum(1 for r in v2_rows if not _numeric_ok(r))
    checks.append(
        _check(
            "numeric_ranges",
            STATUS_FAIL if bad_numeric else STATUS_PASS,
            len(v2_rows),
            bad_numeric,
            (
                f"{bad_numeric} v2 row(s) violate numeric ranges "
                "(area>0, bedrooms/bathrooms>=1, parking 0-3, age>=0, "
                "floor>=0, total>=1)."
                if bad_numeric
                else "All v2 rows satisfy numeric ranges."
            ),
        )
    )

    # 6. Categorical membership on v2 rows (fail).
    def _categorical_ok(r: Dict[str, Any]) -> bool:
        return (
            isinstance(r.get("location"), str)
            and r["location"] in LOCALITY_RATES
            and isinstance(r.get("furnishing_status"), str)
            and r["furnishing_status"] in FURNISHING_VALUES
            and isinstance(r.get("property_type"), str)
            and r["property_type"] in PROPERTY_TYPE_VALUES
        )

    bad_cat = sum(1 for r in v2_rows if not _categorical_ok(r))
    checks.append(
        _check(
            "categorical_values",
            STATUS_FAIL if bad_cat else STATUS_PASS,
            len(v2_rows),
            bad_cat,
            (
                f"{bad_cat} v2 row(s) carry unknown categorical values "
                "(location/furnishing/property type)."
                if bad_cat
                else "All v2 rows carry known categorical values."
            ),
        )
    )

    # 7. Cross-feature relationships on v2 rows (fail).
    def _relations_ok(r: Dict[str, Any]) -> bool:
        try:
            if not _is_int_like(r.get("floor_number")) or not _is_int_like(
                r.get("total_floors")
            ):
                return False
            if int(r["floor_number"]) > int(r["total_floors"]):
                return False
            if not _is_int_like(r.get("bathrooms")) or not _is_int_like(
                r.get("bedrooms")
            ):
                return False
            if int(r["bathrooms"]) > int(r["bedrooms"]) + 2:
                return False
            return True
        except (TypeError, ValueError):
            return False

    bad_rel = sum(1 for r in v2_rows if not _relations_ok(r))
    checks.append(
        _check(
            "relationships",
            STATUS_FAIL if bad_rel else STATUS_PASS,
            len(v2_rows),
            bad_rel,
            (
                f"{bad_rel} v2 row(s) break building relationships "
                "(floor<=total, bathrooms<=bedrooms+2)."
                if bad_rel
                else "All v2 rows satisfy building relationships."
            ),
        )
    )

    # 8. Outliers via BHK area bands (warn heuristic).
    def _band_ok(r: Dict[str, Any]) -> bool:
        try:
            bhk = r.get("bedrooms")
            area = float(r.get("area_sqft"))
            if not _is_int_like(bhk) or not math.isfinite(area):
                return False
            band = BHK_AREA_BANDS.get(int(bhk))
            if band is None:
                return True  # no documented band: no claim
            lo, hi = band
            return lo <= area <= hi
        except (TypeError, ValueError):
            return False

    outlier_rows = sum(1 for r in v2_rows if not _band_ok(r))
    checks.append(
        _check(
            "outliers",
            STATUS_WARN if outlier_rows else STATUS_PASS,
            len(v2_rows),
            outlier_rows,
            (
                f"{outlier_rows} v2 row(s) fall outside the documented BHK "
                "carpet-area band; flagged for review, not rejected."
                if outlier_rows
                else "All v2 rows sit inside their BHK area band."
            ),
        )
    )

    failed = any(c["status"] == STATUS_FAIL for c in checks)
    warned = any(c["status"] == STATUS_WARN for c in checks)
    status = STATUS_FAIL if failed else (STATUS_WARN if warned else STATUS_PASS)
    logger.info(
        "Data-quality check completed: status=%s v2=%d legacy=%d malformed=%d.",
        status,
        len(v2_rows),
        len(legacy_rows),
        malformed,
    )
    return {
        "status": status,
        "total_rows": total,
        "v2_rows": len(v2_rows),
        "legacy_rows": len(legacy_rows),
        "malformed_rows": malformed,
        "checks": checks,
    }


def summarize_data_quality(result: Dict[str, Any]) -> Dict[str, Any]:
    """Returns a compact additive summary of a data-quality result."""
    checks = result.get("checks", []) if isinstance(result, dict) else []
    passed = sum(1 for c in checks if c.get("status") == STATUS_PASS)
    failed = sum(1 for c in checks if c.get("status") == STATUS_FAIL)
    warned = sum(1 for c in checks if c.get("status") == STATUS_WARN)
    return {
        "status": result.get("status", STATUS_UNKNOWN),
        "checks_run": len(checks),
        "checks_passed": passed,
        "checks_warned": warned,
        "checks_failed": failed,
        "total_rows": result.get("total_rows", 0),
        "v2_rows": result.get("v2_rows", 0),
        "legacy_rows": result.get("legacy_rows", 0),
        "malformed_rows": result.get("malformed_rows", 0),
    }
