"""Deterministic demo traffic scenarios for AdaptiveAI (Phase 6).

Generates simulated House Price (Pune schema v2) samples for faculty
demonstrations of the full MLOps lifecycle (serve → monitor → drift →
retrain → gate → promote) WITHOUT polluting production prediction
history. Every sample is isolated by the caller via
``source="demo"`` + ``demo_session_id``; nothing here writes to any
database, model store, or pointer.

Determinism: every scenario consumes ONLY ``random.Random(seed)`` —
no global random state, no wall-clock sampling — so the same
(scenario, sample_count, seed) always yields byte-identical samples.

Supported scenarios (all use valid v2 categories only):

- ``healthy_baseline``: in-distribution rows resampled from the
  versioned reference CSV. Expected monitoring result: healthy / no
  meaningful drift.
- ``area_shift``: in-distribution rows with ``area_sqft`` redrawn from
  a high band (default 2400–3000 sqft). Expected: numeric area drift.
- ``locality_shift``: rows concentrated in one valid locality
  (default ``Kothrud`` at 90%). Expected: categorical location drift.
- ``config_shift``: valid but shifted configuration mix (mostly
  Fully-Furnished Villas with high BHK). Expected: furnishing /
  property-type / bedroom drift.
- ``data_quality``: controlled, documented quality issues — duplicated
  rows, ``floor_number > total_floors`` violations,
  ``bathrooms > bedrooms + 2`` violations, and BHK-band outliers. All
  values stay type-valid (no NaN) so the real drift detectors still
  run; the real data-quality layer reports the failures honestly.
"""

import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.exceptions import AdaptiveAIError
from app.use_cases.house_price.market_config import (
    FEATURE_COLUMNS,
    FURNISHING_VALUES,
    LOCALITY_RATES,
    PROPERTY_TYPE_VALUES,
)

logger = logging.getLogger(__name__)

SCENARIO_HEALTHY_BASELINE = "healthy_baseline"
SCENARIO_AREA_SHIFT = "area_shift"
SCENARIO_LOCALITY_SHIFT = "locality_shift"
SCENARIO_CONFIG_SHIFT = "config_shift"
SCENARIO_DATA_QUALITY = "data_quality"

SUPPORTED_SCENARIOS = (
    SCENARIO_HEALTHY_BASELINE,
    SCENARIO_AREA_SHIFT,
    SCENARIO_LOCALITY_SHIFT,
    SCENARIO_CONFIG_SHIFT,
    SCENARIO_DATA_QUALITY,
)

SCENARIO_METADATA = {
    SCENARIO_HEALTHY_BASELINE: {
        "name": SCENARIO_HEALTHY_BASELINE,
        "title": "Healthy Baseline",
        "description": "Realistic in-distribution Pune samples resampled from the reference population.",
        "expected": "Healthy — no meaningful drift.",
    },
    SCENARIO_AREA_SHIFT: {
        "name": SCENARIO_AREA_SHIFT,
        "title": "Area Distribution Shift",
        "description": "Mostly high-area properties (2400–3000 sqft) against a normal reference mix.",
        "expected": "Numeric drift on area_sqft.",
    },
    SCENARIO_LOCALITY_SHIFT: {
        "name": SCENARIO_LOCALITY_SHIFT,
        "title": "Locality Concentration Shift",
        "description": "Samples concentrated heavily in one valid locality (default Kothrud).",
        "expected": "Categorical drift on location.",
    },
    SCENARIO_CONFIG_SHIFT: {
        "name": SCENARIO_CONFIG_SHIFT,
        "title": "Property Configuration Shift",
        "description": "Shifted but valid mix: mostly Fully-Furnished Villas with high bedroom counts.",
        "expected": "Categorical/numeric drift on furnishing_status, property_type, bedrooms.",
    },
    SCENARIO_DATA_QUALITY: {
        "name": SCENARIO_DATA_QUALITY,
        "title": "Data Quality Issue",
        "description": "Controlled duplicates, floor/total and bathroom/bedroom violations, BHK-band outliers.",
        "expected": "Data-quality failure (relationships) with duplicate/outlier warnings.",
    },
}

DEFAULT_REFERENCE_CSV = "data/reference/house_price_reference.csv"

_reference_pool: Optional[List[Dict[str, Any]]] = None
_reference_pool_path: Optional[str] = None


def _coerce_row(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Coerces a reference row to JSON-safe v2 python natives."""
    return {
        "area_sqft": float(raw["area_sqft"]),
        "bedrooms": int(raw["bedrooms"]),
        "bathrooms": int(raw["bathrooms"]),
        "parking": int(raw["parking"]),
        "location": str(raw["location"]),
        "property_age": float(raw["property_age"]),
        "floor_number": int(raw["floor_number"]),
        "total_floors": int(raw["total_floors"]),
        "furnishing_status": str(raw["furnishing_status"]),
        "property_type": str(raw["property_type"]),
    }


def load_reference_pool(reference_csv_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """Loads the reference population as v2 sample dicts (cached per path).

    Rows keep CSV order so indexed sampling is deterministic. Read-only:
    the reference dataset is never modified.
    """
    global _reference_pool, _reference_pool_path
    resolved = reference_csv_path or DEFAULT_REFERENCE_CSV
    if _reference_pool is not None and _reference_pool_path == resolved:
        return _reference_pool
    try:
        from app.core.config import resolve_project_path

        path = Path(resolve_project_path(resolved))
    except Exception:
        path = Path(resolved)
    if not path.is_file():
        raise AdaptiveAIError(f"Reference dataset not found: {path}")
    try:
        import pandas as pd

        df = pd.read_csv(str(path))
    except Exception as e:
        raise AdaptiveAIError(f"Failed to read reference dataset at {path}: {e}") from e
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise AdaptiveAIError(
            f"Reference dataset is missing required columns: {', '.join(missing)}"
        )
    pool = [_coerce_row(dict(row)) for _, row in df[FEATURE_COLUMNS].iterrows()]
    if not pool:
        raise AdaptiveAIError("Reference dataset holds no samplable rows.")
    _reference_pool = pool
    _reference_pool_path = resolved
    return pool


def _validate_generation_args(
    scenario: str, sample_count: Any, seed: Any
) -> int:
    """Validates scenario/count/seed, returning a clean sample count."""
    if scenario not in SUPPORTED_SCENARIOS:
        raise AdaptiveAIError(
            "Unknown demo scenario "
            f"'{scenario}'. Supported: {', '.join(SUPPORTED_SCENARIOS)}."
        )
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise AdaptiveAIError("sample_count must be an integer.")
    if sample_count < 1 or sample_count > 1000:
        raise AdaptiveAIError("sample_count must be between 1 and 1000.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise AdaptiveAIError("seed must be a non-negative integer.")
    return int(sample_count)


def generate_scenario_samples(
    scenario: str,
    sample_count: int,
    seed: int,
    params: Optional[Dict[str, Any]] = None,
    reference_csv_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generates deterministic demo samples for one scenario.

    Args:
        scenario: One of ``SUPPORTED_SCENARIOS``.
        sample_count: Rows to generate (1–1000).
        seed: Deterministic seed; identical inputs reproduce identical
            outputs.
        params: Optional scenario tuning (area band, locality,
            concentration). Unknown keys raise.
        reference_csv_path: Reference CSV override (tests only).

    Returns:
        List of v2 input dicts (JSON-safe python natives). No database
        or model access; persistence is the caller's job (isolated via
        ``source="demo"``).
    """
    count = _validate_generation_args(scenario, sample_count, seed)
    tunables = dict(params) if params is not None else {}
    if not isinstance(tunables, dict):
        raise AdaptiveAIError("params must be a mapping or None.")
    pool = load_reference_pool(reference_csv_path)
    rng = random.Random(int(seed))
    if scenario == SCENARIO_HEALTHY_BASELINE:
        _reject_unknown_params(tunables, set())
        return [_baseline_row(pool, rng) for _ in range(count)]
    if scenario == SCENARIO_AREA_SHIFT:
        return _area_shift_rows(pool, rng, count, tunables)
    if scenario == SCENARIO_LOCALITY_SHIFT:
        return _locality_shift_rows(pool, rng, count, tunables)
    if scenario == SCENARIO_CONFIG_SHIFT:
        _reject_unknown_params(tunables, set())
        return [_config_shift_row(pool, rng) for _ in range(count)]
    _reject_unknown_params(tunables, set())
    return _data_quality_rows(pool, rng, count)


def _reject_unknown_params(params: Dict[str, Any], allowed: set) -> None:
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise AdaptiveAIError(
            f"Unknown scenario params: {', '.join(unknown)}."
        )


def _baseline_row(pool: List[Dict[str, Any]], rng: random.Random) -> Dict[str, Any]:
    """One in-distribution row (uniform resample with replacement)."""
    return dict(pool[rng.randrange(len(pool))])


def _area_shift_rows(
    pool: List[Dict[str, Any]],
    rng: random.Random,
    count: int,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    _reject_unknown_params(params, {"area_min", "area_max"})
    area_min = float(params.get("area_min", 2400.0))
    area_max = float(params.get("area_max", 3000.0))
    if not area_min > 0 or not area_max > area_min:
        raise AdaptiveAIError("area_min/area_max must satisfy 0 < min < max.")
    rows = []
    for _ in range(count):
        row = _baseline_row(pool, rng)
        row["area_sqft"] = round(rng.uniform(area_min, area_max), 1)
        rows.append(row)
    return rows


def _locality_shift_rows(
    pool: List[Dict[str, Any]],
    rng: random.Random,
    count: int,
    params: Dict[str, Any],
) -> List[Dict[str, Any]]:
    _reject_unknown_params(params, {"locality", "concentration"})
    locality = str(params.get("locality", "Kothrud"))
    if locality not in LOCALITY_RATES:
        raise AdaptiveAIError(f"Unknown locality for demo scenario: {locality}")
    concentration = float(params.get("concentration", 0.9))
    if not 0.0 < concentration <= 1.0:
        raise AdaptiveAIError("concentration must be in (0, 1].")
    rows = []
    for _ in range(count):
        row = _baseline_row(pool, rng)
        if rng.random() < concentration:
            row["location"] = locality
        rows.append(row)
    return rows


def _config_shift_row(pool: List[Dict[str, Any]], rng: random.Random) -> Dict[str, Any]:
    """Shifted-but-valid configuration mix (deterministic per seed)."""
    row = _baseline_row(pool, rng)
    if rng.random() < 0.80:
        row["furnishing_status"] = "Fully-Furnished"
    if rng.random() < 0.70:
        row["property_type"] = "Villa"
    if rng.random() < 0.75:
        row["bedrooms"] = 5
        row["bathrooms"] = rng.choice([4, 5])
    return row


def _data_quality_rows(
    pool: List[Dict[str, Any]], rng: random.Random, count: int
) -> List[Dict[str, Any]]:
    """Controlled quality issues, all type-valid (detectors still run).

    Mix (deterministic): 40% exact duplicates of one valid row
    (duplicates warning), 30% ``floor_number > total_floors``
    violations (relationships failure), 20% ``bathrooms > bedrooms +
    2`` violations (relationships failure), 10% 1-BHK 2600 sqft
    outliers (outlier warning). Categories always stay valid.
    """
    anchor = dict(pool[rng.randrange(len(pool))])
    rows: List[Dict[str, Any]] = []
    for i in range(count):
        slot = (i * 10) // max(count, 1)
        if slot < 4:
            rows.append(dict(anchor))
            continue
        row = _baseline_row(pool, rng)
        if slot < 7:
            total = rng.randint(1, 6)
            row["total_floors"] = total
            row["floor_number"] = total + rng.randint(1, 3)
        elif slot < 9:
            bedrooms = rng.randint(1, 3)
            row["bedrooms"] = bedrooms
            row["bathrooms"] = bedrooms + 3
        else:
            row["bedrooms"] = 1
            row["bathrooms"] = 1
            row["area_sqft"] = round(rng.uniform(2400.0, 2800.0), 1)
        rows.append(row)
    return rows


def describe_scenarios() -> List[Dict[str, str]]:
    """Returns UI-ready metadata for every supported scenario."""
    return [dict(SCENARIO_METADATA[name]) for name in SUPPORTED_SCENARIOS]
