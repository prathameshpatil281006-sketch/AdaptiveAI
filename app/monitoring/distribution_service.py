"""Distribution service for Monitoring page (Phase 4).

Provides reference vs production distributions for the empirical
distribution comparison chart. Built on top of the
MonitoringDataProvider full-schema frames so reference/production
sourcing stays identical to drift detection.

Numeric features: shared-range histograms plus mean/std/count (as
before; response shape for numeric entries is unchanged apart from
an added ``type: "numeric"`` marker).
Categorical features: per-category reference/production counts and
shares (no fabricated bins; shares are exact observed fractions).

Empty production windows yield zero counts and ``sufficient_data``
False explicitly — never fabricated bars.

No API or HTTP work lives here. The FastAPI layer calls
``get_house_price_distributions`` as a thin adapter.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


def _histogram(values: List[float], bins: int = 20) -> Dict[str, Any]:
    """Computes a shared-range histogram for one value list."""
    import numpy as np

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"bins": [], "counts": []}
    try:
        counts, edges = np.histogram(arr, bins=bins)
        return {
            "bins": [float(x) for x in edges.tolist()],
            "counts": [int(x) for x in counts.tolist()],
        }
    except Exception:
        return {"bins": [], "counts": []}


def get_house_price_distributions(
    production_limit: int = 1000,
    bins: int = 20,
    db_manager: Any = None,
    repository: Any = None,
    source: Optional[str] = "production",
    demo_session_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Returns per-feature reference/production distributions.

    Args:
        production_limit: Window size for production history.
        bins: Histogram bins for numeric features.
        db_manager: Optional DatabaseManager override (tests only).
            Defaults to the configured database.
        repository: Optional PredictionRepository override (tests
            only). Takes precedence over ``db_manager``.
        source: Traffic class observed. Defaults to ``"production"``
            (Phase 6: demo rows never leak into production charts);
            pass ``"demo"`` with ``demo_session_id`` for simulator
            views, or None for the legacy unfiltered window.

    Returns:
        Dict with features (numeric entries keep their existing
        shape plus ``type``; categorical entries carry ``type`` and
        ``categories`` with per-category counts/shares),
        production_rows, reference_rows, and ``sufficient_data``
        (False when the production window holds no v2 rows).
    """
    from app.core.exceptions import AdaptiveAIError
    from app.database.database import DatabaseManager
    from app.database.repository import PredictionRepository
    from app.monitoring.data_provider import create_house_price_provider

    if isinstance(production_limit, bool) or not isinstance(production_limit, int):
        raise AdaptiveAIError("production_limit must be a positive integer.")
    if production_limit < 1:
        raise AdaptiveAIError("production_limit must be a positive integer.")

    try:
        if repository is None:
            db_manager = db_manager if db_manager is not None else DatabaseManager()
            repository = PredictionRepository(db_manager)
        provider = create_house_price_provider(
            repository=repository, source=source, demo_session_id=demo_session_id
        )
        data = provider.get_full_monitoring_data(limit=production_limit)
        reference: pd.DataFrame = data["reference_full"]
        production: pd.DataFrame = data["production_full"]
    except Exception as e:
        raise AdaptiveAIError(f"Failed to load distribution data: {e}") from e

    try:
        numeric_cols = list(provider.monitored_feature_names)
    except Exception:
        numeric_cols = []
    return build_distributions(reference, production, numeric_cols, bins=bins)


def build_distributions(
    reference: "pd.DataFrame",
    production: "pd.DataFrame",
    numeric_cols: List[str],
    bins: int = 20,
) -> Dict[str, Any]:
    """Builds reference/production distributions from full frames.

    Pure function of DataFrames (injectable for tests): numeric
    columns become histogram entries, every other column becomes a
    categorical entry. No database or filesystem access.
    """
    import numpy as np
    import pandas as pd

    features: List[Dict[str, Any]] = []
    try:
        feature_names = list(reference.columns)
    except Exception:
        feature_names = []
    numeric_set = set(numeric_cols or [])

    for name in feature_names:
        if name in numeric_set:
            entry = _numeric_distribution(reference, production, name, bins)
            if entry is not None:
                features.append(entry)
        else:
            try:
                features.append(
                    _categorical_distribution(reference, production, name)
                )
            except Exception as e:
                logger.warning(f"Failed to build distribution for {name}: {e}")

    return {
        "features": features,
        "reference_rows": int(len(reference)),
        "production_rows": int(len(production)),
        "sufficient_data": bool(len(production) > 0),
    }


def _numeric_distribution(
    reference: "pd.DataFrame", production: "pd.DataFrame", name: str, bins: int
) -> Optional[Dict[str, Any]]:
    """Builds one numeric histogram entry (pre-existing shape + type)."""
    import numpy as np
    import pandas as pd

    try:
        ref_vals = pd.to_numeric(reference[name], errors="coerce").dropna().to_numpy(dtype=float)
        prod_vals = (
            pd.to_numeric(production[name], errors="coerce").dropna().to_numpy(dtype=float)
            if name in production.columns and len(production) > 0
            else np.asarray([], dtype=float)
        )
        ref_vals = ref_vals[np.isfinite(ref_vals)]
        prod_vals = prod_vals[np.isfinite(prod_vals)] if prod_vals.size else prod_vals

        # Shared bin edges so both histograms are comparable.
        combined = np.concatenate([ref_vals, prod_vals]) if prod_vals.size else ref_vals
        if combined.size == 0:
            ref_hist: Dict[str, Any] = {"bins": [], "counts": []}
            prod_hist: Dict[str, Any] = {"bins": [], "counts": []}
        else:
            try:
                _, edges = np.histogram(combined, bins=bins)
                edges_list = [float(x) for x in edges.tolist()]
                r_counts, _ = np.histogram(ref_vals, bins=edges)
                if prod_vals.size:
                    p_counts, _ = np.histogram(prod_vals, bins=edges)
                else:
                    p_counts = np.zeros(len(edges) - 1, dtype=int)
                ref_hist = {
                    "bins": edges_list,
                    "counts": [int(x) for x in r_counts.tolist()],
                }
                prod_hist = {
                    "bins": edges_list,
                    "counts": [int(x) for x in p_counts.tolist()],
                }
            except Exception:
                ref_hist = _histogram(ref_vals.tolist(), bins=bins)
                prod_hist = _histogram(prod_vals.tolist(), bins=bins)

        def _stats(arr: Any) -> Dict[str, Any]:
            try:
                if arr.size == 0:
                    return {"mean": None, "std": None, "count": 0}
                return {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                    "count": int(arr.size),
                }
            except Exception:
                return {"mean": None, "std": None, "count": 0}

        ref_stats = _stats(ref_vals)
        prod_stats = _stats(prod_vals)
        return {
            "feature": str(name),
            "type": "numeric",
            "reference": {**ref_stats, "histogram": ref_hist},
            "production": {**prod_stats, "histogram": prod_hist},
        }
    except Exception as e:
        logger.warning(f"Failed to build distribution for {name}: {e}")
        return None


def _categorical_distribution(
    reference: "pd.DataFrame", production: "pd.DataFrame", name: str
) -> Dict[str, Any]:
    """Builds one categorical entry with per-category counts/shares."""
    ref_series = reference[name]
    prod_series = production[name] if name in production.columns else []

    def _counts(values: Any) -> Dict[str, int]:
        tally: Dict[str, int] = {}
        try:
            items = values.tolist() if hasattr(values, "tolist") else list(values or [])
        except Exception:
            return tally
        for value in items:
            if value is None:
                continue
            key = str(value)
            if key in ("", "nan", "None", "NaT"):
                continue
            tally[key] = tally.get(key, 0) + 1
        return tally

    ref_counts = _counts(ref_series)
    prod_counts = _counts(prod_series)
    ref_total = sum(ref_counts.values())
    prod_total = sum(prod_counts.values())
    categories = []
    for value in sorted(set(ref_counts) | set(prod_counts)):
        ref_c = int(ref_counts.get(value, 0))
        prod_c = int(prod_counts.get(value, 0))
        categories.append(
            {
                "value": value,
                "reference_count": ref_c,
                "reference_share": (ref_c / ref_total) if ref_total else 0.0,
                "production_count": prod_c,
                "production_share": (prod_c / prod_total) if prod_total else 0.0,
            }
        )
    return {
        "feature": str(name),
        "type": "categorical",
        "categories": categories,
        "reference_rows": int(ref_total),
        "production_rows": int(prod_total),
    }
