"""Categorical drift detector for AdaptiveAI (Phase 4).

Compares production vs reference distributions for categorical
features (``location``, ``furnishing_status``, ``property_type``)
with the chi-square test of homogeneity on a 2 x K contingency table
(reference row vs production row over the union of observed
categories).

Why chi-square (and NOT KS on encoded integers):
- Categorical values have no order or metric; encoding them as
  arbitrary integers and running a KS test would manufacture a false
  ordering and meaningless distances. The chi-square test works
  directly on category counts with no encoding.

Validity handling (documented):
- Categories are pooled over the union of reference and production
  values, so unseen categories (production-only) and missing
  categories (reference-only, production count 0) are both handled.
- Buckets with expected counts below ``MIN_EXPECTED_COUNT`` (5.0,
  the standard chi-square rule of thumb) are merged into a single
  ``OTHER_RARE`` bucket so the chi-square approximation stays valid.
- If fewer than 2 buckets remain after merging, the feature cannot
  be tested: it is reported with ``status="insufficient"``,
  ``drifted=False``, and null statistics (conservative: no drift
  claim on untestable data).
- Production windows smaller than ``min_production_rows`` return the
  whole result with ``status="insufficient_data"`` and empty
  features — the same convention as the numeric KS detector.

Multiple testing: Holm-Bonferroni step-down (shared
``holm_adjusted_pvalues``) is applied WITHIN the categorical family
only. The numeric KS family keeps its own existing correction, so
numeric behavior is unchanged.

Effect size: Population Stability Index (PSI) over the UNMERGED union
of categories, plus Cramer's V on the merged test table as
supplementary evidence. PSI needs no bucketing (zero counts are
handled with additive smoothing), so it keeps full locality
resolution even for small production windows where chi-square
bucket-merging would dilute the signal. Conventional PSI bands
(>= 0.25 large shift, 0.10-0.25 moderate) drive severity; the
chi-square p-value alone drives the drifted True/False decision.

The detector accepts DataFrames/Series only. It never reads CSV
files, SQLite, MLflow, or House Price-specific code.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from app.core.exceptions import AdaptiveAIError
from app.monitoring.drift_detector import (
    DEFAULT_ALPHA,
    DEFAULT_MIN_PRODUCTION_ROWS,
    holm_adjusted_pvalues,
)

logger = logging.getLogger(__name__)

CATEGORICAL_METHOD = "chi-square"
CATEGORICAL_CORRECTION = "holm-bonferroni"

# Standard chi-square rule of thumb: every bucket should have an
# expected count of at least 5 for the approximation to hold.
MIN_EXPECTED_COUNT = 5.0

RARE_BUCKET_LABEL = "OTHER_RARE"


def _category_counts(values: pd.Series) -> Dict[str, int]:
    """Counts stringified categories, dropping nulls (nulls are a data-quality issue, not drift)."""
    counts: Dict[str, int] = {}
    for value in values.tolist():
        if value is None:
            continue
        try:
            if isinstance(value, float) and math.isnan(value):
                continue
        except (TypeError, ValueError):
            pass
        key = str(value)
        if key in ("", "nan", "None", "NaT"):
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _merge_rare_categories(
    ref_counts: Dict[str, int], prod_counts: Dict[str, int]
) -> List[str]:
    """Returns the bucket list with rare categories merged.

    A category is rare when its expected count under homogeneity is
    below MIN_EXPECTED_COUNT. All rare categories merge into one
    OTHER_RARE bucket.
    """
    categories = sorted(set(ref_counts) | set(prod_counts))
    ref_total = sum(ref_counts.values())
    prod_total = sum(prod_counts.values())
    grand = ref_total + prod_total
    if grand == 0:
        return []
    keep: List[str] = []
    rare = False
    for cat in categories:
        col_total = ref_counts.get(cat, 0) + prod_counts.get(cat, 0)
        # Minimum expected cell count for this column across both rows.
        min_expected = min(ref_total, prod_total) * col_total / grand
        if min_expected < MIN_EXPECTED_COUNT:
            rare = True
        else:
            keep.append(cat)
    if rare:
        keep.append(RARE_BUCKET_LABEL)
    return keep


def _bucket_table(
    ref_counts: Dict[str, int], prod_counts: Dict[str, int], buckets: List[str]
) -> np.ndarray:
    """Builds the 2 x K observed-count table for the bucket list."""
    rare_ref = sum(v for k, v in ref_counts.items() if k not in buckets)
    rare_prod = sum(v for k, v in prod_counts.items() if k not in buckets)
    table = []
    for row_counts, rare in ((ref_counts, rare_ref), (prod_counts, rare_prod)):
        row = [int(row_counts.get(cat, 0)) for cat in buckets if cat != RARE_BUCKET_LABEL]
        if RARE_BUCKET_LABEL in buckets:
            row.append(int(rare))
        table.append(row)
    return np.asarray(table, dtype=float)


def cramers_v(chi2: float, n: int) -> float:
    """Cramer's V for a 2 x K table: sqrt(chi2 / n)."""
    if n <= 0 or not math.isfinite(chi2) or chi2 < 0:
        return 0.0
    return float(math.sqrt(chi2 / n))


# Additive smoothing mass for PSI (keeps shares proper and finite
# when a category is unobserved on one side).
PSI_SMOOTHING_EPSILON = 1e-4


def population_stability_index(
    ref_counts: Dict[str, int], prod_counts: Dict[str, int]
) -> float:
    """Computes PSI between reference and production category shares.

    PSI = sum((prod_share - ref_share) * ln(prod_share / ref_share))
    over the union of categories, with additive smoothing so unseen
    categories stay finite. Non-negative; 0 means identical shares.
    """
    categories = sorted(set(ref_counts) | set(prod_counts))
    if not categories:
        return 0.0
    ref_total = sum(ref_counts.values())
    prod_total = sum(prod_counts.values())
    if ref_total <= 0 or prod_total <= 0:
        return 0.0
    k = len(categories)
    psi = 0.0
    for cat in categories:
        ref_share = (ref_counts.get(cat, 0) + PSI_SMOOTHING_EPSILON) / (
            ref_total + PSI_SMOOTHING_EPSILON * k
        )
        prod_share = (prod_counts.get(cat, 0) + PSI_SMOOTHING_EPSILON) / (
            prod_total + PSI_SMOOTHING_EPSILON * k
        )
        psi += (prod_share - ref_share) * math.log(prod_share / ref_share)
    if not math.isfinite(psi) or psi < 0:
        return 0.0
    return float(psi)


class CategoricalDriftDetector:
    """Chi-square drift detector for categorical feature DataFrames."""

    def __init__(
        self,
        alpha: float = DEFAULT_ALPHA,
        min_production_rows: int = DEFAULT_MIN_PRODUCTION_ROWS,
    ) -> None:
        """Initializes the detector with configurable thresholds."""
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise AdaptiveAIError("alpha must be a number in (0, 1).")
        alpha = float(alpha)
        if not 0.0 < alpha < 1.0:
            raise AdaptiveAIError("alpha must be in (0, 1).")
        if isinstance(min_production_rows, bool) or not isinstance(
            min_production_rows, int
        ):
            raise AdaptiveAIError("min_production_rows must be a positive integer.")
        if min_production_rows < 1:
            raise AdaptiveAIError("min_production_rows must be a positive integer.")
        self._alpha = alpha
        self._min_production_rows = int(min_production_rows)

    @property
    def alpha(self) -> float:
        """Returns the configured significance level."""
        return self._alpha

    @property
    def min_production_rows(self) -> int:
        """Returns the minimum production sample size."""
        return self._min_production_rows

    def detect(
        self,
        reference: pd.DataFrame,
        production: pd.DataFrame,
        features: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Compares production vs reference category distributions.

        Args:
            reference: Reference feature DataFrame (must be non-empty).
            production: Production feature DataFrame.
            features: Optional explicit categorical feature list. When
                None, the reference columns are used.

        Returns:
            Serializable dict with overall_drifted, alpha,
            min_production_rows, production_rows, sufficient_data,
            status ("ok" or "insufficient_data"), method
            ("chi-square"), correction ("holm-bonferroni"), and
            per-feature {feature_name, type ("categorical"), method,
            statistic (chi2), p_value, p_value_corrected, dof,
            effect_size (PSI), cramers_v, drifted, status
            ("ok"/"insufficient")}.
        """
        if not isinstance(reference, pd.DataFrame):
            raise AdaptiveAIError("reference must be a pandas DataFrame.")
        if not isinstance(production, pd.DataFrame):
            raise AdaptiveAIError("production must be a pandas DataFrame.")

        if features is None:
            feature_names = list(reference.columns)
        else:
            if not isinstance(features, list) or not features:
                raise AdaptiveAIError("features must be a non-empty list when provided.")
            feature_names = list(features)
            missing_ref = [c for c in feature_names if c not in reference.columns]
            missing_prod = [c for c in feature_names if c not in production.columns]
            if missing_ref or missing_prod:
                raise AdaptiveAIError(
                    "Required feature columns missing: "
                    f"missing_in_reference={missing_ref}, "
                    f"missing_in_production={missing_prod}."
                )

        if len(reference) == 0:
            raise AdaptiveAIError("Reference data must not be empty.")
        if len(feature_names) == 0:
            raise AdaptiveAIError("No features available for comparison.")

        production_rows = int(len(production))
        if production_rows < self._min_production_rows:
            logger.info(
                "Insufficient production data for categorical drift detection: "
                f"{production_rows} < {self._min_production_rows}."
            )
            return {
                "overall_drifted": False,
                "alpha": float(self._alpha),
                "method": CATEGORICAL_METHOD,
                "correction": CATEGORICAL_CORRECTION,
                "min_production_rows": int(self._min_production_rows),
                "production_rows": production_rows,
                "sufficient_data": False,
                "status": "insufficient_data",
                "features": [],
            }

        raw_pvalues: List[float] = []
        tested: List[Dict[str, Any]] = []
        for name in feature_names:
            ref_counts = _category_counts(reference[name])
            prod_counts = _category_counts(production[name])
            buckets = _merge_rare_categories(ref_counts, prod_counts)
            if len(buckets) < 2:
                tested.append(
                    {
                        "feature_name": str(name),
                        "type": "categorical",
                        "method": CATEGORICAL_METHOD,
                        "statistic": None,
                        "p_value": None,
                        "dof": None,
                        "effect_size": None,
                        "drifted": False,
                        "status": "insufficient",
                        "buckets": int(len(buckets)),
                    }
                )
                raw_pvalues.append(1.0)
                continue
            table = _bucket_table(ref_counts, prod_counts, buckets)
            try:
                chi2, p_value, dof, _ = chi2_contingency(table, correction=False)
                chi2 = float(chi2)
                p_value = float(p_value)
                dof = int(dof)
            except Exception:
                chi2, p_value, dof = 0.0, 1.0, 0
            if not math.isfinite(chi2) or chi2 < 0:
                chi2, p_value = 0.0, 1.0
            if not math.isfinite(p_value) or p_value < 0.0 or p_value > 1.0:
                p_value = 1.0
            n = int(table.sum())
            psi = population_stability_index(ref_counts, prod_counts)
            tested.append(
                {
                    "feature_name": str(name),
                    "type": "categorical",
                    "method": CATEGORICAL_METHOD,
                    "statistic": chi2,
                    "p_value": p_value,
                    "dof": dof,
                    "effect_size": psi,
                    "cramers_v": cramers_v(chi2, n),
                    "drifted": False,  # decided below from adjusted p-values
                    "status": "ok",
                    "buckets": int(len(buckets)),
                }
            )
            raw_pvalues.append(p_value)

        adjusted = holm_adjusted_pvalues(raw_pvalues)
        results: List[Dict[str, Any]] = []
        for idx, entry in enumerate(tested):
            if entry["status"] == "ok" and entry["p_value"] is not None:
                p_corrected = float(adjusted[idx])
                entry["p_value_corrected"] = p_corrected
                entry["drifted"] = bool(p_corrected < self._alpha)
            else:
                entry["p_value_corrected"] = None
            results.append(entry)

        overall = bool(any(r["drifted"] for r in results))
        return {
            "overall_drifted": overall,
            "alpha": float(self._alpha),
            "method": CATEGORICAL_METHOD,
            "correction": CATEGORICAL_CORRECTION,
            "min_production_rows": int(self._min_production_rows),
            "production_rows": production_rows,
            "sufficient_data": True,
            "status": "ok",
            "features": results,
        }
