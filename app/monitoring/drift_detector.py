"""Statistical drift detector for AdaptiveAI (Steps 26 + 62).

Compares production feature distributions against reference feature
distributions using the two-sample Kolmogorov-Smirnov (KS) test
(``scipy.stats.ks_2samp``).

Why KS:
- Non-parametric: no normality assumption about House Price features.
- Works on continuous numeric features with unequal sample sizes.
- Returns an interpretable statistic (max CDF distance) and p-value.

Decision rule (Step 62, I-13 — Holm-Bonferroni, documented):
    Per-feature raw p-values come from the two-sided KS test. Across
    the m tested features, Holm step-down adjusted p-values are
    computed from the p-values sorted ascending p_(1) <= ... <= p_(m):

        p_adj_(i) = max_{j<=i} min(1, (m - j + 1) * p_(j))

    A feature is drifted when ``p_adj < alpha`` (default alpha=0.05),
    and overall drift holds when ANY feature is drifted under the
    adjusted values. With m == 1 this reduces exactly to the legacy
    ``p < alpha`` rule. The family-wise error rate is controlled at
    alpha without any dependence assumptions between features.

Non-finite or out-of-range raw p-values (NaN/inf, never expected from
``ks_2samp``) are sanitized to 1.0 (conservative: no drift claim) so a
single bad value can neither crash detection nor force a false alarm.

The detector accepts DataFrames only. It never reads CSV files,
SQLite, MLflow, or House Price-specific code, so it works with any
compatible numeric DataFrames.
"""

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from app.core.exceptions import AdaptiveAIError

logger = logging.getLogger(__name__)

DEFAULT_ALPHA = 0.05
DEFAULT_MIN_PRODUCTION_ROWS = 30

CORRECTION_METHOD = "holm-bonferroni"


def holm_adjusted_pvalues(p_values: List[float]) -> List[float]:
    """Computes Holm-Bonferroni step-down adjusted p-values.

    Args:
        p_values: Raw per-feature p-values in original feature order.

    Returns:
        Adjusted p-values in the same order, each in [0, 1].
        Sorting is stable (ties broken by original index), so the
        output is fully deterministic for a given input.

    Non-finite or out-of-range inputs are sanitized to 1.0
    (conservative: never a drift claim on a bad value).
    """
    m = len(p_values)
    safe: List[float] = []
    for p in p_values:
        try:
            f = float(p)
        except (TypeError, ValueError):
            f = 1.0
        if not math.isfinite(f) or f < 0.0 or f > 1.0:
            if isinstance(f, float) and math.isfinite(f):
                f = min(1.0, max(0.0, f))
            else:
                f = 1.0
        safe.append(f)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: (safe[i], i))
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        val = safe[idx] * (m - rank)
        if val > 1.0:
            val = 1.0
        if val < running_max:
            val = running_max
        else:
            running_max = val
        adjusted[idx] = float(val)
    return adjusted


class DriftDetector:
    """KS-based drift detector for numeric feature DataFrames."""

    def __init__(
        self, alpha: float = DEFAULT_ALPHA, min_production_rows: int = DEFAULT_MIN_PRODUCTION_ROWS
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
        """Compares production vs reference distributions per feature.

        Args:
            reference: Reference feature DataFrame (must be non-empty).
            production: Production feature DataFrame.
            features: Optional explicit feature list. When None, the
                reference columns are used and production must contain
                exactly the same set (no silent mismatches).

        Returns:
            Serializable dict with overall_drifted, alpha,
            min_production_rows, production_rows, sufficient_data,
            status ("ok" or "insufficient_data"), correction
            ("holm-bonferroni"), and per-feature
            {feature_name, statistic, p_value, p_value_corrected,
            drifted}.

            ``p_value`` is the raw two-sided KS p-value; ``drifted``
            is decided from the Holm-adjusted value
            (``p_value_corrected < alpha``). With a single feature the
            adjusted value equals the raw value, preserving legacy
            single-test behavior exactly.

            Insufficient production data (rows < minimum, including
            empty) returns overall_drifted=False with empty features
            and status "insufficient_data"; no p-values are manufactured.
            This is distinguishable from a healthy "no drift" result,
            which has status "ok" and populated features.
        """
        if not isinstance(reference, pd.DataFrame):
            raise AdaptiveAIError("reference must be a pandas DataFrame.")
        if not isinstance(production, pd.DataFrame):
            raise AdaptiveAIError("production must be a pandas DataFrame.")

        if features is None:
            feature_names = list(reference.columns)
            if list(production.columns) != [] and set(production.columns) != set(
                reference.columns
            ):
                raise AdaptiveAIError(
                    "Reference and production feature sets must match: "
                    f"reference={sorted(reference.columns)}, "
                    f"production={sorted(production.columns)}."
                )
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
                "Insufficient production data for drift detection: "
                f"{production_rows} < {self._min_production_rows}."
            )
            return {
                "overall_drifted": False,
                "alpha": float(self._alpha),
                "min_production_rows": int(self._min_production_rows),
                "production_rows": production_rows,
                "sufficient_data": False,
                "status": "insufficient_data",
                "features": [],
            }

        results: List[Dict[str, Any]] = []
        raw_pvalues: List[float] = []
        statistics: List[float] = []
        for name in feature_names:
            ref_values = self._validated_numeric_values(reference[name], name, "reference")
            prod_values = self._validated_numeric_values(production[name], name, "production")
            outcome = ks_2samp(ref_values, prod_values, alternative="two-sided")
            try:
                statistic = float(outcome.statistic)
            except (TypeError, ValueError):
                statistic = 0.0
            if not math.isfinite(statistic):
                statistic = 0.0
            try:
                p_value = float(outcome.pvalue)
            except (TypeError, ValueError):
                p_value = 1.0
            if not math.isfinite(p_value) or p_value < 0.0 or p_value > 1.0:
                p_value = 1.0 if not math.isfinite(p_value) else min(1.0, max(0.0, p_value))
            statistics.append(statistic)
            raw_pvalues.append(p_value)

        adjusted = holm_adjusted_pvalues(raw_pvalues)
        for idx, name in enumerate(feature_names):
            p_corrected = float(adjusted[idx])
            drifted = bool(p_corrected < self._alpha)
            results.append(
                {
                    "feature_name": str(name),
                    "statistic": statistics[idx],
                    "p_value": raw_pvalues[idx],
                    "p_value_corrected": p_corrected,
                    "drifted": drifted,
                }
            )

        overall = bool(any(r["drifted"] for r in results))
        return {
            "overall_drifted": overall,
            "alpha": float(self._alpha),
            "correction": CORRECTION_METHOD,
            "min_production_rows": int(self._min_production_rows),
            "production_rows": production_rows,
            "sufficient_data": True,
            "status": "ok",
            "features": results,
        }

    @staticmethod
    def _validated_numeric_values(series: pd.Series, name: str, side: str) -> np.ndarray:
        """Validates a feature column and returns finite float values."""
        if series.dtype.kind not in ("i", "u", "f"):
            raise AdaptiveAIError(
                f"Feature '{name}' in {side} data must be numeric, "
                f"got dtype '{series.dtype}'."
            )
        values = series.to_numpy(dtype=float)
        if values.size == 0:
            raise AdaptiveAIError(f"Feature '{name}' in {side} data must not be empty.")
        if not bool(np.all(np.isfinite(values))):
            raise AdaptiveAIError(
                f"Feature '{name}' in {side} data contains NaN or infinite values."
            )
        return values
