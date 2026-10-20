"""Candidate-vs-production model evaluation for AdaptiveAI (Step 31).

Evaluation-only component that compares a retrained candidate model
against the currently deployed/production model on the SAME evaluation
dataset::

    RetrainingService
        |
        v
    produces candidate (best_pipeline / best_model_name)
        |
        v
    ModelEvaluator
        |
        v
    compares candidate vs production (no promotion, no persistence)

House Price (regression) is the first supported implementation. The
primary selection metric is R2. The decision rule is strictly::

    candidate_better = candidate_r2 > production_r2
    improvement = candidate_r2 - production_r2

The evaluator has no side effects: it never saves, overwrites,
registers, or promotes models, and never touches SQLite or the MLflow
registry. Production loading reuses the existing House Price persistence
utility (:func:`load_house_price_model`); no joblib logic is duplicated
here.
"""

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.core.exceptions import ModelNotFoundError, RetrainingError
from app.use_cases.house_price.model_persistence import load_house_price_model

logger = logging.getLogger(__name__)

PRIMARY_METRIC = "r2"

CANDIDATE_BETTER = "candidate_better"
PRODUCTION_BETTER = "production_better"
NO_PRODUCTION_MODEL = "no_production_model"


def compute_regression_metrics(y_true: Any, y_pred: Any) -> Dict[str, float]:
    """Computes MAE/RMSE/R2 following the existing MLEngine conventions.

    Args:
        y_true: Ground-truth targets.
        y_pred: Predicted targets.

    Returns:
        Dict with ``mae``, ``rmse``, ``r2`` as Python floats.
    """
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _default_production_loader(
    model_name: str, models_dir: Optional[Union[str, Path]] = None
) -> Any:
    """Loads a production model reusing the House Price persistence utility."""
    if models_dir is None:
        return load_house_price_model(model_name)
    return load_house_price_model(model_name, models_dir=models_dir)


class ModelEvaluator:
    """Compares a candidate model against a production model on identical data.

    The evaluator is use-case oriented but generic in its core: it evaluates
    two already-trained sklearn Pipelines on the same ``X_test``/``y_test``
    and decides using R2 for House Price regression.

    Dependency injection keeps the evaluator testable:

    Args:
        metrics_fn: Callable ``(y_true, y_pred) -> {mae, rmse, r2}``.
            Defaults to :func:`compute_regression_metrics` (MLEngine
            conventions). Tests may inject a stub.
        production_loader: Callable ``(model_name, models_dir) -> pipeline``.
            Defaults to reusing :func:`load_house_price_model`. Tests may
            inject a stub or pass an already-loaded production model
            directly to :meth:`evaluate_candidate`.
    """

    PRIMARY_METRIC = PRIMARY_METRIC

    def __init__(
        self,
        metrics_fn: Optional[Callable[[Any, Any], Dict[str, float]]] = None,
        production_loader: Optional[Callable[..., Any]] = None,
    ) -> None:
        if metrics_fn is not None and not callable(metrics_fn):
            raise RetrainingError("metrics_fn must be callable or None.")
        if production_loader is not None and not callable(production_loader):
            raise RetrainingError("production_loader must be callable or None.")
        self._metrics_fn = metrics_fn or compute_regression_metrics
        self._production_loader = production_loader or _default_production_loader

    @property
    def metrics_fn(self) -> Callable[[Any, Any], Dict[str, float]]:
        """Returns the injected metrics callable."""
        return self._metrics_fn

    @property
    def production_loader(self) -> Callable[..., Any]:
        """Returns the injected production-model loader."""
        return self._production_loader

    def _evaluate_model(
        self, model: Any, X_test: Any, y_test: Any, *, role: str
    ) -> Dict[str, float]:
        """Runs ``model.predict(X_test)`` and computes MAE/RMSE/R2 as floats.

        Both candidate and production receive exactly the same ``X_test`` /
        ``y_test`` objects; no copying or re-splitting happens here.

        Raises:
            RetrainingError: If prediction or metric computation fails.
        """
        if model is None:
            raise RetrainingError(f"{role} model must not be None.")
        if X_test is None or y_test is None:
            raise RetrainingError(
                f"{role} evaluation requires both X_test and y_test."
            )
        if not hasattr(model, "predict"):
            raise RetrainingError(f"{role} model does not expose predict().")
        try:
            # Pass the raw feature frame straight into the sklearn Pipeline
            # (which already contains preprocessing). No manual transform.
            y_pred = model.predict(X_test)
        except Exception as e:
            raise RetrainingError(
                f"{role} model evaluation failed during predict: {e}"
            ) from e
        try:
            raw = self._metrics_fn(y_test, y_pred)
            metrics = {
                "mae": float(raw["mae"]),
                "rmse": float(raw["rmse"]),
                "r2": float(raw["r2"]),
            }
        except Exception as e:
            if isinstance(e, RetrainingError):
                raise
            raise RetrainingError(
                f"{role} model evaluation failed during metric computation: {e}"
            ) from e
        for key in ("mae", "rmse", "r2"):
            if not isinstance(metrics[key], float):
                raise RetrainingError(
                    f"{role} model metric '{key}' must be a float."
                )
        return metrics

    def _load_production_model(
        self,
        production_model_name: str,
        models_dir: Optional[Union[str, Path]] = None,
    ) -> Any:
        """Loads a production artifact via the injected loader.

        Raises:
            ModelNotFoundError: If the artifact genuinely does not exist
                (propagated as-is so callers map it to ``no_production_model``).
            RetrainingError: If loading fails for any other reason (a broken
                production model must NOT be treated as missing).
        """
        try:
            return self._production_loader(
                production_model_name, models_dir
            )
        except ModelNotFoundError:
            raise
        except RetrainingError:
            raise
        except Exception as e:
            # Covers AdaptiveAIError from a corrupt artifact and any other
            # loader failure: broken production is an error, not "missing".
            raise RetrainingError(
                f"Failed to load production model '{production_model_name}': {e}"
            ) from e

    def evaluate_candidate(
        self,
        candidate_model: Any,
        production_model: Any = None,
        X_test: Any = None,
        y_test: Any = None,
        candidate_model_name: str = "candidate",
        production_model_name: Optional[str] = "production",
        models_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Compares candidate vs production on the SAME ``X_test``/``y_test``.

        The caller may either pass an already-loaded ``production_model``
        (preferred for unit testing) or leave it as ``None`` and provide
        ``production_model_name`` (+ optional ``models_dir``) so the
        evaluator loads the artifact via the House Price persistence
        utility. Passing both a loaded model and a name uses the loaded
        model; the name is then used only as its identity label.

        Args:
            candidate_model: Trained candidate pipeline (e.g. best_pipeline
                from :class:`RetrainingService`).
            production_model: Already-loaded production pipeline, or ``None``.
            X_test: Shared evaluation features (raw DataFrame for Pipelines).
            y_test: Shared evaluation targets.
            candidate_model_name: Identity label for the candidate
                (e.g. ``best_model_name`` from retraining).
            production_model_name: Identity label for production, following
                the persistence naming convention
                (``<model_name>.joblib``). When ``production_model`` is
                ``None`` and this name is provided, the artifact is loaded
                from disk. When both are ``None``, the result is
                ``no_production_model``.
            models_dir: Optional production models directory. ``None`` uses
                the default ``models/house_price/``.

        Returns:
            Dict with ``candidate``, ``production``, ``comparison``, ``status``::

                {
                    "candidate": {"model_name": ..., "metrics": {"mae","rmse","r2"}},
                    "production": {"model_name": ..., "metrics": {...}} | None,
                    "comparison": {
                        "primary_metric": "r2",
                        "candidate_score": float,
                        "production_score": float | None,
                        "improvement": float | None,
                        "candidate_better": bool,
                    },
                    "status": "candidate_better" | "production_better"
                              | "no_production_model",
                }

        Raises:
            RetrainingError: If candidate evaluation fails, or if production
                loading (non-missing) or production evaluation fails.
        """
        if not candidate_model_name or not str(candidate_model_name).strip():
            raise RetrainingError("candidate_model_name must be a non-empty string.")

        # Candidate is always evaluated (also in the no-production path).
        candidate_metrics = self._evaluate_model(
            candidate_model, X_test, y_test, role="Candidate"
        )
        candidate_block: Dict[str, Any] = {
            "model_name": str(candidate_model_name),
            "metrics": dict(candidate_metrics),
        }

        def _no_production_result() -> Dict[str, Any]:
            return {
                "candidate": candidate_block,
                "production": None,
                "comparison": {
                    "primary_metric": PRIMARY_METRIC,
                    "candidate_score": float(candidate_metrics["r2"]),
                    "production_score": None,
                    "improvement": None,
                    "candidate_better": False,
                },
                "status": NO_PRODUCTION_MODEL,
            }

        # Resolve the production model: already-loaded object wins; else try
        # loading from disk when a name was supplied; else no production.
        resolved_production = production_model
        resolved_production_name = production_model_name
        if resolved_production is None:
            if production_model_name is None:
                logger.info("No production model supplied; returning no_production_model.")
                return _no_production_result()
            try:
                resolved_production = self._load_production_model(
                    str(production_model_name), models_dir
                )
            except ModelNotFoundError:
                logger.info(
                    "Production model '%s' not found; returning no_production_model.",
                    production_model_name,
                )
                return _no_production_result()

        if resolved_production_name is None or not str(resolved_production_name).strip():
            resolved_production_name = "production"

        production_metrics = self._evaluate_model(
            resolved_production, X_test, y_test, role="Production"
        )
        production_block: Dict[str, Any] = {
            "model_name": str(resolved_production_name),
            "metrics": dict(production_metrics),
        }

        candidate_score = float(candidate_metrics["r2"])
        production_score = float(production_metrics["r2"])
        improvement = float(candidate_score - production_score)
        candidate_better = bool(candidate_score > production_score)
        status = CANDIDATE_BETTER if candidate_better else PRODUCTION_BETTER

        logger.info(
            "Candidate-vs-production evaluated: candidate=%s r2=%.4f production=%s r2=%.4f "
            "improvement=%+.4f status=%s",
            candidate_block["model_name"],
            candidate_score,
            production_block["model_name"],
            production_score,
            improvement,
            status,
        )
        return {
            "candidate": candidate_block,
            "production": production_block,
            "comparison": {
                "primary_metric": PRIMARY_METRIC,
                "candidate_score": candidate_score,
                "production_score": production_score,
                "improvement": improvement,
                "candidate_better": candidate_better,
            },
            "status": status,
        }

    def evaluate_candidate_vs_production_file(
        self,
        candidate_model: Any,
        X_test: Any,
        y_test: Any,
        candidate_model_name: str = "candidate",
        production_model_name: Optional[str] = None,
        models_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Loads production from disk (if named) and compares on shared data.

        A genuinely missing artifact (``ModelNotFoundError``) yields
        ``status == "no_production_model"`` with candidate metrics intact. A
        corrupt/unloadable artifact or a production evaluation failure raises
        :class:`RetrainingError` and is never treated as missing.
        """
        return self.evaluate_candidate(
            candidate_model=candidate_model,
            production_model=None,
            X_test=X_test,
            y_test=y_test,
            candidate_model_name=candidate_model_name,
            production_model_name=production_model_name,
            models_dir=models_dir,
        )

    def evaluate_house_price_candidate(
        self,
        candidate_model: Any,
        data: Union[pd.DataFrame, Path, str],
        candidate_model_name: str = "candidate",
        production_model_name: Optional[str] = None,
        models_dir: Optional[Union[str, Path]] = None,
        production_model: Any = None,
    ) -> Dict[str, Any]:
        """Validates/splits House Price data, then compares on the test split.

        Reuses the existing House Price validation/split/feature-ordering
        utilities so candidate and production share exactly the same
        validated ``X_test``/``y_test``. ``data`` may be a DataFrame or a CSV
        path. When ``production_model`` is given it is used directly (unit
        testing); otherwise ``production_model_name``/``models_dir`` trigger
        a persistence-utility load (missing file -> ``no_production_model``).

        Raises:
            RetrainingError: If data loading/validation/splitting fails, the
                candidate evaluation fails, or production loading/evaluation
                fails (non-missing).
        """
        from app.use_cases.house_price.split import split_house_price_data

        if isinstance(data, pd.DataFrame):
            frame = data.copy()
        elif isinstance(data, (str, Path)):
            from app.core.config import resolve_project_path

            try:
                frame = pd.read_csv(resolve_project_path(data))
            except FileNotFoundError as e:
                raise RetrainingError(f"Evaluation data not found: {data}") from e
            except Exception as e:
                raise RetrainingError(f"Failed to load evaluation data: {e}") from e
        else:
            raise RetrainingError(
                f"Invalid data type: {type(data).__name__}. "
                "Expected a DataFrame or CSV path."
            )
        try:
            _, X_test, _, y_test = split_house_price_data(frame)
        except Exception as e:
            if isinstance(e, RetrainingError):
                raise
            raise RetrainingError(f"Evaluation data preparation failed: {e}") from e

        return self.evaluate_candidate(
            candidate_model=candidate_model,
            production_model=production_model,
            X_test=X_test,
            y_test=y_test,
            candidate_model_name=candidate_model_name,
            production_model_name=production_model_name,
            models_dir=models_dir,
        )


def evaluate_retraining_candidate(
    candidate_pipeline: Any,
    X_test: Any,
    y_test: Any,
    candidate_model_name: str = "candidate",
    production_model: Any = None,
    production_model_name: Optional[str] = None,
    models_dir: Optional[Union[str, Path]] = None,
    evaluator: Optional[ModelEvaluator] = None,
) -> Dict[str, Any]:
    """Small adapter connecting a RetrainingService result to ModelEvaluator.

    Takes ``best_pipeline``/``best_model_name`` from
    :meth:`RetrainingService.retrain` plus the shared test split and returns
    the standard candidate-vs-production comparison dict. The existing
    ``retrain()`` contract is untouched; this helper lives separately and
    performs evaluation only (no promotion, persistence, registry, or DB).
    """
    active = evaluator if evaluator is not None else ModelEvaluator()
    if not isinstance(active, ModelEvaluator):
        raise RetrainingError("evaluator must be a ModelEvaluator instance or None.")
    return active.evaluate_candidate(
        candidate_model=candidate_pipeline,
        production_model=production_model,
        X_test=X_test,
        y_test=y_test,
        candidate_model_name=candidate_model_name,
        production_model_name=production_model_name,
        models_dir=models_dir,
    )
