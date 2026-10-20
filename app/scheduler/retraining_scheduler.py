"""Automated retraining scheduler foundation for AdaptiveAI (Step 37).

Lightweight application-level orchestration that triggers exactly one
retraining cycle on explicit demand::

    RetrainingScheduler
        |
        v
    RetrainingService.retrain(use_case, data_path, promote,
                              production_model, models_dir)
        |
        v
    structured retraining result (returned unchanged)

Design notes:

- Dependency injection only. No global service objects, no singletons,
  no import-time side effects.
- No loops, no sleeps, no threads, no background workers, no automatic
  execution. Recurring execution will be connected to a real scheduling
  mechanism in a later step. ``enabled=False`` means nothing runs
  automatically.
- No model persistence, no registry writes, no database access, no
  monitoring/drift logic here. The injected retraining service remains
  the sole owner of retraining orchestration and promotion.
- ``run_once()`` propagates :class:`RetrainingError` unchanged and wraps
  any unexpected failure in :class:`RetrainingError` so a failed cycle
  is never reported as success.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from app.core.config import resolve_project_path
from app.core.exceptions import AdaptiveAIError, ConfigurationError, RetrainingError
from app.use_cases.base import BaseUseCase

logger = logging.getLogger(__name__)

DEFAULT_ENABLED = False
DEFAULT_INTERVAL_MINUTES = 60
# Absolute project-root anchored default (Step 56): scheduled cycles
# resolve the same reference dataset from any working directory.
# Explicit per-instance data_path values still pass through unchanged.
DEFAULT_DATA_PATH = str(
    resolve_project_path(
        Path("data") / "reference" / "house_price_reference.csv"
    )
)


def parse_scheduler_config(raw: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validates a raw scheduler mapping and returns normalized values.

    Args:
        raw: Optional mapping with ``enabled`` and ``interval_minutes``.
            ``None`` (or an empty mapping) yields the defaults.

    Returns:
        Dict with ``enabled`` (bool) and ``interval_minutes`` (int).

    Raises:
        ConfigurationError: If the mapping is not a dict, if ``enabled``
            is not a boolean, or if ``interval_minutes`` is not a
            positive integer.
    """
    if raw is None:
        return {"enabled": DEFAULT_ENABLED, "interval_minutes": DEFAULT_INTERVAL_MINUTES}
    if not isinstance(raw, dict):
        raise ConfigurationError("scheduler configuration must be a mapping or None.")
    enabled = raw.get("enabled", DEFAULT_ENABLED)
    if not isinstance(enabled, bool):
        raise ConfigurationError("scheduler 'enabled' must be a boolean.")
    interval = raw.get("interval_minutes", DEFAULT_INTERVAL_MINUTES)
    if isinstance(interval, bool) or not isinstance(interval, int):
        raise ConfigurationError("scheduler 'interval_minutes' must be a positive integer.")
    if interval < 1:
        raise ConfigurationError("scheduler 'interval_minutes' must be a positive integer.")
    return {"enabled": bool(enabled), "interval_minutes": int(interval)}


def resolve_scheduler_config(app_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Extracts and validates the scheduler section from a platform config.

    Args:
        app_config: Full platform configuration mapping (as returned by
            ``load_config``). ``None`` yields the defaults.

    Returns:
        Normalized scheduler dict with ``enabled`` and ``interval_minutes``.

    Raises:
        ConfigurationError: If the scheduler section is present but invalid.
    """
    if app_config is None:
        return parse_scheduler_config(None)
    if not isinstance(app_config, dict):
        raise ConfigurationError("application configuration must be a mapping or None.")
    return parse_scheduler_config(app_config.get("scheduler", None))


class RetrainingScheduler:
    """Explicit, single-cycle trigger for retraining jobs.

    The scheduler holds injected collaborators and configuration only.
    Work happens exclusively inside :meth:`run_once`, which delegates to
    the injected retraining service exactly once and returns its result
    object unchanged (no reconstruction, no copying of the outcome).
    """

    def __init__(
        self,
        retraining_service: Any,
        use_case: BaseUseCase,
        data_path: Union[str, Path, Any] = DEFAULT_DATA_PATH,
        enabled: bool = DEFAULT_ENABLED,
        interval_minutes: int = DEFAULT_INTERVAL_MINUTES,
        production_model: Any = None,
        models_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        """Wires the scheduler with injected dependencies.

        Args:
            retraining_service: Object exposing ``retrain(use_case=...,
                data_path=..., promote=...)`` (normally a
                ``RetrainingService``). Duck-typed so tests can inject
                stubs/mocks; never instantiated globally here.
            use_case: Use-case abstraction forwarded to the service.
            data_path: CSV path (or DataFrame) forwarded unchanged to
                the service.
            enabled: Scheduler flag. ``False`` (default) means nothing
                runs automatically. Stored for status reporting only;
                it never gates an explicit :meth:`run_once` call.
            interval_minutes: Intended cadence for a future recurring
                mechanism. Must be a positive integer. Stored only.
            production_model: Optional pre-loaded production pipeline
                forwarded to the service (Step 66 I-16 fix: previously
                supported by the service but silently dropped here).
                ``None`` (default) preserves legacy behavior where the
                service resolves production itself.
            models_dir: Optional production directory override
                forwarded to the service (same I-16 fix). ``None``
                (default) preserves legacy behavior.

        Raises:
            RetrainingError: If the service, use case, or data path is invalid.
            ConfigurationError: If ``enabled``/``interval_minutes`` are invalid.
        """
        retrain_fn = getattr(retraining_service, "retrain", None)
        if not callable(retrain_fn):
            raise RetrainingError(
                "retraining_service must expose a callable retrain() or be provided."
            )
        if not isinstance(use_case, BaseUseCase):
            raise RetrainingError("use_case must be a BaseUseCase instance.")
        if isinstance(data_path, str):
            if not data_path.strip():
                raise RetrainingError("data_path must be a non-empty CSV path or DataFrame.")
        elif isinstance(data_path, Path):
            pass
        else:
            # Allow pandas DataFrame passthrough without importing pandas
            # here (the service owns data loading/validation).
            if not (hasattr(data_path, "columns") and hasattr(data_path, "copy")):
                raise RetrainingError(
                    "data_path must be a CSV path or pandas DataFrame."
                )
        if not isinstance(enabled, bool):
            raise ConfigurationError("scheduler 'enabled' must be a boolean.")
        if isinstance(interval_minutes, bool) or not isinstance(interval_minutes, int):
            raise ConfigurationError(
                "scheduler 'interval_minutes' must be a positive integer."
            )
        if interval_minutes < 1:
            raise ConfigurationError(
                "scheduler 'interval_minutes' must be a positive integer."
            )
        if models_dir is not None and not isinstance(models_dir, (str, Path)):
            raise RetrainingError("models_dir must be a path string, Path, or None.")
        if isinstance(models_dir, str) and not models_dir.strip():
            raise RetrainingError("models_dir must be a non-empty path string or None.")
        self._retraining_service = retraining_service
        self._use_case = use_case
        self._data_path = data_path
        self._enabled = bool(enabled)
        self._interval_minutes = int(interval_minutes)
        self._production_model = production_model
        self._models_dir = models_dir

    @property
    def retraining_service(self) -> Any:
        """Returns the injected retraining service."""
        return self._retraining_service

    @property
    def use_case(self) -> BaseUseCase:
        """Returns the injected use case."""
        return self._use_case

    @property
    def data_path(self) -> Union[str, Path, Any]:
        """Returns the configured data path."""
        return self._data_path

    @property
    def enabled(self) -> bool:
        """Returns whether automatic scheduling is enabled (status only)."""
        return self._enabled

    @property
    def interval_minutes(self) -> int:
        """Returns the configured interval in minutes (status only)."""
        return self._interval_minutes

    @property
    def production_model(self) -> Any:
        """Returns the configured production model override (may be None)."""
        return self._production_model

    @property
    def models_dir(self) -> Optional[Union[str, Path]]:
        """Returns the configured models directory override (may be None)."""
        return self._models_dir

    @classmethod
    def from_config(
        cls,
        retraining_service: Any,
        use_case: BaseUseCase,
        data_path: Union[str, Path, Any] = DEFAULT_DATA_PATH,
        scheduler_config: Optional[Dict[str, Any]] = None,
    ) -> "RetrainingScheduler":
        """Builds a scheduler from a raw scheduler config mapping.

        Args:
            retraining_service: Injected service exposing ``retrain()``.
            use_case: Injected use-case abstraction.
            data_path: CSV path (or DataFrame) forwarded to the service.
            scheduler_config: Optional raw mapping with ``enabled`` and
                ``interval_minutes`` (e.g. ``platform.yaml["scheduler"]``).

        Returns:
            A configured :class:`RetrainingScheduler`.
        """
        parsed = parse_scheduler_config(scheduler_config)
        return cls(
            retraining_service=retraining_service,
            use_case=use_case,
            data_path=data_path,
            enabled=parsed["enabled"],
            interval_minutes=parsed["interval_minutes"],
        )

    def run_once(self, promote: bool = False) -> Dict[str, Any]:
        """Executes exactly one retraining cycle via the injected service.

        Step 67 (I-17) contract: the scheduler NEVER promotes unless
        promotion is explicitly requested. The default
        ``promote=False`` matches ``RetrainingService.retrain()`` and
        the Steps 55/58 safe semantics (train/track/select only, no
        production evaluation side effects beyond comparison, no
        promotion writes). Pass ``promote=True`` explicitly to allow
        the service's promotion phase.

        Args:
            promote: Explicit promotion flag forwarded unchanged to
                ``RetrainingService.retrain()``. Defaults to False.

        Returns:
            The retraining service result dict, returned as-is without
            reconstruction.

        Raises:
            RetrainingError: If ``promote`` is not a boolean, if the
                underlying service raises :class:`RetrainingError`
                (propagated unchanged), or if any unexpected failure
                occurs (wrapped with the original error chained).
        """
        if not isinstance(promote, bool):
            raise RetrainingError("promote must be a boolean.")
        logger.info(
            "Retraining scheduler cycle started: use_case=%s promote=%s",
            self._use_case.name,
            promote,
        )
        try:
            result = self._retraining_service.retrain(
                use_case=self._use_case,
                data_path=self._data_path,
                promote=promote,
                production_model=self._production_model,
                models_dir=self._models_dir,
            )
        except RetrainingError:
            raise
        except AdaptiveAIError as e:
            raise RetrainingError(f"Scheduled retraining failed: {e}") from e
        except Exception as e:
            raise RetrainingError(f"Scheduled retraining failed: {e}") from e
        logger.info(
            "Retraining scheduler cycle completed: use_case=%s",
            self._use_case.name,
        )
        return result

    def get_status(self) -> Dict[str, Any]:
        """Reports scheduler configuration without side effects.

        Returns:
            Dict with ``enabled``, ``interval_minutes``, ``running``
            (always False: no background execution in this step), and
            ``last_run`` (always None: recurring execution is not yet
            implemented, so no run is ever pretended).
        """
        return {
            "enabled": self._enabled,
            "interval_minutes": self._interval_minutes,
            "running": False,
            "last_run": None,
        }
