"""Monitoring data provider for AdaptiveAI (Step 25).

Reusable layer that supplies reference and production feature data
in a consistent format for future drift detection (Step 26).

Contract:
    reference_features: pandas.DataFrame (feature columns only)
    production_features: pandas.DataFrame (same feature columns, same order)

Empty production history returns an empty DataFrame with the
appropriate feature columns (documented choice); it does not raise.

No drift calculation lives here.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from app.core.exceptions import AdaptiveAIError, DataValidationError
from app.database.repository import PredictionRepository
from app.use_cases.base import BaseUseCase

logger = logging.getLogger(__name__)

DEFAULT_HOUSE_PRICE_REFERENCE = Path("data") / "reference" / "house_price_reference.csv"


def _resolve_default_reference_path(use_case_name: str) -> Path:
    """Resolves the default reference CSV path for a known use case."""
    if use_case_name == "house_price":
        try:
            from app.core.config import get_project_root

            return Path(get_project_root()) / DEFAULT_HOUSE_PRICE_REFERENCE
        except Exception:
            return Path(DEFAULT_HOUSE_PRICE_REFERENCE)
    raise AdaptiveAIError(
        f"No default reference dataset configured for use case '{use_case_name}'. "
        "Provide an explicit reference_csv_path."
    )


class MonitoringDataProvider:
    """Retrieves reference and production feature data for monitoring.

    Reference data comes from a CSV file (feature columns validated).
    Production data comes exclusively through PredictionRepository
    (no raw SQL in this layer).
    """

    def __init__(
        self,
        use_case: BaseUseCase,
        repository: PredictionRepository,
        reference_csv_path: Optional[Union[str, Path]] = None,
        source: Optional[str] = None,
        demo_session_id: Optional[str] = None,
    ) -> None:
        """Initializes the provider with dependency injection.

        Phase 6 traffic isolation: ``source`` selects which prediction
        traffic this provider observes — None (legacy default: all
        rows), ``"production"`` (real serving traffic only, legacy
        NULL-source rows included), or ``"demo"`` (simulated traffic
        only, optionally narrowed to ``demo_session_id``). Validation
        of the pairing lives in the repository layer.
        """
        if not isinstance(use_case, BaseUseCase):
            raise AdaptiveAIError("use_case must be a BaseUseCase instance.")
        if not isinstance(repository, PredictionRepository):
            raise AdaptiveAIError("repository must be a PredictionRepository instance.")
        self._use_case = use_case
        self._repository = repository
        if reference_csv_path is None:
            self._reference_path = _resolve_default_reference_path(use_case.name)
        else:
            self._reference_path = Path(reference_csv_path)
        self._source = source
        self._demo_session_id = demo_session_id

    @property
    def source(self) -> Optional[str]:
        """Returns the traffic-source filter (None = all traffic)."""
        return self._source

    @property
    def demo_session_id(self) -> Optional[str]:
        """Returns the demo-session filter (None = all demo traffic)."""
        return self._demo_session_id

    @property
    def use_case_name(self) -> str:
        """Returns the use-case name this provider serves."""
        return self._use_case.name

    @property
    def schema_version(self) -> Optional[int]:
        """Returns the use-case data-schema version when declared."""
        version = getattr(self._use_case, "schema_version", None)
        if isinstance(version, bool) or not isinstance(version, int):
            return None
        return int(version)

    @property
    def feature_names(self) -> List[str]:
        """Returns the expected feature names from the use-case abstraction."""
        return list(self._use_case.get_feature_names())

    @property
    def monitored_feature_names(self) -> List[str]:
        """Returns the drift-monitored (numeric) feature names.

        Falls back to all features for use cases without a monitored
        subset. The KS detector requires numeric data; categorical
        drift needs a categorical test (out of scope).
        """
        monitored = getattr(self._use_case, "get_monitored_feature_names", None)
        if callable(monitored):
            return list(monitored())
        return list(self._use_case.get_feature_names())

    def get_reference_features(self) -> pd.DataFrame:
        """Loads reference feature data (monitored columns only, in order)."""
        return self._read_reference_columns(self.monitored_feature_names)

    def get_full_reference_features(self) -> pd.DataFrame:
        """Loads reference feature data (all use-case features, in order).

        Used by categorical drift, data quality, and distributions so
        every consumer compares against the same reference population.
        """
        return self._read_reference_columns(self.feature_names)

    def _read_reference_columns(self, columns: List[str]) -> pd.DataFrame:
        """Reads the requested columns from the reference CSV."""
        path = self._reference_path
        if not path.is_file():
            raise AdaptiveAIError(f"Reference dataset not found: {path}")
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise AdaptiveAIError(f"Failed to read reference dataset at {path}: {e}") from e
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise DataValidationError(
                f"Reference dataset is missing required columns: {', '.join(missing)}"
            )
        try:
            return df[columns].copy()
        except Exception as e:
            raise AdaptiveAIError(f"Failed to extract reference features: {e}") from e

    def get_production_features(self, limit: int = 1000) -> pd.DataFrame:
        """Returns production features from prediction history.

        Empty history returns an empty DataFrame with the expected
        feature columns. Malformed stored input_data raises an
        AdaptiveAI error instead of silently producing bad data.
        """
        feature_names = self.feature_names
        monitored_names = self.monitored_feature_names
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise AdaptiveAIError("limit must be a non-negative integer.")
        try:
            records = self._repository.list_predictions(
                limit=limit,
                use_case=self._use_case.name,
                source=self._source,
                demo_session_id=self._demo_session_id,
            )
        except AdaptiveAIError:
            raise
        except Exception as e:
            raise AdaptiveAIError(f"Failed to read production history: {e}") from e

        if not records:
            return pd.DataFrame(columns=monitored_names)

        rows: List[Dict[str, object]] = []
        skipped_legacy = 0
        for record in records:
            input_data = record.get("input_data")
            if not isinstance(input_data, dict):
                raise AdaptiveAIError(
                    "Stored prediction input_data must decode to a dict."
                )
            # Skip records stored under an older data schema (e.g. legacy
            # 5-feature rows) so historical rows stay readable in history
            # while monitoring only compares current-schema rows.
            if any(c not in input_data for c in feature_names):
                skipped_legacy += 1
                continue
            rows.append({c: input_data[c] for c in monitored_names})
        if skipped_legacy:
            logger.info(
                "Skipped %d historical prediction(s) stored under an older "
                "data schema for monitoring.",
                skipped_legacy,
            )
        if not rows:
            return pd.DataFrame(columns=monitored_names)
        try:
            return pd.DataFrame(rows, columns=monitored_names)
        except Exception as e:
            raise AdaptiveAIError(f"Failed to build production features: {e}") from e

    def get_monitoring_data(self, limit: int = 1000) -> Dict[str, pd.DataFrame]:
        """Returns both feature frames for future drift detection."""
        return {
            "reference_features": self.get_reference_features(),
            "production_features": self.get_production_features(limit=limit),
        }

    def get_full_monitoring_data(self, limit: int = 1000) -> Dict[str, Any]:
        """Returns full-schema frames plus window accounting.

        Unlike ``get_production_features`` (which raises on malformed
        stored rows), malformed rows are COUNTED here so data quality
        can report them instead of failing the whole health check.

        Returns:
            Dict with ``reference_full`` (all features),
            ``production_full`` (v2 rows only, all features),
            ``raw_rows`` (parsed input mappings, None for unparseable
            rows),
            ``total_records``, ``v2_records``, ``legacy_records``,
            ``malformed_records``.
        """
        feature_names = self.feature_names
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
            raise AdaptiveAIError("limit must be a non-negative integer.")
        try:
            raw_texts = self._repository.list_raw_inputs(
                limit=limit,
                use_case=self._use_case.name,
                source=self._source,
                demo_session_id=self._demo_session_id,
            )
        except AdaptiveAIError:
            raise
        except Exception as e:
            raise AdaptiveAIError(f"Failed to read production history: {e}") from e

        import json

        reference_full = self.get_full_reference_features()
        raw_rows: List[Any] = []
        v2_rows: List[Dict[str, object]] = []
        legacy_records = 0
        malformed_records = 0
        for text in raw_texts:
            try:
                input_data = json.loads(text)
            except (TypeError, ValueError):
                input_data = None
            raw_rows.append(input_data)
            if not isinstance(input_data, dict):
                malformed_records += 1
                continue
            if any(c not in input_data for c in feature_names):
                legacy_records += 1
                continue
            v2_rows.append({c: input_data[c] for c in feature_names})
        if legacy_records:
            logger.info(
                "Skipped %d historical prediction(s) stored under an older "
                "data schema for monitoring.",
                legacy_records,
            )
        try:
            production_full = pd.DataFrame(v2_rows, columns=feature_names)
        except Exception as e:
            raise AdaptiveAIError(f"Failed to build production features: {e}") from e
        return {
            "reference_full": reference_full,
            "production_full": production_full,
            "raw_rows": raw_rows,
            "total_records": len(raw_rows),
            "v2_records": len(v2_rows),
            "legacy_records": legacy_records,
            "malformed_records": malformed_records,
        }


def create_house_price_provider(
    repository: PredictionRepository,
    reference_csv_path: Optional[Union[str, Path]] = None,
    source: Optional[str] = None,
    demo_session_id: Optional[str] = None,
) -> MonitoringDataProvider:
    """Convenience factory for the House Price use case.

    Phase 6: ``source``/``demo_session_id`` select the observed
    traffic class (None = all rows, legacy default).
    """
    from app.use_cases.house_price.use_case import HousePriceUseCase

    return MonitoringDataProvider(
        use_case=HousePriceUseCase(),
        repository=repository,
        reference_csv_path=reference_csv_path,
        source=source,
        demo_session_id=demo_session_id,
    )
