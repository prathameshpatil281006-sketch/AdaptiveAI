import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import joblib
from sklearn.pipeline import Pipeline

from app.core.config import get_project_root
from app.core.exceptions import ModelNotFoundError, AdaptiveAIError

logger = logging.getLogger(__name__)

# Single production-serving reference for House Price (Step 55).
# Designates which persisted artifact + registry version is production.
# The prediction API resolves this first; a legacy named artifact is
# used only when no production reference exists (pre-pointer
# deployments). Exactly one reference file per models directory.
PRODUCTION_POINTER_FILENAME = "production.json"
PRODUCTION_POINTER_SCHEMA_VERSION = 1

# Fields that must be present, non-empty strings for serving.
_REQUIRED_POINTER_FIELDS = (
    "model_name",
    "registered_model_name",
    "model_version",
    "run_id",
)


def _get_default_models_dir() -> Path:
    """Returns the default directory for storing house price models."""
    return Path(get_project_root()) / "models" / "house_price"


def save_house_price_model(
    model: Pipeline,
    model_name: str,
    models_dir: Optional[Union[str, Path]] = None
) -> Path:
    """
    Saves a trained scikit-learn pipeline to disk using joblib.

    Args:
        model (Pipeline): The trained scikit-learn Pipeline to serialize.
        model_name (str): The name/identifier of the model (e.g., 'LinearRegression').
        models_dir (Optional[Union[str, Path]]): Optional custom directory. Defaults to 'models/house_price/'.

    Returns:
        Path: The absolute path to the saved model file.

    Raises:
        AdaptiveAIError: If saving the model fails.
    """
    try:
        base_dir = Path(models_dir) if models_dir is not None else _get_default_models_dir()
        base_dir.mkdir(parents=True, exist_ok=True)

        filename = model_name if model_name.endswith(".joblib") else f"{model_name}.joblib"
        target_path = base_dir / filename

        logger.info(f"Saving House Price model '{model_name}' to {target_path}")
        joblib.dump(model, target_path)
        logger.info(f"Successfully saved House Price model '{model_name}' to {target_path}")

        return target_path

    except Exception as e:
        logger.error(f"Failed to save House Price model '{model_name}': {e}")
        raise AdaptiveAIError(f"Failed to save model '{model_name}': {str(e)}") from e


def load_house_price_model(
    model_name: str,
    models_dir: Optional[Union[str, Path]] = None
) -> Pipeline:
    """
    Loads a serialized scikit-learn pipeline from disk.

    Args:
        model_name (str): The name/identifier of the model to load.
        models_dir (Optional[Union[str, Path]]): Optional custom directory. Defaults to 'models/house_price/'.

    Returns:
        Pipeline: The deserialized scikit-learn Pipeline.

    Raises:
        ModelNotFoundError: If the model file does not exist.
        AdaptiveAIError: If deserializing or reading the model fails.
    """
    base_dir = Path(models_dir) if models_dir is not None else _get_default_models_dir()
    filename = model_name if model_name.endswith(".joblib") else f"{model_name}.joblib"
    target_path = base_dir / filename

    if not target_path.is_file():
        logger.error(f"House Price model file not found at: {target_path}")
        raise ModelNotFoundError(f"Model file not found: {target_path}")

    try:
        logger.info(f"Loading House Price model from {target_path}")
        model = joblib.load(target_path)
        logger.info(f"Successfully loaded House Price model '{model_name}' from {target_path}")
        return model

    except Exception as e:
        logger.error(f"Failed to load House Price model from {target_path}: {e}")
        raise AdaptiveAIError(f"Failed to load model from {target_path}: {str(e)}") from e


def get_production_pointer_path(
    models_dir: Optional[Union[str, Path]] = None
) -> Path:
    """Returns the path of the production serving reference file."""
    base_dir = Path(models_dir) if models_dir is not None else _get_default_models_dir()
    return base_dir / PRODUCTION_POINTER_FILENAME


def _validate_production_pointer(pointer: Any) -> Dict[str, Any]:
    """Validates a production pointer mapping without mutating it."""
    if not isinstance(pointer, dict):
        raise AdaptiveAIError("Production pointer must be a JSON object.")
    normalized: Dict[str, Any] = dict(pointer)
    for field in _REQUIRED_POINTER_FIELDS:
        value = normalized.get(field)
        if not isinstance(value, str) or not value.strip():
            raise AdaptiveAIError(
                f"Production pointer field '{field}' must be a non-empty string."
            )
        normalized[field] = value.strip()
    for numeric_field in ("candidate_score", "production_score", "improvement"):
        value = normalized.get(numeric_field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AdaptiveAIError(
                f"Production pointer field '{numeric_field}' must be a number or null."
            )
        try:
            as_float = float(value)
        except (TypeError, ValueError, OverflowError) as e:
            raise AdaptiveAIError(
                f"Production pointer field '{numeric_field}' must be a number or null: {e}"
            ) from e
        import math

        if not math.isfinite(as_float):
            raise AdaptiveAIError(
                f"Production pointer field '{numeric_field}' must be a finite number or null."
            )
        normalized[numeric_field] = as_float
    return normalized


def read_production_pointer(
    models_dir: Optional[Union[str, Path]] = None
) -> Dict[str, Any]:
    """Reads the production serving reference for House Price.

    Args:
        models_dir: Optional custom directory. Defaults to 'models/house_price/'.

    Returns:
        The validated pointer mapping (model_name, registered_model_name,
        model_version, run_id, plus optional metadata).

    Raises:
        ModelNotFoundError: If no production reference has been deployed.
        AdaptiveAIError: If the reference exists but is corrupt/invalid.
    """
    target_path = get_production_pointer_path(models_dir)
    if not target_path.is_file():
        raise ModelNotFoundError(
            f"No production House Price model is deployed (missing {target_path})."
        )
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            pointer = json.load(f)
    except (OSError, ValueError) as e:
        logger.error(f"Production pointer at {target_path} is unreadable: {e}")
        raise AdaptiveAIError(
            f"Production pointer at {target_path} is corrupt: {e}"
        ) from e
    try:
        return _validate_production_pointer(pointer)
    except AdaptiveAIError as e:
        logger.error(f"Production pointer at {target_path} is invalid: {e}")
        raise AdaptiveAIError(
            f"Production pointer at {target_path} is invalid: {e}"
        ) from e


def write_production_pointer(
    pointer: Dict[str, Any],
    models_dir: Optional[Union[str, Path]] = None,
) -> Path:
    """Atomically writes the production serving reference.

    The reference is validated, stamped with the schema version and a
    UTC deployment timestamp, written to a temporary file in the same
    directory, then atomically moved over the previous reference (when
    the platform supports atomic rename). A failed write leaves the
    previous reference untouched.

    Args:
        pointer: Mapping with model_name, registered_model_name,
            model_version, run_id (non-empty strings) plus optional
            metadata (use_case, primary_metric, scores, reason).
        models_dir: Optional custom directory. Defaults to 'models/house_price/'.

    Returns:
        Path: The absolute path of the production reference file.

    Raises:
        AdaptiveAIError: If validation or the atomic write fails.
    """
    try:
        base_dir = Path(models_dir) if models_dir is not None else _get_default_models_dir()
        base_dir.mkdir(parents=True, exist_ok=True)
        target_path = base_dir / PRODUCTION_POINTER_FILENAME

        validated = _validate_production_pointer(pointer)
        stamped: Dict[str, Any] = {
            "schema_version": PRODUCTION_POINTER_SCHEMA_VERSION,
            "deployed_at": datetime.now(timezone.utc).isoformat(),
        }
        stamped.update(validated)

        payload = json.dumps(stamped, indent=2, sort_keys=True)
        tmp_path = target_path.with_name(target_path.name + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp_path, target_path)

        logger.info(
            "Production House Price reference now points at '%s' (%s v%s, run %s).",
            stamped["model_name"],
            stamped["registered_model_name"],
            stamped["model_version"],
            stamped["run_id"],
        )
        return target_path
    except AdaptiveAIError:
        raise
    except Exception as e:
        logger.error(f"Failed to write production pointer: {e}")
        raise AdaptiveAIError(f"Failed to write production pointer: {str(e)}") from e


def load_production_house_price_model(
    models_dir: Optional[Union[str, Path]] = None
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Loads the currently designated production model and its reference.

    Args:
        models_dir: Optional custom directory. Defaults to 'models/house_price/'.

    Returns:
        Tuple of (pipeline, production pointer mapping).

    Raises:
        ModelNotFoundError: If no production reference exists or the
            designated artifact file is missing.
        AdaptiveAIError: If the reference is corrupt or loading fails.
    """
    pointer = read_production_pointer(models_dir)
    model = load_house_price_model(pointer["model_name"], models_dir=models_dir)
    return model, pointer
