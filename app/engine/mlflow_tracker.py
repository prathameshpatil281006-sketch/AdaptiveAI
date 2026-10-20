import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.pipeline import Pipeline

from app.core.exceptions import ModelPromotionError

# Enable local file store tracking in newer MLflow versions without requiring external DB
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

logger = logging.getLogger(__name__)

DEFAULT_TRACKING_DIRNAME = "mlruns"

# Step 61 (I-12): ONE canonical House Price experiment + lineage convention.
# Both normal selection and orchestration evaluation write candidate runs
# here, distinguished by lineage tags (not by divergent experiments).
HOUSE_PRICE_EXPERIMENT_NAME = "house_price"
HOUSE_PRICE_USE_CASE_NAME = "house_price"
TRAINING_CONTEXT_SELECTION = "selection"
TRAINING_CONTEXT_RETRAINING = "retraining"


def get_default_tracking_dir() -> Path:
    """Returns the default local MLflow store directory (project-root anchored).

    Step 56: stable regardless of the process working directory, so runs
    and registry versions always land in the repository ``mlruns/`` when
    launched from anywhere.
    """
    from app.core.config import get_project_root

    return Path(get_project_root()) / DEFAULT_TRACKING_DIRNAME


class MLflowTracker:
    """
    Reusable tracker for logging ML experiments, candidate parameters,
    evaluation metrics, and trained sklearn pipelines to a local MLflow backend.

    Also supports registering the best model in the MLflow Model Registry
    with versioning (Step 20).
    """

    REGISTERED_MODEL_NAME = "HousePriceModel"

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        experiment_name: str = HOUSE_PRICE_EXPERIMENT_NAME
    ):
        self.tracking_uri = tracking_uri or str(get_default_tracking_dir())
        self.experiment_name = experiment_name
        self.set_tracking_uri(self.tracking_uri)
        self.set_experiment(self.experiment_name)

    def set_tracking_uri(self, uri: str) -> None:
        """Sets the MLflow tracking URI, formatting local paths for cross-platform support.

        Step 56 path contract: relative local paths resolve against the
        project root (never the process working directory), so the
        default local store is stable wherever the app is launched
        from. Absolute paths and remote URIs pass through unchanged.
        """
        if not any(uri.startswith(scheme) for scheme in ("http://", "https://", "file://", "sqlite://")):
            candidate = Path(uri)
            if not candidate.is_absolute():
                from app.core.config import get_project_root

                candidate = Path(get_project_root()) / candidate
            resolved_uri = candidate.resolve().as_uri()
        else:
            resolved_uri = uri

        self.tracking_uri = resolved_uri
        mlflow.set_tracking_uri(resolved_uri)
        logger.info(f"MLflow tracking URI set to: {resolved_uri}")

    def set_experiment(self, experiment_name: str) -> None:
        """Creates or sets the active MLflow experiment."""
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        logger.info(f"MLflow experiment set to: {experiment_name}")

    def start_run(self, run_name: Optional[str] = None) -> mlflow.ActiveRun:
        """Starts a new MLflow run."""
        logger.info(f"Starting MLflow run: {run_name}")
        return mlflow.start_run(run_name=run_name)

    def log_model_type(self, model_type: str) -> None:
        """Logs the model type as tag and parameter."""
        mlflow.set_tag("model_type", model_type)
        mlflow.log_param("model_type", model_type)

    def log_hyperparameters(self, hyperparameters: Dict[str, Any]) -> None:
        """Logs model hyperparameters to MLflow."""
        if hyperparameters:
            cleaned_params = {
                k: ("None" if v is None else v)
                for k, v in hyperparameters.items()
            }
            mlflow.log_params(cleaned_params)

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        """Logs evaluation metrics (MAE, RMSE, R²) to MLflow."""
        for metric_name, value in metrics.items():
            mlflow.log_metric(metric_name, float(value))

    def log_pipeline(self, pipeline: Pipeline, artifact_path: str = "model") -> None:
        """Logs a trained scikit-learn pipeline as an MLflow artifact using cloudpickle."""
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name=artifact_path,
            serialization_format="cloudpickle"
        )

    def end_run(self) -> None:
        """Safely ends the active MLflow run if one exists."""
        if mlflow.active_run():
            mlflow.end_run()
            logger.info("Ended active MLflow run.")

    def track_candidate(
        self,
        model_type: str,
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        pipeline: Pipeline,
        run_name: Optional[str] = None,
        use_case: Optional[str] = None,
        training_context: Optional[str] = None,
        extra_tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Convenience workflow to execute a complete MLflow tracking cycle for a candidate:
        start run -> log info, hyperparameters, metrics, and pipeline -> end run.

        Step 61 (I-12): coherent lineage tags are set inside the run so
        selection and orchestration evaluation share one convention:
        ``model_type`` (legacy, preserved) + ``model_name`` mirror,
        ``use_case`` (e.g. ``house_price``), ``training_context``
        (``selection``/``retraining``) with a ``training_stage`` alias
        for backwards compatibility, plus any ``extra_tags`` (e.g.
        ``candidate_role``). Registry writes never happen here (Step 58).

        Returns:
            str: The run ID of the completed run (exact lineage for promotion).
        """
        self.start_run(run_name=run_name or model_type)
        try:
            self.log_model_type(model_type)
            self.log_hyperparameters(hyperparameters)
            self.log_metrics(metrics)
            self.log_pipeline(pipeline)
            # Coherent lineage (additive only; legacy model_type preserved).
            try:
                mlflow.set_tag("model_name", str(model_type))
            except Exception:
                pass
            if use_case is not None and str(use_case).strip():
                try:
                    mlflow.set_tag("use_case", str(use_case).strip())
                except Exception:
                    pass
            if training_context is not None and str(training_context).strip():
                context_value = str(training_context).strip()
                for key in ("training_context", "training_stage"):
                    try:
                        mlflow.set_tag(key, context_value)
                    except Exception:
                        pass
            if extra_tags:
                for key, value in dict(extra_tags).items():
                    try:
                        mlflow.set_tag(str(key), str(value))
                    except Exception:
                        pass
            run = mlflow.active_run()
            run_id = run.info.run_id if run else ""
            return run_id
        finally:
            self.end_run()

    def add_tags_to_run(self, run_id: str, tags: Dict[str, str]) -> None:
        """Best-effort post-hoc tagging (e.g. best-candidate designation).

        Step 61 (I-12): used to mark the winning candidate run without
        changing its run_id lineage. Failures only warn; tracking-only
        policy is unchanged (no registry writes).
        """
        if not run_id or not str(run_id).strip() or not tags:
            return
        try:
            client = self._get_client()
            for key, value in dict(tags).items():
                client.set_tag(str(run_id), str(key), str(value))
        except Exception as e:
            logger.warning(f"Failed to add lineage tags to run '{run_id}': {e}")

    def _get_client(self) -> MlflowClient:
        """Returns an MlflowClient bound to this tracker's tracking URI."""
        return MlflowClient(tracking_uri=self.tracking_uri)

    def ensure_registered_model(
        self, registered_model_name: str = REGISTERED_MODEL_NAME
    ):
        """
        Ensures a registered model exists, creating it if necessary.

        Args:
            registered_model_name: Name of the registered model.

        Returns:
            The registered model object.

        Raises:
            ModelPromotionError: If creation/retrieval fails.
        """
        if not registered_model_name or not str(registered_model_name).strip():
            raise ModelPromotionError("Registered model name must be a non-empty string.")
        client = self._get_client()
        try:
            try:
                client.create_registered_model(registered_model_name)
                logger.info(f"Created registered model: {registered_model_name}")
            except MlflowException as e:
                if getattr(e, "error_code", "") == "RESOURCE_ALREADY_EXISTS" or \
                        "already exists" in str(e).lower():
                    logger.info(f"Registered model already exists: {registered_model_name}")
                else:
                    raise
            return client.get_registered_model(registered_model_name)
        except MlflowException as e:
            logger.error(f"Failed to ensure registered model '{registered_model_name}': {e}")
            raise ModelPromotionError(
                f"Failed to ensure registered model '{registered_model_name}': {e}"
            ) from e

    def create_model_version(
        self,
        registered_model_name: str,
        source: str,
        run_id: Optional[str] = None,
    ):
        """
        Creates a new model version under an existing registered model.

        Args:
            registered_model_name: Name of the registered model.
            source: Model artifact URI (e.g. 'runs:/<run_id>/model').
            run_id: Source run ID that generated the model.

        Returns:
            The created ModelVersion object.

        Raises:
            ModelPromotionError: If version creation fails.
        """
        if not registered_model_name or not str(registered_model_name).strip():
            raise ModelPromotionError("Registered model name must be a non-empty string.")
        if not source or not str(source).strip():
            raise ModelPromotionError("Model source URI must be a non-empty string.")
        client = self._get_client()
        try:
            model_version = client.create_model_version(
                name=registered_model_name,
                source=source,
                run_id=run_id,
            )
            logger.info(
                f"Created model version '{model_version.version}' "
                f"for registered model '{registered_model_name}' from run '{run_id}'."
            )
            return model_version
        except Exception as e:
            logger.error(f"Failed to create model version for '{registered_model_name}': {e}")
            if isinstance(e, ModelPromotionError):
                raise
            raise ModelPromotionError(
                f"Failed to create model version for '{registered_model_name}': {e}"
            ) from e

    def register_model(
        self,
        run_id: str,
        registered_model_name: str = REGISTERED_MODEL_NAME,
        artifact_path: str = "model",
    ) -> Dict[str, Any]:
        """
        Registers a logged MLflow model artifact as a new version under a
        registered model, reusing the existing run/artifact (no retraining).

        Args:
            run_id: ID of the run containing the logged model artifact.
            registered_model_name: Target registered model name (default 'HousePriceModel').
            artifact_path: Artifact path used when logging the pipeline (default 'model').

        Returns:
            Dict with 'registered_model_name' and 'version' (plus run_id/source).

        Raises:
            ModelPromotionError: If registration fails.
        """
        if not run_id or not str(run_id).strip():
            raise ModelPromotionError("run_id must be a non-empty string for registration.")
        if not registered_model_name or not str(registered_model_name).strip():
            raise ModelPromotionError("Registered model name must be a non-empty string.")
        if not artifact_path or not str(artifact_path).strip():
            raise ModelPromotionError("artifact_path must be a non-empty string.")

        source = f"runs:/{run_id}/{artifact_path}"
        try:
            self.ensure_registered_model(registered_model_name)
            model_version = self.create_model_version(
                registered_model_name=registered_model_name,
                source=source,
                run_id=run_id,
            )
            return {
                "registered_model_name": model_version.name,
                "version": str(model_version.version),
                "run_id": model_version.run_id,
                "source": model_version.source,
            }
        except ModelPromotionError:
            raise
        except Exception as e:
            logger.error(f"Failed to register model '{registered_model_name}': {e}")
            raise ModelPromotionError(
                f"Failed to register model '{registered_model_name}': {e}"
            ) from e

    def list_model_versions(
        self, registered_model_name: str = REGISTERED_MODEL_NAME
    ) -> List[Any]:
        """
        Lists all versions for a registered model.

        Args:
            registered_model_name: Name of the registered model.

        Returns:
            List of ModelVersion objects (empty if none exist).

        Raises:
            ModelPromotionError: If listing fails.
        """
        if not registered_model_name or not str(registered_model_name).strip():
            raise ModelPromotionError("Registered model name must be a non-empty string.")
        client = self._get_client()
        try:
            return list(client.search_model_versions(f"name='{registered_model_name}'"))
        except Exception as e:
            logger.error(f"Failed to list versions for '{registered_model_name}': {e}")
            raise ModelPromotionError(
                f"Failed to list versions for '{registered_model_name}': {e}"
            ) from e

    def get_latest_model_version(
        self, registered_model_name: str = REGISTERED_MODEL_NAME
    ) -> Optional[Any]:
        """
        Retrieves the latest (highest version number) model version, if any.

        Args:
            registered_model_name: Name of the registered model.

        Returns:
            The latest ModelVersion, or None if no versions exist.

        Raises:
            ModelPromotionError: If retrieval fails.
        """
        versions = self.list_model_versions(registered_model_name)
        if not versions:
            return None
        try:
            return max(versions, key=lambda v: int(v.version))
        except (ValueError, TypeError) as e:
            raise ModelPromotionError(
                f"Failed to determine latest version for '{registered_model_name}': {e}"
            ) from e
