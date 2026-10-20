import logging
from typing import Dict, Any, Union
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.core.exceptions import TrainingError

logger = logging.getLogger(__name__)


class MLEngine:
    """
    Generic machine learning engine for training and evaluating models.

    This engine is decoupled from specific use cases, datasets, or algorithms.
    """

    def train_and_evaluate(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
        preprocessor: Any,
        model: BaseEstimator
    ) -> Dict[str, Any]:
        """
        Builds a scikit-learn Pipeline (preprocessor -> model), fits it on the training set,
        and computes evaluation metrics (MAE, RMSE, R²) on the test set.

        Args:
            X_train: Training features.
            X_test: Evaluation features.
            y_train: Training targets.
            y_test: Evaluation targets.
            preprocessor: Transformer or scikit-learn preprocessing component.
            model: An unfitted scikit-learn estimator/regressor.

        Returns:
            Dict[str, Any]: {
                "pipeline": trained_pipeline,
                "metrics": {
                    "mae": float,
                    "rmse": float,
                    "r2": float
                }
            }

        Raises:
            TrainingError: If pipeline fitting or evaluation fails.
        """
        model_class_name = model.__class__.__name__ if hasattr(model, "__class__") else str(model)
        logger.info(f"Starting pipeline training with model '{model_class_name}'")

        try:
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            mae = float(mean_absolute_error(y_test, y_pred))
            mse = float(mean_squared_error(y_test, y_pred))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_test, y_pred))

            metrics = {
                "mae": mae,
                "rmse": rmse,
                "r2": r2
            }

            logger.info(
                f"Completed evaluation for '{model_class_name}' - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}"
            )

            return {
                "pipeline": pipeline,
                "metrics": metrics
            }

        except Exception as e:
            logger.error(f"Error during training and evaluation of '{model_class_name}': {e}")
            raise TrainingError(f"Training failed for model '{model_class_name}': {str(e)}") from e
