from typing import List, Dict, Any
from app.use_cases.base import BaseUseCase
from app.use_cases.house_price.market_config import (
    CATEGORICAL_FEATURES,
    DATA_SCHEMA_VERSION,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)

class HousePriceUseCase(BaseUseCase):
    """
    Pune-aware House Price Prediction Use Case (schema v2).

    Synthetic residential property data calibrated against publicly
    available Pune locality asking-rate benchmarks. Target is ``price``
    in INR (numeric).
    """

    SCHEMA_VERSION = DATA_SCHEMA_VERSION

    @property
    def name(self) -> str:
        return "house_price"

    @property
    def display_name(self) -> str:
        return "House Price Prediction"

    @property
    def problem_type(self) -> str:
        return "regression"

    @property
    def target_column(self) -> str:
        return TARGET_COLUMN

    @property
    def schema_version(self) -> int:
        return DATA_SCHEMA_VERSION

    def get_feature_names(self) -> List[str]:
        return list(FEATURE_COLUMNS)

    def get_numeric_feature_names(self) -> List[str]:
        return list(NUMERIC_FEATURES)

    def get_categorical_feature_names(self) -> List[str]:
        return list(CATEGORICAL_FEATURES)

    def get_monitored_feature_names(self) -> List[str]:
        """Numeric features monitored with the KS drift detector.

        Categorical drift (location/furnishing/type) needs a categorical
        test and is intentionally out of scope for the KS detector.
        """
        return list(NUMERIC_FEATURES)

    def get_model_candidates(self) -> List[Dict[str, Any]]:
        return [
            {
                "model_type": "LinearRegression",
                "hyperparameters": {
                    "fit_intercept": True
                }
            },
            {
                "model_type": "RandomForestRegressor",
                "hyperparameters": {
                    "n_estimators": 100,
                    "max_depth": None,
                    "random_state": 42
                }
            },
            {
                "model_type": "GradientBoostingRegressor",
                "hyperparameters": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 3,
                    "random_state": 42
                }
            }
        ]
