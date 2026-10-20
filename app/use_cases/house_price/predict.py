import logging
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from app.core.exceptions import PredictionError, DataValidationError
from app.use_cases.house_price.validation import validate_house_price_data

logger = logging.getLogger(__name__)

def predict_house_price(model: Pipeline, input_data: pd.DataFrame) -> np.ndarray:
    """
    Generates house price predictions using a trained model.
    
    Args:
        model (Pipeline): The trained scikit-learn pipeline.
        input_data (pd.DataFrame): The feature data for prediction.
        
    Returns:
        np.ndarray: An array of predicted house prices.
        
    Raises:
        DataValidationError: If the input data is structurally or logically invalid.
        PredictionError: If prediction fails due to model or runtime issues.
    """
    logger.info(f"House Price prediction started for {len(input_data)} samples.")
    
    try:
        # Validate the input using existing logic.
        # Since the existing validator enforces the presence and positivity of 'price',
        # we temporarily add a dummy valid target column solely for validation purposes.
        df_val = input_data.copy()
        if "price" not in df_val.columns:
            df_val["price"] = 1.0
            
        validate_house_price_data(df_val)
        
        # Make predictions
        predictions = model.predict(input_data)
        
        logger.info("House Price prediction completed successfully.")
        return np.array(predictions)
        
    except DataValidationError:
        # Re-raise validation errors natively
        raise
    except Exception as e:
        logger.error(f"House Price prediction failed: {e}")
        raise PredictionError(f"Failed to generate predictions: {str(e)}") from e
