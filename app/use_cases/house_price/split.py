import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split
from app.core.config import load_config
from app.use_cases.house_price.validation import validate_house_price_data
from app.use_cases.house_price.use_case import HousePriceUseCase

def split_house_price_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Validates and splits the House Price dataset into training and testing sets.
    
    Args:
        df (pd.DataFrame): The raw input dataset containing features and target.
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]: 
        X_train, X_test, y_train, y_test
    """
    # Validate the dataset first
    validate_house_price_data(df)
    
    use_case = HousePriceUseCase()
    features = use_case.get_feature_names()
    target = use_case.target_column
    
    X = df[features]
    y = df[target]
    
    config = load_config()
    seed = config.get("random_seed", 42)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=seed
    )
    
    return X_train, X_test, y_train, y_test
