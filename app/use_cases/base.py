"""
Base use-case interface for the AdaptiveAI platform.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any

class BaseUseCase(ABC):
    """
    Abstract base class for all machine learning use cases.
    
    This interface defines the fundamental contract that every specific
    use case (e.g., house price prediction, demand forecasting) must implement.
    It provides the necessary metadata and configuration for the common engine
    without tightly coupling the application to specific algorithms.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """The internal unique identifier for the use case (e.g., 'house_price')."""
        pass

    @property
    @abstractmethod
    def display_name(self) -> str:
        """The human-readable name for the use case (e.g., 'House Price Prediction')."""
        pass

    @property
    @abstractmethod
    def problem_type(self) -> str:
        """
        The type of machine learning problem this use case solves.
        Common values: 'regression', 'classification', 'time-series'.
        """
        pass

    @property
    @abstractmethod
    def target_column(self) -> str:
        """The name of the target column/variable to predict."""
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        Returns a list of feature column names used for training the model.
        
        Returns:
            List[str]: The list of feature names.
        """
        pass

    @abstractmethod
    def get_model_candidates(self) -> List[Dict[str, Any]]:
        """
        Returns a list of candidate model definitions to be evaluated by the engine.
        
        Each candidate is represented generically (e.g., as a dictionary or 
        structured object containing algorithm names and hyperparameter spaces),
        rather than instantiated ML models, keeping the interface decoupled from 
        specific ML libraries.
        
        Returns:
            List[Dict[str, Any]]: A list of model candidate representations.
        """
        pass
