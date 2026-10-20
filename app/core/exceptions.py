"""
Core exceptions for the AdaptiveAI application.
This module defines a clean hierarchy of exception types to be used across the system.
"""

class AdaptiveAIError(Exception):
    """
    Base exception for all custom AdaptiveAI errors.
    All specific application exceptions should inherit from this class.
    """
    pass

class ConfigurationError(AdaptiveAIError):
    """Raised when there is an issue loading or validating the application configuration."""
    pass

class DataValidationError(AdaptiveAIError):
    """Raised when data provided to the system fails schema or quality validation checks."""
    pass

class ModelNotFoundError(AdaptiveAIError):
    """Raised when a requested machine learning model cannot be found in the registry or filesystem."""
    pass

class UseCaseNotFoundError(AdaptiveAIError):
    """Raised when a specified use case is not registered or supported by the system."""
    pass

class TrainingError(AdaptiveAIError):
    """Raised when an error occurs during the model training pipeline."""
    pass

class PredictionError(AdaptiveAIError):
    """Raised when an error occurs while generating predictions using a trained model."""
    pass

class DriftDetectionError(AdaptiveAIError):
    """Raised when an error occurs during the data or concept drift detection process."""
    pass

class RetrainingError(AdaptiveAIError):
    """Raised when the automated model retraining pipeline fails."""
    pass

class ModelPromotionError(AdaptiveAIError):
    """Raised when an error occurs while attempting to promote a model to a production environment."""
    pass
