import pytest
from app.core.exceptions import (
    AdaptiveAIError,
    ConfigurationError,
    DataValidationError,
    ModelNotFoundError,
    UseCaseNotFoundError,
    TrainingError,
    PredictionError,
    DriftDetectionError,
    RetrainingError,
    ModelPromotionError
)

def test_adaptive_ai_error_is_exception():
    """Verify that the base error inherits from Python's built-in Exception."""
    assert issubclass(AdaptiveAIError, Exception)

def test_specific_exceptions_inherit_from_base():
    """Verify that all specific exceptions inherit from the base AdaptiveAIError."""
    specific_exceptions = [
        ConfigurationError,
        DataValidationError,
        ModelNotFoundError,
        UseCaseNotFoundError,
        TrainingError,
        PredictionError,
        DriftDetectionError,
        RetrainingError,
        ModelPromotionError
    ]
    for exc in specific_exceptions:
        assert issubclass(exc, AdaptiveAIError)

@pytest.mark.parametrize("exc_class", [
    AdaptiveAIError,
    ConfigurationError,
    DataValidationError,
    ModelNotFoundError,
    UseCaseNotFoundError,
    TrainingError,
    PredictionError,
    DriftDetectionError,
    RetrainingError,
    ModelPromotionError
])
def test_exception_raise_and_catch_with_message(exc_class):
    """Verify that each exception can be raised, caught, and preserves its message."""
    test_message = f"Test message for {exc_class.__name__}"
    
    with pytest.raises(exc_class) as exc_info:
        raise exc_class(test_message)
        
    # Verify the message is preserved
    assert str(exc_info.value) == test_message
    
    # Verify it can also be caught as the base AdaptiveAIError (if it is a subclass or the class itself)
    with pytest.raises(AdaptiveAIError) as base_exc_info:
        raise exc_class("Caught as base")
        
    assert str(base_exc_info.value) == "Caught as base"
