import pytest
from typing import List, Dict, Any
from app.use_cases.base import BaseUseCase

def test_base_use_case_cannot_be_instantiated():
    """Verify that the BaseUseCase cannot be instantiated directly due to abstract methods."""
    with pytest.raises(TypeError) as exc_info:
        BaseUseCase()
    
    assert "Can't instantiate abstract class BaseUseCase" in str(exc_info.value)

class DummyUseCase(BaseUseCase):
    """A minimal implementation of BaseUseCase for testing purposes."""
    
    @property
    def name(self) -> str:
        return "dummy_case"
        
    @property
    def display_name(self) -> str:
        return "Dummy Use Case"
        
    @property
    def problem_type(self) -> str:
        return "classification"
        
    @property
    def target_column(self) -> str:
        return "is_fraud"
        
    def get_feature_names(self) -> List[str]:
        return ["amount", "time", "distance"]
        
    def get_model_candidates(self) -> List[Dict[str, Any]]:
        return [
            {
                "model_type": "random_forest",
                "hyperparameters": {"n_estimators": 100}
            }
        ]

def test_dummy_use_case_implementation():
    """Verify that a class correctly implementing the interface can be instantiated and accessed."""
    use_case = DummyUseCase()
    
    assert use_case.name == "dummy_case"
    assert use_case.display_name == "Dummy Use Case"
    assert use_case.problem_type == "classification"
    assert use_case.target_column == "is_fraud"
    
    features = use_case.get_feature_names()
    assert len(features) == 3
    assert "amount" in features
    
    candidates = use_case.get_model_candidates()
    assert len(candidates) == 1
    assert candidates[0]["model_type"] == "random_forest"
    
    # Verify we are returning generic structures without actual ML instantiation
    assert isinstance(candidates[0], dict)
