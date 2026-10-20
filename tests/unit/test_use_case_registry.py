import pytest
from typing import List, Dict, Any
from app.use_cases.base import BaseUseCase
from app.use_cases.registry import UseCaseRegistry
from app.core.exceptions import UseCaseNotFoundError

class DummyUseCase(BaseUseCase):
    """A minimal implementation for testing the registry."""
    def __init__(self, name: str):
        self._name = name
        
    @property
    def name(self) -> str:
        return self._name
        
    @property
    def display_name(self) -> str:
        return f"Dummy {self._name}"
        
    @property
    def problem_type(self) -> str:
        return "classification"
        
    @property
    def target_column(self) -> str:
        return "target"
        
    def get_feature_names(self) -> List[str]:
        return ["f1", "f2"]
        
    def get_model_candidates(self) -> List[Dict[str, Any]]:
        return []

@pytest.fixture
def registry():
    return UseCaseRegistry()

def test_register_and_retrieve_use_case(registry):
    use_case = DummyUseCase("test_case")
    registry.register(use_case)
    
    retrieved = registry.get("test_case")
    assert retrieved is use_case
    assert retrieved.name == "test_case"

def test_check_use_case_exists(registry):
    use_case = DummyUseCase("test_case")
    assert not registry.exists("test_case")
    
    registry.register(use_case)
    assert registry.exists("test_case")

def test_list_all_use_cases(registry):
    uc1 = DummyUseCase("case_1")
    uc2 = DummyUseCase("case_2")
    
    registry.register(uc1)
    registry.register(uc2)
    
    all_cases = registry.list_all()
    assert len(all_cases) == 2
    assert uc1 in all_cases
    assert uc2 in all_cases

def test_retrieve_unknown_use_case_raises_error(registry):
    with pytest.raises(UseCaseNotFoundError) as exc_info:
        registry.get("unknown_case")
        
    assert "Use case 'unknown_case' is not registered." in str(exc_info.value)
