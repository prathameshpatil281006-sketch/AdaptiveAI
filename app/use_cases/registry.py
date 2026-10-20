from typing import Dict, List
from app.use_cases.base import BaseUseCase
from app.core.exceptions import UseCaseNotFoundError

class UseCaseRegistry:
    """
    Registry for managing machine learning use cases in the AdaptiveAI platform.
    """
    
    def __init__(self):
        self._use_cases: Dict[str, BaseUseCase] = {}
        
    def register(self, use_case: BaseUseCase) -> None:
        """
        Registers a new use case in the registry.
        
        Args:
            use_case (BaseUseCase): An instance of a class implementing BaseUseCase.
        """
        self._use_cases[use_case.name] = use_case
        
    def get(self, name: str) -> BaseUseCase:
        """
        Retrieves a use case by its unique name.
        
        Args:
            name (str): The internal name of the use case.
            
        Returns:
            BaseUseCase: The registered use case.
            
        Raises:
            UseCaseNotFoundError: If the use case is not registered.
        """
        if name not in self._use_cases:
            raise UseCaseNotFoundError(f"Use case '{name}' is not registered.")
        return self._use_cases[name]
        
    def exists(self, name: str) -> bool:
        """
        Checks if a use case is registered.
        
        Args:
            name (str): The internal name of the use case.
            
        Returns:
            bool: True if registered, False otherwise.
        """
        return name in self._use_cases
        
    def list_all(self) -> List[BaseUseCase]:
        """
        Lists all registered use cases.
        
        Returns:
            List[BaseUseCase]: A list of all registered use cases.
        """
        return list(self._use_cases.values())
