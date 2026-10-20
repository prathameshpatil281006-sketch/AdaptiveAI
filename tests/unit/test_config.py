import pytest
from app.core.config import load_config

def test_load_config_success():
    """Test that the default configuration loads successfully with expected values."""
    config = load_config()
    
    assert config["application"]["name"] == "AdaptiveAI"
    assert config["application"]["environment"] == "development"
    assert config["random_seed"] == 42
    assert config["logging"]["level"] == "INFO"

def test_load_config_file_not_found():
    """Test that loading a non-existent configuration file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_config("config/non_existent.yaml")
