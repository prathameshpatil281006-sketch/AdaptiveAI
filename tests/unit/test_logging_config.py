import logging
import pytest
from unittest.mock import patch
from app.core.logging_config import setup_logging

def test_setup_logging_works_and_respects_level():
    root_logger = logging.getLogger()
    initial_level = root_logger.level
    
    setup_logging()
    
    # Platform.yaml configures it to INFO
    assert root_logger.level == logging.INFO
    
    adaptive_handlers = [h for h in root_logger.handlers if getattr(h, "_is_adaptive_ai_handler", False)]
    assert len(adaptive_handlers) == 1
    
    # Cleanup
    root_logger.removeHandler(adaptive_handlers[0])
    root_logger.setLevel(initial_level)

def test_setup_logging_no_duplicate_handlers():
    root_logger = logging.getLogger()
    
    # Call multiple times
    setup_logging()
    setup_logging()
    setup_logging()
    
    adaptive_handlers = [h for h in root_logger.handlers if getattr(h, "_is_adaptive_ai_handler", False)]
    assert len(adaptive_handlers) == 1
    
    # Cleanup
    root_logger.removeHandler(adaptive_handlers[0])

def test_setup_logging_invalid_level():
    with patch("app.core.logging_config.load_config") as mock_load:
        mock_load.return_value = {
            "application": {"name": "AdaptiveAI"},
            "logging": {"level": "INVALID_LEVEL"}
        }
        
        with pytest.raises(ValueError, match="Invalid logging level"):
            setup_logging()

def test_info_messages_can_be_logged(caplog):
    # Pytest caplog fixture captures standard logging automatically
    setup_logging()
    
    logger = logging.getLogger("test_logger")
    logger.info("This is an expected INFO message for testing.")
    
    assert "This is an expected INFO message for testing." in caplog.text
    
    # Ensure debug is NOT captured since level is INFO
    logger.debug("This is a DEBUG message that should be ignored.")
    assert "This is a DEBUG message that should be ignored." not in caplog.text

    # Cleanup
    root_logger = logging.getLogger()
    adaptive_handlers = [h for h in root_logger.handlers if getattr(h, "_is_adaptive_ai_handler", False)]
    if adaptive_handlers:
        root_logger.removeHandler(adaptive_handlers[0])
