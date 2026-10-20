import logging
from app.core.config import load_config

def setup_logging() -> None:
    """
    Configures the standard Python logging system using the application configuration.
    
    - Reads the logging level from config.
    - Sets up the root logger with a standard format.
    - Prevents adding duplicate handlers if called multiple times.
    
    Raises:
        ValueError: If the configured logging level is invalid.
    """
    config = load_config()
    app_name = config["application"]["name"]
    level_str = config["logging"]["level"].upper()
    
    valid_levels = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL
    }
    
    if level_str not in valid_levels:
        raise ValueError(f"Invalid logging level '{level_str}'. Must be one of {list(valid_levels.keys())}")
        
    level = valid_levels[level_str]
    
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Check if we already added our specific handler to avoid duplicates
    for handler in root_logger.handlers:
        if getattr(handler, "_is_adaptive_ai_handler", False):
            # Also update the level of the existing handler just in case config changed
            handler.setLevel(level)
            return
            
    log_format = f"%(asctime)s | %(levelname)s | {app_name} | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    
    formatter = logging.Formatter(fmt=log_format, datefmt=date_format)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    
    # Mark this handler so we can identify it later
    setattr(console_handler, "_is_adaptive_ai_handler", True)
    
    root_logger.addHandler(console_handler)
