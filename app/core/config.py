import os
import yaml
from pathlib import Path
from typing import Dict, Any, Union

def get_project_root() -> str:
    """Returns the project root directory reliably."""
    # This file is in app/core/config.py, so the root is two directories up
    current_dir = os.path.dirname(os.path.abspath(__file__))
    app_dir = os.path.dirname(current_dir)
    return os.path.dirname(app_dir)

def resolve_project_path(path: Union[str, Path]) -> Path:
    """Resolves a project file path independently of the working directory.

    Step 56 path contract: relative paths are anchored at the project
    root (so default data/model/MLflow locations behave identically no
    matter where the process is launched from); absolute paths pass
    through unchanged (normalized). Explicit caller-supplied absolute
    locations (temporary test directories, configured overrides) are
    therefore never rewritten.
    """
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(get_project_root()) / candidate

def load_config(config_file: str = "config/platform.yaml") -> Dict[str, Any]:
    """
    Loads and validates the platform configuration from a YAML file.
    
    Args:
        config_file (str): The relative path to the configuration file from the project root.
        
    Returns:
        Dict[str, Any]: The loaded configuration dictionary.
        
    Raises:
        FileNotFoundError: If the configuration file is not found.
        ValueError: If the configuration is missing required keys or is invalid.
    """
    root_dir = get_project_root()
    config_path = os.path.join(root_dir, config_file)
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML configuration: {e}")
        
    if not isinstance(config, dict):
        raise ValueError("Configuration file must contain a valid YAML mapping")
        
    # Validate required basic settings exist
    required_keys = ["application", "random_seed", "logging"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required top-level configuration key: '{key}'")
            
    if "name" not in config["application"]:
        raise ValueError("Missing 'name' in 'application' configuration")
        
    if "environment" not in config["application"]:
        raise ValueError("Missing 'environment' in 'application' configuration")
        
    if "level" not in config["logging"]:
        raise ValueError("Missing 'level' in 'logging' configuration")

    # Optional scheduler section (Step 37 foundation). When present it is
    # validated; when absent the scheduler defaults apply. Never required.
    if "scheduler" in config:
        scheduler = config["scheduler"]
        if not isinstance(scheduler, dict):
            raise ValueError("'scheduler' configuration must be a mapping")
        if "enabled" in scheduler and not isinstance(scheduler["enabled"], bool):
            raise ValueError("'scheduler.enabled' must be a boolean")
        if "interval_minutes" in scheduler:
            interval = scheduler["interval_minutes"]
            if isinstance(interval, bool) or not isinstance(interval, int):
                raise ValueError("'scheduler.interval_minutes' must be a positive integer")
            if interval < 1:
                raise ValueError("'scheduler.interval_minutes' must be a positive integer")

    return config
