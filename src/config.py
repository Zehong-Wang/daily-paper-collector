from pathlib import Path
import logging
import os

import yaml
from dotenv import load_dotenv

load_dotenv()


def get_project_root() -> Path:
    """Walk up from this file's directory to find the project root
    (identified by the presence of pyproject.toml).
    Returns the absolute Path to the project root."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError("Could not find project root (no pyproject.toml found)")


def load_config(path: str = None) -> dict:
    """Load and return the YAML config dict.
    If path is None, defaults to config/config.yaml relative to project root.
    Resolves the database path to an absolute path relative to project root."""
    root = get_project_root()
    if path is None:
        config_path = root / "config" / "config.yaml"
    else:
        config_path = Path(path)
        if not config_path.is_absolute():
            config_path = root / config_path
    with open(config_path) as f:
        config = yaml.safe_load(f)
    # Resolve database path relative to project root
    db_path = config.get("database", {}).get("path", "data/papers.db")
    if not Path(db_path).is_absolute():
        config["database"]["path"] = str(root / db_path)
    return config


def get_env(key: str) -> str:
    """Read from os.environ. Raises ValueError if the key is missing."""
    value = os.environ.get(key)
    if value is None:
        raise ValueError(f"Required environment variable '{key}' is not set")
    return value


def setup_logging(level: str = "INFO"):
    """Configure project-wide logging. Call once at startup."""
    log_level = getattr(logging, level.upper())
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger().setLevel(log_level)
