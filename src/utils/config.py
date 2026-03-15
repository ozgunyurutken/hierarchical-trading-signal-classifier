"""
Centralized configuration loader.
Reads config.yaml and provides typed access to all project parameters.
"""

import yaml
from pathlib import Path
from typing import Any


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML configuration file."""
    cfg_path = Path(path) if path else _CONFIG_PATH
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


def get_project_root() -> Path:
    """Return absolute path to project root."""
    return _PROJECT_ROOT


# Singleton config instance
_cfg: dict[str, Any] | None = None


def cfg() -> dict[str, Any]:
    """Get cached config dictionary."""
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg
