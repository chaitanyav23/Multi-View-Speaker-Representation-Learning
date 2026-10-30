"""
scripts/utils/config.py
=======================
YAML configuration loader with dot-notation access and runtime override
support.  All pipeline scripts import their settings from here.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Optional

import yaml


# ── Sentinel for missing keys ────────────────────────────────────────────────
_MISSING = object()

_DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "stage1_config.yaml"
)


class ConfigNode:
    """
    Recursive wrapper that exposes YAML dict keys as attributes and supports
    ``cfg["key"]`` access, ``cfg.get("key", default)``, and ``repr(cfg)``.
    """

    def __init__(self, mapping: dict) -> None:
        object.__setattr__(self, "_data", {})
        for k, v in mapping.items():
            self._data[k] = ConfigNode(v) if isinstance(v, dict) else v

    # ── Attribute access ─────────────────────────────────────────────────────
    def __getattr__(self, key: str) -> Any:
        data = object.__getattribute__(self, "_data")
        if key in data:
            return data[key]
        raise AttributeError(f"Config has no key '{key}'")

    def __setattr__(self, key: str, value: Any) -> None:
        self._data[key] = value

    # ── Dict-style access ────────────────────────────────────────────────────
    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    # ── Serialisation ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        result = {}
        for k, v in self._data.items():
            result[k] = v.to_dict() if isinstance(v, ConfigNode) else v
        return result

    def __repr__(self) -> str:
        return f"ConfigNode({self.to_dict()!r})"


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (override wins)."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(
    config_path: Optional[Path | str] = None,
    overrides: Optional[dict] = None,
) -> ConfigNode:
    """
    Load the YAML config file and return a :class:`ConfigNode`.

    Parameters
    ----------
    config_path:
        Path to the YAML file.  Defaults to
        ``configs/stage1_config.yaml`` relative to the project root.
    overrides:
        Optional dict of values to merge on top of the file (useful for
        CLI flag injection).

    Returns
    -------
    ConfigNode
    """
    path = Path(config_path) if config_path else _DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        raw: dict = yaml.safe_load(fh) or {}

    if overrides:
        raw = _deep_merge(raw, overrides)

    # Expand environment variable placeholders  ${ENV_VAR}
    raw = _expand_env_vars(raw)

    return ConfigNode(raw)


def _expand_env_vars(obj: Any) -> Any:
    """Walk *obj* and expand ``${VAR}`` strings using os.environ."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand_env_vars(i) for i in obj]
    return obj
