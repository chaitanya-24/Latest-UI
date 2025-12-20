"""Configuration loader for iAgentOps.

Search order (first found wins):
1) Env var IAGENTOPS_CONFIG -> absolute/relative path to JSON file
2) ./iagentops.config.json (current working directory)
3) ~/.iagentops/config.json (user home directory)

Config schema (partial):
{
  "pricing": {
    "openai": {
      "gpt-4": [0.03, 0.06],
      "gpt-3.5-turbo": [0.0005, 0.0015],
      "default": [0.002, 0.002]
    },
    "anthropic": { "default": [0.003, 0.015] }
  }
}
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Tuple

DEFAULT_CONFIG: Dict[str, Any] = {
    "pricing": {}
}

_CONFIG_CACHE: Dict[str, Any] | None = None


def _candidate_paths() -> list[Path]:
    paths: list[Path] = []
    # 1) Env var
    env_path = os.getenv("IAGENTOPS_CONFIG")
    if env_path:
        paths.append(Path(env_path).expanduser())
    # 2) CWD
    paths.append(Path.cwd() / "iagentops.config.json")
    # 3) HOME
    paths.append(Path.home() / ".iagentops" / "config.json")
    return paths


essential_keys = {"pricing"}


def load_config() -> Dict[str, Any]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    for p in _candidate_paths():
        try:
            if p.is_file():
                with p.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                # minimal validation
                if not isinstance(data, dict):
                    continue
                # Merge over defaults
                config = {**DEFAULT_CONFIG, **data}
                _CONFIG_CACHE = config
                return _CONFIG_CACHE
        except Exception:
            # Ignore bad files and continue
            continue

    _CONFIG_CACHE = DEFAULT_CONFIG.copy()
    return _CONFIG_CACHE


def get_pricing() -> Dict[str, Dict[str, Tuple[float, float]]]:
    cfg = load_config()
    pricing = cfg.get("pricing", {})
    # Normalize values to tuple[float,float]
    normalized: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for provider, models in pricing.items():
        if not isinstance(models, dict):
            continue
        normalized[provider.lower()] = {}
        for model, price in models.items():
            if isinstance(price, (list, tuple)) and len(price) == 2:
                normalized[provider.lower()][model.lower()] = (float(price[0]), float(price[1]))
            else:
                # single value -> same for input/output
                v = float(price)
                normalized[provider.lower()][model.lower()] = (v, v)
    return normalized
