"""Configuration system with YAML cascading overrides."""

import os
from pathlib import Path
from typing import Any, Optional
import yaml


class Config:
    """Loads YAML with overrides: default -> mode-specific -> env vars."""

    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = str(Path(__file__).parent.parent.parent / "config")
        self._config_dir = Path(config_dir)
        self.data: dict = {}
        self._load("default.yaml")
        mode = self.data.get("mode", "offline")
        mode_file = f"{mode}.yaml"
        if (self._config_dir / mode_file).exists():
            self._load(mode_file)
        self._apply_env_overrides()

    def _load(self, filename: str) -> None:
        path = self._config_dir / filename
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        if d:
            self._deep_update(self.data, d)

    @staticmethod
    def _deep_update(base: dict, override: dict) -> None:
        for k, v in override.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                Config._deep_update(base[k], v)
            else:
                base[k] = v

    def _apply_env_overrides(self) -> None:
        for key in ["models_base", "mode", "device", "seed"]:
            env_key = f"SCR_{key.upper()}"
            if env_key in os.environ:
                self.data[key] = os.environ[env_key]

    def get(self, key_path: str, default: Any = None) -> Any:
        keys = key_path.split(".")
        d = self.data
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    def set(self, key_path: str, value: Any) -> None:
        keys = key_path.split(".")
        d = self.data
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    @property
    def models_base(self) -> Path:
        return Path(self.data.get("models_base", "."))

    @property
    def mode(self) -> str:
        return self.data.get("mode", "offline")

    @property
    def device(self) -> str:
        return self.data.get("device", "cpu")


_config_instance: Optional[Config] = None


def get_config(config_dir: str = None) -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_dir)
    return _config_instance
