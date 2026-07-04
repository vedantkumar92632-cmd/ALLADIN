import os
import tomllib
from typing import Any, Dict

def _load_env(filepath: str = ".env"):
    """Parse .env file and set environment variables if not already set."""
    if os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    key, val = line.split("=", 1)
                    key = key.strip()
                    val = val.strip().strip("'\"")
                    if key and key not in os.environ:
                        os.environ[key] = val

# Load .env variables first
_load_env()

DEFAULT_CONFIG = """
[general]
env = "development"
log_level = "INFO"

[database]
sqlite_db_path = "data/market.db"

[broker]
openalgo_api_url = "http://localhost:5000"
openalgo_api_key = ""

[risk]
max_position_size_pct = 0.05
max_sector_exposure_pct = 0.20
max_leverage = 1.0
max_daily_drawdown_pct = 0.02
max_weekly_drawdown_pct = 0.05
max_portfolio_drawdown_pct = 0.10
emergency_kill_switch_enabled = true
consecutive_loss_limit = 5
max_slippage_bps = 50
"""

class Settings:
    def __init__(self, config_path: str = "config/config.toml"):
        self.config_path = config_path
        self._settings: Dict[str, Any] = {}
        self.load()

    def load(self):
        # 1. Start with default config
        self._settings = tomllib.loads(DEFAULT_CONFIG)

        # 2. Load from config.toml if it exists
        if os.path.exists(self.config_path):
            with open(self.config_path, "rb") as f:
                toml_data = tomllib.load(f)
                self._update_recursive(self._settings, toml_data)

        # 3. Load overrides from Environment Variables
        self._apply_env_overrides()

    def _update_recursive(self, target: Dict[str, Any], source: Dict[str, Any]):
        for k, v in source.items():
            if isinstance(v, dict) and k in target and isinstance(target[k], dict):
                self._update_recursive(target[k], v)
            else:
                target[k] = v

    def _apply_env_overrides(self):
        # Translate environment variables like BROKER_OPENALGO_API_URL to nested keys
        for env_key, env_val in os.environ.items():
            parts = env_key.lower().split("_")
            if len(parts) < 2:
                continue
                
            section = parts[0]
            if section in self._settings and isinstance(self._settings[section], dict):
                key = "_".join(parts[1:])
                if key in self._settings[section]:
                    default_val = self._settings[section][key]
                    if isinstance(default_val, bool):
                        self._settings[section][key] = env_val.lower() in ("true", "1", "yes")
                    elif isinstance(default_val, int):
                        self._settings[section][key] = int(env_val)
                    elif isinstance(default_val, float):
                        self._settings[section][key] = float(env_val)
                    else:
                        self._settings[section][key] = env_val

    def get(self, path: str, default: Any = None) -> Any:
        """Get value using dot notation, e.g., settings.get("database.sqlite_db_path")"""
        parts = path.split(".")
        current = self._settings
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

# Global settings instance
settings = Settings()
