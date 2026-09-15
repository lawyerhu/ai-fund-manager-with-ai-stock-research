from pathlib import Path
import os
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = ROOT / ".env"


def load_project_env(path: str | Path | None = None, *, override: bool = False) -> Path:
    """Load the project environment independently of the process working directory."""
    env_path = Path(path) if path is not None else ENV_PATH
    load_dotenv(dotenv_path=env_path, override=override)
    return env_path


load_project_env()


def load_config(path: str | Path | None = None) -> dict:
    config_path = Path(path) if path else ROOT / "config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    execution = config.setdefault("execution", {})
    overrides = {
        "ENTRY_ENGINE_ENABLED": ("entry_engine_enabled", lambda value: value.strip().lower() in {"1", "true", "yes", "on"}),
        "ENTRY_DELAY_AFTER_OPEN_SECONDS": ("entry_delay_after_open_seconds", int),
        "ENTRY_MAX_QUOTE_AGE_SECONDS": ("entry_max_quote_age_seconds", float),
        "ENTRY_LIMIT_SLIPPAGE_BPS": ("limit_slippage_bps", float),
        "ENTRY_MAX_TOTAL_SLIPPAGE_BPS": ("max_total_slippage_bps", float),
        "ENTRY_ORDER_TIMEOUT_SECONDS": ("entry_order_timeout_seconds", float),
        "MAX_ENTRY_REQUOTES": ("max_entry_requotes", int),
    }
    for env_name, (key, converter) in overrides.items():
        value = os.getenv(env_name)
        if value is not None:
            execution[key] = converter(value)
    return config


def env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)
