"""Environment-based configuration."""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if not raw:
        return default
    p = Path(raw)
    return p if p.is_absolute() else BASE_DIR / p


class BaseConfig:
    APP_NAME = "Home Value Estimator"
    SECRET_KEY = os.getenv("SECRET_KEY", "")
    DEBUG = False
    TESTING = False
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    MODEL_PATH = _path("MODEL_PATH", BASE_DIR / "models" / "boston_home_price_prediction.h5")
    PREPROCESSING_PATH = _path("PREPROCESSING_PATH", BASE_DIR / "models" / "preprocessing.json")
    MODEL_EAGER_LOAD = _bool("MODEL_EAGER_LOAD", True)

    # A prediction request is a small JSON object (13 numbers); 16 KB is generous.
    MAX_CONTENT_LENGTH = int(os.getenv("MAX_CONTENT_LENGTH", 16 * 1024))
    ENABLE_HSTS = _bool("ENABLE_HSTS", False)


class DevelopmentConfig(BaseConfig):
    DEBUG = _bool("FLASK_DEBUG", True)
    SECRET_KEY = os.getenv("SECRET_KEY") or "dev-only-insecure-key"


class TestingConfig(BaseConfig):
    TESTING = True
    SECRET_KEY = "testing-key"
    MODEL_EAGER_LOAD = False
    LOG_LEVEL = "WARNING"


class ProductionConfig(BaseConfig):
    DEBUG = False


_CONFIGS = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def get_config(name: str | None = None) -> type[BaseConfig]:
    name = (name or os.getenv("APP_ENV") or "production").strip().lower()
    if name not in _CONFIGS:
        raise ValueError(f"Unknown APP_ENV '{name}'. Use one of: {', '.join(_CONFIGS)}")
    return _CONFIGS[name]
