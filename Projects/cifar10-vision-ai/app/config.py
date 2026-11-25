"""Environment-based configuration.

Values are read from environment variables (optionally loaded from a .env
file by python-dotenv in ``app.create_app``). Production defaults are safe:
DEBUG is off and a missing SECRET_KEY is a hard error.
"""

from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


class BaseConfig:
    APP_ENV = "base"
    DEBUG = False
    TESTING = False
    SECRET_KEY = os.getenv("SECRET_KEY", "")

    # Model
    MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "cifar10_pred.h5"))
    # Load the model when the app is created (recommended). If False, it is
    # loaded lazily on the first prediction.
    MODEL_EAGER_LOAD = _env_bool("MODEL_EAGER_LOAD", True)

    # Upload limits
    MAX_UPLOAD_MB = _env_int("MAX_UPLOAD_MB", 5)
    MAX_CONTENT_LENGTH = MAX_UPLOAD_MB * 1024 * 1024
    MIN_IMAGE_DIMENSION = _env_int("MIN_IMAGE_DIMENSION", 8)
    MAX_IMAGE_DIMENSION = _env_int("MAX_IMAGE_DIMENSION", 6000)

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

    # Response headers
    ENABLE_HSTS = _env_bool("ENABLE_HSTS", False)


class DevelopmentConfig(BaseConfig):
    APP_ENV = "development"
    DEBUG = _env_bool("FLASK_DEBUG", True)
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-insecure-key")


class TestingConfig(BaseConfig):
    APP_ENV = "testing"
    TESTING = True
    SECRET_KEY = "testing-key"
    MODEL_EAGER_LOAD = False


class ProductionConfig(BaseConfig):
    APP_ENV = "production"
    DEBUG = False  # never configurable in production
    ENABLE_HSTS = _env_bool("ENABLE_HSTS", True)


CONFIG_BY_NAME: dict[str, type[BaseConfig]] = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def get_config(name: str | None = None) -> type[BaseConfig]:
    name = (name or os.getenv("APP_ENV", "production")).strip().lower()
    try:
        return CONFIG_BY_NAME[name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown APP_ENV '{name}'. Use one of: {', '.join(CONFIG_BY_NAME)}"
        ) from exc
