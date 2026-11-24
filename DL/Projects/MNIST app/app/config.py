"""Environment-based configuration."""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


class BaseConfig:
    APP_NAME = "MNIST AI"
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-insecure-key")
    DEBUG = False
    TESTING = False

    # Model
    MODEL_PATH = os.getenv("MODEL_PATH", str(BASE_DIR / "models" / "digit_brain_intel.h5"))
    # "mnist": MNIST-style normalisation (invert, crop, 20x20 fit, centre-of-mass in 28x28)
    # "notebook": exact replication of the notebook (grayscale -> resize 28x28 -> /255)
    PREPROCESS_MODE = os.getenv("PREPROCESS_MODE", "mnist").strip().lower()

    # Upload validation
    MAX_CONTENT_LENGTH = _env_int("MAX_UPLOAD_MB", 5) * 1024 * 1024
    ALLOWED_EXTENSIONS = frozenset({"png", "jpg", "jpeg"})
    ALLOWED_FORMATS = frozenset({"PNG", "JPEG"})
    ALLOWED_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/jpg", "image/pjpeg"})
    MIN_IMAGE_DIM = _env_int("MIN_IMAGE_DIM", 8)
    MAX_IMAGE_DIM = _env_int("MAX_IMAGE_DIM", 4096)

    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
    JSON_SORT_KEYS = False


class DevelopmentConfig(BaseConfig):
    DEBUG = _env_bool("FLASK_DEBUG", True)
    LOG_LEVEL = os.getenv("LOG_LEVEL", "DEBUG").upper()


class TestingConfig(BaseConfig):
    TESTING = True
    MAX_CONTENT_LENGTH = 1 * 1024 * 1024  # small limit to test oversize handling


class ProductionConfig(BaseConfig):
    DEBUG = False
    SESSION_COOKIE_SECURE = True
    SESSION_COOKIE_HTTPONLY = True
    SESSION_COOKIE_SAMESITE = "Lax"


CONFIG_BY_NAME = {
    "development": DevelopmentConfig,
    "testing": TestingConfig,
    "production": ProductionConfig,
}


def get_config(name: str | None = None) -> type[BaseConfig]:
    name = (name or os.getenv("APP_ENV", "production")).strip().lower()
    return CONFIG_BY_NAME.get(name, ProductionConfig)
