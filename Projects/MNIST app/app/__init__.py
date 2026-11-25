"""Flask application factory."""
from __future__ import annotations

import logging

from dotenv import load_dotenv
from flask import Flask

load_dotenv()  # reads .env if present (never committed)

from .config import get_config  # noqa: E402  (config reads env vars at import)
from .errors import register_error_handlers  # noqa: E402
from .logging_config import configure_logging  # noqa: E402
from .security import register_security_headers  # noqa: E402


def create_app(config_name: str | None = None, *, load_model: bool = True) -> Flask:
    config = get_config(config_name)
    configure_logging(config.LOG_LEVEL)
    logger = logging.getLogger(__name__)

    app = Flask(__name__)
    app.config.from_object(config)

    if not app.debug and not app.testing and app.config["SECRET_KEY"] == "dev-only-insecure-key":
        logger.warning("SECRET_KEY is not set; set it via environment variable in production.")

    register_error_handlers(app)
    register_security_headers(app)

    from .api import api_bp
    from .main import main_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp)

    if load_model:
        from .services import model_service

        model_service.load(app.config["MODEL_PATH"])

    logger.info("%s started (env=%s, preprocess=%s)", app.config["APP_NAME"], config.__name__, app.config["PREPROCESS_MODE"])
    return app
