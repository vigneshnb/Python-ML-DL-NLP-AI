"""Flask application factory."""

from __future__ import annotations

import logging
import sys

from dotenv import load_dotenv
from flask import Flask, request
from werkzeug.exceptions import HTTPException

from app.config import BASE_DIR, get_config
from app.features import FEATURE_NAMES
from app.responses import error
from app.services.model_service import ModelNotAvailableError, ModelService
from app.services.prediction_service import PredictionService
from app.services.preprocessing import Preprocessor

CSP = (
    "default-src 'self'; "
    "script-src 'self'; "
    "style-src 'self' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data:; "
    "connect-src 'self'; "
    "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
)


def _configure_logging(level: str) -> None:
    root = logging.getLogger()
    if not any(getattr(h, "_hve_handler", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
        handler._hve_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(level)


def create_app(
    config_name: str | None = None,
    model_service: ModelService | None = None,
) -> Flask:
    load_dotenv(BASE_DIR / ".env")
    cfg = get_config(config_name)

    app = Flask(__name__)
    app.config.from_object(cfg)
    app.json.sort_keys = False  # keep feature_summary in model feature order
    _configure_logging(app.config["LOG_LEVEL"])
    log = logging.getLogger("app")

    if not app.config.get("SECRET_KEY"):
        raise RuntimeError("SECRET_KEY must be set (see .env.example).")

    # ---- preprocessing + model (one instance per process) ----
    preprocessor = Preprocessor(app.config["PREPROCESSING_PATH"])
    svc = model_service or ModelService(app.config["MODEL_PATH"], n_features=len(FEATURE_NAMES))
    app.extensions["preprocessor"] = preprocessor
    app.extensions["model_service"] = svc
    app.extensions["prediction_service"] = PredictionService(svc, preprocessor)
    if app.config["MODEL_EAGER_LOAD"]:
        try:
            svc.load()
        except ModelNotAvailableError:
            log.error("Starting without a model; /api/predict/ will return 503.")

    # ---- blueprints ----
    from app.api import api_bp
    from app.web import web_bp

    app.url_map.strict_slashes = False  # /api/predict and /api/predict/ both work
    app.register_blueprint(api_bp)
    app.register_blueprint(web_bp)

    # ---- security headers ----
    @app.after_request
    def _security_headers(resp):
        h = resp.headers
        h.setdefault("X-Content-Type-Options", "nosniff")
        h.setdefault("X-Frame-Options", "DENY")
        h.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        h.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=(), payment=()")
        h.setdefault("Cross-Origin-Opener-Policy", "same-origin")
        h.setdefault("Content-Security-Policy", CSP)
        if app.config["ENABLE_HSTS"]:
            h.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        if request.path.startswith("/api/"):
            h["Cache-Control"] = "no-store"
        return resp

    # ---- errors: never leak internals ----
    @app.errorhandler(HTTPException)
    def _http_error(exc: HTTPException):
        if request.path.startswith("/api/"):
            messages = {
                404: "Endpoint not found.",
                405: "Method not allowed.",
                413: "Request body is too large.",
            }
            return error(messages.get(exc.code, exc.name), exc.code or 500)
        return exc

    @app.errorhandler(Exception)
    def _unhandled(exc: Exception):
        log.exception("Unhandled error on %s", request.path)
        if request.path.startswith("/api/"):
            return error("An internal error occurred.", 500)
        return "Internal Server Error", 500

    return app
