"""Flask application factory."""

from __future__ import annotations

import logging
import os
import sys

from dotenv import load_dotenv
from flask import Flask, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from app.config import BASE_DIR, get_config
from app.responses import error
from app.services.model_service import ModelNotAvailableError, ModelService

CSP = (
    "default-src 'self'; "
    "script-src 'self' https://cdn.jsdelivr.net; "
    "style-src 'self' https://fonts.googleapis.com; "
    "font-src 'self' https://fonts.gstatic.com; "
    "img-src 'self' data: blob:; "
    "connect-src 'self'; "
    "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'"
)


def _configure_logging(level: str) -> None:
    root = logging.getLogger()
    if not any(getattr(h, "_cifar_handler", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        )
        handler._cifar_handler = True  # type: ignore[attr-defined]
        root.addHandler(handler)
    root.setLevel(level)


def create_app(config_name: str | None = None, model_service: ModelService | None = None) -> Flask:
    load_dotenv(BASE_DIR / ".env")
    cfg = get_config(config_name)

    app = Flask(__name__)
    app.config.from_object(cfg)
    _configure_logging(app.config["LOG_LEVEL"])
    log = logging.getLogger("app")

    if not app.config.get("SECRET_KEY"):
        raise RuntimeError("SECRET_KEY must be set (see .env.example).")

    # ---- model (one instance per process) ----
    svc = model_service or ModelService(app.config["MODEL_PATH"])
    app.extensions["model_service"] = svc
    if app.config["MODEL_EAGER_LOAD"]:
        try:
            svc.load()
        except ModelNotAvailableError:
            log.error("Starting without a model; /api/predict/ will return 503.")

    # ---- blueprints ----
    from app.api import api_bp
    from app.web import web_bp

    app.register_blueprint(api_bp)
    app.register_blueprint(web_bp)

    # ---- security headers ----
    @app.after_request
    def _security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
        resp.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
        resp.headers.setdefault("Content-Security-Policy", CSP)
        if app.config["ENABLE_HSTS"]:
            resp.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        if request.path.startswith("/api/"):
            resp.headers["Cache-Control"] = "no-store"
        return resp

    # ---- error handlers (JSON for /api, never leak internals) ----
    @app.errorhandler(RequestEntityTooLarge)
    def _too_large(_exc):
        return error(
            f"File is too large. Maximum size is {app.config['MAX_UPLOAD_MB']} MB.",
            "file_too_large",
            413,
        )

    @app.errorhandler(HTTPException)
    def _http_error(exc: HTTPException):
        if request.path.startswith("/api/"):
            return error(exc.description or exc.name, exc.name.lower().replace(" ", "_"), exc.code or 500)
        return exc

    @app.errorhandler(Exception)
    def _unhandled(exc: Exception):
        log.exception("Unhandled error on %s %s", request.method, request.path)
        return error("An unexpected error occurred. Please try again.", "internal_error", 500)

    log.info("App created (env=%s, debug=%s, model=%s)", cfg.APP_ENV, app.debug, os.path.basename(app.config["MODEL_PATH"]))
    return app
