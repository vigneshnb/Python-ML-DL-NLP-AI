"""Custom exceptions and JSON error handlers with a consistent response format."""
from __future__ import annotations

import logging

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base application error rendered as a JSON response."""

    status_code = 400
    code = "bad_request"

    def __init__(self, message: str, *, status_code: int | None = None, code: str | None = None):
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if code is not None:
            self.code = code


class ValidationError(AppError):
    status_code = 400
    code = "validation_error"


class UnsupportedMediaError(AppError):
    status_code = 415
    code = "unsupported_media_type"


class PreprocessingError(AppError):
    status_code = 422
    code = "preprocessing_error"


class ModelNotReadyError(AppError):
    status_code = 503
    code = "model_unavailable"


def error_response(message: str, status_code: int, code: str):
    body = {"success": False, "message": message, "error": {"code": code, "status": status_code}}
    return jsonify(body), status_code


def register_error_handlers(app: Flask) -> None:
    @app.errorhandler(AppError)
    def handle_app_error(err: AppError):
        logger.info("Request rejected (%s): %s", err.code, err.message)
        return error_response(err.message, err.status_code, err.code)

    @app.errorhandler(413)
    def handle_too_large(_err):
        limit_mb = app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)
        return error_response(
            f"File is too large. Maximum upload size is {limit_mb:g} MB.", 413, "payload_too_large"
        )

    @app.errorhandler(HTTPException)
    def handle_http_error(err: HTTPException):
        if request.path.startswith("/api/"):
            return error_response(err.description or err.name, err.code or 500, err.name.lower().replace(" ", "_"))
        return err

    @app.errorhandler(Exception)
    def handle_unexpected(err: Exception):
        logger.exception("Unhandled error on %s %s", request.method, request.path)
        return error_response("An unexpected server error occurred.", 500, "internal_error")
