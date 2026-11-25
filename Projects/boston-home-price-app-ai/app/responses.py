"""Consistent JSON response envelope: {success, message, data | errors}."""

from __future__ import annotations

from typing import Any

from flask import jsonify


def success(message: str, data: Any = None, status: int = 200):
    return jsonify({"success": True, "message": message, "data": data}), status


def error(message: str, status: int = 400, errors: Any = None):
    body: dict[str, Any] = {"success": False, "message": message}
    if errors:
        body["errors"] = errors
    return jsonify(body), status
