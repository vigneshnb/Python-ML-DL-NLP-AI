"""Consistent JSON envelope for every API response.

Success: {"success": true,  "message": str, "data": {...}}
Error:   {"success": false, "message": str, "error": {"code": str}}
"""

from __future__ import annotations

from typing import Any

from flask import jsonify


def success(data: Any = None, message: str = "OK", status: int = 200):
    return jsonify({"success": True, "message": message, "data": data}), status


def error(message: str, code: str, status: int = 400):
    return (
        jsonify({"success": False, "message": message, "error": {"code": code}}),
        status,
    )
