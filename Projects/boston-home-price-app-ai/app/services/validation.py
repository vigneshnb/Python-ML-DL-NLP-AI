"""Validation of prediction request payloads against the central feature schema."""

from __future__ import annotations

import math
from typing import Any

from app.features import FEATURES, FEATURE_NAMES

MAX_FIELDS = 64  # refuse absurdly large objects before iterating them


class ValidationFailed(ValueError):
    def __init__(self, errors: dict[str, str], message: str = "Some inputs are invalid."):
        super().__init__(message)
        self.errors = errors
        self.message = message


def _coerce(feature, raw: Any) -> tuple[float | None, str | None]:
    # JSON booleans are ints in Python - reject them explicitly.
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        return None, "Must be a number."
    value = float(raw)
    if not math.isfinite(value):
        return None, "Must be a finite number."
    if feature.dtype in {"int", "binary"} and not value.is_integer():
        return None, "Must be a whole number."
    if feature.dtype == "binary" and value not in (0.0, 1.0):
        return None, "Must be 0 or 1."
    if value < feature.min or value > feature.max:
        return None, f"Must be between {feature.min:g} and {feature.max:g}."
    return value, None


def validate_payload(payload: Any) -> dict[str, float]:
    """Return {feature: value} in model order or raise ValidationFailed.

    Accepts either ``{"features": {...}}`` or a flat ``{...}`` object.
    Missing values are errors - nothing is defaulted silently.
    """
    if not isinstance(payload, dict):
        raise ValidationFailed({}, "Request body must be a JSON object.")
    data = payload["features"] if "features" in payload else payload
    if not isinstance(data, dict):
        raise ValidationFailed({}, "'features' must be a JSON object.")
    if len(data) > MAX_FIELDS:
        raise ValidationFailed({}, "Too many fields in request.")

    errors: dict[str, str] = {}
    unknown = [k for k in data if k not in FEATURE_NAMES]
    for key in unknown[:10]:
        errors[str(key)[:40]] = "Unknown feature."

    values: dict[str, float] = {}
    for feature in FEATURES:
        if feature.name not in data or data[feature.name] is None or data[feature.name] == "":
            errors[feature.name] = "This value is required."
            continue
        value, err = _coerce(feature, data[feature.name])
        if err:
            errors[feature.name] = err
        else:
            values[feature.name] = value

    if errors:
        raise ValidationFailed(errors)
    return {name: values[name] for name in FEATURE_NAMES}
