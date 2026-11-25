"""JSON API: /api/predict/, /api/health/, /api/model-info/."""

from __future__ import annotations

import logging

from flask import current_app, request

from app.api import api_bp
from app.features import FEATURE_NAMES
from app.responses import error, success
from app.services.model_service import ModelNotAvailableError, PredictionError
from app.services.validation import ValidationFailed

log = logging.getLogger(__name__)


def _ext(name: str):
    return current_app.extensions[name]


@api_bp.post("/predict/")
def predict():
    if not request.is_json:
        return error("Content-Type must be application/json.", 415)
    payload = request.get_json(silent=True)
    if payload is None:
        return error("Request body is not valid JSON.", 400)

    try:
        result = _ext("prediction_service").predict(payload)
    except ValidationFailed as exc:
        return error(exc.message, 422, exc.errors)
    except ModelNotAvailableError:
        return error("The prediction model is currently unavailable.", 503)
    except PredictionError:
        return error("Prediction failed. Please try again.", 500)

    log.info("prediction ok value=%.4f time_ms=%.2f", result["predicted_value"], result["execution_time_ms"])
    return success("Prediction completed successfully", result)


@api_bp.get("/health/")
def health():
    svc = _ext("model_service")
    return success(
        "Service is running",
        {
            "status": "ok" if svc.is_loaded else "degraded",
            "application": current_app.config["APP_NAME"],
            "model_status": svc.status,
            "model_loaded": svc.is_loaded,
        },
    )


@api_bp.get("/model-info/")
def model_info():
    svc = _ext("model_service")
    pre = _ext("preprocessor")
    return success(
        "Model information",
        {
            "model_name": svc.model_path.name,
            "model_status": svc.status,
            "framework": svc.framework or {"keras": "not loaded yet"},
            "input_shape": list(svc.input_shape),
            "output_shape": list(svc.output_shape),
            "input_dtype": "float32",
            "feature_count": len(FEATURE_NAMES),
            "feature_names": list(FEATURE_NAMES),
            "feature_order_verified": True,
            "preprocessing": {
                "type": pre.meta["scaler"],
                "fitted_on": f"{pre.meta['fitted_on']} training split ({pre.meta['train_rows']} rows)",
                "formula": "(x - mean) / scale",
            },
            "output": {
                "target": pre.target.get("name"),
                "description": pre.target.get("description"),
                "unit": pre.target.get("unit"),
                "training_range": [pre.target.get("train_min"), pre.target.get("train_max")],
            },
        },
    )
