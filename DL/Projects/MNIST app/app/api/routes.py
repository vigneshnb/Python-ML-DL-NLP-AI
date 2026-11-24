"""JSON API: /api/predict/, /api/health/, /api/model-info/."""
from __future__ import annotations

import logging
import time

from flask import Blueprint, current_app, jsonify, request

from ..services import PredictionService, model_service
from ..services.validation import validate_upload

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")
_started_at = time.time()


def success(message: str, data: dict, status: int = 200):
    return jsonify({"success": True, "message": message, "data": data}), status


@api_bp.post("/predict/", strict_slashes=False)
def predict():
    upload = request.files.get("image") or request.files.get("file")
    filename, raw, image = validate_upload(upload)

    service = PredictionService(model_service, current_app.config["PREPROCESS_MODE"])
    result = service.predict(image)

    data = result.to_dict()
    data.update(
        {
            "filename": filename,
            "image": {"width": image.width, "height": image.height, "format": image.format, "size_bytes": len(raw)},
            "preprocess_mode": service.preprocess_mode,
        }
    )
    return success("Prediction completed successfully", data)


@api_bp.get("/health/", strict_slashes=False)
def health():
    loaded = model_service.is_loaded
    data = {
        "status": "healthy" if loaded else "degraded",
        "model_loaded": loaded,
        "model_error": model_service.load_error,
        "uptime_seconds": round(time.time() - _started_at, 1),
    }
    return success("Service is healthy" if loaded else "Model is not loaded", data, 200 if loaded else 503)


@api_bp.get("/model-info/", strict_slashes=False)
def model_info():
    data = model_service.metadata()
    data["preprocessing"] = {
        "mode": current_app.config["PREPROCESS_MODE"],
        "input": "28x28 grayscale, white digit on black, scaled to [0, 1] (x / 255)",
        "steps": (
            ["grayscale", "resize 28x28", "divide by 255"]
            if current_app.config["PREPROCESS_MODE"] == "notebook"
            else [
                "grayscale (transparency composited on white)",
                "auto-invert light backgrounds",
                "background removal + contrast stretch",
                "crop to digit bounding box",
                "fit into 20x20 box (aspect preserved)",
                "pad to 28x28, centre by centre of mass",
                "divide by 255",
            ]
        ),
    }
    data["upload_limits"] = {
        "max_size_mb": current_app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024),
        "allowed_extensions": sorted(current_app.config["ALLOWED_EXTENSIONS"]),
        "min_dimension": current_app.config["MIN_IMAGE_DIM"],
        "max_dimension": current_app.config["MAX_IMAGE_DIM"],
    }
    return success("Model information", data)
