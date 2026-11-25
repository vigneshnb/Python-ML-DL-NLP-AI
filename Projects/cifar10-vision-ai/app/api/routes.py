"""JSON API: /api/health/, /api/model-info/, /api/predict/"""

from __future__ import annotations

import logging
import time

from flask import current_app, request

from app.api import api_bp
from app.constants import APP_NAME, CLASS_NAMES, UPLOAD_FIELD_NAME
from app.responses import error, success
from app.services import (
    ImageValidationError,
    InferenceError,
    ModelNotAvailableError,
    ModelService,
    load_and_validate,
    preprocess,
)

logger = logging.getLogger(__name__)


def _model_service() -> ModelService:
    return current_app.extensions["model_service"]


@api_bp.get("/health/")
def health():
    svc = _model_service()
    return success(
        {
            "status": "ok",
            "app_name": APP_NAME,
            "model_loaded": svc.is_loaded,
            "model_status": "ready" if svc.is_loaded else ("error" if svc.load_error else "not_loaded"),
        },
        "Service is healthy",
    )


@api_bp.get("/model-info/")
def model_info():
    svc = _model_service()
    try:
        svc.load()
    except ModelNotAvailableError:
        return error("The AI model is currently unavailable.", "model_unavailable", 503)
    return success(
        {
            "model_name": svc.model_path.name,
            "input_shape": list(svc.input_shape),
            "output_shape": list(svc.output_shape),
            "output_type": "probabilities (softmax)" if svc.outputs_probabilities else "logits",
            "num_classes": len(CLASS_NAMES),
            "class_names": list(CLASS_NAMES),
            "preprocessing": {
                "color_mode": "RGB",
                "resize": f"{svc.input_shape[2]}x{svc.input_shape[1]} (bicubic)",
                "scaling": "pixel / 255.0 -> [0, 1]",
                "dtype": "float32",
            },
            "framework": svc.framework,
        },
        "Model information",
    )


@api_bp.post("/predict/")
def predict():
    cfg = current_app.config
    file = request.files.get(UPLOAD_FIELD_NAME)
    if file is None or not file.filename:
        return error("No image was provided. Please choose an image to upload.", "missing_file")

    data = file.read(cfg["MAX_CONTENT_LENGTH"] + 1)
    if len(data) > cfg["MAX_CONTENT_LENGTH"]:
        return error(
            f"File is too large. Maximum size is {cfg['MAX_UPLOAD_MB']} MB.", "file_too_large", 413
        )

    svc = _model_service()
    try:
        svc.load()
    except ModelNotAvailableError:
        return error("The AI model is currently unavailable.", "model_unavailable", 503)

    started = time.perf_counter()
    try:
        img = load_and_validate(
            data,
            file.filename,
            min_dim=cfg["MIN_IMAGE_DIMENSION"],
            max_dim=cfg["MAX_IMAGE_DIMENSION"],
        )
        batch = preprocess(img.image, svc.input_shape)
        result = svc.predict(batch)
    except ImageValidationError as exc:
        return error(exc.message, exc.code, exc.status)
    except InferenceError:
        return error("The model could not classify this image.", "inference_failed", 500)
    total_ms = round((time.perf_counter() - started) * 1000, 2)

    logger.info(
        "prediction label=%s confidence=%.2f%% size=%dx%d fmt=%s time=%.1fms",
        result.label, result.confidence, img.width, img.height, img.format, total_ms,
    )
    return success(
        {
            "predicted_class": result.class_index,
            "predicted_label": result.label,
            "confidence": result.confidence,
            "probabilities": result.probabilities,
            "execution_time_ms": total_ms,
            "inference_time_ms": result.execution_time_ms,
            "image": {"width": img.width, "height": img.height, "format": img.format},
        },
        "Image classified successfully",
    )
