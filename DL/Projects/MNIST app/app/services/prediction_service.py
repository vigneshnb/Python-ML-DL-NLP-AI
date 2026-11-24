"""Orchestrates preprocessing + inference and shapes the result."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .model_service import ModelService
from .preprocessing import preprocess, to_data_url

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    predicted_digit: int
    confidence: float
    probabilities: dict[str, float]
    execution_time_ms: float
    processed_image: str

    def to_dict(self) -> dict:
        return {
            "predicted_digit": self.predicted_digit,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "execution_time_ms": self.execution_time_ms,
            "processed_image": self.processed_image,
        }


class PredictionService:
    def __init__(self, model: ModelService, preprocess_mode: str = "mnist") -> None:
        self.model = model
        self.preprocess_mode = preprocess_mode

    def predict(self, image: Image.Image) -> PredictionResult:
        start = time.perf_counter()
        batch = preprocess(image, self.preprocess_mode)
        probs = self.model.predict(batch)[0]
        elapsed_ms = (time.perf_counter() - start) * 1000

        digit = int(np.argmax(probs))
        percentages = {str(i): round(float(p) * 100, 2) for i, p in enumerate(probs)}
        result = PredictionResult(
            predicted_digit=digit,
            confidence=round(float(probs[digit]) * 100, 2),
            probabilities=percentages,
            execution_time_ms=round(elapsed_ms, 2),
            processed_image=to_data_url(batch[0]),
        )
        logger.info("Predicted %d (%.2f%%) in %.1f ms", digit, result.confidence, elapsed_ms)
        return result
