"""Orchestrates validation -> preprocessing -> inference -> presentation."""

from __future__ import annotations

import time

from app.features import FEATURE_NAMES, FEATURES_BY_NAME
from app.services.model_service import ModelService
from app.services.preprocessing import Preprocessor
from app.services.validation import validate_payload


def format_usd(thousands: float) -> str:
    return f"${thousands * 1000:,.0f}"


class PredictionService:
    def __init__(self, model_service: ModelService, preprocessor: Preprocessor):
        self.model = model_service
        self.pre = preprocessor

    def predict(self, payload) -> dict:
        values = validate_payload(payload)  # raises ValidationFailed

        start = time.perf_counter()
        x = self.pre.transform(values)
        raw = self.model.predict(x)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        target = self.pre.target
        lo, hi = float(target.get("train_min", 5.0)), float(target.get("train_max", 50.0))
        warnings: list[str] = []

        outside = self.pre.out_of_training_range(values)
        if outside:
            labels = ", ".join(f"{FEATURES_BY_NAME[n].label} ({n})" for n in outside)
            warnings.append(
                f"Outside the range seen in training: {labels}. The model is extrapolating, "
                "so treat this estimate with extra caution."
            )
        if raw < 0:
            warnings.append("The model returned a negative value, which has no monetary meaning.")
        elif raw > float(target.get("censored_at", 50.0)):
            warnings.append(
                "The estimate is above $50,000, the cap applied to every value in the "
                "original dataset; the model never saw higher targets."
            )

        z = x[0].tolist()
        return {
            "predicted_value": round(raw, 4),
            "display_value": format_usd(raw) if raw >= 0 else None,
            "unit": "thousands of US dollars (1970s)",
            "execution_time_ms": round(elapsed_ms, 2),
            "feature_summary": {name: values[name] for name in FEATURE_NAMES},
            "standardized_inputs": {name: round(float(v), 4) for name, v in zip(FEATURE_NAMES, z)},
            "target_context": {
                "train_min": lo,
                "train_max": hi,
                "train_median": float(target.get("train_median", 0.0)),
            },
            "warnings": warnings,
        }
