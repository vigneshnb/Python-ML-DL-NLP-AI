"""Input preprocessing that reproduces the training pipeline.

Training (see scripts/export_preprocessing.py) used a StandardScaler fitted on the
training split of keras.datasets.boston_housing. The fitted parameters are loaded
from ``models/preprocessing.json``; no parameter is invented or hard-coded here.

    model_input = (x - mean) / scale        # float32, shape (1, 13)
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np

from app.features import FEATURE_NAMES

log = logging.getLogger(__name__)


class PreprocessingError(RuntimeError):
    """Preprocessing artefact missing or inconsistent with the feature schema."""


class Preprocessor:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PreprocessingError("Preprocessing parameters file not found.") from exc
        except (OSError, json.JSONDecodeError) as exc:
            raise PreprocessingError("Preprocessing parameters file is unreadable.") from exc

        order = raw.get("feature_order")
        if list(order or []) != list(FEATURE_NAMES):
            raise PreprocessingError(
                "Feature order in preprocessing parameters does not match the feature schema."
            )
        n = len(FEATURE_NAMES)
        arrays = {}
        for key in ("mean", "scale", "train_min", "train_max", "train_median"):
            arr = np.asarray(raw.get(key, []), dtype=np.float64)
            if arr.shape != (n,) or not np.all(np.isfinite(arr)):
                raise PreprocessingError(f"Preprocessing parameter '{key}' is invalid.")
            arrays[key] = arr
        if np.any(arrays["scale"] <= 0):
            raise PreprocessingError("Preprocessing 'scale' values must be positive.")

        self.mean = arrays["mean"]
        self.scale = arrays["scale"]
        self.train_min = arrays["train_min"]
        self.train_max = arrays["train_max"]
        self.train_median = arrays["train_median"]
        self.target = dict(raw.get("target") or {})
        self.model_sha256 = raw.get("model_sha256")
        self.meta = {
            "scaler": raw.get("scaler", {}).get("type", "StandardScaler"),
            "fitted_on": raw.get("dataset", {}).get("source"),
            "train_rows": raw.get("dataset", {}).get("train_rows"),
        }
        log.info("Loaded preprocessing parameters (%s, %d features)", self.meta["scaler"], n)

    # ------------------------------------------------------------------ helpers
    def vector(self, values: dict[str, float]) -> np.ndarray:
        """Assemble raw values in the exact model feature order -> shape (1, 13)."""
        missing = [name for name in FEATURE_NAMES if name not in values]
        if missing:
            raise ValueError(f"Missing features: {', '.join(missing)}")
        row = [float(values[name]) for name in FEATURE_NAMES]
        if not all(math.isfinite(v) for v in row):
            raise ValueError("Feature values must be finite numbers.")
        return np.asarray([row], dtype=np.float64)

    def transform(self, values: dict[str, float]) -> np.ndarray:
        """Raw feature dict -> standardised float32 model input of shape (1, 13)."""
        x = self.vector(values)
        return ((x - self.mean) / self.scale).astype(np.float32)

    def training_ranges(self) -> dict[str, tuple[float, float]]:
        return {
            name: (float(lo), float(hi))
            for name, lo, hi in zip(FEATURE_NAMES, self.train_min, self.train_max)
        }

    def out_of_training_range(self, values: dict[str, float]) -> list[str]:
        ranges = self.training_ranges()
        return [n for n in FEATURE_NAMES if not (ranges[n][0] <= values[n] <= ranges[n][1])]
