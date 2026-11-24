"""Loads the trained Keras model once and exposes thread-safe inference."""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

# Quieten TensorFlow before it is imported.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from ..errors import ModelNotReadyError  # noqa: E402

logger = logging.getLogger(__name__)


class ModelService:
    """Wraps the user's trained MNIST model (digit_brain_intel.h5).

    The model is loaded a single time per process. Inference is guarded by a lock so
    that a threaded server (Flask dev server, gunicorn gthread workers) is safe.
    """

    def __init__(self) -> None:
        self._model = None
        self._lock = threading.Lock()
        self._load_lock = threading.Lock()
        self.model_path: str | None = None
        self.load_error: str | None = None
        self.load_time_ms: float | None = None
        self.loaded_at: float | None = None

    # ------------------------------------------------------------------ loading
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self, model_path: str) -> None:
        with self._load_lock:
            if self._model is not None and self.model_path == model_path:
                return
            path = Path(model_path)
            self.model_path = str(path)
            if not path.is_file():
                self.load_error = f"Model file not found: {path}"
                logger.error(self.load_error)
                return
            start = time.perf_counter()
            try:
                import keras  # imported lazily: heavy dependency

                model = keras.models.load_model(path, compile=False)
                # Warm-up so the first real request is not slow.
                model(np.zeros((1, 28, 28), dtype=np.float32), training=False)
            except Exception as exc:  # pragma: no cover - depends on corrupt model files
                self.load_error = f"Failed to load model: {exc}"
                logger.exception("Failed to load model from %s", path)
                return
            self._model = model
            self.load_error = None
            self.load_time_ms = round((time.perf_counter() - start) * 1000, 1)
            self.loaded_at = time.time()
            logger.info(
                "Model loaded from %s in %.1f ms (input=%s, output=%s)",
                path, self.load_time_ms, self.input_shape, self.output_shape,
            )

    # ---------------------------------------------------------------- metadata
    @property
    def input_shape(self) -> tuple | None:
        if self._model is None:
            return None
        return (None, *tuple(self._model.input_shape[1:]))

    @property
    def output_shape(self) -> tuple | None:
        if self._model is None:
            return None
        return (None, *tuple(self._model.output_shape[1:]))

    def metadata(self) -> dict[str, Any]:
        info: dict[str, Any] = {
            "loaded": self.is_loaded,
            "file": Path(self.model_path).name if self.model_path else None,
            "format": "Keras HDF5 (.h5)",
            "error": self.load_error,
        }
        if self._model is None:
            return info
        import keras

        layers = []
        for layer in self._model.layers:
            entry = {"name": layer.name, "type": layer.__class__.__name__, "params": int(layer.count_params())}
            cfg = layer.get_config()
            if "units" in cfg:
                entry["units"] = cfg["units"]
            if "activation" in cfg:
                entry["activation"] = cfg["activation"]
            layers.append(entry)
        info.update(
            {
                "name": self._model.name,
                "framework": f"Keras {keras.__version__}",
                "input_shape": list(self.input_shape),
                "output_shape": list(self.output_shape),
                "num_classes": int(self._model.output_shape[-1]),
                "total_params": int(self._model.count_params()),
                "layers": layers,
                "load_time_ms": self.load_time_ms,
            }
        )
        return info

    # --------------------------------------------------------------- inference
    def predict(self, batch: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ModelNotReadyError(self.load_error or "The AI model is not loaded.")
        with self._lock:
            output = self._model(batch, training=False)
        return np.asarray(output, dtype=np.float64)


# Process-wide singleton
model_service = ModelService()
