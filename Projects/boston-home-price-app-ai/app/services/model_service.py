"""Loads the existing Keras model once per process and runs inference."""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

EXPECTED_INPUT_SHAPE = (None, 13)
EXPECTED_OUTPUT_SHAPE = (None, 1)


class ModelNotAvailableError(RuntimeError):
    """The model could not be loaded."""


class PredictionError(RuntimeError):
    """Inference failed."""


class ModelService:
    """Thread-safe, lazily-initialised wrapper around the .h5 model.

    The model is loaded at most once per process (Gunicorn worker). It is opened
    read-only with ``compile=False``: weights are never modified.
    """

    def __init__(self, model_path: Path | str, n_features: int = 13):
        self.model_path = Path(model_path)
        self.n_features = n_features
        self._model = None
        self._load_error: str | None = None
        self._load_lock = threading.Lock()
        self._predict_lock = threading.Lock()
        self.framework: dict[str, str] = {}

    # ------------------------------------------------------------------ status
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def input_shape(self) -> tuple:
        return tuple(self._model.input_shape) if self._model is not None else EXPECTED_INPUT_SHAPE

    @property
    def output_shape(self) -> tuple:
        return tuple(self._model.output_shape) if self._model is not None else EXPECTED_OUTPUT_SHAPE

    @property
    def status(self) -> str:
        if self.is_loaded:
            return "loaded"
        return "error" if self._load_error else "not_loaded"

    # ------------------------------------------------------------------ loading
    def load(self):
        if self._model is not None:
            return self._model
        with self._load_lock:
            if self._model is not None:
                return self._model
            if not self.model_path.is_file():
                self._load_error = "Model file not found."
                log.error("Model file missing: %s", self.model_path.name)
                raise ModelNotAvailableError(self._load_error)
            try:
                os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
                import keras  # imported lazily: heavy

                model = keras.models.load_model(self.model_path, compile=False)
            except Exception as exc:  # noqa: BLE001 - any loader failure
                self._load_error = "Model could not be loaded."
                log.exception("Failed to load model %s", self.model_path.name)
                raise ModelNotAvailableError(self._load_error) from exc

            self._validate(model)
            # Warm-up so the first user request does not pay graph-building cost.
            model(np.zeros((1, self.n_features), dtype=np.float32), training=False)
            try:
                import tensorflow as tf

                self.framework = {"tensorflow": tf.__version__, "keras": keras.__version__}
            except Exception:  # noqa: BLE001
                self.framework = {"keras": keras.__version__}
            self._model = model
            self._load_error = None
            log.info("Model loaded: %s in=%s out=%s", self.model_path.name,
                     model.input_shape, model.output_shape)
            return model

    def _validate(self, model) -> None:
        in_shape = tuple(model.input_shape)
        out_shape = tuple(model.output_shape)
        if len(in_shape) != 2 or in_shape[1] != self.n_features:
            self._load_error = "Model input shape is incompatible."
            log.error("Unexpected input shape %s (expected %s)", in_shape, EXPECTED_INPUT_SHAPE)
            raise ModelNotAvailableError(self._load_error)
        if len(out_shape) != 2 or out_shape[1] != 1:
            self._load_error = "Model output shape is incompatible."
            log.error("Unexpected output shape %s (expected %s)", out_shape, EXPECTED_OUTPUT_SHAPE)
            raise ModelNotAvailableError(self._load_error)

    # ------------------------------------------------------------------ inference
    def predict(self, x: np.ndarray) -> float:
        """Run the model on an already-preprocessed (1, n_features) float32 array."""
        model = self.load()
        if x.shape != (1, self.n_features):
            raise PredictionError("Model input has the wrong shape.")
        try:
            with self._predict_lock:
                y = model(x.astype(np.float32, copy=False), training=False)
            y = np.asarray(y)
        except Exception as exc:  # noqa: BLE001
            log.exception("Inference failed")
            raise PredictionError("Prediction failed.") from exc
        if y.shape != (1, 1) or not np.isfinite(y).all():
            raise PredictionError("Model returned an invalid output.")
        return float(y[0, 0])
