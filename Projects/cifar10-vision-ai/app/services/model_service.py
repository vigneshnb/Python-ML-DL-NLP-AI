"""Loads the trained Keras model once per process and runs inference.

Findings from inspecting ``cifar10_pred.h5`` (Keras 3 / TF 2.21, legacy HDF5):

* Architecture: Flatten -> Dense(512) -> Dropout -> Dense(256) -> Dropout ->
  Dense(128) -> Dropout -> Dense(10, softmax)
* Input  ``(None, 32, 32, 3)`` float32
* Output ``(None, 10)``; the last layer's activation is ``softmax`` so the
  output already *is* a probability distribution. Softmax is only applied here
  if a future model reports a ``linear`` (logits) final activation -- never twice.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.constants import CLASS_NAMES, NUM_CLASSES

logger = logging.getLogger(__name__)


class ModelNotAvailableError(RuntimeError):
    """The model could not be loaded or is not loaded yet."""


class InferenceError(RuntimeError):
    """The model failed while running a prediction."""


@dataclass(frozen=True)
class Prediction:
    class_index: int
    label: str
    confidence: float  # percentage 0-100
    probabilities: dict[str, float]  # label -> percentage
    execution_time_ms: float


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class ModelService:
    def __init__(self, model_path: str | os.PathLike):
        self.model_path = Path(model_path)
        self._model = None
        self._load_error: str | None = None
        self._outputs_probabilities = True
        self._load_lock = threading.Lock()
        # Keras models are not guaranteed thread-safe for concurrent calls;
        # serialise inference inside a process (inference is ~ms for this model).
        self._infer_lock = threading.Lock()
        self.framework: dict[str, str] = {}

    # ------------------------------------------------------------------ loading
    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def load(self) -> None:
        """Load the model (idempotent, thread-safe)."""
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            if not self.model_path.is_file():
                self._load_error = "Model file not found."
                logger.error("Model file not found at %s", self.model_path)
                raise ModelNotAvailableError(self._load_error)
            try:
                os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
                import keras
                import tensorflow as tf

                started = time.perf_counter()
                # compile=False: we only need inference, not the optimizer state.
                model = keras.models.load_model(str(self.model_path), compile=False)
                self._validate(model)
                self.framework = {
                    "tensorflow": tf.__version__,
                    "keras": keras.__version__,
                    "format": "HDF5 (.h5)",
                }
                # Warm-up so the first user request isn't slow.
                model(np.zeros((1, *model.input_shape[1:]), dtype=np.float32), training=False)
                self._model = model
                self._load_error = None
                logger.info(
                    "Model loaded from %s in %.0f ms (input=%s, output=%s, softmax_output=%s)",
                    self.model_path.name,
                    (time.perf_counter() - started) * 1000,
                    model.input_shape,
                    model.output_shape,
                    self._outputs_probabilities,
                )
            except ModelNotAvailableError:
                raise
            except Exception as exc:  # noqa: BLE001 - surface as unavailable
                self._load_error = "Model failed to load."
                logger.exception("Failed to load model: %s", exc)
                raise ModelNotAvailableError(self._load_error) from exc

    def _validate(self, model) -> None:
        in_shape = model.input_shape
        out_shape = model.output_shape
        if not (isinstance(in_shape, tuple) and len(in_shape) == 4):
            raise ModelNotAvailableError(f"Unexpected model input shape {in_shape}")
        if not (isinstance(out_shape, tuple) and out_shape[-1] == NUM_CLASSES):
            raise ModelNotAvailableError(
                f"Model must output {NUM_CLASSES} classes, got {out_shape}"
            )
        activation = None
        try:
            activation = model.layers[-1].get_config().get("activation")
        except Exception:  # noqa: BLE001
            pass
        if activation == "softmax":
            self._outputs_probabilities = True
        elif activation in (None, "linear"):
            self._outputs_probabilities = False
            logger.warning("Model output is treated as logits (activation=%s)", activation)
        else:
            raise ModelNotAvailableError(f"Unsupported output activation '{activation}'")

    # ---------------------------------------------------------------- metadata
    def _require_model(self):
        if self._model is None:
            self.load()
        return self._model

    @property
    def input_shape(self) -> tuple:
        return tuple(self._require_model().input_shape)

    @property
    def output_shape(self) -> tuple:
        return tuple(self._require_model().output_shape)

    @property
    def outputs_probabilities(self) -> bool:
        self._require_model()
        return self._outputs_probabilities

    # --------------------------------------------------------------- inference
    def predict_proba(self, batch: np.ndarray) -> np.ndarray:
        model = self._require_model()
        expected = tuple(model.input_shape[1:])
        if batch.ndim != 4 or tuple(batch.shape[1:]) != expected:
            raise InferenceError(
                f"Input tensor shape {batch.shape} does not match model input {expected}"
            )
        try:
            with self._infer_lock:
                raw = model(batch.astype(np.float32, copy=False), training=False)
            out = np.asarray(raw, dtype=np.float64)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Inference failed")
            raise InferenceError("Model inference failed.") from exc

        if out.shape != (batch.shape[0], NUM_CLASSES):
            raise InferenceError(f"Unexpected model output shape {out.shape}")
        probs = out if self._outputs_probabilities else stable_softmax(out)
        if not np.all(np.isfinite(probs)):
            raise InferenceError("Model returned non-finite values.")
        return probs

    def predict(self, batch: np.ndarray) -> Prediction:
        started = time.perf_counter()
        probs = self.predict_proba(batch)[0]
        elapsed_ms = (time.perf_counter() - started) * 1000
        idx = int(np.argmax(probs))
        return Prediction(
            class_index=idx,
            label=CLASS_NAMES[idx],
            confidence=round(float(probs[idx]) * 100, 2),
            probabilities={
                name: round(float(p) * 100, 2) for name, p in zip(CLASS_NAMES, probs)
            },
            execution_time_ms=round(elapsed_ms, 2),
        )
