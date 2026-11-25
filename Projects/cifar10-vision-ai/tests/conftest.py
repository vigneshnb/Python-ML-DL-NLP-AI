"""Shared fixtures.

Two kinds of app fixtures:

* ``real_app`` / ``real_client`` -- loads the actual trained ``cifar10_pred.h5``.
  Used to verify real inference end to end.
* ``stub_app`` / ``stub_client`` -- injects ``StubModelService`` (a tiny
  deterministic stand-in) so request-validation tests run in isolation and
  don't depend on TensorFlow. These tests never assert on prediction quality.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from app import create_app  # noqa: E402
from app.services.model_service import ModelService  # noqa: E402

MODEL_PATH = ROOT / "models" / "cifar10_pred.h5"
SAMPLE_DIR = ROOT / "tests" / "samples"


# ----------------------------------------------------------------- helpers
def make_image_bytes(fmt: str = "PNG", size=(64, 64), color=(120, 80, 200), mode="RGB") -> bytes:
    img = Image.new(mode, size, color if mode != "L" else 128)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


class StubModelService(ModelService):
    """Isolated stand-in for unit tests of the HTTP layer (NOT real inference)."""

    def __init__(self):
        super().__init__("stub.h5")
        self.calls = 0

    @property
    def is_loaded(self):
        return True

    def load(self):
        self.framework = {"stub": "1"}

    @property
    def input_shape(self):
        return (None, 32, 32, 3)

    @property
    def output_shape(self):
        return (None, 10)

    @property
    def outputs_probabilities(self):
        return True

    def predict_proba(self, batch):
        self.calls += 1
        assert batch.shape == (1, 32, 32, 3) and batch.dtype == np.float32
        p = np.full((1, 10), 0.05)
        p[0, 3] = 0.55
        return p


# ----------------------------------------------------------------- fixtures
@pytest.fixture(scope="session")
def real_app():
    if not MODEL_PATH.is_file():
        pytest.skip("models/cifar10_pred.h5 not present")
    os.environ["MODEL_PATH"] = str(MODEL_PATH)
    app = create_app("testing")
    app.extensions["model_service"].load()
    return app


@pytest.fixture()
def real_client(real_app):
    return real_app.test_client()


@pytest.fixture()
def stub_app():
    return create_app("testing", model_service=StubModelService())


@pytest.fixture()
def stub_client(stub_app):
    return stub_app.test_client()


@pytest.fixture()
def png_bytes():
    return make_image_bytes("PNG")
