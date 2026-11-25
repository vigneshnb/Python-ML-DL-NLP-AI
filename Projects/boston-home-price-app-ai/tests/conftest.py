"""Shared fixtures.

* ``stub_*`` fixtures use a fake model: for isolated API/validation unit tests only.
* ``real_*`` fixtures load the actual models/boston_home_price_prediction.h5.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "boston_home_price_prediction.h5"

# Sample row from the training notebook ("9. Model Prediction (New Observation)").
NOTEBOOK_SAMPLE = {
    "CRIM": 0.02, "ZN": 18, "INDUS": 2.5, "CHAS": 0, "NOX": 0.45, "RM": 6.5, "AGE": 65,
    "DIS": 4.5, "RAD": 1, "TAX": 300, "PTRATIO": 15, "B": 390, "LSTAT": 5,
}


class StubModelService:
    """Stand-in for ModelService: deterministic fake output, no TensorFlow."""

    def __init__(self, fail_load: bool = False, fail_predict: bool = False, value: float = 22.5):
        self.model_path = MODEL_PATH
        self.fail_load = fail_load
        self.fail_predict = fail_predict
        self.value = value
        self.framework = {"keras": "stub"}
        self.calls: list[np.ndarray] = []

    is_loaded = property(lambda self: not self.fail_load)
    status = property(lambda self: "error" if self.fail_load else "loaded")
    input_shape = (None, 13)
    output_shape = (None, 1)

    def load(self):
        from app.services.model_service import ModelNotAvailableError

        if self.fail_load:
            raise ModelNotAvailableError("Model file not found.")
        return self

    def predict(self, x):
        from app.services.model_service import PredictionError

        self.load()
        self.calls.append(x)
        if self.fail_predict:
            raise PredictionError("Prediction failed.")
        return self.value


@pytest.fixture
def stub_model():
    return StubModelService()


@pytest.fixture
def stub_app(stub_model):
    from app import create_app

    return create_app("testing", model_service=stub_model)


@pytest.fixture
def stub_client(stub_app):
    return stub_app.test_client()


@pytest.fixture
def sample():
    return dict(NOTEBOOK_SAMPLE)


@pytest.fixture(scope="session")
def real_app():
    if not MODEL_PATH.is_file():
        pytest.skip("Model file not present")
    from app import create_app

    app = create_app("testing")
    app.extensions["model_service"].load()
    return app


@pytest.fixture(scope="session")
def real_client(real_app):
    return real_app.test_client()
