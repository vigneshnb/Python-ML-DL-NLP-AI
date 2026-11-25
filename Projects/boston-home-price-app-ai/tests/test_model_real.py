"""Inference with the REAL model file (no stubs)."""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import MODEL_PATH, NOTEBOOK_SAMPLE

ROOT = Path(__file__).resolve().parent.parent
PRE = json.loads((ROOT / "models" / "preprocessing.json").read_text(encoding="utf-8"))

pytestmark = pytest.mark.real_model


def test_model_loads_once_with_expected_shapes(real_app):
    svc = real_app.extensions["model_service"]
    model = svc.load()
    assert svc.load() is model  # cached, not reloaded
    assert svc.is_loaded and svc.status == "loaded"
    assert tuple(model.input_shape) == (None, 13)
    assert tuple(model.output_shape) == (None, 1)
    assert model.inputs[0].dtype == "float32"
    assert "keras" in svc.framework


def test_real_inference_matches_independent_keras_predict(real_app):
    """Service output == plain keras model.predict on the same standardised row."""
    import keras

    pre = real_app.extensions["preprocessor"]
    x = pre.transform(NOTEBOOK_SAMPLE)
    reference = keras.models.load_model(MODEL_PATH, compile=False)
    expected = float(reference.predict(x, verbose=0)[0, 0])
    got = real_app.extensions["model_service"].predict(x)
    assert got == pytest.approx(expected, abs=1e-4)
    assert got == pytest.approx(PRE["verification"]["prediction_with_training_scaler"], abs=1e-4)


def test_provenance_notebook_value_is_reproduced(real_app):
    """The notebook printed 29.239614 using its X_test-fitted scaler; reproduce it exactly."""
    cache = Path.home() / ".keras" / "datasets" / "boston_housing.npz"
    if not cache.exists():
        pytest.skip("boston_housing.npz not cached")
    from keras.datasets import boston_housing

    _, (x_test, _) = boston_housing.load_data()
    mean, std = x_test.mean(axis=0), x_test.std(axis=0)
    row = np.array([[NOTEBOOK_SAMPLE[k] for k in PRE["feature_order"]]], dtype=np.float64)
    out = real_app.extensions["model_service"].predict(((row - mean) / std).astype(np.float32))
    assert out == pytest.approx(29.239614, abs=1e-4)


def test_test_split_error_is_reasonable(real_app):
    """Sanity: MAE on the held-out split with training-split scaling (~2.8 k$)."""
    cache = Path.home() / ".keras" / "datasets" / "boston_housing.npz"
    if not cache.exists():
        pytest.skip("boston_housing.npz not cached")
    from keras.datasets import boston_housing

    _, (x_test, y_test) = boston_housing.load_data()
    pre = real_app.extensions["preprocessor"]
    model = real_app.extensions["model_service"].load()
    x = ((x_test - pre.mean) / pre.scale).astype(np.float32)
    mae = float(np.abs(model.predict(x, verbose=0).ravel() - y_test).mean())
    assert mae < 3.5


def test_predict_endpoint_real_model(real_client):
    r = real_client.post("/api/predict/", json={"features": NOTEBOOK_SAMPLE})
    body = r.get_json()
    assert r.status_code == 200 and body["success"] is True
    d = body["data"]
    assert d["predicted_value"] == pytest.approx(28.0736, abs=1e-3)
    assert d["display_value"] == "$28,074"
    assert d["execution_time_ms"] > 0


def test_different_inputs_give_different_predictions(real_client):
    """Guards against hard-coded output: more rooms / lower LSTAT -> higher value."""
    low = dict(NOTEBOOK_SAMPLE, RM=5.0, LSTAT=25.0)
    high = dict(NOTEBOOK_SAMPLE, RM=7.8, LSTAT=3.0)
    v_low = real_client.post("/api/predict/", json=low).get_json()["data"]["predicted_value"]
    v_high = real_client.post("/api/predict/", json=high).get_json()["data"]["predicted_value"]
    assert v_high > v_low


def test_model_file_not_modified():
    import hashlib

    assert hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() == PRE["model_sha256"]
