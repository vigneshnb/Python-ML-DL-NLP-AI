"""Real-model tests: these load the actual trained cifar10_pred.h5 and run
genuine inference. No mocks."""

import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.constants import CLASS_NAMES

SAMPLES = Path(__file__).parent / "samples"


def test_model_loads(real_app):
    svc = real_app.extensions["model_service"]
    assert svc.is_loaded
    assert svc.load_error is None


def test_model_input_shape(real_app):
    assert real_app.extensions["model_service"].input_shape == (None, 32, 32, 3)


def test_model_output_shape_and_softmax(real_app):
    svc = real_app.extensions["model_service"]
    assert svc.output_shape == (None, 10)
    assert svc.outputs_probabilities is True  # final layer is softmax -> no 2nd softmax


def test_probabilities_sum_to_one(real_app):
    svc = real_app.extensions["model_service"]
    rng = np.random.default_rng(1)
    batch = rng.random((1, 32, 32, 3), dtype=np.float32)
    p = svc.predict_proba(batch)
    assert p.shape == (1, 10)
    assert np.isclose(p.sum(), 1.0, atol=1e-4)
    assert (p >= 0).all()


def test_wrong_tensor_shape_rejected(real_app):
    from app.services.model_service import InferenceError

    with pytest.raises(InferenceError):
        real_app.extensions["model_service"].predict_proba(np.zeros((1, 28, 28, 3), np.float32))


def test_model_info_api(real_client):
    r = real_client.get("/api/model-info/")
    d = r.get_json()["data"]
    assert r.status_code == 200
    assert d["model_name"] == "cifar10_pred.h5"
    assert d["input_shape"] == [None, 32, 32, 3]
    assert d["output_shape"] == [None, 10]
    assert d["class_names"] == list(CLASS_NAMES)
    assert "tensorflow" in d["framework"]


def test_health_reports_model_ready(real_client):
    d = real_client.get("/api/health/").get_json()["data"]
    assert d["model_loaded"] is True and d["model_status"] == "ready"


@pytest.mark.parametrize(
    "filename, expected",
    [("horse.png", "Horse"), ("bird.png", "Bird")],
)
def test_prediction_api_real_samples(real_client, filename, expected):
    """Sample images from the project (Screenshot1/2) -> known labels."""
    path = SAMPLES / filename
    if not path.is_file():
        pytest.skip(f"sample {filename} missing")
    r = real_client.post(
        "/api/predict/",
        data={"image": (io.BytesIO(path.read_bytes()), filename)},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200, r.get_json()
    d = r.get_json()["data"]
    assert d["predicted_label"] == expected
    assert d["predicted_class"] == CLASS_NAMES.index(expected)
    assert list(d["probabilities"]) == list(CLASS_NAMES)
    assert abs(sum(d["probabilities"].values()) - 100) < 0.2
    assert d["confidence"] == max(d["probabilities"].values())
    assert d["execution_time_ms"] > 0


def test_api_matches_direct_model_call(real_app, real_client):
    """The API's numbers are exactly what the model produces (nothing fabricated)."""
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, (48, 48, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")

    r = real_client.post(
        "/api/predict/",
        data={"image": (io.BytesIO(buf.getvalue()), "rand.png")},
        content_type="multipart/form-data",
    )
    d = r.get_json()["data"]

    import keras

    model = keras.models.load_model(str(real_app.extensions["model_service"].model_path), compile=False)
    x = np.expand_dims(
        np.asarray(Image.fromarray(arr).resize((32, 32), Image.Resampling.BICUBIC), np.float32) / 255.0, 0
    )
    direct = model.predict(x, verbose=0)[0]
    assert d["predicted_class"] == int(np.argmax(direct))
    for name, pct in d["probabilities"].items():
        assert pct == pytest.approx(round(float(direct[CLASS_NAMES.index(name)]) * 100, 2), abs=0.011)
