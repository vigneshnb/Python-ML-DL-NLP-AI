import numpy as np
import pytest
from PIL import Image

from app.services.model_service import stable_softmax
from app.services.preprocessing import ImageValidationError, load_and_validate, preprocess
from tests.conftest import make_image_bytes


def test_preprocess_shape_dtype_range():
    img = Image.new("RGB", (100, 60), (255, 0, 128))
    x = preprocess(img, (None, 32, 32, 3))
    assert x.shape == (1, 32, 32, 3)
    assert x.dtype == np.float32
    assert x.min() >= 0.0 and x.max() <= 1.0
    # RGB order preserved and scaled by /255
    assert np.allclose(x[0, 16, 16], [1.0, 0.0, 128 / 255], atol=1e-6)


def test_preprocess_matches_notebook_pipeline():
    """Same result as the notebook's own cifar10_prediction() preprocessing."""
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (50, 70, 3), dtype=np.uint8))
    ours = preprocess(img, (None, 32, 32, 3))
    nb = np.expand_dims(np.array(img.resize((32, 32)).convert("RGB"), dtype=np.float32) / 255.0, 0)
    assert np.allclose(ours, nb)


@pytest.mark.parametrize("mode", ["RGBA", "L", "P"])
def test_non_rgb_inputs_are_converted(mode):
    data = make_image_bytes("PNG", mode=mode, color=(10, 20, 30, 255) if mode == "RGBA" else (10, 20, 30))
    v = load_and_validate(data, "a.png", min_dim=8, max_dim=6000)
    assert v.image.mode == "RGB"
    assert preprocess(v.image, (None, 32, 32, 3)).shape == (1, 32, 32, 3)


def test_rejects_wrong_channels():
    with pytest.raises(ValueError):
        preprocess(Image.new("RGB", (32, 32)), (None, 32, 32, 1))


def test_validation_errors():
    with pytest.raises(ImageValidationError) as e:
        load_and_validate(b"", "a.png", min_dim=8, max_dim=6000)
    assert e.value.code == "empty_file"
    with pytest.raises(ImageValidationError) as e:
        load_and_validate(make_image_bytes(), "a.exe", min_dim=8, max_dim=6000)
    assert e.value.code == "unsupported_type"


def test_stable_softmax():
    p = stable_softmax(np.array([[1000.0, 1001.0, 1002.0]]))
    assert np.isfinite(p).all() and np.isclose(p.sum(), 1.0)
