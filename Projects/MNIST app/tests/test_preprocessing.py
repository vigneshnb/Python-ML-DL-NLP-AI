"""Preprocessing produces the tensor the model was trained on."""
import numpy as np
import pytest
from PIL import Image

from app.errors import PreprocessingError
from app.services.preprocessing import preprocess


def test_output_shape_dtype_and_range(digit_path):
    batch = preprocess(Image.open(digit_path(4)), "mnist")
    assert batch.shape == (1, 28, 28)
    assert batch.dtype == np.float32
    assert batch.min() >= 0.0 and batch.max() <= 1.0


def test_light_background_is_inverted(digit_path):
    batch = preprocess(Image.open(digit_path(0)), "mnist")[0]
    # MNIST convention: black background (corners ~0), white ink.
    assert batch[0, 0] == 0 and batch[-1, -1] == 0
    assert batch.max() == pytest.approx(1.0, abs=0.01)


def test_digit_is_centred_by_mass(digit_path):
    arr = preprocess(Image.open(digit_path(2)), "mnist")[0]
    rows, cols = np.indices(arr.shape)
    cy, cx = (rows * arr).sum() / arr.sum(), (cols * arr).sum() / arr.sum()
    assert abs(cy - 13.5) <= 1.0 and abs(cx - 13.5) <= 1.0


def test_digit_fits_20px_box(digit_path):
    arr = preprocess(Image.open(digit_path(1)), "mnist")[0]
    ys, xs = np.nonzero(arr)
    assert ys.max() - ys.min() + 1 <= 20
    assert xs.max() - xs.min() + 1 <= 20


def test_transparent_png_supported():
    img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    for x in range(30, 70):
        for y in range(45, 55):
            img.putpixel((x, y), (0, 0, 0, 255))
    batch = preprocess(img, "mnist")
    assert batch.shape == (1, 28, 28) and batch.max() > 0.9


def test_notebook_mode_matches_notebook_pipeline(digit_path):
    img = Image.open(digit_path(6))
    expected = np.asarray(img.convert("L").resize((28, 28)), dtype=np.float32) / 255.0
    np.testing.assert_allclose(preprocess(img, "notebook")[0], expected)


def test_blank_image_raises():
    with pytest.raises(PreprocessingError):
        preprocess(Image.new("L", (50, 50), 255), "mnist")


def test_unknown_mode_raises(digit_path):
    with pytest.raises(ValueError):
        preprocess(Image.open(digit_path(1)), "bogus")
