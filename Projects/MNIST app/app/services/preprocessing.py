"""Image preprocessing that reproduces the model's training input distribution.

Facts established from ``MNIST Digit Classification.ipynb`` and the model file:

* Training data: ``keras.datasets.mnist`` -> uint8 arrays of shape (28, 28),
  white digit on a black background, digit fitted in a ~20x20 box and centred by
  centre of mass inside the 28x28 frame.
* Normalisation: ``X / 255.0`` -> float values in [0, 1].
* Model input: ``(batch, 28, 28)`` float32 -- no channel axis (first layer is Flatten).
* Output: 10-way softmax probabilities.

Two modes are provided:

``notebook``  -- the exact steps used in the notebook for outside images:
                 grayscale -> resize to 28x28 -> /255.
``mnist``     -- (default) the notebook steps plus the normalisation that the MNIST
                 dataset itself applied to its digits, so real-world uploads (dark ink on
                 white paper, off-centre, any size) look like the training data:
                 auto-invert, background removal, crop, fit to 20x20, centre of mass.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageOps

from ..errors import PreprocessingError

MNIST_SIZE = 28
MNIST_BOX = 20  # MNIST digits are size-normalised to fit a 20x20 box


def _to_grayscale(image: Image.Image) -> np.ndarray:
    """Convert any PIL image to a float32 grayscale array (0..255).

    Transparent pixels are composited onto white so transparent-background PNGs work.
    """
    image = ImageOps.exif_transpose(image)
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)
    return np.asarray(image.convert("L"), dtype=np.float32)


def _border_level(arr: np.ndarray) -> float:
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    return float(np.median(border))


def _shift_to_center_of_mass(canvas: np.ndarray) -> np.ndarray:
    total = canvas.sum()
    if total <= 0:
        return canvas
    rows, cols = np.indices(canvas.shape)
    cy = (rows * canvas).sum() / total
    cx = (cols * canvas).sum() / total
    dy = int(round(MNIST_SIZE / 2 - 0.5 - cy))
    dx = int(round(MNIST_SIZE / 2 - 0.5 - cx))

    # Never shift ink outside the frame.
    ys, xs = np.nonzero(canvas)
    dy = int(np.clip(dy, -ys.min(), MNIST_SIZE - 1 - ys.max()))
    dx = int(np.clip(dx, -xs.min(), MNIST_SIZE - 1 - xs.max()))
    return np.roll(np.roll(canvas, dy, axis=0), dx, axis=1)


def preprocess_notebook(image: Image.Image) -> np.ndarray:
    """Exact notebook pipeline: convert('L') -> resize((28, 28)) -> /255."""
    gray = ImageOps.exif_transpose(image).convert("L").resize((MNIST_SIZE, MNIST_SIZE))
    return np.asarray(gray, dtype=np.float32) / 255.0


def preprocess_mnist(image: Image.Image) -> np.ndarray:
    """MNIST-style normalisation. Returns a (28, 28) float32 array in [0, 1]."""
    arr = _to_grayscale(image)

    # 1. MNIST is light ink on a dark background: invert light-background images.
    if _border_level(arr) > 127:
        arr = 255.0 - arr

    # 2. Remove background / paper texture and stretch contrast.
    background = _border_level(arr)
    arr = np.clip(arr - background, 0, None)
    peak = float(arr.max())
    if peak < 25:  # practically blank after removing the background
        raise PreprocessingError("No digit could be detected in the image. Try a clearer, higher-contrast image.")
    arr = arr / peak * 255.0
    arr[arr < 0.2 * 255] = 0  # suppress faint noise

    # 3. Crop to the digit's bounding box.
    ys, xs = np.nonzero(arr)
    if ys.size == 0:
        raise PreprocessingError("No digit could be detected in the image.")
    arr = arr[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]

    # 4. Fit into a 20x20 box preserving the aspect ratio.
    h, w = arr.shape
    scale = MNIST_BOX / max(h, w)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    digit = Image.fromarray(arr.astype(np.uint8)).resize((new_w, new_h), Image.Resampling.LANCZOS)
    digit_arr = np.asarray(digit, dtype=np.float32)
    if digit_arr.max() > 0:  # thin strokes fade when shrinking a large image; restore intensity
        digit_arr = np.clip(digit_arr / digit_arr.max() * 255.0, 0, 255)

    # 5. Paste into a 28x28 canvas and centre by centre of mass.
    canvas = np.zeros((MNIST_SIZE, MNIST_SIZE), dtype=np.float32)
    top, left = (MNIST_SIZE - new_h) // 2, (MNIST_SIZE - new_w) // 2
    canvas[top: top + new_h, left: left + new_w] = digit_arr
    canvas = _shift_to_center_of_mass(canvas)

    # 6. Same normalisation as training: /255.
    return (canvas / 255.0).astype(np.float32)


def preprocess(image: Image.Image, mode: str = "mnist") -> np.ndarray:
    """Return a model-ready batch of shape (1, 28, 28), float32 in [0, 1]."""
    if mode == "notebook":
        arr = preprocess_notebook(image)
    elif mode == "mnist":
        arr = preprocess_mnist(image)
    else:
        raise ValueError(f"Unknown PREPROCESS_MODE '{mode}' (expected 'mnist' or 'notebook')")
    return np.expand_dims(arr, axis=0)


def to_data_url(arr: np.ndarray, upscale: int = 6) -> str:
    """Encode a (28, 28) [0,1] array as a PNG data URL (what the model 'sees')."""
    img = Image.fromarray((np.squeeze(arr) * 255).astype(np.uint8))
    img = img.resize((MNIST_SIZE * upscale, MNIST_SIZE * upscale), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
