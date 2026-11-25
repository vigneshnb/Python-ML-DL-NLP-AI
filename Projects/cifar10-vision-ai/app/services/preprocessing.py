"""Image validation and preprocessing.

Preprocessing decisions (derived from inspecting ``cifar10_pred.h5`` and the
training notebook ``CIFAR10.ipynb``):

* Input shape ``(None, 32, 32, 3)``  -> resize to 32x32, 3 channels, add a batch axis.
* Channel order RGB                  -> training data came from
  ``keras.datasets.cifar10`` (RGB uint8); Pillow also decodes to RGB, so no
  channel swap is needed.
* Pixel range                        -> training used ``x / 255.0`` (no mean/std
  standardisation), so we divide by 255 to get floats in [0, 1].
* Dtype                              -> model weights are float32; we feed float32.
* Alpha / palette / greyscale input  -> converted with ``Image.convert("RGB")``,
  exactly as the notebook's own ``cifar10_prediction`` helper did.
* Resizing                           -> plain resize to 32x32 (no crop / padding),
  matching the notebook. Bicubic resampling is used explicitly (Pillow's
  default for RGB images) so results don't depend on the Pillow version.

The input tensor shape is not hard-coded: the model service passes the shape
it read from the loaded model.
"""

from __future__ import annotations

import io
import warnings
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from app.constants import ALLOWED_EXTENSIONS, ALLOWED_IMAGE_FORMATS

# Decompression-bomb guard: refuse anything above ~40 MP regardless of config.
Image.MAX_IMAGE_PIXELS = 40_000_000


class ImageValidationError(ValueError):
    """Raised for any user-correctable problem with an uploaded image."""

    def __init__(self, message: str, code: str = "invalid_image", status: int = 400):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status = status


@dataclass(frozen=True)
class ValidatedImage:
    image: Image.Image  # decoded image, RGB, EXIF orientation applied
    format: str
    width: int
    height: int


def allowed_extension(filename: str | None) -> bool:
    if not filename or "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def load_and_validate(
    data: bytes,
    filename: str | None,
    *,
    min_dim: int,
    max_dim: int,
) -> ValidatedImage:
    """Validate raw upload bytes and return a decoded RGB image.

    The filename is only used for the extension check; it is never touched on
    the filesystem. Everything happens in memory.
    """
    if not data:
        raise ImageValidationError("The uploaded file is empty.", "empty_file")

    if not allowed_extension(filename):
        raise ImageValidationError(
            "Unsupported file type. Please upload a PNG, JPG, JPEG or WEBP image.",
            "unsupported_type",
            415,
        )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            # 1) structural check of the file
            with Image.open(io.BytesIO(data)) as probe:
                fmt = probe.format
                probe.verify()
            # 2) verify() invalidates the object: reopen and fully decode
            img = Image.open(io.BytesIO(data))
            img.load()
    except (Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ImageValidationError(
            "Image resolution is too large.", "image_too_large", 413
        ) from None
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise ImageValidationError(
            "The file is corrupted or is not a valid image.", "corrupted_image"
        ) from None

    if fmt not in ALLOWED_IMAGE_FORMATS:
        raise ImageValidationError(
            "Unsupported image format. Please upload a PNG, JPG, JPEG or WEBP image.",
            "unsupported_type",
            415,
        )

    width, height = img.size
    if width < min_dim or height < min_dim:
        raise ImageValidationError(
            f"Image is too small. Minimum size is {min_dim}x{min_dim} pixels.",
            "invalid_dimensions",
        )
    if width > max_dim or height > max_dim:
        raise ImageValidationError(
            f"Image is too large. Maximum size is {max_dim}x{max_dim} pixels.",
            "invalid_dimensions",
        )

    # Respect camera orientation so phone photos aren't classified sideways.
    img = ImageOps.exif_transpose(img)
    rgb = img.convert("RGB")
    return ValidatedImage(image=rgb, format=fmt, width=width, height=height)


def preprocess(image: Image.Image, input_shape: tuple) -> np.ndarray:
    """Convert a PIL image into the tensor the model expects.

    ``input_shape`` is the model's ``input_shape`` e.g. ``(None, 32, 32, 3)``.
    Returns a float32 array of shape ``(1, H, W, C)`` with values in [0, 1].
    """
    if len(input_shape) != 4:
        raise ValueError(f"Unsupported model input shape: {input_shape}")
    _, height, width, channels = input_shape
    if channels != 3:
        raise ValueError(f"Model expects {channels} channels; only RGB is supported.")

    rgb = image if image.mode == "RGB" else image.convert("RGB")
    resized = rgb.resize((int(width), int(height)), Image.Resampling.BICUBIC)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)
