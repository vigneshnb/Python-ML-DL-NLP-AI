"""Upload validation: filename, extension, MIME type, decodability and dimensions."""
from __future__ import annotations

import io
import warnings

from flask import current_app
from PIL import Image, UnidentifiedImageError
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from ..errors import UnsupportedMediaError, ValidationError

# Guard against decompression bombs (4096 x 4096 is our dimension ceiling anyway).
Image.MAX_IMAGE_PIXELS = 4096 * 4096 * 2


def allowed_extension(filename: str, allowed: frozenset[str]) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed


def validate_upload(file: FileStorage | None) -> tuple[str, bytes, Image.Image]:
    """Validate an uploaded file and return (safe_filename, raw_bytes, decoded_image).

    Everything is processed in memory; nothing is written to disk.
    """
    cfg = current_app.config

    if file is None:
        raise ValidationError("No image file provided. Send the image in the 'image' form field.", code="missing_file")
    if not file.filename:
        raise ValidationError("No file selected.", code="missing_file")

    filename = secure_filename(file.filename) or "upload"
    if not allowed_extension(file.filename, cfg["ALLOWED_EXTENSIONS"]):
        raise UnsupportedMediaError("Unsupported file type. Please upload a PNG, JPG or JPEG image.")

    mimetype = (file.mimetype or "").lower()
    if mimetype and mimetype not in cfg["ALLOWED_MIME_TYPES"] and mimetype != "application/octet-stream":
        raise UnsupportedMediaError(f"Unsupported content type '{mimetype}'. Please upload a PNG or JPEG image.")

    raw = file.read()
    if not raw:
        raise ValidationError("The uploaded file is empty.", code="empty_file")
    if len(raw) > cfg["MAX_CONTENT_LENGTH"]:
        raise ValidationError("File is too large.", status_code=413, code="payload_too_large")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            probe = Image.open(io.BytesIO(raw))
            probe.verify()  # detects truncated / corrupted files
            image = Image.open(io.BytesIO(raw))
            image.load()
    except (UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise ValidationError("The file is not a valid image or is corrupted.", code="invalid_image")
    except (OSError, SyntaxError, ValueError):
        raise ValidationError("The image file is corrupted or truncated.", code="corrupted_image")

    if image.format not in cfg["ALLOWED_FORMATS"]:
        raise UnsupportedMediaError(
            f"Image content is {image.format or 'unknown'}, not PNG or JPEG.", code="content_mismatch"
        )

    width, height = image.size
    if width < cfg["MIN_IMAGE_DIM"] or height < cfg["MIN_IMAGE_DIM"]:
        raise ValidationError(
            f"Image is too small ({width}x{height}). Minimum is {cfg['MIN_IMAGE_DIM']}px per side.",
            code="image_too_small",
        )
    if width > cfg["MAX_IMAGE_DIM"] or height > cfg["MAX_IMAGE_DIM"]:
        raise ValidationError(
            f"Image is too large ({width}x{height}). Maximum is {cfg['MAX_IMAGE_DIM']}px per side.",
            code="image_too_large",
        )

    return filename, raw, image
