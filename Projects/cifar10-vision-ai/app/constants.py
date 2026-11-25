"""Single source of truth for application-wide constants.

The CIFAR-10 class mapping lives here and nowhere else. Every other module
(API, model service, templates via the model-info endpoint, tests) imports it.
"""

from __future__ import annotations

APP_NAME = "CIFAR Vision AI"
APP_SUBTITLE = "AI Image Classification"

# Standard CIFAR-10 label order: index == model output neuron.
CLASS_NAMES: tuple[str, ...] = (
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
)
NUM_CLASSES = len(CLASS_NAMES)

# Upload validation
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({"png", "jpg", "jpeg", "webp"})
# Formats as reported by Pillow after decoding the actual bytes (not the filename).
ALLOWED_IMAGE_FORMATS: frozenset[str] = frozenset({"PNG", "JPEG", "WEBP"})
ALLOWED_MIME_TYPES: frozenset[str] = frozenset(
    {"image/png", "image/jpeg", "image/webp"}
)

# Multipart field name expected by POST /api/predict/
UPLOAD_FIELD_NAME = "image"
