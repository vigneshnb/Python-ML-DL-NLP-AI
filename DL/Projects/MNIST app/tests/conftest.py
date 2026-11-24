import io
import os
from pathlib import Path

import pytest
from PIL import Image

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from app import create_app  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def app():
    application = create_app("testing")
    yield application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture(scope="session")
def digit_path():
    return lambda d: FIXTURES / f"digit_{d}.png"


def make_image_bytes(fmt="PNG", size=(64, 64), color=255, mode="L"):
    buf = io.BytesIO()
    Image.new(mode, size, color).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture()
def image_bytes():
    return make_image_bytes
