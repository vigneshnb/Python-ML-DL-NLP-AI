"""HTTP-layer tests with the stub model (isolated; no real-inference claims)."""

import io

from tests.conftest import make_image_bytes


def post(client, data: bytes | None, filename="img.png"):
    payload = {} if data is None else {"image": (io.BytesIO(data), filename)}
    return client.post("/api/predict/", data=payload, content_type="multipart/form-data")


def test_app_initialises(stub_app):
    assert stub_app.config["TESTING"] is True
    assert "model_service" in stub_app.extensions
    rules = {r.rule for r in stub_app.url_map.iter_rules()}
    assert {"/api/predict/", "/api/health/", "/api/model-info/", "/"} <= rules


def test_production_config_has_debug_off(monkeypatch):
    from app.config import get_config

    assert get_config("production").DEBUG is False


def test_health(stub_client):
    r = stub_client.get("/api/health/")
    body = r.get_json()
    assert r.status_code == 200 and body["success"] is True
    assert body["data"]["app_name"] == "CIFAR Vision AI"
    assert body["data"]["model_loaded"] is True


def test_missing_file(stub_client):
    r = post(stub_client, None)
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "missing_file"


def test_empty_filename(stub_client):
    r = post(stub_client, b"abc", filename="")
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "missing_file"


def test_unsupported_extension(stub_client):
    r = post(stub_client, make_image_bytes("GIF"), filename="anim.gif")
    assert r.status_code == 415
    assert r.get_json()["error"]["code"] == "unsupported_type"


def test_unsupported_real_format_with_allowed_extension(stub_client):
    # A BMP renamed to .png must be rejected by content inspection.
    r = post(stub_client, make_image_bytes("BMP"), filename="fake.png")
    assert r.status_code == 415
    assert r.get_json()["error"]["code"] == "unsupported_type"


def test_corrupted_image(stub_client):
    good = make_image_bytes("PNG")
    r = post(stub_client, good[: len(good) // 2], filename="broken.png")
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "corrupted_image"


def test_not_an_image(stub_client):
    r = post(stub_client, b"this is plain text, not pixels", filename="notes.jpg")
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "corrupted_image"


def test_oversized_upload(stub_app):
    stub_app.config["MAX_CONTENT_LENGTH"] = 1024
    r = post(stub_app.test_client(), make_image_bytes("PNG", size=(400, 400), color=None, mode="RGB") + b"\0" * 4096)
    assert r.status_code == 413
    body = r.get_json()
    assert body["success"] is False and body["error"]["code"] == "file_too_large"


def test_image_too_small(stub_client):
    r = post(stub_client, make_image_bytes("PNG", size=(4, 4)))
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "invalid_dimensions"


def test_image_too_large_dimensions(stub_app):
    stub_app.config["MAX_IMAGE_DIMENSION"] = 100
    r = post(stub_app.test_client(), make_image_bytes("PNG", size=(150, 50)))
    assert r.status_code == 400
    assert r.get_json()["error"]["code"] == "invalid_dimensions"


def test_valid_upload_response_structure(stub_client, stub_app):
    r = post(stub_client, make_image_bytes("JPEG"), filename="photo.JPG")
    assert r.status_code == 200
    body = r.get_json()
    assert body["success"] is True and body["message"] == "Image classified successfully"
    d = body["data"]
    assert set(d) >= {"predicted_class", "predicted_label", "confidence", "probabilities", "execution_time_ms"}
    assert len(d["probabilities"]) == 10
    assert stub_app.extensions["model_service"].calls == 1


def test_webp_accepted(stub_client):
    r = post(stub_client, make_image_bytes("WEBP"), filename="x.webp")
    assert r.status_code == 200


def test_security_headers_and_no_leak(stub_client):
    r = stub_client.get("/api/does-not-exist/")
    assert r.status_code == 404
    assert r.get_json()["success"] is False
    for h in ("X-Content-Type-Options", "X-Frame-Options", "Content-Security-Policy", "Referrer-Policy"):
        assert h in r.headers
    assert r.headers["Cache-Control"] == "no-store"


def test_model_unavailable_returns_503(tmp_path):
    from app import create_app
    from app.services.model_service import ModelService

    app = create_app("testing", model_service=ModelService(tmp_path / "missing.h5"))
    c = app.test_client()
    r = post(c, make_image_bytes("PNG"))
    assert r.status_code == 503
    body = r.get_json()
    assert body["error"]["code"] == "model_unavailable"
    assert str(tmp_path) not in r.get_data(as_text=True)  # no filesystem path leak
    assert c.get("/api/health/").get_json()["data"]["model_status"] == "error"
