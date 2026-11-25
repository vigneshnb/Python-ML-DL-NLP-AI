"""POST /api/predict/ — valid predictions and every validation failure path."""
import io

import pytest


def post_image(client, data: bytes, filename: str, mimetype: str = "image/png", field: str = "image"):
    return client.post(
        "/api/predict/",
        data={field: (io.BytesIO(data), filename, mimetype)},
        content_type="multipart/form-data",
    )


@pytest.mark.parametrize("digit", range(10))
def test_predicts_real_handwritten_digits(client, digit_path, digit):
    res = post_image(client, digit_path(digit).read_bytes(), f"digit_{digit}.png")
    assert res.status_code == 200, res.get_json()
    body = res.get_json()
    assert body["success"] is True
    assert body["message"] == "Prediction completed successfully"
    assert body["data"]["predicted_digit"] == digit


def test_prediction_response_format(client, digit_path):
    res = post_image(client, digit_path(7).read_bytes(), "seven.png")
    data = res.get_json()["data"]
    assert set(data) >= {"predicted_digit", "confidence", "probabilities", "execution_time_ms"}
    assert isinstance(data["predicted_digit"], int)
    assert list(data["probabilities"]) == [str(i) for i in range(10)]
    # Probabilities are percentages that come from a softmax: they sum to ~100.
    assert sum(data["probabilities"].values()) == pytest.approx(100, abs=0.1)
    assert data["confidence"] == max(data["probabilities"].values())
    assert data["probabilities"][str(data["predicted_digit"])] == data["confidence"]
    assert data["execution_time_ms"] > 0
    assert data["processed_image"].startswith("data:image/png;base64,")
    assert data["image"]["width"] == 256 and data["image"]["format"] == "PNG"


def test_jpeg_upload_is_accepted(client, digit_path):
    from PIL import Image

    buf = io.BytesIO()
    Image.open(digit_path(3)).convert("RGB").save(buf, format="JPEG", quality=90)
    res = post_image(client, buf.getvalue(), "three.jpg", "image/jpeg")
    assert res.status_code == 200
    assert res.get_json()["data"]["predicted_digit"] == 3


def test_missing_file(client):
    res = client.post("/api/predict/", data={}, content_type="multipart/form-data")
    assert res.status_code == 400
    body = res.get_json()
    assert body["success"] is False
    assert body["error"]["code"] == "missing_file"


def test_empty_filename(client):
    res = post_image(client, b"", "")
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] == "missing_file"


def test_empty_file(client):
    res = post_image(client, b"", "blank.png")
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] == "empty_file"


@pytest.mark.parametrize("filename,mimetype", [("notes.txt", "text/plain"), ("digit.gif", "image/gif"), ("noext", "image/png")])
def test_unsupported_extension(client, image_bytes, filename, mimetype):
    res = post_image(client, image_bytes(), filename, mimetype)
    assert res.status_code == 415
    assert res.get_json()["success"] is False


def test_unsupported_format_disguised_as_png(client, image_bytes):
    gif = image_bytes(fmt="GIF")
    res = post_image(client, gif, "sneaky.png", "image/png")
    assert res.status_code == 415
    assert res.get_json()["error"]["code"] == "content_mismatch"


def test_wrong_mime_type(client, image_bytes):
    res = post_image(client, image_bytes(), "digit.png", "application/pdf")
    assert res.status_code == 415


def test_non_image_content(client):
    res = post_image(client, b"this is definitely not an image", "fake.png")
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] == "invalid_image"


def test_corrupted_image(client, digit_path):
    data = digit_path(5).read_bytes()
    truncated = data[: len(data) // 2]
    res = post_image(client, truncated, "broken.png")
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] in {"corrupted_image", "invalid_image"}


def test_oversized_upload(client, app):
    too_big = b"\x89PNG\r\n\x1a\n" + b"0" * (app.config["MAX_CONTENT_LENGTH"] + 10)
    res = post_image(client, too_big, "huge.png")
    assert res.status_code == 413
    body = res.get_json()
    assert body["success"] is False
    assert body["error"]["code"] == "payload_too_large"


def test_image_dimensions_too_small(client, image_bytes):
    res = post_image(client, image_bytes(size=(4, 4)), "tiny.png")
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] == "image_too_small"


def test_image_dimensions_too_large(client, app, image_bytes):
    limit = app.config["MAX_IMAGE_DIM"]
    res = post_image(client, image_bytes(size=(limit + 1, 10)), "wide.png")
    assert res.status_code == 400
    assert res.get_json()["error"]["code"] == "image_too_large"


def test_blank_image_reports_no_digit(client, image_bytes):
    res = post_image(client, image_bytes(color=255), "blank.png")
    assert res.status_code == 422
    assert res.get_json()["error"]["code"] == "preprocessing_error"


def test_filename_is_sanitised(client, digit_path):
    res = post_image(client, digit_path(1).read_bytes(), "../../etc/passwd.png")
    assert res.status_code == 200
    assert res.get_json()["data"]["filename"] == "etc_passwd.png"
