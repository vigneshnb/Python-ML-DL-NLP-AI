"""Application startup, model loading, health, model-info and security headers."""
from app.services import model_service


def test_app_starts_in_testing_mode(app):
    assert app.testing is True
    assert app.debug is False


def test_production_config_has_debug_disabled():
    from app.config import ProductionConfig

    assert ProductionConfig.DEBUG is False


def test_model_loaded_once_at_startup(app):
    assert model_service.is_loaded
    assert model_service.load_error is None
    assert model_service.input_shape == (None, 28, 28)
    assert model_service.output_shape == (None, 10)


def test_model_load_is_idempotent(app):
    model_before = model_service._model
    model_service.load(app.config["MODEL_PATH"])
    assert model_service._model is model_before


def test_missing_model_reports_error():
    from app.services.model_service import ModelService

    svc = ModelService()
    svc.load("does/not/exist.h5")
    assert not svc.is_loaded
    assert "not found" in svc.load_error


def test_index_page_renders(client):
    res = client.get("/")
    assert res.status_code == 200
    html = res.get_data(as_text=True)
    assert "MNIST AI" in html
    assert "Turn Handwriting Into" in html
    assert "Powered by Machine Learning | MNIST Digit Recognition" in html


def test_health_endpoint(client):
    res = client.get("/api/health/")
    assert res.status_code == 200
    body = res.get_json()
    assert body["success"] is True
    assert body["data"]["model_loaded"] is True
    assert body["data"]["status"] == "healthy"


def test_health_without_trailing_slash(client):
    assert client.get("/api/health").status_code == 200


def test_model_info_endpoint(client):
    body = client.get("/api/model-info/").get_json()
    assert body["success"] is True
    data = body["data"]
    assert data["file"] == "digit_brain_intel.h5"
    assert data["input_shape"] == [None, 28, 28]
    assert data["num_classes"] == 10
    assert [layer["type"] for layer in data["layers"]] == ["Flatten", "Dense", "Dense"]
    assert data["preprocessing"]["mode"] in {"mnist", "notebook"}


def test_security_headers(client):
    res = client.get("/")
    assert res.headers["X-Content-Type-Options"] == "nosniff"
    assert res.headers["X-Frame-Options"] == "DENY"
    assert "default-src 'self'" in res.headers["Content-Security-Policy"]


def test_unknown_api_route_returns_json_404(client):
    res = client.get("/api/nope/")
    assert res.status_code == 404
    assert res.get_json()["success"] is False


def test_predict_rejects_get(client):
    res = client.get("/api/predict/")
    assert res.status_code == 405
    assert res.get_json()["success"] is False
