"""API tests with a stubbed model (isolated unit tests: no real inference here)."""

import json

import pytest

from app.features import FEATURE_NAMES
from tests.conftest import StubModelService


def _post(client, payload, **kw):
    return client.post("/api/predict/", json=payload, **kw)


# ---------------------------------------------------------------- app factory
def test_app_initialises(stub_app):
    assert stub_app.config["TESTING"] is True
    assert {"model_service", "preprocessor", "prediction_service"} <= set(stub_app.extensions)


def test_production_requires_secret_key(monkeypatch):
    from app import create_app

    monkeypatch.setattr("app.config.ProductionConfig.SECRET_KEY", "")
    monkeypatch.setattr("app.config.ProductionConfig.MODEL_EAGER_LOAD", False)
    with pytest.raises(RuntimeError, match="SECRET_KEY"):
        create_app("production", model_service=StubModelService())


def test_production_debug_is_off(monkeypatch):
    from app import create_app

    monkeypatch.setattr("app.config.ProductionConfig.SECRET_KEY", "x" * 32)
    app = create_app("production", model_service=StubModelService())
    assert app.config["DEBUG"] is False


# ---------------------------------------------------------------- health / info
def test_health(stub_client):
    r = stub_client.get("/api/health/")
    body = r.get_json()
    assert r.status_code == 200 and body["success"] is True
    assert body["data"] == {
        "status": "ok", "application": "Home Value Estimator",
        "model_status": "loaded", "model_loaded": True,
    }


def test_health_reports_degraded_when_model_missing():
    from app import create_app

    app = create_app("testing", model_service=StubModelService(fail_load=True))
    data = app.test_client().get("/api/health").get_json()["data"]
    assert data["status"] == "degraded" and data["model_loaded"] is False


def test_model_info(stub_client):
    data = stub_client.get("/api/model-info/").get_json()["data"]
    assert data["model_name"] == "boston_home_price_prediction.h5"
    assert data["input_shape"] == [None, 13] and data["output_shape"] == [None, 1]
    assert data["feature_count"] == 13
    assert data["feature_names"] == list(FEATURE_NAMES)
    assert data["preprocessing"]["type"] == "StandardScaler"
    assert data["output"]["target"] == "MEDV"
    assert "thousands" in data["output"]["unit"]


# ---------------------------------------------------------------- predict: success
def test_predict_success_envelope(stub_client, stub_model, sample):
    r = _post(stub_client, {"features": sample})
    body = r.get_json()
    assert r.status_code == 200
    assert body["success"] is True and body["message"] == "Prediction completed successfully"
    d = body["data"]
    assert d["predicted_value"] == 22.5          # value comes from the (stub) model
    assert d["display_value"] == "$22,500"
    assert d["execution_time_ms"] >= 0
    assert list(d["feature_summary"]) == list(FEATURE_NAMES)
    assert d["feature_summary"]["RM"] == 6.5
    assert set(d["standardized_inputs"]) == set(FEATURE_NAMES)
    assert d["warnings"] == []
    # the model received a (1, 13) float32 array
    assert stub_model.calls[0].shape == (1, 13) and str(stub_model.calls[0].dtype) == "float32"


def test_predict_accepts_flat_payload_and_no_trailing_slash(stub_client, sample):
    r = stub_client.post("/api/predict", json=sample)
    assert r.status_code == 200


def test_predict_warns_when_extrapolating(stub_client, sample):
    sample["RM"] = 9.5  # above training max 8.725
    d = _post(stub_client, sample).get_json()["data"]
    assert any("Rooms per dwelling (RM)" in w for w in d["warnings"])


@pytest.mark.parametrize("value,expect_display,fragment", [
    (-3.0, None, "negative"),
    (55.0, "$55,000", "$50,000"),
])
def test_predict_flags_implausible_outputs(sample, value, expect_display, fragment):
    from app import create_app

    app = create_app("testing", model_service=StubModelService(value=value))
    d = app.test_client().post("/api/predict/", json=sample).get_json()["data"]
    assert d["display_value"] == expect_display
    assert any(fragment in w for w in d["warnings"])


# ---------------------------------------------------------------- predict: invalid input
def test_missing_inputs(stub_client, sample):
    del sample["RM"], sample["TAX"]
    r = _post(stub_client, sample)
    body = r.get_json()
    assert r.status_code == 422 and body["success"] is False
    assert body["errors"] == {"RM": "This value is required.", "TAX": "This value is required."}


@pytest.mark.parametrize("field,value,msg", [
    ("RM", "6.5", "Must be a number."),
    ("RM", None, "This value is required."),
    ("RM", [6.5], "Must be a number."),
    ("RM", True, "Must be a number."),
    ("RM", {"v": 1}, "Must be a number."),
    ("RAD", 2.5, "Must be a whole number."),
    ("CHAS", 2, "Must be 0 or 1."),
    ("CHAS", 0.5, "Must be a whole number."),
    ("CRIM", -1, "Must be between 0 and 100."),
    ("AGE", 101, "Must be between 0 and 100."),
    ("B", 400, "Must be between 0 and 396.9."),
    ("RM", 0.5, "Must be between 1 and 15."),
])
def test_wrong_types_and_out_of_range(stub_client, stub_model, sample, field, value, msg):
    sample[field] = value
    r = _post(stub_client, sample)
    assert r.status_code == 422
    assert r.get_json()["errors"] == {field: msg}
    assert stub_model.calls == []  # model never called with invalid data


def test_non_finite_numbers_rejected(stub_client, sample):
    raw = json.dumps(sample).replace('"RM": 6.5', '"RM": NaN')
    r = stub_client.post("/api/predict/", data=raw, content_type="application/json")
    assert r.status_code == 422
    assert r.get_json()["errors"]["RM"] == "Must be a finite number."


def test_unknown_field_rejected(stub_client, sample):
    sample["POOL"] = 1
    r = _post(stub_client, sample)
    assert r.status_code == 422 and r.get_json()["errors"] == {"POOL": "Unknown feature."}


@pytest.mark.parametrize("payload", [[1, 2, 3], "text", 42])
def test_non_object_body(stub_client, payload):
    r = _post(stub_client, payload)
    assert r.status_code == 422
    assert r.get_json()["message"] == "Request body must be a JSON object."


def test_features_must_be_object(stub_client):
    r = _post(stub_client, {"features": [1, 2]})
    assert r.status_code == 422


def test_malformed_json(stub_client):
    r = stub_client.post("/api/predict/", data="{bad", content_type="application/json")
    assert r.status_code == 400 and r.get_json()["success"] is False


def test_wrong_content_type(stub_client):
    r = stub_client.post("/api/predict/", data="RM=6", content_type="application/x-www-form-urlencoded")
    assert r.status_code == 415


def test_request_too_large(stub_client):
    r = stub_client.post("/api/predict/", data="{" + " " * 20000 + "}", content_type="application/json")
    assert r.status_code == 413
    assert r.get_json() == {"success": False, "message": "Request body is too large."}


def test_method_not_allowed_and_404_are_json(stub_client):
    assert stub_client.get("/api/predict/").get_json()["success"] is False
    r = stub_client.get("/api/nope")
    assert r.status_code == 404 and r.get_json()["message"] == "Endpoint not found."


# ---------------------------------------------------------------- model failures
def test_model_unavailable_returns_503(sample):
    from app import create_app

    app = create_app("testing", model_service=StubModelService(fail_load=True))
    r = app.test_client().post("/api/predict/", json=sample)
    assert r.status_code == 503
    assert r.get_json()["message"] == "The prediction model is currently unavailable."


def test_prediction_error_does_not_leak_internals(sample):
    from app import create_app

    app = create_app("testing", model_service=StubModelService(fail_predict=True))
    r = app.test_client().post("/api/predict/", json=sample)
    text = r.get_data(as_text=True)
    assert r.status_code == 500
    assert "Traceback" not in text and "models" not in text and ".h5" not in text


def test_real_model_service_missing_file(tmp_path):
    from app.services.model_service import ModelNotAvailableError, ModelService

    svc = ModelService(tmp_path / "nope.h5")
    with pytest.raises(ModelNotAvailableError):
        svc.load()
    assert svc.status == "error" and not svc.is_loaded


# ---------------------------------------------------------------- security headers
def test_security_headers(stub_client):
    r = stub_client.get("/api/health/")
    h = r.headers
    assert h["X-Content-Type-Options"] == "nosniff"
    assert h["X-Frame-Options"] == "DENY"
    assert "default-src 'self'" in h["Content-Security-Policy"]
    assert "script-src 'self';" in h["Content-Security-Policy"]
    assert h["Cache-Control"] == "no-store"
    assert "Strict-Transport-Security" not in h
