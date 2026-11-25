"""Frontend tests.

* The page renders every element the JS depends on, generated from the feature schema.
* Clear Form / result wiring is checked statically in app.js.
* Pure form logic (validation, payload order, formatting, initial state) runs under Node.
"""

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

from app.features import FEATURE_NAMES, FEATURES

ROOT = Path(__file__).resolve().parent.parent
APP_JS = (ROOT / "app" / "static" / "js" / "app.js").read_text(encoding="utf-8")

REQUIRED_IDS = [
    "estimateForm", "predictBtn", "clearBtn", "fillMediansBtn", "newEstimateBtn", "formStatus", "progressFill",
    "resultCard", "resultEmpty", "resultLoading", "resultError", "resultErrorText", "resultBody",
    "resultValue", "resultRaw", "resultTime", "indicatorFill", "indicatorMarker", "indicatorMedian",
    "resultWarnings", "zBars", "inputSummary", "modelStatus", "navToggle", "siteNav", "featureSchema",
]


@pytest.fixture
def html(stub_client):
    r = stub_client.get("/")
    assert r.status_code == 200
    return r.get_data(as_text=True)


def test_page_content(html):
    assert "Home Value Estimator" in html and "Property Intelligence" in html
    assert "Understand Your Property&#39;s Estimated Value" in html or "Understand Your Property's Estimated Value" in html
    assert "Explore an estimated property value using your selected characteristics and a machine learning model." in html
    assert "Home Value Estimator | Machine Learning Demonstration" in html
    assert "Educational ML demonstration" in html
    for phrase in ("Educational only", "Historical data", "Historical bias", "Do not use"):
        assert phrase in html
    for title in ("Property &amp; Neighborhood", "Environmental &amp; Structural",
                  "Accessibility &amp; Tax", "Historical Dataset Variables"):
        assert title in html


def test_required_elements_present(html):
    for el_id in REQUIRED_IDS:
        assert re.search(rf'id="{el_id}"', html), f"missing #{el_id}"
    assert re.search(r'id="predictBtn"\s+disabled', html), "predict must start disabled"
    assert "Estimate Home Value" in html and "Clear Form" in html


def test_every_feature_rendered_with_constraints(html):
    for f in FEATURES:
        if f.control == "toggle":
            assert html.count(f'name="{f.name}"') == 2
            assert 'value="0"' in html and 'value="1"' in html
        else:
            m = re.search(rf'<input[^>]*id="f-{f.name}"[^>]*>', html, re.S)
            assert m, f.name
            tag = m.group(0)
            assert 'type="number"' in tag and f'data-feature="{f.name}"' in tag
            assert 'min="' in tag and 'max="' in tag and 'step="' in tag
            assert re.search(r'\svalue="', tag) is None, f"{f.name} must not be pre-filled"
        assert f'id="err-{f.name}"' in html


def test_embedded_schema_in_model_order(html):
    raw = re.search(r'<script type="application/json" id="featureSchema">(.*?)</script>', html, re.S).group(1)
    schema = json.loads(raw)
    assert [f["name"] for f in schema["features"]] == list(FEATURE_NAMES)
    assert all("train_min" in f and "train_max" in f for f in schema["features"])
    assert schema["target"]["train_min"] == 5.0 and schema["target"]["train_max"] == 50.0


def test_no_inline_scripts_or_styles(html):
    assert re.search(r"<script(?![^>]*\bsrc=)(?![^>]*application/json)[^>]*>", html) is None
    assert " style=" not in html  # CSP has no 'unsafe-inline'


def test_static_assets_served(stub_client):
    for path in ("/static/js/app.js", "/static/js/form-logic.js",
                 "/static/css/style.css", "/static/img/logo.svg"):
        assert stub_client.get(path).status_code == 200, path


def test_clear_form_wiring():
    assert 'els.clearBtn.addEventListener("click", clearAll)' in APP_JS
    assert 'els.newEstimateBtn.addEventListener("click", clearAll)' in APP_JS
    body = APP_JS.split("function clearAll()")[1].split("\n  }\n")[0]
    for needle in (
        "requestSeq++",          # ignore in-flight responses
        "controller.abort()",    # cancel in-flight request
        "els.form.reset()",      # clear inputs
        'setValue(f.name, "")',  # including radio toggle
        'showFieldError(f.name, "")',  # reset validation errors
        "L.initialState()",      # reset state/result
        "is-loading",            # reset loading state
        "resetResultView()",     # clear visualisation + summary
        'setPhase("empty")',
        "validate()",            # disables predict button again
    ):
        assert needle in body, needle
    reset = APP_JS.split("function resetResultView()")[1].split("\n  }\n")[0]
    for needle in ("zBars", "inputSummary", "warnings", "indicatorFill", "resultValue"):
        assert needle in reset


def test_duplicate_submission_guard():
    body = APP_JS.split("function predict()")[1].split("\n  }\n")[0]
    assert "state.submitting" in body and "setSubmitting(true)" in body
    assert "seq !== requestSeq" in body


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_form_logic_with_node():
    res = subprocess.run(
        ["node", str(ROOT / "tests" / "js" / "form-logic.test.mjs")],
        capture_output=True, text=True, timeout=60,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "all form-logic tests passed" in res.stdout
