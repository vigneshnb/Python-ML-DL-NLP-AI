"""Frontend contract tests: the page exposes the controls the JS drives, and the
Clear/Reset handler resets every piece of state listed in the spec.

These are static checks (no browser needed). The full interactive flow is
verified manually / in a browser as described in README.md.
"""
import re
from pathlib import Path

import pytest

JS = (Path(__file__).parent.parent / "app" / "static" / "js" / "app.js").read_text(encoding="utf-8")

REQUIRED_IDS = [
    "dropzone", "fileInput", "previewImg", "fileName", "fileDims", "fileSize", "removeBtn", "alert",
    "predictBtn", "clearBtn", "predDigit", "predConfidence", "predState", "predTime", "probChart",
    "thumbOriginal", "thumbProcessed", "modelStatus",
]


def _function_body(name: str) -> str:
    match = re.search(rf"function {name}\(\)\s*\{{(.*?)\n  \}}", JS, re.S)
    assert match, f"function {name} not found"
    return match.group(1)


@pytest.fixture(scope="module")
def html(app):
    return app.test_client().get("/").get_data(as_text=True)


@pytest.mark.parametrize("element_id", REQUIRED_IDS)
def test_page_has_required_elements(html, element_id):
    assert f'id="{element_id}"' in html


def test_predict_button_disabled_initially(html):
    assert re.search(r'<button[^>]*id="predictBtn"[^>]*disabled', html)


def test_file_input_accepts_only_png_jpeg(html):
    assert 'accept=".png,.jpg,.jpeg,image/png,image/jpeg"' in html


def test_assets_are_served(client):
    assert client.get("/static/js/app.js").status_code == 200
    assert client.get("/static/css/style.css").status_code == 200


def test_clear_button_wired_to_reset(html):
    assert 'el.clearBtn.addEventListener("click", resetAll)' in JS
    assert "resetAll();" in _function_body("init")


def test_reset_all_clears_everything():
    body = _function_body("resetAll")
    for call in ("clearFile()", "resetResult()", "hideAlert()", "updatePredictButton()"):
        assert call in body
    assert "abort()" in body  # cancels an in-flight prediction


def test_reset_result_clears_prediction_and_chart():
    body = _function_body("resetResult")
    for fragment in ('el.digit.textContent = "–"', 'el.confidence.textContent = "—"', 'el.time.textContent = "—"',
                     "resetChart()", "thumbProcessed.hidden = true"):
        assert fragment in body


def test_clear_file_removes_preview():
    body = _function_body("clearFile")
    for fragment in ("revokeObjectURL", 'el.fileInput.value = ""', "el.dropPreview.hidden = true", "el.fileMeta.hidden = true"):
        assert fragment in body


def test_duplicate_submissions_prevented():
    assert "if (!state.file || state.inFlight) return;" in JS
    assert "el.predictBtn.disabled = !state.file || !!state.inFlight;" in JS


def test_predict_posts_multipart_to_api():
    assert 'predict: "/api/predict/"' in JS
    assert 'body.append("image"' in JS
