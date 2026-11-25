"""Frontend tests.

* The page renders with every element the JS needs.
* The Clear/Reset state logic (static/js/state.js) is exercised with Node.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
JS_TEST = ROOT / "tests" / "js" / "state.test.mjs"

REQUIRED_IDS = [
    "dropzone", "fileInput", "browseBtn", "previewImg", "fileName", "fileSize", "fileDims",
    "fileType", "removeBtn", "alertBox", "predictBtn", "clearBtn", "resultCard",
    "resultEmpty", "resultLoading", "resultError", "retryBtn", "resultBody", "resultLabel",
    "resultClass", "resultConfidence", "resultTime", "gaugeFill", "probChart", "probTable",
    "historyList", "clearHistoryBtn", "modelStatus", "navToggle",
]


def test_index_renders(stub_client):
    r = stub_client.get("/")
    html = r.get_data(as_text=True)
    assert r.status_code == 200
    assert "CIFAR Vision AI" in html
    assert "See the World <span class=\"gradient-text\">Through AI</span>" in html
    assert "Upload an image and discover what AI sees." in html
    assert "Powered by CIFAR-10 | AI Image Classification" in html
    for el_id in REQUIRED_IDS:
        assert re.search(rf'id="{el_id}"', html), f"missing #{el_id}"
    # predict button disabled before any upload
    assert re.search(r'id="predictBtn"\s+disabled', html)


def test_static_assets_served(stub_client):
    for path in ("/static/js/app.js", "/static/js/state.js", "/static/css/style.css"):
        assert stub_client.get(path).status_code == 200


def test_clear_button_wired():
    js = (ROOT / "app" / "static" / "js" / "app.js").read_text(encoding="utf-8")
    assert 'els.clearBtn.addEventListener("click", () => clearAll(' in js
    assert 'e.key === "Escape"' in js  # keyboard reset
    body = js.split("function clearAll(")[1].split("\n}\n")[0]
    for needle in ("resetState()", "revokeObjectURL", 'fileInput.value = ""', "resetChart()", "render()", "requestSeq"):
        assert needle in body


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js not installed")
def test_state_logic_with_node():
    res = subprocess.run(["node", str(JS_TEST)], capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stdout + res.stderr
