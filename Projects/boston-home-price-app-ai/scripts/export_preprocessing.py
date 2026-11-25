"""Rebuild the training-time StandardScaler parameters and save them as JSON.

Why this script exists
----------------------
The training notebook (Learning/DL/Chapter-8 (ANN for Regression)/Docs/
"Chapter 8.1 ANN For Regression.ipynb") did:

    (X_train, y_train), (X_test, y_test) = boston_housing.load_data()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)     # <- the model is trained on THIS scaling
    X_test  = scaler.fit_transform(X_test)      # (re-fit on test data - notebook bug)
    model.fit(X_train, y_train, ...)

The fitted scaler object was never saved. However, ``boston_housing.load_data()``
is deterministic (default ``seed=113``, ``test_split=0.2``), so the exact
training-split mean / standard deviation can be recomputed. This script does that
and writes ``models/preprocessing.json``. It does NOT train or modify the model.

It also verifies provenance: the notebook recorded a prediction of 29.239614 for a
sample row scaled with the (test-fitted) scaler. Reproducing that value proves the
.h5 file and the dataset split are the ones used in the notebook.

Run once (requires TensorFlow and the cached dataset ~/.keras/datasets/boston_housing.npz):

    python scripts/export_preprocessing.py
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "boston_home_price_prediction.h5"
OUT_PATH = ROOT / "models" / "preprocessing.json"

# Column order of keras.datasets.boston_housing (same as the original StatLib file).
FEATURE_ORDER = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
    "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT",
]

NOTEBOOK_SAMPLE = [0.02, 18, 2.5, 0, 0.45, 6.5, 65, 4.5, 1, 300, 15, 390, 5]
NOTEBOOK_RECORDED_PREDICTION = 29.239614


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _standard_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Same maths as sklearn StandardScaler (population std, ddof=0)."""
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0.0] = 1.0
    return mean, scale


def main() -> int:
    import keras
    from keras.datasets import boston_housing

    (x_train, y_train), (x_test, _y_test) = boston_housing.load_data()
    x_train = x_train.astype(np.float64)
    x_test = x_test.astype(np.float64)
    assert x_train.shape == (404, 13), x_train.shape

    mean, scale = _standard_scaler(x_train)
    test_mean, test_scale = _standard_scaler(x_test)

    model = keras.models.load_model(MODEL_PATH, compile=False)
    sample = np.array([NOTEBOOK_SAMPLE], dtype=np.float64)
    reproduced = float(model.predict(((sample - test_mean) / test_scale).astype(np.float32), verbose=0)[0, 0])
    correct = float(model.predict(((sample - mean) / scale).astype(np.float32), verbose=0)[0, 0])
    if abs(reproduced - NOTEBOOK_RECORDED_PREDICTION) > 1e-3:
        print(f"Provenance check FAILED: got {reproduced}, expected {NOTEBOOK_RECORDED_PREDICTION}")
        return 1

    dataset_file = Path(os.environ.get("KERAS_HOME", Path.home() / ".keras")) / "datasets" / "boston_housing.npz"

    payload = {
        "description": (
            "StandardScaler parameters fitted on the training split of "
            "keras.datasets.boston_housing.load_data() (seed=113, test_split=0.2), "
            "exactly as in the training notebook. Model input = (x - mean) / scale."
        ),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generated_by": "scripts/export_preprocessing.py",
        "model_file": MODEL_PATH.name,
        "model_sha256": _sha256(MODEL_PATH),
        "dataset": {
            "source": "keras.datasets.boston_housing",
            "file_sha256": _sha256(dataset_file) if dataset_file.exists() else None,
            "seed": 113,
            "test_split": 0.2,
            "train_rows": int(x_train.shape[0]),
        },
        "scaler": {
            "type": "StandardScaler",
            "with_mean": True,
            "with_std": True,
            "ddof": 0,
        },
        "feature_order": FEATURE_ORDER,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "train_min": x_train.min(axis=0).tolist(),
        "train_max": x_train.max(axis=0).tolist(),
        "train_median": np.median(x_train, axis=0).tolist(),
        "target": {
            "name": "MEDV",
            "description": "Median value of owner-occupied homes in a census tract",
            "unit": "thousands of US dollars (1970s)",
            "multiplier": 1000,
            "train_min": float(y_train.min()),
            "train_max": float(y_train.max()),
            "train_median": float(np.median(y_train)),
            "train_mean": float(y_train.mean()),
            "censored_at": 50.0,
        },
        "verification": {
            "notebook_sample": NOTEBOOK_SAMPLE,
            "notebook_recorded_prediction": NOTEBOOK_RECORDED_PREDICTION,
            "reproduced_with_notebook_scaler": round(reproduced, 6),
            "prediction_with_training_scaler": round(correct, 6),
            "note": (
                "The notebook's prediction cell used a scaler that had been re-fitted on "
                "X_test. The model was trained on X_train scaling, which this file uses."
            ),
        },
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUT_PATH.relative_to(ROOT)}")
    print(f"  provenance check: reproduced {reproduced:.6f} (notebook: {NOTEBOOK_RECORDED_PREDICTION})")
    print(f"  sample with training scaler: {correct:.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
