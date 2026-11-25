# Home Value Estimator · Property Intelligence

A production-ready Flask app that serves the existing Keras model
`models/boston_home_price_prediction.h5` behind a light, editorial real-estate interface.
Users enter 13 tract characteristics, get the model's estimate with full context, and can
clear everything and start again without reloading the page.

> **Educational demonstration only.** The model was trained on the 1970s Boston Housing
> dataset. It estimates a historical census-tract **median** in **1970s dollars** (capped at
> $50,000). It is not a valuation of any real property today, and the dataset contains
> documented bias (`B`, `LSTAT`). Do not use it for financial, housing, lending or
> investment decisions.

## What the model expects (verified)

| | |
|---|---|
| Framework | Keras 3 / TensorFlow (`.h5`), loaded with `compile=False` |
| Input | `(None, 13)` float32, a single feature vector |
| Feature order | `CRIM, ZN, INDUS, CHAS, NOX, RM, AGE, DIS, RAD, TAX, PTRATIO, B, LSTAT` |
| Preprocessing | `StandardScaler` fitted on the training split of `keras.datasets.boston_housing` |
| Output | `(None, 1)`: MEDV in **thousands of 1970s USD** (training range 5–50) |

The training notebook was located and the scaler was rebuilt exactly. The app reproduces the
notebook's recorded output `29.239614` to six decimals. See **[docs/MODEL.md](docs/MODEL.md)** for
the full analysis, including a scaler bug in the notebook's prediction cell and how the app avoids it.

## Quick start (development)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env            # set SECRET_KEY; APP_ENV=development for local debugging
python run.py                   # http://127.0.0.1:5000
```

Python 3.12 is the target. It was also developed and tested on Python 3.13 with TF 2.21.0 /
Keras 3.15.1 / Flask 3.1.3.

## API

All responses use `{"success": bool, "message": str, "data": {...}}`. Errors return
`{"success": false, "message": str, "errors": {field: msg}}`.

### `POST /api/predict/`

```json
{"features": {"CRIM": 0.02, "ZN": 18, "INDUS": 2.5, "CHAS": 0, "NOX": 0.45, "RM": 6.5,
              "AGE": 65, "DIS": 4.5, "RAD": 1, "TAX": 300, "PTRATIO": 15, "B": 390, "LSTAT": 5}}
```

```json
{"success": true, "message": "Prediction completed successfully",
 "data": {"predicted_value": 28.0736, "display_value": "$28,074",
          "unit": "thousands of US dollars (1970s)", "execution_time_ms": 3.4,
          "feature_summary": {"CRIM": 0.02, "...": "..."},
          "standardized_inputs": {"CRIM": -0.4036, "...": "..."},
          "target_context": {"train_min": 5.0, "train_max": 50.0, "train_median": 20.75},
          "warnings": []}}
```

Status codes: `200` OK, `400` malformed JSON, `413` body > 16 KB, `415` not JSON,
`422` validation errors (missing, wrong type, NaN, out of range, unknown field), `503` model
unavailable, `500` inference failure (no internals leaked).

### `GET /api/health/`

Returns application status, model status and application name.

### `GET /api/model-info/`

Returns the model name, input and output shapes, dtype, feature count and names,
framework versions, preprocessing and output unit.

## Tests

```bash
python -m pytest                       # everything (65 tests)
python -m pytest -m "not real_model"   # skip tests that load TensorFlow
node tests/js/form-logic.test.mjs      # front-end logic only
```

* **Isolated unit tests** (stub model): API envelope, validation (missing, types, booleans,
  NaN, ranges, integers, unknown fields), 400/413/415/404/405/503/500, security headers,
  and the production `DEBUG=False` / `SECRET_KEY` rules.
* **Real-model tests** (`real_model`): load once, shapes and dtype, service output equals plain
  `keras.predict`, notebook provenance value reproduced, test-split MAE, end-to-end `/api/predict/`,
  outputs vary with inputs, model file hash unchanged.
* **Schema / preprocessing:** order equals training columns, defaults equal training medians,
  scaler equals sklearn, bad artefacts rejected.
* **Front-end:** every feature is rendered with constraints and no pre-filled values, there are no
  inline scripts (CSP), Clear Form and duplicate-submit wiring are in place, and the pure logic runs under Node.

## Project layout

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Deployment to Ubuntu / AWS EC2 with Gunicorn,
Nginx, systemd and Let's Encrypt is covered in [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Regenerating preprocessing parameters

`models/preprocessing.json` is committed. To rebuild it from the cached Keras dataset
(this does **not** train or modify the model):

```bash
python scripts/export_preprocessing.py
```
