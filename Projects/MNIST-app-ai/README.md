# MNIST AI — Handwritten Digit Recognition

A Flask web application that serves the trained model `digit_brain_intel.h5`, built in
`MNIST Digit Classification.ipynb`. You upload a handwritten digit (PNG/JPG/JPEG), click **Predict Digit**,
and the page shows the predicted digit, its confidence, how long the prediction took, and a
Chart.js chart of the probability for every digit 0–9.

## 1. The model

| Item | Value |
|---|---|
| File | `models/digit_brain_intel.h5` (Keras 3.15.1, HDF5) |
| Architecture | `Flatten → Dense(100, relu) → Dense(10, softmax)`, 79,510 params |
| Input | `(batch, 28, 28)` float32, values in `[0, 1]`, no channel axis |
| Output | 10 softmax probabilities |
| Training data | `keras.datasets.mnist` — white digit on black, ~20×20 digit centred in 28×28 |
| Normalisation | `x / 255.0` |
| Test-set accuracy (measured) | 97.66 % |

The model is loaded **once per process** at startup (`app/services/model_service.py`). It is warmed up
on load, and calls to it are serialised by a lock.

## 2. Preprocessing

`app/services/preprocessing.py`. You choose the mode with `PREPROCESS_MODE`:

* **`mnist` (default):** grayscale (transparent pixels placed on white) → invert light
  backgrounds → remove background and stretch contrast → crop to the digit → fit into a 20×20 box
  (aspect ratio kept) → pad to 28×28 and centre by centre of mass → `/255`.
  This produces the same kind of input as the training data.
* **`notebook`:** exactly the notebook's steps: `convert('L') → resize((28,28)) → /255`.

Why `mnist` is the default: we tested both modes on 500 MNIST test digits drawn as dark ink on white,
off-centre and enlarged (the way a photo or scan looks):

| Mode | Raw MNIST arrays | Realistic uploads |
|---|---|---|
| `mnist` | 97.0 % | **97.2 %** |
| `notebook` | 98.8 % | 8.6 % |

The notebook pipeline only works on images that already look like MNIST (white on black). Real uploads
are usually dark ink on light paper, and the model was never trained on that.

## 3. Project structure

```
MNIST app/
├── app/
│   ├── __init__.py            # application factory (create_app)
│   ├── config.py              # env-based Development / Testing / Production config
│   ├── logging_config.py      # Python logging setup
│   ├── errors.py              # exceptions + consistent JSON error handlers
│   ├── security.py            # security headers (CSP, nosniff, frame-deny…)
│   ├── api/routes.py          # /api/predict/, /api/health/, /api/model-info/
│   ├── main/__init__.py       # dashboard page
│   ├── services/
│   │   ├── model_service.py       # loads the model once, thread-safe inference
│   │   ├── preprocessing.py       # training-matched preprocessing
│   │   ├── prediction_service.py  # preprocess → infer → shape result
│   │   └── validation.py          # filename / extension / MIME / content / size checks
│   ├── templates/             # base.html, index.html
│   └── static/                # css/style.css, js/app.js, img/favicon.svg
├── models/digit_brain_intel.h5
├── tests/                     # pytest suite + fixtures (real MNIST test digits)
├── deploy/                    # gunicorn.conf.py, nginx/, systemd/
├── docs/DEPLOYMENT.md         # AWS EC2 deployment guide
├── run.py                     # dev server
├── wsgi.py                    # production WSGI entry
├── requirements.txt
└── .env.example
```

## 4. Run locally

```bash
python -m venv .venv
```
```bash
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
```
```bash
pip install -r requirements.txt
```
```bash
copy .env.example .env          # Linux/macOS: cp .env.example .env  — then set APP_ENV=development
```
```bash
python run.py
```

Open http://127.0.0.1:5000.

## 5. API

All responses share one format: `{"success": bool, "message": str, "data": {...}}`.
Errors have the form `{"success": false, "message": str, "error": {"code": str, "status": int}}`.

### `POST /api/predict/`
Multipart form with the file in the `image` field.

```bash
curl -F "image=@tests/fixtures/digit_7.png" http://127.0.0.1:5000/api/predict/
```

```json
{
  "success": true,
  "message": "Prediction completed successfully",
  "data": {
    "predicted_digit": 7,
    "confidence": 99.93,
    "probabilities": {"0": 0.0, "1": 0.01, "...": "...", "7": 99.93, "9": 0.05},
    "execution_time_ms": 4.8,
    "processed_image": "data:image/png;base64,...",
    "filename": "digit_7.png",
    "image": {"width": 256, "height": 256, "format": "PNG", "size_bytes": 4213},
    "preprocess_mode": "mnist"
  }
}
```

The probabilities are percentages taken from the model's softmax output. The `processed_image` field is
the 28×28 image the model actually received.

| Status | Code | When |
|---|---|---|
| 400 | `missing_file`, `empty_file`, `invalid_image`, `corrupted_image`, `image_too_small`, `image_too_large` | Bad input |
| 413 | `payload_too_large` | File is larger than `MAX_UPLOAD_MB` |
| 415 | `unsupported_media_type`, `content_mismatch` | Not a PNG/JPEG, or the content doesn't match the extension |
| 422 | `preprocessing_error` | No digit found (blank image) |
| 503 | `model_unavailable` | Model failed to load |

### `GET /api/health/`
Returns 200 with `model_loaded: true`, or 503 when the model isn't loaded.

### `GET /api/model-info/`
Returns the architecture, shapes, parameter count, preprocessing steps and upload limits.

## 6. Security

* Uploads are validated by filename (`secure_filename`), extension, MIME type, actual decoded format
  (PIL), `verify()` (catches corrupted files), dimensions, and size (`MAX_CONTENT_LENGTH`). There is also a
  decompression-bomb guard.
* Images are processed **in memory only**; nothing is written to disk.
* Responses carry CSP, `X-Frame-Options`, `nosniff`, `Referrer-Policy` and `Permissions-Policy` headers.
  API responses are sent with `no-store`.
* Secrets come from environment variables. `DEBUG=False` in production. Never commit `.env`.

## 7. Tests

```bash
python -m pytest
```

The suite has 74 tests. They cover app startup, model loading, real predictions for digits 0–9 (PNG and
JPEG), invalid, unsupported, disguised and corrupted files, oversized files and images, a missing file,
blank images, the health and model-info endpoints, preprocessing invariants (shape, inversion, centring,
the 20 px box, notebook parity), security headers, and the frontend Predict and Clear/Reset contract.

## 8. Production

See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md). Quick start on Linux:

```bash
gunicorn -c deploy/gunicorn.conf.py wsgi:app
```
