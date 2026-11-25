# CIFAR Vision AI

AI image classification web app built with Flask around an **existing trained
Keras model** (`models/cifar10_pred.h5`). Upload an image and get the predicted
CIFAR-10 class, the confidence, and the full probability distribution for all
10 classes.

Airplane · Automobile · Bird · Cat · Deer · Dog · Frog · Horse · Ship · Truck

## Features
- Large drag-and-drop upload area with a **Browse files** button and clipboard paste (PNG, JPG, JPEG, WEBP)
- Image preview with file name, size, dimensions and type, plus a remove action
- Predict button is disabled until a valid image is chosen and while a request runs (no duplicate submissions)
- Result card states: empty, skeleton loading, success, and error with **Try again**
- Success card: predicted class (large), class number, preview, prediction time and a confidence ring (High / Moderate / Low)
- Chart.js horizontal bar chart of all 10 classes with % labels and the predicted class highlighted; exact values in an expandable table
- **Clear** restores the initial UI (image, preview, result, chart, errors, time) without reloading
- Keyboard: `Ctrl+Enter` predict, `Esc` clear (or close the mobile menu), `Ctrl+V` paste; visible focus rings; screen-reader announcements
- Recent-predictions history held only in page memory; live model status and model info from the API
- Responsive dark UI (desktop, tablet, mobile); respects `prefers-reduced-motion`

## Model
| | |
|---|---|
| Input | `(None, 32, 32, 3)` float32, RGB, values in [0, 1] |
| Output | `(None, 10)`, softmax probabilities |
| Preprocessing | RGB → resize 32×32 (bicubic) → `/255.0` → add batch axis |
| Test accuracy | 43.6 % (reported by the training notebook) |

Full findings and preprocessing rationale: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start (local)

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env                   # set APP_ENV=development for local work
python run.py                          # http://127.0.0.1:5000
```

The model must be at `models/cifar10_pred.h5` (or set `MODEL_PATH`).

## Tests

```bash
python -m pytest
```
* `test_model_real.py`: loads the real model and checks shapes, softmax, real predictions on sample images, and that API numbers equal a direct model call.
* `test_api_validation.py`: request validation using a stub model (isolated HTTP-layer tests).
* `test_preprocessing.py`: tensor shape, dtype, range, parity with the notebook pipeline.
* `test_frontend.py`: page rendering, and Clear/Reset state logic run in Node.js (`tests/js/state.test.mjs`; skipped if Node is missing).

## API

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/predict/` | multipart field `image` → prediction |
| `GET` | `/api/health/` | app + model status |
| `GET` | `/api/model-info/` | model name, shapes, classes, framework |

```json
{
  "success": true,
  "message": "Image classified successfully",
  "data": {
    "predicted_class": 7,
    "predicted_label": "Horse",
    "confidence": 98.3,
    "probabilities": { "Airplane": 0.2, "...": "...", "Truck": 0.1 },
    "execution_time_ms": 12.4,
    "inference_time_ms": 9.8,
    "image": { "width": 114, "height": 111, "format": "PNG" }
  }
}
```
Errors: `{"success": false, "message": "...", "error": {"code": "unsupported_type"}}`.
Codes: `missing_file`, `empty_file`, `unsupported_type` (415), `corrupted_image`,
`invalid_dimensions`, `file_too_large` (413), `image_too_large` (413),
`model_unavailable` (503), `inference_failed` (500).

```bash
curl -F "image=@tests/samples/horse.png" http://127.0.0.1:5000/api/predict/
```

## Configuration (`.env`)
See [.env.example](.env.example). Key settings: `APP_ENV`, `SECRET_KEY`,
`MODEL_PATH`, `MAX_UPLOAD_MB`, `MIN_IMAGE_DIMENSION`, `MAX_IMAGE_DIMENSION`, `LOG_LEVEL`, `ENABLE_HSTS`.
Production forces `DEBUG=False` and refuses to start without `SECRET_KEY`.

## Security
- Extension and decoded-format allow-lists (a renamed BMP is rejected), size, dimension and 40 MP decompression-bomb limits
- Uploads processed in memory only, never written to disk; filename never used as a path
- CSP, `X-Frame-Options`, `nosniff`, `Referrer-Policy`, `Permissions-Policy`, optional HSTS
- JSON errors never include stack traces or filesystem paths

## Production
Gunicorn + Nginx + systemd on Ubuntu / AWS EC2: see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)
(sizing, SSL, restart, logs, troubleshooting). Gunicorn does not run on Windows;
use `python run.py` for local development there.

## Model file & Git
`*.h5` is git-ignored on purpose (20 MB binary). Ship it with `scp`/S3, or use
Git LFS if you decide to version it.
