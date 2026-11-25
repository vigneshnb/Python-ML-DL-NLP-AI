# Architecture & model integration

## 1. Model inspection (`models/cifar10_pred.h5`)

Inspected with Keras 3.15 / TensorFlow 2.21 (`keras.models.load_model(..., compile=False)`).

| Property | Value |
|---|---|
| Format | Legacy Keras HDF5 (`.h5`), loads without modification |
| Architecture | `Flatten → Dense(512, relu) → Dropout(0.2) → Dense(256, relu) → Dropout(0.1) → Dense(128, relu) → Dropout(0.5) → Dense(10, softmax)` |
| Parameters | 1,738,890 (≈ 6.6 MB of float32 weights) |
| Input | `(None, 32, 32, 3)`, float32 |
| Output | `(None, 10)` |
| Output activation | **softmax** → outputs are already probabilities (verified: rows sum to 1.0) |
| Test accuracy (from the training notebook) | **43.6 %** on the CIFAR-10 test set |

> The model is a fully-connected network (no convolutions), so its accuracy is
> modest. Expect frequent mistakes on real-world photos; the app reports the
> model's genuine output and never adjusts it.

### Softmax handling
`ModelService._validate` reads the last layer's activation:
* `softmax` → output used as-is (this model).
* `linear` / none → numerically stable softmax applied once.
* anything else → refuse to serve.

Softmax is therefore never applied twice.

## 2. Preprocessing decisions

Derived from `CIFAR10.ipynb` (training: `cifar10.load_data()` then `x / 255.0`; inference helper:
`resize((32,32)) → convert('RGB') → float32 → /255 → expand_dims`).

| Step | Decision | Reason |
|---|---|---|
| Decode | Pillow, in memory, `verify()` then full `load()` | detects truncated/corrupt files |
| Orientation | `ImageOps.exif_transpose` | phone photos are stored rotated |
| Colour | `convert("RGB")` | CIFAR-10 arrays are RGB; Pillow decodes to RGB, so no BGR swap |
| Resize | 32×32 straight resize, bicubic, no crop | matches the notebook (Pillow's default for RGB is bicubic; set explicitly) |
| Scale | `/ 255.0` → [0, 1] | the exact normalisation used in training (no mean/std) |
| Dtype | float32 | model weights are float32 |
| Batch | `np.expand_dims(..., 0)` → `(1, 32, 32, 3)` | model expects a batch axis |

The target size is read from the loaded model's `input_shape`, not hard-coded.
`tests/test_preprocessing.py::test_preprocess_matches_notebook_pipeline` checks
our tensor equals the notebook's.

## 3. Code layout

```
app/
  __init__.py          create_app(): config, logging, model load, blueprints, security headers, error handlers
  config.py            env-based Development / Testing / Production config
  constants.py         CLASS_NAMES (only place the mapping exists), allowed formats
  responses.py         consistent JSON envelope
  api/routes.py        /api/health/, /api/model-info/, /api/predict/
  web/__init__.py      GET / (renders the UI)
  services/
    model_service.py   load-once ModelService, validation, inference lock, softmax logic
    preprocessing.py   upload validation + tensor preparation
  templates/index.html
  static/css/style.css
  static/js/state.js   pure UI state logic (unit-tested with Node)
  static/js/app.js     DOM wiring, Chart.js, fetch
models/cifar10_pred.h5
tests/                 pytest (+ tests/js for Node)
deploy/                gunicorn, nginx, systemd
wsgi.py / run.py       production / development entry points
```

## 4. Request flow — `POST /api/predict/`

1. Field `image` present? else `400 missing_file`.
2. Size ≤ `MAX_UPLOAD_MB` (Flask `MAX_CONTENT_LENGTH` + explicit check) else `413 file_too_large`.
3. Model loaded? else `503 model_unavailable`.
4. Extension in {png,jpg,jpeg,webp} else `415`; decoded format in {PNG,JPEG,WEBP} else `415`
   (a renamed BMP is rejected); undecodable → `400 corrupted_image`;
   >40 MP → `413`; dimensions outside `[MIN, MAX]` → `400 invalid_dimensions`.
5. Preprocess → `ModelService.predict` (under a lock) → JSON.

The upload is never written to disk and the filename is never used as a path.
`execution_time_ms` = preprocessing + inference; `inference_time_ms` = model call only.

## 5. Concurrency

One `ModelService` per process, loaded at start-up. Inference is guarded by a
`threading.Lock` (Keras models are not guaranteed thread-safe); this model runs
in a few milliseconds, so the lock is not a bottleneck. See
`deploy/gunicorn.conf.py` for worker/thread sizing and memory notes.

## 6. Frontend

* `state.js` holds the state machine (`idle → ready → loading → success | error`),
  `resultView()` (which result panel to show) and `resetState()`. `app.js`
  renders everything from that state, so every view is derived, never toggled ad hoc.
* File problems (type, size, dimensions, unreadable) show under the upload and
  block Predict; API failures show in the result card with **Try again**.
* Predict is disabled until a valid image is selected and while a request is in
  flight (no duplicate submissions).
* Clear (button, the preview's x, or `Esc`) revokes the preview object URL,
  resets the file input, state, result card, gauge, table and chart, returns focus
  to **Browse files**, and bumps a request counter so a late response from a
  cancelled prediction is ignored.
* Accessibility: real `<button>`s for every action, visible focus rings, an
  `aria-live` region announcing the result, `aria-busy` while loading, and
  `prefers-reduced-motion` support. Only a few animations remain: fades, the
  loading skeleton and the gauge fill.
* History lives only in page memory (160 px JPEG data-URL thumbnails); nothing
  is persisted on the server or in browser storage.
