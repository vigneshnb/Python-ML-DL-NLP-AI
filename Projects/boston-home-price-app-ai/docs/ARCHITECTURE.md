# Architecture

```
Browser ──JSON──▶ Nginx ──▶ Gunicorn (gthread) ──▶ Flask app
                                                    │
     /           web blueprint ── renders form from app/features.py
     /api/*      api blueprint ── PredictionService
                                    ├─ validation.validate_payload   (schema-driven, 422 on error)
                                    ├─ Preprocessor.transform        ((x-mean)/scale from preprocessing.json)
                                    └─ ModelService.predict          (Keras model, loaded once per worker)
```

## Layout

```
app/
  __init__.py            application factory, security headers, JSON error handlers, logging
  config.py              environment-based config (development / testing / production)
  features.py            CENTRAL FEATURE SCHEMA: order, labels, units, bounds, steps, groups, defaults
  responses.py           {success, message, data|errors} envelope
  api/routes.py          POST /api/predict/, GET /api/health/, GET /api/model-info/
  web/__init__.py        GET / (form generated from the schema)
  services/
    model_service.py     loads .h5 once (thread-safe), validates shapes, warm-up, inference lock
    preprocessing.py     loads/validates scaler params, assembles the vector in model order
    validation.py        request validation against the schema
    prediction_service.py  orchestration + presentation (formatting, warnings, z-scores)
  templates/             base.html, index.html
  static/
    css/style.css        light editorial theme
    js/form-logic.js     pure logic (validation, payload, formatting, initial state), tested in Node
    js/app.js            DOM wiring: validation UI, submit, loading, result, Clear Form
models/
  boston_home_price_prediction.h5   the existing model (unchanged)
  preprocessing.json                training-split StandardScaler params + provenance
scripts/export_preprocessing.py     regenerates preprocessing.json (does not train anything)
tests/                              pytest + Node tests
deploy/                             gunicorn, nginx, systemd
```

## Single source of truth

`app/features.py` defines every feature once. It is used by:

* the Jinja template, which generates every input, label, unit, min/max/step and hint;
* the front-end, through a JSON `<script type="application/json">` block for client validation;
* server-side validation;
* the preprocessor, which checks at startup that `feature_order` in `preprocessing.json` equals the schema order.

UI groups never affect vector order. `Preprocessor.vector()` always iterates `FEATURE_NAMES`.

## Model lifecycle

* One `ModelService` per process, created in `create_app`. With `MODEL_EAGER_LOAD=true`, each
  Gunicorn worker loads the model once at startup and runs a warm-up call.
* `load()` uses double-checked locking. Inference calls `model(x, training=False)` under a
  lock, which takes about 3 ms and is cheaper than `model.predict` for a single row.
* If loading fails, the app still starts. `/api/health/` reports `degraded` and `/api/predict/` returns 503.

## Security

* Strict CSP (`script-src 'self'`, no inline scripts or styles). Fonts come from Google Fonts only.
* `X-Content-Type-Options`, `X-Frame-Options: DENY`, `Referrer-Policy`, `Permissions-Policy`,
  `Cross-Origin-Opener-Policy`, and optional HSTS.
* `MAX_CONTENT_LENGTH` is 16 KB. JSON content-type is required. NaN/Infinity, booleans, strings
  and unknown fields are rejected.
* All API errors use the JSON envelope. Exceptions are logged server-side and never returned
  (no paths, no tracebacks).
* No database and no persistence. Submitted values exist only for the duration of the request.

## Front-end states

`empty → loading → result | error`. **Clear Form** (and "Start a new estimate") does the following:
aborts the in-flight request and bumps a sequence number so late responses are ignored; resets every
input including the CHAS toggle; clears errors, warnings, the visualisation and the summary; restores
the initial state object; and disables the predict button again. The page is not reloaded.
