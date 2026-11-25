"""Gunicorn config tuned for a TensorFlow model.

Memory: each worker *process* loads its own copy of TensorFlow + the model.
TensorFlow runtime ~350-500 MB RSS; this model adds ~20 MB. So:
  1 worker  ≈ 0.4-0.6 GB
  2 workers ≈ 0.8-1.2 GB
On a 1 GB instance (t3.micro) use 1 worker. Scale concurrency with threads
(inference is serialised by a lock but takes only a few ms; threads still
overlap upload/decoding I/O) and add workers only if you have RAM to spare.
"""

import multiprocessing
import os

bind = os.getenv("GUNICORN_BIND", "127.0.0.1:8000")
workers = int(os.getenv("GUNICORN_WORKERS", "1"))
threads = int(os.getenv("GUNICORN_THREADS", "4"))
worker_class = "gthread"

# Do NOT preload: forking after TensorFlow initialises its thread pools can
# deadlock. Each worker loads the model once at start-up instead.
preload_app = False

timeout = int(os.getenv("GUNICORN_TIMEOUT", "60"))     # model load + first request
graceful_timeout = 30
keepalive = 5
max_requests = 1000          # recycle workers periodically to cap memory growth
max_requests_jitter = 100

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info").lower()

# Avoid TF grabbing every core per worker
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(max(1, multiprocessing.cpu_count() // max(1, workers))))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
