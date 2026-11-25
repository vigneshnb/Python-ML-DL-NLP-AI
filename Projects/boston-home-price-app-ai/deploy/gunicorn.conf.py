"""Gunicorn configuration for the Home Value Estimator.

Memory: each worker *process* loads its own TensorFlow runtime (~350-500 MB RSS);
the model itself is tiny (3,521 parameters, ~80 KB). Rough sizing:
  1 worker  ~ 0.5 GB      (t3.micro / 1 GB RAM)
  2 workers ~ 1.0 GB      (t3.small / 2 GB RAM)  <- default
Inference takes a few milliseconds and is serialised per worker by a lock, so a
couple of threads per worker is enough; add workers only if RAM allows.
"""

import multiprocessing
import os

bind = os.getenv("GUNICORN_BIND", "127.0.0.1:8000")
workers = int(os.getenv("GUNICORN_WORKERS", "2"))
threads = int(os.getenv("GUNICORN_THREADS", "2"))
worker_class = "gthread"

# Do not preload: forking after TensorFlow has initialised its thread pools can
# deadlock. Each worker loads the model exactly once when it starts.
preload_app = False

timeout = int(os.getenv("GUNICORN_TIMEOUT", "60"))  # covers TF import + model load
graceful_timeout = 30
keepalive = 5
max_requests = 2000            # recycle workers periodically to cap memory growth
max_requests_jitter = 200
limit_request_line = 4094
limit_request_fields = 50

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info").lower()
forwarded_allow_ips = "127.0.0.1"

# Keep TensorFlow from spawning a thread per core in every worker.
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(max(1, multiprocessing.cpu_count() // max(1, workers))))
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
