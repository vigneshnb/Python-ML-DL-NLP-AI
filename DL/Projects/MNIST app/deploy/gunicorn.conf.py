"""Gunicorn configuration for MNIST AI.

Run with:  gunicorn -c deploy/gunicorn.conf.py wsgi:app

TensorFlow notes:
* `preload_app = False` — TensorFlow is not fork-safe once initialised, so every
  worker imports TF and loads the model itself (once, at worker start).
* Each worker holds its own model copy (~ small for this MLP, TF runtime ~300-400 MB RAM).
  Size workers by RAM: on a 1 GB t3.micro use 1-2 workers, on 4 GB use 2-4.
* `gthread` workers let one process serve concurrent I/O while inference is
  serialised by a lock inside ModelService.
"""
import multiprocessing
import os

bind = os.getenv("GUNICORN_BIND", "127.0.0.1:8000")
workers = int(os.getenv("GUNICORN_WORKERS", min(2, multiprocessing.cpu_count())))
worker_class = "gthread"
threads = int(os.getenv("GUNICORN_THREADS", 4))
timeout = int(os.getenv("GUNICORN_TIMEOUT", 60))
graceful_timeout = 30
keepalive = 5
preload_app = False

# Recycle workers periodically to bound memory growth.
max_requests = 1000
max_requests_jitter = 100

# Limit request line/header sizes (body size is enforced by Flask + Nginx).
limit_request_line = 4094
limit_request_fields = 100

accesslog = "-"
errorlog = "-"
loglevel = os.getenv("LOG_LEVEL", "info").lower()
proc_name = "mnist-ai"

raw_env = [f"TF_CPP_MIN_LOG_LEVEL={os.getenv('TF_CPP_MIN_LOG_LEVEL', '2')}"]
