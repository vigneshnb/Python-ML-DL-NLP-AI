# Production deployment — Ubuntu 22.04/24.04 on AWS EC2

## 0. Sizing

TensorFlow is the main memory cost (~400–600 MB per Gunicorn worker process;
the model itself is ~20 MB). Recommended:

| Instance | RAM | `GUNICORN_WORKERS` | `GUNICORN_THREADS` |
|---|---|---|---|
| t3.micro | 1 GB | 1 (add 1–2 GB swap) | 4 |
| t3.small | 2 GB | 1–2 | 4 |
| t3.medium | 4 GB | 2–3 | 4 |

Do not raise workers blindly — every worker loads its own TensorFlow + model.
Security group: allow 22 (your IP), 80, 443.

## 1. System packages

```bash
sudo apt update && sudo apt -y upgrade
sudo apt -y install python3.12 python3.12-venv python3-pip nginx git
# Ubuntu 22.04 ships 3.10; get 3.12 via: sudo add-apt-repository ppa:deadsnakes/ppa
```

## 2. App user and code

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin cifar
sudo mkdir -p /opt/cifar-vision-ai && sudo chown cifar:www-data /opt/cifar-vision-ai
sudo -u cifar git clone <your-repo-url> /opt/cifar-vision-ai   # or rsync/scp the folder
```

The model is git-ignored by default. Copy it separately:

```bash
scp models/cifar10_pred.h5 ubuntu@<ec2-ip>:/tmp/
sudo mv /tmp/cifar10_pred.h5 /opt/cifar-vision-ai/models/
sudo chown cifar:www-data /opt/cifar-vision-ai/models/cifar10_pred.h5
sudo chmod 640 /opt/cifar-vision-ai/models/cifar10_pred.h5
```
(Alternatives: `aws s3 cp s3://bucket/cifar10_pred.h5 ...`, or Git LFS.)

## 3. Virtual environment

```bash
cd /opt/cifar-vision-ai
sudo -u cifar python3.12 -m venv .venv
sudo -u cifar .venv/bin/pip install --upgrade pip
sudo -u cifar .venv/bin/pip install -r requirements.txt
# CPU-only server, smaller install:  pip install tensorflow-cpu==2.21.0 (instead of tensorflow)
```

## 4. Environment

```bash
sudo -u cifar cp .env.example .env
sudo -u cifar nano .env      # APP_ENV=production, SECRET_KEY=<random>, MODEL_PATH=/opt/cifar-vision-ai/models/cifar10_pred.h5
sudo chmod 600 .env
```

Generate a key: `python3 -c "import secrets; print(secrets.token_hex(32))"`.

Smoke test:
```bash
sudo -u cifar .venv/bin/python -m pytest -q        # needs requirements-dev.txt
sudo -u cifar .venv/bin/gunicorn -c deploy/gunicorn.conf.py wsgi:app   # Ctrl+C after "Model loaded"
```

## 5. systemd

```bash
sudo cp deploy/systemd/cifar-vision-ai.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now cifar-vision-ai
sudo systemctl status cifar-vision-ai
curl -s http://127.0.0.1:8000/api/health/
```

## 6. Nginx

```bash
sudo cp deploy/nginx/cifar-vision-ai.conf /etc/nginx/sites-available/
sudo sed -i 's/your-domain.com/example.com/g' /etc/nginx/sites-available/cifar-vision-ai.conf
sudo ln -s /etc/nginx/sites-available/cifar-vision-ai.conf /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```
Nginx's `client_max_body_size` (6M) must stay ≥ `MAX_UPLOAD_MB`.
Nginx needs read access to `app/static/` (group `www-data`).

## 7. SSL (Let's Encrypt)

Point your domain's A record at the EC2 Elastic IP, then:

```bash
sudo apt -y install certbot python3-certbot-nginx
sudo certbot --nginx -d example.com -d www.example.com --redirect
sudo systemctl status certbot.timer      # auto-renewal
```
Keep `ENABLE_HSTS=true` only once HTTPS works. Behind an AWS ALB instead,
terminate TLS with an ACM certificate on the load balancer.

## 8. Operating

| Task | Command |
|---|---|
| Restart | `sudo systemctl restart cifar-vision-ai` |
| Graceful reload (code change) | `sudo systemctl reload cifar-vision-ai` |
| Stop / start | `sudo systemctl stop|start cifar-vision-ai` |
| App logs (live) | `sudo journalctl -u cifar-vision-ai -f` |
| Recent errors | `sudo journalctl -u cifar-vision-ai -p err --since "1 hour ago"` |
| Nginx logs | `/var/log/nginx/access.log`, `/var/log/nginx/error.log` |
| Health | `curl -s https://example.com/api/health/` |

Updating: `git pull && .venv/bin/pip install -r requirements.txt && sudo systemctl restart cifar-vision-ai`.

## 9. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `/api/health/` shows `model_status: "error"` | Wrong `MODEL_PATH`, missing file, or permissions. Check `journalctl` for "Model file not found" / "Failed to load model". |
| 502 Bad Gateway | Gunicorn not running or still loading TF: `systemctl status`, increase `TimeoutStartSec`. |
| 413 from Nginx (HTML page) | `client_max_body_size` smaller than the upload. |
| Worker killed / `SIGKILL` / OOM in `dmesg` | Too many workers for RAM: set `GUNICORN_WORKERS=1`, add swap, or a larger instance. |
| Very slow first request | Model warm-up happens at worker start; ensure `MODEL_EAGER_LOAD=true`. |
| `Illegal instruction` on import tensorflow | CPU lacks AVX (rare on EC2); use a newer instance family. |
| `SECRET_KEY must be set` | `.env` not loaded: check `EnvironmentFile=` path. |
