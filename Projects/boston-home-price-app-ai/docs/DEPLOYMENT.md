# Production deployment: Ubuntu 22.04/24.04 on AWS EC2

Assumes the app lives in `/opt/home-value-estimator`, runs as user `hve`, and sits behind Nginx.

## 1. Instance

* **Size:** t3.small (2 GB) for 2 workers, or t3.micro (1 GB) with `GUNICORN_WORKERS=1`.
  TensorFlow takes about 0.4–0.5 GB per worker. The model itself is tiny.
* **Security group:** allow 22 (from your IP only), 80 and 443. Do **not** open 8000.

## 2. System packages

```bash
sudo apt update && sudo apt -y upgrade
sudo apt -y install python3.12 python3.12-venv python3-pip nginx git
# Ubuntu 22.04 does not ship 3.12: sudo add-apt-repository ppa:deadsnakes/ppa && sudo apt install python3.12 python3.12-venv
```

## 3. Code, virtualenv and configuration

```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin hve
sudo mkdir -p /opt/home-value-estimator && sudo chown $USER:www-data /opt/home-value-estimator
git clone <your-repo-url> /opt/home-value-estimator      # or rsync/scp the project folder
cd /opt/home-value-estimator

python3.12 -m venv .venv
.venv/bin/pip install --upgrade pip
# On a CPU-only server you can swap tensorflow for tensorflow-cpu==2.21.0 (smaller download)
.venv/bin/pip install -r requirements.txt

cp .env.example .env
python3 -c "import secrets; print(secrets.token_hex(32))"   # paste into SECRET_KEY
nano .env                                                  # APP_ENV=production, SECRET_KEY=...
chmod 640 .env

sudo chown -R hve:www-data /opt/home-value-estimator
```

Check that the model and preprocessing files are present:

```bash
ls -l models/boston_home_price_prediction.h5 models/preprocessing.json
```

Smoke test before wiring systemd:

```bash
sudo -u hve .venv/bin/gunicorn -c deploy/gunicorn.conf.py wsgi:app &
curl -s localhost:8000/api/health/
curl -s -X POST localhost:8000/api/predict/ -H 'Content-Type: application/json' \
  -d '{"features":{"CRIM":0.02,"ZN":18,"INDUS":2.5,"CHAS":0,"NOX":0.45,"RM":6.5,"AGE":65,"DIS":4.5,"RAD":1,"TAX":300,"PTRATIO":15,"B":390,"LSTAT":5}}'
# expect "predicted_value": 28.07...
kill %1
```

## 4. systemd

```bash
sudo cp deploy/systemd/home-value-estimator.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now home-value-estimator
sudo systemctl status home-value-estimator
```

## 5. Nginx

```bash
sudo cp deploy/nginx/home-value-estimator.conf /etc/nginx/sites-available/home-value-estimator
sudo sed -i 's/example.com/your-domain.com/g' /etc/nginx/sites-available/home-value-estimator
sudo ln -s /etc/nginx/sites-available/home-value-estimator /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

## 6. SSL (Let's Encrypt)

Point your domain's DNS A record at the instance's Elastic IP, then run:

```bash
sudo apt -y install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com -d www.your-domain.com --redirect
sudo certbot renew --dry-run             # renewal is automatic via systemd timer
```

Then set `ENABLE_HSTS=true` in `.env` and restart the app.

## 7. Restart, update, logs

```bash
sudo systemctl restart home-value-estimator      # full restart (reloads model)
sudo systemctl reload home-value-estimator       # graceful worker reload (HUP)

# update
cd /opt/home-value-estimator && sudo -u hve git pull
sudo -u hve .venv/bin/pip install -r requirements.txt
sudo systemctl restart home-value-estimator

# logs
sudo journalctl -u home-value-estimator -f                # app + gunicorn
sudo tail -f /var/log/nginx/home-value-estimator.access.log
sudo tail -f /var/log/nginx/home-value-estimator.error.log
```

## 8. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `502 Bad Gateway` | Gunicorn is not running. Check `systemctl status` and `journalctl -u home-value-estimator`. |
| Worker killed / `SIGKILL` / OOM | Too many workers for the RAM. Set `GUNICORN_WORKERS=1` or use a bigger instance. |
| `WORKER TIMEOUT` at startup | TensorFlow import is slow on small instances. Raise `GUNICORN_TIMEOUT` (e.g. 120). |
| `/api/health/` shows `degraded` | Model failed to load. Check that `models/*.h5` exists and is readable by `hve`, and look in the logs for the cause. |
| `RuntimeError: SECRET_KEY must be set` | `.env` is missing or not readable, or `SECRET_KEY` is empty. |
| `Feature order ... does not match` at startup | `models/preprocessing.json` was edited or replaced. Regenerate it with `scripts/export_preprocessing.py`. |
| `413` from the API | Body over 16 KB. A valid request is about 300 bytes. |
| `429` from Nginx | Rate limit hit (10 req/s per IP, burst 20). Adjust `limit_req_zone`. |
| Fonts look different | Google Fonts are blocked. The UI falls back to system fonts, which is harmless. |
| `ImportError: libstdc++` / illegal instruction | Use a 64-bit x86_64 or arm64 Ubuntu AMI with Python 3.12 and TF 2.21 wheels. |
