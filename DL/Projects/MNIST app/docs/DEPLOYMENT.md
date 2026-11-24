# Deploying MNIST AI on AWS EC2 (Ubuntu 24.04)

## 1. Instance

* **Type:** at least t3.small (2 GB RAM). TensorFlow uses roughly 300–400 MB per Gunicorn worker.
* **Security group:** allow 22 (SSH, your IP only), 80 and 443.
* **AMI:** Ubuntu 24.04 LTS. You can install Python 3.12 from the system packages.

## 2. System packages

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv nginx git
```
```bash
sudo useradd --system --create-home --shell /usr/sbin/nologin mnist
```

## 3. Application

```bash
sudo mkdir -p /opt/mnist-ai && sudo chown $USER:www-data /opt/mnist-ai
```
```bash
git clone <your-repo-url> /opt/mnist-ai     # or: scp -r "MNIST app"/* ec2:/opt/mnist-ai/
```
```bash
cd /opt/mnist-ai && python3.12 -m venv .venv && .venv/bin/pip install --upgrade pip
```
```bash
.venv/bin/pip install -r requirements.txt   # tip: swap tensorflow for tensorflow-cpu==2.21.0 to save space
```

Make sure the model is in place at `models/digit_brain_intel.h5`. It is not generated during deployment.

## 4. Environment

```bash
cp .env.example .env
```
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"   # put the output in SECRET_KEY
```

In `.env`, set `APP_ENV=production` and `SECRET_KEY`. Change `GUNICORN_WORKERS` if needed: 1–2 for 2 GB of
RAM, 2–4 for 4 GB or more.

```bash
chmod 600 .env && sudo chown -R mnist:www-data /opt/mnist-ai
```

## 5. Verify

```bash
sudo -u mnist .venv/bin/python -m pytest -q
```
```bash
sudo -u mnist .venv/bin/gunicorn -c deploy/gunicorn.conf.py wsgi:app   # Ctrl+C after checking
```
```bash
curl -s http://127.0.0.1:8000/api/health/
```

## 6. systemd service

```bash
sudo cp deploy/systemd/mnist-ai.service /etc/systemd/system/
```
```bash
sudo systemctl daemon-reload && sudo systemctl enable --now mnist-ai
```
```bash
sudo journalctl -u mnist-ai -f
```

## 7. Nginx

Edit `server_name` in `deploy/nginx/mnist-ai.conf`, then:

```bash
sudo cp deploy/nginx/mnist-ai.conf /etc/nginx/sites-available/
```
```bash
sudo ln -sf /etc/nginx/sites-available/mnist-ai.conf /etc/nginx/sites-enabled/ && sudo rm -f /etc/nginx/sites-enabled/default
```
```bash
sudo nginx -t && sudo systemctl reload nginx
```

## 8. HTTPS (recommended)

```bash
sudo apt install -y certbot python3-certbot-nginx && sudo certbot --nginx -d your-domain.com
```

After that, uncomment the `Strict-Transport-Security` header in the Nginx config and reload Nginx.

## 9. Operations

| Task | Command |
|---|---|
| Restart the app | `sudo systemctl restart mnist-ai` |
| Reload workers gracefully | `sudo systemctl reload mnist-ai` |
| Logs | `sudo journalctl -u mnist-ai --since "1 hour ago"` |
| Health check (for a load balancer / uptime monitor) | `GET /api/health/` (200 = healthy, 503 = model not loaded) |

When you change `MAX_UPLOAD_MB`, also update `client_max_body_size` in Nginx.
