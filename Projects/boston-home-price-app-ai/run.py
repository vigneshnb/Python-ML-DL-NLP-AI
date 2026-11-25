"""Development server entry point.  Production: gunicorn -c deploy/gunicorn.conf.py wsgi:app"""

import os

os.environ.setdefault("APP_ENV", "development")

from app import create_app  # noqa: E402

app = create_app()

if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=app.config["DEBUG"],
        use_reloader=False,  # avoid loading TensorFlow twice
    )
