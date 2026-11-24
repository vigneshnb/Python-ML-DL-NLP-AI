"""Local development server: `python run.py`."""
import os

from app import create_app

app = create_app(os.getenv("APP_ENV", "development"))

if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "127.0.0.1"),
        port=int(os.getenv("PORT", "5000")),
        debug=app.config["DEBUG"],
        use_reloader=False,  # avoid loading the model twice
        threaded=True,
    )
