from flask import Blueprint, current_app, render_template

from app.constants import APP_NAME, APP_SUBTITLE, CLASS_NAMES

web_bp = Blueprint("web", __name__)


@web_bp.get("/")
def index():
    return render_template(
        "index.html",
        app_name=APP_NAME,
        app_subtitle=APP_SUBTITLE,
        class_names=CLASS_NAMES,
        max_upload_mb=current_app.config["MAX_UPLOAD_MB"],
        min_dim=current_app.config["MIN_IMAGE_DIMENSION"],
        max_dim=current_app.config["MAX_IMAGE_DIMENSION"],
    )
