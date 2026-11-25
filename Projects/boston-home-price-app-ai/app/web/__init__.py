"""HTML pages."""

from flask import Blueprint, current_app, render_template

from app.features import FEATURES, GROUPS, schema_for_client

web_bp = Blueprint("web", __name__)


@web_bp.get("/")
def index():
    pre = current_app.extensions["preprocessor"]
    ranges = pre.training_ranges()
    schema = schema_for_client(ranges)
    schema["target"] = {
        "train_min": pre.target.get("train_min"),
        "train_max": pre.target.get("train_max"),
        "train_median": pre.target.get("train_median"),
    }
    groups = [
        {**g, "features": [f for f in FEATURES if f.group == g["id"]]}
        for g in GROUPS
    ]
    return render_template("index.html", groups=groups, ranges=ranges, schema=schema)
