"""CropVision Flask application. Routes only - ML logic lives in predict.py / imaging.py."""
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from flask import Flask, flash, jsonify, redirect, render_template, request, url_for, abort
from PIL import Image

import db
import disease_info
from config import Config
from imaging import InvalidImage, open_image, quality_note
from predict import ModelNotConfigured, Predictor

log = logging.getLogger("cropvision")


def create_app(config_overrides: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if config_overrides:
        app.config.update(config_overrides)
    for d in (app.config["UPLOAD_DIR"], app.config["THUMB_DIR"], Path(app.config["DATABASE"]).parent):
        Path(d).mkdir(parents=True, exist_ok=True)
    db.init_db(app)
    # A pre-built predictor may be injected (tests); otherwise load from models/.
    app.extensions["predictor"] = app.config.get("PREDICTOR") or Predictor(
        app.config["MODELS_DIR"], app.config["MODEL_PATH"], app.config["LOW_CONFIDENCE"])

    @app.template_filter("localtime")
    def localtime(value):
        try:
            return datetime.fromisoformat(value).astimezone().strftime("%d %b %Y, %I:%M %p")
        except (TypeError, ValueError):
            return value or ""

    @app.template_filter("pct")
    def pct(value):
        return f"{value * 100:.1f}%"

    @app.context_processor
    def inject_globals():
        return {"model_status": app.extensions["predictor"].status()}

    # ---- pages ----
    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/detect")
    def detect():
        return render_template("detect.html", max_mb=app.config["MAX_CONTENT_LENGTH"] // (1024 * 1024))

    @app.get("/results/<int:scan_id>")
    def results(scan_id):
        return render_scan(scan_id, "results.html")

    @app.get("/history")
    def history():
        scans = db.list_scans()
        rows = [{"scan": s, "name": disease_info.display_name(s["label"])} for s in scans]
        return render_template("history.html", rows=rows)

    @app.get("/about")
    def about():
        return render_template("about.html", sources=disease_info.SOURCES)

    @app.post("/history/<int:scan_id>/delete")
    def delete(scan_id):
        row = db.delete_scan(scan_id)
        if row:
            for folder, name in ((app.config["UPLOAD_DIR"], row["filename"]), (app.config["THUMB_DIR"], row["thumb"])):
                try:
                    (Path(folder) / name).unlink(missing_ok=True)
                except OSError:
                    log.warning("Could not remove file %s", name)
            flash("Scan deleted.", "success")
        return redirect(url_for("history"))

    def render_scan(scan_id, template):
        scan = db.get_scan(scan_id)
        if scan is None:
            abort(404)
        top = [{"name": disease_info.display_name(t["label"]), "confidence": t["confidence"]}
               for t in json.loads(scan["top_json"])]
        return render_template(template, scan=scan, name=disease_info.display_name(scan["label"]),
                               info=disease_info.lookup(scan["label"]), top=top,
                               sources=disease_info.SOURCES)

    # ---- API ----
    @app.get("/api/status")
    def api_status():
        return jsonify(app.extensions["predictor"].status())

    @app.post("/api/predict")
    def api_predict():
        predictor = app.extensions["predictor"]
        if not predictor.available:
            msg = predictor.status()["error"] or "Model inference is not configured."
            return jsonify(error=f"Model inference is not configured. {msg}", code="model_not_configured"), 503
        file = request.files.get("image")
        if file is None or not file.filename:
            return jsonify(error="No image was uploaded.", code="no_file"), 400
        ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
        if ext not in app.config["ALLOWED_EXTENSIONS"]:
            return jsonify(error="Unsupported file type. Please upload a JPG, PNG or WebP image.", code="bad_type"), 415
        try:
            img = open_image(file.stream)
        except InvalidImage as exc:
            return jsonify(error=str(exc), code="invalid_image"), 422
        try:
            result = predictor.predict(img)
        except ModelNotConfigured as exc:
            return jsonify(error=str(exc), code="model_not_configured"), 503
        except Exception:
            log.exception("Inference failed")
            return jsonify(error="The model failed to process this image.", code="inference_error"), 500

        # Store a re-encoded copy (never the raw upload) under a random filename.
        stem = uuid.uuid4().hex
        filename, thumb = f"{stem}.jpg", f"{stem}.jpg"
        stored = img.copy()
        stored.thumbnail((app.config["MAX_STORED_SIDE"],) * 2)
        stored.save(Path(app.config["UPLOAD_DIR"]) / filename, "JPEG", quality=90)
        small = img.copy()
        small.thumbnail((app.config["THUMB_SIDE"],) * 2)
        small.save(Path(app.config["THUMB_DIR"]) / thumb, "JPEG", quality=85)
        scan_id = db.add_scan(filename, thumb, result, quality_note(img))
        return jsonify(scan_id=scan_id, redirect=url_for("results", scan_id=scan_id))

    # ---- errors ----
    @app.errorhandler(404)
    def not_found(_e):
        return render_template("error.html", code=404, message="That page or scan could not be found."), 404

    @app.errorhandler(413)
    def too_large(_e):
        msg = f"That file is larger than the {app.config['MAX_CONTENT_LENGTH'] // (1024 * 1024)} MB limit."
        if request.path.startswith("/api/"):
            return jsonify(error=msg, code="too_large"), 413
        return render_template("error.html", code=413, message=msg), 413

    @app.errorhandler(500)
    def server_error(_e):
        if request.path.startswith("/api/"):
            return jsonify(error="Something went wrong on the server.", code="server_error"), 500
        return render_template("error.html", code=500, message="Something went wrong on our side. Please try again."), 500

    return app


app = create_app()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app.run(host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", 5000)),
            debug=os.environ.get("FLASK_DEBUG") == "1")
