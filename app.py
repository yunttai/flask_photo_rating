import os
import uuid
from datetime import datetime
from typing import Tuple

import numpy as np
from PIL import Image, ImageOps
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, abort, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import cv2

# ------------------- Config -------------------
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, os.getenv("FLASK_UPLOAD_DIR", "uploads"))
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-me")
app.config["SQLALCHEMY_DATABASE_URI"] = os.getenv("SQLALCHEMY_DATABASE_URI", "sqlite:///photo_rating.db")
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25MB
ALLOWED_EXTS = {"png", "jpg", "jpeg", "webp"}

db = SQLAlchemy(app)

# ------------------- DB Model -------------------
class Photo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False, unique=True)
    original_name = db.Column(db.String(255), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    score = db.Column(db.Float, nullable=False)

    def url(self):
        return url_for("uploaded_file", filename=self.filename)

# ------------------- Utils -------------------
def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTS

def pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    # Convert to RGB numpy array [H,W,3], dtype=float32 in [0,255]
    if img.mode not in ("RGB", "RGBA", "L"):
        img = img.convert("RGB")
    if img.mode == "RGBA":
        img = img.convert("RGB")
    if img.mode == "L":
        img = ImageOps.colorize(img, black="black", white="white").convert("RGB")
    arr = np.asarray(img).astype(np.float32)
    return arr

def compute_colorfulness(arr_rgb: np.ndarray) -> float:
    # Hasler & Süsstrunk (2003)
    R, G, B = arr_rgb[...,0], arr_rgb[...,1], arr_rgb[...,2]
    rg = R - G
    yb = 0.5*(R + G) - B
    rg_std, rg_mean = np.std(rg), np.mean(rg)
    yb_std, yb_mean = np.std(yb), np.mean(yb)
    return np.sqrt(rg_std**2 + yb_std**2) + 0.3*np.sqrt(rg_mean**2 + yb_mean**2)

def compute_sharpness(gray: np.ndarray) -> float:
    # Variance of Laplacian
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())

def compute_exposure_contrast(luma: np.ndarray) -> Tuple[float, float]:
    # luma in [0,255]
    mean = float(luma.mean())
    std = float(luma.std())
    # Exposure score: ideal around mid (120~140). Penalize extremes.
    ideal = 130.0
    rng = 130.0  # tolerance
    exposure = 1.0 - min(abs(mean - ideal)/rng, 1.0)
    # Contrast score: normalize by a reference std
    ref_std = 50.0
    contrast = min(std / ref_std, 1.5) / 1.5  # clamp to [0,1]
    return exposure, contrast

def compute_rule_of_thirds_score(edges: np.ndarray) -> float:
    # edges: binary edge map (0/255)
    h, w = edges.shape
    ys, xs = np.nonzero(edges)
    if len(xs) == 0:
        return 0.4  # no edges → neutral-ish
    cx = xs.mean() / w
    cy = ys.mean() / h
    thirds = np.array([1/3, 2/3], dtype=np.float32)
    dx = min(abs(cx - thirds[0]), abs(cx - thirds[1]))
    dy = min(abs(cy - thirds[0]), abs(cy - thirds[1]))
    # Closer to thirds → higher score
    dist = np.sqrt(dx*dx + dy*dy)
    # Max possible distance in unit square ~0.75; map to [0,1]
    score = 1.0 - min(dist / 0.75, 1.0)
    return float(score)

def aesthetic_score(img_path: str) -> float:
    # Load image (downscale for speed if huge)
    with Image.open(img_path) as im:
        im = ImageOps.exif_transpose(im)  # respect EXIF orientation
        max_side = 1080
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side))
        arr = pil_to_np_rgb(im)
    # Luma
    luma = 0.2126*arr[...,0] + 0.7152*arr[...,1] + 0.0722*arr[...,2]
    luma = luma.astype(np.float32)
    # Sharpness via Laplacian on grayscale
    gray = (luma/255.0).astype(np.float32)
    gray8 = (gray*255).astype(np.uint8)
    sharp = compute_sharpness(gray8)
    # Normalize sharpness ~ empirical
    sharp_score = min(sharp/1000.0, 1.0)

    exposure, contrast = compute_exposure_contrast(luma)
    colorfulness = compute_colorfulness(arr)
    # Normalize colorfulness by a reference (empirical 100)
    color_score = min(colorfulness/100.0, 1.0)

    # Edges for composition
    edges = cv2.Canny(gray8, 50, 150)
    thirds_score = compute_rule_of_thirds_score(edges)

    # Weighted sum → [0,1]
    w_expo, w_con, w_sharp, w_color, w_thirds = 0.2, 0.2, 0.25, 0.15, 0.2
    s = (
        w_expo * exposure +
        w_con * contrast +
        w_sharp * sharp_score +
        w_color * color_score +
        w_thirds * thirds_score
    )
    s = max(0.0, min(1.0, float(s)))
    # Map to 1.0~5.0 stars
    stars = 1.0 + 4.0*s
    # one decimal place
    return round(stars, 1)

# ------------------- Routes -------------------
@app.route("/")
def index():
    count = Photo.query.count()
    latest = Photo.query.order_by(Photo.uploaded_at.desc()).limit(12).all()
    return render_template("index.html", count=count, latest=latest)

@app.route("/upload", methods=["POST"])
def upload():
    if "photo" not in request.files:
        flash("파일이 없습니다.", "danger")
        return redirect(url_for("index"))
    file = request.files["photo"]
    if file.filename == "":
        flash("파일명이 비어 있습니다.", "warning")
        return redirect(url_for("index"))
    if not allowed_file(file.filename):
        flash("허용되지 않는 파일 형식입니다. (png, jpg, jpeg, webp)", "warning")
        return redirect(url_for("index"))

    original = secure_filename(file.filename)
    ext = original.rsplit(".", 1)[-1].lower()
    fname = f"{uuid.uuid4().hex}.{ext}"
    save_path = os.path.join(UPLOAD_DIR, fname)
    file.save(save_path)

    # Rate
    try:
        score = aesthetic_score(save_path)
    except Exception as e:
        # Remove bad file
        if os.path.exists(save_path):
            os.remove(save_path)
        flash(f"평가 중 오류: {e}", "danger")
        return redirect(url_for("index"))

    p = Photo(filename=fname, original_name=original, score=score)
    db.session.add(p)
    db.session.commit()
    flash("업로드 및 평점 계산 완료!", "success")
    return redirect(url_for("gallery"))

@app.route("/gallery")
def gallery():
    photos = Photo.query.order_by(Photo.uploaded_at.desc()).all()
    return render_template("gallery.html", photos=photos)

@app.route("/image/<int:pid>")
def image_detail(pid: int):
    photo = Photo.query.get_or_404(pid)
    return render_template("detail.html", p=photo)

@app.route("/uploads/<path:filename>")
def uploaded_file(filename: str):
    # Only serve from UPLOAD_DIR
    return send_from_directory(UPLOAD_DIR, filename)

# Simple JSON API to re-rate an image (debug)
@app.route("/api/rate/<int:pid>", methods=["POST"])
def api_rerate(pid: int):
    photo = Photo.query.get_or_404(pid)
    path = os.path.join(UPLOAD_DIR, photo.filename)
    if not os.path.exists(path):
        abort(404)
    score = aesthetic_score(path)
    photo.score = score
    db.session.commit()
    return jsonify({"id": photo.id, "score": photo.score})

# ------------------- CLI -------------------
@app.cli.command("init-db")
def init_db():
    """Initialize the SQLite database."""
    db.create_all()
    print("[OK] Database initialized.")

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(host="0.0.0.0", port=5000, debug=True)
