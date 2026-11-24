"""Central configuration. Secrets come from environment variables / .env, never from source."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


class Config:
    SECRET_KEY = os.environ.get("CROPVISION_SECRET_KEY") or os.urandom(24).hex()
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024  # 8 MB upload cap
    DATABASE = str(BASE_DIR / "database" / "cropvision.db")
    UPLOAD_DIR = BASE_DIR / "static" / "uploads"
    THUMB_DIR = BASE_DIR / "static" / "uploads" / "thumbs"
    MODELS_DIR = BASE_DIR / "models"
    MODEL_PATH = os.environ.get("CROPVISION_MODEL_PATH") or None
    LOW_CONFIDENCE = float(os.environ.get("CROPVISION_LOW_CONFIDENCE", "0.60"))
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
    MAX_STORED_SIDE = 1280   # stored copy is downscaled to this longest side
    THUMB_SIDE = 320
