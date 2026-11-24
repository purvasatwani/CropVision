"""Image validation and preprocessing shared by training-data preparation and inference.

The SAME `resize_for_model` function is used by prepare_dataset.py (training data) and by
predict.py (uploads), so train/inference preprocessing cannot drift apart. Pixel scaling is
done INSIDE the model (MobileNetV3 `include_preprocessing=True`), so the model takes raw
0-255 RGB values.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MODEL_INPUT_SIZE = 224
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
MIN_SIDE = 64
Image.MAX_IMAGE_PIXELS = 40_000_000  # decompression-bomb guard


class InvalidImage(ValueError):
    """Raised with a user-readable message when an upload cannot be used."""


def open_image(stream) -> Image.Image:
    """Validate and decode an uploaded image into an RGB PIL image (EXIF rotation applied)."""
    try:
        probe = Image.open(stream)
        probe.verify()  # detects truncated/corrupt files
        stream.seek(0)
        img = Image.open(stream)
        if img.format not in ALLOWED_FORMATS:
            raise InvalidImage("Unsupported image format. Please upload a JPG, PNG or WebP photo.")
        img = ImageOps.exif_transpose(img)
        img.load()
    except InvalidImage:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise InvalidImage("This file could not be read as an image. Try another photo.") from None
    img = img.convert("RGB")
    if min(img.size) < MIN_SIDE:
        raise InvalidImage(f"Image is too small ({img.width}x{img.height}). Use at least {MIN_SIDE}px per side.")
    return img


def resize_for_model(img: Image.Image, size: int = MODEL_INPUT_SIZE) -> Image.Image:
    """Centre-crop to a square, then resize (bilinear) to size x size."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    img = img.crop((left, top, left + s, top + s))
    return img.resize((size, size), Image.BILINEAR)


def to_model_array(img: Image.Image, size: int = MODEL_INPUT_SIZE) -> np.ndarray:
    """Return float32 array shaped (1, size, size, 3) with raw 0-255 values."""
    arr = np.asarray(resize_for_model(img, size), dtype=np.float32)
    return arr[None, ...]


def blur_score(img: Image.Image) -> float | None:
    """Variance of the Laplacian (higher = sharper). None if OpenCV is unavailable."""
    try:
        import cv2
    except ImportError:
        return None
    gray = cv2.cvtColor(np.asarray(img.resize((512, 512))), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def quality_note(img: Image.Image) -> str | None:
    score = blur_score(img)
    if score is not None and score < 30:
        return "The photo looks blurry. For a more reliable result, retake it in good light with the leaf in focus."
    return None
