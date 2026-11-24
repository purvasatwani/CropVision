import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import create_app  # noqa: E402


class StubPredictor:
    """TEST DOUBLE ONLY. Lets the web flow be tested without a trained model; never used by the app itself."""
    available = True

    def status(self):
        return {"available": True, "error": None, "backend": "stub", "model_file": "stub", "num_classes": 2}

    def predict(self, img, top_k=3):
        top = [{"label": "Tomato___Late_blight", "confidence": 0.9}, {"label": "Tomato___healthy", "confidence": 0.1}]
        return {"label": top[0]["label"], "confidence": 0.9, "top": top, "backend": "stub",
                "inference_ms": 1.0, "low_confidence": False}


def make_app(tmp_path, predictor=None):
    cfg = {"TESTING": True, "DATABASE": str(tmp_path / "t.db"), "UPLOAD_DIR": tmp_path / "up",
           "THUMB_DIR": tmp_path / "up" / "thumbs", "MODELS_DIR": tmp_path / "no_models", "MODEL_PATH": None}
    if predictor:
        cfg["PREDICTOR"] = predictor
    return create_app(cfg)


@pytest.fixture
def client_no_model(tmp_path):
    return make_app(tmp_path).test_client()


@pytest.fixture
def client_stub(tmp_path):
    return make_app(tmp_path, StubPredictor()).test_client()


def png_bytes(size=(300, 200), color=(40, 160, 80)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "PNG")
    buf.seek(0)
    return buf
