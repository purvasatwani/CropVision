"""Model loading and inference. No Flask code lives here.

Backends (first match wins):
  1. $CROPVISION_MODEL_PATH
  2. models/cropvision.tflite   (TensorFlow Lite - the edge model)
  3. models/cropvision.keras    (full Keras model)
Requires models/class_names.json next to the model. If anything is missing the Predictor
reports `available == False`; it never fabricates a prediction.

CLI (offline, no web server):   python predict.py path/to/leaf.jpg
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

from imaging import MODEL_INPUT_SIZE, open_image, to_model_array

BASE_DIR = Path(__file__).resolve().parent


class ModelNotConfigured(RuntimeError):
    pass


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def _as_probabilities(out: np.ndarray) -> np.ndarray:
    out = np.asarray(out, dtype=np.float64).reshape(-1)
    if out.min() >= 0 and out.max() <= 1.0 + 1e-6 and abs(out.sum() - 1) < 1e-3:
        return out
    return _softmax(out)


def load_tflite_interpreter(model_path: str):
    """Use the lightest TFLite runtime installed (works on Raspberry Pi without full TensorFlow)."""
    for mod in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter"):
        try:
            interp_mod = __import__(mod, fromlist=["Interpreter"])
            return interp_mod.Interpreter(model_path=model_path)
        except ImportError:
            continue
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ModelNotConfigured(
            "No TensorFlow Lite runtime installed (install tensorflow, ai-edge-litert or tflite-runtime)."
        ) from exc
    return tf.lite.Interpreter(model_path=model_path)


class TFLiteRunner:
    def __init__(self, model_path: str):
        self.interp = load_tflite_interpreter(model_path)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        self.out = self.interp.get_output_details()[0]

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        x = batch
        if self.inp["dtype"] != np.float32:  # quantised input: apply scale / zero point
            scale, zero = self.inp["quantization"]
            info = np.iinfo(self.inp["dtype"])
            x = np.clip(np.round(batch / scale + zero), info.min, info.max).astype(self.inp["dtype"])
        self.interp.set_tensor(self.inp["index"], x)
        self.interp.invoke()
        y = self.interp.get_tensor(self.out["index"])
        if self.out["dtype"] != np.float32:
            scale, zero = self.out["quantization"]
            y = (y.astype(np.float32) - zero) * scale
        return y


class KerasRunner:
    def __init__(self, model_path: str):
        try:
            import tensorflow as tf
        except ImportError as exc:
            raise ModelNotConfigured("TensorFlow is not installed.") from exc
        self.model = tf.keras.models.load_model(model_path, compile=False)

    def __call__(self, batch: np.ndarray) -> np.ndarray:
        return self.model(batch, training=False).numpy()


class Predictor:
    def __init__(self, models_dir: Path | str = BASE_DIR / "models", model_path: str | None = None,
                 low_confidence: float = 0.60):
        self.models_dir = Path(models_dir)
        self.low_confidence = low_confidence
        self.model_path = self._resolve(model_path)
        self.class_names: list[str] = []
        self._runner = None
        self._error: str | None = None
        self.backend = None
        self.meta: dict = {}

    def _resolve(self, explicit):
        if explicit:
            p = Path(explicit)
            return p if p.is_absolute() else BASE_DIR / p
        for name in ("cropvision.tflite", "cropvision.keras"):
            if (self.models_dir / name).exists():
                return self.models_dir / name
        return None

    def _load(self):
        if self._runner is not None or self._error:
            return
        try:
            if not self.model_path or not Path(self.model_path).exists():
                raise ModelNotConfigured("No trained model found in the models/ folder. Run train_model.py first.")
            labels = self.models_dir / "class_names.json"
            if not labels.exists():
                raise ModelNotConfigured("models/class_names.json is missing (it is written by train_model.py).")
            self.class_names = json.loads(labels.read_text(encoding="utf-8"))
            meta = self.models_dir / "model_meta.json"
            self.meta = json.loads(meta.read_text(encoding="utf-8")) if meta.exists() else {}
            if str(self.model_path).endswith(".tflite"):
                self._runner, self.backend = TFLiteRunner(str(self.model_path)), "tflite"
            else:
                self._runner, self.backend = KerasRunner(str(self.model_path)), "keras"
        except ModelNotConfigured as exc:
            self._error = str(exc)
        except Exception as exc:  # corrupt model file etc.
            self._error = f"Model could not be loaded: {exc}"

    @property
    def available(self) -> bool:
        self._load()
        return self._runner is not None

    def status(self) -> dict:
        self._load()
        return {"available": self.available, "error": self._error, "backend": self.backend,
                "model_file": Path(self.model_path).name if self.model_path else None,
                "num_classes": len(self.class_names)}

    def predict(self, img: Image.Image, top_k: int = 3) -> dict:
        self._load()
        if self._runner is None:
            raise ModelNotConfigured(self._error or "Model inference is not configured.")
        batch = to_model_array(img, MODEL_INPUT_SIZE)
        t0 = time.perf_counter()
        probs = _as_probabilities(self._runner(batch))
        ms = (time.perf_counter() - t0) * 1000
        if len(probs) != len(self.class_names):
            raise ModelNotConfigured("Model output size does not match class_names.json.")
        order = np.argsort(probs)[::-1][:top_k]
        top = [{"label": self.class_names[i], "confidence": float(probs[i])} for i in order]
        return {"label": top[0]["label"], "confidence": top[0]["confidence"], "top": top,
                "backend": self.backend, "inference_ms": round(ms, 2),
                "low_confidence": top[0]["confidence"] < self.low_confidence}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python predict.py <image>")
    pred = Predictor()
    with open(sys.argv[1], "rb") as fh:
        result = pred.predict(open_image(fh))
    print(json.dumps(result, indent=2))
