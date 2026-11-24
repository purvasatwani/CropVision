"""Convert the Keras model to TensorFlow Lite variants and compare them.

    python convert_model.py                 # needs models/cropvision.keras + dataset/processed
    python convert_model.py --deploy dynamic

Variants: float32, dynamic-range int8 weights, float16, and full-integer int8 (calibrated on a
sample of TRAINING images). For each variant it records file size, mean CPU latency on THIS
machine, and test accuracy on the held-out split. Results: reports/edge_comparison.json and .md.
Latency measured on a PC says nothing about a Raspberry Pi/Jetson - re-run on the target device
(copy the .tflite files and run  python convert_model.py --benchmark-only ).
"""
import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image

from evaluate_model import evaluate
from imaging import MODEL_INPUT_SIZE
from predict import KerasRunner, TFLiteRunner


def representative_data(train_dir: Path, n: int):
    files = sorted(train_dir.rglob("*.jpg"))
    rng = np.random.default_rng(0)
    rng.shuffle(files)
    def gen():
        for f in files[:n]:
            a = np.asarray(Image.open(f).convert("RGB").resize((MODEL_INPUT_SIZE,) * 2), dtype=np.float32)
            yield [a[None]]
    return gen


def convert(saved_model_dir: str, kind: str, rep=None) -> bytes:
    # Keras 3 (TF >= 2.16): convert from an exported SavedModel; from_keras_model() is unreliable.
    conv = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    if kind in ("dynamic", "fp16", "int8"):
        conv.optimizations = [tf.lite.Optimize.DEFAULT]
    if kind == "fp16":
        conv.target_spec.supported_types = [tf.float16]
    if kind == "int8":
        conv.representative_dataset = rep          # float32 input/output kept for easy use
    return conv.convert()


def latency_ms(runner, runs=50, warmup=5):
    x = np.random.default_rng(0).uniform(0, 255, (1, MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, 3)).astype(np.float32)
    for _ in range(warmup):
        runner(x)
    t = []
    for _ in range(runs):
        s = time.perf_counter(); runner(x); t.append((time.perf_counter() - s) * 1000)
    return {"mean": float(np.mean(t)), "p95": float(np.percentile(t, 95))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras", type=Path, default=Path("models/cropvision.keras"))
    ap.add_argument("--data", type=Path, default=Path("dataset/processed"))
    ap.add_argument("--out", type=Path, default=Path("models/tflite"))
    ap.add_argument("--calibration-images", type=int, default=200)
    ap.add_argument("--deploy", choices=["float32", "dynamic", "fp16", "int8"], default="dynamic",
                    help="variant to copy to models/cropvision.tflite for the web app")
    ap.add_argument("--benchmark-only", action="store_true", help="skip conversion; benchmark existing files")
    ap.add_argument("--limit", type=int, default=None, help="limit test images (quick check)")
    args = ap.parse_args()
    classes = json.loads(Path("models/class_names.json").read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    names = {"float32": "cropvision_float32.tflite", "dynamic": "cropvision_dynamic.tflite",
             "fp16": "cropvision_fp16.tflite", "int8": "cropvision_int8.tflite"}

    if not args.benchmark_only:
        model = tf.keras.models.load_model(args.keras, compile=False)
        rep = representative_data(args.data / "train", args.calibration_images)
        with tempfile.TemporaryDirectory() as tmp:
            model.export(tmp, format="tf_saved_model")
            for kind, fname in names.items():
                (args.out / fname).write_bytes(convert(tmp, kind, rep))
                print("wrote", fname)
        shutil.copy(args.out / names[args.deploy], Path("models/cropvision.tflite"))

    rows = []
    entries = [("keras (original)", args.keras)] if args.keras.exists() else []
    entries += [(k, args.out / f) for k, f in names.items() if (args.out / f).exists()]
    for label, path in entries:
        runner = KerasRunner(str(path)) if path.suffix == ".keras" else TFLiteRunner(str(path))
        lat = latency_ms(runner)
        acc = None
        if (args.data / "test").is_dir():
            y, p = evaluate(path, args.data, classes, args.limit)
            acc = float((y == p).mean()) if len(y) else None
        rows.append({"variant": label, "size_mb": round(path.stat().st_size / 1e6, 3),
                     "latency_ms_mean": round(lat["mean"], 2), "latency_ms_p95": round(lat["p95"], 2),
                     "test_accuracy": acc})
        print(rows[-1])
    meta = {"device_note": "Measured on the machine that ran this script (CPU, single image, batch=1).",
            "tensorflow": tf.__version__, "results": rows}
    Path("reports").mkdir(exist_ok=True)
    Path("reports/edge_comparison.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    md = ["| Variant | Size (MB) | Mean latency (ms) | p95 (ms) | Test accuracy |", "|---|---|---|---|---|"]
    md += [f"| {r['variant']} | {r['size_mb']} | {r['latency_ms_mean']} | {r['latency_ms_p95']} | "
           f"{'n/a' if r['test_accuracy'] is None else format(r['test_accuracy'], '.4f')} |" for r in rows]
    Path("reports/edge_comparison.md").write_text("\n".join(md) + "\n\n" + meta["device_note"] + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
