"""Evaluate a model on the held-out TEST split (never used for training or model selection).

    python evaluate_model.py                       # models/cropvision.keras
    python evaluate_model.py --model models/cropvision.tflite

Writes reports/evaluation_<model>.json (accuracy, per-class precision/recall/F1) and a confusion
matrix PNG. These are the only numbers you should quote as accuracy in your report.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from imaging import MODEL_INPUT_SIZE
from predict import KerasRunner, TFLiteRunner, _as_probabilities


def iter_test(root: Path, class_names):
    for idx, cls in enumerate(class_names):
        for f in sorted((root / cls).glob("*")):
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                yield f, idx


def evaluate(model_path: Path, data: Path, class_names: list[str], limit: int | None = None):
    runner = TFLiteRunner(str(model_path)) if model_path.suffix == ".tflite" else KerasRunner(str(model_path))
    y_true, y_pred = [], []
    for n, (f, idx) in enumerate(iter_test(data / "test", class_names)):
        if limit and n >= limit:
            break
        arr = np.asarray(Image.open(f).convert("RGB").resize((MODEL_INPUT_SIZE,) * 2), dtype=np.float32)[None]
        y_true.append(idx)
        y_pred.append(int(np.argmax(_as_probabilities(runner(arr)))))
    return np.array(y_true), np.array(y_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=Path("models/cropvision.keras"))
    ap.add_argument("--data", type=Path, default=Path("dataset/processed"))
    ap.add_argument("--classes", type=Path, default=Path("models/class_names.json"))
    ap.add_argument("--limit", type=int, default=None, help="evaluate only first N images (quick check)")
    args = ap.parse_args()
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    class_names = json.loads(args.classes.read_text(encoding="utf-8"))
    y_true, y_pred = evaluate(args.model, args.data, class_names, args.limit)
    if len(y_true) == 0:
        raise SystemExit(f"No test images found in {args.data / 'test'}")
    acc = float(accuracy_score(y_true, y_pred))
    labels = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   output_dict=True, zero_division=0)
    Path("reports").mkdir(exist_ok=True)
    out = Path("reports") / f"evaluation_{args.model.stem}.json"
    out.write_text(json.dumps({"model": str(args.model), "test_images": int(len(y_true)),
                               "accuracy": acc, "per_class": report}, indent=2), encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 9))
        ax.imshow(cm, cmap="Greens")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion matrix - {args.model.name}")
        fig.tight_layout(); fig.savefig(Path("reports") / f"confusion_{args.model.stem}.png", dpi=120)
    except ImportError:
        pass
    print(f"Test images: {len(y_true)}   Accuracy: {acc:.4f}   (report: {out})")


if __name__ == "__main__":
    main()
