"""Train MobileNetV3-Small with transfer learning.

    python train_model.py --data dataset/processed                # real training (downloads ImageNet weights)
    python train_model.py --data dataset/processed --epochs 3 --fine-tune-epochs 3

Stage 1 trains only the new classifier head on a frozen backbone; stage 2 unfreezes the top of the
backbone at a low learning rate. Model input is raw 0-255 RGB (scaling is inside the model).
Outputs in models/: cropvision.keras, class_names.json, model_meta.json, and reports/training_history.json.
Test accuracy is NOT computed here - run evaluate_model.py so it is measured once on the held-out set.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import tensorflow as tf

from imaging import MODEL_INPUT_SIZE

SIZE = (MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)


def load_split(root: Path, split: str, batch: int, shuffle: bool, class_names=None):
    return tf.keras.utils.image_dataset_from_directory(
        root / split, image_size=SIZE, batch_size=batch, shuffle=shuffle, seed=42,
        label_mode="int", class_names=class_names)


def build_model(num_classes: int, weights: str | None, dropout: float):
    base = tf.keras.applications.MobileNetV3Small(
        input_shape=SIZE + (3,), include_top=False, weights=weights, include_preprocessing=True)
    base.trainable = False
    inputs = tf.keras.Input(shape=SIZE + (3,), name="image")   # raw 0-255 RGB
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return tf.keras.Model(inputs, outputs, name="cropvision_mobilenetv3s"), base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("dataset/processed"))
    ap.add_argument("--out", type=Path, default=Path("models"))
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--fine-tune-epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--fine-tune-lr", type=float, default=1e-4)
    ap.add_argument("--unfreeze-layers", type=int, default=40)
    ap.add_argument("--weights", default="imagenet", help="'imagenet' (default) or 'none' (smoke tests only)")
    args = ap.parse_args()
    weights = None if args.weights.lower() == "none" else args.weights
    tf.keras.utils.set_random_seed(42)

    for split in ("train", "val"):
        if not (args.data / split).is_dir():
            raise SystemExit(f"Missing {args.data / split}. Run prepare_dataset.py first (see README).")

    train = load_split(args.data, "train", args.batch_size, True)
    class_names = train.class_names
    val = load_split(args.data, "val", args.batch_size, False, class_names)
    print(f"{len(class_names)} classes")

    augment = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal_and_vertical"), tf.keras.layers.RandomRotation(0.15),
        tf.keras.layers.RandomZoom(0.15), tf.keras.layers.RandomContrast(0.15),
        tf.keras.layers.RandomBrightness(0.15, value_range=(0, 255))], name="augment")
    train = train.map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    train, val = train.prefetch(tf.data.AUTOTUNE), val.prefetch(tf.data.AUTOTUNE)

    model, base = build_model(len(class_names), weights, dropout=0.3)
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3, restore_best_weights=True)]
    history = {}

    model.compile(optimizer=tf.keras.optimizers.Adam(args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h = model.fit(train, validation_data=val, epochs=args.epochs, callbacks=callbacks)
    history["head"] = h.history

    if args.fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-args.unfreeze_layers]:
            layer.trainable = False
        for layer in base.layers:                       # keep BatchNorm statistics frozen
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(args.fine_tune_lr),
                      loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        h = model.fit(train, validation_data=val, epochs=args.fine_tune_epochs, callbacks=callbacks)
        history["fine_tune"] = h.history

    args.out.mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    model.save(args.out / "cropvision.keras")
    (args.out / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    (args.out / "model_meta.json").write_text(json.dumps({
        "architecture": "MobileNetV3-Small (transfer learning)", "input_size": MODEL_INPUT_SIZE,
        "weights": weights or "random (smoke test)", "num_classes": len(class_names),
        "trained_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tensorflow": tf.__version__, "data_dir": str(args.data)}, indent=2), encoding="utf-8")
    Path("reports/training_history.json").write_text(
        json.dumps({k: {m: [float(x) for x in v] for m, v in d.items()} for k, d in history.items()}, indent=2))
    print("Saved model to", args.out / "cropvision.keras", "- now run evaluate_model.py")


if __name__ == "__main__":
    main()
