"""Runs only when a real trained model exists in models/ (skipped otherwise)."""
from pathlib import Path

import pytest
from PIL import Image

from predict import Predictor

ROOT = Path(__file__).resolve().parents[1]
pred = Predictor(ROOT / "models")


@pytest.mark.skipif(not pred.available, reason="no trained model in models/")
def test_real_model_returns_valid_distribution():
    res = pred.predict(Image.new("RGB", (300, 300), (30, 140, 60)))
    assert 0 <= res["confidence"] <= 1 and res["label"] in pred.class_names
    assert abs(sum(t["confidence"] for t in res["top"]) - 1) <= 1 and res["backend"] in {"tflite", "keras"}


@pytest.mark.skipif(not pred.available, reason="no trained model in models/")
def test_test_split_images_are_classified():
    test_dir = ROOT / "dataset" / "processed" / "test"
    imgs = list(test_dir.rglob("*.jpg"))[:20] if test_dir.exists() else []
    if not imgs:
        pytest.skip("no test split prepared")
    for p in imgs:
        assert pred.predict(Image.open(p))["label"] in pred.class_names
