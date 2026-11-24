"""Split a class-per-folder image dataset (e.g. PlantVillage) into train/val/test.

    python prepare_dataset.py --src dataset/raw/color --dst dataset/processed

* Exact duplicate files (same MD5) are dropped so they cannot land in different splits.
* Split is stratified per class (default 70/15/15), seeded and reproducible.
* Every image is resized with imaging.resize_for_model - the SAME function used at inference.
* Writes dataset/processed/split_manifest.json (counts + class list) for your report.

LEAKAGE WARNING: PlantVillage contains several photos of the same physical leaf. Exact-duplicate
removal does not catch those, so test accuracy can be optimistic. Say so in your report, and if
you have leaf/plant IDs, group by them before splitting.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from PIL import Image

from imaging import MODEL_INPUT_SIZE, resize_for_model

EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="folder with one sub-folder per class")
    ap.add_argument("--dst", default=Path("dataset/processed"), type=Path)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-per-class", type=int, default=10)
    args = ap.parse_args()

    classes = sorted(p.name for p in args.src.iterdir() if p.is_dir())
    if not classes:
        raise SystemExit(f"No class folders found in {args.src}")
    rng = random.Random(args.seed)
    seen, manifest = set(), {"classes": classes, "seed": args.seed, "counts": {}, "duplicates_removed": 0}

    for cls in classes:
        files = sorted(f for f in (args.src / cls).rglob("*") if f.suffix.lower() in EXTS)
        unique = []
        for f in files:
            h = md5(f)
            if h in seen:
                manifest["duplicates_removed"] += 1
                continue
            seen.add(h)
            unique.append(f)
        if len(unique) < args.min_per_class:
            print(f"skipping {cls}: only {len(unique)} images")
            continue
        rng.shuffle(unique)
        n_test, n_val = max(1, int(len(unique) * args.test)), max(1, int(len(unique) * args.val))
        parts = {"test": unique[:n_test], "val": unique[n_test:n_test + n_val], "train": unique[n_test + n_val:]}
        for split, items in parts.items():
            out = args.dst / split / cls
            out.mkdir(parents=True, exist_ok=True)
            for i, f in enumerate(items):
                try:
                    with Image.open(f) as im:
                        resize_for_model(im.convert("RGB"), MODEL_INPUT_SIZE).save(out / f"{i:05d}.jpg", quality=95)
                except OSError as exc:
                    print("unreadable, skipped:", f, exc)
            manifest["counts"].setdefault(split, {})[cls] = len(items)
        print(f"{cls}: " + ", ".join(f"{k}={len(v)}" for k, v in parts.items()))

    (args.dst / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Done. Manifest written to", args.dst / "split_manifest.json")


if __name__ == "__main__":
    main()
