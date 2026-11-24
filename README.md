# CropVision: Edge AI Plant Disease Detection

Flask web app + MobileNetV3-Small classifier (TensorFlow/Keras) with TensorFlow Lite export for edge devices.
Upload or capture a leaf photo, get a prediction with the model's real confidence, and keep a scan history in SQLite.
The UI works fully offline (Bootstrap, icons and fonts are bundled in `static/vendor`).

> **No trained model is included.** Until you train one, the app runs but disables scanning and says
> "Model inference is not configured". Accuracy numbers must come from `evaluate_model.py` on the held-out test split.

## 1. Architecture

```
Browser (Bootstrap UI) --upload--> Flask routes (app.py)
                                      |-- imaging.py   validate, EXIF fix, centre-crop, resize 224x224 (shared with training)
                                      |-- predict.py   TFLite (preferred) or Keras backend -> class + softmax confidence
                                      |-- disease_info.py  reference text per class
                                      `-- db.py        SQLite scan history (database/cropvision.db)
Offline pipeline: prepare_dataset.py -> train_model.py -> evaluate_model.py -> convert_model.py
```

## 2. Install (Windows, VS Code)

Use **Python 3.10-3.12 (64-bit)**; TensorFlow 2.16 does not support 3.13. In the VS Code terminal (PowerShell):

```powershell
cd CropVision
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env
```
If activation is blocked: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, then retry. In VS Code choose the `.venv` interpreter (Ctrl+Shift+P, "Python: Select Interpreter").

## 3. Run the website

```powershell
python app.py        # open http://127.0.0.1:5000
```
Without a model you can browse every page; `/detect` shows a clear "not configured" warning.

## 4. Get and prepare the dataset

1. Download **PlantVillage** (38 classes, about 54k leaf images). Sources: the TensorFlow Datasets catalogue (`plant_village`), the original
   repository `spMohanty/PlantVillage-Dataset` on GitHub, or a Kaggle mirror. Use the **colour** images, with one sub-folder per class
   (e.g. `Tomato___Late_blight`). **Check the licence on the copy you download** (TFDS lists CC0 1.0) and cite Hughes & Salathé (2015) and Mohanty et al. (2016).
2. Extract so the layout is `dataset/raw/<class_name>/*.jpg`.
3. Split (stratified 70/15/15, exact duplicates removed, resized with the same function used at inference):
```powershell
python prepare_dataset.py --src dataset/raw --dst dataset/processed
```
**Leakage caveat:** PlantVillage has several photos of the same leaf, which can land in different splits and inflate test accuracy. State this in your report.

## 5. Train, evaluate, convert

```powershell
python train_model.py --data dataset/processed          # ImageNet weights are downloaded on first run (needs internet once)
python evaluate_model.py                                # test-set accuracy + per-class report -> reports/
python convert_model.py --deploy dynamic                # TFLite float32 / dynamic-int8 / fp16 / full-int8 + comparison table
python evaluate_model.py --model models/cropvision.tflite
```
* CPU-only training on Windows can take hours. Google Colab (free GPU) works well: upload `dataset/processed`, run the same commands, download `models/`.
  Quick trial: add `--epochs 2 --fine-tune-epochs 2`.
* `convert_model.py` writes `models/tflite/*.tflite`, copies the `--deploy` variant to `models/cropvision.tflite` (the app prefers it),
  and produces `reports/edge_comparison.md` (size, latency, test accuracy per variant). Pick the variant from your measured table.
* Latency from your PC is not edge-device latency. Re-measure on the target device (section 7).

## 6. Test everything

```powershell
python -m pytest -q
```
* 18 tests cover pages, upload validation, DB persistence/deletion, preprocessing, and the "model missing" path (they use a clearly marked stub predictor, never in the app).
* `tests/test_model_integration.py` runs only when a real model is in `models/`.
* Manual check: open Home, Disease Detection, Scan History, About; upload a leaf from `dataset/processed/test/...`; confirm the result page, history entry, "View" and "Delete".
* Command-line inference, no server: `python predict.py path\to\leaf.jpg`.

## 7. Raspberry Pi / Jetson deployment (not tested by this project)

1. Copy the project plus `models/cropvision.tflite` and `models/class_names.json` (no dataset needed).
2. On a 64-bit Raspberry Pi OS: `python3 -m venv .venv && source .venv/bin/activate && pip install flask python-dotenv pillow numpy opencv-python-headless ai-edge-litert`
   (if unavailable for your OS/Python, use `tflite-runtime`; `predict.py` tries both before full TensorFlow).
3. `HOST=0.0.0.0 python app.py` and open `http://<pi-ip>:5000` from a phone on the same network. Browsers only allow live camera on `localhost`/HTTPS; over plain HTTP the "Use camera" button falls back to the phone's camera picker.
4. Jetson: same steps; the TFLite CPU path works, and TensorRT/ONNX conversion is an optional extension.
5. Run `python convert_model.py --benchmark-only` on the device to record real latency, then report those numbers as device measurements.

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `No matching distribution for tensorflow` | Use Python 3.10-3.12, 64-bit. |
| Venv will not activate | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`. |
| "Model inference is not configured" | Train first; ensure `models/cropvision.tflite` (or `.keras`) and `models/class_names.json` exist. |
| `Missing dataset/processed/train` | Run `prepare_dataset.py` (section 4). |
| Weights download fails | Needs internet once for ImageNet weights; or use `--weights none` (smoke test only). |
| Upload rejected | JPG/PNG/WebP, under 8 MB, at least 64 px. |
| Camera button does nothing on a LAN IP | Browsers need HTTPS or localhost; use the file picker. |
| Port 5000 busy | `$env:PORT=5001; python app.py` |
| Long TensorFlow/oneDNN log lines | Harmless; set `$env:TF_CPP_MIN_LOG_LEVEL=2`. |

## 9. Feature status

| Feature | Status |
|---|---|
| Five-page responsive UI, offline assets | Implemented and route-tested |
| Upload (drag-drop, picker, camera), validation, safe filenames, re-encoded storage | Implemented and tested |
| Inference via real model, missing-model handling | Implemented; verified with a throwaway model on synthetic data |
| SQLite history (view, delete, empty state) | Implemented and tested |
| Train / evaluate / convert / benchmark scripts | Implemented; pipeline verified end to end on synthetic data |
| Trained PlantVillage model and real accuracy | **Needs you to download data and train** |
| Real size/latency/accuracy comparison | **Needs a trained model** |
| Raspberry Pi / Jetson deployment and measurements | **Needs hardware; not tested** |
| ONNX export | Not implemented (TFLite chosen) |
| Disease text | Written for the project; **verify each entry against sources you cite** |

## Safety

Predictions are preliminary and not a substitute for expert diagnosis. The model always picks one of its known classes, even for non-leaf or unfamiliar images; low-confidence results are flagged.
