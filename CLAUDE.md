# CLAUDE.md — Project Reference for Surface Defect Classifier

## Quick Start

**Environment**: Windows + Python 3.13 + virtual env in `./venv/`

**Activate venv**: `./venv/Scripts/activate` (PowerShell) or `source venv/bin/activate` (bash)

**Run everything with PYTHONPATH**:
```powershell
$env:PYTHONPATH="."; python -m src.prepare_data
$env:PYTHONPATH="."; python -m src.train
$env:PYTHONPATH="."; python -m src.evaluate
$env:PYTHONPATH="."; python -m src.predict data/processed/val/crazing/crazing_10.jpg
```

## Modules (in dependency order)

| File | Purpose | Key function |
|------|---------|--------------|
| `src/prepare_data.py` | Organize raw JPEGs into `data/processed/train|val/<class>/` (ImageFolder format) | `main()` |
| `src/model.py` | Build ResNet18 with pretrained ImageNet weights, replace final FC layer for 6 classes | `build_model(num_classes)` |
| `src/dataset.py` | DataLoaders with train augmentations (flip, rotate) + val normalization | `get_dataloaders()` |
| `src/train.py` | 15-epoch training loop, Adam lr=1e-4, saves best `models/best_model.pth` by val_acc | `main()` |
| `src/evaluate.py` | Load best checkpoint, print classification_report + confusion matrix | `main()` |
| `src/predict.py` | Single-image CLI inference: `python -m src.predict <path>` | `predict_image()` |
| `app.py` | Gradio web UI for drag-and-drop inference | `python app.py` |

## Class Order (alphabetical, matches ImageFolder)
1. `crazing`
2. `inclusion`
3. `patches`
4. `pitted_surface`
5. `rolled-in_scale`
6. `scratches`

## Key Constants (hardcoded, editable in source)
- `EPOCHS = 15` (src/train.py)
- `BATCH_SIZE = 32` (src/dataset.py)
- `IMG_SIZE = 224` (src/dataset.py)
- `LR = 1e-4` (src/train.py)
- `VAL_FRACTION = 0.2`, `SEED = 42` (src/prepare_data.py)

## Expected Outputs
- Training: reaches ~1.0000 val_acc by epoch 5-10
- Evaluation: perfect classification report (all 1.0000)
- Inference: single-image prediction with confidence

## Limitations
- No automated tests
- No seed set for training (non-reproducible across runs)
- No CLI args / config file — all params hardcoded
- No error handling in predict.py (crashes on bad files)
- ImageNet mean/std duplicated in dataset.py and predict.py
- Models folder only tracks best_model.pth (gitignored otherwise)