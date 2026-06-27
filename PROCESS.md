# Surface Defect Classifier - Process Documentation

## Project Overview
This project classifies steel surface defects into 6 categories using transfer learning with ResNet18.
Dataset: 1,770 images (NEU Surface Defect Database) across 6 classes.

## Directory Structure
```
surface_defect_classifier/
├── app.py                      # Step 7: Gradio web UI for drag-and-drop inference
├── setup.ps1                   # Windows setup script (venv + deps + run hint)
├── CLAUDE.md                   # Project reference for Claude Code
├── README.md                   # Public-facing project documentation
├── PROCESS.md                  # This file: internal process documentation
├── data/
│   ├── raw/                    # 1,770 original images (flat, class encoded in filename)
│   └── processed/              # ImageFolder format (created by prepare_data.py)
│       ├── train/              # 1,416 images (80%)
│       └── val/                # 354 images (20%)
├── models/
│   ├── best_model.pth          # Best validation checkpoint (created by train.py, tracked by git)
│   └── confusion_matrix.png    # Optional: evaluation artifact (tracked by git if created)
├── src/
│   ├── prepare_data.py         # Step 1: Organize raw data into ImageFolder format
│   ├── model.py                # Step 2: Build ResNet18 with custom classification head
│   ├── dataset.py              # Step 3: Create DataLoaders with transforms
│   ├── train.py                # Step 4: Train model, save best checkpoint
│   ├── evaluate.py             # Step 5: Evaluate best model with detailed metrics
│   └── predict.py              # Step 6: Single-image inference (CLI)
├── requirements.txt            # Python dependencies (includes gradio)
└── .gitignore                  # Excludes venv, __pycache__, data/raw, models/* except best_model.pth/confusion_matrix.png
```

## Class Labels (alphabetical order, matches ImageFolder)
1. crazing
2. inclusion
3. patches
4. pitted_surface
5. rolled-in_scale
6. scratches

## Step-by-Step Process

### STEP 1: Prepare Data (src/prepare_data.py)
**Purpose**: Convert flat raw directory (class_1.jpg, class_2.jpg) into ImageFolder structure expected by torchvision.

**What it does**:
1. Scans `data/raw/` for all `.jpg` files
2. Extracts class name from filename using regex: `crazing_10.jpg` → `crazing`
3. Groups files by class
4. Shuffles with fixed seed (42) for reproducibility
5. Splits each class 80% train / 20% val
6. Copies to `data/processed/train/<class>/` and `data/processed/val/<class>/`
6. Prints per-class counts

**Run**:
```bash
PYTHONPATH=. ./venv/Scripts/python.exe -m src.prepare_data
```

**Output example**:
```
crazing            train=236  val= 59
inclusion          train=237  val= 59
patches            train=236  val= 59
pitted_surface     train=235  val= 59
rolled-in_scale    train=236  val= 59
scratches          train=236  val= 59
```

**Key constants** (editable in file):
- `VAL_FRACTION = 0.2`
- `SEED = 42`

---

### STEP 2: Model Definition (src/model.py)
**Purpose**: Create a transfer-learned ResNet18 with 6-output classification head.

**What it does**:
1. Loads `torchvision.models.resnet18(weights=IMAGENET1K_V1)` — pretrained on ImageNet
2. Gets input features of final FC layer (512 for ResNet18)
3. Replaces `model.fc` with `nn.Linear(512, num_classes)`
4. Returns modified model

**Why ResNet18**:
- Small (~11M params), fast on CPU/single GPU
- Pretrained features (edges, textures) transfer well to defect images
- Deeper models would overfit on 1,416 training images

**Function**:
```python
build_model(num_classes: int) -> nn.Module
```

**Run** (sanity check):
```bash
PYTHONPATH=. ./venv/Scripts/python.exe -m src.model
# Output: Linear(in_features=512, out_features=6, bias=True)
```

---

### STEP 3: DataLoaders (src/dataset.py)
**Purpose**: Provide batched, transformed data for training and validation.

**Transforms**:
- **Train**: Resize(224) → RandomHorizontalFlip → RandomRotation(10°) → ToTensor → Normalize(ImageNet mean/std)
- **Val**: Resize(224) → ToTensor → Normalize(ImageNet mean/std)  (NO augmentation)

**Why separate transforms**:
- Augmentation only during training to reduce overfitting
- Validation must reflect real-world performance on unmodified images

**Function**:
```python
get_dataloaders(data_dir="data/processed", batch_size=32, img_size=224) 
    -> (train_loader, val_loader, class_names)
```

**Run** (sanity check):
```bash
PYTHONPATH=. ./venv/Scripts/python.exe -m src.dataset
# Output:
# Classes: ['crazing', 'inclusion', 'patches', 'pitted_surface', 'rolled-in_scale', 'scratches']
# Train batches: 45
# Batch image shape: torch.Size([32, 3, 224, 224])
```

**ImageNet normalization constants** (used in both dataset.py and predict.py):
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

---

### STEP 4: Training (src/train.py)
**Purpose**: Train the model for 15 epochs, save best checkpoint by validation accuracy.

**Configuration** (module-level constants):
- `EPOCHS = 15`
- Optimizer: `Adam(lr=1e-4)`
- Loss: `CrossEntropyLoss()`
- Device: CUDA if available, else CPU

**Loop per epoch**:
1. `train_one_epoch()`: model.train(), forward, loss.backward(), optimizer.step()
2. `evaluate()`: model.eval(), no_grad, compute val_loss and val_acc
3. Print: `Epoch {n}/15 - train_loss={:.4f} val_loss={:.4f} val_acc={:.4f}`
4. If `val_acc > best_val_acc`: save `model.state_dict()` to `models/best_model.pth`

**Why save best by val_acc, not last epoch**:
Model can overfit in later epochs; best validation checkpoint usually generalizes better.

**Run**:
```bash
PYTHONPATH=. ./venv/Scripts/python.exe -m src.train
```

**Output example**:
```
Using device: cuda
Training on 1416 images, validating on 354 images
Epoch 1/15 - train_loss=0.2751 val_loss=0.0465 val_acc=0.9859
  -> saved new best model (val_acc=0.9859)
Epoch 2/15 - train_loss=0.0343 val_loss=0.0192 val_acc=0.9944
  -> saved new best model (val_acc=0.9944)
...
Epoch 15/15 - train_loss=0.0303 val_loss=0.0193 val_acc=0.9915
```

---

### STEP 5: Evaluation (src/evaluate.py)
**Purpose**: Load best checkpoint and compute detailed metrics on validation set.

**What it does**:
1. Loads `build_model(num_classes=6)` and `models/best_model.pth`
2. Runs `evaluate_model()` on val_loader with `torch.no_grad()`
3. Prints:
   - Per-class precision, recall, f1-score, support (sklearn classification_report)
   - Raw confusion matrix (numpy array)
   - Normalized confusion matrix (per true class)
   - Final loss and accuracy

**Run**:
```bash
PYTHONPATH=. ./venv/Scripts/python.exe -m src.evaluate
```

**Output example**:
```
Classification Report:
                 precision    recall  f1-score   support
        crazing     1.0000    1.0000    1.0000        59
      inclusion     1.0000    1.0000    1.0000        59
        patches     1.0000    1.0000    1.0000        59
 pitted_surface     1.0000    1.0000    1.0000        59
rolled-in_scale     1.0000    1.0000    1.0000        59
      scratches     1.0000    1.0000    1.0000        59
       accuracy                         1.0000       354
      macro avg     1.0000    1.0000    1.0000       354
   weighted avg     1.0000    1.0000    1.0000       354

Confusion Matrix:
[[59  0  0  0  0  0]
 [ 0 59  0  0  0  0]
 [ 0  0 59  0  0  0]
 [ 0  0  0 59  0  0]
 [ 0  0  0  0 59  0]
 [ 0  0  0  0  0 59]]

Final Validation: loss=0.0043 accuracy=1.0000
```

---

### STEP 6: Inference CLI (src/predict.py)
**Purpose**: Predict class of a single image file.

**What it does**:
1. Reads image path from `sys.argv[1]`
2. Hardcodes class_names in ImageFolder alphabetical order
3. Loads model + best checkpoint
4. Applies validation transform (Resize → ToTensor → Normalize)
5. Runs `model(image_tensor)` with `torch.no_grad()`
6. Softmax → argmax → confidence
7. Prints: `Predicted: {class} (confidence: {XX.XX%})`

**Run**:
```bash
PYTHONPATH=. ./venv/Scripts/python.exe -m src.predict <image_path>
```

**Example**:
```bash
PYTHONPATH=. ./venv/Scripts/python.exe -m src.predict data/processed/val/crazing/crazing_10.jpg
# Output: Predicted: crazing (confidence: 100.00%)
```

---

### STEP 7: Web UI (app.py)
**Purpose**: Provide a drag-and-drop web interface for the trained model.

**What it does**:
1. Loads model + best checkpoint at startup (same as predict.py)
2. Defines Gradio `Interface` with:
   - Input: `gr.Image(type="pil")` — upload widget
   - Output: `gr.Label(num_top_classes=6)` — ranked bar list of all 6 classes
   - Title/description for the UI
3. Prediction function applies validation transform, runs model, returns `{class: prob}` dict
4. Launches local server on `http://127.0.0.1:7860` (port may vary)

**Run**:
```bash
PYTHONPATH=. ./venv/Scripts/python.exe app.py
```

**Output**:
```
* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.
```

Open the printed URL in a browser, upload an image, see predictions.

**Note**: This is intentionally separate from `src/predict.py` (the CLI version) — both share the same prediction logic conceptually, but `app.py` exists purely to give a no-terminal, click-based way to use the trained model.

---

## Complete Pipeline Run (Fresh)

```bash
# 1. Prepare data (only needed once, or if raw data changes)
PYTHONPATH=. ./venv/Scripts/python.exe -m src.prepare_data

# 2. Train (creates models/best_model.pth)
PYTHONPATH=. ./venv/Scripts/python.exe -m src.train

# 3. Evaluate (loads best_model.pth)
PYTHONPATH=. ./venv/Scripts/python.exe -m src.evaluate

# 4. Predict on new images (CLI)
PYTHONPATH=. ./venv/Scripts/python.exe -m src.predict data/processed/val/scratches/scratches_1.jpg

# 5. Launch web UI (optional)
PYTHONPATH=. ./venv/Scripts/python.exe app.py
```

---

## Dependencies (requirements.txt)
```
torch==2.11.0+cu128
torchvision==0.26.0+cu128
pillow==12.2.0
numpy==2.5.0
scikit-learn==1.9.0
scipy==1.18.0
matplotlib==3.11.0
gradio
```

Install:
```bash
./venv/Scripts/pip.exe install -r requirements.txt
```

---

## Known Limitations (for maintainers)

1. **No automated tests** — any refactor must be manually verified
2. **Not fully reproducible** — only `prepare_data.py` sets a seed (42); training uses no torch/numpy/python seeds
3. **No error handling**:
   - `predict.py` crashes on corrupt/missing/non-image files (PIL.UnidentifiedImageError, FileNotFoundError)
   - `train.py`/`evaluate.py` crash on missing data dirs or checkpoint
4. **Hardcoded values**:
   - Class names list in `predict.py` must match ImageFolder alphabetical order
   - Data paths (`data/processed`, `models/best_model.pth`) assume fixed structure
   - ImageNet mean/std duplicated in `dataset.py` and `predict.py`
   - Batch size (32), lr (1e-4), epochs (15), img_size (224) all in source
5. **No training enhancements**: no LR scheduler, gradient clipping, early stopping, mixed precision
6. **Single GPU only** — no DDP/multi-GPU support
7. **No experiment tracking** — no MLflow/W&B, no config versioning

---

## File Reference (src/)

| File | Lines | Purpose |
|------|-------|---------|
| `prepare_data.py` | 55 | Raw → ImageFolder split |
| `model.py` | 44 | `build_model(num_classes)` — ResNet18 + new head |
| `dataset.py` | 91 | `get_dataloaders()` — train/val DataLoaders with transforms |
| `train.py` | 107 | Training loop, best checkpoint saving |
| `evaluate.py` | 90 | Detailed validation metrics (report, confusion matrix) |
| `predict.py` | 86 | Single-image inference CLI |

---

## Root-Level File Reference

| File | Purpose |
|------|---------|
| `app.py` | Gradio web UI for drag-and-drop inference |
| `setup.ps1` | Windows one-command setup (venv + deps + run hint) |
| `CLAUDE.md` | Project reference for Claude Code |
| `README.md` | Public-facing documentation |
| `PROCESS.md` | This file: internal process documentation |
| `requirements.txt` | Python dependencies |

---

## Notes for Future Changes

- **Add new classes**: Place images in `data/raw/` as `<class>_<num>.jpg`, re-run `prepare_data.py`, update `class_names` list in `predict.py`, retrain
- **Change model**: Modify `build_model()` in `model.py` (e.g., use `resnet50`, `efficientnet_b0`, or custom architecture)
- **Adjust augmentation**: Edit `train_transform` in `dataset.py`
- **Change hyperparameters**: Edit constants at top of `train.py` (EPOCHS, lr, batch_size via get_dataloaders call)
- **Make configurable**: Add argparse to each script or use a config file (YAML/JSON) + config loader