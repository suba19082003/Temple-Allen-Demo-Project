# Surface Defect Classifier

A fine-tuned ResNet18 model that classifies steel surface images into
6 defect types, with a one-click web UI. Built as a machine learning
demo project, applying image-classification techniques (originally
used in a biomedical-imaging context) to an industrial quality-control
problem.

## Quick Start (no training required)

The trained model is already included in this repo — you can run the
demo immediately:
git clone https://github.com/suba19082003/Temple-Allen-Demo-Project.git

cd Temple-Allen-Demo-Project

python -m venv venv
Windows: `.\venv\Scripts\Activate.ps1`
Mac/Linux: `source venv/bin/activate`
pip install -r requirements.txt

python app.py

Open the local URL it prints (usually `http://127.0.0.1:7860`), upload
a steel surface image, and see the predicted defect class with
confidence scores for all 6 categories.

## What this project does

Classifies images into 6 categories from the NEU Surface Defect
Database: crazing, inclusion, patches, pitted surface, rolled-in
scale, scratches.

| Stage | File | What it does |
|---|---|---|
| Data prep | `src/prepare_data.py` | Splits raw images into train/val folders (80/20, stratified per class) |
| Model | `src/model.py` | Defines the model: pretrained ResNet18 with the final layer replaced for 6 classes |
| Data loading | `src/dataset.py` | Builds PyTorch DataLoaders with augmentation (train only) |
| Training | `src/train.py` | Trains for 15 epochs, saves the best checkpoint by validation accuracy |
| Evaluation | `src/evaluate.py` | Prints per-class precision/recall/F1 and a confusion matrix |
| Inference (CLI) | `src/predict.py` | Classifies a single image from the command line |
| Inference (UI) | `app.py` | Same model, as a drag-and-drop web interface (Gradio) |

## Results

On the held-out validation set (354 images, 59 per class):
             precision    recall  f1-score   support
    crazing     1.0000    1.0000    1.0000        59
  inclusion     1.0000    1.0000    1.0000        59
    patches     1.0000    1.0000    1.0000        59
pitted_surface     1.0000    1.0000    1.0000        59

rolled-in_scale     1.0000    1.0000    1.0000        59

scratches     1.0000    1.0000    1.0000        59

accuracy                         1.0000       354

**Caveat worth stating honestly:** this is a small validation set (59
images per class). 100% here means "59/59 correct," not "tested at
large scale." We verified this wasn't due to data leakage (no
duplicate files between train/val, checked by hash) and confirmed it
independently via single-image inference outside the batch evaluation
path. The NEU-CLS dataset is also widely reported as achieving 98-100%
accuracy with simple CNNs in published literature, which is consistent
with what we see here.

## To retrain from scratch (optional)

1. Get the dataset images:
git clone --depth 1 https://github.com/siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection.git temp_dataset_repo
   Copy `temp_dataset_repo/IMAGES/*` into `data/raw/` in this repo, then
   delete `temp_dataset_repo`.
2. `python -m src.prepare_data`
3. `python -m src.train`
4. `python -m src.evaluate` (optional — prints metrics)

## Honest limitations

- No automated tests
- Training is not fully reproducible (only the train/val data split is
  seeded; model initialization and batch order are not)
- No error handling for corrupt/invalid image files in `predict.py`
- No learning rate scheduler, gradient clipping, or early stopping
- Validated at small scale (1,770 images); not tested on a large
  dataset

## Future work

- **Multi-GPU training/inference**: not used here — a single GPU
  comfortably fits this model and dataset size. Would use PyTorch's
  `DistributedDataParallel` if data or model size grew significantly.
- **ROS2 integration**: could wrap `src/predict.py`'s logic in a ROS2
  node subscribing to a camera topic, for use in a real-time robotic
  inspection cell.
- **Isaac Sim**: could generate synthetic training images (varied
  lighting/camera angles) to augment the real dataset, or simulate a
  camera mounted on a robotic arm for inspection planning.

## Dataset credit

Images from the NEU Surface Defect Database (NEU-CLS), sourced via a
GitHub mirror (siddhartamukherjee/NEU-DET-Steel-Surface-Defect-Detection)
since the original Northeastern University hosting page is no longer
available. All modeling code in this repo is original.