"""
prepare_data.py

Reads images out of data/raw/ (where the class name is baked into each
filename, e.g. "crazing_1.jpg"), and reorganizes them into:

    data/processed/train/<class_name>/*.jpg
    data/processed/val/<class_name>/*.jpg

This folder-per-class layout is the format torchvision's ImageFolder
dataset expects, so once this runs, loading the data in PyTorch is trivial.
"""
import random
import re
import shutil
from pathlib import Path

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
VAL_FRACTION = 0.2   # 20% of each class held out for validation
SEED = 42            # fixes the random split so it's reproducible


def get_class_name(filename: str) -> str:
    """'crazing_104.jpg' -> 'crazing'. Strips the trailing _<number>.<ext>."""
    return re.sub(r"_\d+\.\w+$", "", filename)


def main():
    random.seed(SEED)

    images_by_class: dict[str, list[Path]] = {}
    for path in RAW_DIR.iterdir():
        if not path.is_file():
            continue
        class_name = get_class_name(path.name)
        images_by_class.setdefault(class_name, []).append(path)

    for class_name, paths in sorted(images_by_class.items()):
        random.shuffle(paths)
        n_val = round(len(paths) * VAL_FRACTION)
        val_paths = paths[:n_val]
        train_paths = paths[n_val:]

        for split_name, split_paths in [("train", train_paths), ("val", val_paths)]:
            dest_dir = OUT_DIR / split_name / class_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            for src_path in split_paths:
                shutil.copy2(src_path, dest_dir / src_path.name)

        print(f"{class_name:18s} train={len(train_paths):3d}  val={len(val_paths):3d}")


if __name__ == "__main__":
    main()
