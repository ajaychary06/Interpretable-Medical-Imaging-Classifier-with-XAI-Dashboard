# src/split_data.py
"""
Run from project root (xai_brain_tumor).
Usage:
    conda activate xai_proj
    python src\split_data.py
This script expects images under data\data_raw (one subfolder per class).
It creates data\processed\train|val|test\<class>\image.jpg
"""
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

ROOT = Path.cwd()
RAW_DIR = ROOT / "data" / "data_raw"
PROC_DIR = ROOT / "data" / "processed"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1  # remaining

RANDOM_STATE = 42

def gather_classes(raw_dir):
    classes = [p.name for p in raw_dir.iterdir() if p.is_dir()]
    classes = sorted(classes)
    if not classes:
        raise SystemExit(f"No class subfolders found in {raw_dir}. Check dataset layout.")
    # Remove accidental folders (like brain_tumor_dataset)
    if "brain_tumor_dataset" in classes:
        classes.remove("brain_tumor_dataset")
    return classes

def make_dirs(proc_dir, classes):
    for split in ("train","val","test"):
        for c in classes:
            p = proc_dir / split / c
            p.mkdir(parents=True, exist_ok=True)

def copy_files_for_class(class_name):
    src_dir = RAW_DIR / class_name
    files = [p for p in src_dir.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png")]
    if not files:
        print(f"[WARN] No images found for class {class_name} at {src_dir}")
        return
    # first split off test
    trainval, test = train_test_split(files, test_size=TEST_RATIO, random_state=RANDOM_STATE)
    # split trainval into train and val proportionally
    val_size = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    train, val = train_test_split(trainval, test_size=val_size, random_state=RANDOM_STATE)

    for p in train:
        shutil.copy2(p, PROC_DIR / "train" / class_name / p.name)
    for p in val:
        shutil.copy2(p, PROC_DIR / "val" / class_name / p.name)
    for p in test:
        shutil.copy2(p, PROC_DIR / "test" / class_name / p.name)

def main():
    print("Root:", ROOT)
    print("Raw dir:", RAW_DIR)
    print("Processed dir:", PROC_DIR)
    classes = gather_classes(RAW_DIR)
    print("Found classes:", classes)
    make_dirs(PROC_DIR, classes)
    for c in classes:
        print("Processing class:", c)
        copy_files_for_class(c)
    print("Done. Processed data in:", PROC_DIR)

if __name__ == "__main__":
    main()
