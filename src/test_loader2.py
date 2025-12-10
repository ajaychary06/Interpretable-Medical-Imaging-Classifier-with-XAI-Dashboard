# src/test_loader2.py
# Loads src/dataset.py directly (works around import path issues on Windows)

import importlib.util
from pathlib import Path
import sys

project_root = Path(__file__).resolve().parent.parent
dataset_path = project_root / "src" / "dataset.py"

spec = importlib.util.spec_from_file_location("dataset_from_path", str(dataset_path))
dataset_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_mod)

make_loader = dataset_mod.make_loader

def main():
    loader = make_loader(str(project_root / "data" / "processed"), split="train", batch_size=8, image_size=224, num_workers=0)
    print("Num batches:", len(loader))
    for x, y in loader:
        print("batch x", x.shape, "y", y.shape)
        break

if __name__ == "__main__":
    main()
