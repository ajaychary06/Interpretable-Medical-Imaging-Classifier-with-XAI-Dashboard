# scripts/count_classes.py
from pathlib import Path

ROOT = Path("data/processed")

for split in ["train", "val", "test"]:
    p = ROOT / split
    if not p.exists():
        print(f"{split}: missing ({p})")
        continue
    classes = {}
    for d in sorted(p.iterdir()):
        if d.is_dir():
            count = sum(1 for _ in d.glob("*.*"))
            classes[d.name] = count
    print(f"{split}: {classes}")
