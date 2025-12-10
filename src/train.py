# src/train.py
"""
Run from project root:
    conda activate xai_proj
    python src\train.py

This script will:
- Load dataset module from src/dataset.py (safe import-by-path)
- Load model builder from src/model.py (safe import-by-path)
- Train for `NUM_EPOCHS` (default 1) on CPU (or GPU if available)
- Save best model to checkpoints/best_model.pth
"""

import importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
import os
from datetime import datetime

# ----------------------------
# Config (edit values if needed)
# ----------------------------
NUM_CLASSES = 2
BATCH_SIZE = 8          # safe for CPU; increase if you have more RAM
IMAGE_SIZE = 224
NUM_EPOCHS = 1          # smoke-run: 1 epoch
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0         # windows: keep 0; increase on Linux

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(exist_ok=True, parents=True)

# ----------------------------
# Determinism (optional)
# ----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------------------
# Helper: import module by path
# ----------------------------
def import_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# load dataset maker
dataset_mod = import_from_path(SRC_DIR / "dataset.py", "dataset_by_path")
make_loader = dataset_mod.make_loader

# load model builder
model_mod = import_from_path(SRC_DIR / "model.py", "model_by_path")
get_resnet18 = model_mod.get_resnet18

# ----------------------------
# Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ----------------------------
# Data loaders
# ----------------------------
train_loader = make_loader(str(PROJECT_ROOT / "data" / "processed"), split="train", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, num_workers=NUM_WORKERS)
val_loader = make_loader(str(PROJECT_ROOT / "data" / "processed"), split="val", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, num_workers=NUM_WORKERS)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ----------------------------
# Model, optimizer, scheduler
# ----------------------------
model = get_resnet18(num_classes=NUM_CLASSES, pretrained=False)
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# ----------------------------
# Training / validation loops
# ----------------------------
def train_one_epoch(model, loader, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 20 == 0:
            print(f"  batch {batch_idx+1}/{len(loader)}  loss={running_loss/(batch_idx+1):.4f}")

    avg_loss = running_loss / len(loader)
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc

def eval_one_epoch(model, loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = F.cross_entropy(outputs, labels)
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    avg_loss = running_loss / len(loader)
    acc = correct / total if total > 0 else 0.0
    return avg_loss, acc

# ----------------------------
# Run training (simple)
# ----------------------------
best_val_acc = 0.0
start_time = datetime.now()
for epoch in range(1, NUM_EPOCHS + 1):
    print(f"\nEpoch {epoch}/{NUM_EPOCHS} --------------------")
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
    print(f"Train  loss={train_loss:.4f}  acc={train_acc:.4f}")
    val_loss, val_acc = eval_one_epoch(model, val_loader, device)
    print(f"Val    loss={val_loss:.4f}  acc={val_acc:.4f}")

    # save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        ckpt_path = CHECKPOINTS_DIR / "best_model.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_acc": val_acc
        }, ckpt_path)
        print(f"Saved best model to {ckpt_path}")

total_time = datetime.now() - start_time
print(f"\nDone. Best val acc={best_val_acc:.4f}. Time elapsed: {total_time}")
