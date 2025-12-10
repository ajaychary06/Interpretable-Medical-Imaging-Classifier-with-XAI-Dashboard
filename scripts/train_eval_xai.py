# scripts/train_eval_xai.py
"""
Training + evaluation script with a small XAI demo at the end.
Windows-safe: main training loop is guarded by `if __name__ == "__main__":`
and num_workers defaults to 0 on Windows to avoid multiprocessing spawn issues.
"""

import os
from pathlib import Path
import importlib.util
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.optim as optim
import platform

# --- Dynamically import the local scripts/xai_extended.py module so imports always work ---
THIS_DIR = Path(__file__).resolve().parent
XAI_PATH = THIS_DIR / "xai_extended.py"
if not XAI_PATH.exists():
    raise FileNotFoundError(f"Expected {XAI_PATH} to exist. Make sure xai_extended.py is in the scripts folder.")
spec = importlib.util.spec_from_file_location("scripts.xai_extended", str(XAI_PATH))
xai_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xai_mod)
explain_image_with_models = getattr(xai_mod, "explain_image_with_models")

# --- Paths (adjusted to your repo layout) ---
PROJECT_ROOT = Path.cwd()
DATA_ROOT = PROJECT_ROOT / "data" / "processed"   # uses data/processed as you listed
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
TEST_DIR = DATA_ROOT / "test"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

OUT_XAI_DIR = PROJECT_ROOT / "outputs" / "xai_demo_from_train"
OUT_XAI_DIR.mkdir(parents=True, exist_ok=True)

# --- Device and hyperparams ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 2
batch_size_train = 16
batch_size_val = 32
epochs = 5
lr = 1e-4

# Windows-safe num_workers: use 0 on Windows by default to avoid spawn issues
if platform.system().lower().startswith("win"):
    num_workers = 0
else:
    try:
        cpu_count = os.cpu_count() or 1
    except Exception:
        cpu_count = 1
    num_workers = min(4, max(0, cpu_count - 1))

# --- Transforms ---
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])
transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# --- Helpers for checkpoint save/load ---
def save_checkpoint(model: nn.Module, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict()}, str(path))

def load_checkpoint_into_model(model: nn.Module, path: Path, device=torch.device("cpu")):
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    state = torch.load(str(path), map_location=device)
    if state is None:
        raise RuntimeError(f"torch.load returned None for checkpoint: {path}")
    if isinstance(state, dict) and "state_dict" in state:
        state_dict = state["state_dict"]
    else:
        state_dict = state
    # strip DataParallel prefix if present
    if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
        new_state = {k.replace("module.", ""): v for k, v in state_dict.items()}
        state_dict = new_state
    # load with strict=False to be forgiving about minor mismatch
    try:
        model.load_state_dict(state_dict)
    except Exception:
        model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    return model

def build_dataloaders():
    # --- Sanity checks for data paths ---
    for p in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        if not p.exists():
            raise FileNotFoundError(f"Expected data directory not found: {p}. Please ensure data/processed has train/val/test subfolders.")

    # --- Datasets & loaders ---
    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=transform_train)
    val_ds = datasets.ImageFolder(str(VAL_DIR), transform=transform_val)
    test_ds = datasets.ImageFolder(str(TEST_DIR), transform=transform_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size_val, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size_val, shuffle=False, num_workers=num_workers)

    return train_ds, val_ds, test_ds, train_loader, val_loader, test_loader

def build_model():
    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(device)

def run_training_and_eval():
    # Build everything
    train_ds, val_ds, test_ds, train_loader, val_loader, test_loader = build_dataloaders()
    model = build_model()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- Training loop ---
    best_val_acc = 0.0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)

        # validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        val_acc = correct / len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} loss={epoch_loss:.4f} val_acc={val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = CHECKPOINT_DIR / "best_model.pth"
            save_checkpoint(model, ckpt_path)
            print("Saved checkpoint:", ckpt_path)

    # --- Evaluate on test set (use saved best checkpoint if present) ---
    ckpt_path = CHECKPOINT_DIR / "best_model.pth"
    if ckpt_path.exists():
        model = load_checkpoint_into_model(model, ckpt_path, device=device)
    else:
        print("No best_model.pth found in checkpoints; using current model weights for test evaluation.")

    test_correct = 0
    model.eval()
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            test_correct += (outputs.argmax(dim=1) == labels).sum().item()
    test_acc = test_correct / len(test_loader.dataset)
    print("Test accuracy:", test_acc)

    # --- Run XAI on the first few test images and save overlays ---
    out_dir = OUT_XAI_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    test_samples = [str(p) for p, _ in test_ds.samples][:5]

    for p in test_samples:
        try:
            res = explain_image_with_models(p, model=model, use_gradcam=True, use_ig=True)
        except Exception as e:
            print(f"XAI generation failed for {p}: {e}")
            continue

        stem = Path(p).stem
        if res.get("gradcam_overlay") is not None:
            try:
                res["gradcam_overlay"].save(out_dir / f"{stem}_gradcam.jpg")
            except Exception as e:
                print(f"Failed to save gradcam overlay for {p}: {e}")
        if res.get("ig_overlay") is not None:
            try:
                res["ig_overlay"].save(out_dir / f"{stem}_ig.jpg")
            except Exception as e:
                print(f"Failed to save IG overlay for {p}: {e}")

    print("Saved XAI overlays to", out_dir)

if __name__ == "__main__":
    run_training_and_eval()
