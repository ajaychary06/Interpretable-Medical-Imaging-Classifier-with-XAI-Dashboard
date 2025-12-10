#!/usr/bin/env python3
"""
train_advanced.py

Train a transfer-learning classifier with improved performance features:
- Pretrained ResNet (default) or EfficientNet-b0 (optional)
- Mixed precision (torch.cuda.amp)
- LR scheduler (OneCycleLR default, can use CosineAnnealingLR)
- Useful metrics: Accuracy, Precision, Recall, F1, AUC
- Checkpointing (best val F1), periodic saving
- Gradient clipping, weight decay, label smoothing option
- Basic image augmentations with torchvision
- Class-weighting to mitigate class imbalance
- Command-line hyperparameter config
"""
import os
import argparse
import random
from pathlib import Path
from collections import defaultdict, Counter
import time
import copy

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import label_binarize

# ============ Utilities ============
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # deterministic flags (may slow training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def save_checkpoint(state, checkpoint_dir, name="checkpoint.pth.tar"):
    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.save(state, os.path.join(checkpoint_dir, name))

def get_num_classes_from_loader(loader):
    """
    Robustly infer number of classes from a DataLoader's dataset.
    Handles ImageFolder, Subset(ImageFolder), etc.
    """
    try:
        ds = loader.dataset
        # Subset -> .dataset
        if hasattr(ds, "classes"):
            return len(ds.classes)
        if hasattr(ds, "dataset") and hasattr(ds.dataset, "classes"):
            return len(ds.dataset.classes)
    except Exception:
        pass
    return None

# ============ Model builders ============
def build_model(model_name="resnet50", num_classes=2, pretrained=True, dropout=0.5):
    model_name = model_name.lower()
    if model_name.startswith("resnet"):
        # torchvision >=0.13 warns about 'pretrained' param — it's fine to use for compatibility
        model = getattr(models, model_name)(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    elif model_name in ("efficientnet_b0",):
        model = getattr(models, model_name)(pretrained=pretrained)
        # handle classifier attr differences across torchvision versions
        try:
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes)
            )
        except Exception:
            # fallback
            try:
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes)
            except Exception:
                raise ValueError("Unable to set efficientnet classifier — unexpected model API.")
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return model

# ============ Dataloaders ============
def make_dataloaders(data_dir, img_size=224, batch_size=32, num_workers=4, augment=True, val_split=0.2, seed=42):
    # Expect data_dir structured as: data_dir/train/<class>/*.jpg and data_dir/val/<class>/*.jpg
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_transforms = [transforms.Resize((img_size, img_size))]
    if augment:
        train_transforms += [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.05,0.05))
        ]
    train_transforms += [transforms.ToTensor(), normalize]
    train_transform = transforms.Compose(train_transforms)

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])

    train_dir = Path(data_dir) / "train"
    val_dir = Path(data_dir) / "val"

    if train_dir.exists() and val_dir.exists():
        train_ds = datasets.ImageFolder(str(train_dir), transform=train_transform)
        val_ds = datasets.ImageFolder(str(val_dir), transform=val_transform)
    else:
        dataset = datasets.ImageFolder(str(data_dir), transform=train_transform)
        n = len(dataset)
        indices = list(range(n))
        random.Random(seed).shuffle(indices)
        split = int(np.floor(val_split * n))
        train_idx, val_idx = indices[split:], indices[:split]
        from torch.utils.data import Subset
        train_ds = Subset(dataset, train_idx)
        val_ds = Subset(copy.deepcopy(dataset), val_idx)
        # ensure val uses val_transform
        val_ds.dataset.transform = val_transform

    pin_mem = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)

    return train_loader, val_loader

# ============ Training / Validation Steps ============
def compute_metrics(y_true, y_pred, y_probs=None, average="binary"):
    metrics = {}
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    metrics["precision"] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics["recall"] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics["f1"] = f1_score(y_true, y_pred, average=average, zero_division=0)
    if y_probs is not None:
        try:
            if y_probs.ndim == 1 or (y_probs.ndim == 2 and y_probs.shape[1] == 1):
                metrics["auc"] = roc_auc_score(y_true, y_probs)
            else:
                y_true_bin = label_binarize(y_true, classes=list(range(y_probs.shape[1])))
                metrics["auc"] = roc_auc_score(y_true_bin, y_probs)
        except Exception:
            metrics["auc"] = float("nan")
    else:
        metrics["auc"] = float("nan")
    return metrics

def train_one_epoch(model, loader, criterion, optimizer, device, scaler=None, grad_clip=None, scheduler=None):
    model.train()
    losses = []
    all_preds = []
    all_targets = []
    all_probs = []

    pbar = tqdm(loader, desc="Train", leave=False)
    for imgs, targets in pbar:
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(imgs)
            loss = criterion(outputs, targets)

        if scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # Step scheduler per-batch if using OneCycleLR (user may choose onecycle)
        if scheduler is not None and isinstance(scheduler, torch.optim.lr_scheduler.OneCycleLR):
            try:
                scheduler.step()
            except Exception:
                pass

        losses.append(loss.item())

        probs = torch.softmax(outputs.detach().cpu(), dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        all_preds.extend(preds.tolist())
        all_targets.extend(targets.detach().cpu().numpy().tolist())
        if probs.shape[1] == 1:
            all_probs.extend(probs[:, 0].tolist())
        else:
            all_probs.extend(probs.tolist())

        pbar.set_postfix(loss=np.mean(losses))

    metrics = compute_metrics(np.array(all_targets), np.array(all_preds), np.array(all_probs),
                              average="binary" if len(np.unique(all_targets))==2 else "macro")
    metrics["loss"] = float(np.mean(losses))
    return metrics

def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Valid", leave=False)
        for imgs, targets in pbar:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            probs = torch.softmax(outputs.detach().cpu(), dim=1).numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds.tolist())
            all_targets.extend(targets.detach().cpu().numpy().tolist())
            if probs.shape[1] == 1:
                all_probs.extend(probs[:, 0].tolist())
            else:
                all_probs.extend(probs.tolist())
            pbar.set_postfix(loss=np.mean(losses))

    metrics = compute_metrics(np.array(all_targets), np.array(all_preds), np.array(all_probs),
                              average="binary" if len(np.unique(all_targets))==2 else "macro")
    metrics["loss"] = float(np.mean(losses))
    return metrics

# ============ Main ============
def parse_args():
    parser = argparse.ArgumentParser(description="Train classifier with transfer learning")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to data directory (ImageFolder style). If contains train/val subfolders, they will be used.")
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet18","resnet34","resnet50","efficientnet_b0"], help="Model backbone")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--scheduler", type=str, default="onecycle", choices=["onecycle","cosine","none"])
    parser.add_argument("--max_lr", type=float, default=None, help="Max LR for OneCycle; defaults to lr*10")
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--save_every", type=int, default=5, help="Save checkpoint every N epochs")
    parser.add_argument("--num_classes", type=int, default=None, help="Override number of classes (inferred from dataset if omitted)")
    return parser.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, val_loader = make_dataloaders(args.data_dir, img_size=args.img_size,
                                                batch_size=args.batch_size, num_workers=args.num_workers,
                                                augment=True, seed=args.seed)

    # infer num_classes if not provided
    if args.num_classes is None:
        try:
            num_classes = get_num_classes_from_loader(train_loader)
        except Exception:
            num_classes = None
        if num_classes is None:
            num_classes = 2
            print("Warning: num_classes inferred failed. Defaulting to 2.")
    else:
        num_classes = args.num_classes

    print(f"Num classes: {num_classes}")

    # ----------------- build model -----------------
    model = build_model(args.model, num_classes=num_classes, pretrained=args.pretrained, dropout=args.dropout)
    model = model.to(device)

    # ----------------- class weights (imbalance fix) -----------------
    # compute class weights using training set
    all_targets = []
    for _, targets in train_loader:
        all_targets.extend(targets.tolist())

    counts = Counter(all_targets)
    num_classes_in_counts = len(counts)
    total = float(sum(counts.values()))
    # make sure class order 0..num_classes-1
    class_weights = [ total / (counts.get(i, 1) ) for i in range(num_classes) ]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print("Class weights:", class_weights)

    # ----------------- criterion (with weights + optional label smoothing) -----------------
    if args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ----------------- optimizer & scheduler -----------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.scheduler == "onecycle":
        max_lr = args.max_lr or args.lr * 10
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,
                                                        steps_per_epoch=steps_per_epoch,
                                                        epochs=args.epochs,
                                                        pct_start=0.1,
                                                        anneal_strategy="cos")
    elif args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = None

    scaler = torch.cuda.amp.GradScaler() if (args.mixed_precision and device.type == "cuda") else None

    start_epoch = 0
    best_val_f1 = -1.0
    history = defaultdict(list)

    # optionally resume
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "scheduler_state" in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_f1 = checkpoint.get("best_val_f1", best_val_f1)
        print(f"Resumed from {args.resume}, starting at epoch {start_epoch}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    # training loop
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        t0 = time.time()

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler=scaler, grad_clip=args.grad_clip, scheduler=scheduler if args.scheduler=="onecycle" else None)
        val_metrics = validate(model, val_loader, criterion, device)

        # scheduler step for cosine (per-epoch)
        if args.scheduler == "cosine" and scheduler is not None:
            scheduler.step()

        # log
        print("Train Loss: {:.4f} | Val Loss: {:.4f}".format(train_metrics["loss"], val_metrics["loss"]))
        print("Train F1: {:.4f} | Val F1: {:.4f}".format(train_metrics["f1"], val_metrics["f1"]))
        print("Val Acc: {:.4f} | Val Prec: {:.4f} | Val Rec: {:.4f} | Val AUC: {:.4f}".format(
            val_metrics["accuracy"], val_metrics["precision"], val_metrics["recall"], val_metrics.get("auc", float("nan"))))

        # append history
        for k,v in train_metrics.items():
            history[f"train_{k}"].append(v)
        for k,v in val_metrics.items():
            history[f"val_{k}"].append(v)

        # checkpointing
        is_best = val_metrics["f1"] > best_val_f1
        if is_best:
            best_val_f1 = val_metrics["f1"]
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "best_val_f1": best_val_f1,
                "history": dict(history)
            }, args.checkpoint_dir, name="best_model.pth.tar")
            print("Saved new best model (best_val_f1={:.4f})".format(best_val_f1))

        # periodic save
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "best_val_f1": best_val_f1,
                "history": dict(history)
            }, args.checkpoint_dir, name=f"checkpoint_epoch_{epoch+1}.pth.tar")
            print(f"Saved checkpoint for epoch {epoch+1}")

        t1 = time.time()
        print(f"Epoch duration: {(t1-t0):.1f}s")

    # final save + report
    save_checkpoint({
        "epoch": args.epochs-1,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_val_f1": best_val_f1,
        "history": dict(history)
    }, args.checkpoint_dir, name="final_model.pth.tar")
    print("Training complete. Best val F1: {:.4f}".format(best_val_f1))

if __name__ == "__main__":
    main()
