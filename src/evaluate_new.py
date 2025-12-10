#!/usr/bin/env python3
"""
evaluate_new.py

Improved evaluation script for ImageFolder-style test sets.
Produces:
 - Confusion matrix PNG
 - Classification report TXT
 - AUC (robust handling) TXT
 - Prob histograms (per-class)
 - predictions CSV with file paths, true labels, predicted labels, probabilities
 - meta.json

Usage examples:
python src/evaluate_new.py --data_dir "data/processed" --checkpoint "checkpoints/best_model.pth.tar" --model resnet18 --batch_size 8 --num_workers 0 --output_dir eval_results
python src/evaluate_new.py --data_dir "data/processed/test" --checkpoint "checkpoints/final_model.pth.tar" --model resnet50 --batch_size 16
"""
import os
import argparse
from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models, datasets
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---------------- model builder ----------------
def build_model(model_name="resnet18", num_classes=2, dropout=0.5):
    """
    Build a model compatible with our training script.
    This does NOT load pretrained weights here — checkpoint loader will set weights.
    """
    model_name = model_name.lower()
    if model_name.startswith("resnet"):
        model = getattr(models, model_name)(weights=None)
        in_features = model.fc.in_features
        model.fc = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(in_features, num_classes)
        )
    elif model_name == "efficientnet_b0":
        # torchvision efficientnet API: classifier can differ by version; handle common cases
        model = models.efficientnet_b0(weights=None)
        try:
            in_features = model.classifier[1].in_features
            model.classifier = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(in_features, num_classes)
            )
        except Exception:
            # fallback
            model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
    return model

# ---------------- dataloader ----------------
def make_loader(data_dir, img_size=224, batch_size=16, num_workers=0):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize
    ])
    ds = datasets.ImageFolder(str(data_dir), transform=transform)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False)
    return loader, ds

# ---------------- plotting helpers ----------------
def plot_confusion_matrix(cm, classes, out_path):
    fig, ax = plt.subplots(figsize=(5,5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)),
           xticklabels=classes, yticklabels=classes,
           ylabel="True label", xlabel="Predicted label", title="Confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2. if cm.max() != 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(int(cm[i, j]), "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

def plot_prob_histograms(all_probs, classes, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    probs = np.array(all_probs)
    if probs.size == 0:
        return
    if probs.ndim == 1:
        plt.figure()
        plt.hist(probs, bins=30)
        plt.title("Predicted probabilities")
        plt.savefig(Path(out_dir)/"probs_hist.png")
        plt.close()
    else:
        for i, cls in enumerate(classes):
            plt.figure()
            plt.hist(probs[:, i], bins=30)
            plt.title(f"Prob histogram: {cls}")
            plt.savefig(Path(out_dir)/f"prob_hist_{i}_{cls}.png")
            plt.close()

# ---------------- main ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True, help="Path to test folder or parent with test/ subfolder")
    p.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth.tar", help="Checkpoint (state dict or wrapped dict)")
    p.add_argument("--model", type=str, default="resnet18", choices=["resnet18","resnet34","resnet50","efficientnet_b0"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--output_dir", type=str, default="eval_results")
    p.add_argument("--num_classes", type=int, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_dir)
    if (data_root / "test").exists():
        test_dir = data_root / "test"
    else:
        test_dir = data_root

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    loader, ds = make_loader(test_dir, img_size=args.img_size, batch_size=args.batch_size, num_workers=args.num_workers)
    classes = ds.classes
    if args.num_classes is None:
        num_classes = len(classes)
    else:
        num_classes = args.num_classes
    print(f"Num classes: {num_classes} Classes: {classes}")

    # build model and load checkpoint
    model = build_model(args.model, num_classes=num_classes, dropout=0.5)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    # handle DataParallel 'module.' prefix
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        new_state = { (k[len("module."):]): v for k,v in state.items() }
    else:
        new_state = state
    model.load_state_dict(new_state)
    model = model.to(device)
    model.eval()

    soft = nn.Softmax(dim=1)
    all_targets = []
    all_preds = []
    all_probs = []
    sample_paths = []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            if outputs.ndim == 1:
                outputs = outputs.unsqueeze(1)
            probs = soft(outputs).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_targets.extend(targets.numpy().tolist())
            all_preds.extend(preds.tolist())
            if probs.shape[1] == 1:
                all_probs.extend(probs[:,0].tolist())
            else:
                all_probs.extend(probs.tolist())
            # record file paths
            sample_paths.extend([p[0] for p in loader.dataset.samples[len(sample_paths): len(sample_paths)+imgs.size(0)]])

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    probs_arr = np.array(all_probs)

    cm = confusion_matrix(y_true, y_pred)
    print("Confusion matrix:\n", cm)
    plot_confusion_matrix(cm, classes, out_dir / "confusion_matrix.png")

    report = classification_report(y_true, y_pred, target_names=classes, digits=4)
    print("Classification report:\n", report)
    (out_dir / "classification_report.txt").write_text(report)

    # robust AUC
    try:
        if num_classes == 2:
            if probs_arr.ndim == 1:
                auc_val = roc_auc_score(y_true, probs_arr)
            else:
                auc_val = roc_auc_score(y_true, probs_arr[:, 1])
        else:
            y_bin = label_binarize(y_true, classes=list(range(num_classes)))
            auc_val = roc_auc_score(y_bin, probs_arr)
        print(f"AUC: {auc_val:.4f}")
        (out_dir / "auc.txt").write_text(str(auc_val))
    except Exception as e:
        print("AUC calculation failed:", e)
        (out_dir / "auc.txt").write_text("failed: " + str(e))

    # save preds csv
    df = pd.DataFrame({
        "file": sample_paths,
        "true": y_true,
        "pred": y_pred
    })
    if probs_arr.ndim == 1:
        df["prob_pos"] = probs_arr
    else:
        for i,cls in enumerate(classes):
            df[f"prob_{i}_{cls}"] = probs_arr[:, i]
    df.to_csv(out_dir / "predictions.csv", index=False)

    plot_prob_histograms(all_probs, classes, out_dir / "prob_histograms")

    meta = {
        "num_samples": int(len(y_true)),
        "num_classes": int(num_classes),
        "classes": classes,
        "checkpoint": str(ckpt_path)
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    print("Evaluation complete. Results saved to:", out_dir)

if __name__ == "__main__":
    main()
