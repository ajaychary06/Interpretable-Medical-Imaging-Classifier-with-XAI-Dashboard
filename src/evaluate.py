# src/evaluate.py
"""
Run from project root:
    conda activate xai_proj
    python src\evaluate.py

Outputs:
 - prints classification report & accuracy
 - saves results/predictions.csv
 - saves results/confusion_matrix.png
"""
import importlib.util
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import itertools
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)

# helper to import module by path
def import_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# load dataset maker and model builder
dataset_mod = import_from_path(SRC_DIR / "dataset.py", "dataset_by_path")
make_loader = dataset_mod.make_loader

model_mod = import_from_path(SRC_DIR / "model.py", "model_by_path")
get_resnet18 = model_mod.get_resnet18

# configs (match train.py defaults)
BATCH_SIZE = 8
IMAGE_SIZE = 224
NUM_CLASSES = 2
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# loaders
test_loader = make_loader(str(PROJECT_ROOT / "data" / "processed"), split="test", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, num_workers=0)
print("Test batches:", len(test_loader))

# model
model = get_resnet18(num_classes=NUM_CLASSES, pretrained=False)
model = model.to(device)
if CHECKPOINT.exists():
    ck = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ck["model_state_dict"])
    print("Loaded checkpoint:", CHECKPOINT)
else:
    print("No checkpoint found at", CHECKPOINT)
    raise SystemExit(1)

model.eval()
y_true = []
y_pred = []
filenames = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1).cpu().numpy()
        y_pred.extend(preds.tolist())
        y_true.extend(labels.numpy().tolist())

# class names inferred from processed folder
classes = sorted([p.name for p in (PROJECT_ROOT/"data"/"processed"/"train").iterdir() if p.is_dir()])
print("Classes:", classes)

# metrics
print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=classes, digits=4))
acc = accuracy_score(y_true, y_pred)
print("Test accuracy: {:.4f}".format(acc))

# save predictions CSV (image path + true + pred)
# note: to keep simple we re-iterate to get filenames aligned with preds
rows = []
for cls_idx, cls in enumerate(classes):
    src_dir = PROJECT_ROOT / "data" / "processed" / "test" / cls
    files = sorted([p.name for p in src_dir.iterdir() if p.suffix.lower() in (".jpg",".jpeg",".png")])
    for fname in files:
        rows.append((str(src_dir / fname), cls, None))  # we'll fill preds later

# If counts mismatch, try fallback: assign sequentially
if len(rows) != len(y_pred):
    # rebuild rows by walking test folder in arbitrary order
    rows = []
    for p in (PROJECT_ROOT/"data"/"processed"/"test").rglob("*"):
        if p.is_file() and p.suffix.lower() in (".jpg",".jpeg",".png"):
            rows.append((str(p), p.parent.name, None))

# fill preds
for i in range(min(len(rows), len(y_pred))):
    rows[i] = (rows[i][0], rows[i][1], classes[y_pred[i]])

# write CSV
csv_path = RESULTS_DIR / "predictions.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path","true_label","pred_label"])
    writer.writerows(rows)
print("Saved predictions:", csv_path)

# confusion matrix plot
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion matrix")
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.tight_layout()
cm_path = RESULTS_DIR / "confusion_matrix.png"
plt.savefig(cm_path)
print("Saved confusion matrix:", cm_path)
