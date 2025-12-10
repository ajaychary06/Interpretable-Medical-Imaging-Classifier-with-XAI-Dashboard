# src/xai.py
r"""
Run from project root:
    conda activate xai_proj
    python src\xai.py

What it does:
 - Loads dataset loader and model by path (safe import)
 - Loads checkpoint checkpoints/best_model.pth
 - For the first N images in the test set, computes:
     - Integrated Gradients (pixel-level)
     - LayerGradCam (last ResNet block)
 - Saves:
     - raw image (copied)
     - ig overlay (results/xai/ig_<imagename>.png)
     - gradcam overlay (results/xai/gc_<imagename>.png)
     - combined side-by-side image (results/xai/combined_<imagename>.png)
 - Prints saved file paths.

Notes:
 - CPU-only friendly but slower than GPU.
 - Adjust NUM_IMAGES to process more/less.
"""
import importlib.util
from pathlib import Path
import torch, torch.nn.functional as F
from PIL import Image
import numpy as np
import os

# captum
from captum.attr import IntegratedGradients, LayerGradCam
from captum.attr import visualization as viz

# plotting
import matplotlib.pyplot as plt

# -----------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
RESULTS_XAI = PROJECT_ROOT / "results" / "xai"
RESULTS_XAI.mkdir(parents=True, exist_ok=True)

# config
NUM_IMAGES = 12   # change as needed (small number to start)
BATCH_SIZE = 1
IMAGE_SIZE = 224
NUM_CLASSES = 2
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
def import_from_path(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# load dataset & model modules
dataset_mod = import_from_path(SRC_DIR / "dataset.py", "dataset_mod")
make_loader = dataset_mod.make_loader

model_mod = import_from_path(SRC_DIR / "model.py", "model_mod")
get_resnet18 = model_mod.get_resnet18

# create loaders (single-batch)
test_loader = make_loader(str(PROJECT_ROOT / "data" / "processed"), split="test", batch_size=BATCH_SIZE, image_size=IMAGE_SIZE, num_workers=0)

# load model
model = get_resnet18(num_classes=NUM_CLASSES, pretrained=False).to(DEVICE)
if CHECKPOINT.exists():
    ck = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    print("Loaded checkpoint:", CHECKPOINT)
else:
    raise SystemExit("Checkpoint not found. Train first.")

model.eval()

# choose layer for LayerGradCam (ResNet18 last conv)
# for ResNet18 the last block is model.layer4[-1]; use its conv2
target_layer = model.layer4[-1].conv2

# helper: convert tensor -> numpy image (H,W,3) uint8 (denormalize)
def tensor_to_pil(img_tensor):
    # img_tensor: [3,H,W] tensor (normalized)
    img = img_tensor.cpu().numpy()
    img = np.transpose(img, (1,2,0))
    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])
    img = (img * std + mean)
    img = np.clip(img*255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

# helper: normalize attribution to 0-1
def normalize_attr(attr):
    a = attr.copy()
    a = np.abs(a).sum(axis=0) if attr.ndim==3 else np.abs(a)
    a = a - a.min()
    if a.max() > 0:
        a = a / a.max()
    return a

# run explainers on first NUM_IMAGES images
ig = IntegratedGradients(model)
lgc = LayerGradCam(model, target_layer)

saved = []
count = 0
for imgs, labels in test_loader:
    imgs = imgs.to(DEVICE)
    labels = labels.to(DEVICE)
    with torch.no_grad():
        outputs = model(imgs)
        probs = F.softmax(outputs, dim=1)
        preds = probs.argmax(dim=1)

    # for each item in batch (batch_size==1)
    for i in range(imgs.size(0)):
        if count >= NUM_IMAGES:
            break
        img_tensor = imgs[i].unsqueeze(0)       # 1,C,H,W
        label = int(labels[i].item())
        pred = int(preds[i].item())

        # baseline for IG: zero image
        baseline = torch.zeros_like(img_tensor).to(DEVICE)

        # Integrated Gradients (pixel-level)
        attributions_ig = ig.attribute(img_tensor, baselines=baseline, target=pred, internal_batch_size=1)
        # convert to numpy HxW
        attr_ig_np = attributions_ig.squeeze(0).cpu().numpy()
        attr_ig_norm = normalize_attr(attr_ig_np)  # HxW

        # Layer GradCam
        attributions_lgc = lgc.attribute(img_tensor, target=pred)
        # attributions_lgc may be HxW, CxHxW, or 1xHxW etc. Normalize shape to (1, C, H, W)
        if attributions_lgc.dim() == 2:
            inp = attributions_lgc.unsqueeze(0).unsqueeze(0)   # -> (1,1,H,W)
        elif attributions_lgc.dim() == 3:
            inp = attributions_lgc.unsqueeze(0)               # -> (1,C,H,W)
        else:
            inp = attributions_lgc                              # assume already (N,C,H,W)

        # upsample to input size (N,C,H,W) -> (N,C,IMAGE_SIZE,IMAGE_SIZE)
        upsampled = torch.nn.functional.interpolate(inp, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)

        # collapse channel dimension to single 2D map (average across channels) and remove batch dim
        upsampled = upsampled.squeeze(0)   # (C,H,W) or (1,H,W)
        if upsampled.dim() == 3:
            # average over channels
            up2d = upsampled.mean(dim=0)
        else:
            up2d = upsampled  # already 2D

        attr_gc_np = up2d.cpu().detach().numpy()
        attr_gc_norm = normalize_attr(attr_gc_np)  # HxW


        # convert original image to PIL
        pil_img = tensor_to_pil(img_tensor.squeeze(0))

        # create overlays using matplotlib colormap and blend with original
        def save_overlay(attr_norm, out_path, cmap='jet', alpha=0.45):
            # attr_norm: HxW float 0-1
            cmap_fn = plt.get_cmap(cmap)
            colored = cmap_fn(attr_norm)[:,:,:3]  # HxWx3
            colored = (colored * 255).astype(np.uint8)
            colored_pil = Image.fromarray(colored).resize(pil_img.size)
            blended = Image.blend(pil_img.convert("RGBA"), colored_pil.convert("RGBA"), alpha=alpha)
            blended.convert("RGB").save(out_path)

        base_name = f"img_{count}_true{label}_pred{pred}"
        raw_path = RESULTS_XAI / f"{base_name}_orig.png"
        pil_img.save(raw_path)

        ig_path = RESULTS_XAI / f"{base_name}_ig.png"
        save_overlay(attr_ig_norm, ig_path, cmap='jet', alpha=0.45)

        gc_path = RESULTS_XAI / f"{base_name}_gc.png"
        save_overlay(attr_gc_norm, gc_path, cmap='jet', alpha=0.5)

        # combined: side-by-side (orig | IG | GradCam)
        combined = Image.new('RGB', (pil_img.width * 3 + 20, pil_img.height))
        combined.paste(pil_img, (0,0))
        combined.paste(Image.open(ig_path), (pil_img.width + 10, 0))
        combined.paste(Image.open(gc_path), (pil_img.width*2 + 20, 0))
        combined_path = RESULTS_XAI / f"{base_name}_combined.png"
        combined.save(combined_path)

        print("Saved:", raw_path.name, ig_path.name, gc_path.name, combined_path.name)
        saved.append((raw_path, ig_path, gc_path, combined_path))
        count += 1

    if count >= NUM_IMAGES:
        break

print(f"Done. Saved {len(saved)} xai outputs in {RESULTS_XAI}")
