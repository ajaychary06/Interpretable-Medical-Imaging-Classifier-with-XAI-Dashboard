# src/xai_generate.py
"""
xai_generate.py

Generates Grad-CAM, Guided Backprop, Guided Grad-CAM and SmoothGrad visualizations.

Examples:
# Single image
python src/xai_generate.py --img "data/processed/test/yes/Y1.jpg" --checkpoint "checkpoints/best_model.pth.tar" --model resnet18 --outdir xai_out --layer layer4

# Whole folder (process up to max_images)
python src/xai_generate.py --img_dir "data/processed/test" --checkpoint "checkpoints/best_model.pth.tar" --model resnet18 --outdir xai_out --max_images 20
"""
import argparse
from pathlib import Path
import torch
from PIL import Image
import numpy as np

from xai_methods import preprocess_image, GradCAM, GuidedBackprop, guided_grad_cam, apply_colormap_on_image, smooth_grad

from torchvision import models
import torch.nn as nn

def build_model_for_infer(model_name="resnet18", num_classes=2, pretrained=False, dropout=0.5):
    model_name = model_name.lower()
    if model_name.startswith("resnet"):
        model = getattr(models, model_name)(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        try:
            in_features = model.classifier[1].in_features
            model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        except Exception:
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    else:
        raise ValueError("Unsupported model")
    return model

def find_layer(model, layer_name="layer4"):
    for n,m in reversed(list(model.named_modules())):
        if layer_name in n:
            return m
    raise ValueError(f"Layer with name containing '{layer_name}' not found. Try 'layer4' for ResNet.")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img", type=str, default=None, help="Single image path")
    p.add_argument("--img_dir", type=str, default=None, help="ImageFolder-style folder (will iterate classes)")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model", type=str, default="resnet18", choices=["resnet18","resnet34","resnet50","efficientnet_b0"])
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--outdir", type=str, default="xai_out")
    p.add_argument("--layer", type=str, default="layer4", help="Target conv layer name fragment (e.g. layer4)")
    p.add_argument("--num_classes", type=int, default=None)
    p.add_argument("--max_images", type=int, default=50)
    return p.parse_args()

def load_checkpoint(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    state = checkpoint.get("model_state", checkpoint)
    if isinstance(state, dict) and any(k.startswith("module.") for k in state.keys()):
        state = {k[len("module."):]:v for k,v in state.items()}
    model.load_state_dict(state)
    return model

def process_image_path(p: Path, model, device, layer_module, outdir: Path, img_size=224):
    img = Image.open(p).convert("RGB")
    inp = preprocess_image(img, img_size=img_size).to(device)
    with torch.no_grad():
        out = model(inp)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))
    # Grad-CAM
    gcam = GradCAM(model, layer_module)
    cam_mask = gcam.generate_cam(inp, pred)
    gcam.close()
    # Guided BP
    gb = GuidedBackprop(model)
    guided = gb.generate_gradients(inp, pred)
    gb.close()
    # Guided Grad-CAM
    guided_gc = guided_grad_cam(cam_mask, guided)
    # SmoothGrad (optional, can be slow)
    try:
        sg = smooth_grad(model, inp, pred, n_samples=12)
    except Exception:
        sg = None

    outdir.mkdir(parents=True, exist_ok=True)
    info_txt = outdir / f"{p.stem}_pred.txt"
    info_txt.write_text(f"file: {p}\n pred: {pred}\n probs: {probs.tolist()}\n")

    # overlay cam
    mask_resized = Image.fromarray((cam_mask * 255).astype(np.uint8)).resize(img.size)
    mask_arr = np.array(mask_resized) / 255.0
    overlay = apply_colormap_on_image(img, mask_arr, alpha=0.45)
    overlay.save(outdir / f"{p.stem}_gradcam.jpg")

    # guided gradcam overlay
    gg_resized = Image.fromarray((guided_gc * 255).astype(np.uint8)).resize(img.size)
    gg_arr = np.array(gg_resized) / 255.0
    overlay2 = apply_colormap_on_image(img, gg_arr, alpha=0.5)
    overlay2.save(outdir / f"{p.stem}_guided_gradcam.jpg")

    # guided only (as heatmap)
    guided_resized = Image.fromarray((guided * 255).astype(np.uint8)).resize(img.size)
    guided_arr = np.array(guided_resized) / 255.0
    overlay3 = apply_colormap_on_image(img, guided_arr, alpha=0.6)
    overlay3.save(outdir / f"{p.stem}_guided.jpg")

    # smoothgrad if present
    if sg is not None:
        sg_resized = Image.fromarray((sg * 255).astype(np.uint8)).resize(img.size)
        sg_arr = np.array(sg_resized) / 255.0
        overlay4 = apply_colormap_on_image(img, sg_arr, alpha=0.5)
        overlay4.save(outdir / f"{p.stem}_smoothgrad.jpg")

    print("Saved XAI images for:", p.name)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # build and load model
    model = build_model_for_infer(args.model, num_classes=args.num_classes or 2, pretrained=False)
    model = model.to(device)
    model = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    # find conv layer
    layer_module = find_layer(model, args.layer)

    if args.img:
        p = Path(args.img)
        process_image_path(p, model, device, layer_module, outdir, img_size=args.img_size)
        return

    if args.img_dir:
        root = Path(args.img_dir)
        processed = 0
        for cls in sorted(root.iterdir()):
            if not cls.is_dir():
                continue
            for imgf in sorted(cls.glob("*.*")):
                process_image_path(imgf, model, device, layer_module, outdir, img_size=args.img_size)
                processed += 1
                if processed >= args.max_images:
                    print("Processed max_images limit.")
                    return

if __name__ == "__main__":
    main()
