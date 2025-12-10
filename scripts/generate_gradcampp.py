# scripts/generate_gradcampp.py
import sys, os
proj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, proj_root)
sys.path.insert(0, os.path.join(proj_root, "src"))

import argparse
from pathlib import Path
from PIL import Image
import numpy as np
import torch
from xai_methods import preprocess_image, apply_colormap_on_image
from xai_extra import GradCAMPlusPlus, integrated_gradients, fuse_maps
from xai_generate import build_model_for_infer, load_checkpoint, find_layer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--img_dir", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--model", type=str, default="resnet18")
    p.add_argument("--outdir", type=str, default="xai_out_extra")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--max_images", type=int, default=100)
    p.add_argument("--layer", type=str, default="layer4")
    p.add_argument("--weights", type=float, nargs=2, default=[0.6,0.4], help="weights for [gradcampp, integrated_gradients]")
    return p.parse_args()

def process_file(p, model, device, layer_module, outdir, img_size, weights):
    p = Path(p)
    img = Image.open(p).convert("RGB")
    inp = preprocess_image(img, img_size=img_size).to(device)

    # forward to get prediction
    with torch.no_grad():
        out = model(inp)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
        pred = int(np.argmax(probs))

    # Grad-CAM++
    gcpp = GradCAMPlusPlus(model, layer_module)
    campp = gcpp.generate_cam(inp, pred)
    gcpp.close()

    # Integrated Gradients (may be slower)
    ig = integrated_gradients(model, inp, target_class=pred, steps=30, device=device)

    # fuse
    fused = fuse_maps([campp, ig], method="weighted", weights=weights)

    outdir.mkdir(parents=True, exist_ok=True)
    stem = p.stem
    # save pred info
    (outdir / f"{stem}_pred.txt").write_text(f"file: {p}\npred: {pred}\nprobs: {probs.tolist()}\n")

    # save gradcampp overlay
    cam_mask = Image.fromarray((campp * 255).astype(np.uint8)).resize(img.size)
    cam_arr = np.array(cam_mask) / 255.0
    overlay = apply_colormap_on_image(img, cam_arr, alpha=0.45)
    overlay.save(outdir / f"{stem}_gradcampp.jpg")

    # save ig overlay
    ig_mask = Image.fromarray((ig * 255).astype(np.uint8)).resize(img.size)
    ig_arr = np.array(ig_mask) / 255.0
    overlay2 = apply_colormap_on_image(img, ig_arr, alpha=0.45)
    overlay2.save(outdir / f"{stem}_ig.jpg")

    # save fused overlay
    fused_mask = Image.fromarray((fused * 255).astype(np.uint8)).resize(img.size)
    fused_arr = np.array(fused_mask) / 255.0
    overlay3 = apply_colormap_on_image(img, fused_arr, alpha=0.5)
    overlay3.save(outdir / f"{stem}_fused.jpg")

    print("Saved:", stem, "pred=", pred)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model_for_infer(args.model, num_classes=2, pretrained=False).to(device)
    model = load_checkpoint(model, args.checkpoint, device)
    model.eval()

    layer_module = find_layer(model, args.layer)

    img_root = Path(args.img_dir)
    processed = 0
    for cls in sorted(img_root.iterdir()):
        if not cls.is_dir():
            continue
        for imgf in sorted(cls.glob("*.*")):
            process_file(imgf, model, device, layer_module, Path(args.outdir), args.img_size, args.weights)
            processed += 1
            if processed >= args.max_images:
                print("Processed max_images limit.")
                return

if __name__ == "__main__":
    main()
