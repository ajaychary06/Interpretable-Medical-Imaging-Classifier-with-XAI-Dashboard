#!/usr/bin/env python3
"""
smoke_xai_extra.py - Auto-adapting Grad-CAM loader for ResNet-style checkpoints.

Saves:
  xai_smoke_campp.jpg  (grayscale upsampled CAM)
  xai_smoke_ig.jpg     (original image)
  xai_smoke_fused.jpg  (overlay)

Usage:
python scripts\smoke_xai_extra.py --img test.jpg --checkpoint "checkpoints/best_model.pth" --num-classes 2 --device cpu
"""

import os
import argparse
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import torchvision.transforms as T

# Attempt to import ResNet variants and weights enum
try:
    import torchvision.models as models
    from torchvision.models import ResNet50_Weights
    _HAS_WEIGHTS_ENUM = True
except Exception:
    import torchvision.models as models
    _HAS_WEIGHTS_ENUM = False

# ---------------- Helpers ----------------

def find_last_conv_module(model):
    import torch.nn as nn
    last_conv = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv

class ActivationsAndGradients:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            if grad_out is None:
                return
            g = grad_out[0]
            if g is not None:
                self.gradients = g.detach()

        self.fwd_handle = self.target_layer.register_forward_hook(forward_hook)
        if hasattr(self.target_layer, "register_full_backward_hook"):
            self.bwd_handle = self.target_layer.register_full_backward_hook(
                lambda module, grad_in, grad_out: backward_hook(module, grad_in, grad_out)
            )
        else:
            self.bwd_handle = self.target_layer.register_backward_hook(backward_hook)

    def remove_hooks(self):
        try:
            self.fwd_handle.remove()
        except Exception:
            pass
        try:
            self.bwd_handle.remove()
        except Exception:
            pass

def compute_gradcam_from_activations(activations, gradients):
    weights = gradients.mean(dim=(2,3), keepdim=True)
    weighted = (weights * activations).sum(dim=1, keepdim=True)
    cam = F.relu(weighted)
    cam = cam.squeeze(1)
    cam_np = cam.cpu().numpy()
    return cam_np[0]

def postprocess_and_save(cam, orig_img_bgr, save_prefix="xai_smoke"):
    cam = np.squeeze(cam)
    ih, iw = orig_img_bgr.shape[:2]  # H, W
    cam_up = cv2.resize(cam, (iw, ih), interpolation=cv2.INTER_LINEAR)

    print(
        f"Shapes -> campp (raw HxW): {cam.shape}  "
        f"campp (upsampled HxW): {cam_up.shape}  "
        f"ig (HxW): {(ih, iw)}"
    )

    cam_min, cam_max = cam_up.min(), cam_up.max()
    if cam_max - cam_min < 1e-8:
        cam_norm = np.zeros_like(cam_up, dtype=np.uint8)
    else:
        cam_norm = np.uint8(255 * (cam_up - cam_min) / (cam_max - cam_min))

    cam_gray_filename = f"{save_prefix}_campp.jpg"
    fused_filename = f"{save_prefix}_fused.jpg"
    orig_filename = f"{save_prefix}_ig.jpg"

    cv2.imwrite(cam_gray_filename, cam_norm)
    cam_color = cv2.applyColorMap(cam_norm, cv2.COLORMAP_JET)
    fused = cv2.addWeighted(orig_img_bgr, 0.6, cam_color, 0.4, 0)
    cv2.imwrite(fused_filename, fused)
    cv2.imwrite(orig_filename, orig_img_bgr)

    print(f"Saved smoke images: {cam_gray_filename}, {orig_filename}, {fused_filename}")
    return cam_gray_filename, orig_filename, fused_filename

# ---------------- Robust loader that infers architecture ----------------

def _build_resnet_variant(name, in_channels, num_classes):
    """
    Build a torchvision ResNet variant and adapt conv1 to in_channels and fc to num_classes.
    name: 'resnet18','resnet34','resnet50','resnet101','resnet152'
    """
    # weights=None ensures no pretrained ImageNet weights are loaded
    constructor = getattr(models, name)
    try:
        model = constructor(weights=None)  # torchvision >= 0.13
    except Exception:
        model = constructor(pretrained=False)
    # adapt conv1 if needed
    import torch.nn as nn
    # default conv1 = nn.Conv2d(3, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)
    # We'll examine existing model.conv1 and replace with matching in_channels and same kernel/stride/padding/out_channels
    old_conv = model.conv1
    old_ks = tuple(old_conv.kernel_size)
    old_stride = tuple(old_conv.stride)
    old_padding = tuple(old_conv.padding)
    out_channels = old_conv.out_channels
    bias = old_conv.bias is not None
    model.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=old_ks,
                            stride=old_stride, padding=old_padding, bias=bias)
    # adapt final fc
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    else:
        # some resnets may have classifier; fallback
        model.fc = nn.Linear(512, num_classes)
    return model

def load_model(device, checkpoint_path=None, num_classes=2):
    """
    Loads a ResNet-like model whose structure is auto-inferred from the checkpoint.
    Strategy:
      - load checkpoint dict and extract state dict (model_state_dict / state_dict / top-level)
      - check shapes of conv1.weight and fc.weight if present
      - infer in_channels and fc.in_features
      - try candidate resnet variants that match inferred fc in_features (512 -> resnet18/34, 2048 -> resnet50/101/152)
      - replace conv1 in_channels and fc to match num_classes, then attempt to load state
    """
    import torch.nn as nn

    # Helper to extract state dict
    state = None
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=device)
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                state = ckpt['model_state_dict']
                print("Using checkpoint['model_state_dict']")
            elif 'state_dict' in ckpt:
                state = ckpt['state_dict']
                print("Using checkpoint['state_dict']")
            else:
                # top-level might already be a state_dict
                has_param_keys = any(isinstance(k, str) and (k.endswith('.weight') or k.endswith('.bias')) for k in ckpt.keys())
                if has_param_keys:
                    state = ckpt
                    print("Using top-level checkpoint as state_dict")
                else:
                    print("Checkpoint keys:", list(ckpt.keys())[:20])
                    raise RuntimeError("Can't find state_dict in checkpoint.")
        else:
            raise RuntimeError("Checkpoint loaded but is not a dict.")

    # show sample keys for debugging
    sample_keys = list(state.keys())[:12]
    print("Sample keys from state_dict:", sample_keys)

    # If keys are prefixed like 'module.' or 'model.', strip them for inspection
    def _strip_prefixes(sd):
        sample = list(sd.keys())[:8]
        if any(k.startswith("module.") for k in sample):
            return {k.replace("module.", ""): v for k,v in sd.items()}
        return sd

    state = _strip_prefixes(state)
    sample_keys = list(state.keys())[:12]
    print("Post-strip sample keys:", sample_keys)

    # infer conv1 parameters if present
    conv1_weight = state.get("conv1.weight", None)
    fc_w = state.get("fc.weight", None)

    inferred_in_channels = None
    inferred_conv_kernel = None
    inferred_fc_in = None

    if conv1_weight is not None:
        # conv1.weight shape: (out_channels, in_channels, kh, kw)
        inferred_in_channels = conv1_weight.shape[1]
        inferred_conv_kernel = (conv1_weight.shape[2], conv1_weight.shape[3])
        print(f"Inferred conv1 in_channels={inferred_in_channels}, kernel={inferred_conv_kernel}")

    if fc_w is not None:
        # fc.weight shape: (out_features, in_features)
        inferred_fc_in = fc_w.shape[1]
        print(f"Inferred fc in_features={inferred_fc_in}")

    # Choose candidate resnet variants
    candidates = []
    if inferred_fc_in == 2048:
        candidates = ["resnet50","resnet101","resnet152"]
    elif inferred_fc_in == 512:
        # Could be resnet18 or resnet34 (both have fc in_features=512)
        # Prefer resnet18 first (smaller), then resnet34
        candidates = ["resnet18","resnet34","resnet50"]
    else:
        # unknown -> try smaller to larger
        candidates = ["resnet18","resnet34","resnet50","resnet101","resnet152"]

    last_error = None
    for name in candidates:
        try:
            print(f"Trying to build {name} with in_channels={inferred_in_channels} num_classes={num_classes}")
            in_ch = int(inferred_in_channels) if inferred_in_channels is not None else 3
            model = _build_resnet_variant(name, in_ch, num_classes)
            # Prepare cleaned state for load: strip common wrappers
            cleaned = {}
            for k,v in state.items():
                new_k = k
                for p in ("module.", "model.", "net."):
                    if new_k.startswith(p):
                        new_k = new_k[len(p):]
                cleaned[new_k] = v
            # Attempt load
            model.load_state_dict(cleaned)
            print(f"Loaded checkpoint into {name} successfully.")
            model.eval()
            model.to(device)
            # pick preprocess transform
            if _HAS_WEIGHTS_ENUM and name == "resnet50":
                preprocess = ResNet50_Weights.IMAGENET1K_V2.transforms()
            else:
                preprocess = T.Compose([
                    T.Resize((224,224)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
                ])
            return model, preprocess
        except Exception as e:
            print(f"{name} load failed: {e}")
            last_error = e
            continue

    # If we reach here, all attempts failed
    print("All candidate model loads failed. Last error:")
    raise last_error

# ---------------- Input prep & Grad-CAM run ----------------

def prepare_input(img_path, preprocess):
    pil = Image.open(img_path).convert('RGB')
    orig_rgb = np.array(pil)
    orig_bgr = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2BGR)
    input_tensor = preprocess(pil).unsqueeze(0)
    return orig_bgr, input_tensor

def run_gradcam_on_image(model, preprocess, img_path, device, target_class=None):
    orig_bgr, input_tensor = prepare_input(img_path, preprocess)
    input_tensor = input_tensor.to(device)

    target_layer = find_last_conv_module(model)
    if target_layer is None:
        raise RuntimeError("No Conv2d layer found in the model to attach Grad-CAM hooks.")

    ag = ActivationsAndGradients(model, target_layer)

    output = model(input_tensor)
    probs = torch.nn.functional.softmax(output, dim=1)
    pred_prob, pred_idx = torch.max(probs, dim=1)
    pred_idx = pred_idx.item()
    pred_prob = pred_prob.item()

    if target_class is None:
        target_class = pred_idx

    model.zero_grad()
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1.0
    output.backward(gradient=one_hot)

    activations = ag.activations
    gradients = ag.gradients
    if activations is None or gradients is None:
        raise RuntimeError("Failed to collect activations/gradients from the target layer. Check hooks.")

    cam = compute_gradcam_from_activations(activations, gradients)
    ag.remove_hooks()
    return cam, orig_bgr, pred_idx, pred_prob

# ---------------- CLI & main ----------------

def main():
    parser = argparse.ArgumentParser(description="Auto-adapting Grad-CAM for ResNet checkpoints")
    parser.add_argument("--img", type=str, required=True, help="input image path")
    parser.add_argument("--checkpoint", type=str, default=None, help="path to model checkpoint (.pth)")
    parser.add_argument("--num-classes", type=int, default=2, help="number of classes (default 2)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device")
    parser.add_argument("--target-class", type=int, default=None, help="target class index (optional)")
    args = parser.parse_args()

    img_path = args.img
    device = torch.device(args.device)

    if not os.path.exists(img_path):
        print(f"Input image not found: {img_path}")
        return

    model, preprocess = load_model(device, checkpoint_path=args.checkpoint, num_classes=args.num_classes)
    cam, orig_bgr, pred_idx, pred_prob = run_gradcam_on_image(model, preprocess, img_path, device, args.target_class)

    ih, iw = orig_bgr.shape[:2]
    cam_up = cv2.resize(cam, (iw, ih), interpolation=cv2.INTER_LINEAR)
    print(f"Shapes -> campp (raw HxW): {cam.shape}  campp (upsampled HxW): {cam_up.shape}  ig (HxW): {orig_bgr.shape[:2]}")

    postprocess_and_save(cam, orig_bgr, save_prefix="xai_smoke")

    labels = ['no', 'yes'] if args.num_classes == 2 else [f"class_{i}" for i in range(args.num_classes)]
    pred_label = labels[pred_idx] if pred_idx < len(labels) else f"class_{pred_idx}"
    print(f"Predicted index: {pred_idx}, label: {pred_label}, probability: {pred_prob:.4f}")

if __name__ == "__main__":
    main()
