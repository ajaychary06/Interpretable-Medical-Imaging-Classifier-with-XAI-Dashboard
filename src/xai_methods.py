# src/xai_methods.py
"""
xai_methods.py

Contains:
 - GradCAM
 - GuidedBackprop
 - GuidedGradCAM
 - SmoothGrad
 - helpers to preprocess and overlay heatmaps

Works with PyTorch models (CPU/GPU).
"""

from typing import Tuple, List
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

# ---------------- utilities ----------------
def preprocess_image(pil_img: Image.Image, img_size=224):
    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    return tf(pil_img).unsqueeze(0)  # 1,C,H,W

def normalize_cam(cam: np.ndarray):
    cam = np.maximum(cam, 0)
    if cam.max() - cam.min() != 0:
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    else:
        cam = cam * 0.0
    return cam

def apply_colormap_on_image(org_img: Image.Image, mask: np.ndarray, colormap=plt.cm.jet, alpha=0.5):
    """
    mask: 2D array [0..1], same spatial resolution as org_img
    Returns PIL Image with heatmap overlay
    """
    cmap = colormap
    heatmap = cmap(mask)[:, :, :3]  # H,W,3
    heatmap = (heatmap * 255).astype(np.uint8)
    heat_img = Image.fromarray(heatmap).convert("RGBA")
    org_img = org_img.convert("RGBA").resize(heat_img.size)
    blended = Image.blend(org_img, heat_img, alpha=alpha)
    return blended.convert("RGB")

# ---------------- Grad-CAM ----------------
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        # register hooks
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        # note: backward hook may produce a FutureWarning on some PyTorch versions
        self.fwd_handle = target_layer.register_forward_hook(forward_hook)
        self.bwd_handle = target_layer.register_backward_hook(backward_hook)

    def _get_cam(self, class_idx: int, input_tensor: torch.Tensor):
        outputs = self.model(input_tensor)
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(0)
        self.model.zero_grad()
        score = outputs[0, class_idx]
        score.backward(retain_graph=True)

        grads = self.gradients  # [B,C,H,W]
        acts  = self.activations  # [B,C,H,W]
        weights = torch.mean(grads, dim=(2,3), keepdim=True)  # [B,C,1,1]
        cam = torch.sum(weights * acts, dim=1, keepdim=True)  # [B,1,H,W]
        cam = cam.squeeze().cpu().numpy()
        cam = normalize_cam(cam)
        return cam

    def generate_cam(self, input_tensor: torch.Tensor, class_idx: int):
        return self._get_cam(class_idx, input_tensor)

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

# ---------------- Guided Backprop ----------------
class GuidedBackprop:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.eval()
        self.handles = []
        # Replace ReLU backward with guided version
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                handle = module.register_backward_hook(self._relu_backward_hook)
                self.handles.append(handle)

    def _relu_backward_hook(self, module, grad_in, grad_out):
        if isinstance(grad_in, tuple):
            modified = tuple([torch.clamp(g, min=0.0) if g is not None else None for g in grad_in])
            return modified
        return grad_in

    def generate_gradients(self, input_tensor: torch.Tensor, class_idx: int):
        inp = input_tensor.clone().requires_grad_(True)
        outputs = self.model(inp)
        if outputs.ndim == 1:
            outputs = outputs.unsqueeze(0)
        score = outputs[0, class_idx]
        self.model.zero_grad()
        score.backward(retain_graph=True)
        grads = inp.grad.detach().cpu().squeeze().numpy()  # C,H,W
        grads = np.transpose(grads, (1,2,0))  # H,W,C
        grads = np.abs(grads).sum(axis=2)    # H,W
        grads = normalize_cam(grads)
        return grads

    def close(self):
        for h in self.handles:
            h.remove()

# ---------------- Guided Grad-CAM ----------------
def guided_grad_cam(gcam_mask: np.ndarray, guided_grads: np.ndarray):
    """
    gcam_mask: Hc x Wc (feature map size), values in [0..1]
    guided_grads: Hg x Wg (input image size), values in [0..1]
    Returns elementwise product normalized to [0..1] at guided_grads resolution.
    """
    # if shapes already match, just multiply
    if gcam_mask.shape == guided_grads.shape:
        g = guided_grads * gcam_mask
        return normalize_cam(g)

    # otherwise, resize gcam_mask to guided_grads shape using bilinear upsampling
    cam_img = Image.fromarray((gcam_mask * 255).astype(np.uint8))
    cam_resized = cam_img.resize((guided_grads.shape[1], guided_grads.shape[0]), resample=Image.BILINEAR)
    cam_arr = np.array(cam_resized).astype(np.float32) / 255.0

    g = guided_grads * cam_arr
    return normalize_cam(g)

# ---------------- SmoothGrad ----------------
def smooth_grad(model, input_tensor: torch.Tensor, class_idx: int, stdev_spread=0.15, n_samples=25):
    mean = 0.0
    stdev = stdev_spread * (input_tensor.max() - input_tensor.min()).item()
    gb = GuidedBackprop(model)
    total = None
    for i in range(n_samples):
        noise = torch.randn_like(input_tensor) * stdev + mean
        noisy_input = input_tensor + noise
        grads = gb.generate_gradients(noisy_input, class_idx)
        if total is None:
            total = grads
        else:
            total += grads
    gb.close()
    avg = total / float(n_samples)
    avg = normalize_cam(avg)
    return avg
