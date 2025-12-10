# src/xai_extra.py
"""
Stable Grad-CAM++ + Integrated Gradients + fusion helpers.

This version avoids backward hooks (which can cause view/inplace autograd errors)
by capturing activations via a forward hook and computing gradients with
torch.autograd.grad(score, activations).
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation that is robust to hook/view issues.
    Usage:
      gcpp = GradCAMPlusPlus(model, target_layer)
      cam = gcpp.generate_cam(input_tensor, class_idx)
      gcpp.close()
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self._fh = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module, inp, out):
        # store activation tensor (keep as-is so it stays in graph)
        self.activations = out

    def generate_cam(self, input_tensor: torch.Tensor, target_class: int):
        """
        input_tensor: 1xC x H x W (on device)
        returns: Hc x Wc numpy array normalized [0..1]
        """
        # forward
        output = self.model(input_tensor)  # shape [1, num_classes] or [1]
        if output.ndim == 1:
            output = output.unsqueeze(0)
        # pick score for target class
        score = output[0, target_class]

        # ensure activations were captured
        acts = self.activations  # Tensor [1, C, h, w]
        if acts is None:
            raise RuntimeError("Activations not captured. Check target layer name.")

        # compute gradients of score w.r.t activations explicitly
        # returns a tensor with same shape as acts
        grads = torch.autograd.grad(score, acts, retain_graph=True, create_graph=False)[0]

        # move to CPU numpy and compute Grad-CAM++ weights
        # grads: [1, C, h, w], acts: [1, C, h, w]
        with torch.no_grad():
            grads_val = grads  # still a tensor
            acts_val = acts

            # compute squared and cubed grads
            grads_pow_2 = grads_val * grads_val
            grads_pow_3 = grads_pow_2 * grads_val

            # sum over spatial dims
            sum_acts = torch.sum(acts_val, dim=(2, 3), keepdim=True)  # [1,C,1,1]
            eps = 1e-8

            denom = 2.0 * grads_pow_2 + sum_acts * grads_pow_3
            denom = denom + eps

            alpha = grads_pow_2 / denom  # [1,C,H,W]
            alpha = torch.clamp(alpha, min=0.0)

            positive_grads = F.relu(grads_val)
            weights = torch.sum(alpha * positive_grads, dim=(2, 3), keepdim=True)  # [1,C,1,1]

            cam = torch.sum(weights * acts_val, dim=1)  # [1,H,W]
            cam = cam.squeeze(0).cpu().numpy()

            # relu + normalize
            cam = np.maximum(cam, 0)
            if cam.max() - cam.min() > 0:
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            else:
                cam = np.zeros_like(cam)

        return cam

    def close(self):
        try:
            self._fh.remove()
        except Exception:
            pass


def integrated_gradients(model, input_tensor, target_class, steps=30, device=None):
    """
    Simple integrated gradients implementation.
    input_tensor: 1xC x H x W
    returns: H x W numpy attribution map normalized [0..1]
    """
    if device is None:
        device = input_tensor.device
    baseline = torch.zeros_like(input_tensor).to(device)

    # generate scaled inputs and accumulate gradients
    scaled_inputs = [baseline + (float(i) / steps) * (input_tensor - baseline) for i in range(1, steps + 1)]
    grads = []
    model.eval()
    for scaled in scaled_inputs:
        scaled = scaled.requires_grad_(True)
        out = model(scaled)
        if out.ndim == 1:
            out = out.unsqueeze(0)
        score = out[0, target_class]
        model.zero_grad()
        score.backward(retain_graph=True)
        if scaled.grad is None:
            g = torch.zeros_like(scaled)
        else:
            g = scaled.grad.detach().cpu().numpy()  # 1 x C x H x W
        grads.append(g)
    avg_grads = np.mean(np.concatenate(grads, axis=0), axis=0)  # C x H x W
    delta = (input_tensor.detach().cpu().numpy()[0] - baseline.detach().cpu().numpy()[0])  # C x H x W
    integrated = delta * avg_grads
    attributions = np.abs(integrated).sum(axis=0)  # H x W
    if attributions.max() - attributions.min() > 0:
        attributions = (attributions - attributions.min()) / (attributions.max() - attributions.min() + 1e-8)
    else:
        attributions = np.zeros_like(attributions)
    return attributions


def fuse_maps(maps, method="weighted", weights=None):
    """
    Fuse multiple attribution maps (HxW arrays).
    """
    if len(maps) == 0:
        raise ValueError("No maps to fuse")
    shapes = [m.shape for m in maps]
    target = shapes[0]
    # resize mismatched maps to target using PIL bilinear
    resized = []
    for m in maps:
        if m.shape == target:
            resized.append(m)
        else:
            im = Image.fromarray((m * 255).astype('uint8')).resize((target[1], target[0]), resample=Image.BILINEAR)
            resized.append(np.array(im).astype(np.float32) / 255.0)
    arr = np.stack(resized, axis=0)
    if method == "mean":
        fused = np.mean(arr, axis=0)
    elif method == "max":
        fused = np.max(arr, axis=0)
    elif method == "weighted":
        if weights is None:
            weights = [1.0 / arr.shape[0]] * arr.shape[0]
        w = np.array(weights).reshape(-1, 1, 1)
        fused = np.sum(arr * w, axis=0) / (np.sum(w) + 1e-12)
    else:
        raise ValueError("Unknown method")

    if fused.max() - fused.min() > 0:
        fused = (fused - fused.min()) / (fused.max() - fused.min() + 1e-8)
    else:
        fused = fused * 0.0
    return fused
