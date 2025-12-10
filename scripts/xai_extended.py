"""
xai_extended.py
Utilities for loading a PyTorch model, making predictions, and generating XAI attributions
(Grad-CAM via Captum or pytorch-grad-cam fallback, and Integrated Gradients via Captum).

Author: (updated)
Notes:
 - Replace MODEL_PATH with your model checkpoint path if you have one.
 - Replace TARGET_LAYER_NAME with the name of the convolutional layer used for Grad-CAM.
 - This file uses only spaces (4) for indentation to avoid TabError.
"""

import os
import sys
from typing import Tuple, Optional, Union, IO, Dict, Any

import warnings
import inspect
import io

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision import models

# Optional: Captum for attributions (preferred)
try:
    from captum.attr import IntegratedGradients, LayerGradCam
except Exception:
    IntegratedGradients = None
    LayerGradCam = None

# Optional fallback: pytorch-grad-cam (if installed)
try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
except Exception:
    GradCAMPlusPlus = None
    ClassifierOutputTarget = None

# --- Configuration to change as needed ---
MODEL_PATH = None  # e.g., r"C:\path\to\model.pth" or None to use a pretrained ResNet18
TARGET_LAYER_NAME = "layer4"  # default candidate for ResNet-like models (used for Grad-CAM)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ----------------------------------------

# --- Preprocessing pipeline ---
IMG_SIZE = 224
_preprocess = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def _is_file_like(obj) -> bool:
    return hasattr(obj, "read") and callable(getattr(obj, "read"))


def load_image(image_source: Union[str, Image.Image, IO, bytes]) -> Image.Image:
    """
    Load an image from disk/path, a file-like object (stream), bytes, or return if already PIL Image.
    Returns a PIL Image in RGB mode.
    """
    # If already PIL Image
    if isinstance(image_source, Image.Image):
        return image_source.convert("RGB")

    # If source is bytes
    if isinstance(image_source, (bytes, bytearray)):
        try:
            return Image.open(io.BytesIO(image_source)).convert("RGB")
        except Exception as e:
            raise IOError(f"Could not open image from bytes: {e}")

    # If source is file-like (has read())
    if _is_file_like(image_source):
        try:
            # read bytes (do not assume file pointer at start)
            try:
                image_source.seek(0)
            except Exception:
                pass
            data = image_source.read()
            if isinstance(data, str):
                data = data.encode("utf-8")
            return Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            raise IOError(f"Could not open image from file-like object: {e}")

    # If source is a path string
    if isinstance(image_source, str):
        if not image_source or not os.path.exists(image_source):
            raise FileNotFoundError(f"Image path not found: {image_source}")
        try:
            return Image.open(image_source).convert("RGB")
        except Exception as e:
            raise IOError(f"Could not open image from path '{image_source}': {e}")

    raise TypeError("Unsupported image_source type. Pass a path, bytes, file-like, or PIL Image.")


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Convert PIL image to a normalized torch tensor of shape (1, C, H, W) on DEVICE.
    """
    if not isinstance(img, Image.Image):
        raise TypeError("pil_to_tensor requires a PIL Image")
    tensor = _preprocess(img).unsqueeze(0).to(DEVICE)
    return tensor


def load_model(model_path: Optional[str] = None, num_classes: int = 2) -> nn.Module:
    """
    Load a model. If model_path is None, use a pretrained resnet18 and adapt final layer.
    If model_path is provided, attempt to load state_dict into a resnet18 skeleton.
    """
    # Use a ResNet skeleton by default; attempt modern weights API first
    try:
        # torchvision >= 0.13 has ResNet18_Weights
        from torchvision.models import ResNet18_Weights  # type: ignore

        if model_path is None:
            weights = ResNet18_Weights.IMAGENET1K_V1
        else:
            weights = None
        model = models.resnet18(weights=weights)
    except Exception:
        # fallback for older torchvision versions that still use pretrained=...
        try:
            model = models.resnet18(pretrained=(model_path is None))
        except Exception as e:
            raise RuntimeError(f"Failed to create resnet18 model: {e}")

    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    model = model.to(DEVICE)

    if model_path:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        state = torch.load(model_path, map_location=DEVICE)
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        # strip 'module.' if present
        if isinstance(state_dict, dict) and any(k.startswith("module.") for k in state_dict.keys()):
            new_state = {}
            for k, v in state_dict.items():
                new_state[k.replace("module.", "")] = v
            state_dict = new_state
        # attempt to load safely
        try:
            model.load_state_dict(state_dict)
        except Exception:
            model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def predict_image(image_tensor: torch.Tensor, model: nn.Module) -> Tuple[int, np.ndarray, torch.Tensor]:
    """
    Run forward pass on single-image tensor and return predicted class index, softmax probs (numpy),
    and raw logits (tensor).
    Args:
        image_tensor: shape (1, C, H, W) on DEVICE
        model: torch.nn.Module on DEVICE
    Returns:
        pred_idx (int), probs (np.ndarray), logits (torch.Tensor)
    """
    if image_tensor.device != DEVICE:
        image_tensor = image_tensor.to(DEVICE)
    model = model.to(DEVICE)
    # model should be eval(); still use no_grad for safety in prediction wrapper
    with torch.no_grad():
        logits = model(image_tensor)  # shape (1, num_classes)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred_idx = int(torch.argmax(logits, dim=1).cpu().item())
    return pred_idx, probs, logits


def _find_module_by_name(model: nn.Module, layer_name: str) -> nn.Module:
    """
    Return the module corresponding to layer_name (dotted path or top-level name).
    Example: 'layer4' or 'layer4.1.conv2'
    """
    if not layer_name:
        raise ValueError("layer_name must be provided for Grad-CAM")
    parts = layer_name.split(".")
    module = model
    for p in parts:
        if hasattr(module, p):
            module = getattr(module, p)
        else:
            # try numeric index if module is a Sequential
            try:
                idx = int(p)
                module = module[idx]
            except Exception:
                raise AttributeError(f"Layer '{layer_name}' not found in model. Failed at '{p}'")
    return module


def _normalize_mask_to_0_1(mask: np.ndarray) -> np.ndarray:
    mn, mx = float(mask.min()), float(mask.max())
    if mx - mn > 1e-8:
        return (mask - mn) / (mx - mn)
    else:
        return np.zeros_like(mask)


def get_gradcam_mask(
    image_tensor: torch.Tensor,
    model: nn.Module,
    target_layer_name: str = TARGET_LAYER_NAME,
    target_class: Optional[int] = None,
) -> np.ndarray:
    """
    Return a Grad-CAM heatmap mask (H x W, numpy, values 0..1) for the input image_tensor.
    Tries Captum's LayerGradCam first; if not available, uses pytorch-grad-cam (GradCAMPlusPlus) if installed.
    image_tensor: torch.Tensor shape (1, C, H, W) on DEVICE
    """
    # Ensure model/tensor on DEVICE
    model = model.to(DEVICE)
    if image_tensor.device != DEVICE:
        image_tensor = image_tensor.to(DEVICE)

    if target_class is None:
        pred_idx, _, _ = predict_image(image_tensor, model)
        target_class = pred_idx

    # --- Option 1: Captum's LayerGradCam (preferred) ---
    if LayerGradCam is not None:
        target_layer = _find_module_by_name(model, target_layer_name)
        layer_gc = LayerGradCam(model, target_layer)

        # compute attributions (shape: 1 x C x H' x W')
        attributions = layer_gc.attribute(image_tensor, target=target_class)

        # detach before converting to numpy to avoid "requires_grad" issues
        at_np = attributions.detach().cpu().numpy()

        # sum across channels if multi-channel
        if at_np.ndim == 4:
            aggregated = np.sum(np.abs(at_np[0]), axis=0)
        elif at_np.ndim == 3:
            aggregated = np.sum(np.abs(at_np), axis=0)
        else:
            raise RuntimeError("Unexpected attribution shape from LayerGradCam: " + str(at_np.shape))

        heatmap = _normalize_mask_to_0_1(aggregated)

        # upsample to original IMG_SIZE if needed
        heatmap_resized = Image.fromarray(np.uint8(heatmap * 255)).resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
        heatmap_resized = np.asarray(heatmap_resized).astype(np.float32) / 255.0
        return heatmap_resized

    # --- Option 2: pytorch-grad-cam fallback (if installed) ---
    if GradCAMPlusPlus is not None and ClassifierOutputTarget is not None:
        # find the module object for the target layer (pytorch-grad-cam expects module object)
        target_layer = _find_module_by_name(model, target_layer_name)

        # cam expects cpu/cuda string or device param depending on version; we'll try both
        # prepare input tensor (pytorch-grad-cam expects numpy or tensor depending on version)
        # Use the model-normalized tensor; pytorch-grad-cam expects unnormalized image for some utilities,
        # but the cam core works with normalized tensors as well if provided as input_tensor.
        try:
            # try to construct with device argument (modern versions)
            try:
                cam = GradCAMPlusPlus(model=model, target_layers=[target_layer], device=DEVICE)
            except TypeError:
                # older versions may expect device as string 'cuda' or a use_cuda kwarg
                try:
                    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer], use_cuda=torch.cuda.is_available())
                except TypeError:
                    # final fallback: try string device
                    cam = GradCAMPlusPlus(model=model, target_layers=[target_layer], device=str(DEVICE))
        except Exception as e:
            raise RuntimeError(f"Failed to instantiate GradCAMPlusPlus from pytorch-grad-cam: {e}")

        # Build targets
        targets = [ClassifierOutputTarget(int(target_class))]

        # Run cam; many versions accept input_tensor= and return numpy mask
        try:
            cam_result = cam(input_tensor=image_tensor, targets=targets)  # returns array shape (B, H, W)
            if isinstance(cam_result, np.ndarray):
                mask = cam_result[0]
            else:
                # some versions return list/other - try to convert
                mask = np.array(cam_result)[0]
        except TypeError:
            # older API may be cam.compute_cam(input_tensor,..) or attribute; try attribute method
            try:
                mask = cam.attribute(image_tensor, targets=targets)
                if isinstance(mask, torch.Tensor):
                    mask = mask.detach().cpu().numpy()
                mask = mask[0]
            except Exception as e:
                raise RuntimeError(f"Failed to compute CAM with fallback API: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to compute CAM with pytorch-grad-cam: {e}")

        # Normalize and resize
        mask_norm = _normalize_mask_to_0_1(mask.astype(np.float32))
        heatmap_resized = Image.fromarray(np.uint8(mask_norm * 255)).resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
        heatmap_resized = np.asarray(heatmap_resized).astype(np.float32) / 255.0
        return heatmap_resized

    # If neither backend available, raise informative error
    raise ImportError(
        "No Grad-CAM backend available. Install captum (`pip install captum`) "
        "or pytorch-grad-cam (`pip install grad-cam`)."
    )


def get_integrated_gradients(
    image_tensor: torch.Tensor,
    model: nn.Module,
    target_class: Optional[int] = None,
    baseline: Optional[torch.Tensor] = None,
    n_steps: int = 50,
) -> np.ndarray:
    """
    Return Integrated Gradients attribution map as numpy array shape (H, W).
    Requires captum.
    """
    if IntegratedGradients is None:
        raise ImportError("captum is required for Integrated Gradients. Install with `pip install captum`")

    model = model.to(DEVICE)
    if image_tensor.device != DEVICE:
        image_tensor = image_tensor.to(DEVICE)

    if target_class is None:
        target_class, _, _ = predict_image(image_tensor, model)

    ig = IntegratedGradients(model)

    if baseline is None:
        baseline = torch.zeros_like(image_tensor).to(DEVICE)

    attributions, _ = ig.attribute(
        image_tensor,
        baselines=baseline,
        target=target_class,
        return_convergence_delta=True,
        n_steps=n_steps,
    )

    # detach before converting to numpy
    at_np = attributions.detach().cpu().numpy()  # (1, C, H, W)

    # Sum absolute across channels -> (H, W)
    agg = np.sum(np.abs(at_np[0]), axis=0)
    agg_norm = _normalize_mask_to_0_1(agg)

    agg_resized = Image.fromarray(np.uint8(agg_norm * 255)).resize((IMG_SIZE, IMG_SIZE), resample=Image.BILINEAR)
    agg_resized = np.asarray(agg_resized).astype(np.float32) / 255.0
    return agg_resized


def overlay_heatmap_on_pil(original_pil: Image.Image, heatmap: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """
    Overlay a heatmap (H x W, values 0..1) onto a PIL image and return combined PIL image.
    """
    if not isinstance(original_pil, Image.Image):
        raise ValueError("original_pil must be a PIL Image")

    if heatmap.ndim != 2:
        raise ValueError("heatmap must be 2D array")

    orig_resized = original_pil.resize((heatmap.shape[1], heatmap.shape[0])).convert("RGBA")
    heat_uint8 = np.uint8(heatmap * 255)

    heat_rgba = np.zeros((heatmap.shape[0], heatmap.shape[1], 4), dtype=np.uint8)
    # red channel
    heat_rgba[..., 0] = heat_uint8
    # alpha scaled by requested alpha
    heat_rgba[..., 3] = np.uint8(heatmap * 255 * alpha)
    heat_pil = Image.fromarray(heat_rgba, mode="RGBA")
    combined = Image.alpha_composite(orig_resized, heat_pil)
    return combined.convert("RGB")


def explain_image_with_models(
    image_source: Union[str, Image.Image, IO, bytes],
    model: Optional[nn.Module] = None,
    model_path: Optional[str] = None,
    use_gradcam: bool = True,
    use_ig: bool = True,
    target_layer_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    High-level pipeline:
     - loads model (if not provided)
     - loads image (path / PIL / buffer / bytes), preprocess
     - returns prediction, probs, and explanations (PIL overlays and raw masks)
    Returns a dict with keys:
     - 'pred_idx', 'probs', 'logits', 'original_pil', 'gradcam_mask', 'gradcam_overlay',
       'ig_mask', 'ig_overlay'
    Some keys may be None if captum missing or use flags false.
    """
    if model is None:
        model = load_model(model_path or MODEL_PATH)
    target_layer_name = target_layer_name or TARGET_LAYER_NAME

    original = load_image(image_source)
    tensor = pil_to_tensor(original)

    pred_idx, probs, logits = predict_image(tensor, model)

    result: Dict[str, Any] = {
        "pred_idx": pred_idx,
        "probs": probs,
        "logits": logits,
        "original_pil": original,
        "gradcam_mask": None,
        "gradcam_overlay": None,
        "ig_mask": None,
        "ig_overlay": None,
    }

    if use_gradcam:
        try:
            mask_gc = get_gradcam_mask(tensor, model, target_layer_name, target_class=pred_idx)
            result["gradcam_mask"] = mask_gc
            result["gradcam_overlay"] = overlay_heatmap_on_pil(original, mask_gc, alpha=0.5)
        except Exception as e:
            result["gradcam_error"] = str(e)

    if use_ig:
        try:
            mask_ig = get_integrated_gradients(tensor, model, target_class=pred_idx)
            result["ig_mask"] = mask_ig
            result["ig_overlay"] = overlay_heatmap_on_pil(original, mask_ig, alpha=0.5)
        except Exception as e:
            result["ig_error"] = str(e)

    return result


# ---------- CLI test entry point ----------
def _cli_test(image_path: str):
    print("Device:", DEVICE)
    model = load_model(MODEL_PATH)
    print("Model loaded.")
    result = explain_image_with_models(image_path, model=model, use_gradcam=True, use_ig=True)
    print("Predicted class:", result["pred_idx"])
    print("Probabilities:", result["probs"])
    out_dir = os.path.abspath("xai_outputs")
    os.makedirs(out_dir, exist_ok=True)
    if result.get("gradcam_overlay") is not None:
        p = os.path.join(out_dir, "gradcam_overlay.jpg")
        result["gradcam_overlay"].save(p)
        print("Saved Grad-CAM overlay to", p)
    else:
        print("No Grad-CAM overlay generated.", result.get("gradcam_error", ""))

    if result.get("ig_overlay") is not None:
        p = os.path.join(out_dir, "ig_overlay.jpg")
        result["ig_overlay"].save(p)
        print("Saved IG overlay to", p)
    else:
        print("No Integrated Gradients overlay generated.", result.get("ig_error", ""))


# ----------------- Improved CLI entrypoint -----------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="xai_extended CLI - run explainers on an image")
    parser.add_argument(
        "--img",
        "-i",
        required=False,
        help="Path to input image (jpg/png). If omitted, the script will exit with usage instructions.",
    )
    parser.add_argument(
        "--model",
        "-m",
        required=False,
        help="Optional path to model checkpoint (.pth/.pt). If omitted, a pretrained ResNet18 skeleton is used.",
    )
    parser.add_argument("--no-gradcam", action="store_true", help="Disable Grad-CAM computation.")
    parser.add_argument("--no-ig", action="store_true", help="Disable Integrated Gradients computation.")
    parser.add_argument("--outdir", "-o", default="xai_outputs", help="Directory to save overlays.")
    args = parser.parse_args()

    if not args.img:
        print("No image provided. Example usage:")
        print("  python -m scripts.xai_extended --img C:\\path\\to\\image.jpg")
        print("  python scripts\\xai_extended.py -i C:\\path\\to\\image.jpg --model C:\\path\\to\\model.pth")
        sys.exit(1)

    image_path = args.img
    model_path = args.model or MODEL_PATH

    try:
        # Load model once, pass into explain pipeline
        model = load_model(model_path)
    except Exception as e:
        print("Error loading model:", e)
        print("Continuing with model=None (will attempt to use default ResNet skeleton).")
        model = None

    use_gradcam = not args.no_gradcam
    use_ig = not args.no_ig

    print("Device:", DEVICE)
    print("Image:", image_path)
    print(f"Model path: {model_path or 'None (using pretrained skeleton)'}")
    print(f"Grad-CAM: {'ON' if use_gradcam else 'OFF'} | IntegratedGradients: {'ON' if use_ig else 'OFF'}")

    try:
        result = explain_image_with_models(
            image_path,
            model=model,
            model_path=model_path,
            use_gradcam=use_gradcam,
            use_ig=use_ig,
        )
    except Exception as e:
        print("Error during explanation pipeline:", e)
        raise

    out_dir = os.path.abspath(args.outdir)
    os.makedirs(out_dir, exist_ok=True)

    if result.get("gradcam_overlay") is not None:
        p = os.path.join(out_dir, "gradcam_overlay.jpg")
        result["gradcam_overlay"].save(p)
        print("Saved Grad-CAM overlay to", p)
    else:
        print("No Grad-CAM overlay generated.", result.get("gradcam_error", ""))

    if result.get("ig_overlay") is not None:
        p = os.path.join(out_dir, "ig_overlay.jpg")
        result["ig_overlay"].save(p)
        print("Saved IG overlay to", p)
    else:
        print("No Integrated Gradients overlay generated.", result.get("ig_error", ""))
