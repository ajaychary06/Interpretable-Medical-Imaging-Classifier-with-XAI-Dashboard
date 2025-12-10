# app/streamlit_app.py
import streamlit as st
from pathlib import Path
import importlib.util
import io
from PIL import Image
import tempfile
import os
import sys
from typing import Optional

# --- load local xai_extended module safely (works regardless of how Streamlit is launched) ---
SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
XAI_PATH = SCRIPTS_DIR / "xai_extended.py"
if not XAI_PATH.exists():
    raise FileNotFoundError(f"Expected {XAI_PATH} to exist. Put xai_extended.py into scripts/")

spec = importlib.util.spec_from_file_location("scripts.xai_extended", str(XAI_PATH))
xai_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(xai_mod)

# functions we expect
load_model_fn = getattr(xai_mod, "load_model", None)
explain_image_with_models = getattr(xai_mod, "explain_image_with_models", None)
if explain_image_with_models is None:
    raise RuntimeError("explain_image_with_models not found in scripts/xai_extended.py")

# --- UI configuration ---
st.set_page_config(page_title="Brain Tumor XAI Demo", layout="wide")
st.title("Brain Tumor Classifier — XAI Demo")
st.markdown("Upload a brain MRI slice (jpg/png) or pick an example. The app shows prediction, probabilities, Grad-CAM and Integrated Gradients overlays.")

PROJECT_ROOT = Path.cwd()
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"
OUT_DIR = PROJECT_ROOT / "outputs" / "streamlit_xai"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sidebar: model & options
with st.sidebar:
    st.header("Options")
    use_gradcam = st.checkbox("Use Grad-CAM", True)
    use_ig = st.checkbox("Use Integrated Gradients", True)
    st.write("---")
    st.write("Model checkpoint:")
    if CHECKPOINT_PATH.exists():
        st.success(f"Found: {CHECKPOINT_PATH.name}")
    else:
        st.info("No checkpoint found. Using default pretrained skeleton.")

# Cached model loader (so model loads once)
@st.cache_resource
def _load_model(checkpoint_path: Optional[str]):
    try:
        if load_model_fn is None:
            return None
        # load_model accepts None or path
        model = load_model_fn(checkpoint_path)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

MODEL = _load_model(str(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else None)

# Helper to read uploaded file to PIL and also provide a temp path if needed
def _pil_from_upload(uploaded) -> Image.Image:
    data = uploaded.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return img, data

# Build list of example images from processed/test (yes/no)
def _get_examples():
    ex = []
    test_dir = PROJECT_ROOT / "data" / "processed" / "test"
    if test_dir.exists():
        for cls in sorted([p.name for p in test_dir.iterdir() if p.is_dir()]):
            for f in (test_dir / cls).glob("*.*"):
                ex.append((f"{cls}/{f.name}", str(f)))
    return ex

examples = _get_examples()
example_labels = ["None"] + [e[0] for e in examples]

# Main UI: upload or pick example
col_left, col_right = st.columns([1, 2])
with col_left:
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"])
    selected = st.selectbox("Or pick an example", example_labels)
    run = st.button("Run XAI")

with col_right:
    result_area = st.empty()
    outputs_cols = st.columns(3)
    orig_area = outputs_cols[0].empty()
    gc_area = outputs_cols[1].empty()
    ig_area = outputs_cols[2].empty()
    info_area = st.empty()

# If run pressed, prepare input and call explain pipeline
if run:
    image_source = None
    pil_img = None
    # prefer uploaded image
    if uploaded is not None:
        try:
            pil_img, uploaded_bytes = _pil_from_upload(uploaded)
            image_source = uploaded_bytes  # explain_image_with_models accepts bytes
        except Exception as e:
            st.error(f"Could not read uploaded image: {e}")
            image_source = None
    elif selected != "None":
        # find matching path
        sel_idx = example_labels.index(selected) - 1
        example_path = examples[sel_idx][1]
        try:
            pil_img = Image.open(example_path).convert("RGB")
            image_source = example_path
        except Exception as e:
            st.error(f"Could not open example {selected}: {e}")
            image_source = None
    else:
        st.info("Please upload an image or choose an example.")
        image_source = None

    if image_source is not None:
        # show original
        try:
            orig_area.image(pil_img, caption="Original image", use_column_width=True)
        except Exception:
            # fallback: show via bytes
            try:
                if isinstance(image_source, (bytes, bytearray)):
                    orig_area.image(Image.open(io.BytesIO(image_source)))
                else:
                    orig_area.image(str(image_source))
            except Exception:
                orig_area.text("Original image (could not render)")

        info_area.info("Running explainers — this can take a few seconds.")

        # run explain pipeline (safe try/except)
        try:
            res = explain_image_with_models(image_source, model=MODEL, use_gradcam=use_gradcam, use_ig=use_ig)
        except Exception as e:
            st.error(f"Error during explain pipeline: {e}")
            res = {}

        # show prediction & probs
        pred_idx = res.get("pred_idx", None)
        probs = res.get("probs", None)
        label_map = {0: "no tumor", 1: "tumor"}
        pred_label = label_map.get(pred_idx, str(pred_idx)) if pred_idx is not None else "N/A"
        result_area.markdown(f"### Prediction: **{pred_label}**")
        if probs is not None:
            result_area.markdown(f"**Probabilities:** {probs.tolist()}")

        # collect overlays and save them into OUT_DIR with friendly names
        saved_files = {}
        if res.get("gradcam_overlay") is not None:
            p = OUT_DIR / "streamlit_gradcam.jpg"
            try:
                res["gradcam_overlay"].save(p)
                saved_files["Grad-CAM"] = p
                gc_area.image(res["gradcam_overlay"], caption="Grad-CAM", use_column_width=True)
                buf = io.BytesIO()
                res["gradcam_overlay"].save(buf, format="JPEG")
                st.download_button("Download Grad-CAM", data=buf.getvalue(), file_name=p.name, mime="image/jpeg")
            except Exception as e:
                gc_area.error(f"Failed to save/display Grad-CAM: {e}")
        else:
            gc_area.text("Grad-CAM (not generated)")

        if res.get("ig_overlay") is not None:
            p = OUT_DIR / "streamlit_ig.jpg"
            try:
                res["ig_overlay"].save(p)
                saved_files["IntegratedGradients"] = p
                ig_area.image(res["ig_overlay"], caption="Integrated Gradients", use_column_width=True)
                buf = io.BytesIO()
                res["ig_overlay"].save(buf, format="JPEG")
                st.download_button("Download IG", data=buf.getvalue(), file_name=p.name, mime="image/jpeg")
            except Exception as e:
                ig_area.error(f"Failed to save/display IG: {e}")
        else:
            ig_area.text("Integrated Gradients (not generated)")

        # show any errors returned
        if res.get("gradcam_error"):
            st.error(f"Grad-CAM error: {res.get('gradcam_error')}")
        if res.get("ig_error"):
            st.error(f"IG error: {res.get('ig_error')}")

        st.success(f"Saved overlays to: {OUT_DIR}")

# Show gallery of precomputed overlays (from training run)
st.markdown("---")
st.markdown("## Gallery — saved XAI overlays")
gallery_dir = PROJECT_ROOT / "outputs" / "xai_demo_from_train"
if gallery_dir.exists():
    imgs = sorted(list(gallery_dir.glob("*.*")))
    if imgs:
        cols = st.columns(4)
        for i, img_path in enumerate(imgs):
            with cols[i % 4]:
                try:
                    st.image(str(img_path), caption=img_path.name, use_column_width=True)
                    with open(img_path, "rb") as fh:
                        st.download_button("Download", data=fh.read(), file_name=img_path.name, key=f"dl_{i}")
                except Exception as e:
                    st.text(f"Failed to load {img_path.name}: {e}")
    else:
        st.info("No saved overlays found in outputs/xai_demo_from_train.")
else:
    st.info("No gallery directory found. Run train_eval_xai.py to generate overlays.")
