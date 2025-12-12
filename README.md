# Interpretable-Medical-Imaging-Classifier-with-XAI-Dashboard

# README.md
# Explainable AI — Brain Tumor Classification & XAI Visualizations
#
# Project: XAI Brain Tumor Detection
# Author: Ajaychary Kandukuri
# Repo: xai_brain_tumor
#
# Short description:
# End-to-end pipeline that trains deep learning models to classify brain tumors from MRI scans
# and provides explainable AI outputs (Grad-CAM / Grad-CAM++ / saliency + visual overlays) so
# clinicians and stakeholders can understand model decisions. Includes training, evaluation,
# inference, and a Streamlit demo for interactive visualization.

---
## Key features
- Preprocessing & dataset loaders for MRI images (train / val / test)
- PyTorch training scripts with checkpointing and reproducible config
- Multiple XAI methods (Grad-CAM, Grad-CAM++, saliency maps, guided backprop)
- Streamlit-based visualization app to view predictions + explanations
- Utility scripts: inference, batch eval, metrics (accuracy, F1, confusion matrix)
- Exportable outputs for reports & PowerPoint-ready images

---
## Quick start (Windows example)
> From project root (example path used in development):
> `C:\Users\ajayc\Documents\xai_brain_tumor`

```bash
# 1) Create & activate conda environment (recommended)
conda create -n xai_proj python=3.10 -y
conda activate xai_proj

# 2) Install dependencies
# Prefer using the provided requirements file (pip) or environment.yml if available
pip install -r requirements.txt

# If you prefer conda (if environment.yml exists):
# conda env create -f environment.yml
# conda activate xai_proj
