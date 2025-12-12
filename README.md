# Interpretable-Medical-Imaging-Classifier-with-XAI-Dashboard

### Project: XAI Brain Tumor Detection
### Author: Ajaychary Kandukuri
### Repo: xai_brain_tumor

## Short description:
#### End-to-end pipeline that trains deep learning models to classify brain tumors from MRI scans and provides explainable AI outputs (Grad-CAM / Grad-CAM++ / saliency + visual overlays) so clinicians and stakeholders can understand model decisions. Includes training, evaluation, inference, and a Streamlit demo for interactive visualization.

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
```

## Project layout (recommended)

```
xai_brain_tumor/
├─ data/
│  ├─ train/
│  ├─ val/
│  └─ test/
├─ models/
│  └─ checkpoints/
├─ outputs/
│  └─ streamlit_xai/
├─ scripts/
│  ├─ train_eval_xai.py       # training + evaluation orchestration
│  ├─ inference.py            # single-image & batch inference
│  ├─ xai_methods.py          # Grad-CAM, Grad-CAM++, saliency utilities
│  ├─ utils.py                # helper functions (transforms, metrics)
│  └─ export_visuals.py       # save visual overlays for reports
├─ app/
│  └─ streamlit_app.py        # interactive demo
├─ notebooks/
│  └─ Final_Report.ipynb
├─ requirements.txt
└─ README.md
```

## Common commands
### Prepare data

(Place MRI image folders under data/train, data/val, data/test with subfolders per class)


```
# Example: view dataset sizes
python - <<'PY'
from pathlib import Path
p = Path('data')
for split in ['train','val','test']:
    s = p / split
    print(split, sum(1 for _ in s.rglob('*.png')), "images")
PY
```

## Train a model

```
# Train with default config
python scripts/train_eval_xai.py --config configs/train_resnet.yaml

# Or run with CLI args (example)
python scripts/train_eval_xai.py \
  --epochs 30 \
  --batch-size 16 \
  --lr 1e-4 \
  --data-root ./data \
  --save-dir ./models/checkpoints

```

## Evaluate / run inference on a checkpoint

```
# Single-image inference (prints predicted class + probability)
python scripts/inference.py --checkpoint models/checkpoints/best.pth --image tests/sample1.png

# Batch inference producing CSV results + XAI overlays
python scripts/inference.py --checkpoint models/checkpoints/best.pth --input-dir ./data/test --output-dir ./outputs/streamlit_xai --xai-method gradcampp

```

## Generate XAI visualizations (standalone)

```
python scripts/export_visuals.py --checkpoint models/checkpoints/best.pth --input-dir data/test --out visuals/
```

## Run Streamlit demo (interactive)

```
# From repository root
streamlit run app/streamlit_app.py --server.port 8501
# Then open browser: http://localhost:8501
```

## Configuration

Use `configs/*.yaml (or .json)` to manage model hyperparameters, transforms, and dataset paths.

Example fields:

- model: resnet50

- input_size: 224

- batch_size: 16

- optimizer: adam

- lr: 1e-4

- augmentations: True / False


## Training tips & reproducibility

- Set deterministic seeds in `train_eval_xai.py` (seed value configurable).

- Use mixed precision if available (AMP) to speed training.

- Monitor logs with TensorBoard `(tensorboard --logdir runs/)`.

- Save best checkpoints by validation metric (e.g., F1 or AUC).

## Evaluation & metrics

- Scripts compute: Accuracy, Precision, Recall, F1-score, Confusion Matrix, AUC (if applicable).

- Visual artifacts saved per-sample: original, model heatmap overlay, guided backprop overlay.

- Example: `outputs/streamlit_xai` contains per-image JSON metadata + PNG overlays ready for reports.

## Troubleshooting

### Grad-CAM or Grad-CAM++ errors:
- Verify matching versions of `torch` and `torchvision`.
- If third-party XAI lib signature changed (e.g., `use_cuda` removed), update calls in `xai_methods.py`.

### "conda not recognized":
- Ensure Anaconda is installed and added to PATH, or use the Anaconda/Miniconda prompt.

### CUDA / GPU:
```python
import torch
torch.cuda.is_available()
```
- If `False`, install correct CUDA toolkit or use CPU mode.

## Notebooks & Final Report

- `notebooks/Final_Report.ipynb` contains EDA, training curves, selected results, and methodology notes.

- Use exported overlays (Grad-CAM / Grad-CAM++ outputs) in the notebook when creating figures for the final report.

## Contributing

- Create an issue for bugs or feature requests.

- For code updates: fork → create feature branch → submit PR with a clear description.

- Keep changes modular (data pipeline, model code, XAI utilities should remain separate).

## License

MIT License

## Credits & References

- Built with PyTorch, torchvision, Streamlit, and custom XAI utilities.

- Inspired by best practices in medical imaging, explainable AI, and deep learning engineering.

Author / Contact

Ajaychary Kandukuri
Email: ajaycharykandukuri06@gmail.com
