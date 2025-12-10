# scripts/generate_xai_report.py
"""
Generate a simple HTML review page for XAI visualizations.
Produces: outputs/xai_visualizations/index.html
"""

import re
from pathlib import Path
import html
import sys

ROOT = Path('.').resolve()
OUT_ROOT = ROOT / 'outputs' / 'xai_visualizations'
INDEX_HTML = OUT_ROOT / 'index.html'

if not OUT_ROOT.exists():
    print(f"No outputs found at {OUT_ROOT}. Run XAI pipeline first.")
    sys.exit(1)

folders = sorted([p for p in OUT_ROOT.iterdir() if p.is_dir()])

rows = []
for folder in folders:
    # find candidate images
    cam = None
    ig = None
    fused = None
    orig = None
    for name in folder.iterdir():
        ln = name.name.lower()
        if 'cam' in ln and name.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            cam = name
        if 'ig' in ln and name.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            ig = name
        if 'fused' in ln and name.suffix.lower() in ('.jpg', '.jpeg', '.png'):
            fused = name
        # heuristics for original image
        if name.suffix.lower() in ('.jpg', '.jpeg', '.png') and ('xai_smoke' not in ln):
            # pick the first image that is not xai_smoke_*
            if orig is None:
                orig = name

    # parse prediction from stdout.txt
    pred_line = ''
    stdout = folder / 'stdout.txt'
    if stdout.exists():
        txt = stdout.read_text(encoding='utf-8', errors='ignore')
        m = re.search(r'Predicted index:.*', txt)
        if m:
            pred_line = m.group(0)
        else:
            # try another common pattern
            m2 = re.search(r'Prediction.*', txt)
            if m2:
                pred_line = m2.group(0)
    else:
        pred_line = ''

    rows.append({
        'folder': folder.name,
        'orig': orig.name if orig else '',
        'cam': cam.name if cam else '',
        'ig': ig.name if ig else '',
        'fused': fused.name if fused else '',
        'pred': pred_line,
    })

# Build HTML
html_parts = []
html_parts.append('<!doctype html>')
html_parts.append('<html><head><meta charset="utf-8"><title>XAI Visualizations</title>')
html_parts.append('<style>body{font-family:Arial,Helvetica,sans-serif;} .card{display:inline-block;border:1px solid #ddd;padding:8px;margin:8px;width:360px;vertical-align:top;} img{max-width:100%;height:auto;border:1px solid #ccc;} .meta{font-size:13px;color:#333;margin-top:6px;}</style>')
html_parts.append('</head><body>')
html_parts.append(f'<h2>XAI Visualizations — {len(rows)} items</h2>')
html_parts.append('<p>Columns: Original | Grad-CAM | Integrated Gradients | Fused — and model prediction</p>')
html_parts.append('<div>')

for r in rows:
    folder_rel = f'./{html.escape(r["folder"])}/'
    html_parts.append('<div class="card">')
    html_parts.append(f'<strong>{html.escape(r["folder"])}</strong><div class="meta">{html.escape(r["pred"])}</div>')
    # original
    if r['orig']:
        html_parts.append(f'<div><em>Original</em><br><img src="{html.escape(folder_rel + r["orig"])}" alt="orig"></div>')
    else:
        html_parts.append('<div><em>Original</em><br><div style="width:320px;height:180px;background:#f5f5f5;display:flex;align-items:center;justify-content:center;color:#999">no original</div></div>')
    # cam
    html_parts.append('<div><em>Grad-CAM</em><br>')
    if r['cam']:
        html_parts.append(f'<img src="{html.escape(folder_rel + r["cam"])}" alt="cam">')
    else:
        html_parts.append('<div style="width:320px;height:120px;background:#f5f5f5;display:flex;align-items:center;justify-content:center;color:#999">no cam</div>')
    html_parts.append('</div>')
    # ig
    html_parts.append('<div><em>Integrated Gradients</em><br>')
    if r['ig']:
        html_parts.append(f'<img src="{html.escape(folder_rel + r["ig"])}" alt="ig">')
    else:
        html_parts.append('<div style="width:320px;height:120px;background:#f5f5f5;display:flex;align-items:center;justify-content:center;color:#999">no ig</div>')
    html_parts.append('</div>')
    # fused
    html_parts.append('<div><em>Fused</em><br>')
    if r['fused']:
        html_parts.append(f'<img src="{html.escape(folder_rel + r["fused"])}" alt="fused">')
    else:
        html_parts.append('<div style="width:320px;height:120px;background:#f5f5f5;display:flex;align-items:center;justify-content:center;color:#999">no fused</div>')
    html_parts.append('</div>')

    html_parts.append('</div>')  # card end

html_parts.append('</div></body></html>')

INDEX_HTML.write_text('\n'.join(html_parts), encoding='utf-8')
print(f'Wrote: {INDEX_HTML}')
print('Open with your browser (double-click the file).')
