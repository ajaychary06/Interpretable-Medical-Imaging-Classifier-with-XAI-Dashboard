# scripts/make_report_from_examples.py
from pathlib import Path, PurePosixPath

OUT = Path('outputs') / 'xai_visualizations'
SRC = OUT / 'for_report'
INDEX = SRC / 'index.html'

html = []
html.append('<html><head><meta charset="utf-8"><title>XAI Example Report</title>')
html.append('<style>')
html.append('img { width: 260px; border:1px solid #ccc; margin:10px; }')
html.append('.folder { margin-bottom:40px; }')
html.append('</style></head><body>')
html.append('<h1>XAI Example Report (TP/TN/FP/FN)</h1>')

for folder in sorted(SRC.iterdir()):
    if not folder.is_dir():
        continue

    html.append(f'<div class="folder"><h2>{folder.name}</h2>')

    for fname in ["xai_smoke_campp.jpg", "xai_smoke_ig.jpg", "xai_smoke_fused.jpg"]:
        img_file = folder / fname
        if img_file.exists():
            rel = fname  # relative path inside the same folder
            html.append(f'<img src="{folder.name}/{rel}" alt="{fname}">')
        else:
            html.append(f'<div style="width:260px;height:260px;background:#eee;display:inline-block;margin:10px;">Missing {fname}</div>')

    html.append('</div>')

html.append('</body></html>')

INDEX.write_text("\n".join(html), encoding="utf-8")
print("Wrote", INDEX)
