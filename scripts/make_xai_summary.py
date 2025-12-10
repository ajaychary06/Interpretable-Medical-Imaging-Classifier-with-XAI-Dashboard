# scripts/make_xai_summary.py
import csv
from pathlib import Path
import re

ROOT = Path('.').resolve()
OUT_ROOT = ROOT / 'outputs' / 'xai_visualizations'
PROCESSED = ROOT / 'data' / 'processed' / 'test'

# Build mapping basename -> true label by scanning processed/test/yes and /no
label_map = {}
for lab in ['yes', 'no']:
    d = PROCESSED / lab
    if d.exists():
        for p in d.iterdir():
            if p.is_file():
                label_map[p.stem] = lab

rows = []
for folder in sorted([p for p in OUT_ROOT.iterdir() if p.is_dir()]):
    basename = folder.name
    stdout = folder / 'stdout.txt'
    pred_label = ''
    pred_prob = ''
    pred_idx = ''
    if stdout.exists():
        txt = stdout.read_text(encoding='utf-8', errors='ignore')
        m = re.search(r'Predicted index:\s*(\d+).*label:\s*([a-zA-Z]+).*probability:\s*([0-9.]+)', txt)
        if m:
            pred_idx = m.group(1)
            pred_label = m.group(2)
            pred_prob = m.group(3)
        else:
            # fallback: find any 'Predicted' line
            m2 = re.search(r'Predicted.*', txt)
            pred_label = m2.group(0) if m2 else ''
    true_label = label_map.get(basename, '')
    rows.append((basename, true_label, pred_idx, pred_label, pred_prob, str(stdout)))

# write CSV
out_csv = OUT_ROOT / 'xai_summary.csv'
with open(out_csv, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['basename','true_label','pred_idx','pred_label','pred_prob','stdout_path'])
    writer.writerows(rows)

print('Wrote', out_csv)
