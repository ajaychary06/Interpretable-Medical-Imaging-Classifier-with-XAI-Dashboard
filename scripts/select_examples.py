# scripts/select_examples.py
import csv
from pathlib import Path
OUT_ROOT = Path('outputs') / 'xai_visualizations'
csvf = OUT_ROOT / 'xai_summary.csv'
sel_root = OUT_ROOT / 'for_report'
sel_root.mkdir(parents=True, exist_ok=True)
rows = []
with open(csvf, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# classify rows
tp = []; tn = []; fp = []; fn = []
for r in rows:
    t = r['true_label'].lower() if r['true_label'] else ''
    p = r['pred_label'].lower() if r['pred_label'] else ''
    try:
        prob = float(r['pred_prob']) if r['pred_prob'] else 0.0
    except:
        prob = 0.0
    if not t:
        continue
    if t == 'yes' and p == 'yes':
        tp.append((prob, r))
    elif t == 'no' and p == 'no':
        tn.append((prob, r))
    elif t == 'no' and p == 'yes':
        fp.append((prob, r))
    elif t == 'yes' and p == 'no':
        fn.append((prob, r))

# sort by probability (descending for TP/FP to get confident examples, ascending for FN for weakest)
tp = sorted(tp, key=lambda x: -x[0])
tn = sorted(tn, key=lambda x: -x[0])
fp = sorted(fp, key=lambda x: -x[0])
fn = sorted(fn, key=lambda x: x[0])

# pick up to 3 examples each
chosen = []
for cat, arr in [('TP', tp), ('TN', tn), ('FP', fp), ('FN', fn)]:
    for prob, r in arr[:3]:
        src = OUT_ROOT / r['basename']
        # copy available images (campp, ig, fused, orig) into sel_root/<cat>_<basename>/
        dst = sel_root / f"{cat}_{r['basename']}"
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.suffix.lower() in ('.jpg', '.jpeg', '.png'):
                (dst / f.name).write_bytes(f.read_bytes())
        chosen.append((cat, r['basename'], r['true_label'], r['pred_label'], r['pred_prob']))
# write summary
with open(OUT_ROOT / 'for_report' / 'chosen_examples.csv', 'w', encoding='utf-8') as f:
    f.write('category,basename,true,pred,prob\\n')
    for c in chosen:
        f.write(','.join(map(str,c)) + '\\n')
print('Selected examples copied to', OUT_ROOT / 'for_report')
