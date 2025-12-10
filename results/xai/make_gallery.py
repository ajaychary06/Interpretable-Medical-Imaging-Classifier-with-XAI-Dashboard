# results/xai/make_gallery.py
from pathlib import Path
d = Path(__file__).resolve().parent
imgs = sorted([p for p in d.glob("*_combined.png")])
html_lines = [
    "<!doctype html>",
    "<html><head><meta charset='utf-8'><title>XAI Gallery</title></head><body>",
    "<h1>XAI Gallery</h1>"
]
for p in imgs:
    file_url = "file:///" + str(p.resolve()).replace("\\", "/")
    html_lines.append(f'<div style="margin:12px 0"><h4>{p.name}</h4>')
    html_lines.append(f'<img src="{file_url}" style="max-width:95%; width:800px; display:block; border:1px solid #ccc; padding:4px; background:#fff" />')
    html_lines.append("</div>")
html_lines.append("</body></html>")
out = d / "gallery_fixed.html"
out.write_text("\n".join(html_lines), encoding="utf-8")
print("Wrote:", out)
