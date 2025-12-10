# scripts/make_xai_montages.py
from pathlib import Path
from PIL import Image, ImageOps
import os

OUT = Path("xai_out_test")
OUT_MONT = Path("xai_out_test_montages")
OUT_MONT.mkdir(exist_ok=True)

def find_versions(stem):
    items = list(OUT.glob(f"{stem}*"))
    # prefer patterns ending with _gradcam, _guided_gradcam
    gcam = None; ggrad=None; orig=None
    for p in items:
        name = p.name.lower()
        if "_gradcam" in name and p.suffix.lower() in [".jpg",".png"]:
            gcam = p
        if "_guided_gradcam" in name and p.suffix.lower() in [".jpg",".png"]:
            ggrad = p
        # original file might not have suffix _orig; find any original-size image without those tags
        if ("gradcam" not in name) and ("guided" not in name) and p.suffix.lower() in [".jpg",".png"]:
            orig = p
    return orig, gcam, ggrad

# gather unique stems (before first space or underscore)
stems = set()
for p in OUT.iterdir():
    if p.suffix.lower() not in [".jpg",".png"]:
        continue
    # use full stem without suffix
    s = p.stem
    # normalize: remove trailing suffix tags like _gradcam etc
    for tag in ["_gradcam","_guided_gradcam","_guided","_smoothgrad","_pred"]:
        if s.endswith(tag):
            s = s[: -len(tag)]
    stems.add(s)

for stem in sorted(stems):
    orig,gcam,ggrad = find_versions(stem)
    to_open = []
    # open original (if exists) else try fallback like stem + ".jpg"
    if orig and orig.exists():
        to_open.append(Image.open(orig).convert("RGB"))
    else:
        # try common fallback
        fallback = OUT / (stem + ".jpg")
        if fallback.exists():
            to_open.append(Image.open(fallback).convert("RGB"))
        else:
            # skip if no original
            continue
    # gradcam
    if gcam and gcam.exists():
        to_open.append(Image.open(gcam).convert("RGB"))
    else:
        # placeholder
        to_open.append(Image.new("RGB", to_open[0].size, (255,255,255)))
    # guided gradcam
    if ggrad and ggrad.exists():
        to_open.append(Image.open(ggrad).convert("RGB"))
    else:
        to_open.append(Image.new("RGB", to_open[0].size, (255,255,255)))

    # resize all to same height
    widths, heights = zip(*(im.size for im in to_open))
    maxh = max(heights)
    resized = [ImageOps.fit(im, (int(maxh* (im.width/im.height)), maxh)) for im in to_open]

    # stitch horizontally
    total_w = sum(im.width for im in resized)
    outimg = Image.new("RGB", (total_w, maxh))
    x=0
    for im in resized:
        outimg.paste(im, (x,0))
        x += im.width

    outpath = OUT_MONT / f"{stem}_montage.jpg"
    outimg.save(outpath)
    print("Saved montage:", outpath)
