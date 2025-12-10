# scripts/run_xai_pipeline.py
"""
Run the smoke_xai_extra.py script over a folder of images and collect outputs.

Usage (from project root):
  conda activate xai_proj
  python scripts/run_xai_pipeline.py --img_dir data/processed/test/yes --checkpoint checkpoints/best_model.pth --num_classes 2 --device cpu

Notes:
- This wrapper assumes your smoke script accepts:
    --img <image_path>
    --checkpoint <checkpoint_path>
    --num-classes <int>
    --device <cpu|cuda>
- The wrapper collects files that include the image basename and also any 'xai_smoke_*' files the smoke script produces.
"""

import argparse
import subprocess
import shutil
import sys
import time
import logging
from pathlib import Path

def run_command(cmd, cwd=None, timeout=None):
    try:
        completed = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
        return completed.returncode, completed.stdout, completed.stderr
    except Exception as e:
        return -1, "", str(e)

def collect_and_cleanup(project_root: Path, basename: str, image_out_dir: Path):
    """
    Copy files that match basename and xai_smoke_* into image_out_dir.
    Remove xai_smoke_* from project root to avoid mixing with next image.
    Returns True if any file was copied.
    """
    moved_any = False

    # 1) Collect any files that include the basename
    for f in project_root.rglob('*'):
        if f.is_file() and basename in f.name:
            try:
                shutil.copy2(f, image_out_dir / f.name)
                moved_any = True
            except Exception as e:
                logging.warning(f'Failed to copy {f} -> {image_out_dir}: {e}')

    # 2) Also collect any xai_smoke_* files in project root (or in project_root / outputs)
    # Search common locations
    candidates = list(project_root.glob('xai_smoke_*'))
    candidates += list((project_root / 'outputs').glob('xai_smoke_*')) if (project_root / 'outputs').exists() else []
    for f in candidates:
        if f.is_file():
            try:
                shutil.copy2(f, image_out_dir / f.name)
                moved_any = True
            except Exception as e:
                logging.warning(f'Failed to copy {f} -> {image_out_dir}: {e}')
            # attempt to remove the temporary file so it doesn't affect next image
            try:
                f.unlink()
            except Exception:
                pass

    return moved_any

def main(args):
    project_root = Path('.').resolve()
    img_dir = Path(args.img_dir)
    checkpoint = Path(args.checkpoint)
    out_root = Path(args.out_dir)
    smoke_script = Path('scripts') / 'smoke_xai_extra.py'

    out_root.mkdir(parents=True, exist_ok=True)
    log_file = out_root / 'run_xai_pipeline.log'
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file, mode='a')])

    logging.info('Run XAI pipeline started')
    logging.info(f'Project root: {project_root}')
    logging.info(f'Image dir: {img_dir}')
    logging.info(f'Checkpoint: {checkpoint}')
    logging.info(f'Output root: {out_root}')
    logging.info(f'Smoke script: {smoke_script}')

    if not smoke_script.exists():
        logging.error(f"Could not find smoke script at: {smoke_script}. Adjust path and try again.")
        sys.exit(2)

    if not checkpoint.exists():
        logging.error(f"Checkpoint not found: {checkpoint}")
        sys.exit(2)

    if not img_dir.exists():
        logging.error(f"Image folder not found: {img_dir}")
        sys.exit(2)

    # Find images recursively
    images = sorted([p for p in img_dir.rglob('*') if p.is_file() and p.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']])
    if len(images) == 0:
        logging.error('No images found in img_dir')
        sys.exit(2)

    logging.info(f'Found {len(images)} images. Starting pipeline (timeout={args.timeout}s per image)')

    for i, img_path in enumerate(images, start=1):
        start_t = time.time()
        basename = img_path.stem
        image_out_dir = out_root / basename
        image_out_dir.mkdir(parents=True, exist_ok=True)

        logging.info(f'[{i}/{len(images)}] Processing: {img_path.name}')

        cmd = [sys.executable, str(smoke_script), '--img', str(img_path), '--checkpoint', str(checkpoint),
               '--num-classes', str(args.num_classes), '--device', args.device]
        logging.info('CMD: ' + ' '.join(cmd))

        rc, out, err = run_command(cmd, cwd=str(project_root), timeout=args.timeout)

        # save stdout/stderr
        try:
            (image_out_dir / 'stdout.txt').write_text(out)
        except Exception:
            logging.warning(f'Could not write stdout for {basename}')
        try:
            (image_out_dir / 'stderr.txt').write_text(err)
        except Exception:
            logging.warning(f'Could not write stderr for {basename}')

        if rc != 0:
            logging.error(f'Error running smoke script for {img_path.name} (rc={rc}). See {image_out_dir / "stderr.txt"} for details')
            # continue on error
            continue

        moved_any = collect_and_cleanup(project_root, basename, image_out_dir)

        if not moved_any:
            logging.warning(f'No generated files matched image basename `{basename}` or temp xai_smoke_* files. Check smoke script naming or inspect {image_out_dir / "stdout.txt"}')

        elapsed = time.time() - start_t
        logging.info(f'[{i}/{len(images)}] Completed {img_path.name} in {elapsed:.1f}s')

    logging.info('XAI pipeline finished')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', type=str, default='data/processed/test/yes', help='Folder with test images (recursive)')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth', help='Path to model checkpoint')
    parser.add_argument('--out_dir', type=str, default='outputs/xai_visualizations', help='Root output folder')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes for model')
    parser.add_argument('--device', type=str, default='cpu', help='cpu or cuda')
    parser.add_argument('--timeout', type=int, default=600, help='Timeout in seconds per image')
    args = parser.parse_args()
    main(args)
