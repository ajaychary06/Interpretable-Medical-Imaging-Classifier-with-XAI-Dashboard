# scripts/run_full_pipeline.py
"""
Unified wrapper: optionally runs training (if provided), then evaluation, XAI, and report creation.
Usage:
  python scripts/run_full_pipeline.py --skip_train --device cpu
"""
import subprocess, sys, argparse, os
from pathlib import Path

ROOT = Path('.').resolve()

def run(cmd):
    print("RUN:", " ".join(cmd))
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    print(proc.stdout)
    if proc.stderr:
        print("ERR:", proc.stderr)
    return proc.returncode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Run training if a training script exists')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    # 1) optional training
    if args.train:
        train_script = ROOT / 'scripts' / 'train.py'
        if train_script.exists():
            rc = run([sys.executable, str(train_script)])
            if rc != 0:
                print("Training failed or exited non-zero. Aborting.")
                return
        else:
            print("No train.py found, skipping training.")

    # 2) compute metrics (script previously created compute_metrics.py)
    metrics_script = ROOT / 'scripts' / 'compute_metrics.py'
    if metrics_script.exists():
        run([sys.executable, str(metrics_script)])
    else:
        print("compute_metrics.py not found; skip metrics.")

    # 3) run XAI pipeline (use existing wrapper)
    xai_wrapper = ROOT / 'scripts' / 'run_xai_pipeline.py'
    if xai_wrapper.exists():
        # run for yes and no (test splits)
        run([sys.executable, str(xai_wrapper), '--img_dir', 'data/processed/test/yes', '--checkpoint', 'checkpoints/best_model.pth', '--num_classes', '2', '--device', args.device])
        run([sys.executable, str(xai_wrapper), '--img_dir', 'data/processed/test/no', '--checkpoint', 'checkpoints/best_model.pth', '--num_classes', '2', '--device', args.device])
    else:
        print("run_xai_pipeline.py not found; skipping XAI batch.")

    # 4) generate summary and report
    for script in ['make_xai_summary.py', 'select_examples.py', 'make_report_from_examples.py']:
        s = ROOT / 'scripts' / script
        if s.exists():
            run([sys.executable, str(s)])
        else:
            print(f"{script} not present; skipping.")

    print("Full pipeline completed.")

if __name__ == '__main__':
    main()
