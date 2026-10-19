"""Retrain the GRAD pre-execution gate on the v2 (format-unified) dataset.

2026-09-17: audit found the v1 gate was trained on ``arg=``/``observation=``
node text while the live hook emitted ``args=<json>``/``result=`` with no
action status. v2 rebuilds the training data with the shared
``grad_map.format_event_text`` formatter and a ``status`` field, then retrains
with the identical config. The v1 checkpoint is kept.

Usage:
  python code/repro/v10/retrain_gate_v2_20260917.py
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

REPO = Path('.')
sys.path.insert(0, str(REPO / 'code'))
sys.path.insert(0, str(REPO / 'code/repro'))
sys.path.insert(0, str(REPO / 'code/repro/v10'))

import train_gnn_v2  # noqa: E402

OUT = REPO / 'code/repro/out/online_gate_v2'
RUN_DIR = OUT / 'train_run_v2'
CKPT = OUT / 'gnn_gate_v2.pt'


def main() -> int:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    train_gnn_v2.MODEL_DIR = RUN_DIR
    sys.argv = [
        'train_gnn_v2.py',
        '--train', str(OUT / 'train.jsonl'),
        '--val', str(OUT / 'dev.jsonl'),
        '--test', str(OUT / 'test.jsonl'),
        '--mode', 'graph',
        '--epochs', '100',
        '--patience', '25',
        '--hidden-dim', '192',
        '--batch-size', '48',
        '--lr', '1e-3',
        '--seed', '211',
        '--target-fpr', '0.1',
        '--fixed-fpr-targets', '0.05,0.10,0.20',
        '--checkpoint-metric', 'fixed_fpr_f1',
        '--hard-benign-weight', '3.0',
        '--hard-benign-top-k', '128',
        '--hard-benign-start-epoch', '2',
        '--out-dir', str(RUN_DIR),
    ]
    rc = train_gnn_v2.main()
    raw = sorted(RUN_DIR.glob('gnn_v2_*.pt'), key=lambda p: p.stat().st_mtime)
    if not raw:
        print('ERROR: no checkpoint written by train_gnn_v2', flush=True)
        return 1
    shutil.copyfile(raw[-1], CKPT)
    print(f'copied {raw[-1]} -> {CKPT}', flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
