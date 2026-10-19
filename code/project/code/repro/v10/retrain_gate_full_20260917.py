"""Retrain the GRAD pre-execution gate on the full-context dataset (2026-09-17).

Thin wrapper around ``code/train_gnn_v2.py`` with the exact config of the
original gate run (train_run3: mode=graph, epochs=100, patience=25,
hidden_dim=192, batch_size=48, lr=1e-3, seed=211, target_fpr=0.1,
checkpoint_metric=fixed_fpr_f1, hard-benign defaults). ``MODEL_DIR`` is
monkeypatched into the run directory so no frozen path (``code/models/``) is
touched; the produced checkpoint is then copied to
``code/repro/out/online_gate_full/gnn_gate_full.pt`` in the exact format
``GnnGate`` expects (the format written by ``train_gnn_v2.main``).

Usage:
  python code/repro/v10/retrain_gate_full_20260917.py
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

OUT = REPO / 'code/repro/out/online_gate_full'
RUN_DIR = OUT / 'train_run1'
CKPT = OUT / 'gnn_gate_full.pt'


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
