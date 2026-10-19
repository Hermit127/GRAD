#!/usr/bin/env python3
"""Train the G3-512 closed-loop gate on the v2 gate protocol dataset.

Runs `train_grad_repair.py` (G3 backbone, A_moddrop) on
`online_gate_v2/{train,dev,test}.jsonl` with the v2 gate protocol
(100 epochs/patience 25/batch 48, dev fixed-FPR F1 selection, hard-benign
mining w=3 top-128 from epoch 2), windows 512/160, and saves a packed gate
checkpoint consumable by `final_table4.g3_gate.G3Gate`.

Zero API. Writes `code/repro/out/online_gate_g3/` (manifest + checkpoint).
"""
from __future__ import annotations

import hashlib
import os
import json
import sys
from pathlib import Path

REPO = Path(os.environ.get('GRAD_BASE',
                           '.'))
sys.path.insert(0, str(REPO / 'code'))
sys.path.insert(0, str(REPO / 'code/repro'))
sys.path.insert(0, str(REPO / 'code/repro/final_five_tables'))

import train_grad_repair as tgr  # noqa: E402

DS = REPO / 'code/repro/out/online_gate_v2'
OUT = REPO / 'code/repro/out/online_gate_g3'
CKPT = OUT / 'g3_gate_512.pt'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    sys.argv = [
        'train_grad_repair.py',
        '--train', str(DS / 'train.jsonl'),
        '--val', str(DS / 'dev.jsonl'),
        '--test', str(DS / 'test.jsonl'),
        '--candidate', 'A_moddrop',
        '--lambda-aux', '0.1', '--moddrop', '0.25',
        '--epochs', '100', '--patience', '25', '--batch-size', '48',
        '--hidden-dim', '192', '--lr', '1e-3', '--seed', '211',
        '--max-node-chars', '512', '--max-edge-chars', '160',
        '--target-fpr', '0.1', '--hard-benign-weight', '3.0',
        '--hard-benign-top-k', '128',
        '--out-dir', str(OUT),
        '--save-checkpoint', str(CKPT),
    ]
    rc = tgr.main()
    if rc != 0:
        return rc
    manifest = {
        'gate': 'G3', 'candidate': 'A_moddrop',
        'windows': {'node': 512, 'edge': 160},
        'protocol': '100 epochs / patience 25 / batch 48 / lr 1e-3 / '
                    'dev fixed-FPR F1 / hard-benign w=3 top-128 from epoch 2',
        'dataset': {ph: str(DS / f'{ph}.jsonl')
                    for ph in ('train', 'dev', 'test')},
        'dataset_sha256': {ph: sha(DS / f'{ph}.jsonl')
                           for ph in ('train', 'dev', 'test')},
        'checkpoint': str(CKPT), 'checkpoint_sha256': sha(CKPT),
        'trainer_sha256': sha(REPO / 'code/repro/final_five_tables/'
                                     'train_grad_repair.py'),
        'gate_class_sha256': sha(REPO / 'code/repro/final_table4/g3_gate.py'),
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
