#!/usr/bin/env python3
"""Transfer-adapt the Table 1 GRAD-512 detector to the closed-loop gate
distribution (zero API, GPU only).

Init: Table 1 GRAD-512 candidate (Decomp fold0, seed 211) retrained with
`--save-checkpoint` under the frozen confirmation protocol.
Data: `online_gate_v2/{train,dev,test}.jsonl` (same as the from-scratch gate).
Protocol: A_moddrop, windows 512/160, batch 48, seed 211, lr 3e-4 (10x lower
for transfer), 60 epochs / patience 15, dev fixed-FPR F1 selection,
hard-benign mining w=3 top-128 from epoch 2.

Pre-declared deployment gates: dev AUROC >= 0.90 AND test AUROC >= 0.90 AND
dev fixed-FPR-10 F1 >= 0.75. Otherwise no closed-loop rerun (report only).

Writes `code/repro/out/online_gate_g3/transfer/{ckpt,manifest.json,phase2_*.json}`.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get('GRAD_BASE',
                           '.'))
sys.path.insert(0, str(REPO / 'code'))
sys.path.insert(0, str(REPO / 'code/repro'))
sys.path.insert(0, str(REPO / 'code/repro/final_five_tables'))
sys.path.insert(0, str(REPO / 'code/repro/final_table4'))

import train_grad_repair as tgr  # noqa: E402

DS = REPO / 'code/repro/out/online_gate_v2'
INIT = REPO / 'code/repro/out/grad_repair/transfer_ckpt/G3_512_decomp_fold0_s211.pt'
OUT = REPO / 'code/repro/out/online_gate_g3/transfer'
CKPT = OUT / 'g3_gate_512_transfer.pt'


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
        '--epochs', '60', '--patience', '15', '--batch-size', '48',
        '--hidden-dim', '192', '--lr', '3e-4', '--seed', '211',
        '--max-node-chars', '512', '--max-edge-chars', '160',
        '--target-fpr', '0.1', '--hard-benign-weight', '3.0',
        '--hard-benign-top-k', '128',
        '--init-from', str(INIT),
        '--out-dir', str(OUT),
        '--save-checkpoint', str(CKPT),
    ]
    rc = tgr.main()
    if rc != 0:
        return rc
    best = json.loads((OUT / 'manifest.json').read_text()) if (
        OUT / 'manifest.json').exists() else {}
    manifest = {
        'init_from': str(INIT), 'init_from_sha256': sha(INIT),
        'checkpoint': str(CKPT), 'checkpoint_sha256': sha(CKPT),
        'protocol': 'transfer fine-tune: lr 3e-4, 60 epochs / patience 15, '
                    'dev fixed-FPR F1, hard-benign w=3 top-128 from epoch 2',
        'dataset': {ph: sha(DS / f'{ph}.jsonl')
                    for ph in ('train', 'dev', 'test')},
        'trainer_sha256': sha(REPO / 'code/repro/final_five_tables/'
                                     'train_grad_repair.py'),
        'gate_class_sha256': sha(REPO / 'code/repro/final_table4/g3_gate.py'),
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
