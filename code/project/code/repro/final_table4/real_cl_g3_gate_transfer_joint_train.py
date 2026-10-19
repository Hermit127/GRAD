#!/usr/bin/env python3
"""Joint-transfer gate: Table 1 GRAD-512 init, trained on Decomp train +
closed-loop gate train (retention + deployment), selected on gate dev.

Zero API. Writes code/repro/out/online_gate_g3/transfer_joint/.
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
DECOMP = REPO / 'code/repro/out/unified_gnn/fix2_splits/decomp/fold0'
INIT = REPO / 'code/repro/out/grad_repair/transfer_ckpt/G3_512_decomp_fold0_s211.pt'
OUT = REPO / 'code/repro/out/online_gate_g3/transfer_joint'
CKPT = OUT / 'g3_gate_512_transfer_joint.pt'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    train_joint = OUT / 'train_joint.jsonl'
    rows = [l for l in (DECOMP / 'train.jsonl').read_text().splitlines()
            if l.strip()]
    rows += [l for l in (DS / 'train.jsonl').read_text().splitlines()
             if l.strip()]
    train_joint.write_text('\n'.join(rows) + '\n')
    sys.argv = [
        'train_grad_repair.py',
        '--train', str(train_joint),
        '--val', str(DS / 'dev.jsonl'),
        '--test', str(DS / 'test.jsonl'),
        '--candidate', 'A_moddrop',
        '--lambda-aux', '0.1', '--moddrop', '0.25',
        '--epochs', '100', '--patience', '25', '--batch-size', '48',
        '--hidden-dim', '192', '--lr', '1e-3', '--seed', '211',
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
    manifest = {
        'init_from': str(INIT), 'init_from_sha256': sha(INIT),
        'checkpoint': str(CKPT), 'checkpoint_sha256': sha(CKPT),
        'train_joint': {'path': str(train_joint),
                        'sha256': sha(train_joint),
                        'n_rows': len(rows),
                        'parts': [str(DECOMP / 'train.jsonl'),
                                  str(DS / 'train.jsonl')]},
        'val': str(DS / 'dev.jsonl'), 'test': str(DS / 'test.jsonl'),
        'protocol': 'joint transfer: Table 1 init, Decomp train + gate train, '
                    'lr 1e-3, 100 epochs / patience 25, gate-dev fixed-FPR F1',
        'trainer_sha256': sha(REPO / 'code/repro/final_five_tables/'
                                     'train_grad_repair.py'),
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
