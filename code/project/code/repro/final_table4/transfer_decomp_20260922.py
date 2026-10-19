#!/usr/bin/env python3
"""Cross-domain transfer check between the Table 1 GRAD-512 detector and the
closed-loop G3 gate (zero API; GPU only for one retrain).

Transfer A: closed-loop gate (`online_gate_g3/g3_gate_512.pt`, trained on
online_gate_v2 trace nodes) evaluated on the Table 1 Decomp test split
(`unified_gnn/fix2_splits/decomp/fold0/test.jsonl`).

Transfer B: Table 1 GRAD-512 candidate retrained exactly per the frozen
confirmation protocol on Decomp fold0 seed 211 (with `--save-checkpoint`),
then evaluated on the closed-loop gate dataset (`online_gate_v2/{dev,test}`).

Writes LOGS/final_table4_g3/transfer_decomp_20260922.json.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path('.')
sys.path.insert(0, str(REPO / 'code'))
sys.path.insert(0, str(REPO / 'code/repro'))
sys.path.insert(0, str(REPO / 'code/repro/v10'))
sys.path.insert(0, str(REPO / 'code/repro/final_table4'))
sys.path.insert(0, str(REPO / 'code/repro/final_five_tables'))

from sklearn.metrics import roc_auc_score  # noqa: E402

OUT = REPO / 'LOGS/final_table4_g3'
SPLIT = REPO / 'code/repro/out/unified_gnn/fix2_splits/decomp/fold0'
DS = REPO / 'code/repro/out/online_gate_v2'
T1_CKPT = REPO / 'code/repro/out/grad_repair/transfer_ckpt/G3_512_decomp_fold0_s211.pt'
PY = 'python'


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def score_graphs(gate, graphs):
    ps, ys = [], []
    for g in graphs:
        p, _ = gate.score_nodes(g['nodes'])
        ps.append(float(p))
        ys.append(int(g['label']))
    return np.array(ys), np.array(ps)


def metrics(y, p, threshold=None):
    auc = roc_auc_score(y, p)
    order = np.argsort(-p)
    y_s = y[order]
    pos, neg = y.sum(), len(y) - y.sum()
    tpr, fpr, best = [], [], 0.0
    tp = fp = 0
    for i, yy in enumerate(y_s):
        tp += int(yy == 1)
        fp += int(yy == 0)
        f = fp / neg
        if f <= 0.10:
            best = max(best, tp / pos)
    out = {'n': len(y), 'pos': int(pos), 'neg': int(neg),
           'auroc': round(float(auc), 4),
           'tpr_at_fpr10': round(best, 4)}
    if threshold is not None:
        pred = p >= threshold
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        out['threshold'] = float(threshold)
        out['tpr_at_thr'] = round(tp / max(1, int(pos)), 4)
        out['fpr_at_thr'] = round(fp / max(1, int(neg)), 4)
    return out


def main() -> int:
    from g3_gate import G3Gate

    out = {}
    # ---- Transfer A: closed-loop gate -> Decomp test
    gate_loop = G3Gate(ckpt=REPO / 'code/repro/out/online_gate_g3/'
                              'g3_gate_512.pt')
    y, p = score_graphs(gate_loop, rows(SPLIT / 'test.jsonl'))
    out['A_closedloop_gate_on_decomp_test'] = metrics(
        y, p, threshold=None)
    out['A_closedloop_gate_threshold'] = gate_loop.threshold

    # ---- Transfer B: Table 1 GRAD-512 (retrain with checkpoint) -> gate data
    if not T1_CKPT.exists():
        T1_CKPT.parent.mkdir(parents=True, exist_ok=True)
        cmd = [PY, str(REPO / 'code/repro/final_five_tables/train_grad_repair.py'),
               '--train', str(SPLIT / 'train.jsonl'),
               '--val', str(SPLIT / 'dev.jsonl'),
               '--test', str(SPLIT / 'test.jsonl'),
               '--seed', '211', '--candidate', 'A_moddrop', '--lambda-aux', '0.1',
               '--moddrop', '0.25', '--max-node-chars', '512',
               '--out-dir', str(T1_CKPT.parent),
               '--save-checkpoint', str(T1_CKPT)]
        rc = subprocess.call(cmd)
        assert rc == 0 and T1_CKPT.exists(), 'Table 1 retrain failed'
    gate_t1 = G3Gate(ckpt=T1_CKPT)
    out['B_table1_threshold'] = gate_t1.threshold
    yg, pg = score_graphs(gate_t1, rows(DS / 'test.jsonl'))
    out['B_table1_on_closedloop_test'] = metrics(yg, pg, threshold=None)
    out['B_table1_on_closedloop_test_at_own_thr'] = metrics(
        yg, pg, threshold=gate_t1.threshold)
    yd, pd = score_graphs(gate_t1, rows(DS / 'dev.jsonl'))
    out['B_table1_on_closedloop_dev'] = metrics(yd, pd, threshold=None)
    yt, pt = score_graphs(gate_t1, rows(SPLIT / 'test.jsonl'))
    out['B_table1_on_decomp_test_sanity'] = metrics(yt, pt, threshold=None)

    (OUT / 'transfer_decomp_20260922.json').write_text(
        json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
