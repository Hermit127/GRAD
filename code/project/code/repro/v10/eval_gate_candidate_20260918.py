#!/usr/bin/env python3
"""Zero-API evaluation of any gate checkpoint on all available graph sets:
online_gate_v2 dev/test, authorization pairs (train/dev), decomp_test, mg_test.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path('.')
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code/repro'))
sys.path.insert(0, str(ROOT / 'code/repro/v10'))

import torch  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from torch_geometric.data import DataLoader  # noqa: E402

from train_gnn_v2 import (CausalCharGNN, V2GraphDataset, batch_logits,  # noqa: E402
                          metrics_at_threshold, move_batch,
                          threshold_for_target_fpr)

SETS = {
    'orig_dev': ROOT / 'code/repro/out/online_gate_v2/dev.jsonl',
    'orig_test': ROOT / 'code/repro/out/online_gate_v2/test.jsonl',
    'pairs_train': ROOT / 'LOGS/final_table2/auth_pairs_20260918/auth_pairs_train.jsonl',
    'pairs_dev': ROOT / 'LOGS/final_table2/auth_pairs_20260918/auth_pairs_dev.jsonl',
    'decomp_test': ROOT / 'code/repro/out/online_graphs/decomp_test.jsonl',
    'mg_test': ROOT / 'code/repro/out/online_graphs/mg_test.jsonl',
}


def score_ds(model, ds):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in DataLoader(ds, batch_size=64):
            probs.extend(torch.sigmoid(batch_logits(model, move_batch(batch))).cpu().tolist())
    return probs


def pair_rank_accuracy(ds, probs):
    by = {}
    for i, g in enumerate(ds.graphs):
        by.setdefault(g.get('pair_id'), {})[g['label']] = i
    pairs = [v for v in by.values() if 0 in v and 1 in v]
    if not pairs:
        return None
    return sum(1 for v in pairs if probs[v[1]] > probs[v[0]]) / len(pairs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', default='')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    cfg = ck['config']
    model = CausalCharGNN(cfg['node_base_dim'], cfg['edge_base_dim'])
    model.load_state_dict(ck['model_state'])
    model.eval()

    out = {'ckpt': args.ckpt, 'tag': args.tag}
    probs_cache = {}
    for name, path in SETS.items():
        if not path.exists():
            out[name] = {'status': 'missing', 'path': str(path)}
            continue
        ds = V2GraphDataset(path)
        probs = score_ds(model, ds)
        probs_cache[name] = probs
        y = [int(g['label']) for g in ds.graphs]
        n_pos = sum(y)
        res = {'n': len(y), 'pos': n_pos,
               'auroc': roc_auc_score(y, probs) if len(set(y)) > 1 else None,
               'f1_at_0.5': metrics_at_threshold(y, probs, 0.5)['f1']}
        if 'pairs' in name:
            res['ranking_accuracy'] = pair_rank_accuracy(ds, probs)
            res['pr_auc'] = average_precision_score(y, probs)
        out[name] = res
    if 'orig_dev' in probs_cache:
        y = [int(g['label']) for g in V2GraphDataset(SETS['orig_dev']).graphs]
        thr, m = threshold_for_target_fpr(y, probs_cache['orig_dev'], 0.10)
        out['dev_fpr10_rule'] = {'threshold': thr, **m}
        for name in ('pairs_train', 'pairs_dev'):
            if name not in probs_cache:
                continue
            y2 = [int(g['label']) for g in V2GraphDataset(SETS[name]).graphs]
            pred = [1 if p >= thr else 0 for p in probs_cache[name]]
            tp = sum(1 for a, b in zip(y2, pred) if a == 1 and b == 1)
            fp = sum(1 for a, b in zip(y2, pred) if a == 0 and b == 1)
            out[name]['at_dev_fpr10_rule'] = {
                'tpr': tp / max(sum(y2), 1), 'fpr': fp / max(len(y2) - sum(y2), 1)}
    text = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(text)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
