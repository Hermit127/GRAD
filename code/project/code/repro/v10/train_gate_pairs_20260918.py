#!/usr/bin/env python3
"""Phase C: continued-training pilot for the GRAD gate (classification + paired
authorization ranking).

Arms (all start from the frozen v2 checkpoint, identical budget):
  C0  frozen reference   (no training; evaluated with the same protocol)
  C1  continued classification training on the original corpus, no pairs
  C2  C1 data + same-tool authorization pairs (classification)
  C3  C2 + pairwise ranking loss on raw logits:
      L = L_cls + lambda * mean(max(0, m - s(G_harm) + s(G_benign)))
      with m=1, pair batch share fixed, lambda in {0.1, 0.5}

No API calls; CPU or local GPU only. All candidate checkpoints are kept.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path('.')
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code/repro'))
sys.path.insert(0, str(ROOT / 'code/repro/v10'))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: E402
from torch_geometric.data import Batch, DataLoader  # noqa: E402

import train_gnn_v2 as T  # noqa: E402
from train_gnn_v2 import (CausalCharGNN, V2GraphDataset, batch_logits,  # noqa: E402
                          metrics_at_threshold, set_seed, threshold_for_target_fpr)

DATA = ROOT / 'code/repro/out/online_gate_v2'
PAIRS = ROOT / 'LOGS/final_table2/auth_pairs_20260918'
OUT = ROOT / 'LOGS/final_table2/grad_pairs_train_20260918'
INIT_CKPT = ROOT / 'code/repro/out/online_gate_v2/gnn_gate_v2.pt'


def read_jsonl(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def write_jsonl(path, rows):
    Path(path).write_text('\n'.join(json.dumps(r) for r in rows) + '\n')


@torch.no_grad()
def predict(model, ds, batch_size=64):
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size)
    probs = []
    for batch in loader:
        batch = T.move_batch(batch)
        probs.extend(torch.sigmoid(batch_logits(model, batch)).cpu().tolist())
    return probs


def pair_indices(rows):
    by_pid = {}
    for i, r in enumerate(rows):
        by_pid.setdefault(r.get('pair_id'), {})[r['label']] = i
    return [(v[1], v[0]) for v in by_pid.values() if 0 in v and 1 in v]


def eval_groups(model, dev_orig, dev_pairs, test_orig, c0_threshold):
    p_orig = predict(model, dev_orig)
    y_orig = [int(g['label']) for g in dev_orig.graphs]
    p_pairs = predict(model, dev_pairs)
    y_pairs = [int(g['label']) for g in dev_pairs.graphs]
    thr, fixed = threshold_for_target_fpr(y_orig, p_orig, 0.10)
    idx = pair_indices(dev_pairs.graphs)
    rank_acc = (sum(1 for pos, neg in idx if p_pairs[pos] > p_pairs[neg]) / len(idx)
                if idx else None)

    def at(th):
        pred = [1 if p >= th else 0 for p in p_pairs]
        tp = sum(1 for y, q in zip(y_pairs, pred) if y == 1 and q == 1)
        fp = sum(1 for y, q in zip(y_pairs, pred) if y == 0 and q == 1)
        pos = sum(y_pairs)
        return {'threshold': th, 'tpr': tp / max(pos, 1),
                'fpr': fp / max(len(y_pairs) - pos, 1)}

    out = {
        'orig_dev': {'auroc': roc_auc_score(y_orig, p_orig),
                     'f1_at_0.5': metrics_at_threshold(y_orig, p_orig, 0.5)['f1'],
                     'fixed_fpr_10': fixed},
        'pair_dev': {'auroc': roc_auc_score(y_pairs, p_pairs),
                     'pr_auc': average_precision_score(y_pairs, p_pairs),
                     'ranking_accuracy': rank_acc,
                     'at_dev_fpr10_threshold': at(thr),
                     'at_c0_threshold': at(c0_threshold)},
    }
    if test_orig is not None:
        p_test = predict(model, test_orig)
        y_test = [int(g['label']) for g in test_orig.graphs]
        out['orig_test'] = {'auroc': roc_auc_score(y_test, p_test),
                            'f1_at_0.5': metrics_at_threshold(y_test, p_test, 0.5)['f1']}
    return out


def train_custom_epoch(model, train_ds, optimizer, pair_pool, n_pairs, lam,
                       margin, pos_weight, batch_size):
    model.train()
    total_bce, total_rank, total_loss, n = 0.0, 0.0, 0.0, 0
    order = list(range(len(train_ds)))
    random.shuffle(order)
    for start in range(0, len(order), batch_size):
        base_items = [train_ds[i] for i in order[start:start + batch_size]]
        extra = []
        if pair_pool and n_pairs > 0 and base_items:
            for _ in range(n_pairs):
                pos_d, neg_d = pair_pool[random.randrange(len(pair_pool))]
                extra += [pos_d, neg_d]
        b = T.move_batch(Batch.from_data_list(base_items + extra))
        logits = batch_logits(model, b)
        y = b.y.view(-1)
        loss_vec = F.binary_cross_entropy_with_logits(
            logits, y, pos_weight=torch.tensor(pos_weight, device=logits.device),
            reduction='none')
        if hasattr(b, 'sample_weight'):
            loss_vec = loss_vec * b.sample_weight.view_as(loss_vec)
        bce = loss_vec.mean()
        rank = torch.tensor(0.0, device=logits.device)
        if extra:
            ex_logits = logits[-len(extra):]
            pos = ex_logits[0::2]
            neg = ex_logits[1::2]
            rank = F.relu(margin - pos + neg).mean()
        loss = bce + lam * rank
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_bce += float(bce.item())
        total_rank += float(rank.item())
        total_loss += float(loss.item())
        n += 1
    return total_bce / max(n, 1), total_rank / max(n, 1)


def build_ds(train_rows, dev_rows):
    p = OUT / '_tmp'
    p.mkdir(parents=True, exist_ok=True)
    write_jsonl(p / 'train.jsonl', train_rows)
    write_jsonl(p / 'dev.jsonl', dev_rows)
    return V2GraphDataset(p / 'train.jsonl'), V2GraphDataset(p / 'dev.jsonl')


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['C0', 'C1', 'C2', 'C3'], required=True)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--lam', type=float, default=0.1)
    ap.add_argument('--pair-share', type=float, default=0.25)
    ap.add_argument('--epochs', type=int, default=8)
    ap.add_argument('--patience', type=int, default=3)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--batch-size', type=int, default=48)
    ap.add_argument('--hard-benign-weight', type=float, default=3.0)
    ap.add_argument('--hard-benign-top-k', type=int, default=128)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    run_id = f'{args.arm}_seed{args.seed}' + (f'_lam{args.lam}' if args.arm == 'C3' else '')
    torch.set_num_threads(int(__import__('os').environ.get('TORCH_THREADS', '8')))
    set_seed(args.seed)
    base_train = read_jsonl(DATA / 'train.jsonl')
    base_dev = read_jsonl(DATA / 'dev.jsonl')
    pair_train = read_jsonl(PAIRS / 'auth_pairs_train.jsonl')
    pair_dev = read_jsonl(PAIRS / 'auth_pairs_dev.jsonl')

    ck = torch.load(INIT_CKPT, map_location='cpu', weights_only=False)
    cfg = ck['config']
    model = CausalCharGNN(node_base_dim=cfg['node_base_dim'],
                          edge_base_dim=cfg['edge_base_dim'])
    model.load_state_dict(ck['model_state'])
    c0_threshold = float((ck.get('threshold_reports') or {}).get('fpr_10', {})
                         .get('threshold', 0.9999856948852539))

    dev_orig = V2GraphDataset(DATA / 'dev.jsonl')
    dev_pairs = V2GraphDataset(PAIRS / 'auth_pairs_dev.jsonl')
    test_orig = V2GraphDataset(DATA / 'test.jsonl')

    result = {'run_id': run_id, 'args': vars(args), 'c0_threshold': c0_threshold,
              'init_ckpt': str(INIT_CKPT)}

    if args.arm == 'C0':
        result['eval'] = eval_groups(model, dev_orig, dev_pairs, test_orig, c0_threshold)
        (OUT / f'{run_id}.json').write_text(json.dumps(result, indent=1))
        print(json.dumps(result['eval'], indent=1))
        return 0

    train_rows = base_train + (pair_train if args.arm in ('C2', 'C3') else [])
    train_ds, dev_ds = build_ds(train_rows, base_dev + (pair_dev if args.arm in ('C2', 'C3') else []))
    pair_pool = None
    if args.arm == 'C3':
        by_pid = {}
        for i, r in enumerate(train_rows):
            if r.get('pair_id'):
                by_pid.setdefault(r['pair_id'], {})[r['label']] = i
        pair_pool = [(train_ds[v[1]], train_ds[v[0]]) for v in by_pid.values()
                     if 0 in v and 1 in v]
    pos = sum(1 for g in train_ds.graphs if int(g['label']) == 1)
    pos_weight = (len(train_ds.graphs) - pos) / max(pos, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)
    base_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size)
    best = {'score': -1.0, 'epoch': 0, 'state': None, 'eval': None}
    history = []
    n_pairs = max(1, int(round(args.batch_size * args.pair_share
                               / (1 - args.pair_share) / 2)))
    for epoch in range(1, args.epochs + 1):
        if args.arm == 'C3':
            bce, rank = train_custom_epoch(
                model, train_ds, optimizer, pair_pool, n_pairs, args.lam, 1.0,
                pos_weight, args.batch_size)
        else:
            bce = T.train_epoch(model, base_loader, optimizer, pos_weight)
            rank = 0.0
        ev = eval_groups(model, dev_orig, dev_pairs, None, c0_threshold)
        score = ev['orig_dev']['fixed_fpr_10']['f1']
        scheduler.step(bce)
        history.append({'epoch': epoch, 'bce': bce, 'rank': rank,
                        'orig_dev_f1_fpr10': score,
                        'pair_dev_rank_acc': ev['pair_dev']['ranking_accuracy'],
                        'pair_dev_auroc': ev['pair_dev']['auroc']})
        if score > best['score']:
            best = {'score': score, 'epoch': epoch,
                    'state': {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()},
                    'eval': ev}
        if epoch - best['epoch'] >= args.patience:
            break
    model.load_state_dict(best['state'])
    final = eval_groups(model, dev_orig, dev_pairs, test_orig, c0_threshold)
    result.update({'history': history, 'best_epoch': best['epoch'],
                   'val_fixed_fpr10_f1': best['score'], 'eval': final})
    torch.save({'model_state': best['state'], 'config': cfg,
                'run_id': run_id, 'c0_threshold': c0_threshold,
                'threshold_reports': ck.get('threshold_reports')},
               OUT / f'{run_id}.pt')
    (OUT / f'{run_id}.json').write_text(json.dumps(result, indent=1))
    print(f"[{run_id}] best_epoch={best['epoch']} orig_dev_f1={best['score']:.4f} "
          f"pair_dev_rank_acc={final['pair_dev']['ranking_accuracy']} "
          f"pair_dev_auroc={final['pair_dev']['auroc']:.4f} "
          f"orig_test_auroc={final['orig_test']['auroc']:.4f}", flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
