#!/usr/bin/env python
"""Paired-repair diagnostic pilot for the GRAD gate (P1 objective).

Trains a fresh CausalCharGNN on the frozen paired dataset
(`code/repro/out/paired_repair_v1`) with BCE on both pair endpoints plus a
same-prefix ranking margin. Pre-registered in
`LOGS/final_table4/paired_repair_pilot_prereg_20260918.md`; the frozen v2
checkpoint is never modified.

Usage:
  python code/repro/final_table4/train_paired_gate.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'code'))
sys.path.insert(0, str(REPO / 'code/repro/v10'))
sys.path.insert(0, str(REPO / 'code/repro/final_table4'))

import torch  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

import train_gnn_v2 as T  # noqa: E402

PAIRS = REPO / 'code/repro/out/paired_repair_v1'
V2 = REPO / 'code/repro/out/online_gate_v2'
OUT = PAIRS
LAMBDA, MARGIN = 1.0, 1.0
HIDDEN, SEED, EPOCHS, PATIENCE, BATCH, LR, WD = 192, 211, 100, 25, 48, 1e-3, 1e-4


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def to_data(nodes, label):
    d = T.graph_to_v2_data({'nodes': nodes})
    d.y = torch.tensor([float(label)])
    d.y_graph = d.y
    return d


def load_pairs(path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    out = []
    for p in rows:
        pair = {'pair_id': p['pair_id'], 'type': p['type'],
                'a': to_data(p['nodes_a'], p['label_a_harm']
                             if 'label_a_harm' in p else p.get(
                                 'candidate_is_bad', 0)),
                'b': to_data(p['nodes_b'], p['label_b_harm']
                             if 'label_b_harm' in p else 0),
                'expected_order': p.get('expected_order', 'A>B')}
        if p['type'] == 'evidence_swap':
            # endpoint labels are not defined as harm for evidence swaps:
            # use BCE only for candidate endpoints; here labels follow whether
            # the candidate is the dangerous target.
            lab = int(bool(p.get('candidate_is_bad')))
            pair['a'].y = torch.tensor([float(lab)])
            pair['b'].y = torch.tensor([float(lab)])
            if p.get('expected_order') == 'weak':
                pair['expected_order'] = 'none'
        out.append(pair)
    return out


def pair_batches(pairs, batch_size, shuffle, device='cpu'):
    idx = list(range(len(pairs)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        chunk = [pairs[j] for j in idx[i:i + batch_size]]
        batch_a = Batch.from_data_list([c['a'] for c in chunk]).to(device)
        batch_b = Batch.from_data_list([c['b'] for c in chunk]).to(device)
        yield chunk, batch_a, batch_b


def scores(model, batch):
    logits = T.batch_logits(model, batch)
    return logits


def evaluate_pairs(model, pairs, device='cpu'):
    model.eval()
    correct = n = 0
    with torch.no_grad():
        for _, ba, bb in pair_batches(pairs, BATCH, False, device):
            la = scores(model, ba)
            lb = scores(model, bb)
            for i, c in enumerate(_):
                order = c['expected_order']
                if order == 'none':
                    continue
                want_a = order == 'A>B'
                n += 1
                correct += int((float(la[i]) > float(lb[i])) == want_a)
    return correct, n


def evaluate_v2(model, role, device='cpu'):
    rows = [json.loads(l) for l in (V2 / f'{role}.jsonl').read_text()
            .splitlines() if l.strip()]
    model.eval()
    labels, probs = [], []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            chunk = rows[i:i + 64]
            batch = Batch.from_data_list(
                [T.graph_to_v2_data(r) for r in chunk]).to(device)
            p = torch.sigmoid(T.batch_logits(model, batch)).cpu().tolist()
            probs += p
            labels += [r['label'] for r in chunk]
    return labels, probs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--out-name', default='gnn_gate_paired.pt')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--config', choices=['p1', 'p1b'], default='p1')
    ap.add_argument('--init-ckpt', default=None)
    ap.add_argument('--lambda-pair', type=float, default=LAMBDA)
    ap.add_argument('--patience', type=int, default=PATIENCE)
    ap.add_argument('--lr', type=float, default=LR)
    args = ap.parse_args()
    set_seed(SEED)
    device = args.device
    if device == 'auto':
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    train = load_pairs(PAIRS / 'train.jsonl')
    if args.config == 'p1b':
        train = [p for p in train if p['type'] == 'candidate_swap']
    dev = load_pairs(PAIRS / 'dev.jsonl')
    dev_cand = [p for p in dev if p['type'] == 'candidate_swap']
    print(f'pairs train={len(train)} dev={len(dev)} '
          f'(candidate_swap {len(dev_cand)})', flush=True)

    node_base_dim = len(T.PERMISSION) + 1 + len(T.CONTENT_KEYWORDS)
    edge_base_dim = len(T.V2_EDGE_TYPES) + 1
    model = T.CausalCharGNN(node_base_dim, edge_base_dim,
                            hidden_dim=HIDDEN).to(device)
    if args.init_ckpt:
        ck = torch.load(args.init_ckpt, map_location=device,
                        weights_only=False)
        model.load_state_dict(ck['model_state'])
        print(f'initialized from {args.init_ckpt}', flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=WD)
    bce = torch.nn.BCEWithLogitsLoss()

    row_ds = []
    if args.config == 'p1b':
        row_ds = [T.graph_to_v2_data(json.loads(l))
                  for l in (V2 / 'train.jsonl').read_text().splitlines()
                  if l.strip()]
        print(f'original train rows for BCE: {len(row_ds)}', flush=True)

    best = {'dev_pair_acc': -1, 'dev_auroc': 0, 'epoch': -1, 'state': None}
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        nbatch = 0
        for chunk, ba, bb in pair_batches(train, BATCH, True, device):
            optimizer.zero_grad()
            la, lb = scores(model, ba), scores(model, bb)
            ya = torch.cat([c['a'].y for c in chunk]).to(la.device)
            yb = torch.cat([c['b'].y for c in chunk]).to(la.device)
            loss = bce(la, ya) + bce(lb, yb)
            if args.config == 'p1b':
                row_batch = Batch.from_data_list(
                    row_ds[(nbatch * BATCH) % len(row_ds):
                           (nbatch * BATCH) % len(row_ds) + BATCH]).to(device)
                row_logits = T.batch_logits(model, row_batch)
                loss = loss + bce(row_logits, row_batch.y.view(-1).to(device))
            pos, neg = [], []
            for i, c in enumerate(chunk):
                if c['expected_order'] == 'A>B':
                    pos.append(la[i])
                    neg.append(lb[i])
                elif c['expected_order'] == 'B>A':
                    pos.append(lb[i])
                    neg.append(la[i])
            if pos:
                s_pos = torch.stack(pos)
                s_neg = torch.stack(neg)
                loss = loss + args.lambda_pair * torch.clamp(
                    MARGIN - (s_pos - s_neg), min=0).mean()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            nbatch += 1
        correct, n = evaluate_pairs(model, dev, device)
        dev_acc = correct / n if n else 0.0
        labels, probs = evaluate_v2(model, 'dev', device)
        dev_auroc = (roc_auc_score(labels, probs)
                     if len(set(labels)) > 1 else 0.5)
        history.append({'epoch': epoch, 'loss': round(total / max(nbatch, 1), 5),
                        'dev_pair_acc': round(dev_acc, 4),
                        'dev_auroc': round(dev_auroc, 4)})
        guard = (dev_auroc >= 0.90) if args.config == 'p1b' else True
        improved = guard and (dev_acc > best['dev_pair_acc'] or
                              (dev_acc == best['dev_pair_acc'] and
                               dev_auroc > best['dev_auroc']))
        if improved:
            best.update({'dev_pair_acc': dev_acc, 'dev_auroc': dev_auroc,
                         'epoch': epoch,
                         'state': {k: v.clone() for k, v in
                                   model.state_dict().items()}})
        if epoch % 5 == 0 or improved:
            print(f"epoch {epoch}: loss={total / max(nbatch, 1):.4f} "
                  f"dev_pair={dev_acc:.3f} dev_auroc={dev_auroc:.3f}", flush=True)
        if epoch - best['epoch'] >= args.patience:
            print(f'early stop at epoch {epoch}', flush=True)
            break
    if best['state'] is None:
        print('FAIL: no epoch satisfied the dev AUROC guard (>=0.90)',
              flush=True)
    else:
        model.load_state_dict(best['state'])
    # held-out probe (final report only)
    test_pairs = load_pairs(PAIRS / 'test_probe.jsonl')
    all_correct, all_n = evaluate_pairs(model, test_pairs, device)
    cand_correct, cand_n = evaluate_pairs(
        model, [p for p in test_pairs if p['type'] == 'candidate_swap'],
        device)
    labels, probs = evaluate_v2(model, 'test', device)
    test_auroc = (roc_auc_score(labels, probs)
                  if len(set(labels)) > 1 else 0.5)
    # dev FPR-10 threshold report for GnnGate compatibility
    dev_labels, dev_probs = evaluate_v2(model, 'dev', device)
    thr, dev_metrics = T.threshold_for_target_fpr(dev_labels, dev_probs, 0.10)
    ckpt = {
        'config': {'node_base_dim': node_base_dim,
                   'edge_base_dim': edge_base_dim,
                   'train': {'pilot': 'paired_repair_v1', 'lambda': LAMBDA,
                             'margin': MARGIN, 'seed': SEED,
                             'hidden': HIDDEN, 'lr': LR, 'wd': WD,
                             'batch_pairs': BATCH, 'epochs': args.epochs}},
        'model_state': model.state_dict(),
        'threshold_reports': {'fpr_10': {
            'threshold': float(thr), 'val': dev_metrics}},
    }
    out_path = OUT / args.out_name
    torch.save(ckpt, out_path)
    report = {
        'checkpoint': str(out_path),
        'checkpoint_sha256': hashlib.sha256(
            out_path.read_bytes()).hexdigest(),
        'config': args.config, 'init_ckpt': args.init_ckpt,
        'hyperparameters': {'lambda_pair': args.lambda_pair, 'margin': MARGIN,
                            'seed': SEED,
                            'epochs_run': history[-1]['epoch'],
                            'best_epoch': best['epoch'],
                            'hidden': HIDDEN, 'lr': args.lr,
                            'batch_pairs': BATCH},
        'dev': {'pair_acc': best['dev_pair_acc'], 'auroc': best['dev_auroc']},
        'test_probe': {
            'candidate_swap_correct': f'{cand_correct}/{cand_n}',
            'candidate_swap_acc': (cand_correct / cand_n if cand_n else None),
            'all_pairs_correct': f'{all_correct}/{all_n}',
            'all_pairs_acc': all_correct / all_n if all_n else None},
        'original_v2_test': {'auroc': test_auroc},
        'frozen_baselines': {
            'dev_pair_acc': 1.0,
            'test_probe_candidate_swap_acc': 41 / 59,
            'lexical_task_acc': 0.872,
            'original_test_auroc': 0.9171511627906976},
        'device': device,
        'history': history,
    }
    (OUT / 'paired_repair_pilot_report.json').write_text(
        json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
