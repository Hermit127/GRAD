#!/usr/bin/env python3
"""MG clean-protocol evaluation v2 (fixes GPT audit 2026-09-08):
F1: train-internal val split stratified BY LABEL GROUP (attack/benign templates),
    asserts train/val/test all contain both classes; no silent fallback.
F2: named-offset shared featurizer (code/repro/graph_featurizer.py).
Evaluation: GroupKFold by template seed, checkpoint+threshold from val only,
sample std (ddof=1).
"""
import json, random, sys
from collections import defaultdict
sys.path.insert(0, 'code')
sys.path.insert(0, 'code/repro')
from graph_featurizer import to_data, FD, N_EDGE
import torch, torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import GINEConv, global_add_pool
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold
import statistics

DROP_SET = {'shell_exec', 'code_exec', 'memory_write'}


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = GINEConv(torch.nn.Sequential(torch.nn.Linear(FD, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64)), edge_dim=N_EDGE)
        self.c2 = GINEConv(torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 64)), edge_dim=N_EDGE)
        self.h = torch.nn.Linear(64, 1)

    def forward(self, x, ei, ea, b):
        x = self.c1(x, ei, ea).relu()
        x = self.c2(x, ei, ea).relu()
        return self.h(global_add_pool(x, b)).squeeze(-1)


def scores(model, gs):
    model.eval()
    out = []
    with torch.no_grad():
        for d in gs:
            out.append(torch.sigmoid(model(d.x, d.edge_index, d.edge_attr,
                                           torch.zeros(d.x.size(0), dtype=torch.long))).item())
    return out


def fit_fold(train_g, val_g, test_g, seed=42):
    trd = [d for d in (to_data(g, DROP_SET) for g in train_g) if d]
    vad = [d for d in (to_data(g, DROP_SET) for g in val_g) if d]
    ted = [d for d in (to_data(g, DROP_SET) for g in test_g) if d]
    tr_ys = [int(d.y.item()) for d in trd]
    va_ys = [int(d.y.item()) for d in vad]
    te_ys = [int(d.y.item()) for d in ted]
    # hard assertion: no degenerate fold
    assert len(set(tr_ys)) == 2 and len(set(va_ys)) == 2 and len(set(te_ys)) == 2, \
        f'degenerate fold (seed {seed}): train {set(tr_ys)} val {set(va_ys)} test {set(te_ys)}'
    random.seed(seed)
    torch.manual_seed(seed)
    m = Net()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=5e-4)
    best_va = -1
    best_state = None
    for ep in range(100):
        m.train()
        perm = list(range(len(trd)))
        random.shuffle(perm)
        for i in range(0, len(perm), 32):
            b = Batch.from_data_list([trd[j] for j in perm[i:i + 32]])
            opt.zero_grad()
            loss = F.binary_cross_entropy_with_logits(m(b.x, b.edge_index, b.edge_attr, b.batch), b.y)
            loss.backward()
            opt.step()
        va_s = scores(m, vad)
        va_auc = roc_auc_score(va_ys, va_s)
        if va_auc > best_va:
            best_va = va_auc
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
    m.load_state_dict(best_state)
    va_s = scores(m, vad)
    # threshold candidates derived from val scores (N3 fix): adjacent midpoints of
    # distinct val scores, clamped to [0,1]. Never silently falls back to 0 when
    # all positive scores are < 0.05.
    cands = sorted(set(va_s))
    cands = [0.0] + [(a + b) / 2 for a, b in zip(cands, cands[1:])] + [1.0]
    best = (-1.0, None)
    for t in cands:
        f = f1_score(va_ys, [1 if p >= t else 0 for p in va_s])
        if f > best[0]:
            best = (f, t)
    assert best[1] is not None, 'no valid threshold'
    te_s = scores(m, ted)
    pred = [1 if p >= best[1] else 0 for p in te_s]
    per_item = [{'task': tg.get('task_id'), 'y': y, 'score': round(sc, 6), 'pred': int(pr)}
                for tg, y, sc, pr in zip(test_g, te_ys, te_s, pred)]
    return (roc_auc_score(te_ys, te_s), f1_score(te_ys, pred),
            precision_score(te_ys, pred), recall_score(te_ys, pred), per_item, best[1])


def main():
    graphs = [json.loads(l) for l in open('ref/dataset/memorygraft_augmented.jsonl')]
    att = defaultdict(list)
    ben = defaultdict(list)
    for g in graphs:
        seed = g['task_id'].split('_v')[0]
        (att if g['label'] else ben)[seed].append(g)
    seeds = sorted(att) + sorted(ben)
    labels = [1] * len(att) + [0] * len(ben)
    gkf = GroupKFold(n_splits=5)
    aucs, f1s, pres, recs = [], [], [], []
    manifest = []
    for fold, (tr_idx, te_idx) in enumerate(gkf.split(seeds, labels, groups=seeds)):
        tr_seeds = [seeds[i] for i in tr_idx]
        te_g = [g for s in (seeds[i] for i in te_idx) for g in (att[s] if s in att else ben[s])]
        # ---- stratified val: keep BOTH classes by group ----
        att_tr = [s for s in tr_seeds if s in att]
        ben_tr = [s for s in tr_seeds if s in ben]
        random.seed(42 + fold)
        random.shuffle(att_tr)
        random.shuffle(ben_tr)
        n_att_val = max(1, len(att_tr) // 5)
        n_ben_val = max(1, len(ben_tr) // 10)
        val_seeds = set(att_tr[:n_att_val]) | set(ben_tr[:n_ben_val])
        va_g = [g for s in val_seeds for g in (att[s] if s in att else ben[s])]
        tr_g = [g for s in tr_seeds if s not in val_seeds
                for g in (att[s] if s in att else ben[s])]
        a, f, p, r, per_item, thr = fit_fold(tr_g, va_g, te_g, seed=42 + fold)
        aucs.append(a)
        f1s.append(f)
        pres.append(p)
        recs.append(r)
        print(f'fold{fold}: AUC={a:.4f} F1={f:.4f} P={p:.3f} R={r:.3f} '
              f'(val att={n_att_val} ben={n_ben_val}, test={len(te_g)}, thr={thr:.4f})', flush=True)
        manifest.append({'fold': fold, 'val_threshold': thr, 'items': per_item})
    import json as _json
    out = 'code/repro/out/mg_grouped_cv_manifest.json'
    import os as _os
    _os.makedirs('code/repro/out', exist_ok=True)
    _json.dump({'auc': aucs, 'f1': f1s, 'pre': pres, 'rec': recs,
                'auc_mean': statistics.mean(aucs), 'auc_std': statistics.stdev(aucs),
                'f1_mean': statistics.mean(f1s), 'f1_std': statistics.stdev(f1s),
                'folds': manifest}, open(out, 'w'), indent=1)
    print(f'manifest saved: {out}')
    print(f'\nGroupKFold(stratified-val) 5-fold: AUC={statistics.mean(aucs):.4f}±{statistics.stdev(aucs):.4f} '
          f'F1={statistics.mean(f1s):.4f}±{statistics.stdev(f1s):.4f} (ddof=1)')


if __name__ == '__main__':
    main()
