#!/usr/bin/env python3
"""E-Q5b: Decomp uniform-split baselines on the TRUE size-matched subset
(same split as Ours text 0.930). Non-torch process: BoW, Agent-Sentry-17d,
AgentArmor (full + double-drop view), 1NN, node-count. Struct GNN runs in a
separate torch process and is merged afterwards.
"""
import json, re, random, sys
from collections import Counter, defaultdict
import numpy as np
import xgboost as xgb
sys.path.insert(0, 'code')
sys.path.insert(0, 'code/repro')
from agentarmor_repro import agentarmor_graph, NO_DANGER_SET
from agent_sentry_repro import per_action_features, ACTION
from sklearn.metrics import roc_auc_score, f1_score

_tr = [json.loads(l) for l in open('ref/dataset/decomp_graphs_train_mixed.jsonl')]
_te = [json.loads(l) for l in open('ref/dataset/decomp_graphs_test_mixed.jsonl')]

def size_matched(gs, seed=42):
    rnd = random.Random(seed)
    by_n = defaultdict(lambda: {'a': [], 'b': []})
    for g in gs:
        (by_n[len(g['nodes'])]['a'] if g['label'] else by_n[len(g['nodes'])]['b']).append(g)
    out = []
    for n in sorted(by_n):
        d = by_n[n]
        if not d['a'] or not d['b']:
            continue
        k = min(len(d['a']), len(d['b']))
        rnd.shuffle(d['a']); rnd.shuffle(d['b'])
        out += d['a'][:k] + d['b'][:k]
    return out

tr, te = size_matched(_tr), size_matched(_te)
print(f'size-matched train {len(tr)} test {len(te)}')
ys = [g['label'] for g in te]

TOK = re.compile(r"[a-z0-9]+")
def text_of(g):
    return ' '.join(str(n.get('features', {}).get('subtask', '')) for n in g['nodes'])

# ---- BoW (Ours text, reference) ----
vocab = Counter()
for g in tr:
    vocab.update(TOK.findall(text_of(g).lower()))
keep = {w for w, c in vocab.items() if c >= 3}
idx = {w: i for i, w in enumerate(sorted(keep))}
def fe(g):
    v = [0] * len(idx)
    for w in TOK.findall(text_of(g).lower()):
        if w in idx:
            v[idx[w]] += 1
    return v
m_bow = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42,
                          scale_pos_weight=(sum(1 for g in tr if g['label'] == 0) /
                          max(1, sum(1 for g in tr if g['label'] == 1))), eval_metric='logloss')
m_bow.fit([fe(g) for g in tr], [g['label'] for g in tr])
ps_bow = m_bow.predict_proba([fe(g) for g in te])[:, 1]

# ---- Agent-Sentry 17-dim (graph max-pool of per-action probs) ----
def sentry_feats(g):
    return [per_action_features(g, i) for i, nd in enumerate(g['nodes']) if nd['type'] in ACTION]
Xtr, ytr = [], []
for g in tr:
    for f in sentry_feats(g):
        Xtr.append(f); ytr.append(g['label'])
m_as = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1, random_state=42,
                         eval_metric='logloss')
if Xtr:
    m_as.fit(Xtr, ytr)
    ps_as = []
    for g in te:
        fs = sentry_feats(g)
        ps_as.append(max(m_as.predict_proba(fs)[:, 1]) if fs else 0.5)
else:
    ps_as = [0.5] * len(te)

# ---- AgentArmor (full + double-drop view) ----
ps_aa_full = [agentarmor_graph(g) for g in te]
ps_aa_drop = [agentarmor_graph(g, NO_DANGER_SET) for g in te]

# ---- 1NN tool-seq + node-count ----
cnt = Counter((tuple(sorted(n['type'] for n in g['nodes'])), g['label']) for g in tr)
def sim(a, b):
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0
ps_1nn = []
for g in te:
    ts = tuple(sorted(n['type'] for n in g['nodes']))
    bt, bs = None, -1
    for (tt, lab), _ in cnt.items():
        s = sim(ts, tt)
        if s > bs:
            bs, bt = s, lab
    ps_1nn.append(bt)
# node-count threshold from train (max F1)
best = (0, 0)
for t in range(2, 15):
    p = [1 if len(g['nodes']) > t else 0 for g in tr]
    f = f1_score([g['label'] for g in tr], p)
    if f > best[0]:
        best = (f, t)
ps_nc = [1 if len(g['nodes']) > best[1] else 0 for g in te]

def rep(name, ps):
    ps = [float(x) for x in ps]
    auc = roc_auc_score(ys, ps) if len(set(ps)) > 1 else 0.5
    f = f1_score(ys, [1 if x >= 0.5 else 0 for x in ps])
    print(f'  {name:22s}: AUC={auc:.4f} F1={f:.4f}')

print('=== E-Q5b Decomp size-matched uniform-split baselines ===')
rep('Ours text (BoW)', ps_bow)
rep('Agent-Sentry 17d', ps_as)
rep('AgentArmor full', ps_aa_full)
rep('AgentArmor 2drop', ps_aa_drop)
rep('1NN node-type-set', ps_1nn)
rep(f'node-count>{best[1]}', ps_nc)
# paired bootstrap: AUC(Ours) - AUC(next best) CI
rng = np.random.default_rng(42)
ys_a = np.array(ys)
best_other = max([ps_as, [float(x) for x in ps_aa_full], [float(x) for x in ps_1nn]],
                 key=lambda p: roc_auc_score(ys_a, p))
diffs = []
for _ in range(2000):
    idx = rng.integers(0, len(ys_a), len(ys_a))
    if len(set(ys_a[idx])) < 2:
        continue
    diffs.append(roc_auc_score(ys_a[idx], np.array(ps_bow)[idx]) -
                 roc_auc_score(ys_a[idx], np.array(best_other)[idx]))
lo, hi = np.percentile(diffs, [2.5, 97.5])
print(f'paired bootstrap AUC diff (Ours - next best): mean={np.mean(diffs):.4f} '
      f'95% CI [{lo:.4f}, {hi:.4f}]')
json.dump({'ys': ys, 'bow': list(map(float, ps_bow)), 'as': list(map(float, ps_as)),
           'aa_full': ps_aa_full, 'aa_drop': ps_aa_drop, 'nn': ps_1nn, 'nc': ps_nc,
           'test_ids': [g.get('metadata', {}).get('task', g.get('task_id', '?')) for g in te],
           'train_n': len(tr), 'test_n': len(te),
           'paired_bootstrap_auc_diff': {'mean': float(np.mean(diffs)),
                                         'ci95': [float(lo), float(hi)]}},
          open('code/repro/out/q5b_decomp_baselines.json', 'w'))
print('saved q5b_decomp_baselines.json (with test IDs + paired CI)')
