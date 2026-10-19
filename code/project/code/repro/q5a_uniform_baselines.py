#!/usr/bin/env python3
"""E-Q5a: uniform-split baseline rerun on MG GroupKFold OOF folds (same as GRAD)."""
import json, sys
from collections import Counter, defaultdict
import numpy as np
sys.path.insert(0, 'code')
sys.path.insert(0, 'code/repro')
from agentarmor_repro import agentarmor_graph, NO_DANGER_SET
from compare_all_baselines import baseline_agenttrust, baseline_l1_regex
from agent_sentry_repro import per_action_features, ACTION
import xgboost as xgb
from sklearn.metrics import roc_auc_score, f1_score

mg = [json.loads(l) for l in open('ref/dataset/memorygraft_augmented.jsonl')]
by_task = {g['task_id']: g for g in mg}
mani = json.load(open('code/repro/out/mg_grouped_cv_manifest.json'))
res = defaultdict(list)
for f in mani['folds']:
    test_ids = {it['task'] for it in f['items']}
    te = [by_task[t] for t in test_ids]
    tr = [g for g in mg if g['task_id'] not in test_ids]
    ys = [g['label'] for g in te]
    pa = [agentarmor_graph(g, NO_DANGER_SET) for g in te]
    pat = baseline_agenttrust(te)
    pl1 = baseline_l1_regex(te)
    cnt = Counter((tuple(sorted(n['type'] for n in g['nodes'])), g['label']) for g in tr)
    def sim(a, b):
        sa, sb = set(a), set(b)
        return len(sa & sb) / len(sa | sb) if sa | sb else 0
    pnn = []
    for g in te:
        ts = tuple(sorted(n['type'] for n in g['nodes']))
        bt, bs = None, -1
        for (tt, lab), _c in cnt.items():
            s = sim(ts, tt)
            if s > bs:
                bs, bt = s, lab
        pnn.append(bt)
    # Agent-Sentry 17-dim (graph max-pool), trained on this fold's train
    Xtr, ytr = [], []
    for g in tr:
        for i, nd in enumerate(g['nodes']):
            if nd['type'] in ACTION:
                Xtr.append(per_action_features(g, i)); ytr.append(g['label'])
    ps_as = []
    if Xtr:
        mas = xgb.XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.1,
                                random_state=42, eval_metric='logloss')
        mas.fit(Xtr, ytr)
        for g in te:
            fs = [per_action_features(g, i) for i, nd in enumerate(g['nodes'])
                  if nd['type'] in ACTION]
            ps_as.append(float(max(mas.predict_proba(fs)[:, 1])) if fs else 0.5)
    else:
        ps_as = [0.5] * len(te)
    # node-count (threshold from train)
    bst = (0, 2)
    for t in range(2, 15):
        fscore = f1_score([g['label'] for g in tr],
                          [1 if len(g['nodes']) > t else 0 for g in tr])
        if fscore > bst[0]:
            bst = (fscore, t)
    ps_nc = [1 if len(g['nodes']) > bst[1] else 0 for g in te]
    for name, preds in [('AgentArmor', pa), ('AgentTrust', pat), ('L1-regex', pl1),
                        ('1NN', pnn), ('Agent-Sentry', ps_as),
                        (f'node-count>{bst[1]}', ps_nc)]:
        preds = [float(x) for x in preds]
        a = roc_auc_score(ys, preds) if len(set(preds)) > 1 else 0.5
        fl = f1_score(ys, [1 if x >= 0.5 else 0 for x in preds])
        res[name].append((a, fl))
    print(f'fold{f["fold"]}: ' + ' '.join(f'{k}={res[k][-1][0]:.3f}' for k in res), flush=True)
print('\n=== E-Q5a uniform-split baselines (same folds as GRAD, 5-fold mean) ===')
for k, v in res.items():
    aucs = [x[0] for x in v]
    f1s = [x[1] for x in v]
    print(f'  {k:14s}: AUC={np.mean(aucs):.3f}+/-{np.std(aucs, ddof=1):.3f} '
          f'F1={np.mean(f1s):.3f}+/-{np.std(f1s, ddof=1):.3f}')
print('  GRAD(ref)    : AUC=0.9973+/-0.0032 F1=0.9870+/-0.0189')
json.dump({k: {'auc': [x[0] for x in v], 'f1': [x[1] for x in v]} for k, v in res.items()},
          open('code/repro/out/q5a_uniform_baselines.json', 'w'), indent=1)
