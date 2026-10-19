#!/usr/bin/env python3
"""Fill table D: run AgentTrust / AEGIS-22regex / Agent-Sentry across all 5 attack families."""
import json, re, sys
from pathlib import Path
sys.path.insert(0, './code')
sys.path.insert(0, './code/repro')

from compare_all_baselines import baseline_agenttrust, baseline_l1_regex
from agent_sentry_repro import fit_predict, report
from a3_feature_bench import AEGIS_PATTERNS

R = '.'

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]

def aegis_22(g):
    """AEGIS 22-regex per-call: block if any node text matches any pattern."""
    for n in g['nodes']:
        text = ' '.join(str(v) for v in n.get('features', {}).values())
        for _, pat in AEGIS_PATTERNS:
            if re.search(pat, text):
                return 1
    return 0

def minja_to_graphs():
    d = json.load(open(f'{R}/ref/dataset/minja_real_traces.json'))
    out = []
    for i, t in enumerate(d):
        nodes, edges, prev = [], [], None
        for j, s in enumerate(t.get('steps', [])):
            nid = f'm{j}'
            nodes.append({'id': nid, 'type': s.get('type', 'tool_execute'), 'taint': 0,
                          'turn': j, 'features': {'subtask': str(s.get('action', ''))}})
            if prev:
                edges.append({'source': prev, 'target': nid, 'type': 'data_flow'})
            prev = nid
        out.append({'task_id': f'minja_{i}', 'label': t['label'], 'nodes': nodes, 'edges': edges})
    return out

def metrics(labels, preds):
    tp = sum(1 for l, p in zip(labels, preds) if l and p)
    fp = sum(1 for l, p in zip(labels, preds) if not l and p)
    fn = sum(1 for l, p in zip(labels, preds) if l and not p)
    tn = sum(1 for l, p in zip(labels, preds) if not l and not p)
    pre = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    return f'F1={2*pre*rec/(pre+rec) if pre+rec else 0:.3f} (tp{tp}/fp{fp}/fn{fn})'

datasets = {}

# MG: cross-template holdout 80 test (same split as GNN)
import random
from collections import defaultdict
mg = load(f'{R}/ref/dataset/memorygraft_augmented.jsonl')
def sid(t): return t.split('_v')[0]
att, ben = defaultdict(list), defaultdict(list)
for g in mg: (att if g['label'] else ben)[sid(g['task_id'])].append(g)
as_, bs = sorted(att), sorted(ben)
random.seed(42); random.shuffle(as_); random.shuffle(bs)
mg_tr, mg_te = [], []
for s in as_[:8]: mg_tr += att[s]
for s in as_[8:]: mg_te += att[s]
for s in bs[:80]: mg_tr += ben[s]
for s in bs[80:]: mg_te += ben[s]
datasets['MG'] = (mg_tr, mg_te)

# Decomp
de_tr = load(f'{R}/ref/dataset/decomp_graphs_train_mixed.jsonl')
de_te = load(f'{R}/ref/dataset/decomp_graphs_test_mixed.jsonl')
datasets['Decomp'] = (de_tr, de_te)

# MCP matched
mc_tr = load(f'{R}/ref/dataset/mcp_safetybench_defense_matched_train.jsonl')
mc_te = load(f'{R}/ref/dataset/mcp_safetybench_defense_matched_test.jsonl')
datasets['MCP'] = (mc_tr, mc_te)

# TS (60 graphs, no split - small-sample note)
ts = load(f'{R}/ref/dataset/toolshadow_preexec_graphs.jsonl')
datasets['TS'] = ([], ts)

# MINJA (converted)
mj = minja_to_graphs()
datasets['MINJA'] = ([], mj)

print('=== AgentTrust (per-session chain) ===')
for name, (tr, te) in datasets.items():
    preds = baseline_agenttrust(te)
    labels = [g['label'] for g in te]
    print(f'  {name:8s} {metrics(labels, preds)}')

print('=== AEGIS 22-regex (per-call) ===')
for name, (tr, te) in datasets.items():
    preds = [aegis_22(g) for g in te]
    labels = [g['label'] for g in te]
    print(f'  {name:8s} {metrics(labels, preds)}')

print('=== Agent-Sentry L1 (XGBoost 17-dim) ===')
for name, (tr, te) in datasets.items():
    if not tr:
        print(f'  {name:8s} (skip: no train set, small sample)')
        continue
    if len(tr) < 30:
        print(f'  {name:8s} (skip: train n={len(tr)} too small)')
        continue
    try:
        ys, ps, gpred = fit_predict(tr, te)
        print(f'  {name:8s} {report("", ys, ps, gpred)}')
    except Exception as e:
        print(f'  {name:8s} ERR {e}')
