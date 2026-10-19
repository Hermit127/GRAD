#!/usr/bin/env python3
"""Shortcut audit for the new x4 GRAD runs (InjecAgent / ASB / AgentHarm).

Probes on the SAME family-grouped test splits the models were evaluated on:
  - node-count AUROC
  - single-token document frequency separability (top discriminative tokens)
  - text 1-NN (char-trigram Jaccard) AUROC
If a probe matches or exceeds the model AUROC, the cell is flagged
shortcut-confounded and cannot carry a discriminative claim.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SP = REPO / 'code/repro/out/unified_gnn/fix2_splits_x4'
RUNS = REPO / 'code/repro/out/unified_gnn_x4/runs'
OUT = REPO / 'LOGS/final_five_tables/raw_results/x4_shortcut_audit.json'


def auroc(scores, labels):
    pairs = sorted(zip(scores, labels))
    n1 = sum(labels); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return None
    order = sorted(range(len(pairs)), key=lambda i: pairs[i][0])
    ranks = [0.0]*len(pairs); i = 0
    while i < len(order):
        j = i
        while j+1 < len(order) and pairs[order[j+1]][0] == pairs[order[i]][0]:
            j += 1
        avg = (i+1+j+1)/2
        for k in range(i, j+1):
            ranks[order[k]] = avg
        i = j+1
    s = sum(ranks[i] for i in range(len(labels)) if labels[i] == 1)
    return (s - n1*(n1+1)/2) / (n1*n0)


def graph_text(g):
    parts = []
    for n in g.get('nodes', []):
        parts.append(str((n.get('features') or {}).get('subtask', '')))
    return ' '.join(parts).lower()


def main():
    report = {}
    for name in ('injecagent', 'asb', 'agentharm'):
        test = [json.loads(l) for l in open(SP / name / 'test.jsonl') if l.strip()]
        labels = [int(g['label']) for g in test]
        texts = [graph_text(g) for g in test]
        node_counts = [len(g['nodes']) for g in test]
        probes = {'node_count': auroc(node_counts, labels)}
        # single token: best balanced-accuracy token from a small candidate set
        tokens = ['my', 'please', 'important', 'ignore', 'urgent', 'secret',
                  'password', 'send', 'email', 'http', 'www', 'system']
        best = (0.0, None)
        for tok in tokens:
            pred = [1 if tok in t else 0 for t in texts]
            pos = sum(1 for p, y in zip(pred, labels) if p and y)
            neg = sum(1 for p, y in zip(pred, labels) if p and not y)
            tp = pos / max(1, sum(labels))
            fp = neg / max(1, len(labels)-sum(labels))
            ba = (tp + (1-fp)) / 2
            if ba > best[0]:
                best = (ba, tok)
        probes['best_single_token'] = {'token': best[1], 'balanced_acc': round(best[0], 3)}
        # text 1-NN (char trigram jaccard, leave-one-out)
        def tri(s):
            return {s[i:i+3] for i in range(max(0, len(s)-2))}
        sets = [tri(t) for t in texts]
        preds = []
        for i in range(len(sets)):
            sims = []
            for j in range(len(sets)):
                if i == j:
                    continue
                inter = len(sets[i] & sets[j]); union = len(sets[i] | sets[j]) or 1
                sims.append((inter/union, j))
            sims.sort(reverse=True)
            preds.append(labels[sims[0][1]] if sims else 0)
        tp = sum(1 for p, y in zip(preds, labels) if p and y)
        fp = sum(1 for p, y in zip(preds, labels) if p and not y)
        probes['text_1nn'] = {'balanced_acc': round((tp/max(1,sum(labels)) + (1-fp/max(1,len(labels)-sum(labels))))/2, 3)}
        # model AUROC from the runs
        models = {}
        for seed in (211, 212, 213):
            p = RUNS / f'{name}_s{seed}.json'
            if p.exists():
                d = json.loads(p.read_text())
                models[f's{seed}'] = (d.get('test_graph') or {}).get('auroc')
        report[name] = {'n_test': len(test), 'probes': probes, 'grad_auroc': models}
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
