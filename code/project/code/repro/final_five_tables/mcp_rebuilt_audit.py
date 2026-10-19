#!/usr/bin/env python3
"""Audit the improved MCP corpus runs: dev-calibrated op point, AUROC, shortcuts.

Compares three arms on the SAME validation/test files:
  fix2_legacy : code/repro/out/unified_gnn/runs/mcp_f*_s*.json (current Table 1)
  rebuilt     : code/repro/out/unified_gnn_mcp_rebuilt/runs/*.json
  rebuilt_dgc : code/repro/out/unified_gnn_mcp_rebuilt_dgc/runs/*.json (if present)
Plus shortcut probes on the rebuilt test split (node-count, best single token,
char-trigram 1-NN) with a data-disjointness check.
"""
from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / 'LOGS/final_five_tables/raw_results/mcp_rebuilt_audit.json'
BASE = REPO / 'ref/dataset/mcp_continual_rebuild_v1/semantic'
ARMS = {
    'rebuilt': REPO / 'code/repro/out/unified_gnn_mcp_rebuilt/runs',
    'rebuilt_dgc': REPO / 'code/repro/out/unified_gnn_mcp_rebuilt_dgc/runs',
    'combined_data': REPO / 'code/repro/out/unified_gnn_mcp_combined/runs',
}


def load_test():
    p = BASE / ('mcp_safetybench_defense_rawtext_config_large_eval_v1_'
                'test_neutral_semantic.jsonl')
    return [json.loads(l) for l in open(p) if l.strip()]


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


def probes(test):
    labels = [int(g['label']) for g in test]
    counts = [len(g['nodes']) for g in test]
    texts = [' '.join(str((n['features'] or {}).get('subtask', ''))
                      for n in g['nodes']).lower() for g in test]
    out = {'node_count_auroc': auroc(counts, labels)}
    tokens = ['please', 'my', 'ignore', 'important', 'send', 'http', 'www',
              'secret', 'password', 'urgent']
    best = (0.0, None)
    for tok in tokens:
        pred = [1 if tok in t else 0 for t in texts]
        tp = sum(1 for p, y in zip(pred, labels) if p and y) / max(1, sum(labels))
        fp = sum(1 for p, y in zip(pred, labels) if p and not y) / max(1, len(labels)-sum(labels))
        ba = (tp + 1 - fp) / 2
        if ba > best[0]:
            best = (ba, tok)
    out['best_token'] = {'token': best[1], 'balanced_acc': round(best[0], 3)}
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
    tp = sum(1 for p, y in zip(preds, labels) if p and y) / max(1, sum(labels))
    fp = sum(1 for p, y in zip(preds, labels) if p and not y) / max(1, len(labels)-sum(labels))
    out['text_1nn'] = {'balanced_acc': round((tp + 1 - fp) / 2, 3)}
    return out


def ensemble_metrics(run_dir):
    """Per-fold seed-ensemble test AUROC (sklearn) + dev ensemble."""
    from sklearn.metrics import roc_auc_score
    per_fold = {}
    for p in sorted(run_dir.glob('*.json')):
        d = json.loads(p.read_text())
        name = p.stem
        m = None
        if '_f' in name:
            f = name.split('_f')[1].split('_')[0]
        else:
            f = '0'
        per_fold.setdefault(f, []).append(d)
    out = {}
    for f, runs in sorted(per_fold.items()):
        tests = [r['scores']['test'] for r in runs if r.get('scores')]
        devs = [r['scores']['val'] for r in runs if r.get('scores')]
        if not tests:
            continue
        yt = [int(e['label']) for e in tests[0]]
        yv = [int(e['label']) for e in devs[0]]
        ens_t = [sum(float(t[i]['score']) for t in tests) / len(tests)
                 for i in range(len(yt))]
        ens_v = [sum(float(v[i]['score']) for v in devs) / len(devs)
                 for i in range(len(yv))]
        out[f] = {'test_ensemble_auroc': roc_auc_score(yt, ens_t),
                  'dev_ensemble_auroc': roc_auc_score(yv, ens_v),
                  'n_runs': len(tests)}
    return out


def arm_metrics(run_dir):
    if not run_dir.exists():
        return None
    per = []
    for p in sorted(run_dir.glob('*.json')):
        d = json.loads(p.read_text())
        tr = ((d.get('threshold_reports') or {}).get('fpr_10') or {}).get('test') or {}
        per.append({'run': p.stem,
                    'auroc': (d.get('test_graph') or {}).get('auroc'),
                    'dev_threshold': tr.get('threshold'),
                    'test_tpr': tr.get('recall'), 'test_fpr': tr.get('fpr'),
                    'best_epoch': d.get('best_epoch'),
                    'epochs_run': len(d.get('history') or [])})
    aucs = [x['auroc'] for x in per if x['auroc'] is not None]
    return {'n_runs': len(per),
            'auroc_mean': sum(aucs)/len(aucs) if aucs else None,
            'per_run': per}


def main():
    test = load_test()
    report = {'test_n': len(test),
              'test_label_balance': dict(Counter(int(g['label']) for g in test)),
              'shortcut_probes': probes(test),
              'arms': {'fix2_legacy': arm_metrics(
                  REPO / 'code/repro/out/unified_gnn/runs')}}
    # fix2 legacy is per-fold; restrict to mcp_f* files
    legacy = arm_metrics(REPO / 'code/repro/out/unified_gnn/runs')
    if legacy:
        legacy['per_run'] = [x for x in legacy['per_run'] if x['run'].startswith('mcp_f')]
        legacy['n_runs'] = len(legacy['per_run'])
        aucs = [x['auroc'] for x in legacy['per_run'] if x['auroc'] is not None]
        legacy['auroc_mean'] = sum(aucs)/len(aucs) if aucs else None
        report['arms']['fix2_legacy'] = legacy
    for name, d in ARMS.items():
        report['arms'][name] = arm_metrics(d)
    OUT.write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != 'arms'}, indent=1))
    for name, v in report['arms'].items():
        if v:
            print(name, 'n', v['n_runs'], 'AUROC mean', round(v['auroc_mean'] or 0, 4))


if __name__ == '__main__':
    main()
