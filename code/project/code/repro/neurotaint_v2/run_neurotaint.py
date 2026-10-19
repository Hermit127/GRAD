#!/usr/bin/env python
"""Run NeuroTaint v2 on the frozen splits (zero API).

Datasets: fix2 decomp/mg/minja/mcp; MRT cross-task; HRA v2 base cell as a
negative control. Prereg: LOGS/neurotaint_v2_prereg_20260914.md.

Usage:
  python run_neurotaint.py pilot
  python run_neurotaint.py full
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path('.')
PKG = REPO / 'code/repro/static_big_table_v1'
HERE = REPO / 'code/repro/neurotaint_v2'
FIX2 = REPO / 'code/repro/out/static_big_table_v1/fix2_20260910'
MRT = REPO / 'results/mrt_cross_task_v1'
LOGS = REPO / 'LOGS'

sys.path.insert(0, str(PKG))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / 'code/repro'))

from evaluation import auroc  # noqa: E402
from groups import BENCH_ORDER, load_benchmark  # noqa: E402
from neurotaint import audit, get_model  # noqa: E402
from policy import unmapped_types  # noqa: E402


def tpr_at_fpr(scores, labels, target=0.10):
    pairs = [(float(s), int(y)) for s, y in zip(scores, labels)
             if not math.isnan(float(s))]
    n_ben = sum(1 for _, y in pairs if y == 0)
    n_pos = sum(1 for _, y in pairs if y == 1)
    if not n_ben or not n_pos:
        return float('nan')
    best = 0.0
    for t in sorted({s for s, _ in pairs}):
        tp = sum(1 for s, y in pairs if y == 1 and s >= t)
        fp = sum(1 for s, y in pairs if y == 0 and s >= t)
        if fp / n_ben <= target:
            best = max(best, tp / n_pos)
    return best


def roles_for(bench, payload, fold):
    from collections import defaultdict
    graphs = load_benchmark(bench)
    idx_by_group = defaultdict(list)
    for i, g in enumerate(graphs):
        idx_by_group[g['_group']].append(i)
    gr = payload['group_role'] if payload['mode'] == 'single' \
        else payload['folds'][fold]['group_role']
    roles = {r: [] for r in ('train', 'dev', 'gate_fit', 'threshold_cal',
                             'test')}
    for gid, role in gr.items():
        roles[role].extend(idx_by_group[gid])
    return {r: [graphs[i] for i in v] for r, v in roles.items()}


def fix2_test(bench):
    payload = json.loads((FIX2 / 'split_ids' / f'{bench}.json').read_text())
    n = 1 if payload['mode'] == 'single' else len(payload['folds'])
    return [roles_for(bench, payload, f)['test'] for f in range(n)]


def mrt_test():
    tasks = ('american_airlines_2', 'api_key_calendar_agendas_2',
             'bank_transfer_2', 'expenditure_accounting',
             'spam_filter_update')
    out = []
    for task in tasks:
        p = MRT / f'fold_{task}' / 'admission.jsonl'
        out.append([json.loads(l) for l in open(p) if l.strip()])
    return out


def hra_control():
    sys.path.insert(0, str(REPO / 'code/repro/hra_v2'))
    from gen_hra_v2 import cell
    return [cell(211, 2, 2, 0.2, 0.1, sizes=(100, 20, 100))['test']]


def minja_obs_data():
    """Fair MINJA view for sink-based auditing: obs-v1 confirmation traces
    (action + observation nodes, tool_execute sinks present)."""
    sys.path.insert(0, str(REPO / 'code/repro'))
    # minja_obs_view imports channels at module top (xgboost/torch not needed
    # by trace_to_graph_obs); stub it so the neurotaint venv stays minimal.
    import types
    if 'channels' not in sys.modules:
        stub = types.ModuleType('channels')
        stub.StructChannel = stub.TextChannel = object
        stub.static_fuse = lambda a, b: a  # noqa: E731
        sys.modules['channels'] = stub
    import minja_obs_view as OBS
    manifest = json.load(
        open(REPO / 'code/repro/out/minja_obs_confirm/manifest.json'))
    graphs = []
    for task in manifest['tasks']:
        for cond in manifest['conditions']:
            for rep in manifest['replicates']:
                p = (REPO / 'code/repro/out/minja_obs_confirm/traces' /
                     f"{task['task_id']}__{cond}__r{rep}.json")
                if not p.exists():
                    continue
                rec = json.load(open(p))
                if rec.get('status') != 'ok':
                    continue
                if cond not in ('attack_env', 'benign'):
                    continue
                g = OBS.trace_to_graph_obs(rec, obs=True)
                g['label'] = int(cond == 'attack_env')
                g['task_id'] = f"{task['task_id']}__{cond}__r{rep}"
                graphs.append(g)
    return [graphs]


def eval_folds(folds, model, restricted=False, use_causal=True,
               verbose=False):
    rows = []
    stages = Counter()
    for fi, test in enumerate(folds):
        scores, labels = [], []
        for g in test:
            r = audit(g, model, restricted=restricted,
                      use_causal=use_causal)
            scores.append(r['score'])
            labels.append(int(g['label']))
            stages[r['stage']] += 1
            if verbose and len(scores) <= 5:
                print('   sample', g.get('task_id'), 'label',
                      int(g['label']), '->', r)
        rows.append({'fold': fi, 'n': len(test),
                     'auroc': auroc(scores, labels),
                     'tpr_at_fpr10': tpr_at_fpr(scores, labels)})
    return rows, stages


def summarize(name, rows, stages, extra):
    aucs = [r['auroc'] for r in rows if r['auroc'] is not None]
    tprs = [r['tpr_at_fpr10'] for r in rows
            if r['tpr_at_fpr10'] is not None and
            not math.isnan(r['tpr_at_fpr10'])]
    return {'dataset': name, 'n_folds': len(rows),
            'auroc_mean': sum(aucs) / len(aucs) if aucs else None,
            'auroc_per_fold': [round(a, 4) for a in aucs],
            'tpr_at_fpr10_mean': sum(tprs) / len(tprs) if tprs else None,
            'stage_hist': dict(stages.most_common()), **extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['pilot', 'full'])
    ap.add_argument('--datasets', default='decomp,mg,minja,mcp,mrt,hra')
    ap.add_argument('--restricted', action='store_true',
                    help='also run the DCPG-restricted variant')
    a = ap.parse_args()
    t0 = time.time()
    model = get_model()
    dest = LOGS / f'neurotaint_v2_{a.mode}_20260914.json'
    report = {'mode': a.mode, 'prereg':
              'LOGS/neurotaint_v2_prereg_20260914.md',
              'thresholds': {'theta_str': 0.15, 'theta_sem': 0.60,
                             'theta_sem_rag': 0.85, 'theta_cov': 0.10},
              'notes': ['Tier1 canary not applicable offline (no captured '
                        'canaries)', 'causal analyzer is a deterministic '
                        'proxy, not the paper counterfactual'],
              'datasets': {}, 'restricted_variant': {}}
    names = a.datasets.split(',')
    if a.mode == 'pilot':
        names = ['mg', 'mrt', 'hra']
    elif dest.exists():
        prev = json.loads(dest.read_text())
        report['datasets'] = prev.get('datasets', {})
        report['restricted_variant'] = prev.get('restricted_variant', {})
    for name in names:
        if name in BENCH_ORDER:
            folds = fix2_test(name)
            graphs = [g for f in folds for g in f]
        elif name == 'mrt':
            folds = mrt_test()
            graphs = [g for f in folds for g in f]
        elif name == 'hra':
            folds = hra_control()
            graphs = [g for f in folds for g in f]
        elif name == 'minja_obs':
            folds = minja_obs_data()
            graphs = [g for f in folds for g in f]
        else:
            continue
        fold_iter = folds[:1] if a.mode == 'pilot' else folds
        rows, stages = eval_folds(fold_iter, model, verbose=a.mode == 'pilot')
        extra = {'unmapped_types': unmapped_types(graphs),
                 'n_test_graphs': sum(len(f) for f in fold_iter)}
        report['datasets'][name] = summarize(name, rows, stages, extra)
        print(f'== {name}: auroc_mean='
              f"{report['datasets'][name]['auroc_mean']} "
              f"stages={dict(stages)}", flush=True)
        if a.restricted:
            rrows, rstages = eval_folds(fold_iter, model, restricted=True)
            report['restricted_variant'][name] = summarize(
                name, rrows, rstages, {})
    report['elapsed_s'] = round(time.time() - t0, 1)
    dest.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print('wrote', dest, f'(elapsed {report["elapsed_s"]}s)')


if __name__ == '__main__':
    main()
