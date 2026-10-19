#!/usr/bin/env python
"""Rerun the corrected NeuroTaint on all frozen splits (zero API).

Variants written to one report:
  primary        restricted DCPG reachability, word-level LCS, no proxy
  temporal       unrestricted source-before-sink view (diagnostic)
  lcs_char       character-level LCS sensitivity (shows the unit choice effect)
  proxy_ablation the old keyword causal proxy, kept only as an ablation

Usage:
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    python run_neurotaint_fixed.py full
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
NT = REPO / 'code/repro/neurotaint_v2'
PKG = REPO / 'code/repro/static_big_table_v1'
OUT = REPO / 'code/repro/out/baseline_fidelity_fix3_20260916/neurotaint_fixed'
LOGS = REPO / 'LOGS'

sys.path.insert(0, str(PKG))
sys.path.insert(0, str(NT))
sys.path.insert(0, str(HERE))

from evaluation import auroc  # noqa: E402
from groups import BENCH_ORDER  # noqa: E402
from neurotaint import get_model  # noqa: E402
from run_neurotaint import (fix2_test, hra_control, minja_obs_data,  # noqa: E402
                            mrt_test, tpr_at_fpr)
import neurotaint_fixed as fx  # noqa: E402
from policy import unmapped_types  # noqa: E402

DATASETS = ('decomp', 'mg', 'minja', 'mcp', 'mrt', 'hra', 'minja_obs',
            'injecagent', 'asb', 'agentharm')
X4_DATA = REPO / 'code/repro/x4_datasets/data'


def x4_folds(name):
    graphs = [json.loads(line) for line in
              open(X4_DATA / f'{name}.jsonl') if line.strip()]
    fams = sorted({g['family'] for g in graphs})
    n_test = max(1, int(round(len(fams) * 0.2)))
    test_f = set(fams[-n_test:])
    return [[g for g in graphs if g['family'] in test_f]]
VARIANTS = {
    'primary': {'restricted': True, 'lcs_unit': 'char', 'proxy': False},
    'temporal': {'restricted': False, 'lcs_unit': 'char', 'proxy': False},
    'lcs_word': {'restricted': True, 'lcs_unit': 'word', 'proxy': False},
    'proxy_ablation': {'restricted': True, 'lcs_unit': 'char', 'proxy': True},
}


def get_folds(name):
    if name in BENCH_ORDER:
        return fix2_test(name)
    if name == 'mrt':
        return mrt_test()
    if name == 'hra':
        return hra_control()
    if name == 'minja_obs':
        return minja_obs_data()
    if name in ('injecagent', 'asb', 'agentharm'):
        return x4_folds(name)
    raise KeyError(name)


def eval_folds(folds, model, variant):
    rows, stages, causal = [], Counter(), Counter()
    for fi, test in enumerate(folds):
        scores, labels = [], []
        for g in test:
            r = fx.audit_fixed(g, model, restricted=variant['restricted'],
                               use_causal_proxy=variant['proxy'],
                               lcs_unit=variant['lcs_unit'])
            scores.append(r['score'])
            labels.append(int(g['label']))
            stages[r['stage']] += 1
            causal[r['lcs_causal_status']['status']] += 1
        rows.append({'fold': fi, 'n': len(test),
                     'auroc': auroc(scores, labels),
                     'tpr_at_fpr10': tpr_at_fpr(scores, labels)})
    aucs = [r['auroc'] for r in rows if r['auroc'] is not None]
    tprs = [r['tpr_at_fpr10'] for r in rows
            if r['tpr_at_fpr10'] is not None and not math.isnan(r['tpr_at_fpr10'])]
    return {'n_folds': len(rows),
            'auroc_mean': sum(aucs) / len(aucs) if aucs else None,
            'auroc_per_fold': [round(a, 4) for a in aucs],
            'tpr_at_fpr10_mean': sum(tprs) / len(tprs) if tprs else None,
            'stage_hist': dict(stages.most_common()),
            'causal_status': dict(causal)}, rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['pilot', 'full'])
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--out-tag', default='v2')
    ap.add_argument('--variants', default=None,
                    help='comma list of variant names; default all')
    args = ap.parse_args()
    t0 = time.time()
    model = get_model()
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        'implementation': 'code/repro/fidelity_fix_20260916/neurotaint_fixed.py',
        'plan': 'LOGS/baseline_fidelity_fix_plan_20260916.md',
        'variants': VARIANTS,
        'notes': [
            'Tier1 canary not applicable offline (no captured canaries)',
            'Sink-Driven Causal Analyzer reported not_applicable: one LLM judge '
            'call per (sink, lineage) over a neutralized context is required '
            '(paper 4.3); re-hosted graphs have no full context or arg objects',
            'restricted=True uses DCPG reachability incl. write->read links',
        ],
        'datasets': {},
    }
    for name in args.datasets.split(','):
        if not name:
            continue
        folds = get_folds(name)
        if args.mode == 'pilot':
            folds = folds[:1]
        graphs = [g for f in folds for g in f]
        entry = {'unmapped_types': unmapped_types(graphs),
                 'n_test_graphs': sum(len(f) for f in folds), 'variants': {}}
        wanted = (args.variants.split(',') if args.variants
                  else list(VARIANTS))
        for vname, variant in VARIANTS.items():
            if vname not in wanted:
                continue
            if args.mode == 'pilot' and vname not in ('primary',):
                continue
            res, _ = eval_folds(folds, model, variant)
            entry['variants'][vname] = res
            print(f'== {name:10s} {vname:14s} auroc='
                  f'{None if res["auroc_mean"] is None else round(res["auroc_mean"], 4)} '
                  f'stages={res["stage_hist"]}', flush=True)
        report['datasets'][name] = entry
    report['elapsed_s'] = round(time.time() - t0, 1)
    dest = (OUT / f'neurotaint_fixed_{args.mode}_20260916_{args.out_tag}.json')
    dest.write_text(json.dumps(report, indent=1, ensure_ascii=False))
    md = LOGS / f'neurotaint_fixed_{args.mode}_20260916_{args.out_tag}.json'
    md.write_text(dest.read_text())
    print('wrote', dest, f'({report["elapsed_s"]}s)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
