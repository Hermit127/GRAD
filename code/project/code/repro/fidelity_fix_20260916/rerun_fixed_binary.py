#!/usr/bin/env python3
"""Same-protocol rerun of corrected external baselines (static table 2).

Reads the frozen fix2 protocols (read-only) from
code/repro/out/static_big_table_v1/fix2_20260910 and the MRT cross-task
protocol from results/mrt_cross_task_v1, then evaluates the corrected
baseline implementations on the identical test roles.

Outputs (new tree, old results untouched):
  code/repro/out/baseline_fidelity_fix3_20260916/<run_id>/<method>/<unit>/
    config.json, predictions.jsonl, metrics.json, run.log

Usage:
  python rerun_fixed_binary.py --method agenttrust_prod --benches decomp --sample 20
  python rerun_fixed_binary.py --method agenttrust_prod --benches decomp,mg,minja,mcp,mrt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PKG = REPO / 'code/repro/static_big_table_v1'
FIX2 = REPO / 'code/repro/out/static_big_table_v1/fix2_20260910'
MRT_ROOT = REPO / 'results/mrt_cross_task_v1'
OUTROOT = REPO / 'code/repro/out/baseline_fidelity_fix3_20260916'

sys.path.insert(0, str(PKG))
sys.path.insert(0, str(HERE))

from common import (environment_info, git_info, sha256_json,  # noqa: E402
                    utc_now, write_json)
import groups  # noqa: E402

BENCHES = ('decomp', 'mg', 'minja', 'mcp', 'mrt',
           'injecagent', 'asb', 'agentharm')
X4_DATA = REPO / 'code/repro/x4_datasets/data'
MRT_FOLDS = ('american_airlines_2', 'api_key_calendar_agendas_2',
             'bank_transfer_2', 'expenditure_accounting', 'spam_filter_update')


def x4_test(name: str) -> list[dict]:
    graphs = [json.loads(line) for line in
              open(X4_DATA / f'{name}.jsonl') if line.strip()]
    fams = sorted({g['family'] for g in graphs})
    n_test = max(1, int(round(len(fams) * 0.2)))
    test_f = set(fams[-n_test:])
    return [g for g in graphs if g['family'] in test_f]


def method_runner(name: str):
    if name in ('agenttrust_prod', 'agenttrust_compat'):
        import agenttrust_fixed
        return lambda gs: agenttrust_fixed.predict(gs, name.rsplit('_', 1)[1])
    if name in ('agentarmor', 'agentarmor_conf_exfil'):
        import agentarmor_fixed as aa
        conf = name.endswith('conf_exfil')

        def run_armor(gs):
            preds, ledgers = aa.predict(gs, conf_exfil=conf)
            return {name: preds}, ledgers
        return run_armor
    if name == 'aegis':
        import aegis_fixed
        return lambda gs: aegis_fixed.predict(gs)
    raise SystemExit(f'unknown method {name!r}')


def method_source_hash(name: str) -> str:
    import hashlib
    parts = {}
    if name.startswith('agenttrust'):
        import agenttrust_fixed
        parts.update(agenttrust_fixed.source_hashes())
    else:
        module = {'agentarmor': 'agentarmor_fixed.py',
                  'agentarmor_conf_exfil': 'agentarmor_fixed.py',
                  'aegis': 'aegis_fixed.py'}[name]
        p = HERE / module
        if p.exists():
            parts[module] = hashlib.sha256(p.read_bytes()).hexdigest()
    return sha256_json(parts)


def unit_roles(payload: dict, fold: int | None) -> dict:
    graphs = groups.load_benchmark(payload['benchmark'])
    idx_by_group = defaultdict(list)
    for i, g in enumerate(graphs):
        idx_by_group[g['_group']].append(i)
    group_role = payload['group_role'] if fold is None else payload['folds'][fold]['group_role']
    roles = {r: [] for r in ('train', 'dev', 'gate_fit', 'threshold_cal', 'test')}
    for gid, role in group_role.items():
        roles[role].extend(idx_by_group[gid])
    return roles


def stratified_sample(graphs: list[dict], n: int) -> list[dict]:
    if n <= 0 or n >= len(graphs):
        return graphs
    pos = [g for g in graphs if int(g['label']) == 1][:n // 2]
    neg = [g for g in graphs if int(g['label']) == 0][:n - len(pos)]
    keep = pos + neg
    order = {id(g): i for i, g in enumerate(graphs)}
    return sorted(keep, key=lambda g: order[id(g)])


def confusion(preds, labels):
    tp = sum(1 for p, y in zip(preds, labels) if p and y)
    fp = sum(1 for p, y in zip(preds, labels) if p and not y)
    fn = sum(1 for p, y in zip(preds, labels) if not p and y)
    tn = sum(1 for p, y in zip(preds, labels) if not p and not y)
    return {'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'tpr': tp / (tp + fn) if tp + fn else None,
            'fpr': fp / (fp + tn) if fp + tn else None}


def evaluate_unit(runner, test_graphs: list[dict], unit_name: str,
                  meta: dict) -> dict:
    t0 = time.time()
    try:
        preds, ledgers = runner(test_graphs)
        error = None
    except Exception:
        preds, ledgers = None, None
        error = traceback.format_exc()
    labels = [int(g['label']) for g in test_graphs]
    if error:
        metrics = {'unit': unit_name, 'status': 'failed', 'error': error,
                   'n_test': len(test_graphs), 'n_pos': sum(labels),
                   'n_neg': len(labels) - sum(labels)}
        return {'preds': {}, 'ledgers': [], 'metrics': metrics}

    conf = {k: confusion(v, labels) for k, v in preds.items()}
    verdict_totals = Counter()
    err_adapter = err_interceptor = n_chain_only = 0
    missing = Counter()
    for d in ledgers:
        verdict_totals.update(d.get('verdict_counts', {}))
        err_adapter += len(d.get('adapter_errors', []))
        err_interceptor += len(d.get('interceptor_errors', []))
        n_chain_only += d.get('pred_chain_only', 0)
        for t in d.get('missing_mapping', []):
            missing[t] += 1
    metrics = {
        'unit': unit_name, 'status': 'ok',
        'n_test': len(test_graphs), 'n_pos': sum(labels),
        'n_neg': len(labels) - sum(labels),
        'confusion': conf,
        'verdict_totals': dict(verdict_totals),
        'adapter_errors': err_adapter,
        'interceptor_errors': err_interceptor,
        'pred_chain_only': n_chain_only,
        'missing_mapping': dict(missing),
        'elapsed_s': round(time.time() - t0, 2),
        **meta,
    }
    return {'preds': preds, 'ledgers': ledgers, 'metrics': metrics}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True)
    ap.add_argument('--benches', default=','.join(BENCHES))
    ap.add_argument('--sample', type=int, default=0)
    ap.add_argument('--run-id', default=None)
    args = ap.parse_args()

    benches = [b for b in args.benches.split(',') if b]
    run_id = args.run_id or f'{args.method}_20260916'
    run_dir = OUTROOT / run_id
    runner = method_runner(args.method)
    git = git_info()
    env = environment_info()
    summary = {'run_id': run_id, 'method': args.method, 'sample': args.sample,
               'benches': {}, 'started_utc': utc_now(),
               'git_head': git['head'], 'git_dirty_sha': git['status_sha256'],
               'env': env}

    src_hash = method_source_hash(args.method)

    for bench in benches:
        units = []
        if bench == 'mrt':
            for fold in MRT_FOLDS:
                p = MRT_ROOT / f'fold_{fold}' / 'admission.jsonl'
                gs = [json.loads(line) for line in open(p) if line.strip()]
                units.append((f'mrt_f{fold}', None, gs))
        elif bench in ('injecagent', 'asb', 'agentharm'):
            units.append((bench, None, x4_test(bench)))
        else:
            payload = json.load(open(FIX2 / 'split_ids' / f'{bench}.json'))
            if payload['mode'] == 'blocked_data':
                summary['benches'][bench] = {'status': 'blocked_data'}
                continue
            if payload['mode'] == 'single':
                unit_defs = [(None, -1)]
            else:
                unit_defs = [(f, f) for f in range(len(payload['folds']))]
            graphs_all = groups.load_benchmark(bench)
            for label, fold in unit_defs:
                roles = unit_roles(payload, None if fold < 0 else fold)
                gs = [graphs_all[i] for i in roles['test']]
                units.append((f'{bench}' + ('' if fold < 0 else f'_f{fold}'),
                              fold, gs))
        for unit_name, fold, gs in units:
            gs = stratified_sample(gs, args.sample)
            base = run_dir / args.method / unit_name
            base.mkdir(parents=True, exist_ok=True)
            meta = {'method': args.method, 'benchmark': bench, 'fold': fold,
                    'protocol': 'fix2_20260910 frozen split_ids'
                    if bench != 'mrt' else 'mrt_cross_task_v1 leave-one-task-out',
                    'source_hashes_sha256': src_hash,
                    'sampled': bool(args.sample and args.sample < len(gs))}
            res = evaluate_unit(runner, gs, unit_name, meta)
            if res['metrics']['status'] != 'ok':
                write_json(base / 'metrics.json', res['metrics'])
                print(f'{unit_name}: FAILED')
                continue
            with open(base / 'predictions.jsonl', 'w') as fh:
                for g, led in zip(gs, res['ledgers']):
                    row = dict(led)
                    row['sample_id'] = str(g.get('task_id'))
                    row['label'] = int(g['label'])
                    fh.write(json.dumps(row, ensure_ascii=False) + '\n')
            write_json(base / 'metrics.json', res['metrics'])
            write_json(base / 'config.json',
                       {'meta': meta, 'env': env, 'git': git})
            with open(base / 'run.log', 'w') as fh:
                fh.write(f'{utc_now()} {unit_name} n={len(gs)} '
                         f'elapsed={res["metrics"]["elapsed_s"]}s\n')
            summary['benches'].setdefault(bench, {})[unit_name] = res['metrics']
            m = res['metrics']
            head = m['confusion'].get(f'{args.method}_strict')
            print(f'{unit_name}: n={m["n_test"]} '
                  f'strict TPR={head["tpr"] if head else None} '
                  f'FPR={head["fpr"] if head else None} '
                  f'({m["elapsed_s"]}s)')
    summary['finished_utc'] = utc_now()
    write_json(run_dir / 'summary.json', summary)
    print('wrote', run_dir / 'summary.json')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
