#!/usr/bin/env python3
"""REAL-CL S2 runner: cross-dataset stream MCP -> MRT -> Decomp -> MG.

Reuses real_cl_p15.run_arm (identical arms/metrics/probes) with a custom task
loader. Pre-registration: LOGS/real_cl_s2_prereg_20260914.md.

Usage:
  ... real_cl_s2.py --order forward --seed 211 --device cuda:7 \
      --report-test --probe --out out/real_cl/s2_forward_211
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

BASE = Path(os.environ.get('GRAD_BASE', '.'))
sys.path.insert(0, str(BASE / 'code'))
sys.path.insert(0, str(BASE / 'code/repro'))

import torch  # noqa: E402

import real_cl_p1 as p1  # noqa: E402
import real_cl_p15 as p15  # noqa: E402

TASKS = ['mcp', 'mrt', 'decomp', 'mg']
CAP = 400


def read_jsonl(path):
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out


def cap_strat(gs, cap, seed):
    if not cap or len(gs) <= cap:
        return gs
    pos = [g for g in gs if g.get('label')]
    neg = [g for g in gs if not g.get('label')]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)
    n_pos = max(1, int(round(cap * len(pos) / len(gs))))
    take = pos[:n_pos] + neg[:cap - n_pos]
    rng.shuffle(take)
    return take


def load_task(name, seed, mrt_source='top1_360', mg_source='augmented'):
    if name == 'mcp':
        gs = p15.load_graphs_expanded()
    elif name == 'mrt':
        if mrt_source == 'top1_360':
            # audit-blessed runtime traces (no single-token shortcut, best
            # token BA ~0.78, node-count BA 0.5); see split_overlap report doc
            return cap_strat(read_jsonl(
                BASE / 'results/mrt_shared_adapter_scale360_v1/'
                       'stage_07_top1_360.jsonl'), CAP, seed)
        root = BASE / 'results/mrt_antibody_scale300_v1'
        gs, seen = [], set()
        for f in sorted(glob.glob(str(root / '*.jsonl'))):
            for g in read_jsonl(f):
                h = hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()
                if h not in seen:
                    seen.add(h)
                    gs.append(g)
    elif name == 'decomp':
        gs = (read_jsonl(BASE / 'ref/dataset/decomp_graphs_train_mixed.jsonl') +
              read_jsonl(BASE / 'ref/dataset/decomp_graphs_test_mixed.jsonl'))
    elif name == 'mg':
        if mg_source == 'realchain':
            # real runtime-chain traces (v8 batch); 8% positives, treated as a
            # noisy real-world task and reported with PR-AUC as well
            return cap_strat(read_jsonl(BASE / 'LOGS/realchain_full_20260914.jsonl'),
                             CAP, seed)
        hold = (read_jsonl(BASE / 'ref/dataset/capability_holdout_v1/'
                           'holdout_memory_train.jsonl') +
                read_jsonl(BASE / 'ref/dataset/capability_holdout_v1/'
                           'holdout_memory_test.jsonl'))
        hold_ids = {g['task_id'] for g in hold}
        gs = [g for g in read_jsonl(BASE / 'ref/dataset/memorygraft_augmented.jsonl')
              if g['task_id'] not in hold_ids]
    else:
        raise ValueError(name)
    return cap_strat(gs, CAP, seed)


def build_splits(seed, tasks=None, mrt_source='top1_360', mg_source='augmented'):
    splits = {}
    for i, t in enumerate(tasks or TASKS):
        gs = load_task(t, seed + i, mrt_source, mg_source)
        rng = random.Random(seed + 100 + i)
        pos = [g for g in gs if g.get('label')]
        neg = [g for g in gs if not g.get('label')]
        rng.shuffle(pos)
        rng.shuffle(neg)
        take = pos + neg
        rng.shuffle(take)
        n = len(take)
        n_tr, n_dv = int(n * 0.7), int(n * 0.15)
        splits[t] = {'train': take[:n_tr], 'dev': take[n_tr:n_tr + n_dv],
                     'test': take[n_tr + n_dv:]}
    return splits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default=','.join(TASKS))
    ap.add_argument('--order', default='forward')
    ap.add_argument('--seed', type=int, default=211)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--arms', default=('naive,joint,replay,ewc,specialists,'
                                       'specialists_all,crl,crl_replay'))
    ap.add_argument('--updates', type=int, default=200)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--buffer', type=int, default=100)
    ap.add_argument('--ewc-lambda', type=float, default=100.0)
    ap.add_argument('--joint-mult', type=int, default=3)
    ap.add_argument('--sel-every', type=int, default=25)
    ap.add_argument('--report-test', action='store_true')
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--mrt-source', default='top1_360',
                    choices=['top1_360', 'scale300'])
    ap.add_argument('--mg-source', default='augmented',
                    choices=['augmented', 'realchain'])
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    task_list = [t for t in args.tasks.split(',') if t]
    order = task_list if args.order == 'forward' else list(reversed(task_list))
    outdir = Path(args.out) if args.out else \
        BASE / f'code/repro/out/real_cl/s2_{args.order}_{args.seed}'
    if outdir.exists():
        raise SystemExit(f'refusing to overwrite {outdir}')
    outdir.mkdir(parents=True)
    (outdir / 'config.json').write_text(json.dumps(
        {**vars(args), 'order_list': order, 'base': str(BASE), 'cap': CAP},
        indent=1))
    (outdir / 'commands.sh').write_text(
        f'GRAD_BASE={BASE} PYTHONPATH={BASE}/code:{BASE}/code/repro '
        f'python code/repro/real_cl_s2.py --order {args.order} '
        f'--seed {args.seed} --device {args.device} --arms {args.arms} '
        f'--updates {args.updates} --batch {args.batch} --buffer {args.buffer} '
        f'--ewc-lambda {args.ewc_lambda} --joint-mult {args.joint_mult} '
        + ('--report-test --probe ' if args.report_test else '')
        + f'--out {args.out}\n')
    device = torch.device(args.device)
    t0 = time.time()
    log = [f'REAL-CL S2 order={args.order} seed={args.seed} device={args.device}']
    p15.set_seeds(args.seed)
    splits = build_splits(args.seed, task_list, args.mrt_source, args.mg_source)
    log.append('split sizes: ' + json.dumps(
        {t: {k: len(v) for k, v in ph.items()} for t, ph in splits.items()}))
    data = {g['task_id']: p1.graph_to_v2_data(g)
            for t in splits for ph in splits[t].values() for g in ph}
    d0 = data[splits[order[0]]['train'][0]['task_id']]
    dims = (d0.x_base.size(1), max(d0.edge_attr_base.size(1), 1))
    log.append(f'x_base_dim={dims[0]} edge_base_dim={dims[1]}')
    results, ledger, per_item, score_rows = {}, [], [], []
    for arm in args.arms.split(','):
        t1 = time.time()
        acce, pi, extras, slog = p15.run_arm(arm, splits, data, dims, order,
                                             args, device, args.report_test,
                                             args.probe, log)
        score_rows += slog
        results[arm] = extras
        per_item += pi
        (outdir / 'results.json').write_text(json.dumps(results, indent=1))
        ledger.append({'arm': arm, 'seconds': round(time.time() - t1, 1)})
        log.append(f'[{arm}] done {time.time() - t1:.1f}s bwt={extras["bwt"]} '
                   f'forget={extras["forgetting"]}')
    (outdir / 'per_item.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in per_item) + '\n')
    (outdir / 'scores.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in score_rows) + '\n')
    (outdir / 'ledger.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in ledger) + '\n')
    log.append(f'ALL DONE seconds={time.time() - t0:.1f}')
    (outdir / 'log.txt').write_text('\n'.join(log) + '\n')
    print('\n'.join(log))
    print('saved', outdir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
