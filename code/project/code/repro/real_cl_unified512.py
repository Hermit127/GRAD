#!/usr/bin/env python3
"""Unified GRAD-512 strict antibody ablation (decomp/mcp/mrt; zero API).

All arms share: architecture (CausalCharGNN), node-text window
(`--node-chars`, default 512 = GRAD-512), random init per arm with the same
seed, splits, optimizer/schedule, model-selection and threshold rules.

Arms (per run, same data objects):
- `naive`  : strict sequential fine-tuning (current data only);
- `replay` : strict sequential training with bounded replay (current +
  buffered history; cached-probability distillation w=1.0);
- `stage0` : stage-0 backbone (trained on task 0 only) frozen + per-task
  adapter/head experts (the original specialist design);
- `snap`   : STRICT E1 - at each stage t the backbone is updated with replay
  using only current + history, snapshotted as theta_t, and expert t is
  trained on the frozen theta_t; old experts keep their own theta_j
  snapshots. No future-task information reaches any expert or evaluation.

Routers for expert arms: centroid (deployable, signature only) / oracle
(diagnostic) / all (mean over experts, each with its own snapshot).
Metrics per stage: per-task dev metrics (+ test with --report-test), macro
ACC, F_T. Extra training/params/storage are recorded per arm.
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

BASE = Path(os.environ.get('GRAD_BASE', '.'))
sys.path.insert(0, str(BASE / 'code'))
sys.path.insert(0, str(BASE / 'code/repro'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

import train_gnn_v2  # noqa: E402
import real_cl_p1 as p1  # noqa: E402
import real_cl_p15 as p15  # noqa: E402
import real_cl_s2 as s2  # noqa: E402

TASKS = ['decomp', 'mcp', 'mrt']
FAM_STREAM = BASE / 'LOGS/real_cl_family_confirm_20260921/stream'
FAM_STAGES = ['tool_poisoning', 'shadow_confusion', 'return_replay']


def summarize(acce, order):
    n = len(order)
    bwt, fog = [], []
    for i, d in enumerate(order):
        own = acce.get((i, d), {}).get('acc')
        final = acce.get((n - 1, d), {}).get('acc')
        best = max((acce.get((j, d), {}).get('acc', -1)
                    for j in range(i, n)), default=None)
        if own is not None and final is not None:
            bwt.append(final - own)
        if best is not None and final is not None:
            fog.append(best - final)
    return {
        'acc_matrix': {f'{k[0]}|{k[1]}': round(v['acc'], 4)
                       for k, v in sorted(acce.items())},
        'macro_acc': float(np.mean([acce[(n - 1, d)]['acc'] for d in order])),
        'forgetting': float(np.mean(fog)) if fog else None,
        'bwt': float(np.mean(bwt)) if bwt else None}


class SnapExperts:
    def __init__(self):
        self.models, self.adapters, self.heads = {}, {}, {}

    def add(self, t, model, adapter, head):
        for p_ in model.parameters():
            p_.requires_grad = False
        self.models[t] = model
        self.adapters[t] = copy.deepcopy(adapter).cpu()
        self.heads[t] = copy.deepcopy(head).cpu()

    def bytes(self):
        return sum(v.numel() * v.element_size() for m in self.models.values()
                   for v in m.state_dict().values() if isinstance(v, torch.Tensor))


def eval_snap(m_default, data, gs, experts, router_mode, router, seen, device,
              own_task=None):
    import copy as _copy
    cache = {}

    def _apply(t):
        if t not in cache:
            cache[t] = (_copy.deepcopy(experts.adapters[t]).to(device),
                        _copy.deepcopy(experts.heads[t]).to(device))
        return cache[t]

    sc = {}
    if router_mode == 'all':
        for g in gs:
            ps = []
            for t in seen:
                ad, hd = _apply(t)
                ps.append(p1.expert_scores(experts.models[t], data, [g],
                                           ad, hd, device)[g['task_id']])
            sc[g['task_id']] = float(np.mean(ps))
        return sc
    for g in gs:
        if router_mode == 'oracle':
            t = own_task or g.get('stage')
        else:
            t = router.route(g)
        if t not in experts.models:
            t = seen[0]
        ad, hd = _apply(t)
        sc.update(p1.expert_scores(experts.models[t], data, [g], ad, hd,
                                   device))
    return sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default=','.join(TASKS))
    ap.add_argument('--order', default='forward')
    ap.add_argument('--seed', type=int, default=211)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--updates', type=int, default=200)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--buffer', type=int, default=100)
    ap.add_argument('--sel-every', type=int, default=25)
    ap.add_argument('--node-chars', type=int, default=512)
    ap.add_argument('--arms', default='naive,replay,stage0,snap')
    ap.add_argument('--panel', choices=['dmm', 'fam'], default='dmm')
    ap.add_argument('--routers', default='centroid,oracle,all')
    ap.add_argument('--report-test', action='store_true')
    ap.add_argument('--save-pool', action='store_true')
    ap.add_argument('--stream-dir', default='')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    task_list = [t for t in args.tasks.split(',') if t]

    def parse_order(tasks):
        if args.order == 'forward':
            return list(tasks)
        if args.order == 'reversed':
            return list(reversed(tasks))
        seq = [t for t in args.order.split(',') if t]
        if sorted(seq) != sorted(tasks):
            raise SystemExit(f'order {args.order!r} is not a permutation '
                             f'of {tasks}')
        return seq
    outdir = Path(args.out)
    if outdir.exists():
        raise SystemExit(f'refusing to overwrite {outdir}')
    outdir.mkdir(parents=True)
    device = torch.device(args.device)
    if args.node_chars != train_gnn_v2.MAX_NODE_CHARS:
        train_gnn_v2.MAX_NODE_CHARS = args.node_chars
    log = [f'unified512 order={args.order} seed={args.seed} '
           f'window={args.node_chars} arms={args.arms}']
    t0 = time.time()
    p15.set_seeds(args.seed)
    if args.panel == 'fam':
        stream = Path(args.stream_dir) if args.stream_dir else FAM_STREAM
        splits = collections.defaultdict(lambda: collections.defaultdict(list))
        for ph in ('train', 'dev', 'test'):
            for line in open(stream / f'{ph}.jsonl'):
                if line.strip():
                    g = json.loads(line)
                    splits[g['stage']][ph].append(g)
        task_list = list(FAM_STAGES)
        order = parse_order(task_list)
    else:
        order = parse_order(task_list)
        splits = s2.build_splits(args.seed, task_list)
    data = {g['task_id']: p1.graph_to_v2_data(g)
            for d in splits for ph in splits[d].values() for g in ph}
    d0 = data[splits[order[0]]['train'][0]['task_id']]
    dims = (d0.x_base.size(1), max(d0.edge_attr_base.size(1), 1))
    arms = [a for a in args.arms.split(',') if a]
    routers = [r for r in args.routers.split(',') if r]
    results = {}
    score_rows = []
    pool_scores = []

    def ev_split():
        return 'test' if args.report_test else 'dev'

    for arm in arms:
        t1 = time.time()
        acce = collections.defaultdict(dict)
        extras = {}
        p15.set_seeds(args.seed)
        if arm in ('naive', 'replay', 'replay_matched', 'snap'):
            backbone = p1.make_model(dims, device)
            replay = p1.ReplayBuffer(args.buffer) if arm != 'naive' else None
            experts = SnapExperts() if arm == 'snap' else None
            router = p1.Router()
            seen = []
            for ti, t in enumerate(order):
                if arm == 'snap':
                    p15.train_select(backbone, data, splits[t]['train'],
                                     splits[t]['dev'], device, args.seed + ti,
                                     args.updates, args.batch,
                                     replay=replay, frozen_backbone=False)
                    snap = copy.deepcopy(backbone).to(device)
                    ad = p1.Adapter(backbone.node_proj.out_features,
                                    r=8).to(device)
                    hd = nn.Linear(backbone.node_proj.out_features, 1).to(device)
                    old = snap.classifier
                    snap.classifier = hd
                    for p_ in snap.parameters():
                        p_.requires_grad = False
                    for p_ in list(ad.parameters()) + list(hd.parameters()):
                        p_.requires_grad = True
                    p15.train_select(snap, data, splits[t]['train'],
                                     splits[t]['dev'], device,
                                     args.seed + ti, args.updates, args.batch,
                                     adapter=ad, frozen_backbone=True)
                    snap.classifier = old
                    experts.add(t, snap, ad, hd)
                    router.add_task(t, splits[t]['train'])
                    if replay is not None:
                        sc_tr = p1.expert_scores(snap, data,
                                                 splits[t]['train'], ad, hd,
                                                 device)
                        replay.add_task(splits[t]['train'], sc_tr,
                                        args.seed + ti)
                else:
                    p15.train_select(backbone, data, splits[t]['train'],
                                     splits[t]['dev'], device, args.seed + ti,
                                     args.updates, args.batch, replay=replay)
                    if arm == 'replay_matched':
                        p15.train_select(backbone, data, splits[t]['train'],
                                         splits[t]['dev'], device,
                                         args.seed + 1000 + ti,
                                         args.updates, args.batch,
                                         replay=replay)
                    if replay is not None:
                        sc_tr = p1.scores_for(backbone, data,
                                              splits[t]['train'], device)
                        replay.add_task(splits[t]['train'], sc_tr,
                                        args.seed + ti)
                seen.append(t)
                for mode in (routers if arm == 'snap' else ['-']):
                    key = f'{arm}|{mode}'
                    for d in seen:
                        ev = splits[d][ev_split()]
                        if arm == 'snap':
                            sc = eval_snap(backbone, data, ev, experts, mode,
                                           router, seen, device, own_task=d)
                        else:
                            sc = p1.scores_for(backbone, data, ev, device)
                        acce.setdefault(key, {})[(ti, d)] = p1.metrics(ev, sc)
                        if args.save_pool and arm == 'snap' \
                                and ti == len(order) - 1 and mode == 'centroid':
                            for g in ev:
                                rec = {'arm': 'snap', 'stage': d,
                                       'task_id': g['task_id'],
                                       'label': int(g['label']),
                                       'p': {}, 'centroid_selected': None,
                                       'oracle_selected': d}
                                for t in seen:
                                    ad = copy.deepcopy(
                                        experts.adapters[t]).to(device)
                                    hd = copy.deepcopy(
                                        experts.heads[t]).to(device)
                                    rec['p'][t] = float(p1.expert_scores(
                                        experts.models[t], data, [g], ad, hd,
                                        device)[g['task_id']])
                                rec['centroid_selected'] = router.route(g)
                                pool_scores.append(rec)
                        if args.report_test and ti == len(order) - 1:
                            for g in ev:
                                score_rows.append({
                                    'arm': key, 'stage': ti + 1, 'domain': d,
                                    'task_id': g['task_id'],
                                    'label': g['label'],
                                    'score': sc[g['task_id']]})
                log.append(f'[{arm}] after {t}')
            extras['backbone_params'] = sum(p_.numel()
                                            for p_ in backbone.parameters())
            extras['updates_total'] = args.updates * len(order) * (
                2 if arm == 'replay_matched' else 1)
            if experts is not None:
                extras['snapshot_bytes'] = experts.bytes()
                extras['expert_params'] = sum(
                    sum(v.numel() for v in m_.state_dict().values()
                        if isinstance(v, torch.Tensor))
                    for m_ in experts.models.values()) + sum(
                    sum(v.numel() for v in a_.state_dict().values())
                    for a_ in experts.adapters.values()) + sum(
                    sum(v.numel() for v in h_.state_dict().values())
                    for h_ in experts.heads.values())
        elif arm == 'stage0':
            backbone = p1.make_model(dims, device)
            p15.train_select(backbone, data, splits[order[0]]['train'],
                             splits[order[0]]['dev'], device, args.seed,
                             args.updates, args.batch)
            for p_ in backbone.parameters():
                p_.requires_grad = False
            hidden = backbone.node_proj.out_features
            experts = p15.ExpertSet()
            router = p1.Router()
            seen = []
            for ti, t in enumerate(order):
                ad = p1.Adapter(hidden, r=8).to(device)
                hd = nn.Linear(hidden, 1).to(device)
                if ti == 0:
                    hd = copy.deepcopy(backbone.classifier)
                else:
                    old = backbone.classifier
                    backbone.classifier = hd
                    for p_ in list(ad.parameters()) + list(hd.parameters()):
                        p_.requires_grad = True
                    p15.train_select(backbone, data, splits[t]['train'],
                                     splits[t]['dev'], device,
                                     args.seed + ti, args.updates, args.batch,
                                     adapter=ad, frozen_backbone=True)
                    backbone.classifier = old
                experts.add(t, ad, hd)
                router.add_task(t, splits[t]['train'])
                seen.append(t)
                for mode in routers:
                    key = f'{arm}|{mode}'
                    for d in seen:
                        ev = splits[d][ev_split()]
                        if mode == 'oracle':
                            assign = {g['task_id']: d for g in ev}
                        elif mode == 'centroid':
                            assign = {g['task_id']: router.route(g)
                                      for g in ev}
                        else:
                            assign = None
                        if assign is None:
                            sc = p15.eval_all_experts(backbone, data, ev,
                                                      experts, seen, device)
                        else:
                            sc = p15.eval_routed(backbone, data, ev, experts,
                                                 seen, assign, device)
                        acce.setdefault(key, {})[(ti, d)] = p1.metrics(ev, sc)
                        if args.save_pool and ti == len(order) - 1 \
                                and mode == 'centroid':
                            for g in ev:
                                rec = {'arm': 'stage0', 'stage': d,
                                       'task_id': g['task_id'],
                                       'label': int(g['label']),
                                       'p': {}, 'centroid_selected': None,
                                       'oracle_selected': d}
                                for t in seen:
                                    ad, hd = experts.apply(t, backbone, device)
                                    old = backbone.classifier
                                    backbone.classifier = hd
                                    h = backbone.tf2.register_forward_hook(
                                        lambda mod, i, o: ad(o))
                                    rec['p'][t] = float(p1.scores_for(
                                        backbone, data, [g], device)[g['task_id']])
                                    h.remove()
                                    backbone.classifier = old
                                rec['centroid_selected'] = router.route(g)
                                pool_scores.append(rec)
                        if args.report_test and ti == len(order) - 1:
                            for g in ev:
                                score_rows.append({
                                    'arm': key, 'stage': ti + 1, 'domain': d,
                                    'task_id': g['task_id'],
                                    'label': g['label'],
                                    'score': sc[g['task_id']]})
                log.append(f'[{arm}] after {t}')
            extras['backbone_params'] = sum(p_.numel()
                                            for p_ in backbone.parameters())
            extras['expert_params'] = sum(
                sum(v.numel() for v in a_.state_dict().values())
                for a_ in experts.adapters.values()) + sum(
                sum(v.numel() for v in h_.state_dict().values())
                for h_ in experts.heads.values())
            extras['updates_total'] = args.updates * len(order)
        for key, a in acce.items():
            results[key] = {**summarize(a, order), 'seconds':
                            round(time.time() - t1, 1), **extras}
        if args.save_pool:
            if arm == 'snap':
                torch.save({'backbone': backbone.state_dict(),
                            'snap_models': {t: experts.models[t].state_dict()
                                            for t in experts.models},
                            'adapters': {t: experts.adapters[t].state_dict()
                                         for t in experts.adapters},
                            'heads': {t: experts.heads[t].state_dict()
                                      for t in experts.heads}},
                           outdir / 'pool_snap.pt')
            elif arm == 'stage0':
                torch.save({'backbone': backbone.state_dict(),
                            'adapters': dict(experts.adapters),
                            'heads': dict(experts.heads)},
                           outdir / 'pool_stage0.pt')
            elif arm in ('naive', 'replay', 'replay_matched'):
                torch.save(backbone.state_dict(), outdir / f'{arm}.pt')
    results['meta'] = {'order': order, 'seed': args.seed, 'tasks': task_list,
                       'window': args.node_chars, 'updates': args.updates,
                       'buffer': args.buffer, 'report_test': args.report_test,
                       'stream': args.stream_dir or str(FAM_STREAM),
                       'seconds': round(time.time() - t0, 1)}
    (outdir / 'results.json').write_text(json.dumps(results, indent=1))
    if score_rows:
        (outdir / 'scores.jsonl').write_text(
            '\n'.join(json.dumps(r) for r in score_rows) + '\n')
    if pool_scores:
        (outdir / 'expert_scores.jsonl').write_text(
            '\n'.join(json.dumps(r) for r in pool_scores) + '\n')
    (outdir / 'log.txt').write_text('\n'.join(log) + '\n')
    print('\n'.join(log))
    print(json.dumps({k: {'acc': v['macro_acc'], 'ft': v['forgetting']}
                      for k, v in results.items() if k != 'meta'}, indent=1))
    print('saved', outdir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
