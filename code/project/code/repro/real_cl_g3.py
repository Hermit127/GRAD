#!/usr/bin/env python3
"""CAA comparison chain on the true G3 backbone (G3-512).

Backbone: `train_gnn_ablate2.ModalityNormGNN` (candidate A_moddrop) with
per-modality LayerNorm, training-only auxiliary text/structure heads
(lambda_aux=0.1) and training-time per-graph modality dropout p=0.25;
node char window 512, edge char window 160 (patched at runtime, architecture
unchanged). All arms share architecture, splits, seeds/orders, selection
(dev AUC checkpoint), threshold 0.5 and budget (updates 200/stage, batch 16,
buffer 100, AdamW 1e-3 wd 1e-4, clip 1.0).

Arms (same as `real_cl_unified512.py`, re-used for the G3 re-verification):
- naive  : strict sequential fine-tuning (G3 loss incl. aux);
- replay : sequential + bounded replay (BCE(true) + cached-prob distillation
           w=1.0, both with the G3 aux term during backbone updates);
- stage0 : stage-0 G3 frozen + per-task adapter/head experts (aux heads are
           frozen with the backbone during expert training);
- snap   : strict sequential G3 updates with per-stage snapshots; expert t
           trains on frozen theta_t; old experts keep theta_j.

Frozen protocol re-verification on already-exposed data (not an independent
confirmation).
"""
from __future__ import annotations

import argparse
import collections
import copy
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
sys.path.insert(0, str(BASE / 'code/repro/final_five_tables'))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch_geometric.data import Batch, Data  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

import real_cl_p1 as p1  # noqa: E402
import real_cl_s2 as s2  # noqa: E402
import train_gnn_ablate2 as g3t  # noqa: E402

FAM_STREAM = BASE / 'LOGS/real_cl_family_confirm_20260921/stream'
FAM_STAGES = ['tool_poisoning', 'shadow_confusion', 'return_replay']
DMM_TASKS = ['decomp', 'mcp', 'mrt']

CFG = {'candidate': 'A_moddrop', 'moddrop': 0.25, 'lambda_aux': 0.1,
       'edge_mode': 'rebuilt', 'edge_dropout': 0.0, 'edge_text': 'copy'}


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_g3(device, node_chars, edge_chars):
    in_dim = len(g3t.PERMISSION) + 1 + len(g3t.CONTENT_KEYWORDS)
    edge_dim = len(g3t.V2_EDGE_TYPES) + 1
    g3t.MAX_NODE_CHARS = node_chars
    g3t.MAX_EDGE_CHARS = edge_chars
    m = g3t.ModalityNormGNN(in_dim, edge_dim, 192, use_ln=True,
                            use_aux=True).to(device)
    return m, in_dim, edge_dim


def write_stage_files(panel, split_dir, stage, splits_stage):
    d = Path(split_dir) / stage
    d.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ph in ('train', 'dev', 'test'):
        rows = splits_stage.get(ph) or []
        if not rows:
            continue
        p = d / f'{ph}.jsonl'
        p.write_text('\n'.join(json.dumps(g) for g in rows) + '\n')
        paths[ph] = str(p)
    return paths


def g3_scores(model, data, gs, device, adapter=None, head=None):
    """Per-sample probabilities with optional adapter hook + head swap."""
    if not gs:
        return {}
    old = model.classifier
    handle = None
    if head is not None:
        model.classifier = head
    if adapter is not None:
        handle = model.tf2.register_forward_hook(lambda mod, i, o: adapter(o))
    model.eval()
    out = {}
    with torch.no_grad():
        for i in range(0, len(gs), 16):
            ch = gs[i:i + 16]
            b = Batch.from_data_list([data[g['task_id']] for g in ch]).to(device)
            logits = model(b, moddrop_p=0.0, training=False)['logit']
            for g, p in zip(ch, torch.sigmoid(logits).cpu().tolist()):
                out[g['task_id']] = p
    if handle is not None:
        handle.remove()
    if head is not None:
        model.classifier = old
    return out


def dev_auc_g3(model, data, gs, device, adapter=None, head=None):
    from sklearn.metrics import roc_auc_score
    sc = g3_scores(model, data, gs, device, adapter, head)
    ys = [g['label'] for g in gs]
    ps = [sc[g['task_id']] for g in gs]
    if len(set(ys)) < 2:
        return 0.0
    return float(roc_auc_score(ys, ps))


def train_select_g3(model, data, tr, dv, device, seed, updates, batch,
                    adapter=None, replay=None, sel_every=25,
                    frozen_backbone=False, replay_n=0, cache_w=1.0, log=None):
    """CAA protocol training with the G3 backbone (aux loss kept in the
    backbone-update phase; modality dropout active only when training the
    backbone; expert phase freezes everything, modulo adapter/head)."""
    params = [p for p in model.parameters() if p.requires_grad]
    if adapter is not None:
        adapter = adapter.to(device)
        params = params + [p for p in adapter.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    rng = random.Random(seed)
    best = (-1.0, None, None, 0)
    for step in range(updates):
        model.train()
        if frozen_backbone:
            for p_ in model.parameters():
                p_.requires_grad = False
            for mod in model.modules():
                if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                    mod.eval()
        buf = replay.sample(replay_n, rng) if (replay is not None and replay_n) \
            else []
        cur = [tr[(step * batch + i) % len(tr)] for i in range(batch - len(buf))]
        gs = cur + [b[0] for b in buf]
        b = Batch.from_data_list([data[g['task_id']] for g in gs]).to(device)
        y = torch.tensor([float(g['label']) for g in gs], device=device)
        if adapter is not None:
            handle = model.tf2.register_forward_hook(
                lambda mod, i, o: adapter(o))
        opt.zero_grad()
        if frozen_backbone or adapter is not None:
            logits = model(b, moddrop_p=0.0, training=False)['logit']
        else:
            out = g3t.batch_logits(model, b, CFG, training=True)
            logits = out['logit']
            lam = CFG['lambda_aux']
            aux = lam * (F.binary_cross_entropy_with_logits(out['s_t'], y)
                         + F.binary_cross_entropy_with_logits(out['s_s'], y))
        loss = F.binary_cross_entropy_with_logits(logits, y)
        if adapter is not None:
            handle.remove()
        if not (frozen_backbone or adapter is not None):
            loss = loss + aux
        if buf and cache_w > 0:
            cache = torch.tensor([b_[1] for b_ in buf], device=device)
            loss = loss + cache_w * F.binary_cross_entropy_with_logits(
                logits[len(cur):], cache)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % sel_every == 0 or step == updates - 1:
            va = dev_auc_g3(model, data, dv, device, adapter)
            if va > best[0]:
                best = (va, copy.deepcopy(model.state_dict()),
                        copy.deepcopy(adapter.state_dict())
                        if adapter is not None else None, step + 1)
    if best[1] is not None:
        model.load_state_dict(best[1])
        if adapter is not None and best[2] is not None:
            adapter.load_state_dict(best[2])
    return model, best[0], best[3]


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
        seen, tot = set(), 0
        for m in self.models.values():
            if id(m) in seen:
                continue
            seen.add(id(m))
            tot += sum(v.numel() * v.element_size()
                       for v in m.state_dict().values()
                       if isinstance(v, torch.Tensor))
        return tot


def eval_snap(data, gs, experts, mode, router, seen, device, own_task=None):
    import copy as _copy
    cache = {}

    def _apply(t):
        if t not in cache:
            cache[t] = (_copy.deepcopy(experts.adapters[t]).to(device),
                        _copy.deepcopy(experts.heads[t]).to(device))
        return cache[t]

    sc = {}
    if mode == 'all':
        for g in gs:
            ps = []
            for t in seen:
                ad, hd = _apply(t)
                ps.append(g3_scores(experts.models[t], data, [g], device,
                                    ad, hd)[g['task_id']])
            sc[g['task_id']] = float(np.mean(ps))
        return sc
    for g in gs:
        t = (own_task or g.get('stage')) if mode == 'oracle' else router.route(g)
        if t not in experts.models:
            t = seen[0]
        ad, hd = _apply(t)
        sc.update(g3_scores(experts.models[t], data, [g], device, ad, hd))
    return sc


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
    return {'acc_matrix': {f'{k[0]}|{k[1]}': round(v['acc'], 4)
                           for k, v in sorted(acce.items())},
            'macro_acc': float(np.mean([acce[(n - 1, d)]['acc']
                                        for d in order])),
            'forgetting': float(np.mean(fog)) if fog else None,
            'bwt': float(np.mean(bwt)) if bwt else None}


def file_hash(p):
    try:
        return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--panel', choices=['dmm', 'fam'], default='dmm')
    ap.add_argument('--order', default='forward')
    ap.add_argument('--seed', type=int, default=211)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--updates', type=int, default=200)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--buffer', type=int, default=100)
    ap.add_argument('--sel-every', type=int, default=25)
    ap.add_argument('--node-chars', type=int, default=512)
    ap.add_argument('--edge-chars', type=int, default=160)
    ap.add_argument('--arms', default='naive,replay,stage0,snap')
    ap.add_argument('--routers', default='centroid,oracle,all')
    ap.add_argument('--report-test', action='store_true')
    ap.add_argument('--save-pool', action='store_true')
    ap.add_argument('--latency', action='store_true')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    outdir = Path(args.out)
    if outdir.exists():
        raise SystemExit(f'refusing to overwrite {outdir}')
    outdir.mkdir(parents=True)
    device = torch.device(args.device)
    log = [f'G3 CAA panel={args.panel} order={args.order} seed={args.seed} '
           f'node_chars={args.node_chars} edge_chars={args.edge_chars}']
    t0 = time.time()
    # ---- data ----
    if args.panel == 'dmm':
        task_list = list(DMM_TASKS)
        order = task_list if args.order == 'forward' \
            else list(reversed(task_list))
        s2_splits = s2.build_splits(args.seed, task_list)
        splits = {}
        data_dir = outdir / '_data'
        for d in order:
            splits[d] = s2_splits[d]
            write_stage_files(args.panel, data_dir, d, s2_splits[d])
    else:
        task_list = list(FAM_STAGES)
        order = task_list if args.order == 'forward' \
            else list(reversed(task_list))
        splits = collections.defaultdict(lambda: collections.defaultdict(list))
        for ph in ('train', 'dev', 'test'):
            for line in open(FAM_STREAM / f'{ph}.jsonl'):
                if line.strip():
                    g = json.loads(line)
                    splits[g['stage']][ph].append(g)
        data_dir = outdir / '_data'
        for d in order:
            write_stage_files(args.panel, data_dir, d, splits[d])
    # model dims from a probe conversion
    probe = json.loads(open(Path(data_dir) / order[0] / 'train.jsonl')
                       .readline())
    g3t.MAX_NODE_CHARS = args.node_chars
    g3t.MAX_EDGE_CHARS = args.edge_chars
    d0 = g3t.graph_to_data(probe, CFG, None)
    dims = (d0.x_base.size(1), d0.edge_attr_base.size(1))
    log.append(f'dims={dims}')

    def graphs_of(stage, ph):
        p = Path(data_dir) / stage / f'{ph}.jsonl'
        return [json.loads(l) for l in open(p) if l.strip()]

    def data_of(rows):
        return {g['task_id']: g3t.graph_to_data(g, CFG, None) for g in rows}
    data = {}
    for d in order:
        for ph in ('train', 'dev', 'test'):
            data.update(data_of(graphs_of(d, ph)))
    arms = [a for a in args.arms.split(',') if a]
    routers = [r for r in args.routers.split(',') if r]
    results = {}
    score_rows = []
    pool_scores = []
    latency = {}
    ev_ph = 'test' if args.report_test else 'dev'

    for arm in arms:
        t1 = time.time()
        acce = collections.defaultdict(dict)
        extras = {}
        set_seeds(args.seed)
        if arm in ('naive', 'replay', 'snap'):
            model, _, _ = make_g3(device, args.node_chars, args.edge_chars)
            replay = p1.ReplayBuffer(args.buffer) if arm != 'naive' else None
            experts = SnapExperts() if arm == 'snap' else None
            router = p1.Router()
            seen = []
            for ti, t in enumerate(order):
                tr = graphs_of(t, 'train')
                dv = graphs_of(t, 'dev')
                if arm == 'snap':
                    model, _, _ = train_select_g3(
                        model, data, tr, dv, device, args.seed + ti,
                        args.updates, args.batch, replay=replay,
                        replay_n=args.batch // 2, cache_w=1.0)
                    snap = copy.deepcopy(model).to(device)
                    ad = p1.Adapter(model.node_proj.out_features,
                                    r=8).to(device)
                    hd = nn.Linear(model.node_proj.out_features, 1).to(device)
                    old = snap.classifier
                    snap.classifier = hd
                    snap, _, _ = train_select_g3(
                        snap, data, tr, dv, device, args.seed + ti,
                        args.updates, args.batch, adapter=ad,
                        frozen_backbone=True)
                    snap.classifier = old
                    experts.add(t, snap, ad, hd)
                    router.add_task(t, tr)
                    if replay is not None:
                        sc_tr = g3_scores(snap, data, tr, device, ad, hd)
                        replay.add_task(tr, sc_tr, args.seed + ti)
                else:
                    model, _, _ = train_select_g3(
                        model, data, tr, dv, device, args.seed + ti,
                        args.updates, args.batch, replay=replay,
                        replay_n=args.batch // 2 if replay else 0, cache_w=1.0)
                    if replay is not None:
                        sc_tr = g3_scores(model, data, tr, device)
                        replay.add_task(tr, sc_tr, args.seed + ti)
                seen.append(t)
                for mode in (routers if arm == 'snap' else ['-']):
                    key = f'{arm}|{mode}'
                    for d in seen:
                        ev = graphs_of(d, ev_ph)
                        if arm == 'snap':
                            sc = eval_snap(data, ev, experts, mode, router,
                                           seen, device, own_task=d)
                        else:
                            sc = g3_scores(model, data, ev, device)
                        acce[key][(ti, d)] = p1.metrics(ev, sc)
                        if args.report_test and ti == len(order) - 1:
                            for g in ev:
                                score_rows.append({
                                    'arm': key, 'stage': ti + 1, 'domain': d,
                                    'task_id': g['task_id'],
                                    'label': g['label'],
                                    'score': sc[g['task_id']]})
                        if args.save_pool and arm == 'snap' \
                                and ti == len(order) - 1 and mode == 'centroid':
                            for g in ev:
                                rec = {'arm': 'snap', 'stage': d,
                                       'task_id': g['task_id'],
                                       'label': int(g['label']), 'p': {},
                                       'centroid_selected': None,
                                       'oracle_selected': d}
                                for tt in seen:
                                    ad = copy.deepcopy(
                                        experts.adapters[tt]).to(device)
                                    hd = copy.deepcopy(
                                        experts.heads[tt]).to(device)
                                    rec['p'][tt] = float(g3_scores(
                                        experts.models[tt], data, [g], device,
                                        ad, hd)[g['task_id']])
                                rec['centroid_selected'] = router.route(g)
                                pool_scores.append(rec)
                log.append(f'[{arm}] after {t}')
            if args.latency and arm == 'snap':
                dev_graphs = {d: graphs_of(d, ev_ph) for d in order}
                latency[arm] = measure_latency(data, dev_graphs, experts,
                                               router, seen, device)
            if args.save_pool:
                if arm == 'snap':
                    torch.save({'models': {t_: m_.state_dict()
                                           for t_, m_ in
                                           experts.models.items()},
                                'adapters': {t_: experts.adapters[t_].state_dict()
                                             for t_ in experts.adapters},
                                'heads': {t_: experts.heads[t_].state_dict()
                                          for t_ in experts.heads}},
                               outdir / 'pool_snap.pt')
                    extras['snapshot_bytes'] = experts.bytes()
                else:
                    torch.save(model.state_dict(), outdir / f'{arm}.pt')
        elif arm == 'stage0':
            model, _, _ = make_g3(device, args.node_chars, args.edge_chars)
            model, _, _ = train_select_g3(
                model, data, graphs_of(order[0], 'train'),
                graphs_of(order[0], 'dev'), device, args.seed, args.updates,
                args.batch)
            for p_ in model.parameters():
                p_.requires_grad = False
            hidden = model.node_proj.out_features
            experts = SnapExperts()
            router = p1.Router()
            seen = []
            for ti, t in enumerate(order):
                tr, dv = graphs_of(t, 'train'), graphs_of(t, 'dev')
                ad = p1.Adapter(hidden, r=8).to(device)
                hd = nn.Linear(hidden, 1).to(device)
                if ti == 0:
                    hd = copy.deepcopy(model.classifier)
                else:
                    old = model.classifier
                    model.classifier = hd
                    model, _, _ = train_select_g3(
                        model, data, tr, dv, device, args.seed + ti,
                        args.updates, args.batch, adapter=ad,
                        frozen_backbone=True)
                    model.classifier = old
                experts.add(t, model, ad, hd)
                router.add_task(t, tr)
                seen.append(t)
                for mode in routers:
                    key = f'{arm}|{mode}'
                    for d in seen:
                        ev = graphs_of(d, ev_ph)
                        if mode == 'all':
                            sc = eval_snap(data, ev, experts, 'all', router,
                                           seen, device)
                        else:
                            sc = eval_snap(data, ev, experts, mode, router,
                                           seen, device, own_task=d) \
                                if mode == 'oracle' else \
                                eval_snap(data, ev, experts, 'centroid',
                                          router, seen, device)
                        acce[key][(ti, d)] = p1.metrics(ev, sc)
                        if args.report_test and ti == len(order) - 1:
                            for g in ev:
                                score_rows.append({
                                    'arm': key, 'stage': ti + 1, 'domain': d,
                                    'task_id': g['task_id'],
                                    'label': g['label'],
                                    'score': sc[g['task_id']]})
                log.append(f'[{arm}] after {t}')
            if args.save_pool:
                torch.save({'models': {t_: m_.state_dict()
                                       for t_, m_ in experts.models.items()},
                            'adapters': {t_: experts.adapters[t_].state_dict()
                                         for t_ in experts.adapters},
                            'heads': {t_: experts.heads[t_].state_dict()
                                      for t_ in experts.heads}},
                           outdir / 'pool_stage0.pt')
            extras['snapshot_bytes'] = experts.bytes()
        for key, a in acce.items():
            results[key] = {**summarize(a, order),
                            'seconds': round(time.time() - t1, 1), **extras}
    meta = {'panel': args.panel, 'order': order, 'seed': args.seed,
            'tasks': task_list, 'window': [args.node_chars, args.edge_chars],
            'updates': args.updates, 'batch': args.batch,
            'buffer': args.buffer, 'report_test': args.report_test,
            'g3_flags': {'candidate': CFG['candidate'],
                         'moddrop': CFG['moddrop'],
                         'lambda_aux': CFG['lambda_aux'],
                         'use_ln': True, 'use_aux': True},
            'selection': 'best dev AUC every 25 steps; threshold 0.5',
            'hashes': {
                'real_cl_g3.py': file_hash(__file__),
                'train_gnn_ablate2.py': file_hash(
                    BASE / 'code/repro/final_five_tables/train_gnn_ablate2.py'),
                'train_gnn.py': file_hash(BASE / 'code/train_gnn.py'),
                'permission_spec.py': file_hash(BASE / 'code/permission_spec.py'),
                'stream_manifest': file_hash(
                    FAM_STREAM / 'manifest.json') if args.panel == 'fam'
                else None},
            'seconds': round(time.time() - t0, 1)}
    results['meta'] = meta
    if latency:
        results['latency'] = latency
    (outdir / 'results.json').write_text(json.dumps(results, indent=1,
                                                    default=str))
    if score_rows:
        (outdir / 'scores.jsonl').write_text(
            '\n'.join(json.dumps(r) for r in score_rows) + '\n')
    if pool_scores:
        (outdir / 'expert_scores.jsonl').write_text(
            '\n'.join(json.dumps(r) for r in pool_scores) + '\n')
    (outdir / 'log.txt').write_text('\n'.join(log) + '\n')
    print('\n'.join(log))
    print(json.dumps({k: {'acc': v.get('macro_acc'),
                          'ft': v.get('forgetting')}
                      for k, v in results.items()
                      if k not in ('meta', 'latency')}, indent=1))
    print('saved', outdir)
    return 0


def measure_latency(data, dev_graphs, experts, router, seen, device, n=32):
    """batch=1 latency (ms) for routing + expert forward(s)."""
    import numpy as np

    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize()
    rng = random.Random(0)
    flat = [(d, g) for d, gs in dev_graphs.items() for g in gs]
    rng.shuffle(flat)
    flat = flat[:n]
    out = {}
    for mode in ('centroid', 'all'):
        ts = []
        for d, g in flat:
            sync()
            t0 = time.perf_counter()
            if mode == 'centroid':
                t = router.route(g)
                if t not in experts.models:
                    t = seen[0]
                ad = experts.adapters[t].to(device)
                hd = experts.heads[t].to(device)
                g3_scores(experts.models[t], data, [g], device, ad, hd)
            else:
                for t in seen:
                    ad = experts.adapters[t].to(device)
                    hd = experts.heads[t].to(device)
                    g3_scores(experts.models[t], data, [g], device, ad, hd)
            sync()
            ts.append((time.perf_counter() - t0) * 1000.0)
        arr = np.asarray(ts)
        out[mode] = {'p50_ms': float(np.percentile(arr, 50)),
                     'p95_ms': float(np.percentile(arr, 95)),
                     'mean_ms': float(arr.mean()), 'n': len(ts)}
    return out


if __name__ == '__main__':
    raise SystemExit(main())
