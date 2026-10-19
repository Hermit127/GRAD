#!/usr/bin/env python3
"""REAL-CL P1.5 runner: dev-selected standard CL evaluation.

Extends real_cl_p1 (reused imports) with:
- dev-based best checkpoint for every arm (test read once, only with --report-test)
- standard metric block: ACC matrix / ACC / BWT / forgetting / FWT (stage0 ref)
- specialist ablations: oracle / random / all-experts / single-expert routing
- adapter transfer matrix (reuse analysis) for the specialist arm
- CRL probes: frozen-feature linear probe per stage + adjacent-stage linear CKA
- protected CRL variant (crl_replay)

Usage (tuning: no --report-test):
  ... real_cl_p15.py --order forward --seed 211 --device cuda:7 --arms ewc \
      --ewc-lambda 10 --out out/real_cl/p15tune_forward_211
Usage (final):
  ... real_cl_p15.py --order forward --seed 211 --device cuda:7 \
      --report-test --probe --out out/real_cl/p15_forward_211
"""
from __future__ import annotations

import argparse
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
import torch.nn.functional as F  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

import real_cl_p1 as p1  # noqa: E402

ALL_DOMAINS = p1.ALL_DOMAINS


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def dev_auc(m, data, gs, device):
    sc = p1.scores_for(m, data, gs, device)
    ys = [g['label'] for g in gs]
    ps = [sc[g['task_id']] for g in gs]
    if len(set(ys)) < 2:
        return 0.0
    return float(roc_auc_score(ys, ps))


def train_select(m, data, tr, dv, device, seed, updates, batch,
                 adapter=None, replay=None, penalty=None, sel_every=25,
                 frozen_backbone=False):
    """Dev-selected training with optional adapter hook and soft-target replay.

    Fixes vs the audited version (LOGS/real_cl_independent_review_20260915.md):
    - adapter parameters join the optimizer and grad clipping (previously the
      adapter had gradients but was never updated, and its state was not part
      of the best checkpoint);
    - the best checkpoint restores the adapter as well as the model;
    - frozen_backbone=True keeps BatchNorm running stats frozen (eval mode), so
      "frozen encoder" claims are actually true;
    - replay distillation uses BCE-with-logits on stored probabilities instead
      of MSE(logits, probs).
    """
    params = [p for p in m.parameters() if p.requires_grad]
    if adapter is not None:
        adapter = adapter.to(device)
        params = params + [p for p in adapter.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    rng = random.Random(seed)
    handle = None
    if adapter is not None:
        handle = m.tf2.register_forward_hook(lambda mod, i, o: adapter(o))
    best = (-1.0, None, None, 0)
    for step in range(updates):
        m.train()
        if frozen_backbone:
            for mod in m.modules():
                if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                    mod.eval()
        buf = replay.sample(batch // 2, rng) if replay is not None else []
        cur = [tr[(step * batch + i) % len(tr)] for i in range(batch - len(buf))]
        gs = cur + [b[0] for b in buf]
        b, y = p1.batch_of(data, gs, device)
        opt.zero_grad()
        logits = p1.fwd(m, b)
        loss = F.binary_cross_entropy_with_logits(logits, y)
        if buf:
            old = torch.tensor([b_[1] for b_ in buf], device=device)
            loss = loss + F.binary_cross_entropy_with_logits(
                logits[len(cur):], old)
        if penalty is not None:
            loss = loss + penalty(m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % sel_every == 0 or step == updates - 1:
            m.eval()
            va = dev_auc(m, data, dv, device)
            if va > best[0]:
                best = (va, copy.deepcopy(m.state_dict()),
                        copy.deepcopy(adapter.state_dict())
                        if adapter is not None else None,
                        step + 1)
    if best[1] is not None:
        m.load_state_dict(best[1])
        if adapter is not None and best[2] is not None:
            adapter.load_state_dict(best[2])
    if handle is not None:
        handle.remove()
    return m, best[0], best[3]


def pooled_features(m, data, gs, device):
    old = m.classifier
    m.classifier = nn.Identity()
    feats = {}
    m.eval()
    with torch.no_grad():
        for i in range(0, len(gs), 16):
            ch = gs[i:i + 16]
            b = Batch.from_data_list([data[g['task_id']] for g in ch]).to(device)
            out = p1.fwd(m, b)
            if out.dim() == 1:
                out = out.unsqueeze(-1)
            for g, v in zip(ch, out.cpu().numpy()):
                feats[g['task_id']] = v
    m.classifier = old
    return feats


def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(X.T @ Y) ** 2
    den = np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y)
    return float(num / den) if den > 0 else 0.0


def run_probe(m, data, splits, seen, ref, device, report_test):
    """Frozen-feature linear probe per seen task + CKA on the reference set."""
    out = {}
    for t in seen:
        tr = splits[t]['train'][:40]
        ev = splits[t]['test'] if report_test else splits[t]['dev']
        ftr = pooled_features(m, data, tr, device)
        fev = pooled_features(m, data, ev, device)
        X = np.stack([ftr[g['task_id']] for g in tr])
        y = [g['label'] for g in tr]
        if len(set(y)) < 2:
            out[t] = None
            continue
        clf = LogisticRegression(max_iter=500, C=1.0)
        clf.fit(X, y)
        Xe = np.stack([fev[g['task_id']] for g in ev])
        ye = [g['label'] for g in ev]
        out[t] = float((clf.predict(Xe) == np.array(ye)).mean())
    fre = pooled_features(m, data, ref, device)
    Xr = np.stack([fre[g['task_id']] for g in ref])
    return {'probe_acc': out, 'ref_feats': Xr}


class ExpertSet:
    """Stores task experts as module copies.

    Head architecture follows whatever m.classifier is (Linear or Sequential),
    so the stage-0 expert can reuse the head trained during the shared pretrain
    without a state-dict shape mismatch.
    """

    def __init__(self):
        self.adapters, self.heads = {}, {}

    def add(self, task, adapter, head):
        self.adapters[task] = copy.deepcopy(adapter).cpu()
        self.heads[task] = copy.deepcopy(head).cpu()

    def apply(self, task, m, device):
        ad = copy.deepcopy(self.adapters[task]).to(device)
        hd = copy.deepcopy(self.heads[task]).to(device)
        return ad, hd


def eval_routed(m, data, gs, experts, tasks, assign, device):
    """Evaluate with precomputed routing assignments (task_id -> expert task)."""
    groups = {}
    for g in gs:
        t = assign.get(g['task_id'])
        if t not in experts.adapters:
            t = next(iter(experts.adapters))
        groups.setdefault(t, []).append(g)
    sc = {}
    for t, grp in groups.items():
        ad, hd = experts.apply(t, m, device)
        old = m.classifier
        m.classifier = hd
        handle = m.tf2.register_forward_hook(lambda mod, i, o: ad(o))
        sc.update(p1.scores_for(m, data, grp, device))
        handle.remove()
        m.classifier = old
    return sc


def eval_all_experts(m, data, gs, experts, seen, device):
    sc = {}
    for g in gs:
        ps = []
        for t in seen:
            ad, hd = experts.apply(t, m, device)
            old = m.classifier
            m.classifier = hd
            handle = m.tf2.register_forward_hook(lambda mod, i, o: ad(o))
            ps.append(p1.scores_for(m, data, [g], device)[g['task_id']])
            handle.remove()
            m.classifier = old
        sc[g['task_id']] = float(np.mean(ps))
    return sc


def eval_conf_routed(m, data, gs, experts, seen, device):
    """Confidence routing: score each graph with every seen expert (batched per
    expert) and assign it to the expert with the largest |score - 0.5|."""
    per = {}
    for t in seen:
        ad, hd = experts.apply(t, m, device)
        old = m.classifier
        m.classifier = hd
        handle = m.tf2.register_forward_hook(lambda mod, i, o: ad(o))
        per[t] = p1.scores_for(m, data, gs, device)
        handle.remove()
        m.classifier = old
    sc, assign = {}, {}
    for g in gs:
        tid = g['task_id']
        best_t, best_c = None, -1.0
        for t in seen:
            c = abs(per[t][tid] - 0.5)
            if c > best_c:
                best_t, best_c = t, c
        assign[tid] = best_t
        sc[tid] = per[best_t][tid]
    return sc, assign


def run_arm(arm, splits, data, dims, order, args, device, report_test, probe, log):
    set_seeds(args.seed)
    tasks = list(order)
    m = p1.make_model(dims, device)
    hidden = m.node_proj.out_features
    router = p1.Router() if (arm.startswith('specialists') or
                              arm in ('crl', 'crl_replay')) else None
    experts = ExpertSet()
    replay = p1.ReplayBuffer(args.buffer) if arm in ('replay', 'crl_replay',
                                                     'specialists_replay') \
        else None
    ewc = p1.EWC(args.ewc_lambda)
    acce, per_item, extras = {}, [], {}
    score_log = []
    probe_log, ref = [], [g for g in splits[tasks[0]]['train'][:30]]
    ref_feats_prev = None
    route_hits = {}
    stage0, stage_pre = {}, {}
    for t in tasks:
        stage0[t] = {'acc_dev': p1.metrics(
            splits[t]['dev'], p1.scores_for(m, data, splits[t]['dev'], device))['acc']}
        if report_test:
            stage0[t]['acc_test'] = p1.metrics(
                splits[t]['test'],
                p1.scores_for(m, data, splits[t]['test'], device))['acc']

    def record(ti, evals, log_tag):
        for d in tasks[:ti + 1]:
            ev = splits[d]['test'] if report_test else splits[d]['dev']
            acce[(ti, d)] = ev_metrics[ti][d]
            per_item.append({'arm': arm, 'stage': ti + 1, 'domain': d, 'split':
                             'test' if report_test else 'dev', **ev_metrics[ti][d]})
        log.append(f'[{arm}] {log_tag} ' + '; '.join(
            f'{d[:5]}:{acce[(ti, d)]["acc"]:.3f}/{acce[(ti, d)]["f1"]:.3f}'
            for d in tasks[:ti + 1]))

    def eval_seen(ti):
        out = {}
        seen = tasks[:ti + 1]
        for d in seen:
            ev = splits[d]['test'] if report_test else splits[d]['dev']
            if arm == 'single_expert':
                ad, hd = experts.apply(tasks[0], m, device)
                old = m.classifier
                m.classifier = hd
                handle = m.tf2.register_forward_hook(lambda mod, i, o: ad(o))
                sc = p1.scores_for(m, data, ev, device)
                handle.remove()
                m.classifier = old
            elif arm.startswith('specialists') or arm in ('crl', 'crl_replay'):
                route = None
                if arm in ('specialists', 'crl', 'crl_replay',
                           'specialists_replay'):
                    route = router.route
                elif arm == 'specialists_oracle':
                    route = lambda g: d  # noqa: E731
                elif arm == 'specialists_random':
                    rng_r = random.Random(1234 + ti)
                    route = lambda g: rng_r.choice(seen)  # noqa: E731
                elif arm == 'specialists_conf':
                    route = None  # confidence routing handled below
                if arm == 'specialists_all':
                    route_hits[(ti, d)] = None
                    sc = eval_all_experts(m, data, ev, experts, seen, device)
                elif arm == 'specialists_conf':
                    sc, assign = eval_conf_routed(m, data, ev, experts, seen,
                                                  device)
                    route_hits[(ti, d)] = (
                        sum(1 for g in ev if assign[g['task_id']] == d)
                        / max(len(ev), 1))
                else:
                    assign = {g['task_id']: route(g) for g in ev}
                    route_hits[(ti, d)] = (
                        sum(1 for g in ev if assign[g['task_id']] == d)
                        / max(len(ev), 1))
                    sc = eval_routed(m, data, ev, experts, tasks, assign, device)
            else:
                sc = p1.scores_for(m, data, ev, device)
            out[d] = p1.metrics(ev, sc)
            for g in ev:
                score_log.append({'arm': arm, 'stage': ti + 1, 'domain': d,
                                  'task_id': g['task_id'], 'label': g['label'],
                                  'score': sc[g['task_id']]})
        return out

    ev_metrics = {}
    for ti, t in enumerate(tasks):
        pre_split = splits[t]['test'] if report_test else splits[t]['dev']
        stage_pre[t] = {'acc': p1.metrics(
            pre_split, p1.scores_for(m, data, pre_split, device))['acc']}
        if arm == 'naive':
            train_select(m, data, splits[t]['train'], splits[t]['dev'], device,
                         args.seed + ti, args.updates, args.batch)
        elif arm == 'joint':
            all_train = [g for tt in tasks for g in splits[tt]['train']]
            train_select(m, data, all_train, splits[t]['dev'], device,
                         args.seed, args.updates * args.joint_mult, args.batch)
        elif arm == 'replay':
            train_select(m, data, splits[t]['train'], splits[t]['dev'], device,
                         args.seed + ti, args.updates, args.batch,
                         replay=replay)
            sc_tr = p1.scores_for(m, data, splits[t]['train'], device)
            replay.add_task(splits[t]['train'], sc_tr, args.seed + ti)
        elif arm == 'ewc':
            penalty = ewc.penalty if ti > 0 else None
            train_select(m, data, splits[t]['train'], splits[t]['dev'], device,
                         args.seed + ti, args.updates, args.batch,
                         penalty=penalty)
            ewc.update(m, data, splits[t]['train'], device, 8, args.seed + ti)
        elif arm in ('specialists', 'specialists_oracle', 'specialists_random',
                     'specialists_all', 'specialists_conf',
                     'specialists_replay'):
            if ti == 0:
                train_select(m, data, splits[t]['train'], splits[t]['dev'],
                             device, args.seed, args.updates, args.batch)
                for p in m.parameters():
                    p.requires_grad = False
                # Budget parity with naive: the stage-0 expert reuses the head
                # trained in the shared pretrain; the adapter starts at
                # identity. No extra stage-0 updates.
                adapter = p1.Adapter(hidden, r=8).to(device)
                head = copy.deepcopy(m.classifier)
                experts.add(t, adapter, head)
                router.add_task(t, splits[t]['train'])
                if arm == 'specialists_replay':
                    sc_tr = p1.scores_for(m, data, splits[t]['train'], device)
                    replay.add_task(splits[t]['train'], sc_tr, args.seed)
            else:
                for p in m.parameters():
                    p.requires_grad = False
                adapter = p1.Adapter(hidden, r=8).to(device)
                head = nn.Linear(hidden, 1).to(device)
                old = m.classifier
                m.classifier = head
                for p in list(adapter.parameters()) + list(head.parameters()):
                    p.requires_grad = True
                train_select(m, data, splits[t]['train'], splits[t]['dev'],
                             device, args.seed + ti, args.updates, args.batch,
                             adapter=adapter, frozen_backbone=True,
                             replay=replay if arm == 'specialists_replay'
                             else None)
                sc_tr = (p1.expert_scores(m, data, splits[t]['train'],
                                          adapter, head, device)
                         if arm == 'specialists_replay' else None)
                m.classifier = old
                experts.add(t, adapter, head)
                router.add_task(t, splits[t]['train'])
                if arm == 'specialists_replay':
                    replay.add_task(splits[t]['train'], sc_tr, args.seed + ti)
        elif arm == 'single_expert':
            if ti == 0:
                train_select(m, data, splits[t]['train'], splits[t]['dev'],
                             device, args.seed, args.updates, args.batch)
                for p in m.parameters():
                    p.requires_grad = False
                adapter = p1.Adapter(hidden, r=8).to(device)
                head = copy.deepcopy(m.classifier)
                experts.add(t, adapter, head)
        elif arm in ('crl', 'crl_replay'):
            adapter = p1.Adapter(hidden, r=8).to(device)
            head = nn.Linear(hidden, 1).to(device)
            old = m.classifier
            m.classifier = head
            for p in m.parameters():
                p.requires_grad = True
            train_select(m, data, splits[t]['train'], splits[t]['dev'], device,
                         args.seed + ti, args.updates, args.batch,
                         adapter=adapter,
                         replay=replay if arm == 'crl_replay' else None)
            # Soft targets come from the just-trained expert (adapter+head
            # active), before the expert is unloaded.
            if arm == 'crl_replay':
                sc_tr = p1.expert_scores(m, data, splits[t]['train'], adapter,
                                         head, device)
            m.classifier = old
            experts.add(t, adapter, head)
            router.add_task(t, splits[t]['train'])
            if arm == 'crl_replay':
                replay.add_task(splits[t]['train'], sc_tr, args.seed + ti)
        ev_metrics[ti] = eval_seen(ti)
        record(ti, ev_metrics, f'after {t}')
        if probe and arm in ('specialists', 'crl'):
            pr = run_probe(m, data, splits, tasks[:ti + 1], ref, device,
                           report_test)
            cka = None
            if ref_feats_prev is not None:
                cka = linear_cka(ref_feats_prev, pr['ref_feats'])
            ref_feats_prev = pr['ref_feats']
            probe_log.append({'stage': ti + 1, 'probe_acc': pr['probe_acc'],
                              'cka_prev': cka})

    # transfer matrix for specialists (adapter reuse)
    if arm == 'specialists' and experts.adapters:
        transfer = {}
        for src in experts.adapters:
            row = {}
            for dst in tasks:
                ev = splits[dst]['test'] if report_test else splits[dst]['dev']
                ad, hd = experts.apply(src, m, device)
                old = m.classifier
                m.classifier = hd
                handle = m.tf2.register_forward_hook(lambda mod, i, o: ad(o))
                row[dst] = p1.metrics(ev, p1.scores_for(m, data, ev,
                                                        device))['f1']
                handle.remove()
                m.classifier = old
            transfer[src] = row
        extras['transfer_f1'] = transfer

    # standard summaries (fixed definitions vs LOGS/real_cl_independent_review_20260915.md:
    # FWT is now pre-task accuracy minus stage-0 accuracy; BWT/forgetting keep the
    # all-task average and additionally report the common T-1 variant).
    n = len(tasks)
    fwt, fwt_post, bwt, forget, bwt_t1, forget_t1 = [], [], [], [], [], []
    for i, d in enumerate(tasks):
        own = acce.get((i, d), {}).get('acc')
        final = acce.get((n - 1, d), {}).get('acc')
        best = max((acce.get((j, d), {}).get('acc', -1) for j in range(i, n)),
                   default=None)
        if own is not None and final is not None:
            bwt.append(final - own)
        if best is not None and final is not None:
            forget.append(best - final)
        if i < n - 1:
            if own is not None and final is not None:
                bwt_t1.append(final - own)
            if best is not None and final is not None:
                forget_t1.append(best - final)
        s0 = stage0[d].get('acc_test' if report_test else 'acc_dev')
        pre = stage_pre[d].get('acc')
        if pre is not None and s0 is not None:
            fwt.append(pre - s0)
        if own is not None and s0 is not None:
            fwt_post.append(own - s0)
    extras.update({'acc_matrix': {f'{k[0]}|{k[1]}': round(v['acc'], 4)
                                  for k, v in sorted(acce.items())},
                   'bwt': float(np.mean(bwt)) if bwt else None,
                   'bwt_t1': float(np.mean(bwt_t1)) if bwt_t1 else None,
                   'forgetting': float(np.mean(forget)) if forget else None,
                   'forgetting_t1': float(np.mean(forget_t1)) if forget_t1 else None,
                   'fwt': float(np.mean(fwt)) if fwt else None,
                   'fwt_post_minus_stage0': (float(np.mean(fwt_post))
                                             if fwt_post else None),
                   'stage0_acc': stage0,
                   'stage_pre_acc': stage_pre,
                   'router_hit_per_cell': {f'{k[0]}|{k[1]}': v
                                           for k, v in sorted(route_hits.items())},
                   'router_hit': None})
    rh = [v for v in route_hits.values() if v is not None]
    extras['router_hit'] = float(np.mean(rh)) if rh else None
    if probe_log:
        extras['probe'] = probe_log
    return acce, per_item, extras, score_log



def load_graphs_expanded():
    """All available MCP graph sources (dedup by content hash); domain mapping
    identical to p1.domain_of. Zero API; used per real_cl_p15_prereg §6."""
    import glob as _glob
    import hashlib as _hashlib
    files = ([f"ref/dataset/mcp_safetybench_defense_{s}.jsonl"
              for s in ('train', 'test', 'validation')] +
             [f for f in sorted(_glob.glob(
                 'ref/dataset/mcp_residual_matched_v2_merged/graphs/*.jsonl'))
              if 'generated_all' not in f] +
             sorted(_glob.glob('ref/dataset/mcp_dgc_v0*/*.jsonl')) +
             sorted(_glob.glob('ref/dataset/mcp_dgc_v1_seed*/*.jsonl')) +
             sorted(_glob.glob('ref/dataset/mcp_coverage_deficit_*/*/*.jsonl')) +
             sorted(_glob.glob('ref/dataset/mcp_channel_gap_dgc_v2_seed49*/*.jsonl')))
    seen, out = set(), []
    for f in files:
        try:
            for line in open(BASE / f):
                g = json.loads(line)
                h = _hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                if p1.domain_of(g) != 'other':
                    out.append(g)
        except Exception:
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--domains', default=','.join(ALL_DOMAINS))
    ap.add_argument('--order', default='forward')
    ap.add_argument('--seed', type=int, default=211)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--arms', default=('naive,joint,replay,ewc,specialists,'
                                       'specialists_oracle,specialists_random,'
                                       'specialists_all,single_expert,crl,'
                                       'crl_replay'))
    ap.add_argument('--updates', type=int, default=200)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--buffer', type=int, default=50)
    ap.add_argument('--ewc-lambda', type=float, default=10.0)
    ap.add_argument('--joint-mult', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--sel-every', type=int, default=25)
    ap.add_argument('--report-test', action='store_true')
    ap.add_argument('--data-expanded', action='store_true')
    ap.add_argument('--probe', action='store_true')
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    order = args.domains.split(',')
    if args.order == 'reversed':
        order = list(reversed(order))
    outdir = Path(args.out) if args.out else \
        BASE / f'code/repro/out/real_cl/p15_{args.order}_{args.seed}'
    if outdir.exists():
        raise SystemExit(f'refusing to overwrite {outdir}')
    outdir.mkdir(parents=True)
    (outdir / 'config.json').write_text(json.dumps(
        {**vars(args), 'order_list': order, 'base': str(BASE)}, indent=1))
    (outdir / 'commands.sh').write_text(
        f'GRAD_BASE={BASE} PYTHONPATH={BASE}/code:{BASE}/code/repro '
        f'python code/repro/real_cl_p15.py --order {args.order} '
        f'--seed {args.seed} --device {args.device} --arms {args.arms} '
        f'--updates {args.updates} --batch {args.batch} --buffer {args.buffer} '
        f'--ewc-lambda {args.ewc_lambda} --joint-mult {args.joint_mult} '
        f'--limit {args.limit} ' +
        ('--report-test --probe ' if args.report_test else '') +
        f'--out {args.out}\n')
    device = torch.device(args.device)
    t0 = time.time()
    log = [f'REAL-CL P1.5 order={args.order} seed={args.seed} '
           f'device={args.device} report_test={args.report_test} '
           f'arms={args.arms}']
    set_seeds(args.seed)
    if args.data_expanded:
        graphs = load_graphs_expanded()
    else:
        graphs = p1.load_graphs()
    splits = p1.split_domains(graphs, args.seed, order, args.limit)
    data = {g['task_id']: p1.graph_to_v2_data(g)
            for d in splits for ph in splits[d].values() for g in ph}
    d0 = data[splits[order[0]]['train'][0]['task_id']]
    dims = (d0.x_base.size(1), max(d0.edge_attr_base.size(1), 1))
    log.append(f'x_base_dim={dims[0]} edge_base_dim={dims[1]}')
    results, ledger, per_item, score_rows = {}, [], [], []
    for arm in args.arms.split(','):
        t1 = time.time()
        acce, pi, extras, slog = run_arm(arm, splits, data, dims, order, args,
                                         device, args.report_test, args.probe,
                                         log)
        score_rows += slog
        results[arm] = extras
        per_item += pi
        (outdir / 'results.json').write_text(json.dumps(results, indent=1))
        ledger.append({'arm': arm, 'seconds': round(time.time() - t1, 1)})
        log.append(f'[{arm}] done {time.time() - t1:.1f}s bwt={extras["bwt"]} '
                   f'forget={extras["forgetting"]} fwt={extras["fwt"]}')
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
