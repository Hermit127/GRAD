#!/usr/bin/env python3
"""REAL-CL P1 runner (clean): real MCP 5-domain task sequence on CausalCharGNN.

Arms: naive / joint / replay(DER) / reg(EWC) / specialists / crl.
Router (specialists, crl): nearest-centroid over OBSERVABLE graph signatures
(node-type histogram + size), trained only on seen tasks; no task ID at test.

Usage (server):
  GRAD_BASE=. PYTHONPATH=$GRAD_BASE/code:$GRAD_BASE/code/repro \
    python code/repro/real_cl_p1.py \
    --domains financial_analysis,web_search --order forward --seed 211 \
    --device cuda:7 --updates 10 --limit 40 --out out/real_cl/p1_pilot
"""
from __future__ import annotations

import argparse
import copy
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from sklearn.metrics import f1_score, roc_auc_score  # noqa: E402
from torch_geometric.data import Batch  # noqa: E402

from train_gnn_v2 import CausalCharGNN, graph_to_v2_data  # noqa: E402

ALL_DOMAINS = ['financial_analysis', 'web_search', 'location_navigation',
               'repository_management', 'browser_automation']
NODE_TYPES = ['agent_message', 'code_exec', 'database_query', 'web_request',
              'web_search', 'file_read', 'file_write', 'shell_exec',
              'tool_execute', 'memory_write', 'memory_read', 'system_config',
              'email_send', 'message_send', 'social_media', 'file_delete',
              'account_access', 'version_control', 'data_aggregate',
              'document_create', 'agent_spawn', 'prompt_inject', 'other']


class Adapter(nn.Module):
    def __init__(self, d, r=8):
        super().__init__()
        self.down = nn.Linear(d, r, bias=False)
        self.up = nn.Linear(r, d, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x):
        return x + self.up(F.gelu(self.down(x)))


def domain_of(g):
    md = g.get('metadata', {})
    s = str(md.get('config', '')) + str(md.get('task', ''))
    for d in ALL_DOMAINS:
        if d in s:
            return d
    return 'other'


def load_graphs():
    files = ['ref/dataset/mcp_safetybench_defense_train.jsonl',
             'ref/dataset/mcp_safetybench_defense_test.jsonl',
             'ref/dataset/mcp_safetybench_defense_validation.jsonl'] + \
        [f for f in sorted(glob.glob('ref/dataset/mcp_residual_matched_v2_'
                                     'merged/graphs/*.jsonl'))
         if 'generated_all' not in f]
    seen, out = set(), []
    for f in files:
        for line in open(BASE / f):
            g = json.loads(line)
            h = hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            out.append(g)
    return out


def split_domains(graphs, seed, domains, limit):
    by = {}
    for g in graphs:
        by.setdefault(domain_of(g), []).append(g)
    out = {}
    for d in domains:
        gs = by.get(d, [])
        pos = [g for g in gs if g['label']]
        neg = [g for g in gs if not g['label']]
        rng = random.Random(seed + sum(map(ord, d)))
        rng.shuffle(pos)
        rng.shuffle(neg)
        if limit:
            pos, neg = pos[:limit // 2], neg[:limit - limit // 2]
        take = pos + neg
        rng.shuffle(take)
        n = len(take)
        n_tr, n_dv = int(n * 0.7), int(n * 0.15)
        out[d] = {'train': take[:n_tr], 'dev': take[n_tr:n_tr + n_dv],
                  'test': take[n_tr + n_dv:]}
    return out


def signature(g):
    v = [0.0] * len(NODE_TYPES)
    for n in g['nodes']:
        t = str(n.get('type', 'other'))
        v[NODE_TYPES.index(t) if t in NODE_TYPES else NODE_TYPES.index('other')] += 1
    n = max(len(g['nodes']), 1)
    return np.array([x / n for x in v] +
                    [len(g['nodes']) / 50.0, len(g.get('edges', [])) / 50.0])


class Router:
    def __init__(self):
        self.centroids, self.tasks = [], []

    def add_task(self, task, train_graphs):
        c = np.mean([signature(g) for g in train_graphs], axis=0)
        self.centroids.append(c)
        self.tasks.append(task)

    def route(self, g):
        v = signature(g)
        d = [float(np.linalg.norm(v - c)) for c in self.centroids]
        return self.tasks[int(np.argmin(d))]


def make_model(dims, device):
    return CausalCharGNN(node_base_dim=dims[0], edge_base_dim=dims[1]).to(device)


def fwd(m, b):
    return m(b.x_base, b.node_chars, b.edge_index, b.edge_attr_base,
             b.edge_chars, b.batch)


def batch_of(data, gs, device):
    b = Batch.from_data_list([data[g['task_id']] for g in gs]).to(device)
    y = torch.tensor([float(g['label']) for g in gs], device=device)
    return b, y


def train_updates(m, data, graphs, updates, batch_size, seed, device,
                  replay=None, penalty=None, adapter=None, frozen_backbone=False):
    """Train with optional adapter (hook) and soft-target replay.

    Fixes vs the audited version (LOGS/real_cl_independent_review_20260915.md):
    - adapter parameters are part of the optimizer and grad clipping;
    - frozen_backbone keeps BatchNorm running stats frozen (eval mode);
    - replay soft targets are probabilities, distilled with BCE-with-logits
      (logit space) instead of MSE(logits, probs).
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
    losses = []
    for step in range(updates):
        m.train()
        if frozen_backbone:
            for mod in m.modules():
                if isinstance(mod, nn.modules.batchnorm._BatchNorm):
                    mod.eval()
        buf = replay.sample(batch_size // 2, rng) if replay is not None else []
        cur = [graphs[(step * batch_size + i) % len(graphs)]
               for i in range(batch_size - len(buf))]
        gs = cur + [b[0] for b in buf]
        b, y = batch_of(data, gs, device)
        opt.zero_grad()
        logits = fwd(m, b)
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
        losses.append(float(loss))
    if handle is not None:
        handle.remove()
    return sum(losses) / max(len(losses), 1)


def scores_for(m, data, gs, device):
    m.eval()
    out = {}
    with torch.no_grad():
        for i in range(0, len(gs), 16):
            ch = gs[i:i + 16]
            b = Batch.from_data_list([data[g['task_id']] for g in ch]).to(device)
            for g, p in zip(ch, torch.sigmoid(fwd(m, b)).cpu().tolist()):
                out[g['task_id']] = p
    return out


def metrics(gs, scores):
    ys = [g['label'] for g in gs]
    ps = [scores[g['task_id']] for g in gs]
    pred = [1 if p >= 0.5 else 0 for p in ps]
    tp = sum(1 for y, p in zip(ys, pred) if y and p)
    fp = sum(1 for y, p in zip(ys, pred) if not y and p)
    fn = sum(1 for y, p in zip(ys, pred) if y and not p)
    tn = sum(1 for y, p in zip(ys, pred) if not y and not p)
    try:
        a = float(roc_auc_score(ys, ps)) if len(set(ys)) > 1 else float('nan')
    except ValueError:
        a = float('nan')
    return {'acc': (tp + tn) / max(len(ys), 1), 'auc': a,
            'f1': float(f1_score(ys, pred, zero_division=0)),
            'tpr': tp / max(tp + fn, 1), 'fpr': fp / max(fp + tn, 1),
            'n': len(ys), 'tp': tp, 'fp': fp}


class ReplayBuffer:
    """History-aware bounded replay buffer (DER-style, one quota per task).

    Fixes vs the audited version (LOGS/real_cl_independent_review_20260915.md):
    the buffer is not rebuilt per stage; every seen task contributes a bounded
    quota so the whole history stream is replayed, not just the last task.
    Soft targets are probabilities; the training loop distills them in logit
    space with BCE-with-logits.
    """

    def __init__(self, cap):
        self.cap = cap
        self.stores = []

    def add_task(self, graphs, scores, seed):
        rng = random.Random(seed)
        take = list(graphs)
        rng.shuffle(take)
        quota = max(1, self.cap // (len(self.stores) + 1))
        self.stores.append([(g, scores[g['task_id']]) for g in take[:quota]])
        for i in range(len(self.stores) - 1):
            if len(self.stores[i]) > quota:
                self.stores[i] = self.stores[i][:quota]

    def fill(self, graphs, scores, seed):
        """Backward-compatible alias: adds this task to the history buffer."""
        self.add_task(graphs, scores, seed)

    def sample(self, n, rng):
        pool = [it for s in self.stores for it in s]
        if not pool:
            return []
        return [pool[rng.randrange(len(pool))] for _ in range(n)]


class EWC:
    """Diagonal EWC with differentiable penalty.

    Fixes vs the audited version (LOGS/real_cl_independent_review_20260915.md):
    - Fisher is accumulated from named_parameters().grad after backward(),
      not from detached state_dict() tensors (which are always None -> Fisher 0);
    - the penalty is computed from live parameters, so requires_grad is True
      and it actually constrains optimization.
    """

    def __init__(self, lam):
        self.lam, self.fisher, self.star = lam, {}, {}

    def update(self, m, data, graphs, device, batches, seed):
        m.eval()
        rng = random.Random(seed)
        params = {k: p for k, p in m.named_parameters() if p.requires_grad}
        f = {k: torch.zeros_like(p) for k, p in params.items()}
        for _ in range(batches):
            gs = [graphs[rng.randrange(len(graphs))] for _ in range(8)]
            b, y = batch_of(data, gs, device)
            m.zero_grad()
            F.binary_cross_entropy_with_logits(fwd(m, b), y).backward()
            for k, p in params.items():
                if p.grad is not None:
                    f[k] += p.grad.detach().pow(2) / batches
        for k in f:
            self.fisher[k] = self.fisher.get(k, torch.zeros_like(f[k])) + f[k]
        self.star = {k: p.detach().clone() for k, p in params.items()}

    def penalty(self, m):
        dev = next(m.parameters()).device
        if not self.fisher:
            return torch.zeros((), device=dev)
        s = torch.zeros((), device=dev)
        for k, p in m.named_parameters():
            if k in self.fisher and k in self.star:
                s = s + (self.fisher[k] * (p - self.star[k]).pow(2)).sum()
        return self.lam * s


def expert_scores(m, data, gs, adapter, head, device):
    """Evaluate the shared encoder with a task adapter + task head."""
    hidden = m.node_proj.out_features
    old_cls = m.classifier
    m.classifier = head
    handle = m.tf2.register_forward_hook(lambda mod, i, o: adapter(o))
    sc = scores_for(m, data, gs, device)
    handle.remove()
    m.classifier = old_cls
    return sc


def run_arm(arm, splits, data, dims, order, args, device, outdir, log):
    tasks = list(order)
    evals, per_item = {}, []
    m = make_model(dims, device)
    hidden = m.node_proj.out_features
    experts = {}   # task -> (adapter, head)
    router = Router() if arm in ('specialists', 'crl') else None
    replay = ReplayBuffer(args.buffer_per_task) if arm == 'replay' else None
    ewc = EWC(args.ewc_lambda)

    if arm == 'joint':
        all_train = [g for t in tasks for g in splits[t]['train']]
        train_updates(m, data, all_train, args.updates * len(tasks),
                      args.batch, args.seed, device)
        for t in tasks:
            sc = scores_for(m, data, splits[t]['test'], device)
            evals[(len(tasks) - 1, t)] = metrics(splits[t]['test'], sc)
        torch.save(m.state_dict(), outdir / f'final_{arm}.pt')
        return evals, per_item, {'router_hit': None}

    if arm == 'specialists':
        train_updates(m, data, splits[tasks[0]]['train'], args.updates,
                      args.batch, args.seed, device)
        for p in m.parameters():
            p.requires_grad = False
    if arm == 'crl':
        for p in m.parameters():
            p.requires_grad = True

    for ti, t in enumerate(tasks):
        if arm in ('specialists', 'crl'):
            adapter = Adapter(hidden, r=8).to(device)
            head = nn.Linear(hidden, 1).to(device)
            old_cls = m.classifier
            m.classifier = head
            if arm == 'specialists':
                for p in m.parameters():
                    p.requires_grad = False
                for p in list(adapter.parameters()) + list(head.parameters()):
                    p.requires_grad = True
            train_updates(m, data, splits[t]['train'], args.updates,
                          args.batch, args.seed + ti, device, adapter=adapter,
                          frozen_backbone=(arm == 'specialists'))
            m.classifier = old_cls
            experts[t] = ({k: v.detach().clone() for k, v in adapter.state_dict().items()},
                          {k: v.detach().clone() for k, v in head.state_dict().items()})
            router.add_task(t, splits[t]['train'])
        else:
            penalty = ewc.penalty if arm == 'reg' and ti > 0 else None
            train_updates(m, data, splits[t]['train'], args.updates, args.batch,
                          args.seed + ti, device,
                          replay=replay if arm == 'replay' else None,
                          penalty=penalty)
            if arm == 'reg':
                ewc.update(m, data, splits[t]['train'], device, 8, args.seed + ti)
            if arm == 'replay':
                sc_tr = scores_for(m, data, splits[t]['train'], device)
                replay.add_task(splits[t]['train'], sc_tr, args.seed + ti)

        for d in tasks[:ti + 1]:
            if router is not None:
                sc = {}
                for g in splits[d]['test']:
                    rt = router.route(g)
                    ad_sd, hd_sd = experts[rt]
                    adapter = Adapter(hidden, r=8).to(device)
                    adapter.load_state_dict(ad_sd)
                    head = nn.Linear(hidden, 1).to(device)
                    head.load_state_dict(hd_sd)
                    sc.update(expert_scores(m, data, [g], adapter, head, device))
            else:
                sc = scores_for(m, data, splits[d]['test'], device)
            evals[(ti, d)] = metrics(splits[d]['test'], sc)
            per_item += [{'arm': arm, 'stage': ti + 1, 'domain': d,
                          'task_id': k, 'score': v,
                          'label': next(g['label'] for g in splits[d]['test']
                                        if g['task_id'] == k)}
                         for k, v in sc.items()]
        log.append(f'[{arm}] after {t}: ' + '; '.join(
            f'{d}:acc={evals[(ti, d)]["acc"]:.3f}/{evals[(ti, d)]["auc"]:.3f}'
            for d in tasks[:ti + 1]))

    router_hit = None
    if router is not None:
        hit = tot = 0
        for d in tasks:
            for g in splits[d]['test']:
                hit += int(router.route(g) == d)
                tot += 1
        router_hit = hit / max(tot, 1)
    torch.save(m.state_dict(), outdir / f'final_{arm}.pt')
    if experts:
        torch.save(experts, outdir / f'experts_{arm}.pt')
    return evals, per_item, {'router_hit': router_hit}


def summarize(arm, evals, order):
    n_tasks = len(order)
    acc = {f'{k[0]}|{k[1]}': round(v['acc'], 4) for k, v in evals.items()}
    bwt, forget = [], []
    for i, d in enumerate(order):
        learned = evals.get((i, d), {}).get('acc')
        final = evals.get((n_tasks - 1, d), {}).get('acc')
        best = max((evals.get((j, d), {}).get('acc', -1)
                    for j in range(i, n_tasks)), default=None)
        if learned is not None and final is not None:
            bwt.append(final - learned)
        if best is not None and final is not None:
            forget.append(best - final)
    return {'acc_matrix': acc,
            'bwt': float(np.mean(bwt)) if bwt else None,
            'forgetting': float(np.mean(forget)) if forget else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--domains', default=','.join(ALL_DOMAINS))
    ap.add_argument('--order', default='forward')
    ap.add_argument('--seed', type=int, default=211)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--arms', default='naive,joint,replay,reg,specialists,crl')
    ap.add_argument('--updates', type=int, default=200)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--buffer-per-task', type=int, default=20)
    ap.add_argument('--ewc-lambda', type=float, default=100.0)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', default='')
    args = ap.parse_args()
    order = args.domains.split(',')
    if args.order == 'reversed':
        order = list(reversed(order))
    outdir = Path(args.out) if args.out else \
        BASE / f'code/repro/out/real_cl/p1_{args.order}_{args.seed}'
    if outdir.exists():
        raise SystemExit(f'refusing to overwrite {outdir}')
    outdir.mkdir(parents=True)
    (outdir / 'config.json').write_text(json.dumps(
        {**vars(args), 'order_list': order, 'base': str(BASE)}, indent=1))
    (outdir / 'commands.sh').write_text(
        f'GRAD_BASE={BASE} PYTHONPATH={BASE}/code:{BASE}/code/repro '
        f'python code/repro/real_cl_p1.py --domains {args.domains} '
        f'--order {args.order} --seed {args.seed} --device {args.device} '
        f'--updates {args.updates} --batch {args.batch} '
        f'--arms {args.arms} --limit {args.limit} --out {args.out}\n')
    device = torch.device(args.device)
    t0 = time.time()
    log = [f'REAL-CL P1 order={args.order} seed={args.seed} device={args.device} '
           f'arms={args.arms} updates={args.updates} batch={args.batch}']
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    graphs = load_graphs()
    splits = split_domains(graphs, args.seed, order, args.limit)
    log.append('split sizes: ' + json.dumps(
        {d: {k: len(v) for k, v in ph.items()} for d, ph in splits.items()}))
    data = {g['task_id']: graph_to_v2_data(g)
            for d in splits for ph in splits[d].values() for g in ph}
    d0 = data[splits[order[0]]['train'][0]['task_id']]
    dims = (d0.x_base.size(1), max(d0.edge_attr_base.size(1), 1))
    log.append(f'x_base_dim={dims[0]} edge_base_dim={dims[1]}')
    results, ledger, per_item = {}, [], []
    for arm in args.arms.split(','):
        t1 = time.time()
        evals, pi, extra = run_arm(arm, splits, data, dims, order, args,
                                   device, outdir, log)
        results[arm] = {'evals': {f'{k[0]}|{k[1]}': v for k, v in evals.items()},
                        **extra, **summarize(arm, evals, order)}
        per_item += pi
        ledger.append({'arm': arm, 'seconds': round(time.time() - t1, 1),
                       'finish': time.strftime('%FT%T')})
        (outdir / 'results.json').write_text(json.dumps(results, indent=1))
        log.append(f'[{arm}] done in {time.time() - t1:.1f}s '
                   f'bwt={results[arm]["bwt"]} forget={results[arm]["forgetting"]}')
    (outdir / 'per_item.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in per_item) + '\n')
    (outdir / 'ledger.jsonl').write_text(
        '\n'.join(json.dumps(r) for r in ledger) + '\n')
    log.append(f'ALL DONE seconds={time.time() - t0:.1f}')
    (outdir / 'log.txt').write_text('\n'.join(log) + '\n')
    print('\n'.join(log))
    print('saved', outdir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
