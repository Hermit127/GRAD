#!/usr/bin/env python3
"""Aggregate REAL-CL P1.5 runs: standard CL tables + paired tests + probes.

Reads a tree of run dirs named p15_{order}_{seed}/ (skips PARTIAL/tune),
loads results.json (per-stage), per_item.jsonl (per-domain metrics incl. F1),
and writes markdown + JSON summaries:
- per-arm BWT / forgetting / FWT / router / final macro-ACC (mean±std over runs)
- per-domain final ACC/F1 (mean±std)
- same-seen-set s4->s5 delta per order
- paired tests vs naive (Wilcoxon signed-rank; sign test fallback)
- linear-probe acc + adjacent CKA per stage (specialists/crl)
- specialist adapter transfer matrix (mean F1 src->dst)

Usage:
  python code/repro/real_cl_p15_aggregate.py --src /tmp/realcl/p15all \
      --out-md LOGS/real_cl_p15_results_20260914.md \
      --out-json code/repro/out/real_cl/p15_aggregate.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
from pathlib import Path

import numpy as np

ARMS_ALL = ["naive", "joint", "replay", "ewc", "specialists",
            "specialists_oracle", "specialists_random", "specialists_all",
            "specialists_conf", "specialists_replay",
            "single_expert", "crl", "crl_replay"]
DOMAINS = ["financial_analysis", "web_search", "location_navigation",
           "repository_management", "browser_automation"]


def ms(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    return st.mean(vals), st.pstdev(vals)


def fmt(mean, std):
    return f"{mean:+.3f}±{std:.3f}" if mean is not None else "--"


def bootstrap_ci(vals, n_boot=2000, alpha=0.05, seed=0):
    vals = [v for v in vals if v is not None]
    if len(vals) < 4:
        return None
    rng = np.random.default_rng(seed)
    arr = np.asarray(vals, dtype=float)
    means = [float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
             for _ in range(n_boot)]
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def wilcoxon(a, b):
    """Two-sided Wilcoxon signed-rank without scipy (normal approx)."""
    diffs = [x - y for x, y in zip(a, b) if x is not None and y is not None
             and x != y]
    n = len(diffs)
    if n < 5:
        return None, n
    ranks = sorted(range(n), key=lambda i: abs(diffs[i]))
    rank_of = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(diffs[ranks[j + 1]]) == abs(diffs[ranks[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            rank_of[ranks[k]] = avg
        i = j + 1
    w_pos = sum(rank_of[i] for i in range(n) if diffs[i] > 0)
    w_neg = sum(rank_of[i] for i in range(n) if diffs[i] < 0)
    w = min(w_pos, w_neg)
    mu = n * (n + 1) / 4
    sd = math.sqrt(n * (n + 1) * (2 * n + 1) / 24)
    z = (w - mu + 0.5) / sd if sd else 0.0
    p = 2 * (0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    p = min(1.0, 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2)))))
    return p, n


def load_runs(src):
    runs = {}
    for d in sorted(os.listdir(src)):
        rd = os.path.join(src, d)
        if not os.path.isdir(rd) or 'PARTIAL' in d or 'tune' in d:
            continue
        rj = os.path.join(rd, 'results.json')
        if not os.path.exists(rj):
            continue
        order = 'forward' if 'forward' in d else 'reversed'
        runs[d] = {'order': order, 'seed': d.split('_')[-1], 'dir': d,
                   'res': json.load(open(rj)),
                   'per_item': os.path.join(rd, 'per_item.jsonl')}
    return runs


def arm_metric(run, arm, key):
    return run['res'][arm].get(key)


def macro_acc(run, arm):
    n = len(task_order(run))
    cell = [v for k, v in run['res'][arm]['acc_matrix'].items()
            if k.startswith(f'{n-1}|')]
    return st.mean(cell) if cell else None


def task_order(run):
    am = run['res'][next(iter(run['res']))]['acc_matrix']
    first = {}
    for k in am:
        s, d = k.split('|')
        first[d] = min(first.get(d, 99), int(s))
    return [d for d, _ in sorted(first.items(), key=lambda kv: kv[1])]


def same4_delta(run, arm, order):
    od = order
    n = len(od)
    a = [run['res'][arm]['acc_matrix'].get(f'{n-2}|{d}') for d in od[:n-1]]
    b = [run['res'][arm]['acc_matrix'].get(f'{n-1}|{d}') for d in od[:n-1]]
    a = [x for x in a if x is not None]
    b = [x for x in b if x is not None]
    if not a or not b:
        return None
    return st.mean(b) - st.mean(a)


def per_item_metrics(run, arm):
    """Aggregate per_item.jsonl -> {(stage,domain): metrics}; final F1."""
    out = {}
    p = run['per_item']
    if not os.path.exists(p):
        return out
    for line in open(p):
        if not line.strip():
            continue
        r = json.loads(line)
        if r['arm'] != arm:
            continue
        out[(r['stage'], r['domain'])] = r
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--out-md', required=True)
    ap.add_argument('--title', default='REAL-CL 聚合结果')
    ap.add_argument('--out-json', required=True)
    args = ap.parse_args()
    runs = load_runs(args.src)
    names = sorted(runs)
    orders = sorted({runs[r]['order'] for r in names})
    all_tasks = []
    for r in names:
        for d in task_order(runs[r]):
            if d not in all_tasks:
                all_tasks.append(d)
    arms = [a for a in ARMS_ALL if any(a in runs[r]['res'] for r in names)]
    lines = [f'# {args.title}（{len(names)} runs；'
             f'orders={orders}）', '',
             '数据：真实 MCP 5-domain 流，CausalCharGNN；dev 最优 checkpoint；'
             '基线 dev 定标（buffer=100、EWC λ=100、joint mult=3）。', '',
             '## 主表（mean±std over runs）', '',
             '| 臂 | BWT | forgetting | forgetting(T-1) | FWT | router | final macro-ACC |',
             '|---|---|---|---|---|---|---|']
    summary = {}
    for a in arms:
        bwt = [arm_metric(runs[r], a, 'bwt') for r in names]
        fog = [arm_metric(runs[r], a, 'forgetting') for r in names]
        fog1 = [arm_metric(runs[r], a, 'forgetting_t1') for r in names]
        fwt = [arm_metric(runs[r], a, 'fwt') for r in names]
        rt = [arm_metric(runs[r], a, 'router_hit') for r in names]
        ma = [macro_acc(runs[r], a) for r in names]
        summary[a] = {'bwt': ms(bwt), 'forgetting': ms(fog), 'fwt': ms(fwt),
                      'router': ms(rt), 'macro_acc': ms(ma),
                      'macro_acc_ci': bootstrap_ci(ma),
                      'forgetting_ci': bootstrap_ci(fog),
                      'paired_vs_naive': {}, 'same4': {}}
        lines.append(f"| {a} | {fmt(*ms(bwt))} | {fmt(*ms(fog))} | "
                     f"{fmt(*ms(fog1)) if any(x is not None for x in fog1) else '--'} | "
                     f"{fmt(*ms(fwt))} | "
                     f"{fmt(*ms(rt)) if any(x is not None for x in rt) else '--'} | "
                     f"{fmt(*ms(ma))} |")
    lines += ['', '> 95% bootstrap CI（主表宏指标与配对差的 JSON 字段：'
              '`macro_acc_ci` / `forgetting_ci` / `delta_*_ci`，percentile, 2000 次）', '']
    # paired tests
    lines += ['', '## 配对检验 vs naive（Wilcoxon 双侧；n<5 不给 p）', '',
              '| 臂 | Δmacro-ACC (mean) | p | Δforgetting | p | wins |',
              '|---|---|---|---|---|---|']
    raw_ps = {'macro_acc': {}, 'forgetting': {}}
    for a in arms:
        if a == 'naive':
            continue
        x = [macro_acc(runs[r], a) for r in names]
        y = [macro_acc(runs[r], 'naive') for r in names]
        d = [xx - yy for xx, yy in zip(x, y) if xx is not None and yy is not None]
        p1, n1 = wilcoxon(x, y)
        fx = [arm_metric(runs[r], a, 'forgetting') for r in names]
        fy = [arm_metric(runs[r], 'naive', 'forgetting') for r in names]
        p2, n2 = wilcoxon(fx, fy)
        wins = sum(1 for xx, yy in zip(x, y) if xx is not None and yy is not None
                   and xx > yy)
        raw_ps['macro_acc'][a], raw_ps['forgetting'][a] = p1, p2
        summary[a]['paired_vs_naive'] = {
            'delta_macro_acc': st.mean(d) if d else None, 'p_macro_acc': p1,
            'delta_macro_acc_ci': bootstrap_ci(d),
            'delta_forgetting_ci': bootstrap_ci(
                [aa - bb for aa, bb in zip(fx, fy)
                 if aa is not None and bb is not None]),
            'delta_forgetting': st.mean(
                [aa - bb for aa, bb in zip(fx, fy)
                 if aa is not None and bb is not None]) if fx else None,
            'p_forgetting': p2, 'wins': wins, 'n': len(names)}
        lines.append(f"| {a} | "
                     f"{(st.mean(d) if d else float('nan')):+.3f} | "
                     f"{p1 if p1 is not None else 'n/a'} | "
                     f"{(st.mean([aa - bb for aa, bb in zip(fx, fy) if aa is not None and bb is not None]) if fx else float('nan')):+.3f} | "
                     f"{p2 if p2 is not None else 'n/a'} | {wins}/{len(names)} |")
    # Holm-Bonferroni correction across arms (multiple comparisons)
    def holm(ps):
        items = [(a, p) for a, p in ps.items() if p is not None]
        items.sort(key=lambda kv: kv[1])
        m = len(items)
        out, prev = {}, 0.0
        for i, (a, p) in enumerate(items):
            adj = min(1.0, p * (m - i))
            adj = max(adj, prev)
            prev = adj
            out[a] = adj
        return out
    holm_acc, holm_fog = holm(raw_ps['macro_acc']), holm(raw_ps['forgetting'])
    for a, p in holm_acc.items():
        summary[a]['paired_vs_naive']['p_macro_acc_holm'] = p
    for a, p in holm_fog.items():
        summary[a]['paired_vs_naive']['p_forgetting_holm'] = p
    lines += ['', '**Holm–Bonferroni（按臂校正，见表 p_holm 列于 JSON）**：'
              + '；'.join(f"{a}: ACC {p:.4f} / forget {holm_fog.get(a, float('nan')):.4f}"
                          for a, p in sorted(holm_acc.items()))]

    # same-seen-set by order
    lines += ['', '## 同域集合（前 4 任务）s4→s5 变化', '',
              '| 臂 | forward | reversed |', '|---|---|---|']
    for a in arms:
        row = []
        for o in ('forward', 'reversed'):
            vals = [same4_delta(runs[r], a, task_order(runs[r])) for r in names
                    if runs[r]['order'] == o]
            m, s = ms(vals)
            row.append(fmt(m, s) if m is not None else '--')
        summary[a]['same4'] = dict(zip(('forward', 'reversed'), row))
        lines.append(f"| {a} | {row[0]} | {row[1]} |")
    # per-domain final ACC / F1
    for metric in ('acc', 'f1'):
        lines += ['', f'## 每域 final {metric.upper()}（mean±std over runs）', '',
                  '| 臂 | ' + ' | '.join(d[:8] for d in all_tasks) + ' |',
                  '|' + '---|' * (len(all_tasks) + 1)]
        for a in arms:
            cells = []
            for d in all_tasks:
                vals = []
                for r in names:
                    pi = per_item_metrics(runs[r], a)
                    n = len(task_order(runs[r]))
                    rec = pi.get((n, d)) or pi.get((n - 1, d))
                    if rec and metric in rec:
                        vals.append(rec[metric])
                m, s = ms(vals)
                cells.append(f"{m:.3f}±{s:.3f}" if m is not None else '--')
            lines.append(f"| {a} | " + ' | '.join(cells) + ' |')
    # probes
    lines += ['', '## CRL 探针（linear probe acc / adjacent CKA）', '',
              '| 臂 | s1 | s2 | s3 | s4 | s5 |', '|---|---|---|---|---|---|']
    probe_summary = {}
    for a in ('specialists', 'crl'):
        cells = []
        per_stage = []
        for stage in range(1, 6):
            pas, ckas = [], []
            for r in names:
                pr = runs[r]['res'][a].get('probe', [])
                rec = next((p for p in pr if p['stage'] == stage), None)
                if not rec:
                    continue
                vals = [v for v in rec['probe_acc'].values() if v is not None]
                if vals:
                    pas.append(st.mean(vals))
                if rec.get('cka_prev') is not None:
                    ckas.append(rec['cka_prev'])
            m1, s1 = ms(pas)
            m2, s2 = ms(ckas)
            cells.append(f"{m1:.3f}" + (f"/{m2:.3f}" if m2 is not None else '')
                         if m1 is not None else '--')
            per_stage.append({'probe': ms(pas), 'cka_prev': ms(ckas)})
        probe_summary[a] = per_stage
        lines.append(f"| {a} | " + ' | '.join(cells) + ' |')
    # transfer matrix
    tsum = {}
    for r in names:
        tf = runs[r]['res']['specialists'].get('transfer_f1')
        if not tf:
            continue
        for src, row in tf.items():
            for dst, v in row.items():
                tsum.setdefault((src, dst), []).append(v)
    if tsum:
        lines += ['', '## specialist adapter 迁移矩阵（mean F1，行=专家，列=测试域）', '',
                  '| 专家\\测试 | ' + ' | '.join(d[:8] for d in all_tasks) + ' |',
                  '|' + '---|' * (len(all_tasks) + 1)]
        for src in all_tasks:
            cells = []
            for dst in all_tasks:
                m, s = ms(tsum.get((src, dst), []))
                cells.append(f"{m:.3f}" if m is not None else '--')
            lines.append(f"| {src[:8]} | " + ' | '.join(cells) + ' |')
    out = {'n_runs': len(names), 'runs': names, 'summary': summary,
           'probe': probe_summary}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps(out, indent=1, default=str))
    Path(args.out_md).write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))
    print('saved', args.out_md, args.out_json)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
