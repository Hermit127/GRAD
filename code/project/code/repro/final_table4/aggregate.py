#!/usr/bin/env python
"""Table 4 aggregation: ASR / BU / UA with paired statistics and provenance.

Reads one or more results.json files, resolves one record per frozen case key,
computes per-cell metrics with Wilson CIs, error-aware ASR bounds, per-family
counts, exact paired McNemar tests against the same-backbone None arm, and
writes LOGS/final_table4/table4.{md,tex,json} + cell_provenance.json + a cost
appendix.

Usage:
  python aggregate.py --results LOGS/final_table4/main/results.json \
      --out LOGS/final_table4
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ARMS = ('none', 'monitor', 'sanitizer', 'grad')
ARM_LABELS = {'none': 'None', 'monitor': 'Sequential Monitor',
              'sanitizer': 'Sanitizer', 'grad': 'GRAD (Ours)'}
BACKBONE_LABELS = {'deepseek-flash': 'DeepSeek Flash',
                   'glm-5.3-flash': 'GLM-5.3-Flash'}
COND_ATTACK = 'attack_instruction'
COND_BENIGN = 'benign_instruction'


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (None, None)
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, center - half), min(1.0, center + half))


def binom_two_sided(b: int, c: int) -> float:
    """Exact two-sided binomial test for McNemar (no scipy)."""
    n = b + c
    if n == 0:
        return 1.0
    # two-sided exact p for p=0.5
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def resolve(records: list[dict]) -> dict[tuple, dict]:
    """One record per case key: last error-free if any, else last."""
    by = defaultdict(list)
    for r in records:
        by[r['case_key']].append(r)
    out = {}
    for key, attempts in by.items():
        ok = [a for a in attempts if not a.get('error')]
        out[key] = (ok or attempts)[-1]
    return out


def cell_metrics(runs: list[dict]) -> dict:
    attack = [r for r in runs if r['condition'] == COND_ATTACK]
    benign = [r for r in runs if r['condition'] == COND_BENIGN]
    n_a, n_b = len(attack), len(benign)
    harm = sum(1 for r in attack if r['independent_harm'])
    ua = sum(1 for r in attack if r['independent_completion'])
    bu = sum(1 for r in benign if r['independent_completion'])
    err_a = sum(1 for r in attack if r.get('error'))
    err_b = sum(1 for r in benign if r.get('error'))
    false_alarm = sum(1 for r in benign if r['independent_harm'])
    return {
        'n_attack': n_a, 'n_benign': n_b,
        'harm': harm, 'ua_count': ua, 'bu_count': bu,
        'asr': harm / n_a if n_a else None,
        'bu': bu / n_b if n_b else None,
        'ua': ua / n_a if n_a else None,
        'asr_ci': wilson(harm, n_a), 'bu_ci': wilson(bu, n_b),
        'ua_ci': wilson(ua, n_a),
        'asr_lower': harm / n_a if n_a else None,
        'asr_upper': (harm + err_a) / n_a if n_a else None,
        'errors_attack': err_a, 'errors_benign': err_b,
        'benign_false_alarm': false_alarm,
        'blocked_runs': sum(1 for r in runs if r.get('blocked')),
    }


def per_family(runs: list[dict]) -> dict:
    out = defaultdict(lambda: {'attack': [0, 0], 'benign': [0, 0]})
    for r in runs:
        fam = f"{r['env']}:{r['family']}"
        if r['condition'] == COND_ATTACK:
            out[fam]['attack'][1] += 1
            out[fam]['attack'][0] += int(r['independent_harm'])
        else:
            out[fam]['benign'][1] += 1
            out[fam]['benign'][0] += int(r['independent_completion'])
    return {k: v for k, v in sorted(out.items())}


def mcnemar_vs_none(runs: list[dict], metric_key: str,
                    condition: str) -> dict | None:
    """Paired on (env, family, rep); None arm must be present with same reps."""
    base = {}
    for r in runs:
        if r['condition'] != condition:
            continue
        key = (r['backbone'], r['env'], r['family'], r['repeat'])
        if r['arm'] == 'none':
            base[key] = bool(r[metric_key])
    if not base:
        return None
    results = {}
    for arm in ARMS:
        if arm == 'none':
            continue
        b = c = 0
        pairs = 0
        for r in runs:
            if r['arm'] != arm or r['condition'] != condition:
                continue
            key = (r['backbone'], r['env'], r['family'], r['repeat'])
            if key not in base:
                continue
            pairs += 1
            none_val, arm_val = base[key], bool(r[metric_key])
            if none_val and not arm_val:
                b += 1
            elif arm_val and not none_val:
                c += 1
        results[arm] = {'n_pairs': pairs, 'b': b, 'c': c,
                        'p_exact': binom_two_sided(b, c)}
    return results


def fmt_pct(x, digits=1):
    return '--' if x is None else f'{100 * x:.{digits}f}'


def fmt_p(p):
    if p is None:
        return '--'
    return '<0.001' if p < 0.001 else f'{p:.3f}'


def build_table(results: list[dict], out_dir: Path, title: str,
                cost_summary: dict) -> dict:
    backbones = [b for b in BACKBONE_LABELS
                 if any(r['backbone'] == b for r in results)]
    cells = {}
    for backbone in backbones:
        for arm in ARMS:
            runs = [r for r in results
                    if r['backbone'] == backbone and r['arm'] == arm]
            if runs:
                cells[(backbone, arm)] = cell_metrics(runs)
    paired = {}
    for backbone in backbones:
        for cond, key, name in ((COND_ATTACK, 'independent_harm', 'harm'),
                                (COND_ATTACK, 'independent_completion',
                                 'ua'),
                                (COND_BENIGN, 'independent_completion',
                                 'bu')):
            runs = [r for r in results if r['backbone'] == backbone]
            m = mcnemar_vs_none(runs, key, cond)
            if m:
                paired[f'{backbone}|{name}'] = m
    fam = {}
    for backbone in backbones:
        for arm in ARMS:
            runs = [r for r in results
                    if r['backbone'] == backbone and r['arm'] == arm]
            if runs:
                fam[f'{backbone}|{arm}'] = per_family(runs)
    payload = {'generated_utc': datetime.now(timezone.utc).isoformat(),
               'n_records': len(results), 'cells': {
                   f'{b}|{a}': v for (b, a), v in cells.items()},
               'paired_vs_none': paired, 'per_family': fam,
               'cost': cost_summary}
    (out_dir / 'table4.json').write_text(json.dumps(
        payload, indent=2, ensure_ascii=False))
    write_md(cells, backbones, paired, fam, out_dir, cost_summary, title)
    write_tex(cells, backbones, out_dir)
    provenance = build_provenance(results, cells)
    (out_dir / 'cell_provenance.json').write_text(json.dumps(
        provenance, indent=2, ensure_ascii=False))
    return payload


def write_md(cells, backbones, paired, fam, out_dir, cost_summary, title):
    lines = [f'# {title}', '',
             '> Controlled two-session online closed loop '
             '(`code/repro/v10/envlib.py`), not AgentDojo. Same frozen task '
             'and attack protocol for both backbones; monitor/sanitizer '
             'auxiliary model fixed to DeepSeek Flash; GRAD v2 checkpoint and '
             'threshold frozen (no target-backbone retraining).',
             '']
    header = ('| Method | ' +
              ' | '.join(f'{BACKBONE_LABELS[b]}: ASR↓ | BU↑ | UA↑'
                         for b in backbones) + ' |')
    lines.append(header)
    lines.append('|---|' + '---|' * (3 * len(backbones)))
    for arm in ARMS:
        row = [ARM_LABELS[arm]]
        for b in backbones:
            c = cells.get((b, arm))
            if not c:
                row += ['--', '--', '--']
                continue
            row.append(f'{fmt_pct(c["asr"])} ({c["harm"]}/{c["n_attack"]})')
            row.append(f'{fmt_pct(c["bu"])} ({c["bu_count"]}/{c["n_benign"]})')
            row.append(f'{fmt_pct(c["ua"])} ({c["ua_count"]}/{c["n_attack"]})')
        lines.append('| ' + ' | '.join(row) + ' |')
    lines += ['', '## Paired tests vs same-backbone None (exact McNemar)', '']
    for key, m in sorted(paired.items()):
        for arm, s in sorted(m.items()):
            lines.append(f'- {key} | {ARM_LABELS[arm]}: n={s["n_pairs"]}, '
                         f'b={s["b"]}, c={s["c"]}, p={fmt_p(s["p_exact"])}')
    lines += ['', '## Per-family counts', '']
    for key, fams in sorted(fam.items()):
        parts = [f'{f}: harm {v["attack"][0]}/{v["attack"][1]}, '
                 f'benign-comp {v["benign"][0]}/{v["benign"][1]}'
                 for f, v in fams.items()]
        lines.append(f'- {key}: ' + '; '.join(parts))
    lines += ['', '## Cost', '',
              f'- budget files: {json.dumps(cost_summary, ensure_ascii=False)}']
    (out_dir / 'table4.md').write_text('\n'.join(lines) + '\n')


def write_tex(cells, backbones, out_dir):
    ncols = 1 + 3 * len(backbones)
    colspec = 'l' + 'ccc' * len(backbones)
    head_cols = ' & '.join(
        f'\\multicolumn{{3}}{{c}}{{{BACKBONE_LABELS[b]}}}' for b in backbones)
    sub = ' & '.join('ASR$\\downarrow$ & BU$\\uparrow$ & UA$\\uparrow$'
                     for _ in backbones)
    body = []
    for arm in ARMS:
        vals = []
        for b in backbones:
            c = cells.get((b, arm))
            if not c:
                vals += ['--', '--', '--']
            else:
                vals += [f'{fmt_pct(c["asr"])}', f'{fmt_pct(c["bu"])}',
                         f'{fmt_pct(c["ua"])}']
        row = f'{ARM_LABELS[arm]} & ' + ' & '.join(vals) + ' \\\\'
        if arm == 'none':
            row = '\\rowcolor{gray!15} ' + row
        body.append(row)
    tex = f"""\\begin{{table}}[t]
\\centering
\\caption{{Cross-backbone evaluation in the controlled two-session online
closed-loop environment (not AgentDojo); four held-out task families with five
repetitions per cell (20 runs per arm per backbone, errors kept in the
denominator). Both backbones use the same frozen task and attack protocol; the
Sequential Monitor and Sanitizer auxiliary model is fixed to DeepSeek Flash;
the GRAD checkpoint and threshold are frozen (frozen-transfer, no target
backbone recalibration). ASR/BU/UA are percentages; see the appendix for
counts, intervals, cost and disclosed adaptations.}}
\\label{{tab:cross-backbone}}
\\resizebox{{\\textwidth}}{{!}}{{%
\\begin{{tabular}}{{{colspec}}}
\\toprule
\\multirow{{2}}{{*}}{{Method}} & {head_cols} \\\\
\\cmidrule(lr){{2-{1 + 3 * len(backbones)}}}
 & {sub} \\\\
\\midrule
{chr(10).join(body)}
\\bottomrule
\\end{{tabular}}}}
\\end{{table}}
"""
    (out_dir / 'table4.tex').write_text(tex)


def build_provenance(results, cells) -> dict:
    prov = {}
    for (backbone, arm), c in cells.items():
        runs = [r for r in results
                if r['backbone'] == backbone and r['arm'] == arm]
        prov[f'{backbone}|{arm}'] = {
            'run_ids': sorted(r['run_id'] for r in runs),
            'n_runs': len(runs),
            'errors': [{'run_id': r['run_id'], 'error': r['error']}
                       for r in runs if r.get('error')],
            'blocked_runs': c['blocked_runs'],
            'source': 'results.json (per-run raw traces)',
        }
    return prov


def load_costs(result_files: list[Path]) -> dict:
    out = {}
    for rf in result_files:
        budget_dir = rf.parent / 'budget'
        for p in sorted(budget_dir.glob('budget_*.json')):
            d = json.loads(p.read_text())
            out.setdefault(p.parent.name, {})[p.stem] = {
                'cap_rmb': d['cap_rmb'],
                'charged_upper_rmb': d['charged_upper_rmb'],
                'charged_actual_rmb': d['charged_actual_rmb'],
                'calls': d['calls'], 'failed_calls': d['failed_calls']}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', nargs='+', required=True)
    ap.add_argument('--out', default=str(REPO / 'LOGS/final_table4'))
    ap.add_argument('--plan', default=None,
                    help='optional frozen plan; only its case keys are kept')
    ap.add_argument('--labels', default=None,
                    help='optional recomputed labels (amendment A1)')
    ap.add_argument('--title', default='Table 4. Evaluation across Agent '
                                       'Backbones')
    args = ap.parse_args()
    paths = [Path(p) for p in args.results if Path(p).exists()]
    records = []
    for p in paths:
        data = json.loads(p.read_text())
        records += data.get('runs', [])
    if args.labels:
        labels = json.loads(Path(args.labels).read_text())['labels']
        records = [dict(r, independent_harm=labels[r['run_id']]['new_harm'],
                        independent_completion=labels[r['run_id']]
                        ['new_completion'])
                   if r['run_id'] in labels else r for r in records]
    if args.plan:
        plan = json.loads(Path(args.plan).read_text())
        keep = {'|'.join([e['backbone'], e['env'], e['family'],
                          e['condition'], e['arm'], str(e['rep'])])
                for e in plan['entries']}
        records = [r for r in records if r['case_key'] in keep]
    resolved = resolve(records)
    results = list(resolved.values())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {}
    mp = out_dir / 'frozen_manifest.json'
    if mp.exists():
        manifest = json.loads(mp.read_text())
    for key in ('plan_dev', 'plan_test'):
        if args.plan and Path(manifest.get(key, {}).get('path', '')).name \
                != Path(args.plan).name:
            continue
        if key in manifest:
            expected = set()
            plan = json.loads((REPO / manifest[key]['path']).read_text())
            for e in plan['entries']:
                expected.add('|'.join([e['backbone'], e['env'], e['family'],
                                       e['condition'], e['arm'],
                                       str(e['rep'])]))
            have = set(resolved)
            manifest[key]['coverage'] = {
                'expected_cases': len(expected), 'resolved_cases': len(have),
                'missing_cases': sorted(expected - have)[:20],
                'extra_cases': sorted(have - expected)[:20]}
    payload = build_table(results, out_dir, args.title,
                          load_costs(paths))
    (out_dir / 'coverage.json').write_text(json.dumps(
        {k: manifest[k].get('coverage') for k in manifest
         if k.startswith('plan_')}, indent=2))
    print(json.dumps({'records': len(records), 'resolved_cases':
                      len(resolved), 'out': str(out_dir)}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
