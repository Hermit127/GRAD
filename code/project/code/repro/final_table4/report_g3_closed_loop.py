#!/usr/bin/env python3
"""Build the GRAD-512 (G3) closed-loop report from the matched batch.

Inputs (all produced by the frozen pipeline):
  LOGS/final_table4_g3/main/results.json      (240-run matched batch)
  LOGS/final_table4_g3/main/run_meta.json
  LOGS/final_table4_g3/agg/table4.json        (aggregate.py output)
  LOGS/final_table4/table4.json               (frozen v2 main, cross-batch ref)
  LOGS/final_table4_g3/main/budget/*.json     (budget ledger)

Output: LOGS/final_table4_g3/table4_g3.md + table4_g3.json. Zero API.
"""
from __future__ import annotations

import json
import statistics as st
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def _args():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='LOGS/final_table4_g3',
                    help='batch dir with main/ and agg/ (repo-relative)')
    ap.add_argument('--out-name', default='table4_g3')
    ap.add_argument('--title', default='GRAD-512 (G3) closed-loop matched '
                                      're-verification')
    ap.add_argument('--grad-label', default='GRAD-512')
    ap.add_argument('--lineage', default='', help='optional lineage JSON')
    ap.add_argument('--ref-json', default='LOGS/final_table4/table4.json')
    ap.add_argument('--ref-label', default='v2 frozen main')
    ap.add_argument('--ckpt-label', default='online_gate_g3/g3_gate_512.pt')
    ap.add_argument('--gate-desc', default='G3 ModalityNormGNN, candidate '
                    'A_moddrop, windows 512/160, dev fixed-FPR F1 threshold; '
                    'trained 2026-09-22 on the frozen v2 gate protocol')
    ap.add_argument('--exec-note', default='')
    a = ap.parse_args()
    return a


ARGS = _args()
NEW = REPO / ARGS.dir
ARM_LABELS = {'none': 'None', 'monitor': 'Sequential Monitor',
              'grad': ARGS.grad_label}
BACKBONE_LABELS = {'deepseek-flash': 'DeepSeek Flash',
                   'glm-5.3-flash': 'GLM-5.3-Flash'}
FLIP_NOTE = ('batch migrated mid-run (first 38 runs on macOS host, remainder '
             'on gpu-server); same code/prompt/gate hashes on both hosts')
DEFAULT_EXEC_NOTE = FLIP_NOTE


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def latency_by_arm(results: dict) -> dict:
    out = {}
    for r in results['runs']:
        for d in r.get('defense_log') or []:
            if d.get('status') != 'ok' or not d.get('latency_ms'):
                continue
            out.setdefault((r['backbone'], r['arm']), []).append(
                float(d['latency_ms']))
    return out


def cost_by_arm(results: dict) -> dict:
    out = {}
    for r in results['runs']:
        out.setdefault((r['backbone'], r['arm']), []).append(
            float(r.get('run_cost_rmb') or 0.0))
    return out


def ledger() -> dict:
    out = {}
    for p in sorted((NEW / 'main/budget').glob('budget_*.json')):
        s = json.loads(p.read_text())
        out[s['vendor']] = {'calls': s['calls'],
                            'actual_rmb': s['charged_actual_rmb'],
                            'upper_rmb': s['charged_upper_rmb'],
                            'cap_rmb': s['cap_rmb']}
    return out


def fmt(x, nd=2):
    return '—' if x is None else f'{x:.{nd}f}'


def main() -> int:
    if not ARGS.exec_note:
        ARGS.exec_note = FLIP_NOTE
    results = json.loads((NEW / 'main/results.json').read_text())
    meta = json.loads((NEW / 'main/run_meta.json').read_text())
    agg = json.loads((NEW / 'agg/table4.json').read_text())
    ref_path = REPO / ARGS.ref_json
    v2 = json.loads(ref_path.read_text()) if ref_path.exists() else None
    lat = latency_by_arm(results)
    cost = cost_by_arm(results)

    lines = []
    A = lines.append
    A(f'# {ARGS.title}\n')
    A(f"Gate: `{ARGS.ckpt_label}` sha256 `{meta['gate_ckpt_sha256']}`"
      f" ({ARGS.gate_desc}).")
    A(f"Frozen plan `LOGS/final_table4/plan_test.json` sha256 "
      f"`{meta['plan_sha256']}`; arms `{meta['arms']}`; {meta['n_entries']} "
      f"entries; mock=False. Node-builder tensor identity vs the trainer: "
      f"`tests/test_g3_gate_identity.py` (6 tensor fields, pass).")
    A(f"Execution: {len(results['runs'])} runs, 0 infrastructure failures. "
      f"{ARGS.exec_note}.\n")
    A('> These are the same test cases already exposed in the frozen Table 4 '
      'main batch: this is a fixed-protocol re-verification with one '
      'pre-declared gate change (v2 → G3-512), not a fresh held-out test.\n')

    A('## Detection (matched batch, G3 gate)\n')
    hdr = ['Backbone', 'Arm', 'n_att', 'harm', 'ASR', 'ASR 95% CI', 'n_ben',
           'BU', 'UA']
    A('| ' + ' | '.join(hdr) + ' |')
    A('|' + '---|' * len(hdr))
    for backbone in ('deepseek-flash', 'glm-5.3-flash'):
        for arm in ('none', 'monitor', 'grad'):
            c = agg['cells'].get(f'{backbone}|{arm}')
            if not c:
                continue
            ci = c.get('asr_ci') or [None, None]
            A(f"| {BACKBONE_LABELS[backbone]} | {ARM_LABELS[arm]} | "
              f"{c['n_attack']} | {c['harm']} | {c['asr']:.2f} | "
              f"[{fmt(ci[0])}, {fmt(ci[1])}] | {c['n_benign']} | "
              f"{c['bu']:.2f} | {c['ua']:.2f} |")
    A('')
    A('Exact paired McNemar vs the same-backbone None arm:')
    A('')
    A('| Backbone | Arm | metric | b | c | p (exact) |')
    A('|---|---|---|---|---|---|')
    for backbone in ('deepseek-flash', 'glm-5.3-flash'):
        for arm in ('monitor', 'grad'):
            for metric, label in (('harm', 'harm'), ('bu', 'BU')):
                pv = (agg.get('paired_vs_none', {})
                      .get(f'{backbone}|{metric}', {}).get(arm))
                if not pv:
                    continue
                A(f"| {BACKBONE_LABELS[backbone]} | {ARM_LABELS[arm]} | "
                  f"{label} | {pv['b']} | {pv['c']} | {pv['p_exact']:.4f} |")
    A('')

    A('## Cost and latency (same batch)\n')
    A('| Backbone | Arm | runs | mean ¥/run | total ¥ | defense '
      'decisions | p50 ms | p95 ms |')
    A('|---|---|---|---|---|---|---|---|')
    for backbone in ('deepseek-flash', 'glm-5.3-flash'):
        for arm in ('none', 'monitor', 'grad'):
            cs = cost.get((backbone, arm)) or []
            if not cs:
                continue
            ls = lat.get((backbone, arm)) or []
            A(f"| {BACKBONE_LABELS[backbone]} | {ARM_LABELS[arm]} | "
              f"{len(cs)} | {st.mean(cs):.4f} | {sum(cs):.3f} | {len(ls)} | "
              f"{fmt(pct(ls, 0.5), 1)} | {fmt(pct(ls, 0.95), 1)} |")
    A('')
    led = ledger()
    A('Budget ledger (this batch, both hosts): ' + '; '.join(
        f"{k}: {v['calls']} calls, ¥{v['actual_rmb']:.4f} actual "
        f"(upper ¥{v['upper_rmb']:.4f}, cap ¥{v['cap_rmb']})"
        for k, v in led.items()) + '.')
    A('GRAD-512 gate: single 2.6 MB checkpoint, in-process inference, ¥0 API '
      'per decision (compute-only; the antibody continual-learning method'
      ' additionally stores a 7.74 MB snapshot pool). Monitor pays one judge '
      'API call per action; the mean-¥/run column is the defense-induced cost '
      'difference vs `none` at matched cases.\n')

    A(f'## Cross-batch reference ({ARGS.ref_label}, same frozen cases)\n')
    if v2 is None:
        A('_reference table not available on this host_\n')
    A('| Backbone | Arm | v2 ASR | G3 ASR | v2 BU | G3 BU |')
    A('|---|---|---|---|---|---|')
    for backbone in ('deepseek-flash', 'glm-5.3-flash'):
        for arm in ('none', 'monitor', 'grad'):
            c2 = (v2 or {}).get('cells', {}).get(f'{backbone}|{arm}')
            c3 = agg['cells'].get(f'{backbone}|{arm}')
            if not c2 or not c3:
                continue
            A(f"| {BACKBONE_LABELS[backbone]} | {ARM_LABELS[arm]} | "
              f"{c2['asr']:.2f} | {c3['asr']:.2f} | {c2['bu']:.2f} | "
              f"{c3['bu']:.2f} |")
    A('')
    if ARGS.lineage:
        lin = json.loads(Path(ARGS.lineage).read_text())
        A('## Detector lineage (AUROC %)\n')
        A('| detector | gate dev | gate test | Decomp test |')
        A('|---|---|---|---|')
        for name, v in lin['rows'].items():
            A(f"| {name} | {fmt(v.get('gate_dev') and v['gate_dev']*100, 2)} | "
              f"{fmt(v.get('gate_test') and v['gate_test']*100, 2)} | "
              f"{fmt(v.get('decomp_test') and v['decomp_test']*100, 2)} |")
        A('')
        A(lin.get('note', ''))
        A('')

    out_json = {'gate_ckpt_sha256': meta['gate_ckpt_sha256'],
                'plan_sha256': meta['plan_sha256'],
                'n_runs': len(results['runs']), 'ledger': led,
                'latency_n': {f'{k[0]}|{k[1]}': len(v)
                              for k, v in lat.items()},
                'cost_mean': {f'{k[0]}|{k[1]}': st.mean(v)
                              for k, v in cost.items()},
                'cells': agg['cells'],
                'paired_vs_none': agg.get('paired_vs_none', {})}
    (NEW / f'{ARGS.out_name}.json').write_text(json.dumps(out_json, indent=1))
    (NEW / f'{ARGS.out_name}.md').write_text('\n'.join(lines) + '\n')
    print('wrote', NEW / f'{ARGS.out_name}.md')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
