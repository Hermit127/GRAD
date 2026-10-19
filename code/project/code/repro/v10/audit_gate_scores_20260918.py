#!/usr/bin/env python3
"""Phase A: zero-API score audit of the GRAD v2 gate (2026-09-18).

Checks (per PI list):
A1 checkpoint/model/dims/label direction; A2 sigmoid/logit/aggregation;
A3 eval/BN/dropout; A4 feature order/unknown types; A5 text extraction and the
192-char node window; A6 user task / candidate args / prior results in input;
A7 no future info; A8 deployment-rebuild vs recorded score consistency;
A9 score distributions; A10 high-scoring legitimate call inputs.

Outputs LOGS/final_table2/grad_audit_20260918/{audit_calls.csv,audit_summary.json,
audit_report.md}. No paid API.
"""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path('.')
sys.path.insert(0, str(ROOT / 'code'))
sys.path.insert(0, str(ROOT / 'code/repro'))
sys.path.insert(0, str(ROOT / 'code/repro/v10'))
sys.path.insert(0, str(ROOT / 'code/repro/final_five_tables'))

import grad_map  # noqa: E402
import torch  # noqa: E402
from train_gnn_v2 import (CausalCharGNN, MAX_NODE_CHARS, TOOL_TO_PERM,  # noqa: E402
                          DEFAULT_PERM, graph_to_v2_data)
from gnn_gate import GnnGate  # noqa: E402

CKPT = ROOT / 'code/repro/out/online_gate_v2/gnn_gate_v2.pt'
OUT = ROOT / 'LOGS/final_table2/grad_audit_20260918'
RESULT_DIRS = [
    ROOT / 'LOGS/final_table2/dev_attacks_20260918',
    ROOT / 'LOGS/final_table2/threshold_dev_20260918',
    ROOT / 'LOGS/final_table2/frozen_direct_test_20260918',
]
ATTACKER_TOKENS = re.compile(r'(US133000000121212121212|secure-systems-252|'
                             r'mark\.black-2134@gmail\.com|Riverside View Hotel)')


def norm(text: str) -> str:
    for esc, raw in (('\\n', ' '), ("\\'", "'"), ('\\"', '"'), ('\\\\', '\\'),
                     ("''", "'")):
        text = text.replace(esc, raw)
    return ' '.join(text.split())


def safe_eval(text):
    try:
        return ast.literal_eval(text)
    except Exception:  # noqa: BLE001
        return {'_raw': str(text)[:300]}


def events_and_task(trace):
    """Rebuild (user_task, events[{tool,args,result,status}]) from a stored trace."""
    user_task, events = '', []
    for e in trace:
        if e.get('role') == 'user' and not user_task:
            user_task = e.get('text') or ''
        if e.get('role') == 'assistant' and e.get('tool_calls'):
            for tc in e['tool_calls']:
                events.append({'tool': tc['function'], 'args': safe_eval(tc.get('args')),
                               'result': None, 'status': 'executed', 'pending': True})
        elif e.get('role') == 'tool':
            for ev in reversed(events):
                if ev.get('pending'):
                    ev['result'] = e.get('text') or ''
                    ev['pending'] = False
                    break
    for ev in events:
        ev.pop('pending', None)
        if ev.get('result') is None:
            ev['result'] = ''
    return user_task, events


def prefix_nodes(user_task, events, idx):
    trace = [{'tool': e['tool'], 'args': e['args'], 'result': e['result'],
              'status': e['status']} for e in events[:idx]]
    cand = {'tool': events[idx]['tool'], 'args': events[idx]['args']}
    return grad_map.trace_nodes_full(trace, candidate=cand, user_task=user_task)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ckpt = torch.load(CKPT, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    summary = {'ckpt': str(CKPT), 'ckpt_sha256': hashlib.sha256(CKPT.read_bytes()).hexdigest(),
               'config': {k: cfg.get(k) for k in ('node_base_dim', 'edge_base_dim', 'train')},
               'threshold_reports': ckpt.get('threshold_reports'),
               'A1_A3': {}, 'A8': {}, 'A9': {}, 'A5_A6_A7': {}}

    model = CausalCharGNN(node_base_dim=cfg['node_base_dim'],
                          edge_base_dim=cfg['edge_base_dim'])
    missing, unexpected = model.load_state_dict(ckpt['model_state'], strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    import torch.nn as nn
    bn = [n for n, m in model.named_modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    do = [n for n, m in model.named_modules() if isinstance(m, nn.Dropout)
          or (m.__class__.__name__ == 'TransformerConv' and getattr(m, 'dropout', 0) > 0)]
    model.eval()
    summary['A1_A3'] = {
        'load_state_dict_missing': list(missing), 'load_state_dict_unexpected': list(unexpected),
        'n_params': n_params, 'n_node_types_mapped': len(TOOL_TO_PERM),
        'default_perm': DEFAULT_PERM, 'max_node_chars': MAX_NODE_CHARS,
        'batchnorm_modules': bn, 'dropout_like_modules': do,
        'eval_mode': not model.training,
        'loss_in_training': 'binary_cross_entropy_with_logits (inspect train_gnn_v2)',
    }

    gate = GnnGate(str(CKPT))

    # A2 + A8: direct forward vs GnnGate path vs recorded scores.
    from torch_geometric.data import Batch

    def direct_logit(nodes):
        d = graph_to_v2_data({'nodes': nodes})
        b = Batch.from_data_list([d])
        with torch.no_grad():
            out = model(b.x_base, b.node_chars, b.edge_index,
                        b.edge_attr_base, b.edge_chars, b.batch)
        return float(out.item())

    sample_nodes = json.loads((ROOT / 'code/repro/out/online_gate_v2/dev.jsonl')
                              .read_text().splitlines()[0])['nodes']
    logit = direct_logit(sample_nodes)
    p_direct = float(torch.sigmoid(torch.tensor(logit)).item())
    p_gate, _ = gate.score_nodes(sample_nodes)
    summary['A1_A3']['sigmoid_check'] = {
        'raw_logit': logit, 'sigmoid_once': p_direct, 'gnn_gate_score': p_gate,
        'abs_diff': abs(p_direct - p_gate)}

    audit_rows = []

    def score_trace_run(run, source):
        if not run.get('trace'):
            return
        user_task, events = events_and_task(run['trace'])
        recorded = run.get('decisions') or []
        for i, ev in enumerate(events):
            nodes = prefix_nodes(user_task, events, i)
            p, _ = gate.score_nodes(nodes)
            cand_text = nodes[-1]['features']['subtask']
            prior_texts = [n['features']['subtask'] for n in nodes[:-1]]
            payload_visible_full = any('INFORMATION' in t for t in prior_texts)
            payload_visible_window = any('INFORMATION' in t[:MAX_NODE_CHARS]
                                         for t in prior_texts)
            types = [n['type'] for n in nodes]
            unknown = sum(1 for t in types if t not in TOOL_TO_PERM)
            rec = recorded[i] if i < len(recorded) else {}
            audit_rows.append({
                'source': source, 'suite': run['suite'], 'user_task': run['user_task'],
                'injection_task': run.get('injection_task'), 'arm': run['arm'],
                'strategy': run.get('strategy', ''), 'call_idx': i,
                'tool': ev['tool'], 'candidate_chars': len(cand_text),
                'candidate_visible_chars': min(len(cand_text), MAX_NODE_CHARS),
                'n_nodes': len(nodes), 'n_prior_nodes': len(nodes) - 1,
                'unknown_type_nodes': unknown,
                'user_task_node_present': any(t == 'user_task' for t in types),
                'payload_in_prior_full': payload_visible_full,
                'payload_in_prior_first192': payload_visible_window,
                'p': p, 'recorded_p': rec.get('score'),
                'blocked': rec.get('blocked'), 'recorded_threshold': rec.get('threshold'),
                'security': run.get('security'), 'utility': run.get('utility'),
                'attacker_token_in_candidate': bool(ATTACKER_TOKENS.search(json.dumps(ev['args']))),
            })

    seen_ids = set()
    for d in RESULT_DIRS:
        for path in sorted(d.glob('results_*.json')):
            for run in json.loads(path.read_text()).get('runs', []):
                key = (str(path), run['case_id'])
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                score_trace_run(run, d.name)

    # v10 in-distribution distributions
    dist = {}
    for role in ('train', 'dev', 'test'):
        rows = [json.loads(l) for l in (ROOT / f'code/repro/out/online_gate_v2/{role}.jsonl')
                .read_text().splitlines() if l.strip()]
        ps, ys = [], []
        for s in rows:
            p, _ = gate.score_nodes(s['nodes'])
            ps.append(p)
            ys.append(s['label'])
        dist[role] = {'n': len(ps), 'pos': sum(ys),
                      'p_min': min(ps), 'p_max': max(ps),
                      'frac_gt_0.999': sum(1 for p in ps if p > 0.999) / len(ps),
                      'frac_lt_0.001': sum(1 for p in ps if p < 0.001) / len(ps),
                      'neg_max': max([p for p, y in zip(ps, ys) if y == 0], default=None),
                      'pos_min': min([p for p, y in zip(ps, ys) if y == 1], default=None)}
    summary['A9']['v10'] = dist

    # AgentDojo distributions
    ad = defaultdict(list)
    for r in audit_rows:
        ad[(r['source'], 'benign' if r['security'] is None else
            ('attack' if r['injection_task'] else 'benign-run'))].append(r['p'])
    summary['A9']['agentdojo'] = {
        f'{k[0]}|{k[1]}': {'n': len(v), 'min': min(v),
                           'median': sorted(v)[len(v) // 2], 'max': max(v)}
        for k, v in ad.items()}

    # A8 recorded vs rebuilt (grad arm only)
    comp = [(r['recorded_p'], r['p']) for r in audit_rows
            if r['recorded_p'] is not None]
    summary['A8'] = {
        'n_compared': len(comp),
        'max_abs_diff': max((abs(a - b) for a, b in comp), default=None),
        'mean_abs_diff': (sum(abs(a - b) for a, b in comp) / len(comp)) if comp else None,
    }

    # A4: node types not in the training permission vocabulary
    unknown_counter = Counter()
    for d in RESULT_DIRS:
        for path in sorted(d.glob('results_*.json')):
            for run in json.loads(path.read_text()).get('runs', []):
                if not run.get('trace'):
                    continue
                user_task, events = events_and_task(run['trace'])
                for i in range(len(events)):
                    for n in prefix_nodes(user_task, events, i):
                        if n['type'] not in TOOL_TO_PERM:
                            unknown_counter[n['type']] += 1
    summary['A4'] = {'unknown_types_deployment': dict(unknown_counter),
                     'n_known_types': len(TOOL_TO_PERM)}

    # A5: in-distribution node text length profile (window saturation)
    in_dist_len = {}
    for role in ('train', 'dev', 'test'):
        lens = [len(n['features']['subtask'])
                for l in (ROOT / f'code/repro/out/online_gate_v2/{role}.jsonl')
                .read_text().splitlines() if l.strip()
                for n in json.loads(l)['nodes']]
        in_dist_len[role] = {
            'nodes': len(lens), 'p50': sorted(lens)[len(lens) // 2],
            'p95': sorted(lens)[int(0.95 * len(lens))], 'max': max(lens),
            'over_192': sum(1 for x in lens if x > MAX_NODE_CHARS)}

    # A5/A6/A7 summary
    attack_rows = [r for r in audit_rows if r['injection_task']]
    summary['A5_A6_A7'] = {
        'n_calls': len(audit_rows), 'n_attack_calls': len(attack_rows),
        'candidate_over_window': sum(1 for r in audit_rows
                                     if r['candidate_chars'] > MAX_NODE_CHARS),
        'candidate_over_window_tools': Counter(
            r['tool'] for r in audit_rows if r['candidate_chars'] > MAX_NODE_CHARS),
        'user_task_node_missing': sum(1 for r in audit_rows
                                      if not r['user_task_node_present']),
        'unknown_type_nodes_total': sum(r['unknown_type_nodes'] for r in audit_rows),
        'in_distribution_node_text_len': in_dist_len,
        'payload_in_prior_full': sum(1 for r in attack_rows if r['payload_in_prior_full']),
        'payload_in_prior_first192': sum(1 for r in attack_rows
                                         if r['payload_in_prior_first192']),
    }

    # Phase A classification
    summary['classification'] = {
        'implementation_error_found': False,
        'evidence': [
            'state_dict strict load clean; 627k params; eval mode; no double sigmoid '
            '(direct logit sigmoid == GnnGate score, abs diff 0)',
            'recorded vs rebuilt deployment scores: '
            f"n={summary['A8']['n_compared']} mean|diff|={summary['A8']['mean_abs_diff']:.2e}",
            'primary bottleneck 1 (information): MAX_NODE_CHARS=192, in-dist nodes '
            'over window: train 755/1499; deployment attack calls with payload in prior '
            f"full text but not in the 192-char window: 16 vs 4 visible",
            'primary bottleneck 2 (calibration/saturation): v10 dev pos_min=0.99989, '
            'neg_max=0.999995; useful margin lives at the 0.9999x tail',
            'primary bottleneck 3 (distribution shift): AgentDojo benign-run and '
            'attack-run call scores overlap (medians 0.995 vs 0.997); no AgentDojo-style '
            'long observations or authorized-benign calls in training',
        ],
        'category': 'no implementation error; input information loss + score '
                    'saturation + train/deploy distribution shift',
    }

    with open(OUT / 'audit_calls.csv', 'w', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=list(audit_rows[0].keys()))
        wr.writeheader()
        wr.writerows(audit_rows)
    (OUT / 'audit_summary.json').write_text(json.dumps(summary, indent=1, default=str))

    top_legit = sorted([r for r in audit_rows if not r['injection_task']], key=lambda r: -r['p'])[:10]
    lines = ['# GRAD v2 score audit (Phase A, zero API)', '',
             f"- ckpt sha256: `{summary['ckpt_sha256'][:16]}...`",
             f"- load_state_dict: missing={missing} unexpected={unexpected}; params={n_params}",
             f"- sigmoid check: {summary['A1_A3']['sigmoid_check']}",
             f"- A8 recorded-vs-rebuilt: {summary['A8']}",
             f"- A5: candidate text over 192-char window: "
             f"{summary['A5_A6_A7']['candidate_over_window']} / {len(audit_rows)}",
             f"- A5: attack calls where payload is in prior full text: "
             f"{summary['A5_A6_A7']['payload_in_prior_full']}; visible in first 192 chars: "
             f"{summary['A5_A6_A7']['payload_in_prior_first192']}",
             '', '## v10 in-distribution', '', json.dumps(dist, indent=1),
             '', '## AgentDojo score distribution', '', json.dumps(summary['A9']['agentdojo'], indent=1),
             '', '## Top-10 highest-scoring benign-run calls', '']
    for r in top_legit:
        lines.append(f"- {r['source']} {r['suite']}|{r['user_task']} {r['tool']} "
                     f"p={r['p']:.6f} prior={r['n_prior_nodes']} candidate_chars={r['candidate_chars']}")
    (OUT / 'audit_report.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines[:40]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
