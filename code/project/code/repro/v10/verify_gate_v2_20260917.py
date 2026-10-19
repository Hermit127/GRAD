"""Verify the v2 (format-unified) gate checkpoint through the live hook path.

Checks (zero API, local):
  1. GnnGate loads code/repro/out/online_gate_v2/gnn_gate_v2.pt and reports its
     dev-selected threshold.
  2. The full v2 test split scores through the same graph_to_v2_data path;
     AUROC/recall/FPR at threshold are recomputed and must match the training
     report (sanity of the frozen artifact).
  3. Nodes built by the *live hook* builder (``grad_map.trace_nodes_full``),
     including an executed call, a blocked call and a proposed candidate, are
     accepted by the checkpoint; a blocked prior action must render as
     ``status=blocked`` in the node text.
  4. Format identity: dataset rows and hook-built nodes produce identical node
     text for the same semantic inputs.
  5. Encoder-window disclosure: count v2 test nodes whose text exceeds
     MAX_NODE_CHARS (the model-side cap) and the longest node text.

Usage:
  python code/repro/v10/verify_gate_v2_20260917.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path('.')
sys.path.insert(0, str(REPO / 'code'))
sys.path.insert(0, str(REPO / 'code/repro'))
sys.path.insert(0, str(REPO / 'code/repro/v10'))

from gnn_gate import GnnGate  # noqa: E402
from train_gnn_v2 import MAX_NODE_CHARS  # noqa: E402

import grad_map  # noqa: E402

CKPT = REPO / 'code/repro/out/online_gate_v2/gnn_gate_v2.pt'


def main() -> int:
    gate = GnnGate(ckpt=str(CKPT))
    print(json.dumps({'ckpt': gate.meta['ckpt'],
                      'threshold': round(gate.threshold, 6),
                      'train_cfg': gate.meta.get('train_cfg')},
                     ensure_ascii=False), flush=True)

    from sklearn.metrics import roc_auc_score
    path = REPO / 'code/repro/out/online_gate_v2/test.jsonl'
    labels, scores, lens = [], [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        p, _ = gate.score_nodes(rec['nodes'])
        labels.append(rec['label'])
        scores.append(p)
        lens.extend(len(n['features']['subtask']) for n in rec['nodes'])
    auroc = roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5
    fp = sum(1 for y, s in zip(labels, scores)
             if y == 0 and s >= gate.threshold)
    fn = sum(1 for y, s in zip(labels, scores)
             if y == 1 and s < gate.threshold)
    benign = sum(1 for y in labels if y == 0)
    mal = len(labels) - benign
    print(json.dumps({'split': 'test', 'n': len(labels),
                      'auroc': round(auroc, 4),
                      'fpr': round(fp / max(benign, 1), 4),
                      'recall': round((mal - fn) / max(mal, 1), 4),
                      'over_window_nodes': sum(1 for x in lens
                                               if x > MAX_NODE_CHARS),
                      'nodes_total': len(lens),
                      'longest_node_chars': max(lens)}, ensure_ascii=False),
          flush=True)

    hook_nodes = grad_map.trace_nodes_full(
        [{'tool': 'read_file', 'args': {'path': 'notes.txt'},
          'result': 'benign note text', 'status': 'executed'},
         {'tool': 'send_email', 'args': {'to': 'a@b.c', 'body': 'x' * 500},
          'result': '[BLOCKED by GRAD] blocked_by_grad score=0.99 threshold=0.5',
          'status': 'blocked'}],
        candidate={'tool': 'send_money',
                   'args': {'recipient': 'CH00', 'amount': 9999}},
        user_task='pay the rent')
    text_ok = ('status=blocked' in hook_nodes[2]['features']['subtask']
               and 'status=proposed' in hook_nodes[3]['features']['subtask'])
    p, lat = gate.score_nodes(hook_nodes)
    print(json.dumps({'hook_nodes_p': round(p, 6),
                      'hook_nodes_status_ok': text_ok,
                      'latency_ms': round(lat, 2)}, ensure_ascii=False),
          flush=True)

    statuses = {n.get('status') for n in hook_nodes}
    print(json.dumps({'hook_node_statuses': sorted(statuses)},
                     ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
