#!/usr/bin/env python3
"""CAA attribution Phase A2: does the probe feature set equal the router's?

Zero-training, read-only. Compares, on every graph of the frozen stream:
1. the deployed router features (`real_cl_p1.signature`) against the probe /
   audit features (`real_cl_family_stream_v2.sig`);
2. recoverability of raw node counts, raw edge counts and node-type counts
   (normalised shares * node count);
3. sensitivity checks: text-only edit must not change the signature; node-type
   and edge edits must change it (i.e. the router sees exactly histogram +
   n_nodes + n_edges, and nothing textual).
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get('GRAD_BASE', '.'))
sys.path.insert(0, str(BASE / 'code'))
sys.path.insert(0, str(BASE / 'code/repro'))

import real_cl_p1 as p1  # noqa: E402

STREAM = BASE / 'LOGS/real_cl_family_confirm_20260921/stream'
PHASES = ['train', 'dev', 'test']


def v2_sig(g, node_types):
    v = [0.0] * len(node_types)
    idx = {t: i for i, t in enumerate(node_types)}
    for n in g['nodes']:
        t = str(n.get('type', 'other'))
        v[idx.get(t, idx['other'])] += 1
    n = max(len(g['nodes']), 1)
    return np.array([x / n for x in v] +
                    [len(g['nodes']) / 50.0, len(g.get('edges', [])) / 50.0])


def main() -> int:
    rows = []
    for ph in PHASES:
        for line in open(STREAM / f'{ph}.jsonl'):
            if line.strip():
                rows.append(json.loads(line))
    out = {'n_rows': len(rows), 'n_feature_dims_router': None,
           'equal_all_samples': True, 'max_abs_diff': 0.0}
    maxdiff = 0.0
    for r in rows:
        a = p1.signature(r)
        b = v2_sig(r, p1.NODE_TYPES)
        out['n_feature_dims_router'] = len(a)
        d = float(np.max(np.abs(a - b)))
        maxdiff = max(maxdiff, d)
        if d > 1e-12:
            out['equal_all_samples'] = False
            out['first_mismatch'] = r['task_id']
            break
    out['max_abs_diff'] = maxdiff
    # recoverability of raw quantities
    r0 = rows[0]
    s = p1.signature(r0)
    n_nodes_rec = s[-2] * 50.0
    n_edges_rec = s[-1] * 50.0
    out['raw_recoverable'] = {
        'n_nodes_rec': float(n_nodes_rec), 'n_nodes_true': len(r0['nodes']),
        'n_edges_rec': float(n_edges_rec), 'n_edges_true': len(r0['edges'])}
    # sensitivity: text-only change (subtask text) must not move the signature
    import copy
    g_txt = copy.deepcopy(r0)
    for n in g_txt['nodes']:
        n.setdefault('features', {})
        n['features']['subtask'] = 'totally different text ' * 3
    out['text_edit_changes_signature'] = bool(
        np.max(np.abs(p1.signature(g_txt) - s)) > 1e-12)
    # node-type change must move the signature
    g_type = copy.deepcopy(r0)
    if g_type['nodes']:
        old = g_type['nodes'][0].get('type', 'other')
        g_type['nodes'][0]['type'] = 'shell_exec' if old != 'shell_exec' \
            else 'web_search'
    out['type_edit_changes_signature'] = bool(
        np.max(np.abs(p1.signature(g_type) - s)) > 1e-12)
    # edge-count change must move the signature
    g_edge = copy.deepcopy(r0)
    g_edge.setdefault('edges', [])
    g_edge['edges'] = list(g_edge['edges']) + [{}]
    out['edge_edit_changes_signature'] = bool(
        np.max(np.abs(p1.signature(g_edge) - s)) > 1e-12)
    out['router_source'] = 'real_cl_p1.signature (used by p1.Router)'
    out['probe_source'] = 'real_cl_family_stream_v2.sig / this repo'
    print(json.dumps(out, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
