#!/usr/bin/env python3
"""Shared graph featurizer with NAMED offsets (fixes F2 audit: previous layout
offset by 2 columns -> edge_in index out of bounds at EDGE_TYPES=7).

Layout (FD = N_PERM + N_TYPE + 1 + 2 + 2*N_EDGE):
  [0, N_PERM)            perm onehot
  [N_PERM, N_PERM+N_TYPE) node-type onehot
  pos                    normalized position (1 col)
  out_deg, in_deg        (2 cols)
  edge_out[N_EDGE]       outgoing edge-type counts
  edge_in[N_EDGE]        incoming edge-type counts
"""
import sys
sys.path.insert(0, '.')
from train_gnn import PERMISSION, TOOL_TO_PERM, DEFAULT_PERM
from memorygraft_to_graph import NODE_TYPES, EDGE_TYPES
import torch

EDGE2IDX = {t: i for i, t in enumerate(EDGE_TYPES)}
TYPE2IDX = {t: i for i, t in enumerate(NODE_TYPES)}
N_PERM, N_TYPE, N_EDGE = len(PERMISSION), len(NODE_TYPES), len(EDGE_TYPES)

# named offsets
POS_O = N_PERM + N_TYPE
OUT_DEG_O = POS_O + 1
IN_DEG_O = OUT_DEG_O + 1
EDGE_OUT_O = IN_DEG_O + 1
EDGE_IN_O = EDGE_OUT_O + N_EDGE
FD = EDGE_IN_O + N_EDGE


def to_data(g, drop=(), mode='full'):
    """mode='full': perm+type+pos+deg+edge-counts (contains structural stats);
    mode='no_struct': perm+type+pos ONLY (structural statistics removed)."""
    nodes = [n for n in g['nodes'] if n['type'] not in drop]
    if not nodes:
        return None
    id2i = {n['id']: i for i, n in enumerate(nodes)}
    n = len(nodes)
    dim = FD if mode == 'full' else N_PERM + N_TYPE + 1
    x = torch.zeros(n, dim)
    for i, nd in enumerate(nodes):
        perm = TOOL_TO_PERM.get(nd['type'], DEFAULT_PERM)
        x[i, :N_PERM] = torch.tensor(perm, dtype=torch.float32)
        x[i, N_PERM + TYPE2IDX.get(nd['type'], 0)] = 1.0
        x[i, POS_O] = i / max(n, 1)
    ei, ea_flat = [], []
    for e in g.get('edges', []):
        s, t = id2i.get(e['source']), id2i.get(e['target'])
        if s is None or t is None:
            continue
        if e['type'] not in EDGE2IDX:
            raise ValueError(f"unknown edge type: {e['type']!r}")  # spec 2.3
        ei.append([s, t])
        ea_flat.append(EDGE2IDX[e['type']])
    ei = torch.tensor(ei, dtype=torch.long).t().contiguous() if ei else torch.zeros(2, 0, dtype=torch.long)
    ea = torch.zeros(ei.size(1), N_EDGE)
    for j, idx in enumerate(ea_flat):
        ea[j, idx] = 1.0
    if mode == 'full':
        for j in range(ei.size(1)):
            s, t = ei[0, j].item(), ei[1, j].item()
            idx = ea_flat[j]
            x[s, OUT_DEG_O] += 1
            x[t, IN_DEG_O] += 1
            x[s, EDGE_OUT_O + idx] += 1
            x[t, EDGE_IN_O + idx] += 1
        deg_sum = x[:, OUT_DEG_O:].sum(dim=1, keepdim=True).clamp(min=1)
        x[:, OUT_DEG_O:] /= deg_sum
    return Data(x=x, edge_index=ei, edge_attr=ea, y=torch.tensor([g.get('label', 0)], dtype=torch.float32))


from torch_geometric.data import Data  # noqa: E402
