#!/usr/bin/env python3
"""Contract tests for the G3-512 CAA runner (pre-run verification checklist).

Run:
  GRAD_BASE=. \
  PYTHONPATH=code:code/repro:code/repro/final_five_tables \
  python code/repro/tests/test_real_cl_g3_contracts.py
"""
import copy
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

BASE = Path('.')
sys.path.insert(0, str(BASE / 'code'))
sys.path.insert(0, str(BASE / 'code/repro'))
sys.path.insert(0, str(BASE / 'code/repro/final_five_tables'))

import real_cl_g3 as g3r  # noqa: E402
import train_gnn_ablate2 as g3t  # noqa: E402

PASS = []


def check(name, cond, detail=''):
    PASS.append((name, bool(cond)))
    print(f'{"PASS" if cond else "FAIL"} {name} {detail}')


def state_hash(m):
    h = hashlib.sha1()
    for k, v in sorted(m.state_dict().items()):
        if isinstance(v, torch.Tensor):
            h.update(k.encode())
            h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def synth_data(n=4, in_dim=22, edge_dim=11):
    rng = np.random.RandomState(0)
    data = {}
    for i in range(n):
        nc = torch.tensor(np.random.randint(1, 100, size=(3, 512)),
                          dtype=torch.long)
        xb = torch.tensor(rng.rand(3, in_dim), dtype=torch.float32)
        ei = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
        ea = torch.tensor(rng.rand(2, edge_dim), dtype=torch.float32)
        ec = torch.tensor(np.random.randint(1, 100, size=(2, 160)),
                          dtype=torch.long)
        d = Data(x=torch.empty(3, 0), x_base=xb, node_chars=nc,
                 edge_index=ei, edge_attr_base=ea, edge_chars=ec,
                 y=torch.tensor([float(i % 2)]), num_nodes=3)
        data[f'g{i}'] = d
    return data


def main():
    device = torch.device('cpu')
    g3t.MAX_NODE_CHARS = 512
    g3t.MAX_EDGE_CHARS = 160
    # T01 config flags
    check('G3-T01a candidate A_moddrop', g3r.CFG['candidate'] == 'A_moddrop')
    check('G3-T01b moddrop 0.25', g3r.CFG['moddrop'] == 0.25)
    check('G3-T01c lambda_aux 0.1', g3r.CFG['lambda_aux'] == 0.1)
    model, in_dim, edge_dim = g3r.make_g3(device, 512, 160)
    check('G3-T01d LayerNorm enabled',
          isinstance(model.ln_base, nn.LayerNorm) and
          isinstance(model.ln_char, nn.LayerNorm))
    check('G3-T01e auxiliary heads enabled',
          hasattr(model, 'aux_text_head') and hasattr(model, 'aux_struct_head'))
    check('G3-T01f windows applied',
          g3t.MAX_NODE_CHARS == 512 and g3t.MAX_EDGE_CHARS == 160)

    # T02 modality dropout: stochastic in training, deterministic in eval
    dev = torch.device('cpu')
    model2, _, _ = g3r.make_g3(dev, 512, 160)
    for mod in model2.modules():
        if isinstance(mod, nn.Dropout):
            mod.p = 0.0
    d = synth_data(1)['g0']
    b = Batch.from_data_list([d])
    model2.train()
    with torch.no_grad():
        o1 = model2(b, moddrop_p=0.25, training=True)['logit']
        o2 = model2(b, moddrop_p=0.25, training=True)['logit']
    model2.eval()
    with torch.no_grad():
        e1 = model2(b, moddrop_p=0.25, training=False)['logit']
        e2 = model2(b, moddrop_p=0.25, training=False)['logit']
    check('G3-T02a moddrop stochastic in training',
          not torch.allclose(o1, o2))
    check('G3-T02b moddrop off at inference', torch.allclose(e1, e2))

    # T03 aux loss present in the backbone phase
    y = torch.tensor([1.0])
    out = g3t.batch_logits(model2, b, g3r.CFG, training=True)
    loss, parts = g3t.compute_loss(out, y, g3r.CFG,
                                   sample_weight=None, pos_weight=1.0)
    check('G3-T03a aux terms reported',
          'text' in parts and 'struct' in parts and
          abs(parts['text'] + parts['struct']) > 0)

    # T04 expert phase freezes backbone params AND BN stats
    data = synth_data(4)
    tr = [{'task_id': f'g{i}', 'label': i % 2} for i in range(4)]
    model3, _, _ = g3r.make_g3(dev, 512, 160)
    before = state_hash(model3)
    bn_before = {k: v.clone() for k, v in model3.state_dict().items()
                 if 'running_' in k}
    ad = g3r.p1.Adapter(model3.node_proj.out_features, r=8).to(dev)
    hd = nn.Linear(model3.node_proj.out_features, 1).to(dev)
    old = model3.classifier
    model3.classifier = hd
    g3r.train_select_g3(model3, data, tr, tr, dev, 1, 2, 4, adapter=ad,
                        frozen_backbone=True)
    model3.classifier = old
    after = state_hash(model3)
    bn_same = all(torch.equal(bn_before[k], model3.state_dict()[k])
                  for k in bn_before)
    ad_changed = any(not torch.equal(
        ad.state_dict()[k], torch.zeros_like(ad.state_dict()[k]))
        for k in ad.state_dict())
    check('G3-T04a backbone params frozen in expert phase', before == after)
    check('G3-T04b BN running stats frozen in expert phase', bn_same)
    check('G3-T04c adapter received updates', ad_changed)

    # T05 snapshot save/load round trip
    m_snap = copy.deepcopy(model3)
    sd = m_snap.state_dict()
    m2, _, _ = g3r.make_g3(dev, 512, 160)
    m2.load_state_dict(sd)
    m_snap.eval(); m2.eval()
    with torch.no_grad():
        a = m_snap(b, moddrop_p=0.0, training=False)['logit']
        c = m2(b, moddrop_p=0.0, training=False)['logit']
    check('G3-T05a snapshot round-trip identical', torch.allclose(a, c))

    # T06 replay buffer quota/sample
    buf = g3r.p1.ReplayBuffer(10)
    graphs = [{'task_id': f'r{i}', 'label': i % 2} for i in range(30)]
    scores = {g['task_id']: 0.5 for g in graphs}
    buf.add_task(graphs, scores, 1)
    buf.add_task(graphs, scores, 2)
    pool = buf.sample(4, __import__('random').Random(0))
    check('G3-T06a replay sample size', len(pool) == 4)
    check('G3-T06b replay quota respected',
          all(len(s) <= 10 for s in buf.stores))

    # T07 router on two synthetic signatures
    router = g3r.p1.Router()
    a1 = {'task_id': 'a', 'label': 0, 'nodes': [{'type': 'memory_read'}] * 5,
          'edges': []}
    a2 = {'task_id': 'b', 'label': 0, 'nodes': [{'type': 'web_request'}] * 5,
          'edges': []}
    router.add_task('t1', [a1])
    router.add_task('t2', [a2])
    check('G3-T07a routing hits own centroid',
          router.route(a1) == 't1' and router.route(a2) == 't2')

    # T08 data builder windows
    probe = {'task_id': 'p', 'label': 0,
             'nodes': [{'id': 'n0', 'type': 'agent_message', 'turn': 0,
                        'features': {'subtask': 'x' * 900}},
                       {'id': 'n1', 'type': 'web_request', 'turn': 1,
                        'features': {'subtask': 'y'}}],
             'edges': [{'source': 'n0', 'target': 'n1', 'type': 'data_flow',
                        'rationale': 'z' * 300}]}
    dd = g3t.graph_to_data(probe, g3r.CFG, None)
    check('G3-T08a node window 512', dd.node_chars.shape[1] == 512)
    check('G3-T08b edge window 160', dd.edge_chars.shape[1] == 160)

    print()
    fails = [n for n, ok in PASS if not ok]
    print(f'{len(PASS) - len(fails)}/{len(PASS)} passed')
    if fails:
        print('FAILED:', fails)
        raise SystemExit(1)


if __name__ == '__main__':
    raise SystemExit(main())
