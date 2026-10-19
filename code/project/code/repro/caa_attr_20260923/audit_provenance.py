#!/usr/bin/env python3
"""Audit A: provenance / hash / architecture correspondence (zero training).

Checks, on the server:
1. SHA256 of every code file involved in this round (CAA) and in the final G3
   runs, plus run-time file mtimes vs run start times;
2. the hashes recorded by the G3 runs in results.json::meta.hashes (and which
   copy of train_gnn_ablate2.py was actually imported);
3. architecture fingerprints: CausalCharGNN (train_gnn_v2) vs ModalityNormGNN
   (train_gnn_ablate2), and the state_dict structure of the saved CAA
   checkpoints (pool_snap.pt / pool_stage0.pt / replay.pt);
4. config comparison (windows, params, G3 flags, CL budget, selection, stream).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

BASE = Path(os.environ.get('GRAD_BASE', '.'))
sys.path.insert(0, str(BASE / 'code'))
sys.path.insert(0, str(BASE / 'code/repro'))
sys.path.insert(0, str(BASE / 'code/repro/final_five_tables'))

import torch  # noqa: E402

CAA_DIR = BASE / 'LOGS/real_cl_caa_attribution_20260923'
V2_STREAM = BASE / 'LOGS/real_cl_family_confirm_20260921/stream'
G3_DIR = BASE / 'LOGS/real_cl_g3_20260921'


def file_hashes(p):
    p = Path(p)
    if not p.exists():
        return None
    b = p.read_bytes()
    return {'sha256_16': hashlib.sha256(b).hexdigest()[:16],
            'sha256': hashlib.sha256(b).hexdigest(),
            'bytes': len(b),
            'mtime': p.stat().st_mtime}


def sd_digest(sd):
    if hasattr(sd, 'state_dict'):
        sd = sd.state_dict()
    h = hashlib.sha256()
    n_params = 0
    if not isinstance(sd, dict):
        sd = {k: v for k, v in sd}
    for k in sorted(sd):
        v = sd[k]
        if isinstance(v, torch.Tensor):
            h.update(f'{k}:{tuple(v.shape)}:{v.dtype}'.encode())
            n_params += v.numel()
        else:
            h.update(f'{k}:non-tensor'.encode())
    return h.hexdigest()[:16], n_params


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    rep = {'code_files': {}, 'g3_recorded_hashes': {}, 'architectures': {},
           'checkpoints': {}, 'configs': {}, 'streams': {}}
    files = [
        'code/repro/real_cl_unified512.py', 'code/repro/real_cl_p1.py',
        'code/repro/real_cl_p15.py', 'code/repro/real_cl_s2.py',
        'code/train_gnn_v2.py', 'code/repro/real_cl_g3.py',
        'code/repro/train_gnn_ablate2.py',
        'code/repro/final_five_tables/train_gnn_ablate2.py',
        'code/train_gnn_ablate.py', 'code/permission_spec.py',
        'code/repro/tests/test_real_cl_g3_contracts.py',
    ]
    for f in files:
        rep['code_files'][f] = file_hashes(BASE / f)
    # G3 recorded hashes (constant across runs?)
    rec = {}
    for rd in sorted((BASE / 'code/repro/out/real_cl/g3_formal').glob('*')):
        fp = rd / 'results.json'
        if not fp.exists():
            continue
        m = json.loads(fp.read_text()).get('meta', {})
        rec[rd.name] = m.get('hashes')
    rep['g3_recorded_hashes'] = {
        'distinct': sorted({json.dumps(v, sort_keys=True)
                            for v in rec.values() if v})}
    # architectures
    import train_gnn_v2 as g2
    import real_cl_p1 as p1
    first = json.loads(open(CAA_DIR / 'stream_matched/train.jsonl').readline())
    d0 = p1.graph_to_v2_data(first)
    dims = (d0.x_base.size(1), max(d0.edge_attr_base.size(1), 1))
    causal = g2.CausalCharGNN(node_base_dim=dims[0], edge_base_dim=dims[1])
    rep['architectures']['CausalCharGNN'] = {
        'module': 'train_gnn_v2.CausalCharGNN',
        'params': sum(p.numel() for p in causal.parameters()),
        'state_digest': sd_digest(causal.state_dict())[0],
        'in_dims': dims}
    import train_gnn_ablate2 as g3t
    in_dim = len(g3t.PERMISSION) + 1 + len(g3t.CONTENT_KEYWORDS)
    edge_dim = len(g3t.V2_EDGE_TYPES) + 1
    modnorm = g3t.ModalityNormGNN(in_dim, edge_dim, 192, use_ln=True,
                                  use_aux=True)
    rep['architectures']['ModalityNormGNN'] = {
        'module': 'train_gnn_ablate2.ModalityNormGNN',
        'params': sum(p.numel() for p in modnorm.parameters()),
        'state_digest': sd_digest(modnorm.state_dict())[0],
        'in_dims': [in_dim, edge_dim]}
    # checkpoints from this round
    for rd in sorted((BASE / 'code/repro/out/real_cl/caa_formal').glob('*')):
        for fname in ('pool_snap.pt', 'pool_stage0.pt', 'replay.pt',
                      'replay_matched.pt', 'naive.pt'):
            fp = rd / fname
            if not fp.exists():
                continue
            obj = torch.load(fp, map_location='cpu', weights_only=False)
            entry = {'file_hash': file_hashes(fp)['sha256_16'],
                     'bytes': fp.stat().st_size}
            if isinstance(obj, dict) and 'snap_models' in obj:
                digests = {t: sd_digest(sd) for t, sd in
                           obj['snap_models'].items()}
                entry['snap_models'] = {
                    t: {'digest': d[0], 'params': d[1]}
                    for t, d in digests.items()}
                entry['adapters'] = {t: sd_digest(sd)[1] for t, sd in
                                     obj['adapters'].items()}
                entry['heads'] = {t: sd_digest(sd)[1] for t, sd in
                                  obj['heads'].items()}
            elif isinstance(obj, dict) and 'adapters' in obj:
                entry['backbone_digest'] = sd_digest(obj['backbone'])[0]
                entry['backbone_params'] = sd_digest(obj['backbone'])[1]
                entry['adapters'] = {t: sd_digest(sd)[1] for t, sd in
                                     obj['adapters'].items()}
                entry['heads'] = {t: sd_digest(sd)[1] for t, sd in
                                  obj['heads'].items()}
            else:
                entry['digest'] = sd_digest(obj)[0]
                entry['params'] = sd_digest(obj)[1]
            rep['checkpoints'].setdefault(fname, {})[rd.name] = entry
    # configs
    caa_meta = json.loads(
        (BASE / 'code/repro/out/real_cl/caa_formal/fwd_211/results.json')
        .read_text())['meta']
    g3_meta = json.loads(
        (BASE / 'code/repro/out/real_cl/g3_formal/fam_forward_211/results.json')
        .read_text())['meta']
    rep['configs']['caa_this_round'] = {
        'backbone': 'CausalCharGNN (train_gnn_v2)',
        'window': caa_meta['window'], 'updates': caa_meta['updates'],
        'buffer': caa_meta['buffer'], 'stream': caa_meta['stream'],
        'hashes_in_meta': 'hashes' in caa_meta}
    rep['configs']['g3_final'] = g3_meta
    # streams
    for name, p in (('v2_family_stream', V2_STREAM / 'manifest.json'),
                    ('v3_matched_stream',
                     CAA_DIR / 'stream_matched/manifest.json')):
        fh = file_hashes(p)
        j = json.loads(p.read_text())
        rep['streams'][name] = {
            'file_hash': fh, 'n_rows': j.get('n_rows'),
            'n_rows_matched': j.get('n_rows_matched'),
            'retention': j.get('retention_samples'),
            'split_fractions': j.get('split_fractions', [0.7, 0.15, 0.15]),
            'gates': {k: v for k, v in j.get('gates', {}).items()
                      if 'gate' in k or 'balacc' in k or 'smd' in k}}
    (outdir / 'audit_provenance.json').write_text(json.dumps(rep, indent=1))
    L = ['# Provenance audit: this round vs final G3 (zero training)', '']
    L += ['## Code hashes (sha256[:16], as of audit)', '',
          '| file | sha256_16 | bytes | mtime |', '|---|---|---|---|']
    for f, v in rep['code_files'].items():
        L.append(f"| `{f}` | {v['sha256_16'] if v else '-'} | "
                 f"{v['bytes'] if v else '-'} | "
                 f"{v['mtime'] if v else '-'} |")
    L += ['', '## G3 recorded hashes in results.json::meta.hashes', '',
          f"distinct records: {len(rep['g3_recorded_hashes']['distinct'])}"]
    for d in rep['g3_recorded_hashes']['distinct']:
        L.append(f'- `{d}`')
    L += ['', '## Architectures', '',
          '| model | module | params | state digest |', '|---|---|---|---|']
    for k, v in rep['architectures'].items():
        L.append(f"| {k} | {v['module']} | {v['params']} | "
                 f"{v['state_digest']} |")
    L += ['', '## Saved checkpoints (this round)', '']
    for fname, runs in rep['checkpoints'].items():
        ex = next(iter(runs.values()))
        L.append(f"- `{fname}`: {json.dumps(ex)[:300]}")
    L += ['', '## Configs', '',
          json.dumps(rep['configs'], indent=1)[:1500]]
    L += ['', '## Streams', '',
          json.dumps(rep['streams'], indent=1)[:1200]]
    (outdir / 'audit_provenance.md').write_text('\n'.join(L) + '\n')
    print(json.dumps(rep, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
