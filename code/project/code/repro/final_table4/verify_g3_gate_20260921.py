#!/usr/bin/env python3
"""Verify the G3-512 gate checkpoint: parity with trainer dev scores + latency."""
import glob
import json
import statistics as st
import sys
from pathlib import Path

REPO = Path('.')
sys.path.insert(0, str(REPO / 'code'))
sys.path.insert(0, str(REPO / 'code/repro'))
sys.path.insert(0, str(REPO / 'code/repro/v10'))
sys.path.insert(0, str(REPO / 'code/repro/final_table4'))

from g3_gate import G3Gate  # noqa: E402

gate = G3Gate()
print('meta:', json.dumps(gate.meta))
rows = [json.loads(l) for l in open(REPO / 'code/repro/out/online_gate_v2/dev.jsonl')]
res = json.load(open(sorted(glob.glob(str(
    REPO / 'code/repro/out/online_gate_g3/phase2_A_moddrop_s211_*.json')))[-1]))
vs = res['scores']['val']
diffs, blocks = [], 0
for g, v in zip(rows, vs):
    p, _ = gate.score_nodes(g['nodes'])
    diffs.append(abs(p - float(v['score'])))
lat = sorted(gate.latencies_ms)
print('dev n', len(rows), 'parity max|dp|', max(diffs), 'mean', st.mean(diffs))
print('threshold', gate.threshold)
print('latency p50/p95 ms', lat[len(lat) // 2], lat[int(0.95 * len(lat)) - 1])
