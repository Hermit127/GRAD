#!/usr/bin/env python3
"""Per-fold combined MCP training sets for the legacy 5-fold protocol.

combined_train_f = legacy train_f
                 + {rebuilt/dgc graphs whose task_id is NOT in
                    train_f U dev_f U test_f task ids}
(no test/dev task contamination; legacy-train tasks are not re-added, avoiding
duplicate conversions of the same trace).
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
FIX = REPO / 'code/repro/out/unified_gnn/fix2_splits/mcp'
AUG = REPO / 'ref/dataset/mcp_continual_rebuild_v1/semantic_aug/train_plus_dgc.jsonl'
OUT = REPO / 'code/repro/out/unified_gnn/mcp_combined_splits'


def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    aug = load(AUG)
    OUT.mkdir(parents=True, exist_ok=True)
    for f in range(5):
        tr = load(FIX / f'fold{f}/train.jsonl')
        va = load(FIX / f'fold{f}/dev.jsonl')
        te = load(FIX / f'fold{f}/test.jsonl')
        seen = {g['task_id'] for g in tr} | {g['task_id'] for g in va} | {g['task_id'] for g in te}
        add = [g for g in aug if g['task_id'] not in seen]
        d = OUT / f'fold{f}'
        d.mkdir(parents=True, exist_ok=True)
        with open(d / 'train.jsonl', 'w') as fh:
            for g in tr + add:
                fh.write(json.dumps(g, ensure_ascii=False) + '\n')
        print(f'fold{f}: legacy train {len(tr)} + added {len(add)} = {len(tr)+len(add)}; '
              f'dev {len(va)} test {len(te)}')


if __name__ == '__main__':
    main()
