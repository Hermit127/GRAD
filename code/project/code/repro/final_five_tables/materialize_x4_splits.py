#!/usr/bin/env python3
"""Family-grouped train/dev/test splits for the x4 corpora (InjecAgent/ASB).

Mirrors the archived x4 test rule (`x4_test`: last 20% of families as test) and
adds a family-disjoint dev split (next 20% of the remaining families) for
threshold calibration; the rest is train. Zero API.
"""
from __future__ import annotations
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / 'code/repro/x4_datasets/data'
OUT = REPO / 'code/repro/out/unified_gnn/fix2_splits_x4'


def main():
    manifest = {}
    for name in ('injecagent', 'asb', 'agentharm'):
        graphs = [json.loads(l) for l in open(SRC / f'{name}.jsonl') if l.strip()]
        fams = sorted({g['family'] for g in graphs})
        n_test = max(1, round(len(fams) * 0.2))
        test_f = fams[-n_test:]
        rest = fams[:-n_test]
        n_dev = max(1, round(len(rest) * 0.2))
        dev_f = rest[-n_dev:]
        train_f = rest[:-n_dev]
        splits = {
            'train': [g for g in graphs if g['family'] in set(train_f)],
            'dev': [g for g in graphs if g['family'] in set(dev_f)],
            'test': [g for g in graphs if g['family'] in set(test_f)],
        }
        d = OUT / name
        d.mkdir(parents=True, exist_ok=True)
        counts = {}
        for split, rows in splits.items():
            with open(d / f'{split}.jsonl', 'w') as fh:
                for r in rows:
                    fh.write(json.dumps(r, ensure_ascii=False) + '\n')
            counts[split] = {'n': len(rows),
                             'n_pos': sum(int(r['label']) for r in rows),
                             'families': len({r['family'] for r in rows})}
        manifest[name] = {'counts': counts, 'test_families': test_f,
                          'dev_families': dev_f}
        print(name, json.dumps(counts))
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1))


if __name__ == '__main__':
    main()
