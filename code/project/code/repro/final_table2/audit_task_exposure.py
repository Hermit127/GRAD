#!/usr/bin/env python3
"""Zero-API exposure audit: which AgentDojo (suite, user_task) were ever run.

Scans every JSON file under the given roots, pulls out any dict that carries a
suite/user-task pair (several historical schemas), and aggregates the used
tasks with the source files and the observed security outcomes. The result is
the exclusion basis for the confirmation-set task list.

Output: LOGS/final_table2/attack_dev_v2_20260919/exposure_audit.json/md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'LOGS/final_table2/attack_dev_v2_20260919'
SUITES = ('banking', 'slack', 'travel', 'workspace')


def iter_runs(obj):
    """Yield dicts that look like a run record with suite + user task."""
    if isinstance(obj, dict):
        suite = obj.get('suite')
        uid = obj.get('user_task') or obj.get('user_task_id')
        if suite in SUITES and isinstance(uid, str) and uid.startswith('user_task'):
            yield obj
        for v in obj.values():
            yield from iter_runs(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_runs(v)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--roots', nargs='+',
                    default=['LOGS/final_table2', 'LOGS/final_five_tables'])
    ap.add_argument('--out', default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    used: dict[str, dict[str, set]] = {
        s: {f'user_task_{i}': set() for i in range(41)} for s in SUITES}
    sources: dict[tuple, list] = {}
    scanned = 0
    for root in args.roots:
        for path in sorted((ROOT / root).rglob('*.json')):
            if '_audit' in path.name or 'exposure' in path.name:
                continue
            try:
                obj = json.loads(path.read_text())
            except Exception:  # noqa: BLE001
                continue
            scanned += 1
            for run in iter_runs(obj):
                key = (run['suite'], run['user_task'])
                if key[1] not in used[key[0]]:
                    continue
                src = str(path.relative_to(ROOT))
                used[key[0]][key[1]].add(src)
                sources.setdefault(key, [])
                sec = run.get('security')
                if src not in [s for s, _ in sources[key]]:
                    sources[key].append((src, sec))

    report = {'files_scanned': scanned, 'suites': {}}
    for s in SUITES:
        tasks = {}
        for uid, srcs in used[s].items():
            if srcs:
                tasks[uid] = sorted(srcs)
        report['suites'][s] = tasks

    md = ['# AgentDojo task exposure audit (zero API)', '',
          f'JSON files scanned: {scanned}', '',
          'Tasks listed below appear in at least one historical AgentDojo run.',
          'Confirmation samples must exclude tasks used for attack discovery /',
          'payload selection; other prior runs are disclosed.', '']
    for s in SUITES:
        md.append(f'## {s}')
        md.append('')
        md.append('| user_task | n_sources | earliest-ish sources |')
        md.append('|---|---|---|')
        for uid, srcs in sorted(report['suites'][s].items(),
                                key=lambda kv: int(kv[0].split('_')[-1])):
            md.append(f'| {uid} | {len(srcs)} | {"; ".join(srcs[:3])} |')
        md.append('')
    (out / 'exposure_audit.json').write_text(
        json.dumps(report, indent=1, ensure_ascii=False))
    (out / 'exposure_audit.md').write_text('\n'.join(md) + '\n')
    for s in SUITES:
        n = len(report['suites'][s])
        print(f'{s}: {n} tasks with any prior run -> '
              f'{sorted(report["suites"][s])}', flush=True)
    print(f'wrote {out}/exposure_audit.json/md', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
