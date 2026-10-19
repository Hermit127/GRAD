"""Shared utilities for the static cross-benchmark big-table task.

Spec: LOGS/static_big_table_execution_spec_20260910.md.
This package is an independent snapshot: it never imports CL/antibody modules
and never writes to grad_bundle/, decomp_bundle/, online_adapted_bundle/,
antibody_cl_v2/ or any other historical output directory.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO = Path('.')
PKG = REPO / 'code/repro/static_big_table_v1'
SNAPSHOT = PKG / 'snapshot'
OUTROOT = REPO / 'code/repro/out/static_big_table_v1'
DATASET = REPO / 'ref/dataset'

SPEC_PATH = REPO / 'LOGS/static_big_table_execution_spec_20260910.md'
ACCEPT_PATH = REPO / 'LOGS/q1_q10_acceptance_20260910.md'


def sha256_file(path: Path | str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def sha256_json(obj) -> str:
    return sha256_bytes(json.dumps(obj, sort_keys=True,
                                   ensure_ascii=False).encode('utf-8'))


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


def _jsonable(obj):
    if isinstance(obj, float):
        if obj != obj:
            return None
        if obj == float('inf'):
            return '+inf'
        if obj == float('-inf'):
            return '-inf'
        return obj
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as fh:
        json.dump(_jsonable(obj), fh, indent=2, ensure_ascii=False,
                  allow_nan=False)


def load_json(path: Path):
    with open(path) as fh:
        return json.load(fh)


def load_jsonl(path: Path) -> list[dict]:
    out = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def git_info() -> dict:
    def run(*args):
        try:
            return subprocess.run(args, cwd=REPO, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except subprocess.CalledProcessError as exc:
            return f'ERROR: {exc}'
    info = {
        'head': run('git', 'rev-parse', 'HEAD'),
        'head_subject': run('git', 'log', '-1', '--format=%s'),
        'status_porcelain': run('git', 'status', '--porcelain'),
        'diff': run('git', 'diff'),
    }
    info['diff_sha256'] = sha256_bytes(info['diff'].encode('utf-8'))
    info['status_sha256'] = sha256_bytes(
        info['status_porcelain'].encode('utf-8'))
    return info


def environment_info() -> dict:
    import importlib
    import platform
    import sys
    versions = {}
    for name in ('numpy', 'sklearn', 'xgboost', 'torch',
                 'torch_geometric', 'scipy'):
        try:
            versions[name] = importlib.import_module(name).__version__
        except Exception as exc:  # noqa: BLE001
            versions[name] = f'unavailable: {exc}'
    return {
        'timestamp_utc': utc_now(),
        'python': sys.version,
        'executable': sys.executable,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'processor': platform.processor(),
        'cpu_count': os.cpu_count(),
        'env_vars': {k: os.environ.get(k) for k in
                     ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                      'MKL_NUM_THREADS', 'PYTHONHASHSEED')},
        'versions': versions,
    }


def ensure_run_dir(run_id: str, config: dict, input_hashes: dict,
                   root: Path | None = None) -> Path:
    """Unique run directory; resume only with identical config + input hash."""
    root = root or OUTROOT
    run_dir = root / run_id
    marker = run_dir / 'run_identity.json'
    identity = {'config': config, 'input_hashes': input_hashes}
    if run_dir.exists():
        if not marker.exists():
            raise RuntimeError(
                f'run dir {run_dir} exists without run_identity.json; refuse overwrite')
        prev = load_json(marker)
        if prev != identity:
            raise RuntimeError(
                f'run {run_id} exists with different config/input hashes; refuse')
        return run_dir
    run_dir.mkdir(parents=True)
    write_json(marker, identity)
    return run_dir
