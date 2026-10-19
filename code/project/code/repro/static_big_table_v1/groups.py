"""Benchmark definitions, loaders, and connectivity-based grouping (spec §4.2)."""
from __future__ import annotations

import re
import sys
from pathlib import Path

from common import DATASET, REPO, SNAPSHOT, load_jsonl

SNAP_DIR = SNAPSHOT

BENCH_ORDER = ['decomp', 'mg', 'minja', 'mcp']

BENCH_FILES = {
    'decomp': [
        'ref/dataset/decomp_graphs_train_mixed.jsonl',
        'ref/dataset/decomp_graphs_validation_mixed.jsonl',
        'ref/dataset/decomp_graphs_test_mixed.jsonl',
    ],
    'mg': [
        'ref/dataset/memorygraft_augmented.jsonl',
    ],
    'minja': [
        'ref/dataset/minja_real_traces.json',
        'code/repro/out/static_big_table_v1/minja_recovery/raw/minja_graphs.jsonl',
        'code/repro/out/static_big_table_v1/minja_recovery/raw/ref/minja_attack_instructions.json',
        'code/repro/out/static_big_table_v1/minja_recovery/raw/ref/minja_benign_instructions.json',
    ],
    'mcp': [
        'ref/dataset/mcp_safetybench_defense_matched_train.jsonl',
        'ref/dataset/mcp_safetybench_defense_matched_validation.jsonl',
        'ref/dataset/mcp_safetybench_defense_matched_test.jsonl',
    ],
}

BENCH_PROVENANCE = {
    'decomp': {
        'source_type': 'template-derived (DecomposedHarm / DecompAttack)',
        'real_or_synthetic': 'synthetic-template',
        'generator': 'code/decomposed_to_graph.py',
        'existing_split_use': (
            'test_mixed (352) used for all historical Decomp numbers '
            '(Q1/Q5/GRAD-2); validation_mixed unused historically; '
            'all roles are development/revalidation only'),
    },
    'mg': {
        'source_type': 'template-derived from real MemoryGraft exp4 payloads',
        'real_or_synthetic': 'hybrid-real-templates',
        'generator': 'code/augment_real_traces.py',
        'converter': 'code/memorygraft_to_graph.py',
        'existing_split_use': (
            'memorygraft_augmented used by EXP-2026W35-023 GroupKFold main '
            'number (AUC 0.9973) and by all Q1/Q5/Q10 runs'),
    },
    'minja': {
        'source_type': ('real DeepSeek WebShop runs on the MINJA instruction '
                        'set (prompt-level injection proxy; no environment '
                        'poisoning in the light runner)'),
        'real_or_synthetic': 'real-trace-proxy',
        'generator': 'run_minja_light.py (server, deepseek-chat, temp 0.1)',
        'converter': 'minja_to_graph.py + minja_merged.py (base-task grouping)',
        'existing_split_use': (
            'local 30 used for paper_tables_v3 MINJA AUC 0.738; server 160 '
            'used by pre-fix1 5-fold diagnostics and by the minja_cl_v1 '
            'scaling stages. All are development/revalidation only; the '
            'fix2 grouped-CV protocol was frozen from group counts before '
            'any fix2 score.'),
    },
    'mcp': {
        'source_type': 'real Docker runtime traces (MCP-SafetyBench)',
        'real_or_synthetic': 'real-runtime-trace',
        'generator': 'code/mcp_safetybench_run_batch_trace.py',
        'converter': 'code/mcp_trace_to_graph.py',
        'existing_split_use': (
            'matched train/validation/test used by mcp_text.py and MCP DGC '
            'experiments; all roles are development/revalidation only'),
    },
}

EXCLUDED_SOURCES = [
    {
        'path': 'ref/dataset/unified_test.jsonl',
        'reason': ('CL-sealed outer (MINJA 200 template graphs); spec forbids '
                   'filling static-table numbers from another CL sealed set'),
    },
    {
        'path': 'ref/dataset/toolshadow_preexec_graphs.jsonl',
        'reason': 'ToolShadowing is an independent extension table, not a four-benchmark column',
    },
    {
        'path': 'ref/dataset/memorygraft_bench.jsonl ; ref/dataset/memorygraft_real.jsonl',
        'reason': 'alternate MG views / original payloads; augmented file is the frozen MG corpus',
    },
    {
        'path': ('code/repro/out/static_big_table_v1/minja_recovery/raw/'
                 'minja_cl_v1/'),
        'reason': ('scaling stages 40/80/120/"148" and fixed test-24 are '
                   'derived random subsets of the same 160-graph server pool '
                   '(stage_0148 actually holds 136 rows), not independent '
                   'tasks; excluded from grouping to avoid pseudo-replication'),
    },
    {
        'path': 'ref/dataset/minja_webshop_traces/ (server, not synced)',
        'reason': ('full-environment WebShop runner traces were never '
                   'completed/archived; only the light-runner 160-graph pool '
                   'and the local 30 real traces exist'),
    },
]


def group_decomp(g: dict) -> str:
    m = re.match(r'(?:harmful|benign)_(\d+)', str(g.get('task_id', '')))
    return m.group(1) if m else str(g.get('task_id'))


def group_mg(g: dict) -> str:
    return str(g.get('task_id', '')).split('_v')[0]


def group_mcp(g: dict) -> str:
    task = str((g.get('metadata') or {}).get('task', g.get('task_id', '')))
    task = task.split('/')[-1]
    if task.endswith('.json'):
        task = task[:-5]
    return re.sub(r'_benign$', '', task)


def group_minja(g: dict) -> str:
    """S1: group by normalized original instruction, never by row index."""
    key = g.get('instruction_key')
    if not key:
        if str(SNAP_DIR) not in sys.path:
            sys.path.insert(0, str(SNAP_DIR))
        from minja_converter import normalize_instruction
        key = normalize_instruction(g.get('instruction', ''))
    return f'minja_instr:{key}'


GROUP_FN = {
    'decomp': group_decomp,
    'mg': group_mg,
    'minja': group_minja,
    'mcp': group_mcp,
}


def minja_to_graphs(path: Path | None = None) -> list[dict]:
    """Converter loaded from the frozen snapshot (source hash recorded)."""
    path = path or (DATASET / 'minja_real_traces.json')
    if str(SNAP_DIR) not in sys.path:
        sys.path.insert(0, str(SNAP_DIR))
    from minja_converter import minja_to_graphs as _convert
    return _convert(str(path))


def load_benchmark(name: str) -> list[dict]:
    """Load pooled graph list for a benchmark with _group and _source_file."""
    if name == 'minja':
        from minja_merged import load_merged
        return load_merged()
    graphs = []
    for rel in BENCH_FILES[name]:
        path = REPO / rel
        for i, g in enumerate(load_jsonl(path)):
            g['_source_file'] = rel
            g['_source_index'] = i
            graphs.append(g)
    gf = GROUP_FN[name]
    for g in graphs:
        g['_group'] = gf(g)
    return graphs


def group_label_composition(graphs: list[dict]) -> dict[str, list[int]]:
    comp: dict[str, list[int]] = {}
    for g in graphs:
        comp.setdefault(g['_group'], []).append(int(g['label']))
    return comp
