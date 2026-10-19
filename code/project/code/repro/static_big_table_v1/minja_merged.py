"""Merged MINJA inventory loader (S1 final: instruction/task identity).

Sources (synced read-only from gpu-server, hashes in server_files.sha256.txt):
  * server light traces: 60 attack (10 base tasks x 6 variants) + 100 benign
    DeepSeek runs, graphs in results/minja_graphs/minja_graphs.jsonl (order
    1:1 with the instruction files and the trace files);
  * local real traces: ref/dataset/minja_real_traces.json (30 rows, identical
    copy also present on the server).

Grouping (pre-registered, no pseudo-groups):
  * every row is grouped by its ORIGINAL BASE TASK = instruction text with the
    parenthetical injection note removed and whitespace normalized;
  * the 15 local attack rows attach to 3 of the 10 server attack bases; all 15
    local benign rows attach to server benign bases. Local instructions are
    stored truncated at 100 chars, so a truncated local key is canonicalized to
    the unique server base key that starts with it (verified 1:1, no
    many-to-one collisions; see minja_recovery_audit.py);
  * expected inventory: 190 rows -> 10 attack groups + 100 benign groups = 110
    independent groups.

The instruction/session/variant/source fields are metadata only and never enter
model input.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from common import OUTROOT, REPO

RECOVERY = OUTROOT / 'minja_recovery' / 'raw'

SERVER_ATTACK = 60
SERVER_BENIGN = 100
LOCAL_ROWS = 30
EXPECTED_ATTACK_GROUPS = 10
EXPECTED_BENIGN_GROUPS = 100
EXPECTED_TOTAL_GROUPS = EXPECTED_ATTACK_GROUPS + EXPECTED_BENIGN_GROUPS
EXPECTED_TOTAL_ROWS = SERVER_ATTACK + SERVER_BENIGN + LOCAL_ROWS


def _norm(text) -> str:
    return re.sub(r'\s+', ' ', str(text).strip().lower())


def attack_base_key(instruction) -> str:
    base = re.split(r'\s*\(', str(instruction))[0]
    return _norm(base)


def benign_key(instruction) -> str:
    return _norm(instruction)


def _canonical_base(key: str, server_keys: set[str], side: str,
                    where: str) -> str:
    """Map a (possibly truncated) local key onto the unique server base."""
    if key in server_keys:
        return key
    matches = sorted(k for k in server_keys if k.startswith(key))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(f'{side}/{where}: ambiguous base prefix {key!r} -> '
                           f'{matches[:3]}')
    if side == 'minja_att':
        raise RuntimeError(f'{side}/{where}: local attack instruction does not '
                           f'match any server attack base: {key!r}')
    return key


def _server_rows() -> list[dict]:
    att = json.load(open(RECOVERY / 'ref/minja_attack_instructions.json'))
    ben = json.load(open(RECOVERY / 'ref/minja_benign_instructions.json'))
    graphs = [json.loads(l) for l in open(RECOVERY / 'minja_graphs.jsonl')
              if l.strip()]
    if len(att) != SERVER_ATTACK or len(ben) != SERVER_BENIGN:
        raise RuntimeError('server instruction inventory changed')
    if len(graphs) != SERVER_ATTACK + SERVER_BENIGN:
        raise RuntimeError('server graph inventory changed')
    rows = []
    for i in range(SERVER_ATTACK):
        g = dict(graphs[i])
        g['_source'] = 'server_light'
        g['_orig_task_id'] = g.get('task_id')
        g['task_id'] = f'server_att_{i:04d}'
        g['_instruction'] = att[i]['instruction']
        g['_session'] = att[i]['session']
        g['_variant'] = ('inject' if att[i]['session'].startswith('inject')
                         else att[i]['session'].rsplit('_', 1)[0])
        g['_group'] = 'minja_att:' + attack_base_key(att[i]['instruction'])
        g['_source_file'] = 'minja_recovery/server_light'
        rows.append(g)
    for j in range(SERVER_BENIGN):
        g = dict(graphs[SERVER_ATTACK + j])
        g['_source'] = 'server_light'
        g['_orig_task_id'] = g.get('task_id')
        g['task_id'] = f'server_ben_{j:04d}'
        g['_instruction'] = ben[j]['instruction']
        g['_session'] = ben[j]['session']
        g['_variant'] = ben[j]['session'].rsplit('_', 1)[0]
        g['_group'] = 'minja_ben:' + benign_key(ben[j]['instruction'])
        g['_source_file'] = 'minja_recovery/server_light'
        rows.append(g)
    return rows


def local_trace_to_graph(trace: dict, idx: int) -> dict:
    nodes, edges, prev = [], [], None
    for j, step in enumerate(trace.get('steps', [])):
        nid = f'm{j}'
        nodes.append({'id': nid, 'type': step.get('type', 'tool_execute'),
                      'taint': 0, 'turn': j,
                      'features': {'subtask': str(step.get('action', ''))}})
        if prev:
            edges.append({'source': prev, 'target': nid, 'type': 'data_flow'})
        prev = nid
    label = int(trace['label'])
    instruction = str(trace.get('instruction', ''))
    group = ('minja_att:' if label == 1 else 'minja_ben:') + _norm(instruction)
    return {'task_id': f'local_{idx:04d}', 'label': label,
            'nodes': nodes, 'edges': edges,
            '_source': 'local_real', '_orig_task_id': f'minja_{idx}',
            '_instruction': instruction, '_session': trace.get('session', ''),
            '_variant': 'local_real', '_group': group,
            '_source_file': 'ref/dataset/minja_real_traces.json'}


def _local_rows(server_att_keys: set[str],
                server_ben_keys: set[str]) -> list[dict]:
    traces = json.load(open(REPO / 'ref/dataset/minja_real_traces.json'))
    if len(traces) != LOCAL_ROWS:
        raise RuntimeError('local MINJA trace inventory changed')
    rows = []
    for i, t in enumerate(traces):
        g = local_trace_to_graph(t, i)
        side = 'minja_att' if int(g['label']) == 1 else 'minja_ben'
        raw_key = attack_base_key(t['instruction']) if side == 'minja_att' \
            else benign_key(t['instruction'])
        canon = _canonical_base(raw_key, server_att_keys if side == 'minja_att'
                                else server_ben_keys, side, f'local_{i:04d}')
        g['_raw_key'] = raw_key
        g['_group'] = f'{side}:{canon}'
        rows.append(g)
    return rows


def load_merged() -> list[dict]:
    server = _server_rows()
    server_att_keys = {r['_group'].split(':', 1)[1] for r in server
                       if int(r['label']) == 1}
    server_ben_keys = {r['_group'].split(':', 1)[1] for r in server
                       if int(r['label']) == 0}
    rows = server + _local_rows(server_att_keys, server_ben_keys)
    for i, g in enumerate(rows):
        g['_source_index'] = i
    groups = {g['_group'] for g in rows}
    pos = {g['_group'] for g in rows if int(g['label']) == 1}
    neg = {g['_group'] for g in rows if int(g['label']) == 0}
    if len(rows) != EXPECTED_TOTAL_ROWS:
        raise RuntimeError(f'MINJA merged inventory changed: rows={len(rows)}')
    if len(groups) != EXPECTED_TOTAL_GROUPS or len(pos) != EXPECTED_ATTACK_GROUPS \
            or len(neg) != EXPECTED_BENIGN_GROUPS:
        raise RuntimeError(
            f'MINJA merged inventory changed: groups={len(groups)} '
            f'pos={len(pos)} neg={len(neg)}')
    if pos & neg:
        raise RuntimeError('MINJA group crossed attack/benign labels')
    local_att = {g['_group'] for g in rows
                 if g['_source'] == 'local_real' and int(g['label']) == 1}
    if not local_att <= pos or len({g for g in local_att}) != 3:
        raise RuntimeError('local attack rows must attach to server attack '
                           'base-task groups')
    return rows
