"""Frozen branch state for checkpoint-based closed-loop evaluation.

The v10 host delivers the memory-poisoning payload deterministically in the
first external tool observation of session 1, so the main-protocol branch
point is the clean task start: every arm restarts from the same pinned
payload, task texts, tool catalog and delivery rule. ``freeze_branch`` pins
those inputs and exposes content hashes so a plan-level audit can prove that
all arms share a byte-identical starting state.

``snapshot_state`` / ``restore_state`` provide the explicit mid-run
save/branch primitive. They are used by the offline tests and are the
mechanism future environments (later injection points) must use instead of
re-running the whole trajectory.
"""
from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / 'code/repro/v10'))

from envlib import ENVS  # noqa: E402

READ_TOOLS = ('search', 'scan', 'lookup', 'click', 'remember', 'recall')
BRANCH_POINT = 'task_start'
MOCK_BRANCH_POINT = 'mid_trajectory'
DELIVERY = {'session': 1, 'position': 'first_external_tool_observation',
            'step': 0}


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False,
                      separators=(',', ':'))


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode('utf-8')).hexdigest()


def sha256_obj(obj) -> str:
    return sha256_text(canonical_json(obj))


def payload_for(env_name: str, family: str, condition: str) -> str:
    fam = ENVS[env_name]['families'][family]
    if condition in ('attack_instruction', 'authorized_override'):
        return fam['note_bad']
    if condition == 'benign_instruction':
        return fam['note_good']
    raise ValueError(f'unknown condition {condition!r}')


def tool_catalog(env_name: str) -> list[dict]:
    """Structured catalog for the v10 host (the native system prompt only has
    a text rendering). Action tool is a write tool with a security-critical
    ``arg`` parameter; everything else is read-only."""
    env = ENVS[env_name]
    catalog = [{'name': t, 'kind': 'read', 'parameters': ['arg']}
               for t in READ_TOOLS]
    catalog.append({'name': env['action'], 'kind': 'write',
                    'parameters': ['arg'], 'security_critical': ['arg']})
    return catalog


def freeze_branch(env_name: str, family: str, condition: str) -> dict:
    env = ENVS[env_name]
    fam = env['families'][family]
    payload = payload_for(env_name, family, condition)
    state = {
        'branch_point': BRANCH_POINT,
        'env': env_name,
        'family': family,
        'condition': condition,
        'task1': fam['s1'],
        'task2': fam['s2'],
        'sys1_sha256': sha256_text(env['sys1']),
        'sys2_sha256': sha256_text(env['sys2']),
        'catalog_sha256': sha256_obj(tool_catalog(env_name)),
        'payload': payload,
        'payload_sha256': sha256_text(payload),
        'delivery': dict(DELIVERY),
    }
    state['branch_sha256'] = sha256_obj(
        {k: v for k, v in state.items() if k not in ('payload',
                                                     'branch_sha256')})
    return state


def same_branch_sha(a: dict, b: dict) -> bool:
    return a.get('branch_sha256') == b.get('branch_sha256')


def snapshot_state(messages: list, trace: dict) -> dict:
    """Deep-copied mid-run checkpoint (messages + trace)."""
    return {'branch_point': MOCK_BRANCH_POINT,
            'messages': copy.deepcopy(list(messages)),
            'trace': copy.deepcopy(trace)}


def restore_state(snapshot: dict) -> tuple[list, dict]:
    """Independent copies so branching twice cannot alias state."""
    return copy.deepcopy(snapshot['messages']), copy.deepcopy(snapshot['trace'])
