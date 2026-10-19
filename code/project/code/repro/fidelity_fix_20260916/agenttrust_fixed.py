#!/usr/bin/env python3
"""Faithful AgentTrust adapter v2 (official source, corrected I/O contract).

Fix list vs snapshot/compare_all_baselines.py::baseline_agenttrust (audit 2026-09-16):
  1. subtask is no longer truncated to 200 chars; the full observable text is
     passed as Action.description.
  2. observable structured fields (tool_name / server / record_type) are passed
     through Action.parameters; raw_content stays empty because the re-hosted
     graphs have no separate raw field (recorded in the coverage report).
  3. verdict mapping is grounded in the paper's own metrics (arXiv:2605.04785
     section 6.1): FNR = Pr[v in {allow,warn} | v*=block] -> strict positive is
     verdict in {block, review}; FPR = Pr[v != allow | v* = allow] -> alarm
     positive is verdict != allow. warn is counted separately.
  4. failures are not dropped: the official interceptor's internal error path
     (verdict=review, explanation "Evaluation error:") is counted as
     interceptor_error; wrapper-level exceptions are counted as adapter_error.
     Both are fail-safe positives per the paper's Property 2.
  5. session tracking (SessionTracker + RiskChain) is enabled for every trace;
     chain alerts are recorded per trace and chain-only hits are exposed as a
     separate sensitivity prediction.
  6. both rule sets are available: production (PolicyEngine.default) and
     production + benchmark-compatibility rules (the configuration used in the
     paper's internal benchmark, section 6.2).
"""
from __future__ import annotations

import hashlib
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
AT_ROOT = REPO / 'ref' / 'AgentTrust'

OBSERVABLE_PARAM_FIELDS = ('tool_name', 'server', 'record_type')

NODE_TO_AGENTTRUST = {
    'shell_exec': 'shell_command', 'code_exec': 'code_execution',
    'tool_execute': 'code_execution', 'file_read': 'file_read',
    'file_write': 'file_write', 'file_delete': 'file_delete',
    'web_request': 'network_request', 'web_search': 'network_request',
    'web_download': 'network_request', 'account_access': 'credential_access',
    'database_query': 'database_query', 'information_search': 'file_read',
    'data_aggregate': 'file_read', 'memory_write': 'file_write',
    'memory_read': 'file_read', 'memory_reflect': 'file_read',
    'version_control': 'file_write', 'agent_spawn': 'api_call',
    'agent_message': 'api_call', 'email_send': 'api_call',
    'message_send': 'api_call', 'social_media': 'api_call',
    'document_create': 'file_write', 'prompt_inject': 'unknown',
    'system_config': 'system_config',
}

SOURCE_FILES = (
    'pyproject.toml',
    'src/agent_trust/core/interceptor.py',
    'src/agent_trust/core/analyzer.py',
    'src/agent_trust/core/policy.py',
    'src/agent_trust/core/riskchain.py',
    'src/agent_trust/core/normalizer.py',
    'src/agent_trust/core/safefix.py',
    'src/agent_trust/core/types.py',
    'src/agent_trust/core/utils.py',
    'src/agent_trust/benchmarks/benchmark_compat.yaml',
)


def source_hashes() -> dict:
    out = {}
    for rel in SOURCE_FILES:
        p = AT_ROOT / rel
        out[rel] = (hashlib.sha256(p.read_bytes()).hexdigest()
                    if p.exists() else 'missing')
    return out


def _import_official():
    for rel in ('src', ''):
        p = str(AT_ROOT / rel)
        if p not in sys.path:
            sys.path.insert(0, p)
    from agent_trust.core.interceptor import TrustInterceptor
    from agent_trust.core.policy import PolicyEngine
    from agent_trust.core.types import Action, ActionType, TrustVerdict
    return TrustInterceptor, PolicyEngine, Action, ActionType, TrustVerdict


def build_policy(ruleset: str):
    _, PolicyEngine, _, _, _ = _import_official()
    if ruleset not in ('prod', 'compat'):
        raise ValueError(f'unknown ruleset {ruleset!r}')
    policy = PolicyEngine.default()
    if ruleset == 'compat':
        policy.load_benchmark_rules()
    return policy


def build_observer(ruleset: str):
    TrustInterceptor, _, _, _, _ = _import_official()
    policy = build_policy(ruleset)
    return TrustInterceptor(policy=policy)


def _action_from_node(Action, ActionType, node: dict, task_id: str):
    f = node.get('features') or {}
    at_name = NODE_TO_AGENTTRUST.get(str(node.get('type')), 'unknown')
    params = {}
    for k in OBSERVABLE_PARAM_FIELDS:
        v = f.get(k)
        if isinstance(v, str) and v:
            params[k] = v
    return Action(
        action_type=ActionType(at_name),
        tool_name=str(node.get('type', '')),
        description=str(f.get('subtask', '')),
        parameters=params,
        raw_content='',
        session_id=str(task_id),
        timestamp=datetime.now(timezone.utc),
    )


def evaluate_trace(graph: dict, interceptor, ruleset: str) -> dict:
    """Run the official interceptor over one trace; return verdict ledger."""
    TrustInterceptor, _, Action, ActionType, TrustVerdict = _import_official()
    verdicts = Counter()
    chain_ids = []
    adapter_errors = []
    interceptor_errors = []
    first_strict = None
    first_alarm = None
    missing_mapping = set()
    n_nodes = 0
    for node in graph.get('nodes', []):
        n_nodes += 1
        if str(node.get('type')) not in NODE_TO_AGENTTRUST:
            missing_mapping.add(str(node.get('type')))
        try:
            action = _action_from_node(Action, ActionType, node, graph.get('task_id', ''))
            report = interceptor.verify(action)
        except Exception as exc:  # wrapper-level failure -> fail-safe
            adapter_errors.append(f'{type(exc).__name__}: {exc}')
            verdicts['review'] += 1
            if first_strict is None:
                first_strict = n_nodes - 1
            if first_alarm is None:
                first_alarm = n_nodes - 1
            continue
        v = report.verdict.value
        verdicts[v] += 1
        if v == 'review' and str(report.explanation).startswith('Evaluation error'):
            interceptor_errors.append(str(report.explanation)[:200])
        for alert in getattr(report, 'chain_alerts', []) or []:
            chain = getattr(alert, 'chain', None)
            chain_ids.append(getattr(chain, 'id', str(chain)))
        if v in ('block', 'review') and first_strict is None:
            first_strict = n_nodes - 1
        if v != 'allow' and first_alarm is None:
            first_alarm = n_nodes - 1
    strict = int(first_strict is not None)
    alarm = int(first_alarm is not None)
    chain = int(len(chain_ids) > 0)
    return {
        'task_id': graph.get('task_id'),
        'label': int(graph.get('label', -1)),
        'n_nodes': n_nodes,
        'ruleset': ruleset,
        'verdict_counts': dict(verdicts),
        'n_warn': verdicts.get('warn', 0),
        'n_block': verdicts.get('block', 0),
        'n_review': verdicts.get('review', 0),
        'n_allow': verdicts.get('allow', 0),
        'chain_ids': chain_ids,
        'n_chain_alerts': len(chain_ids),
        'adapter_errors': adapter_errors,
        'interceptor_errors': interceptor_errors,
        'missing_mapping': sorted(missing_mapping),
        'first_flagged_step_strict': first_strict,
        'first_flagged_step_alarm': first_alarm,
        'pred_strict': strict,
        'pred_alarm': alarm,
        'pred_chain': chain,
        'pred_chain_only': int(chain and not strict),
    }


def predict(graphs: list[dict], ruleset: str = 'prod'):
    """Returns (preds dict, ledgers). Interceptor reused per ruleset; a fresh
    session is created per trace to preserve per-trace session isolation."""
    TrustInterceptor, _, _, _, _ = _import_official()
    policy = build_policy(ruleset)
    ledgers = []
    for g in graphs:
        interceptor = TrustInterceptor(policy=policy)
        ledgers.append(evaluate_trace(g, interceptor, ruleset))
    preds = {
        f'agenttrust_{ruleset}_strict': [d['pred_strict'] for d in ledgers],
        f'agenttrust_{ruleset}_alarm': [d['pred_alarm'] for d in ledgers],
        f'agenttrust_{ruleset}_chain': [int(d['pred_strict'] or d['pred_chain']) for d in ledgers],
        f'agenttrust_{ruleset}_chain_only': [d['pred_chain_only'] for d in ledgers],
    }
    return preds, ledgers
