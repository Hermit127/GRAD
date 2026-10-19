#!/usr/bin/env python3
"""AEGIS corrected implementation (audit 2026-09-16 fixes).

Paper: arXiv:2603.12621; pattern catalogue and pipeline ported verbatim from
the vendored official source ref/AEGIS/packages/gateway-mcp/src/services/
classifier.ts (paper v1 differs only in the supply-chain entries; the code is
newer - noted in the README).

Fixes vs the old aegis22 proxy (channels.py: search AEGIS_PATTERNS over
node text and OR the hits):
  1. Stage 1 recursive string extraction with depth 32 / cap 10000 and the
     __SCAN_TRUNCATED__ sentinel (+ MEDIUM large_payload risk);
  2. Stage 2 is the full official risk catalogue with per-pattern severities
     and the content -> name category inference;
  3. Stage 3 JSON Schema policies (draft-07 via jsonschema, allErrors) with
     the category applicability map and violation promotion;
  4. decision follows the SDK fast path: any violation -> block, else allow;
     a pending variant is reported when the request marks blocking=True.

Adaptation boundary: our re-hosted graphs do not carry structured tool
argument objects for most corpora, so when no JSON arguments can be extracted
the node text is treated as the single argument string. This is an adapter
approximation of the SDK argument interception, recorded per node.
"""
from __future__ import annotations

import json
import re

import jsonschema

SEVERITY_ORDER = {'LOW': 1, 'MEDIUM': 2, 'HIGH': 3, 'CRITICAL': 4}
SCAN_TRUNCATED = '__SCAN_TRUNCATED__'
MAX_SCAN_DEPTH = 32
MAX_STRING_COUNT = 10000

NAME_KEYWORDS = [
    (['sql', 'query', 'database', 'db_', '_db', 'sqlite', 'postgres', 'mysql',
      'execute', 'select', 'insert', 'update', 'delete_row', 'fetch_row',
      'run_query', 'exec_query'], 'database'),
    (['file', 'read_', 'write_', 'open_', 'path', 'dir', 'folder', 'ls_',
      'list_', 'mkdir', 'rm_', 'delete_file', 'move_file', 'copy_file',
      'glob', 'stat_'], 'file'),
    (['http', 'request', 'fetch', 'url', 'api_call', 'api_get', 'api_post',
      'webhook', 'get_url', 'post_url', 'download', 'upload', 'curl',
      'web_search', 'browse', 'scrape'], 'network'),
    (['shell', 'exec', 'bash', 'cmd', 'command', 'run_', 'subprocess',
      'terminal', 'powershell', 'spawn', 'popen'], 'shell'),
    (['email', 'send_', 'mail', 'slack', 'notify', 'message', 'sms',
      'telegram', 'discord', 'push_notification', 'alert_'], 'communication'),
    (['publish', 'deploy', 'release', 'push_package', 'upload_artifact',
      'npm_publish', 'docker_push', 'registry', 'promote', 'rollout'],
     'supply-chain'),
]

CATEGORY_DEFAULT_RISK = {'shell': 'CRITICAL', 'database': 'HIGH',
                         'supply-chain': 'HIGH', 'file': 'MEDIUM',
                         'network': 'MEDIUM', 'communication': 'MEDIUM',
                         'data': 'LOW', 'unknown': 'LOW'}


def _rx(pattern, flags=re.I):
    return re.compile(pattern, flags)


SQL_PATTERNS = [
    ('HIGH', _rx(r"\b(OR|AND)\s+['\"]?\d+['\"]?\s*=\s*['\"]?\d+['\"]?|--|;.*DROP|UNION\s+SELECT|'\s*OR\s+'1'\s*=\s*'1")),
    ('HIGH', _rx(r"\bWAITFOR\s+DELAY\b|\bpg_sleep\s*\(|\bBENCHMARK\s*\(|\bSLEEP\s*\(")),
    ('HIGH', _rx(r"/\*[^*]*\*/\s*(?:SELECT|UNION|DROP|INSERT|DELETE|UPDATE|EXEC)\b")),
    ('MEDIUM', _rx(r"\bCONCAT\s*\(|\bCHR\s*\(|\bCHAR\s*\(")),
    ('HIGH', _rx(r";\s*(?:SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER|EXEC|TRUNCATE)\b")),
    ('MEDIUM', _rx(r"0x[0-9a-fA-F]{8,}")),
    ('MEDIUM', _rx(r"\b(DROP|TRUNCATE|ALTER|CREATE\s+TABLE|INSERT\s+INTO|DELETE\s+FROM)\b")),
]

SENSITIVE_PATHS = ['/etc/passwd', '/etc/shadow', '/.ssh/', '/.aws/', '/.env',
                   '/proc/', '/.kube/', '/.terraform/', '/.bashrc', '/.zshrc',
                   '/.profile', '/.gitconfig', '/.npmrc', '/.docker/',
                   '/id_rsa', '/id_ed25519']

INJECT = _rx('|'.join([
    'ignore (previous|above|all|prior|every)',
    'disregard (all|your|the|previous)', 'you are now', 'act as if',
    'new instructions', 'forget (your|all|previous|everything)',
    'override (your|system|all)', 'jailbreak', 'do anything now', 'DAN mode',
    'developer mode', 'bypass (filter|safety|content|restriction)',
    'pretend (you|to be|that)', 'roleplay as', 'system prompt',
    'reveal (your|the|system)',
    'what (are|is) your (instructions|prompt|system)']))


def _first(vals, pred, fmt):
    for v in vals:
        if pred(v):
            return fmt(v)
    return None


def risk_signals(values, args):
    risks = []
    risks.append(('sql_injection', 'HIGH', _first(
        values, lambda v: any(p.search(v) for _, p in SQL_PATTERNS),
        lambda v: f'SQL injection pattern in: "{v[:60]}"')))
    risks.append(('path_traversal', 'HIGH', _first(
        values, lambda v: ('../' in v or '..\\' in v or v.startswith('~/')),
        lambda v: f'Path traversal in: "{v[:60]}"')))
    risks.append(('path_traversal', 'HIGH', _first(
        values, lambda v: bool(_rx(r'%2e%2e(%2f|%5c)', re.I).search(v)),
        lambda v: f'URL-encoded path traversal in: "{v[:60]}"')))
    risks.append(('path_traversal', 'HIGH', _first(
        values, lambda v: bool(_rx(r'%252e%252e', re.I).search(v)),
        lambda v: f'Double-encoded path traversal in: "{v[:60]}"')))
    risks.append(('path_traversal', 'MEDIUM', _first(
        values, lambda v: ('%00' in v or '\x00' in v),
        lambda v: f'Null byte injection in: "{v[:60]}"')))
    risks.append(('sensitive_file', 'CRITICAL', _first(
        values, lambda v: any(s in v for s in SENSITIVE_PATHS),
        lambda v: 'Access to sensitive path: ' + next(
            s for s in SENSITIVE_PATHS if s in v))))
    risks.append(('shell_injection', 'HIGH', _first(
        values, lambda v: bool(_rx(r'[;&|`]|\$\(|\|\|').search(v)),
        lambda v: f'Shell metacharacter in: "{v[:60]}"')))
    risks.append(('shell_injection', 'HIGH', _first(
        values, lambda v: bool(_rx(r'\b(curl|wget|nc|ncat)\b.*\b(https?://|ftp://|//)').search(v)),
        lambda v: f'Dangerous command with network target in: "{v[:60]}"')))
    risks.append(('shell_injection', 'HIGH', _first(
        values, lambda v: bool(_rx(r'\$\{?IFS\}?').search(v)),
        lambda v: f'Shell IFS bypass in: "{v[:60]}"')))
    risks.append(('shell_injection', 'HIGH', _first(
        values, lambda v: bool(_rx(r'<\(|>\(|[<>]\s*/(?:etc|proc|dev|tmp)').search(v)),
        lambda v: f'Shell process substitution or redirect in: "{v[:60]}"')))
    risks.append(('plaintext_url', 'LOW', _first(
        values, lambda v: bool(_rx(r'^http://').search(v)),
        lambda v: f'Plaintext HTTP URL: "{v[:80]}"')))
    risks.append(('large_payload', 'MEDIUM', None if len(
        json.dumps(args, default=str)) <= 50000 else
        f'Large payload: {len(json.dumps(args, default=str)) / 1024:.1f}KB'))
    if any(_rx(r'^https?://').search(v) for v in values) and \
            any(len(v) > 5000 for v in values):
        risks.append(('data_exfiltration', 'HIGH',
                      'Possible data exfiltration: large payload with external URL'))
    risks.append(('prompt_injection', 'CRITICAL', _first(
        values, lambda v: bool(INJECT.search(v)),
        lambda v: f'Prompt injection attempt: "{v[:80]}"')))
    risks.append(('source_map_leak', 'HIGH', _first(
        values, lambda v: bool(_rx(r'\b(npm\s+publish|npx\s+publish|yarn\s+publish|pnpm\s+publish)\b').search(v)),
        lambda v: f'Package publish detected: "{v[:80]}"')))
    risks.append(('source_map_leak', 'HIGH', _first(
        values, lambda v: bool(_rx(r'\.(js|ts|jsx|tsx|css|mjs|cjs)\.map\b').search(v)),
        lambda v: f'Source map file access: "{v[:80]}"')))
    risks.append(('source_map_leak', 'CRITICAL', _first(
        values, lambda v: bool(_rx(r"[\"']?sourcesContent[\"']?\s*:\s*\[|//[#@]\s*sourceMappingURL\s*=").search(v)),
        lambda v: f'Source map content detected: "{v[:80]}"')))
    risks.append(('unsafe_publish', 'HIGH', _first(
        values, lambda v: bool(_rx(r'\b(docker\s+push|helm\s+install|kubectl\s+apply|terraform\s+apply|pulumi\s+up|gcloud\s+deploy|aws\s+(s3\s+cp|ecr\s+push|deploy))\b').search(v)),
        lambda v: f'Deployment command detected: "{v[:80]}"')))
    risks.append(('unsafe_publish', 'HIGH', _first(
        values, lambda v: bool(_rx(r'\b(twine\s+upload|gem\s+push|cargo\s+publish|nuget\s+push|pip.*upload|setuptools.*upload)\b').search(v)),
        lambda v: f'Package registry publish detected: "{v[:80]}"')))
    risks.append(('unsafe_publish', 'MEDIUM', _first(
        values, lambda v: bool(_rx(r'\bgit\s+push\b.*\b(--force|origin\s+main|origin\s+master)\b').search(v)),
        lambda v: f'Git push detected: "{v[:80]}"')))
    risks.append(('secret_in_build', 'CRITICAL', _first(
        values, lambda v: bool(_rx(r"\b(PRIVATE.KEY|SECRET_KEY|API_KEY|ACCESS_TOKEN|AUTH_TOKEN|PASSWORD)\s*[=:]\s*['\"][^'\"]{8,}").search(v)),
        lambda v: f'Secret/credential in build artifact: "{v[:60]}"')))
    risks.append(('secret_in_build', 'HIGH', _first(
        values, lambda v: bool(_rx(r'\b(_authToken|//registry\.npmjs\.org/:_authToken|password\s*=\s*\S+)').search(v)),
        lambda v: f'Build config contains auth credentials: "{v[:60]}"')))
    return [{'type': t, 'severity': s, 'detail': d} for t, s, d in risks
            if d is not None]


def extract_string_values(obj, depth=0, ctx=None):
    if ctx is None:
        ctx = {'count': 0}
    if depth > MAX_SCAN_DEPTH or ctx['count'] > MAX_STRING_COUNT:
        return [SCAN_TRUNCATED]
    if isinstance(obj, str):
        ctx['count'] += 1
        return [obj]
    if isinstance(obj, list):
        out = []
        for v in obj:
            out.extend(extract_string_values(v, depth + 1, ctx))
        return out
    if isinstance(obj, dict):
        out = []
        for v in obj.values():
            out.extend(extract_string_values(v, depth + 1, ctx))
        return out
    return []


def category_from_content(args):
    joined = '\n'.join(extract_string_values(args)).lower()
    if _rx(r'\b(select|insert|update|delete|drop|create table|alter table)\b').search(joined):
        return 'database'
    if _rx(r'\b(npm\s+publish|yarn\s+publish|pnpm\s+publish|docker\s+push|twine\s+upload|cargo\s+publish|terraform\s+apply|kubectl\s+apply)\b').search(joined):
        return 'supply-chain'
    vals = extract_string_values(args)
    if _rx(r'^(https?|ftp)://').search(joined) or any(
            _rx(r'^https?://').search(v) for v in vals):
        return 'network'
    if any(_rx(r'^/[a-z]|^[a-z]:\\').search(v) or './' in v for v in vals):
        return 'file'
    if _rx(r'[;&|`]|\$\(').search(joined):
        return 'shell'
    return None


def category_from_name(tool_name):
    lower = str(tool_name).lower()
    for keywords, category in NAME_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return category
    return None


def classify_tool_call(tool_name, args):
    signals = []
    values = extract_string_values(args)
    risks = risk_signals(values, args)
    if SCAN_TRUNCATED in values:
        risks.append({'type': 'large_payload', 'severity': 'MEDIUM',
                      'detail': 'Argument nesting exceeded scan depth - partial scan'})
        signals.append('scan_truncated')
    cat = category_from_content(args) or category_from_name(tool_name) or 'unknown'
    return {'category': cat,
            'default_risk': CATEGORY_DEFAULT_RISK[cat],
            'risks': risks,
            'signals': signals + [f"{r['type']}:{r['severity']}" for r in risks]}


PUBLISHED_POLICY = {
    'id': 'sql-injection-guard',
    'name': 'Block destructive SQL keywords in query arguments',
    'category': ['database'],
    'risk_level': 'HIGH',
    'schema': {'not': {'properties': {
        'query': {'pattern': 'INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE'}}}},
}

POLICY_CATEGORIES = {
    'sql-injection-guard': ['database'],
    'file-access': ['file'],
    'network-access': ['network'],
    'data-exfiltration': ['network', 'communication'],
    'source-map-leak': ['supply-chain', 'shell'],
    'supply-chain': ['supply-chain', 'shell'],
}


def validate_policies(classification, args, policies=None):
    policies = [PUBLISHED_POLICY] if policies is None else policies
    violations, failed = [], []
    validator_cache = {}
    for policy in policies:
        cats = POLICY_CATEGORIES.get(policy['id'], policy.get('category', []))
        if cats and classification['category'] not in cats:
            continue
        key = policy['id']
        if key not in validator_cache:
            validator_cache[key] = jsonschema.Draft7Validator(policy['schema'])
        errors = sorted(validator_cache[key].iter_errors(args),
                        key=lambda e: tuple(str(part) for part in e.path))
        if errors:
            failed.append(policy['id'])
            violations.extend(f"{policy['id']}: {e.message}" for e in errors)
    return violations, failed


def _node_args(node):
    f = node.get('features') or {}
    payload = f.get('raw_payload')
    if isinstance(payload, dict) and isinstance(payload.get('arguments'), (dict, list)):
        return str(payload.get('tool_name') or f.get('tool_name') or node.get('type')), payload['arguments']
    text = str(f.get('subtask', ''))
    m = re.search(r'tool_name:\s*(?P<tool>.*?)\s*arguments:\s*(?P<args>.*?)\s*response:', text, re.I | re.S)
    if m:
        try:
            return m.group('tool').strip(), json.loads(m.group('args'))
        except ValueError:
            return m.group('tool').strip(), {'_text': m.group('args')}
    tool = str(f.get('tool_name') or node.get('type'))
    return tool, {'_text': text}


def inspect_graph(graph, policies=None):
    violations_all = []
    for node in graph.get('nodes', []):
        tool, args = _node_args(node)
        classification = classify_tool_call(tool, args)
        pol_viol, failed = validate_policies(classification, args, policies)
        details = [f"{r['type']}: {r['detail']}" for r in classification['risks']]
        max_risk = classification['default_risk']
        for r in classification['risks']:
            if SEVERITY_ORDER[r['severity']] > SEVERITY_ORDER[max_risk]:
                max_risk = r['severity']
        for v in pol_viol:
            if SEVERITY_ORDER['HIGH'] > SEVERITY_ORDER[max_risk]:
                max_risk = 'HIGH'
        violations = details + pol_viol
        if violations:
            violations_all.append({'node': node['id'], 'tool': tool,
                                   'category': classification['category'],
                                   'risk_level': max_risk,
                                   'violations': violations[:6]})
    pred_block = int(len(violations_all) > 0)
    top_risk = 'LOW'
    for v in violations_all:
        if SEVERITY_ORDER[v['risk_level']] > SEVERITY_ORDER[top_risk]:
            top_risk = v['risk_level']
    return {'violations': violations_all, 'top_risk': top_risk,
            'pred_block': pred_block,
            'pred_pending': int(len(violations_all) > 0 and
                                SEVERITY_ORDER[top_risk] >= SEVERITY_ORDER['HIGH'])}


def predict(graphs, policies=None):
    ledgers = [inspect_graph(g, policies=policies) for g in graphs]
    preds = {'aegis_block': [d['pred_block'] for d in ledgers],
             'aegis_pending_variant': [d['pred_pending'] for d in ledgers]}
    return preds, ledgers
