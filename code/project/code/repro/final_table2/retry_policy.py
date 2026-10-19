"""Frozen retry / dedup policy for Table 2 (PI 2026-09-17, item E).

Rules (frozen before any formal batch; do not change per-result):

  1. Every attempt is kept. The aggregator never deletes error records.
  2. Errors are classified from the first token of ``error``:
       * ``retryable`` - infrastructure failures only (allowlist below);
       * ``terminal``  - everything else (parser bugs, config errors, logic
         errors). Terminal failures are never silently replaced by a rerun.
  3. Only retryable errors may be followed by a retry; retries never add new
     independent cases (same cell key).
  4. Cell resolution is deterministic and order-frozen: attempts are ordered
     by the frozen panel file list + within-file order (never by glob order);
     the chosen result is the LAST error-free attempt when one exists.
     ``recovered=True`` means at least one retryable error precedes a success.
  5. A cell with no error-free attempt is a final failure; it stays in the
     denominator of its panel and is reported (never dropped).
  6. A cell whose attempts carry different protocol identities
     (runner_contract / benchmark / model / agentdojo version) is flagged
     ``protocol_mismatch`` and is not merged into the same comparison.

Micro aggregation counts one entry per unique case key; repeats of the same
key do not inflate the denominator.
"""
from __future__ import annotations

RETRYABLE_ERROR_PREFIXES = frozenset({
    'APIConnectionError', 'APITimeoutError', 'RateLimitError',
    'InternalServerError', 'ServiceUnavailableError', 'ServerError',
    'TimeoutError', 'ReadTimeout', 'ConnectTimeout',
    'ConnectionError', 'HTTPError',
})

CASE_KEY_FIELDS = ('suite', 'user_task', 'condition', 'arm', 'injection_task')


def cell_key(record: dict) -> tuple:
    return tuple(record.get(f) or '' for f in CASE_KEY_FIELDS)


def protocol_identity(record: dict) -> tuple:
    return (record.get('runner_contract'), record.get('benchmark_version'),
            record.get('model'), record.get('agentdojo_version'))


def classify_error(error) -> str:
    if not error:
        return 'ok'
    token = str(error).split(':')[0].strip()
    return 'retryable' if token in RETRYABLE_ERROR_PREFIXES else 'terminal'


def resolve_cell(attempts: list[dict]) -> dict:
    """Frozen resolution of one case cell. ``attempts`` are in frozen order."""
    n = len(attempts)
    retryable = sum(1 for a in attempts
                    if classify_error(a.get('error')) == 'retryable')
    terminal = sum(1 for a in attempts
                   if classify_error(a.get('error')) == 'terminal')
    successes = [a for a in attempts if classify_error(a.get('error')) == 'ok']
    identities = {protocol_identity(a) for a in successes}
    mismatch = len(identities) > 1
    chosen = successes[-1] if successes else None
    if mismatch:
        chosen = None
        status = 'protocol_mismatch'
    elif terminal:
        chosen = None
        status = 'terminal_failure'
    elif chosen is not None:
        status = 'ok'
    elif terminal:
        status = 'terminal_failure'
    else:
        status = 'retry_failure'
    return {
        'status': status,
        'chosen': chosen,
        'n_attempts': n,
        'n_errors': retryable + terminal,
        'n_retryable_errors': retryable,
        'n_terminal_errors': terminal,
        'recovered': bool(chosen) and retryable > 0,
        'protocol_mismatch': mismatch,
        'protocol_identities': sorted(str(i) for i in identities),
    }


def resolve_cells(records: list[dict]) -> tuple[dict, list[dict]]:
    """Group records by cell key (frozen order) and resolve each cell.

    Returns ``(resolved_by_key, attempt_log)``. The attempt log keeps every
    record with its classification so callers can persist all attempts.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in records:
        groups.setdefault(cell_key(r), []).append(r)
    resolved = {}
    attempt_log = []
    for key, attempts in groups.items():
        res = resolve_cell(attempts)
        resolved[key] = res
        for i, a in enumerate(attempts):
            attempt_log.append({
                'case_key': list(key),
                'attempt_index': i,
                'classification': classify_error(a.get('error')),
                'error': a.get('error'),
                'origin': a.get('origin'),
                'chosen': a is res['chosen'],
            })
    return resolved, attempt_log
