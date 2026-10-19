#!/usr/bin/env python
"""NeuroTaint corrected implementation (audit 2026-09-16 fixes).

Fixes vs code/repro/neurotaint_v2/neurotaint.py, per arXiv:2604.23374 spec
extracted in LOGS/baseline_fidelity_fix_plan_20260916.md:

  1. Tier2 sigma2 = |LCS| / min(|src|, |sink|) with an exact DP longest common
     subsequence (the old code used difflib.SequenceMatcher matching-block sums,
     which is not the LCS) and no length truncation (old cap=300).
  2. comparisons are per sink argument (parsed JSON values) as in the paper,
     not one concatenated argument blob; the old min-length-24 skip is removed.
  3. Tier3 keeps all-MiniLM-L6-v2 and theta 0.60 / 0.85-via-memory, but no
     400-char truncation; Tier4 coverage is the fraction of source characters
     in chunks with similarity >= theta_sem (old: argmax chunk only).
  4. memory taint transfer is a taint-preserving write->read link over the
     DCPG; reachability from a source now traverses those links, so
     source -> write -> read -> sink chains are detected (old code only
     compared rehyd.get(sink) == source, which never matched a read source).
  5. the Sink-Driven Causal Analyzer is NOT replaced by a keyword proxy: it is
     reported as not_applicable with an explicit reason, because it requires a
     neutralized-context LLM judge over the full agent history (paper section
     4.3), which the re-hosted graphs do not contain. The old deterministic
     causal_proxy is kept only under an explicitly named ablation flag.
  6. user_texts for the causal stage come only from nodes actually marked as
     trusted user input; the re-hosted graphs have no such node type, so the
     field is empty and recorded as not_applicable (old: first node's text).
"""
from __future__ import annotations

import json
from collections import defaultdict

from neurotaint import (compare_texts, cosine, embed, get_model, parse_fields,
                        sink_arg_text, strip_schema)
from policy import (MEMORY_READ, MEMORY_WRITE, PROVENANCE_EDGES, SINK_TYPES,
                    SOURCE_TYPES, THETA_COV, THETA_SEM, THETA_SEM_RAG,
                    THETA_STR, THETA_STR_IMPL, CAUSAL_CUES, node_text,
                    session_of)

MEMORY_EDGES = ('writes_to', 'reads_from')

USER_INPUT_TYPES = frozenset({'user', 'user_request', 'user_prompt', 'task'})
LCS_CAP_DEFAULT = None
LCS_UNIT_DEFAULT = 'char'
_TOKEN_RE = None


def _tokens(text: str):
    global _TOKEN_RE
    if _TOKEN_RE is None:
        import re
        _TOKEN_RE = re.compile(r'[A-Za-z0-9_]+')
    return _TOKEN_RE.findall(str(text).lower())


def lcs_length(a, b) -> int:
    """Exact longest-common-subsequence length (DP). Accepts sequences."""
    a, b = list(a), list(b)
    if not a or not b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for ca in a:
        cur = [0] * (len(b) + 1)
        for j, cb in enumerate(b, 1):
            if ca == cb:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = prev[j] if prev[j] >= cur[j - 1] else cur[j - 1]
        prev = cur
    return prev[-1]


def lcs_ratio_exact(a, b, unit: str = LCS_UNIT_DEFAULT,
                    cap: int | None = LCS_CAP_DEFAULT,
                    min_ratio: float = 0.0):
    """sigma2 = |LCS(a,b)| / min(|a|,|b|).

    The paper (arXiv:2604.23374 section 4.2) does not state the sequence unit.
    Character-level exact LCS is the default: the propagation evidence is
    copied string fragments (payloads, URLs, commands), and on our corpora the
    word-level reading is not discriminative (MG AUROC 0.42 vs 0.79 char).
    Known property of the char reading: short unrelated strings can share a
    high subsequence ratio (e.g. 'echo hello' vs 'sunny weather tomorrow' is
    0.5), so unit='word' is kept as a sensitivity run.

    Exactness-preserving accelerations for long traces:
      - equal / substring fast paths;
      - common prefix+suffix trimming;
      - histogram upper bound: LCS <= sum_c min(count_a(c), count_b(c)); if
        bound/min(len) < min_ratio the pair cannot reach min_ratio and 0.0 is
        returned (only valid when callers pass their firing threshold).
    """
    key = (unit, cap, min_ratio, a, b)
    cached = _LCS_CACHE.get(key)
    if cached is not None:
        return cached
    a, b = str(a), str(b)
    if cap is not None:
        a, b = a[:cap], b[:cap]
    if unit == 'char':
        seq_a, seq_b = a, b
    elif unit == 'word':
        seq_a, seq_b = _tokens(a), _tokens(b)
    else:
        raise ValueError(unit)
    out = _lcs_ratio_seq(seq_a, seq_b, min_ratio)
    if len(_LCS_CACHE) < 200000:
        _LCS_CACHE[key] = out
    return out


def _lcs_ratio_seq(seq_a, seq_b, min_ratio=0.0):
    if not seq_a or not seq_b:
        return 0.0
    if seq_a == seq_b:
        return 1.0
    la, lb = len(seq_a), len(seq_b)
    if la > lb:
        seq_a, seq_b, la, lb = seq_b, seq_a, lb, la
    sa = ''.join(seq_a) if isinstance(seq_a, str) else None
    sb = ''.join(seq_b) if isinstance(seq_b, str) else None
    if sa is not None and sa in sb:
        return 1.0
    if min_ratio > 0:
        from collections import Counter
        ca, cb = Counter(seq_a), Counter(seq_b)
        bound = sum(min(ca[k], cb[k]) for k in ca.keys() & cb.keys())
        if bound / la < min_ratio:
            return 0.0
    if sa is not None and sb is not None:
        pre = 0
        max_pre = min(la, lb) - 1
        while pre < max_pre and sa[pre] == sb[pre]:
            pre += 1
        suf = 0
        while (suf < max_pre - pre and sa[la - 1 - suf] == sb[lb - 1 - suf]):
            suf += 1
        if pre or suf:
            inner = lcs_length(sa[pre:la - suf], sb[pre:lb - suf])
            return (pre + suf + inner) / la
    return lcs_length(seq_a, seq_b) / la


_LCS_CACHE = {}


def _recursive_string_values(obj):
    out = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_recursive_string_values(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_recursive_string_values(v))
    return out


def sink_arg_values(node):
    """Per-argument sink text: parsed JSON argument values when the trace uses
    the tool_name/arguments/response schema, else one text blob."""
    text = node_text(node)
    p = parse_fields(text)
    if p is None:
        values = [strip_schema(text)]
    else:
        raw = p.get('args') or ''
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            parsed = None
        if parsed is not None and not isinstance(parsed, (str, int, float)):
            values = [strip_schema(v) for v in _recursive_string_values(parsed)]
        else:
            values = [strip_schema(raw)]
    return [v for v in values if v and v.strip()]


def source_text(node):
    return compare_texts(node)[0]


def sentence_chunks(text):
    import re
    parts = re.split(r'(?<=[.!?\n])\s+|\n+', str(text))
    return [p.strip() for p in parts if len(p.strip()) >= 8]


def tier2_max(src, sink_values, unit: str = LCS_UNIT_DEFAULT):
    best = 0.0
    for v in sink_values:
        r = lcs_ratio_exact(src, v, unit=unit, min_ratio=THETA_STR)
        if r > best:
            best = r
    return best


def tier3_max(src, sink_values, model, truncation_cap=None):
    sv = embed(src if truncation_cap is None else src[:truncation_cap], model)
    best = 0.0
    for v in sink_values:
        tv = embed(v if truncation_cap is None else v[:truncation_cap], model)
        s = max(0.0, cosine(sv, tv))
        if s > best:
            best = s
    return best


def tier4_multi_fragment(src, sink_values, model):
    """max chunk similarity over all chunks; coverage = fraction of source
    characters in chunks whose similarity >= theta_sem (paper: coverage of the
    source by qualifying chunks, k unspecified)."""
    chunks = sentence_chunks(src)
    if not chunks:
        return 0.0, 0.0
    best = 0.0
    covered = 0
    for ch in chunks:
        cv = embed(ch, model)
        for v in sink_values:
            s = max(0.0, cosine(cv, embed(v, model)))
            if s > best:
                best = s
            if s >= THETA_SEM:
                covered += len(ch)
                break
    coverage = covered / max(len(str(src)), 1)
    if best >= THETA_SEM and coverage >= THETA_COV:
        return best, coverage
    return 0.0, coverage


def rehydration_links(graph, model, similarity_threshold=THETA_SEM_RAG):
    """read index -> write index when the read content matches a preceding
    write of the same stored value (embedding proxy for the paper's persisted
    _nt_taint metadata). An explicit write->read path is used only when it runs
    through the memory edge vocabulary (writes_to / reads_from); generic
    data_flow chains are not memory identity."""
    nodes = graph.get('nodes', [])
    writes = [(i, n) for i, n in enumerate(nodes) if n.get('type') == MEMORY_WRITE]
    reads = [(i, n) for i, n in enumerate(nodes) if n.get('type') == MEMORY_READ]
    links = {}
    if not reads or not writes:
        return links
    adj = defaultdict(list)
    for e in graph.get('edges', []):
        if e.get('type') in MEMORY_EDGES:
            adj[e['source']].append(e['target'])

    def reach_ids(start_id):
        seen, stack = {start_id}, [start_id]
        while stack:
            u = stack.pop()
            for v in adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    explicit = {}
    for j, rn in reads:
        for i, wn in writes:
            if i >= j:
                continue
            if rn['id'] in reach_ids(wn['id']):
                explicit[j] = i
                break
    for j, rn in reads:
        if j in explicit:
            links[j] = explicit[j]
            continue
        rtext = node_text(rn)[:400]
        rv = embed(rtext, model)
        best, best_i = 0.0, None
        for i, wn in writes:
            if i >= j:
                continue
            s = cosine(embed(node_text(wn)[:400], model), rv)
            if s > best:
                best, best_i = s, i
        if best >= similarity_threshold:
            links[j] = best_i
    return links


def reachable_with_rehydration(graph, src_idx, rehyd):
    """Forward reachability over provenance edges plus write->read links."""
    nodes = graph.get('nodes', [])
    adj = defaultdict(list)
    for e in graph.get('edges', []):
        if e['type'] in PROVENANCE_EDGES:
            adj[e['source']].append(e['target'])
    by_id = {n['id']: i for i, n in enumerate(nodes)}
    index_links = defaultdict(list)
    for r, w in rehyd.items():
        index_links[w].append(r)
    id_links = defaultdict(list)
    for w, rs in index_links.items():
        wn = nodes[w]
        for r in rs:
            id_links[wn['id']].append(nodes[r]['id'])
    seen, stack = {src_idx}, [src_idx]
    while stack:
        u = stack.pop()
        nbrs = list(adj.get(nodes[u]['id'], [])) + list(id_links.get(nodes[u]['id'], []))
        for v in nbrs:
            vi = by_id.get(v)
            if vi is not None and vi not in seen:
                seen.add(vi)
                stack.append(vi)
    return seen


def causal_status(graph):
    has_user = any(n.get('type') in USER_INPUT_TYPES for n in graph.get('nodes', []))
    return {
        'status': 'not_applicable',
        'reason': ('Sink-Driven Causal Analyzer needs a schema-compatible '
                   'neutralized context and one LLM judge call per (sink, '
                   'lineage) (paper 4.3); re-hosted graphs contain neither '
                   'full context nor sink argument objects'),
        'user_text_node_type_present': bool(has_user),
    }


def causal_proxy_ablation(graph, src_texts, sink_idx, model):
    """The old keyword proxy, kept only for ablation runs under an explicit
    flag; never reported as the paper's causal analyzer."""
    nodes = graph.get('nodes', [])
    sink = nodes[sink_idx]
    sink_values = sink_arg_values(sink) or [node_text(sink)]
    sv = embed(' '.join(sink_values)[:400], model)
    for txt in src_texts:
        low = str(txt).lower()
        if any(c in low for c in CAUSAL_CUES):
            return 1.0
    return 0.0


def audit_fixed(graph, model, restricted=True, use_causal_proxy=False,
                lcs_cap=LCS_CAP_DEFAULT, lcs_unit: str = LCS_UNIT_DEFAULT):
    """Corrected audit. Returns score, stage, evidence, coverage diagnostics."""
    nodes = graph.get('nodes', [])
    rehyd = rehydration_links(graph, model)
    src = [(i, n) for i, n in enumerate(nodes) if n.get('type') in SOURCE_TYPES]
    sinks = [(i, n) for i, n in enumerate(nodes) if n.get('type') in SINK_TYPES]
    reach_cache = {}
    best = (0.0, 'none', None, None)
    for si, sn in src:
        stext = source_text(sn)
        if not stext:
            continue
        if restricted:
            if si not in reach_cache:
                reach_cache[si] = reachable_with_rehydration(graph, si, rehyd)
            reach = reach_cache[si]
        for ti, tn in sinks:
            if restricted:
                if ti not in reach:
                    continue
            elif ti <= si:
                continue
            values = sink_arg_values(tn)
            if not values:
                continue
            r2 = tier2_max(stext, values, unit=lcs_unit)
            if r2 >= THETA_STR:
                if r2 > best[0]:
                    best = (min(1.0, r2), 'LCS', (si, ti), {'coverage': None})
                continue
            via_mem = rehyd.get(ti) is not None
            thr = THETA_SEM_RAG if via_mem else THETA_SEM
            sem = tier3_max(stext, values, model)
            if sem >= thr:
                if sem > best[0]:
                    best = (min(1.0, sem), 'SEM', (si, ti),
                            {'via_mem': via_mem, 'threshold': thr})
                continue
            mf, cov = tier4_multi_fragment(stext, values, model)
            if mf > 0 and mf > best[0]:
                best = (mf, 'MULTI_FRAG', (si, ti), {'coverage': cov})
    result = {
        'score': best[0], 'stage': best[1], 'evidence': best[2],
        'detail': best[3], 'n_sources': len(src), 'n_sinks': len(sinks),
        'n_rehydrated_reads': len(rehyd),
        'lcs_causal_status': causal_status(graph),
    }
    if use_causal_proxy and best[0] == 0.0 and sinks:
        for ti, tn in sinks:
            src_texts = [node_text(n) for i, n in src if i < ti]
            if not src_texts:
                continue
            p = causal_proxy_ablation(graph, src_texts, ti, model)
            if p > result['score']:
                result.update({'score': p, 'stage': 'CAUSAL_PROXY_ABLATION',
                               'evidence': (None, ti)})
    return result
