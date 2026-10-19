#!/usr/bin/env python
"""NeuroTaint v2 offline auditor (faithful core, zero API).

Mechanism per arXiv:2604.23374:
  - DCPG: step nodes with taint lineage; memory_write -> memory_read
    rehydration; optional cross-session persistence (session ids if present).
  - Hybrid Semantic Tracker: Tier2 LCS >= THETA_STR; Tier3 embedding cosine
    >= THETA_SEM (>= THETA_SEM_RAG when reached through memory);
    Tier4 sentence-chunk max similarity >= THETA_SEM with coverage >= THETA_COV.
    Tier1 canary is not available offline (no captured canaries) -> n/a.
  - Sink-Driven Causal Analyzer: deterministic proxy (documented approximation).

Returns a per-trace score in [0,1] and the stage that produced it.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from difflib import SequenceMatcher

from policy import (CAUSAL_CUES, MEMORY_READ, MEMORY_WRITE,
                    PROVENANCE_EDGES, SINK_TYPES, SOURCE_TYPES, THETA_COV,
                    THETA_SEM, THETA_SEM_RAG, THETA_STR, node_text,
                    session_of)

_MODEL = None

_FIELD_RE = re.compile(
    r'tool_name:\s*(?P<tool>.*?)\s*arguments:\s*(?P<args>.*?)'
    r'\s*response:\s*(?P<resp>.*)$', re.I | re.S)


def parse_fields(text):
    """Our re-hosted traces store 'tool_name: X arguments: Y response: Z'.
    The paper compares source *content* with sink *arguments*; comparing the
    shared schema keys is a boilerplate artifact, so parse and keep values."""
    m = _FIELD_RE.search(str(text))
    if not m:
        return None
    return {k: (v or '').strip() for k, v in m.groupdict().items()}


_JSON_KEY_RE = re.compile(r'"(?:[A-Za-z_][A-Za-z0-9_ ]*)"\s*:')
_STRUCT_CHARS_RE = re.compile(r'[{}\[\]",\\:]')


def strip_schema(text):
    """Remove JSON keys and structural punctuation so comparisons are on the
    content values, not on shared schema/boilerplate tokens."""
    t = _JSON_KEY_RE.sub(' ', str(text))
    t = _STRUCT_CHARS_RE.sub(' ', t)
    return re.sub(r'\s+', ' ', t).strip()


def compare_texts(node):
    """Text used for provenance comparison: values only when the trace uses
    the tool_name/arguments/response schema, otherwise raw text; JSON keys
    and structural characters are stripped in all cases."""
    t = node_text(node)
    p = parse_fields(t)
    if p is None:
        return strip_schema(t), t
    src = strip_schema(p['resp'] + ' ' + p['args'])
    return (src or strip_schema(t)), p


def sink_arg_text(node):
    t = node_text(node)
    p = parse_fields(t)
    if p is None:
        return strip_schema(t)
    return strip_schema(p['args'])


def get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer('all-MiniLM-L6-v2')
    return _MODEL


_EMB_CACHE = {}


def embed(text, model):
    key = (id(model), text)
    v = _EMB_CACHE.get(key)
    if v is None:
        v = model.encode([text], normalize_embeddings=True,
                         show_progress_bar=False)[0]
        _EMB_CACHE[key] = v
    return v


def cosine(a, b):
    return float(sum(x * y for x, y in zip(a, b)))


def lcs_ratio(a, b, cap=300):
    a, b = str(a)[:cap], str(b)[:cap]
    if not a or not b:
        return 0.0
    m = SequenceMatcher(None, a, b, autojunk=False)
    total = sum(bl.size for bl in m.get_matching_blocks())
    return total / min(len(a), len(b))


def sentences(text):
    parts = re.split(r'(?<=[.!?])\s+|\n+', str(text))
    return [p.strip() for p in parts if len(p.strip()) >= 8]


def multi_fragment_sim(src, sink, model):
    best, cov = 0.0, 0.0
    sv = embed(sink, model)
    chunks = sentences(src)
    if not chunks:
        return 0.0
    for ch in chunks:
        s = max(0.0, cosine(embed(ch, model), sv))
        if s > best:
            best, cov = s, len(ch) / max(len(str(src)), 1)
    if best >= THETA_SEM and cov >= THETA_COV:
        return best
    return 0.0


def trace_order(graph):
    nodes = graph.get('nodes', [])
    sess = [session_of(n) for n in nodes]
    if any(s is not None for s in sess):
        return list(range(len(nodes)))
    return list(range(len(nodes)))


def memory_rehydration(graph, model):
    """write->read taint transfer: map read node -> write node when the read
    content is semantically close to a preceding write (>= THETA_SEM_RAG)."""
    nodes = graph.get('nodes', [])
    writes = [(i, n) for i, n in enumerate(nodes)
              if n.get('type') == MEMORY_WRITE]
    pairs = {}
    for j, n in enumerate(nodes):
        if n.get('type') != MEMORY_READ:
            continue
        rv = embed(node_text(n)[:400], model)
        best, best_i = 0.0, None
        for i, w in writes:
            if i >= j:
                continue
            s = cosine(embed(node_text(w)[:400], model), rv)
            if s > best:
                best, best_i = s, i
        if best >= THETA_SEM_RAG:
            pairs[j] = best_i
    return pairs


def reachable_from(graph, src_idx):
    """Provenance reachability (DCPG-restricted variant)."""
    nodes = graph.get('nodes', [])
    adj = defaultdict(list)
    for e in graph.get('edges', []):
        if e['type'] in PROVENANCE_EDGES:
            adj[e['source']].append(e['target'])
    by_id = {n['id']: i for i, n in enumerate(nodes)}
    seen, stack = {src_idx}, [src_idx]
    while stack:
        u = stack.pop()
        for v in adj.get(nodes[u]['id'], []):
            vi = by_id.get(v)
            if vi is not None and vi not in seen:
                seen.add(vi)
                stack.append(vi)
    return seen


def causal_proxy(graph, src_texts, sink_idx, user_texts, model):
    """Deterministic approximation of the paper's counterfactual analyzer:
    privileged sink + preceding source + action not derivable from user task +
    source contains imperative/redirect cues."""
    nodes = graph.get('nodes', [])
    sink = nodes[sink_idx]
    sink_text = sink_arg_text(sink) or node_text(sink)
    sv = embed(sink_text[:400], model)
    for ut in user_texts:
        if cosine(embed(ut[:400], model), sv) >= THETA_SEM:
            return 0.0
    cue = False
    for txt in src_texts:
        low = str(txt).lower()
        if any(c in low for c in CAUSAL_CUES):
            cue = True
            break
    return 1.0 if cue else 0.0


def audit(graph, model, restricted=False, use_causal=True):
    """Return {'score', 'stage', 'evidence'} for one trace."""
    nodes = graph.get('nodes', [])
    order = trace_order(graph)
    user_texts = [node_text(nodes[0])] if nodes else []
    rehyd = memory_rehydration(graph, model)
    src = [(i, n) for i, n in enumerate(nodes)
           if n.get('type') in SOURCE_TYPES]
    sinks = [(i, n) for i, n in enumerate(nodes)
             if n.get('type') in SINK_TYPES]
    best = (0.0, 'none', None)
    for si, sn in src:
        stext, _ = compare_texts(sn)
        stv = embed(stext[:400], model)
        reach = reachable_from(graph, si)
        for ti, tn in sinks:
            if restricted and ti not in reach and rehyd.get(ti) != si:
                continue
            if not restricted and ti <= si:
                continue
            ttext = sink_arg_text(tn)
            if not ttext or min(len(stext), len(ttext)) < 24:
                continue
            # Tier 2 LCS
            r2 = lcs_ratio(stext, ttext)
            if r2 >= THETA_STR:
                if r2 > best[0]:
                    best = (min(1.0, r2), 'LCS', (si, ti))
                continue
            # Tier 3/4 semantic (rag threshold if reached via memory)
            tv = embed(ttext[:400], model)
            sem = max(0.0, cosine(stv, tv))
            via_mem = rehyd.get(ti) == si
            thr = THETA_SEM_RAG if via_mem else THETA_SEM
            if sem >= thr:
                if sem > best[0]:
                    best = (min(1.0, sem), 'SEM', (si, ti))
                continue
            mf = multi_fragment_sim(stext, ttext, model)
            if mf > 0 and mf > best[0]:
                best = (mf, 'MULTI_FRAG', (si, ti))
    if best[0] == 0.0 and use_causal and sinks:
        # causal proxy against all source texts preceding each sink
        for ti, tn in sinks:
            src_texts = [node_text(n) for i, n in src if i < ti]
            if not src_texts:
                continue
            p = causal_proxy(graph, src_texts, ti, user_texts, model)
            if p > best[0]:
                best = (p, 'CAUSAL_PROXY', (None, ti))
    return {'score': best[0], 'stage': best[1], 'evidence': best[2],
            'n_sources': len(src), 'n_sinks': len(sinks)}
