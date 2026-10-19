#!/usr/bin/env python3
"""AgentArmor (arXiv:2508.01249) faithful reproduction on behavior-graph data.

Method: PDG-based type system. Type = {security_type, rule_type};
security_type = {confidentiality, integrity} in {low, mid, high}.
- Initial types from registry (paper does not publish the table; we design per
  security common sense and note it): untrusted retrievals (web/file/memory/
  db) = LOW integrity; dangerous exec nodes start HIGH (trusted use); rest MID.
- Lattice propagation along data_flow edges: single-source inherits; multi-source
  join = confidentiality takes most restrictive (HIGH), integrity takes least
  restrictive (LOW) [paper Sec. V-B].
- Inspector rule (paper's only concrete rule): Forbid{Int < Mid} on action nodes
  (dangerous operations); plus lattice check that low-integrity input does not
  influence nodes (captured by the join itself).
"""
import json, random, sys
from collections import defaultdict

LOW, MID, HIGH = 0, 1, 2
INT_MAP = {0: 'L', 1: 'M', 2: 'H'}

UNTRUSTED_SRC = {'web_search', 'web_download', 'information_search', 'file_read',
                 'memory_read', 'database_query', 'account_access', 'data_aggregate'}
DANGEROUS_OPS = {'shell_exec', 'code_exec', 'email_send', 'message_send',
                 'social_media', 'file_delete', 'web_request', 'document_create',
                 'prompt_inject', 'system_config'}
NO_DANGER_SET = {'shell_exec', 'code_exec', 'memory_write'}  # double-drop protocol


def join_integrity(a, b):
    return min(a, b)  # least restrictive (LOW wins)


def agentarmor_graph(g, drop_nodes=()):
    """Returns 1 if AgentArmor blocks the graph (any dangerous op with Int < MID)."""
    nodes = [n for n in g['nodes'] if n['type'] not in drop_nodes]
    if not nodes:
        return 0
    id2i = {n['id']: i for i, n in enumerate(nodes)}
    n = len(nodes)
    # initial types
    integrity = [MID] * n
    for i, nd in enumerate(nodes):
        if nd['type'] in UNTRUSTED_SRC:
            integrity[i] = LOW
        elif nd['type'] in ('tool_execute', 'shell_exec', 'code_exec', 'file_write',
                            'memory_write', 'agent_spawn', 'version_control'):
            integrity[i] = HIGH
    # propagate along data_flow edges (source -> target), fixpoint
    changed = True
    while changed:
        changed = False
        for e in g.get('edges', []):
            s, t = id2i.get(e['source']), id2i.get(e['target'])
            if s is None or t is None:
                continue
            if e.get('type', 'data_flow') not in ('data_flow', 'writes_to', 'reads_from'):
                continue
            nj = join_integrity(integrity[s], integrity[t])
            if nj < integrity[t]:
                integrity[t] = nj
                changed = True
    # inspector: Forbid{Int < Mid} on dangerous operation nodes
    for i, nd in enumerate(nodes):
        if nd['type'] in DANGEROUS_OPS and integrity[i] < MID:
            return 1
    return 0


def run(name, te, drop_nodes=()):
    preds = [agentarmor_graph(g, drop_nodes) for g in te]
    labels = [g['label'] for g in te]
    tp = sum(1 for l, p in zip(labels, preds) if l and p)
    fp = sum(1 for l, p in zip(labels, preds) if not l and p)
    fn = sum(1 for l, p in zip(labels, preds) if l and not p)
    tn = sum(1 for l, p in zip(labels, preds) if not l and not p)
    pre = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * pre * rec / (pre + rec) if pre + rec else 0
    print(f'{name:10s} F1={f1:.3f} P={pre:.3f} R={rec:.3f} (tp{tp}/fp{fp}/fn{fn}/tn{tn})')


def main():
    R = '.'
    # MG: cross-template holdout (same split as table D), double-drop protocol
    mg = [json.loads(l) for l in open(f'{R}/ref/dataset/memorygraft_augmented.jsonl')]
    def sid(t): return t.split('_v')[0]
    att, ben = defaultdict(list), defaultdict(list)
    for g in mg: (att if g['label'] else ben)[sid(g['task_id'])].append(g)
    as_, bs = sorted(att), sorted(ben)
    random.seed(42); random.shuffle(as_); random.shuffle(bs)
    mg_te = []
    for s in as_[8:]: mg_te += att[s]
    for s in bs[80:]: mg_te += ben[s]
    print(f'=== AgentArmor reproduction (n_test={len(mg_te)}) ===')
    run('MG (full)', mg_te)
    run('MG (drop2)', mg_te, NO_DANGER_SET)
    # Decomp
    de = [json.loads(l) for l in open(f'{R}/ref/dataset/decomp_graphs_test_mixed.jsonl')]
    run('Decomp', de)
    # MCP
    mc = [json.loads(l) for l in open(f'{R}/ref/dataset/mcp_safetybench_defense_matched_test.jsonl')]
    run('MCP', mc)
    # TS
    ts = [json.loads(l) for l in open(f'{R}/ref/dataset/toolshadow_preexec_graphs.jsonl')]
    run('TS', ts)
    # MINJA
    mj = json.load(open(f'{R}/ref/dataset/minja_real_traces.json'))
    mj_g = []
    for i, t in enumerate(mj):
        nodes, edges, prev = [], [], None
        for j, s in enumerate(t.get('steps', [])):
            nid = f'm{j}'
            nodes.append({'id': nid, 'type': s.get('type', 'tool_execute'), 'taint': 0,
                          'turn': j, 'features': {'subtask': str(s.get('action', ''))}})
            if prev: edges.append({'source': prev, 'target': nid, 'type': 'data_flow'})
            prev = nid
        mj_g.append({'task_id': f'minja_{i}', 'label': t['label'], 'nodes': nodes, 'edges': edges})
    run('MINJA', mj_g)


if __name__ == '__main__':
    main()
