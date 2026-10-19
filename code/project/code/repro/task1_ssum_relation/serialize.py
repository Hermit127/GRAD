#!/usr/bin/env python3
"""Text renderings for the Task 1 pilot.

The strong text control receives exactly the same node texts and the same
relation set (type + endpoints) as the graph model, in a serialized form.
Node identities are canonical aliases (v0..vk in turn order) so that the
serialized text is byte-identical across counterfactual sides and the text
model is not required to memorize random identifiers; the graph model sees the
same anonymized structure. The chronological control receives the same node
texts and observable attributes without any relation information.
"""
from __future__ import annotations


def ordered_nodes(graph):
    return sorted(graph["nodes"], key=lambda n: (n["turn"], n["id"]))


def alias_map(graph):
    return {n["id"]: f"v{i}" for i, n in enumerate(ordered_nodes(graph))}


def node_line(node, alias):
    text = (node.get("features", {}) or {}).get("subtask", "") or ""
    return f"- {alias} [{node['type']}] {text}".rstrip()


def chrono_text(graph, with_ids=True):
    aliases = alias_map(graph)
    return "\n".join(node_line(n, aliases[n["id"]]) for n in ordered_nodes(graph))


def relation_text(graph, include_types=True, with_ids=True):
    aliases = alias_map(graph)
    lines = [chrono_text(graph, with_ids), "relations:"]
    for e in graph["edges"]:
        if include_types:
            lines.append(f"edge {aliases[e['source']]} {e['type']} {aliases[e['target']]}")
        else:
            lines.append(f"edge {aliases[e['source']]} connection {aliases[e['target']]}")
    return "\n".join(lines)


def graph_text(graph, include_types=True, with_ids=True):
    return relation_text(graph, include_types=include_types, with_ids=with_ids)
