#!/usr/bin/env python3
"""Input-retention audit for the Task 1 relation pilot (GPT check 1).

For every pair in a generated split, verifies at the input level that the
decision-relevant information survives every encoder window:

  * graph arm      : the single counterfactual-differing edge is present in
                     both sides' edge sets; node texts <= 192 chars
                     (MAX_NODE_CHARS); edge rationales <= 160 chars.
  * text arm       : the full serialized relation text fits the token budget
                     (no truncation), and the differing edge line is present
                     in both sides' serialized text.
  * both           : the two sides' node texts / chronological text are
                     byte-identical, as declared by the generator contract.

Outputs a per-pair JSON plus a summary; fails loudly on any violation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_FINAL_FIVE = Path(__file__).resolve().parents[2] / "repro" / "final_five_tables"
if str(_REPO_FINAL_FIVE) not in sys.path:
    sys.path.insert(0, str(_REPO_FINAL_FIVE))

from train_gnn_ablate import MAX_EDGE_CHARS, MAX_NODE_CHARS  # noqa: E402

from graphbuild import build_graph  # noqa: E402
from models_text import MAX_TOKENS, build_vocab, encode_text_audit, tokenize  # noqa: E402
from serialize import alias_map, chrono_text, relation_text  # noqa: E402


def edge_sig(graph):
    return sorted((e["source"], e["target"], e["type"]) for e in graph["edges"])


def audit_pair(harm, safe, vocab):
    ga, gb = build_graph(harm), build_graph(safe)
    sa, sb = set(edge_sig(ga)), set(edge_sig(gb))
    diff_a = sorted(sa - sb)
    diff_b = sorted(sb - sa)
    text_a, text_b = relation_text(ga), relation_text(gb)
    _, n_tok_a, trunc_a = encode_text_audit(text_a, vocab)
    _, n_tok_b, trunc_b = encode_text_audit(text_b, vocab)
    aliases_a, aliases_b = alias_map(ga), alias_map(gb)
    missing_lines = []
    for s, t, etype in diff_a:
        line = f"edge {aliases_a[s]} {etype} {aliases_a[t]}"
        if line not in text_a:
            missing_lines.append(line)
    for s, t, etype in diff_b:
        line = f"edge {aliases_b[s]} {etype} {aliases_b[t]}"
        if line not in text_b:
            missing_lines.append(line)
    node_over = [n["id"] for g in (ga, gb) for n in g["nodes"]
                 if len((n.get("features", {}).get("subtask") or "")) > MAX_NODE_CHARS]
    edge_over = [e["type"] for g in (ga, gb) for e in g["edges"]
                 if len(e.get("rationale", "")) > MAX_EDGE_CHARS]
    return {
        "pair_id": harm["pair_id"],
        "kind": harm["kind"],
        "n_differing_edges": len(diff_a),
        "decisive_edge_present_in_graph": len(diff_a) == 1 and len(diff_b) == 1,
        "decisive_edge_in_text": not missing_lines,
        "text_tokens_harm": n_tok_a,
        "text_tokens_safe": n_tok_b,
        "text_truncated": bool(trunc_a or trunc_b),
        "chrono_identical": chrono_text(ga) == chrono_text(gb),
        "node_text_within_window": not node_over,
        "edge_rationale_within_window": not edge_over,
        "missing_lines": missing_lines,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs-dir", required=True)
    ap.add_argument("--split", default="confirm")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pairs_dir = Path(args.pairs_dir)
    rows = [json.loads(l) for l in open(pairs_dir / f"pairs_{args.split}.jsonl")
            if l.strip()]
    by_pair = {}
    for r in rows:
        by_pair.setdefault(r["pair_id"], {})[r["variant"]] = r
    train_rows = [json.loads(l) for l in open(pairs_dir / "pairs_train.jsonl")
                  if l.strip()]
    vocab = build_vocab([relation_text(build_graph(r)) for r in train_rows])
    results = []
    for pid in sorted(by_pair):
        pair = by_pair[pid]
        if "harm" not in pair or "safe" not in pair:
            continue
        results.append(audit_pair(pair["harm"], pair["safe"], vocab))
    summary = {
        "split": args.split,
        "n_pairs": len(results),
        "max_text_tokens": max((r["text_tokens_harm"] for r in results), default=0),
        "token_budget": MAX_TOKENS,
        "n_text_truncated": sum(r["text_truncated"] for r in results),
        "n_decisive_edge_missing_in_text": sum(
            not r["decisive_edge_in_text"] for r in results),
        "n_decisive_edge_missing_in_graph": sum(
            not r["decisive_edge_present_in_graph"] for r in results),
        "n_chrono_mismatch": sum(not r["chrono_identical"] for r in results),
        "n_node_over_window": sum(not r["node_text_within_window"] for r in results),
        "n_edge_over_window": sum(not r["edge_rationale_within_window"] for r in results),
        "pass": all(r["decisive_edge_in_text"] and r["decisive_edge_present_in_graph"]
                    and not r["text_truncated"] and r["chrono_identical"]
                    and r["node_text_within_window"]
                    and r["edge_rationale_within_window"] for r in results),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({"summary": summary, "pairs": results}, f, indent=1,
                  sort_keys=True)
    print(json.dumps(summary, indent=1, sort_keys=True))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
