#!/usr/bin/env python3
"""A-line round 2: convert validated raw records to obs1 graphs.

Uses the frozen converter `code/decomposed_to_graph.py` and the W1 obs1
truncation (prefix through the proposed harmful step; proposal visible; later
steps dropped; benign unchanged), identical to `build_obs1.py` / W5.

Usage:
  python build_round2.py --validated LOGS/core_review_resolution/data/round2/validated \
      --out LOGS/core_review_resolution/data/round2/confirm
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
import decomposed_to_graph as conv  # noqa: E402


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def truncate(g: dict) -> dict:
    nodes = g.get("nodes", [])
    if int(g.get("label", 0)) == 1:
        keep = len(nodes)
        for i, n in enumerate(nodes):
            if n.get("features", {}).get("harm_step"):
                keep = i + 1
                break
    else:
        keep = len(nodes)
    kept = nodes[:keep]
    ids = {n["id"] for n in kept}
    edges = [e for e in g.get("edges", []) if e["source"] in ids and e["target"] in ids]
    return {"task_id": g["task_id"], "label": int(g["label"]),
            "harm_index": g.get("harm_index"), "num_nodes": len(kept),
            "num_edges": len(edges), "nodes": kept, "edges": edges,
            "sample_id": g.get("task_id"), "group": g.get("group"),
            "w1_protocol": "obs1_pre_harm_proposal_visible",
            "w1_n_nodes_total": len(nodes), "w1_proposal_visible": True}


def visible_text(g: dict) -> str:
    return " | ".join(n["features"].get("subtask", "") for n in g["nodes"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validated", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    va = Path(args.validated)
    if not va.is_absolute():
        va = REPO / va
    out = Path(args.out)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in open(va / "valid_cases.jsonl") if l.strip()]
    graphs, meta = [], []
    for rec in records:
        g = conv.task_to_dict(rec)
        if g is None:
            raise SystemExit(f"converter returned None for {rec['id']}")
        g = truncate(g)
        g["group"] = rec["group"]
        graphs.append(g)
        meta.append({"task_id": g["task_id"], "group": rec["group"],
                     "label": g["label"], "variant": rec.get("variant"),
                     "visible_sha256": hashlib.sha256(
                         visible_text(g).encode()).hexdigest(),
                     "n_nodes_kept": g["num_nodes"],
                     "n_nodes_total": g["w1_n_nodes_total"]})
    with open(out / "graphs.jsonl", "w") as f:
        for g in graphs:
            f.write(json.dumps(g, ensure_ascii=False) + "\n")
    with open(out / "meta.jsonl", "w") as f:
        for m in meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    groups = sorted(set(m["group"] for m in meta))
    validity = json.loads((va / "case_validity.json").read_text())
    design = {
        "protocol": "a_round2_frozen_new_generated_case_validation",
        "label": ("frozen new generated case validation (LLM-generated per the "
                  "mirrored Decomp pipeline); NOT real natural data; offline "
                  "identification scoring, no attack-blocking claim"),
        "generator": "deepseek-flash", "judge": "glm-5.3-flash",
        "n_base_cases_generated_ok": validity["n_generated_ok"],
        "n_base_cases_included": validity["n_included"],
        "n_base_cases_excluded": validity["n_excluded"],
        "n_rows": len(graphs),
        "n_attack_rows": sum(1 for g in graphs if g["label"] == 1),
        "n_benign_rows": sum(1 for g in graphs if g["label"] == 0),
        "n_groups": len(groups),
        "attempts_per_case": 1,
        "exclusion_rule_source": str(va / "case_validity.json"),
        "text_cap": "200 chars/step (frozen converter)",
        "observation_cutoff": "prefix through the proposed harmful step; proposal visible; later steps dropped",
        "benign_length_matching": "benign decomposition truncated to the pre-harm length h of its base case",
        "graphs_sha256": sha256_file(out / "graphs.jsonl"),
        "meta_sha256": sha256_file(out / "meta.jsonl"),
        "valid_cases_sha256": validity["valid_cases_sha256"],
    }
    (out / "design.json").write_text(json.dumps(design, ensure_ascii=False, indent=1))
    print(json.dumps(design, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
