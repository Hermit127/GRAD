#!/usr/bin/env python3
"""W1 fair comparison: build the unified decision-time (obs1) Decomp corpus.

Protocol (frozen in LOGS/w1_fair_comparison/protocol.md):
  * decision time = immediately before executing the harmful step;
  * the proposed harmful action IS visible (hook sees the tool call);
  * no step after the harmful step is visible;
  * benign traces are unchanged (no harmful step exists);
  * text is already capped at 200 chars/step by the source converter and every
    arm (GRAD / Text-only / Temporal / Sequential Monitor) sees the same strings.

Source (unchanged split, legacy test labelled as revised comparison):
  code/repro/out/unified_gnn/fix2_splits/decomp/fold0/{train,dev,test}.jsonl
Output:
  LOGS/w1_fair_comparison/data/obs1/{train,dev,test}.jsonl
  LOGS/w1_fair_comparison/data/obs1/manifest.json
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "code/repro/out/unified_gnn/fix2_splits/decomp/fold0"
OUT = REPO / "LOGS/w1_fair_comparison/data/obs1"
PROTOCOL = "obs1_pre_harm_proposal_visible"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def truncate(g: dict) -> tuple[dict, dict]:
    nodes = g.get("nodes", [])
    n_total = len(nodes)
    if int(g.get("label", 0)) == 1:
        keep_until = None
        for i, n in enumerate(nodes):
            if n.get("features", {}).get("harm_step"):
                keep_until = i + 1
                break
        if keep_until is None:
            keep_until = n_total
    else:
        keep_until = n_total
    kept = nodes[:keep_until]
    kept_ids = {n["id"] for n in kept}
    edges = [e for e in g.get("edges", []) if e["source"] in kept_ids and e["target"] in kept_ids]
    out = {
        "task_id": g["task_id"],
        "label": int(g["label"]),
        "harm_index": g.get("harm_index"),
        "num_nodes": len(kept),
        "num_edges": len(edges),
        "nodes": kept,
        "edges": edges,
        "sample_id": g.get("sample_id"),
        "group": g.get("group"),
        "w1_protocol": PROTOCOL,
        "w1_n_nodes_total": n_total,
        "w1_proposal_visible": True,
    }
    meta = {
        "task_id": g["task_id"],
        "group": g.get("group"),
        "label": int(g["label"]),
        "n_nodes_total": n_total,
        "n_nodes_kept": len(kept),
        "truncated": n_total != len(kept),
        "n_edges_total": len(g.get("edges", [])),
        "n_edges_kept": len(edges),
    }
    return out, meta


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": PROTOCOL,
        "decision_time": "immediately before execution of the harmful step",
        "proposed_action_visible": True,
        "post_harm_steps_visible": False,
        "benign_traces": "unchanged (full trace; no harmful step exists)",
        "text_cap": "200 chars per step as stored in the source converter; identical for all arms",
        "source_dir": str(SRC.relative_to(REPO)),
        "source_sha256": {},
        "splits": {},
    }
    for role in ("train", "dev", "test"):
        sp = SRC / f"{role}.jsonl"
        manifest["source_sha256"][role] = sha256_file(sp)
        rows, metas = [], []
        with open(sp) as f:
            for line in f:
                if line.strip():
                    g = json.loads(line)
                    o, m = truncate(g)
                    rows.append(o)
                    metas.append(m)
        body = "\n".join(json.dumps(r, ensure_ascii=False) for r in rows)
        op = OUT / f"{role}.jsonl"
        op.write_text(body + "\n")
        (OUT / f"{role}.w1meta.jsonl").write_text(
            "\n".join(json.dumps(m, ensure_ascii=False) for m in metas) + "\n")
        attacks = [m for m in metas if m["label"] == 1]
        manifest["splits"][role] = {
            "n": len(rows),
            "n_pos": sum(1 for r in rows if r["label"] == 1),
            "n_attacks_truncated": sum(1 for m in attacks if m["truncated"]),
            "mean_nodes_before": round(sum(m["n_nodes_total"] for m in metas) / len(metas), 3),
            "mean_nodes_after": round(sum(m["n_nodes_kept"] for m in metas) / len(metas), 3),
            "sha256": sha256_file(op),
            "w1meta_sha256": sha256_file(OUT / f"{role}.w1meta.jsonl"),
        }
        print(role, manifest["splits"][role], flush=True)
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
