#!/usr/bin/env python3
"""Real-trace provenance availability audit (GPT check 3).

Scans the existing corpora and reports, per corpus, which runtime provenance
facts are observable in the data at decision time and whether the current
graph construction retains them:

  Q1  which memory record was actually read        (record identity on reads)
  Q2  which source supplied a tool parameter        (argument provenance)
  Q3  which authorization supports the action       (authorization edges)

Verdicts: full / partial / none, with the observed field names and counts.
No files are modified; scanning is capped per corpus.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

Q1_KEYS = ("record_id", "memory_id", "mem_event_id", "read_record")
Q2_KEYS = ("tool_call_id", "span_id", "parent_id", "source_event_id",
           "record_index", "raw_payload", "arguments", "source_kind")
Q3_KEYS = ("authorization", "auth", "permission", "scope")
ALL_KEYS = Q1_KEYS + Q2_KEYS + Q3_KEYS


def graph_iter(path: Path, cap: int = 500):
    if not path.exists():
        return
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= cap:
                break
            if line.strip():
                yield json.loads(line)


def audit_graphs(paths, cap=500):
    stat = {
        "n_graphs": 0, "n_nodes": 0, "n_edges": 0,
        "node_types": Counter(), "edge_types": Counter(),
        "adjacent_edges": 0, "non_adjacent_edges": 0,
        "nodes_with_key": Counter(), "edge_endpoint_is_id_like": 0,
    }
    for p in paths:
        p = Path(p)
        if not p.exists():
            stat.setdefault("missing", []).append(str(p))
            continue
        for g in graph_iter(p, cap):
            stat["n_graphs"] += 1
            nodes = g.get("nodes", [])
            stat["n_nodes"] += len(nodes)
            turn = {n.get("id"): n.get("turn", i) for i, n in enumerate(nodes)}
            for n in nodes:
                stat["node_types"][str(n.get("type", ""))] += 1
                feats = n.get("features", {}) or {}
                for k in ALL_KEYS:
                    if k in feats:
                        stat["nodes_with_key"][k] += 1
            for e in g.get("edges", []):
                stat["n_edges"] += 1
                stat["edge_types"][str(e.get("type", ""))] += 1
                s, t = e.get("source"), e.get("target")
                if s in turn and t in turn:
                    if abs(turn[s] - turn[t]) <= 1:
                        stat["adjacent_edges"] += 1
                    else:
                        stat["non_adjacent_edges"] += 1
    stat["node_types"] = dict(stat["node_types"].most_common(12))
    stat["edge_types"] = dict(stat["edge_types"].most_common(12))
    stat["nodes_with_key"] = dict(stat["nodes_with_key"])
    return stat


def verdicts(stat):
    keys = stat.get("nodes_with_key", {})
    n_nodes = max(stat.get("n_nodes", 0), 1)
    def frac(k):
        return keys.get(k, 0) / n_nodes
    q1 = ("full" if any(frac(k) > 0.05 for k in Q1_KEYS)
          else "none")
    q2 = ("full" if any(frac(k) > 0.05 for k in ("tool_call_id", "span_id", "parent_id"))
          else ("partial" if any(frac(k) > 0.05 for k in
                                 ("raw_payload", "source_event_id", "arguments", "source_kind"))
                else "none"))
    has_mem = any(k in stat.get("node_types", {})
                  for k in ("memory_write", "memory_read", "memory_record"))
    has_auth = any(k in stat.get("edge_types", {})
                   for k in ("authorization", "auth", "permission"))
    if has_auth:
        q3 = "full"
    elif has_mem:
        q3 = "none"
    else:
        q3 = "none"
    return {"Q1_read_record_identity": q1, "Q2_param_source": q2,
            "Q3_authorization": q3,
            "note": "verdicts are data-availability only; the unified graph "
                    "builder may still discard these fields"}


def audit_raw_traces(paths, cap=40):
    out = {"n_traces": 0, "records_with_record_id": 0, "mem_events_with_record_id": 0,
           "records_with_derived_from": 0, "records_total": 0,
           "mem_events_total": 0, "authorization_events": 0, "keys": Counter()}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        try:
            data = json.load(open(p))
        except Exception:
            continue
        runs = data if isinstance(data, list) else data.get("runs", data.get("results", []))
        if not isinstance(runs, list):
            continue
        for run in runs[:cap]:
            tr = run.get("trace", run) if isinstance(run, dict) else {}
            recs = tr.get("records", []) if isinstance(tr, dict) else []
            mems = tr.get("mem_events", []) if isinstance(tr, dict) else []
            out["n_traces"] += 1
            for r in recs:
                out["records_total"] += 1
                out["records_with_record_id"] += 1 if r.get("record_id") else 0
                out["records_with_derived_from"] += 1 if r.get("derived_from") else 0
                out["keys"].update(r.keys())
            for m in mems:
                out["mem_events_total"] += 1
                out["mem_events_with_record_id"] += 1 if m.get("record_id") else 0
            out["authorization_events"] += len(tr.get("authorizations", [])
                                                if isinstance(tr, dict) else [])
    out["keys"] = dict(out["keys"].most_common(20))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=500)
    args = ap.parse_args()
    cap = args.cap
    corpus_specs = {
        "decomp_fix2": ([f"code/repro/out/unified_gnn/fix2_splits/decomp/fold0/{s}.jsonl"
                         for s in ("train", "dev", "test")], cap),
        "mcp_fix2": ([f"code/repro/out/unified_gnn/fix2_splits/mcp/fold{f}/{s}.jsonl"
                      for f in range(2) for s in ("train", "dev", "test")], cap),
        "mcp_rebuilt_semantic": (
            ["ref/dataset/mcp_continual_rebuild_v1/semantic/"
             "mcp_safetybench_defense_rawtext_config_large_eval_v1_"
             f"{s}_neutral_semantic.jsonl" for s in ("train", "validation", "test")],
            min(cap, 200)),
        "mrt_cross_task": (["results/mrt_cross_task_v1/"
                            f"fold_{t}/{s}.jsonl"
                            for t in ("american_airlines_2", "bank_transfer_2")
                            for s in ("train", "admission")], cap),
        "v10_gate_v2": ([f"code/repro/out/online_gate_v2/{s}.jsonl"
                         for s in ("train", "dev", "test")], cap),
    }
    report = {"graph_corpora": {}, "raw_traces": {},
              "builder_gap_statement": (
                  "train_gnn_v2.graph_to_v2_data calls rebuild_v2_edges, which "
                  "never reads g['edges'] (train_gnn_v2.py:194-196,118-159); "
                  "gate_graph.trace_nodes materializes memory-write nodes but "
                  "attaches no record_id (gate_graph.py:52-58).")}
    for name, (paths, c) in corpus_specs.items():
        stat = audit_graphs(paths, cap=c)
        stat.update(verdicts(stat))
        report["graph_corpora"][name] = stat
    report["raw_traces"]["memory_chain_v6"] = audit_raw_traces(
        ["LOGS/memory_chain_v6_runs_20260911.json"])
    report["raw_traces"]["v10_online_all"] = audit_raw_traces(
        ["LOGS/v10_gnn_online_all_20260915.json"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=1, sort_keys=True, default=str)
    for name, stat in report["graph_corpora"].items():
        print(f"{name}: graphs={stat['n_graphs']} nodes={stat['n_nodes']} "
              f"edges={stat['n_edges']} non_adj={stat['non_adjacent_edges']} "
              f"Q1={stat['Q1_read_record_identity']} Q2={stat['Q2_param_source']} "
              f"Q3={stat['Q3_authorization']} keys={stat['nodes_with_key']}")
    for name, stat in report["raw_traces"].items():
        print(f"{name}: traces={stat['n_traces']} records={stat['records_total']} "
              f"with_record_id={stat['records_with_record_id']} "
              f"mem_events={stat['mem_events_total']} "
              f"with_record_id={stat['mem_events_with_record_id']} "
              f"auth_events={stat['authorization_events']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
