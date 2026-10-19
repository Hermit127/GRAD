#!/usr/bin/env python3
"""Convert MCP-SafetyBench runtime traces to behavior graph JSONL splits."""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


NODE_TYPES = {
    "file_read", "file_write", "file_delete", "shell_exec", "code_exec",
    "web_request", "web_search", "web_download", "database_query",
    "memory_write", "memory_read", "memory_reflect", "agent_spawn",
    "agent_message", "version_control", "prompt_inject", "system_config",
    "email_send", "social_media", "document_create", "information_search",
    "account_access", "message_send", "data_aggregate", "tool_execute",
}

MCP_CONFIG_EVIDENCE_KEYS = (
    "mcp_server_modifications",
    "mcp_server_additions",
    "mcp_server_update",
    "modification_return",
)

DANGEROUS_PATTERNS = re.compile(
    r"rm -rf|sudo\b|/etc/passwd|/etc/shadow|\.env\b|id_rsa|authorized_keys|"
    r"ssh-rsa|nc -lvp|/bin/bash|cat ~/.ssh|"
    r"steal(?:ing)? (?:credentials?|secrets?|tokens?|passwords?)|"
    r"(?:dump|leak|exfiltrate) (?:credentials?|secrets?|tokens?|passwords?)|"
    r"exfiltrat|backdoor|malware|payload|chmod 777|"
    r"curl\b[^\n|]*\|\s*(?:sh|bash)|wget\b[^\n|]*\|\s*(?:sh|bash)",
    re.IGNORECASE,
)


def compact(value: Any, max_len: int = 600) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = str(value)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_len]


def to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        return str(value)


def truncate_text(text: str, max_chars: int) -> tuple[str, bool, int]:
    original_chars = len(text)
    if max_chars > 0 and original_chars > max_chars:
        return text[:max_chars], True, original_chars
    return text, False, original_chars


def raw_payload_for_record(data: dict[str, Any], max_chars: int) -> tuple[dict[str, Any], bool, int]:
    rtype = data.get("type")
    if rtype == "tool":
        payload = {
            "type": data.get("type"),
            "server": data.get("server"),
            "tool_name": data.get("tool_name"),
            "arguments": data.get("arguments"),
            "response": data.get("response"),
            "error": data.get("error"),
        }
    elif rtype == "agent":
        payload = {
            "type": data.get("type"),
            "name": data.get("name"),
            "messages": data.get("messages"),
            "response": data.get("response"),
        }
    elif rtype == "llm":
        payload = {
            "type": data.get("type"),
            "messages": data.get("messages"),
            "response": data.get("response"),
        }
    else:
        payload = {"type": data.get("type"), "data": data}

    raw = to_text(payload)
    truncated, is_truncated, original_chars = truncate_text(raw, max_chars)
    if is_truncated:
        return {"truncated_text": truncated}, True, original_chars
    return payload, False, original_chars


def raw_record_text(data: dict[str, Any]) -> str:
    rtype = data.get("type")
    if rtype == "tool":
        return "\n\n".join(
            part for part in [
                "record_type: tool",
                f"server: {data.get('server', '')}",
                f"tool_name: {data.get('tool_name', '')}",
                "arguments:\n" + to_text(data.get("arguments")),
                "response:\n" + to_text(data.get("response")),
                "error:\n" + to_text(data.get("error")),
            ] if part.strip()
        )
    if rtype == "agent":
        return "\n\n".join(
            part for part in [
                "record_type: agent",
                f"name: {data.get('name', '')}",
                "messages:\n" + to_text(data.get("messages")),
                "response:\n" + to_text(data.get("response")),
            ] if part.strip()
        )
    if rtype == "llm":
        parts = ["record_type: llm"]
        messages = data.get("messages") or []
        if isinstance(messages, list):
            for midx, message in enumerate(messages):
                if isinstance(message, dict):
                    role = message.get("role", "")
                    content = message.get("content", "")
                else:
                    role = ""
                    content = message
                parts.append(f"message[{midx}].role: {role}\nmessage[{midx}].content:\n{to_text(content)}")
        else:
            parts.append("messages:\n" + to_text(messages))
        parts.append("response:\n" + to_text(data.get("response")))
        return "\n\n".join(part for part in parts if part.strip())
    return to_text(data)


def mcp_config_evidence(task_config: dict[str, Any] | None) -> dict[str, Any]:
    if not task_config:
        return {}
    evidence = {}
    for key in MCP_CONFIG_EVIDENCE_KEYS:
        value = task_config.get(key)
        if value not in (None, [], {}):
            evidence[key] = value
    return evidence


def format_mcp_config_evidence(evidence: dict[str, Any]) -> str:
    if not evidence:
        return ""
    return "mcp_config_evidence:\n" + to_text(evidence)


def dgc_source_metadata(task_config: dict[str, Any] | None) -> dict[str, Any]:
    if not task_config:
        return {}
    source = task_config.get("dgc_v1_source") or {}
    if not isinstance(source, dict):
        return {}
    keep = {
        "pair_id",
        "side",
        "domain",
        "mechanism",
        "target_gap",
        "safe_family",
    }
    return {
        f"dgc_{key}": value
        for key, value in source.items()
        if key in keep and value not in (None, "", [], {})
    }


def node_type_for_record(data: dict[str, Any]) -> str:
    rtype = str(data.get("type", "")).lower()
    if rtype == "llm":
        return "agent_message"
    if rtype == "agent":
        return "agent_message"
    if rtype != "tool":
        return "tool_execute"

    server = str(data.get("server", "")).lower()
    tool = str(data.get("tool_name", "")).lower()
    args = compact(data.get("arguments", {}), 1000).lower()
    name = f"{server}.{tool}"

    # Neutral mapping by tool FUNCTION ONLY. No attack keywords (curl/rm/chmod/etc.)
    # are inspected: a terminal command is always shell_exec regardless of content.
    if "github" in server or "git" in tool:
        return "version_control"
    if "filesystem" in server or "file" in tool:
        if any(k in tool for k in ("write", "edit", "create")):
            return "file_write"
        if any(k in tool for k in ("delete", "remove", "move")):
            return "file_delete"
        return "file_read"
    if "terminal" in server or tool in {"run_command", "execute_command"}:
        return "shell_exec"
    if "fetch" in server or "fetch" in tool:
        return "web_request"
    if "browser" in server or "playwright" in server or "navigate" in tool:
        return "web_request"
    if "search" in server or "search" in tool:
        return "web_search"
    if "maps" in server or "directions" in tool or "geocode" in tool:
        return "web_request"
    if "yfinance" in server or "finance" in tool:
        return "database_query"
    if "calculator" in server or "calculate" in tool:
        return "code_exec"
    if "memory" in name:
        return "memory_write" if any(k in tool for k in ("write", "create", "add")) else "memory_read"
    return "tool_execute"


def record_text(data: dict[str, Any]) -> str:
    rtype = data.get("type")
    if rtype == "tool":
        return " ".join(
            part for part in [
                str(data.get("server", "")),
                str(data.get("tool_name", "")),
                compact(data.get("arguments", {}), 500),
                compact(data.get("response", {}), 300),
                compact(data.get("error", ""), 100),
            ] if part
        )
    if rtype == "agent":
        return " ".join(
            part for part in [
                str(data.get("name", "")),
                compact(data.get("messages", ""), 300),
                compact(data.get("response", ""), 300),
            ] if part
        )
    if rtype == "llm":
        response = compact(data.get("response", ""), 300)
        messages = data.get("messages") or []
        user_tail = ""
        if isinstance(messages, list) and messages:
            user_tail = compact(messages[-1].get("content", ""), 300) if isinstance(messages[-1], dict) else compact(messages[-1], 300)
        return " ".join(part for part in ["llm", user_tail, response] if part)
    return compact(data, 500)


def flatten_records(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat = []
    for span_index, span in enumerate(trace):
        parent = span.get("parent_id")
        for rec_index, rec in enumerate(span.get("records", [])):
            data = rec.get("data", {})
            if data.get("type") not in {"llm", "tool", "agent"}:
                continue
            flat.append({
                "span_id": span.get("id"),
                "parent_id": parent,
                "span_index": span_index,
                "record_index": rec_index,
                "timestamp": rec.get("timestamp", 0),
                "data": data,
            })
    flat.sort(key=lambda r: (r.get("timestamp") or 0, r["span_index"], r["record_index"]))
    return flat


def graph_from_item(
    item: dict[str, Any],
    result: dict[str, Any],
    trace: list[dict[str, Any]],
    *,
    graph_label: int | None = None,
    label_semantics: str = "attack_success",
    source_results_dir: str = "",
    raw_text_max_chars: int = 20000,
    task_config: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    records = flatten_records(trace)
    if len(records) < 2:
        return None
    task_stem = Path(item["task"]).stem
    attack_success = bool(result.get("attack_success", False))
    if graph_label is None:
        graph_label = 1 if attack_success else 0
    config_evidence = mcp_config_evidence(task_config)
    config_evidence_text = format_mcp_config_evidence(config_evidence)
    nodes = []
    for idx, rec in enumerate(records):
        data = rec["data"]
        ntype = node_type_for_record(data)
        if ntype not in NODE_TYPES:
            ntype = "tool_execute"
        text = record_text(data)
        raw_text_source = raw_record_text(data)
        if idx == 0 and config_evidence_text:
            raw_text_source = config_evidence_text + "\n\n" + raw_text_source
        raw_text, raw_text_truncated, raw_text_chars_original = truncate_text(
            raw_text_source,
            raw_text_max_chars,
        )
        raw_payload, raw_payload_truncated, raw_payload_chars_original = raw_payload_for_record(
            data,
            raw_text_max_chars,
        )
        is_tool = data.get("type") == "tool"
        is_error = bool(data.get("response", {}).get("isError")) if isinstance(data.get("response"), dict) else False
        dangerous = bool(DANGEROUS_PATTERNS.search(text))
        node_label = 1 if graph_label and (is_tool or dangerous) else 0
        nodes.append({
            "id": f"{task_stem}_n{idx:03d}",
            "type": ntype,
            "taint": 1.0 if node_label else (0.35 if dangerous else 0.0),
            "turn": idx,
            "timestamp": rec.get("timestamp", 0),
            "features": {
                "subtask": text,
                "raw_text": raw_text,
                "raw_text_source": "trace_record",
                "raw_text_truncated": raw_text_truncated,
                "raw_text_chars_original": raw_text_chars_original,
                "raw_payload": raw_payload,
                "raw_payload_truncated": raw_payload_truncated,
                "raw_payload_chars_original": raw_payload_chars_original,
                "mcp_config_evidence": config_evidence if idx == 0 else {},
                "mcp_config_evidence_source": "task_config" if idx == 0 and config_evidence else "",
                "node_label": node_label,
                "harm_step": bool(node_label),
                "record_type": data.get("type"),
                "server": data.get("server", ""),
                "tool_name": data.get("tool_name", ""),
                "tool_error": is_error,
                "dangerous_pattern": dangerous,
                "span_id": rec.get("span_id"),
                "parent_id": rec.get("parent_id"),
                "span_index": rec.get("span_index"),
                "record_index": rec.get("record_index"),
            },
        })

    # Q3 fix: collect per-node argument/response PLAIN TEXT for cross-node deps
    def _plain(obj):
        if isinstance(obj, dict):
            return " ".join(_plain(v) for v in obj.values())
        if isinstance(obj, list):
            return " ".join(_plain(v) for v in obj)
        return str(obj)

    _dep = []
    for _nd, _rec in zip(nodes, records):
        _d = _rec.get("data", {})
        _arg = _plain(_d.get("arguments", "")) if isinstance(_d, dict) else ""
        _out = _plain(_d.get("response", "")) if isinstance(_d, dict) else ""
        _dep.append((_nd["id"], _arg, _out))

    edges = []
    for idx in range(len(nodes) - 1):
        edge_type = "causal_derivation"
        cur_type = nodes[idx]["features"]["record_type"]
        nxt_type = nodes[idx + 1]["features"]["record_type"]
        if cur_type == "tool" and nxt_type in {"llm", "agent"}:
            edge_type = "data_flow"
        elif cur_type in {"llm", "agent"} and nxt_type == "tool":
            edge_type = "triggers"
        edges.append({"source": nodes[idx]["id"], "target": nodes[idx + 1]["id"], "type": edge_type})

    # Q3: cross-node dependency edges (argument references an earlier response),
    # distinct from temporal adjacency. Bounded O(n^2) with length>=20 filter.
    _existing = {(e["source"], e["target"]) for e in edges}
    for j in range(len(_dep)):
        jid, jarg, _ = _dep[j]
        if not jarg or len(jarg) < 20:
            continue
        jlow = jarg.lower()
        for i in range(max(0, j - 12), j - 1):  # window, skip adjacency
            iid, _, iout = _dep[i]
            if not iout:
                continue
            # long tokens (>=12 chars) from the earlier response quoted in the
            # current arguments => genuine data dependency, not temporal order
            toks = [t for t in iout.replace('"', ' ').split() if len(t) >= 12]
            if any(t.lower() in jlow for t in toks) and (iid, jid) not in _existing:
                edges.append({"source": iid, "target": jid, "type": "data_flow"})
                _existing.add((iid, jid))

    metadata = {
        "source": "mcp_safetybench",
        "graph_origin": "runtime_trace",
        "evidence_level": "observed_action",
        "task": item.get("task"),
        "config": item.get("config"),
        "trace_id": item.get("trace_id"),
        "status": item.get("status"),
        "trace_records": item.get("trace_records"),
        "reconstructed_summary_entry": bool(item.get("reconstructed_summary_entry")),
        "category": result.get("category"),
        "attack_category": result.get("attack_category"),
        "task_success": result.get("task_success"),
        "attack_success": result.get("attack_success"),
        "label_semantics": label_semantics,
        "source_results_dir": source_results_dir,
    }
    metadata.update(dgc_source_metadata(task_config))

    return {
        "task_id": task_stem,
        "label": graph_label,
        "harm_index": next((i for i, n in enumerate(nodes) if n["features"]["node_label"]), -1),
        "num_nodes": len(nodes),
        "num_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
        "metadata": metadata,
    }


def load_graphs(
    results_dir: Path,
    *,
    graph_label: int | None = None,
    label_semantics: str = "attack_success",
    raw_text_max_chars: int = 20000,
    task_config_root: Path | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    # Cross-model support: scan all model subdirs under benchmark_results
    # (deepseek-chat historically; glm-5.3-flash etc. for cross-model runs).
    bench_root = results_dir / "benchmark_results"
    bench_dirs = sorted([d for d in bench_root.iterdir() if d.is_dir()]) if bench_root.exists() else []
    bench = bench_dirs[0] if bench_dirs else bench_root / "deepseek-chat"
    graphs, skipped = [], []
    converted_tasks: set[tuple[str, str]] = set()
    # Resume logs can contain an older timeout/error followed by a successful
    # retry. Walk newest-first and emit at most one valid graph per task.
    for item in reversed(summary):
        task_key = (str(item.get("config", "")), str(item.get("task", "")))
        if task_key in converted_tasks:
            skipped.append({"task": item.get("task"), "reason": "duplicate_summary_entry", "status": item.get("status")})
            continue
        stem = Path(item["task"]).stem
        result_path = None
        for bd in ([bench] if bench != (bench_root / "deepseek-chat") else bench_dirs) if bench_dirs else [bench]:
            cand = bd / f"{stem}_{bd.name}.json"
            if cand.exists():
                result_path = cand
                break
        if result_path is None and bench_dirs:
            result_path = bench / f"{stem}_{bench.name}.json"
        if result_path is None or not result_path.exists():
            result_path = bench / f"{stem}_{bench.name}.json"
        trace_path = item.get("trace_path")
        if not result_path.exists() or not trace_path or not item.get("trace_records"):
            skipped.append({"task": item.get("task"), "reason": "missing_result_or_trace", "status": item.get("status")})
            continue
        trace_file = results_dir / str(trace_path).replace("/results/", "")
        if not trace_file.exists():
            skipped.append({"task": item.get("task"), "reason": "trace_file_missing", "status": item.get("status")})
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        trace = json.loads(trace_file.read_text(encoding="utf-8"))
        task_config = None
        if task_config_root is not None:
            task_config_path = task_config_root / item["task"]
            if task_config_path.exists():
                task_config = json.loads(task_config_path.read_text(encoding="utf-8"))
        graph = graph_from_item(
            item,
            result,
            trace,
            graph_label=graph_label,
            label_semantics=label_semantics,
            source_results_dir=str(results_dir),
            raw_text_max_chars=raw_text_max_chars,
            task_config=task_config,
        )
        if graph is None:
            skipped.append({"task": item.get("task"), "reason": "too_few_observed_records", "status": item.get("status")})
            continue
        graphs.append(graph)
        converted_tasks.add(task_key)
    graphs.reverse()
    return graphs, skipped


def split_graphs(graphs: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    buckets: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for g in graphs:
        key = (int(g["label"]), str(g.get("metadata", {}).get("attack_category", "unknown")))
        buckets[key].append(g)

    splits = {"train": [], "validation": [], "test": []}
    for bucket in buckets.values():
        rng.shuffle(bucket)
        n = len(bucket)
        if n == 1:
            splits["train"].extend(bucket)
            continue
        n_test = max(1, round(n * 0.2))
        n_val = max(1, round(n * 0.2)) if n >= 3 else 0
        if n_test + n_val >= n:
            n_test = 1
            n_val = 0 if n == 2 else 1
        splits["test"].extend(bucket[:n_test])
        splits["validation"].extend(bucket[n_test:n_test + n_val])
        splits["train"].extend(bucket[n_test + n_val:])
    for values in splits.values():
        rng.shuffle(values)
    return splits


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "labels": dict(Counter(int(g["label"]) for g in rows)),
        "categories": dict(Counter(g.get("metadata", {}).get("attack_category") for g in rows)),
        "avg_nodes": round(sum(g["num_nodes"] for g in rows) / max(len(rows), 1), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results/mcp_safetybench_runtime_full_245_20260616_1820")
    parser.add_argument("--attack-results-dir", action="append", default=[])
    parser.add_argument("--benign-results-dir", action="append", default=[])
    parser.add_argument("--label-mode", choices=["attack_success", "defense"], default="attack_success")
    parser.add_argument(
        "--match-benign-sources",
        action="store_true",
        help="In defense mode, keep only attack traces whose task stem has a generated *_benign counterpart.",
    )
    parser.add_argument("--out-dir", default="ref/dataset")
    parser.add_argument("--prefix", default="mcp_safetybench")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--task-config-root",
        default="baseline/mcp-safetybench/mcpuniverse/benchmark/configs",
        help="Root used to attach MCP mutation evidence from task configs to existing trace nodes.",
    )
    parser.add_argument(
        "--raw-text-max-chars",
        type=int,
        default=20000,
        help="Maximum raw evidence characters stored per trace node; <=0 stores the full record.",
    )
    args = parser.parse_args()
    task_config_root = Path(args.task_config_root) if args.task_config_root else None

    if args.label_mode == "defense":
        if not args.attack_results_dir or not args.benign_results_dir:
            raise SystemExit("--label-mode defense requires --attack-results-dir and --benign-results-dir")
        graphs, skipped = [], []
        attack_graphs, benign_graphs = [], []
        source_results_dirs = {"attack": args.attack_results_dir, "benign": args.benign_results_dir}
        for results_dir in args.attack_results_dir:
            loaded, missed = load_graphs(
                Path(results_dir),
                graph_label=1,
                label_semantics="attack_vs_benign",
                raw_text_max_chars=args.raw_text_max_chars,
                task_config_root=task_config_root,
            )
            attack_graphs.extend(loaded)
            skipped.extend({"source_label": "attack", **row} for row in missed)
        for results_dir in args.benign_results_dir:
            loaded, missed = load_graphs(
                Path(results_dir),
                graph_label=0,
                label_semantics="attack_vs_benign",
                raw_text_max_chars=args.raw_text_max_chars,
                task_config_root=task_config_root,
            )
            benign_graphs.extend(loaded)
            skipped.extend({"source_label": "benign", **row} for row in missed)
        if args.match_benign_sources:
            benign_source_stems = {
                g["task_id"][:-7] if g["task_id"].endswith("_benign") else g["task_id"]
                for g in benign_graphs
            }
            removed = [g for g in attack_graphs if g["task_id"] not in benign_source_stems]
            attack_graphs = [g for g in attack_graphs if g["task_id"] in benign_source_stems]
            skipped.extend({
                "source_label": "attack",
                "task": g.get("metadata", {}).get("task"),
                "reason": "no_generated_benign_counterpart",
                "status": g.get("metadata", {}).get("status"),
            } for g in removed)
        graphs = attack_graphs + benign_graphs
        label_definition = "defense task: label=1 for MCP-SafetyBench attack traces; label=0 for generated benign MCP traces."
    else:
        graphs, skipped = load_graphs(
            Path(args.results_dir),
            raw_text_max_chars=args.raw_text_max_chars,
            task_config_root=task_config_root,
        )
        source_results_dirs = {"attack_success": [args.results_dir]}
        label_definition = "label=1 iff MCP-SafetyBench benchmark_result.attack_success is true; label=0 otherwise."

    splits = split_graphs(graphs, args.seed)
    out_dir = Path(args.out_dir)
    for split, rows in splits.items():
        write_jsonl(out_dir / f"{args.prefix}_{split}.jsonl", rows)
    manifest = {
        "source_results_dir": args.results_dir,
        "source_results_dirs": source_results_dirs,
        "prefix": args.prefix,
        "seed": args.seed,
        "total_graphs": len(graphs),
        "skipped": skipped,
        "splits": {name: summarize(rows) for name, rows in splits.items()},
        "label_mode": args.label_mode,
        "label_definition": label_definition,
        "raw_text_max_chars": args.raw_text_max_chars,
        "task_config_root": str(task_config_root) if task_config_root else "",
        "converter_contract": "one graph node per observed llm/tool/agent trace step; raw MCP evidence is stored on existing node features, not as new node types.",
    }
    (out_dir / f"{args.prefix}_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
