"""Run MCP-SafetyBench tasks one-by-one and export redacted runtime traces."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import tempfile
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from mcpuniverse.benchmark.runner import BenchmarkRunner
from mcpuniverse.tracer.collectors.memory import MemoryCollector


DEFAULT_CONFIGS = [
    "dummy/benchmark_deepseek_minimal.yaml",
    "test/financial_analysis.yaml",
    "test/browser_automation.yaml",
    "test/web_search.yaml",
    "test/location_navigation.yaml",
    "test/repository_management.yaml",
]

SECRET_KEYS = {
    "api_key",
    "access_token",
    "auth_token",
    "bearer_token",
    "client_secret",
    "credential",
    "password",
    "refresh_token",
    "secret",
    "token",
}


def is_secret_key(key: Any) -> bool:
    lowered = str(key).lower()
    return lowered in SECRET_KEYS or lowered.endswith(("_api_key", "_token", "_secret", "_password"))


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out[key] = "<redacted>" if is_secret_key(key) and item else redact(item)
        return out
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def trace_record_to_dict(record: Any) -> dict[str, Any]:
    if hasattr(record, "to_dict"):
        return redact(record.to_dict())
    return redact(record)


def eval_result_to_dict(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        return redact(result.model_dump(mode="json"))
    return redact(dict(result))


def load_config_docs(config_path: str, config_root: str | None = None) -> tuple[list[dict[str, Any]], Path]:
    base = Path(config_root) if config_root else Path("mcpuniverse/benchmark/configs")
    path = Path(config_path)
    if not path.exists():
        path = base / config_path
    if not path.exists():
        raise FileNotFoundError(config_path)
    docs = [doc for doc in yaml.safe_load_all(path.read_text(encoding="utf-8")) if doc]
    return docs, path


def normalize_docs(
    docs: list[dict[str, Any]],
    task: str,
    model_name: str,
    max_iterations: int | None,
    config_root: str | None = None,
) -> list[dict[str, Any]]:
    normalized = deepcopy(docs)
    normalized_task = task
    if config_root and not Path(task).is_absolute():
        normalized_task = str((Path(config_root) / task).resolve())
    for doc in normalized:
        kind = str(doc.get("kind", "")).lower()
        spec = doc.setdefault("spec", {})
        config = spec.setdefault("config", {})
        if kind == "llm":
            spec["type"] = "deepseek"
            config["model_name"] = model_name
            config.setdefault("temperature", 0.0)
            config.setdefault("max_completion_tokens", 1024)
        elif kind == "agent" and max_iterations is not None:
            config["max_iterations"] = max_iterations
        elif kind == "benchmark":
            spec["tasks"] = [normalized_task]
    return normalized


def resolve_task_key(task: str, config_root: str | None = None) -> str:
    if config_root and not Path(task).is_absolute():
        return str((Path(config_root) / task).resolve())
    return task


def rewrite_dgc_evaluators(task_config: dict[str, Any]) -> dict[str, Any]:
    rewritten = deepcopy(task_config)
    evaluators = rewritten.get("evaluators", [])
    if not evaluators:
        rewritten["evaluators"] = [{
            "func": "json -> get(result)",
            "op": "contain",
            "value": "",
            "desc": "Whether the response has a JSON result field.",
        }]
        return rewritten
    for evaluator in evaluators:
        if evaluator.get("op") == "dgc_v1.check_result_shape":
            evaluator["func"] = "json -> get(result)"
            evaluator["op"] = "contain"
            evaluator["value"] = ""
            evaluator.pop("op_args", None)
            evaluator["desc"] = "Whether the response has a JSON result field."
    return rewritten


def prepare_runtime_config_root(source_root: str | None, rewrite_dgc: bool) -> str | None:
    if not source_root or not rewrite_dgc:
        return source_root
    source = Path(source_root).resolve()
    prepared = Path(tempfile.mkdtemp(prefix="dgc_task_configs_"))
    for path in source.rglob("*"):
        rel = path.relative_to(source)
        target = prepared / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".json":
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                data = rewrite_dgc_evaluators(data)
                target.write_text(json.dumps(data, indent=2), encoding="utf-8")
            except Exception:
                target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        else:
            target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return str(prepared)


def task_list(docs: list[dict[str, Any]]) -> list[str]:
    for doc in docs:
        if str(doc.get("kind", "")).lower() == "benchmark":
            return list(doc.get("spec", {}).get("tasks", []))
    return []


def write_temp_config(docs: list[dict[str, Any]]) -> str:
    handle = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8")
    with handle:
        yaml.safe_dump_all(docs, handle, sort_keys=False)
    return handle.name


def latest_summary_items(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    order: list[tuple[str, str]] = []
    for item in summary:
        key = (str(item.get("config", "")), str(item.get("task", "")))
        if key not in latest:
            order.append(key)
        latest[key] = item
    return [latest[key] for key in order]


async def run_one(
    config_name: str,
    config_path: Path,
    docs: list[dict[str, Any]],
    task: str,
    out_dir: Path,
    model_name: str,
    max_iterations: int | None,
    task_timeout_seconds: int | None,
    config_root: str | None,
) -> dict[str, Any]:
    collector = MemoryCollector()
    normalized_task = resolve_task_key(task, config_root=config_root)
    normalized = normalize_docs(
        docs,
        task=task,
        model_name=model_name,
        max_iterations=max_iterations,
        config_root=config_root,
    )
    temp_config = write_temp_config(normalized)
    safe_task_name = Path(task).stem
    task_out_dir = out_dir / "benchmark_results"
    trace_dir = out_dir / "traces"
    trace_dir.mkdir(parents=True, exist_ok=True)
    task_out_dir.mkdir(parents=True, exist_ok=True)

    item: dict[str, Any] = {
        "config": config_name,
        "config_path": str(config_path),
        "task": task,
        "status": "unknown",
        "trace_id": None,
        "trace_records": 0,
        "trace_path": None,
        "evaluation_results": [],
        "passed": None,
        "error": "",
    }
    try:
        runner = BenchmarkRunner(temp_config)
        run_coro = runner.run(
            trace_collector=collector,
            store_folder=str(task_out_dir),
            overwrite=True,
        )
        if task_timeout_seconds:
            outputs = await asyncio.wait_for(run_coro, timeout=task_timeout_seconds)
        else:
            outputs = await run_coro
        for output in outputs:
            result_task = None
            for candidate in (task, normalized_task):
                if candidate in output.task_trace_ids:
                    result_task = candidate
                    break
            if result_task is None and len(output.task_trace_ids) == 1:
                result_task = next(iter(output.task_trace_ids))
            if result_task is None:
                continue
            trace_id = output.task_trace_ids[result_task]
            records = [trace_record_to_dict(record) for record in collector.get(trace_id)]
            trace_path = trace_dir / f"{safe_task_name}_{trace_id}.json"
            trace_path.write_text(json.dumps(records, indent=2), encoding="utf-8")
            evals = [eval_result_to_dict(r) for r in output.task_results[result_task]["evaluation_results"]]
            item.update(
                {
                    "status": "ok",
                    "trace_id": trace_id,
                    "trace_records": len(records),
                    "trace_path": str(trace_path),
                    "evaluation_results": evals,
                    "passed": all(ev.get("passed", False) for ev in evals) if evals else None,
                }
            )
            break
        if item["status"] == "unknown":
            item["status"] = "no_result"
    except asyncio.TimeoutError:
        item.update(
            {
                "status": "timeout",
                "error": f"task exceeded timeout of {task_timeout_seconds} seconds",
            }
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        item.update(
            {
                "status": "error",
                "error": str(exc),
                "traceback": traceback.format_exc(limit=12),
            }
        )
    finally:
        Path(temp_config).unlink(missing_ok=True)
    return item


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", action="append", help="Benchmark YAML config, relative to mcpuniverse/benchmark/configs.")
    parser.add_argument("--config-root", default=None, help="Root directory for benchmark YAML and task JSON files.")
    parser.add_argument("--dgc-rewrite-evaluator", action="store_true",
                        help="Rewrite DGC-v1 placeholder evaluators to built-in JSON result checks in a temp config root.")
    parser.add_argument("--out", default="/results", help="Output directory for summaries and traces.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of tasks to attempt across configs. 0 means all.")
    parser.add_argument("--offset", type=int, default=0, help="Skip this many tasks before running.")
    parser.add_argument("--model-name", default="deepseek-chat")
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--task-timeout-seconds", type=int, default=0, help="Per-task timeout. 0 disables timeout.")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--concurrency", type=int, default=1,
                        help="Number of tasks to execute concurrently inside one container.")
    parser.add_argument("--resume-ok", action="store_true",
                        help="Skip prior status=ok entries whose trace file still exists.")
    parser.add_argument("--summary-flush-every", type=int, default=25,
                        help="Rewrite summary.json every N completed rows; JSONL is appended every row.")
    args = parser.parse_args()

    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    config_root = prepare_runtime_config_root(args.config_root, args.dgc_rewrite_evaluator)
    configs = args.config or DEFAULT_CONFIGS
    summary_path = out_dir / "summary.json"
    summary_jsonl_path = out_dir / "summary.jsonl"

    selected: list[tuple[str, Path, list[dict[str, Any]], str]] = []
    for config_name in configs:
        docs, path = load_config_docs(config_name, config_root=config_root)
        for task in task_list(docs):
            selected.append((config_name, path, docs, task))

    summary: list[dict[str, Any]] = []
    if summary_jsonl_path.exists():
        with summary_jsonl_path.open(encoding="utf-8") as handle:
            summary = [json.loads(line) for line in handle if line.strip()]
    elif summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = []
        with summary_jsonl_path.open("w", encoding="utf-8") as handle:
            for item in summary:
                handle.write(json.dumps(item) + "\n")

    if args.concurrency <= 0 or args.summary_flush_every <= 0:
        raise ValueError("--concurrency and --summary-flush-every must be positive")
    if args.resume_ok:
        completed = {
            (str(item.get("config")), str(item.get("task")))
            for item in summary
            if item.get("status") == "ok"
            and item.get("trace_path")
            and Path(str(item["trace_path"])).exists()
        }
        selected = [
            row for row in selected
            if (str(row[0]), str(row[3])) not in completed
        ]
    if args.offset:
        selected = selected[args.offset :]
    if args.limit:
        selected = selected[: args.limit]

    semaphore = asyncio.Semaphore(args.concurrency)

    async def execute_one(
        index: int,
        config_name: str,
        path: Path,
        docs: list[dict[str, Any]],
        task: str,
    ) -> tuple[int, dict[str, Any]]:
        async with semaphore:
            print(json.dumps({
                "event": "start_task",
                "index": index,
                "total": len(selected),
                "summary_items_before": len(summary),
                "config": config_name,
                "task": task,
            }))
            item = await run_one(
                config_name=config_name,
                config_path=path,
                docs=docs,
                task=task,
                out_dir=out_dir,
                model_name=args.model_name,
                max_iterations=args.max_iterations,
                task_timeout_seconds=args.task_timeout_seconds or None,
                config_root=config_root,
            )
            return index, item

    pending = [
        asyncio.create_task(execute_one(index, config_name, path, docs, task))
        for index, (config_name, path, docs, task) in enumerate(selected, start=1)
    ]
    with summary_jsonl_path.open("a", encoding="utf-8") as summary_jsonl:
        for future in asyncio.as_completed(pending):
            index, item = await future
            summary.append(item)
            summary_jsonl.write(json.dumps(item) + "\n")
            summary_jsonl.flush()
            if len(summary) % args.summary_flush_every == 0:
                summary_path.write_text(
                    json.dumps(latest_summary_items(summary), indent=2),
                    encoding="utf-8",
                )
            print(json.dumps({
                "event": "finish_task",
                "index": index,
                "status": item["status"],
                "passed": item["passed"],
                "trace_records": item["trace_records"],
            }))
            if args.stop_on_error and item["status"] == "error":
                for task in pending:
                    if not task.done():
                        task.cancel()
                break

    compact_summary = latest_summary_items(summary)
    summary_path.write_text(json.dumps(compact_summary, indent=2), encoding="utf-8")

    print(json.dumps({
        "summary_path": str(out_dir / "summary.json"),
        "unique_tasks": len(compact_summary),
        "history_rows": len(summary),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
