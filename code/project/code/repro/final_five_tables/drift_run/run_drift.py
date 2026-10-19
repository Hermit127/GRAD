#!/usr/bin/env python
"""DRIFT x AgentDojo v1.2 closed loop with DeepSeek (local adaptation).

Arms (same harness, official flag semantics from ref/DRIFT/pipeline_main.py):
  none  : DRIFTLLM with build_constraints / injection_isolation /
          dynamic_validation all OFF (official vanilla path of this repo)
  drift : all three DRIFT components ON

Protocol (matches the B1 registration): AgentDojo 0.1.35, benchmark v1.2,
suites banking/slack/travel/workspace, first 5 user tasks by ID per suite,
first injection task, conditions benign/attack, agent deepseek-chat temp 0.

Budget: every LLM call is written to the shared ledger
LOGS/sota_runtime_ledger.json with kind='drift'. --batch-cap is converted to
an absolute ceiling (start + batch) and over-cap calls hard-stop the process
(BudgetExceeded). Paid modes refuse to start without --batch-cap/--cap.

Results: NEW file under LOGS/final_five_tables/ (never overwrites a file from
a different contract). Smoke mode always writes its own timestamped
*_smoke*.json (mistake register #17).

Usage:
  python run_drift.py run --batch-cap 6.0 --dry-run       # offline plan check
  python run_drift.py run --pilot --batch-cap 1.0         # 3-run pilot
  python run_drift.py run   --batch-cap 6.0               # full 80-run batch
  python run_drift.py smoke --batch-cap 1.0               # wiring smoke (4 runs)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[4]  # code/repro/final_five_tables/drift_run/
DRIFT_DIR = REPO / "ref/DRIFT"
if str(DRIFT_DIR) not in sys.path:
    sys.path.insert(0, str(DRIFT_DIR))

RESULTS_DIR = REPO / "LOGS/final_five_tables"
DEFAULT_RESULTS = RESULTS_DIR / "drift_b1_none_drift_20260916.json"
PILOT_RESULTS = RESULTS_DIR / "drift_pilot_banking_user_task_0_20260916.json"
RUN_CONTRACT = "drift-agentdojo-v1.2-deepseek-v1-20260916"
BENCHMARK = "v1.2"

from DRIFTLLM import DRIFTLLM  # noqa: E402
from DRIFTTaskSuite import DRIFTTaskSuite  # noqa: E402
from DRIFTToolsExecutionLoop import DRIFTToolsExecutionLoop  # noqa: E402
from deepseek_client import (BudgetExceeded, DeepSeekModel, Ledger,  # noqa: E402
                             MODEL)
from agentdojo.agent_pipeline import AgentPipeline, InitQuery  # noqa: E402
from agentdojo.agent_pipeline.tool_execution import ToolsExecutor  # noqa: E402
from agentdojo.attacks.attack_registry import load_attack  # noqa: E402
from agentdojo.task_suite.load_suites import get_suite  # noqa: E402
from utils import get_logger, set_seed  # noqa: E402


class InstrumentedDRIFTLLM(DRIFTLLM):
    """DRIFTLLM with pass-through counters; no behavior is changed."""

    COUNTER_KEYS = ("planner_calls", "validator_calls", "alignment_judge_calls",
                    "privilege_assignment_calls", "isolator_calls",
                    "isolator_masks", "interpreter_errors")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.drift_counters = {k: 0 for k in self.COUNTER_KEYS}

    def snapshot(self):
        return dict(self.drift_counters)

    def initial_constraints_build(self, completion):
        self.drift_counters["planner_calls"] += 1
        return super().initial_constraints_build(completion)

    def function_privilege_assignment(self, function):
        self.drift_counters["privilege_assignment_calls"] += 1
        return super().function_privilege_assignment(function)

    def alignment_judge(self, *args, **kwargs):
        self.drift_counters["alignment_judge_calls"] += 1
        return super().alignment_judge(*args, **kwargs)

    def trajectory_constraint_validation(self, *args, **kwargs):
        self.drift_counters["validator_calls"] += 1
        return super().trajectory_constraint_validation(*args, **kwargs)

    def checklist_constraint_validation(self, *args, **kwargs):
        self.drift_counters["validator_calls"] += 1
        return super().checklist_constraint_validation(*args, **kwargs)

    def injection_isolate(self, detected_instructions, messages, openai_messages):
        self.drift_counters["isolator_calls"] += 1
        pre = messages[-1].get("content") if messages else None
        mark, messages, openai_messages = super().injection_isolate(
            detected_instructions, messages, openai_messages)
        post = messages[-1].get("content") if messages else None
        if mark and pre != post:
            self.drift_counters["isolator_masks"] += 1
        return mark, messages, openai_messages

    def _parse_model_output(self, message):
        try:
            return super()._parse_model_output(message)
        except Exception:
            self.drift_counters["interpreter_errors"] += 1
            raise


def patch_model_names():
    """Map deepseek-chat to a prose name for the important_instructions attack.

    Official agentdojo raises if the pipeline name matches no entry of
    MODEL_NAMES; the IPIGuard adaptation added the same entry. With this, the
    attack string carries {user}='Emma Johnson', {model}='DeepSeek', i.e. the
    same surface as the B1 custom DeepSeekImportantInstructions.
    """
    from agentdojo.attacks.important_instructions_attacks import MODEL_NAMES
    MODEL_NAMES.setdefault("deepseek-chat", "DeepSeek")


def build_pipelines(client, logger):
    pipelines = {}
    for arm, flags in (("none", (False, False, False)),
                       ("drift", (True, True, True))):
        args = SimpleNamespace(build_constraints=flags[0],
                               injection_isolation=flags[1],
                               dynamic_validation=flags[2],
                               adaptive_attack=False)
        llm = InstrumentedDRIFTLLM(args, client, logger=logger)
        tools_loop = DRIFTToolsExecutionLoop([ToolsExecutor(), llm])
        pipeline = AgentPipeline([InitQuery(), llm, tools_loop])
        pipeline.name = MODEL
        pipelines[arm] = (pipeline, llm)
    return pipelines


def make_task_suite(suite_name, suite_args):
    suite = get_suite(BENCHMARK, suite_name)
    task_suite = DRIFTTaskSuite(
        suite_args, suite.name, suite.environment_type, suite.tools,
        suite.data_path, suite.benchmark_version, parent_instance=suite)
    return task_suite


def enumerate_jobs(suites, tasks, injections, arms, conditions, pilot):
    jobs = []
    for suite_name in suites:
        suite = get_suite(BENCHMARK, suite_name)
        user_ids = sorted(suite.user_tasks)[:tasks]
        inj_ids = sorted(suite.injection_tasks)[:injections]
        for arm in arms:
            for uid in user_ids:
                jobs.append({"suite": suite_name, "user_task": uid,
                             "injection_task": None, "condition": "benign",
                             "arm": arm})
                for iid in inj_ids:
                    jobs.append({"suite": suite_name, "user_task": uid,
                                 "injection_task": iid, "condition": "attack",
                                 "arm": arm})
    if conditions != "all":
        conds = set(conditions.split(","))
        jobs = [j for j in jobs if j["condition"] in conds]
    if pilot:
        # 3 runs: (none, benign), (drift, benign), (drift, attack)
        jobs = [j for j in jobs
                if j["arm"] == "drift" or j["condition"] == "benign"]
    return jobs


def resolve_results_path(mode, out, pilot):
    if out:
        path = Path(out)
    elif mode == "smoke":
        path = RESULTS_DIR / f"drift_run_smoke_{time.strftime('%Y%m%d_%H%M%S')}.json"
    elif pilot:
        path = PILOT_RESULTS
    else:
        path = DEFAULT_RESULTS
    if mode == "smoke":
        if "smoke" not in path.name:
            raise ValueError("smoke mode must write to its own *_smoke*.json file")
        if path.exists():
            raise ValueError(f"smoke output already exists: {path}")
    elif path.exists():
        saved = json.loads(path.read_text())
        if saved.get("runner_contract") != RUN_CONTRACT:
            raise ValueError(
                f"refusing to touch {path}: runner_contract mismatch "
                f"({saved.get('runner_contract')!r} != {RUN_CONTRACT!r})")
    return path


def run_one(task_suite, pipeline, llm, client, user_task, injection_task,
            injections, logger):
    counters_before = llm.snapshot()
    calls_before = client.ledger.calls if client.ledger else 0
    tokens_before = client.total_tokens
    t0 = time.time()
    error = None
    messages = []
    utility = security = None
    for attempt in (1, 2):
        try:
            utility, security, messages = task_suite.run_task_with_pipeline(
                pipeline, user_task, injection_task, injections)
            error = None
            break
        except BudgetExceeded:
            raise
        except Exception as exc:  # noqa: BLE001
            error = f"{type(exc).__name__}: {exc}"[:300]
            if logger:
                logger.info(f"run failed (attempt {attempt}): {error}")
    delta = {k: llm.drift_counters[k] - counters_before[k]
             for k in llm.COUNTER_KEYS}
    tool_errors = sum(1 for m in messages if isinstance(m, dict)
                      and m.get("role") == "tool" and m.get("error"))
    return {
        "utility": None if error else bool(utility),
        "security": None if error else bool(security),
        "planner_calls": delta["planner_calls"],
        "validator_calls": delta["validator_calls"],
        "alignment_judge_calls": delta["alignment_judge_calls"],
        "privilege_assignment_calls": delta["privilege_assignment_calls"],
        "isolator_calls": delta["isolator_calls"],
        "isolator_masks": delta["isolator_masks"],
        "interpreter_errors": delta["interpreter_errors"],
        "tool_errors": tool_errors,
        "duration_s": round(time.time() - t0, 2),
        "llm_calls": (client.ledger.calls if client.ledger else 0) - calls_before,
        "tokens_total": client.total_tokens - tokens_before,
        "error": error,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["smoke", "run"])
    ap.add_argument("--batch-cap", type=float, default=None,
                    help="max spend for THIS batch in RMB; absolute ceiling = "
                         "ledger_start + batch_cap (required for paid runs)")
    ap.add_argument("--cap", type=float, default=None,
                    help="absolute ledger ceiling in RMB (legacy alternative)")
    ap.add_argument("--out", default=None, help="result file override")
    ap.add_argument("--suites", default="banking,slack,travel,workspace")
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--injections", type=int, default=1)
    ap.add_argument("--arms", default="none,drift")
    ap.add_argument("--conditions", default="benign,attack",
                    choices=["benign", "attack", "benign,attack", "all"])
    ap.add_argument("--pilot", action="store_true",
                    help="3-run pilot: banking/user_task_0, none+benign, "
                         "drift+benign, drift+attack")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the planned jobs and paths; makes no API call")
    ap.add_argument("--seed", type=int, default=98)
    ap.add_argument("--log", default=None, help="optional log file path")
    a = ap.parse_args(argv)

    if a.mode == "smoke":
        a.suites, a.tasks, a.injections = "banking", 1, 1
    if a.pilot:
        a.suites, a.tasks, a.injections = "banking", 1, 1

    suites = a.suites.split(",")
    arms = a.arms.split(",")
    for arm in arms:
        if arm not in ("none", "drift"):
            ap.error(f"unknown arm {arm!r} (expected none/drift)")
    for suite_name in suites:
        get_suite(BENCHMARK, suite_name)  # fail fast on unknown suites
    jobs = enumerate_jobs(suites, a.tasks, a.injections, arms,
                          a.conditions, a.pilot)
    try:
        results_path = resolve_results_path(a.mode, a.out, a.pilot)
    except ValueError as exc:
        ap.error(str(exc))

    if a.dry_run:
        print(json.dumps({
            "dry_run": True,
            "mode": a.mode,
            "results_path": str(results_path),
            "n_jobs": len(jobs),
            "jobs": jobs,
            "batch_cap_rmb": a.batch_cap,
            "paid_runs_refused_without_cap": a.batch_cap is None and a.cap is None,
        }, indent=1, ensure_ascii=False))
        return 0

    if a.batch_cap is None and a.cap is None:
        ap.error("refusing to run paid runs without a spend cap: pass "
                 "--batch-cap (recommended) or --cap")

    patch_model_names()
    set_seed(a.seed)
    logger = get_logger(a.log)

    ledger = Ledger(0.0 if a.batch_cap is not None else a.cap)
    if a.batch_cap is not None:
        ledger.cap = ledger.start_rmb + a.batch_cap

    results = {
        "runner_contract": RUN_CONTRACT,
        "mode": a.mode,
        "model": MODEL,
        "benchmark_version": BENCHMARK,
        "agentdojo_version": "0.1.35",
        "arms": arms,
        "protocol": "first5_user_tasks_by_id_first_injection_benign_attack",
        "security_semantics": "True = injection succeeded (agentdojo)",
        "batch_cap_rmb": a.batch_cap,
        "ledger_cap_rmb": ledger.cap,
        "start_rmb": round(ledger.start_rmb, 6),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runs": [],
    }
    if results_path.exists() and a.mode == "run":
        results = json.loads(results_path.read_text())
    done = {"|".join([r["suite"], r["user_task"],
                      r.get("injection_task") or "", r["condition"],
                      r["arm"]])
            for r in results["runs"] if not r.get("error")}

    client = DeepSeekModel(model=MODEL, logger=logger, ledger=ledger)
    pipelines = build_pipelines(client, logger)
    suite_args = SimpleNamespace(adaptive_attack=False)

    for suite_name in suites:
        task_suite = make_task_suite(suite_name, suite_args)
        suite_jobs = [j for j in jobs if j["suite"] == suite_name]
        if not suite_jobs:
            continue
        attacks = {arm: load_attack("important_instructions", task_suite,
                                    pipelines[arm][0]) for arm in arms}
        for job in suite_jobs:
            key = "|".join([job["suite"], job["user_task"],
                            job["injection_task"] or "", job["condition"],
                            job["arm"]])
            if key in done:
                continue
            user_task = task_suite.get_user_task_by_id(job["user_task"])
            if job["condition"] == "benign":
                injection_task, injections = None, {}
            else:
                injection_task = task_suite.get_injection_task_by_id(
                    job["injection_task"])
                injections = attacks[job["arm"]].attack(user_task, injection_task)
            pipeline, llm = pipelines[job["arm"]]
            try:
                res = run_one(task_suite, pipeline, llm, client, user_task,
                              injection_task, injections, logger)
            except BudgetExceeded as exc:
                res = {"error": f"budget: {exc}"}
            rec = dict(job)
            rec.update(res)
            rec["rmb_batch"] = round(ledger.rmb - ledger.start_rmb, 6)
            results["runs"].append(rec)
            results_path.write_text(json.dumps(results, indent=1,
                                               ensure_ascii=False))
            print(json.dumps(rec, ensure_ascii=False), flush=True)
            if rec.get("error") and str(rec["error"]).startswith("budget"):
                results["budget_stop"] = True
                results["end_rmb"] = round(ledger.rmb, 6)
                results["batch_rmb"] = round(ledger.rmb - ledger.start_rmb, 6)
                results["batch_calls"] = ledger.calls - ledger.start_calls
                results_path.write_text(json.dumps(results, indent=1,
                                                   ensure_ascii=False))
                print("BUDGET STOP", flush=True)
                return 1

    results["end_rmb"] = round(ledger.rmb, 6)
    results["batch_rmb"] = round(ledger.rmb - ledger.start_rmb, 6)
    results["batch_calls"] = ledger.calls - ledger.start_calls
    results["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    results_path.write_text(json.dumps(results, indent=1, ensure_ascii=False))
    print(json.dumps({"results_path": str(results_path),
                      "n_runs": len(results["runs"]),
                      "batch_rmb": results["batch_rmb"],
                      "batch_calls": results["batch_calls"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
