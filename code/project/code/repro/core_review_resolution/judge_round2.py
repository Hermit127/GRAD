#!/usr/bin/env python3
"""A-line round 2: independent validity judging with GLM-5.3-Flash.

Judges only the generated text (goal / sub-tasks / harm index / benign
counterpart); model scores do not exist yet. Frozen prompt: prompts_round2.JUDGE.
Budget-capped and fully logged via the frozen two-vendor budget client.

Usage (server):
  python code/repro/core_review_resolution/judge_round2.py \
      --gen-dir LOGS/core_review_resolution/data/round2 --cap-glm 1.0
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "code/repro/final_table4"))

import prompts_round2 as P  # noqa: E402
from budget_client import BudgetStop, GLM_FLASH, VendorBudgetClient, CallFailed  # noqa: E402

MAX_RETRIES = 2


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_json(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except json.JSONDecodeError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", required=True)
    ap.add_argument("--cap-glm", type=float, default=1.0)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    gen = Path(args.gen_dir)
    if not gen.is_absolute():
        gen = REPO / gen
    report = json.loads((gen / "gen_report.json").read_text())
    invocation = {
        "argv": sys.argv, "cap_glm_rmb": args.cap_glm, "model": GLM_FLASH.model,
        "max_retries": MAX_RETRIES,
        "prompts_sha256": sha256_file(Path(P.__file__)),
        "judge_sha256": sha256_file(Path(__file__)),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (gen / "judge_invocation.json").write_text(json.dumps(invocation, indent=1))
    if args.dry_run:
        from types import SimpleNamespace

        class _Stub:
            def __init__(self):
                self.state = {"calls": 0, "charged_actual_rmb": 0.0}
                outer = self

                class _Completions:
                    def create(self, **kw):
                        outer.state["calls"] += 1
                        body = json.dumps({"goal_explicit": True,
                                           "steps_achieve_goal": True,
                                           "benign_legal": True,
                                           "harm_index_correct": True,
                                           "notes": "dry-run stub"})
                        return SimpleNamespace(content=body, cost_rmb=0.0)

                self.chat = SimpleNamespace(completions=_Completions())

        client = _Stub()
    else:
        client = VendorBudgetClient(gen, GLM_FLASH, args.cap_glm, tag="round2judge")
    out_path = gen / "judge.jsonl"
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["case"])

    with open(out_path, "a") as f:
        for case in report["cases"]:
            ci = case["case"]
            if not case.get("ok") or ci in done:
                continue
            prompt = P.JUDGE.format(
                goal=case["harmful"]["original_task"],
                numbered=P.numbered(case["harmful"]["sub_tasks"]),
                harm_idx=case["harm_index"],
                benign_goal=case["benign"]["original_task"],
                benign_numbered=P.numbered(case["benign"]["sub_tasks"]))
            entry = {"case": ci, "slot": case["slot"]}
            try:
                for attempt in range(1, MAX_RETRIES + 1):
                    try:
                        resp = client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.0, max_tokens=600,
                            _tag=f"case{ci:04d}:judge:try{attempt}")
                    except CallFailed as exc:
                        entry.update(error=str(exc)[:200])
                        break
                    data = parse_json(resp.content)
                    if data is not None:
                        entry["judge"] = data
                        entry["attempt"] = attempt
                        entry["cost_rmb"] = resp.cost_rmb
                        break
                else:
                    entry.update(error="judge JSON parse failed")
            except BudgetStop as exc:
                entry.update(error=f"BudgetStop: {exc}")
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[judge] BUDGET STOP at case {ci}", flush=True)
                break
            except Exception as exc:  # noqa: BLE001
                entry.update(error=f"{type(exc).__name__}: {str(exc)[:200]}")
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[judge] case {ci} {entry.get('judge', entry.get('error'))}",
                  flush=True)

    print(json.dumps(client.state, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
