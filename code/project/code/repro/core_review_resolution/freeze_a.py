#!/usr/bin/env python3
"""A-line freeze manifest (pre-generation and post-generation stages)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CRR = REPO / "code/repro/core_review_resolution"
LOGS = REPO / "LOGS/core_review_resolution"
W1FIN = REPO / "LOGS/w1_final_confirmation"
OUT = LOGS / "a_freeze_manifest.json"

SCRIPTS = ["prompts_round2.py", "gen_round2.py", "judge_round2.py",
           "validate_round2.py", "build_round2.py", "power_round2.py",
           "a_stats.py", "run_score_round2.sh"]
DATA = ["raw_records.jsonl", "judge.jsonl", "gen_report.json",
        "validated/valid_cases.jsonl", "validated/case_validity.json",
        "confirm/graphs.jsonl", "confirm/meta.jsonl", "confirm/design.json"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["pre", "post"], required=True)
    ap.add_argument("--data-root", default=str(LOGS / "data/round2"))
    args = ap.parse_args()
    data_root = Path(args.data_root)

    man = json.loads(OUT.read_text()) if (OUT.exists() and args.stage == "post") else {}
    w1 = json.loads((W1FIN / "freeze_manifest.json").read_text())
    man["frozen_at"] = {"pre": "2026-09-26", "post": "2026-09-26"}[args.stage]
    man["question"] = ("can the frozen W1 GRAD/TXT/TEMP advantage be reproduced on "
                       "frozen, newly generated independent cases?")
    man["models"] = w1["models"]
    man["thresholds"] = {"file": "LOGS/w1_final_confirmation/frozen_thresholds.json",
                         "sha256": sha256_file(W1FIN / "frozen_thresholds.json"),
                         "source": "W1 dev only; frozen; never re-selected"}
    man["obs1_data"] = w1["data"]
    man["observation_cutoff"] = w1["observation_cutoff"]
    man["text_budget"] = w1["text_budget"]
    man["sample_size"] = {"base_cases": 60, "rows": 240,
                          "undersample_rule": "valid base cases < 40 -> underpowered, no top-up",
                          "attempts_per_case": 1}
    if args.stage == "pre":
        man["scripts"] = {s: sha256_file(CRR / s) for s in SCRIPTS}
        pw = LOGS / "power_round2.json"
        if pw.exists():
            man["power"] = json.loads(pw.read_text())
        man["generation_budget"] = {
            "estimate_rmb": "1.5-2.5 (deepseek-flash generation + glm-5.3-flash judge)",
            "requested_cap_rmb": 5.0, "hard_stop": True, "top_up": False}
    else:
        man["round2_data"] = {}
        for rel in DATA:
            p = data_root / rel
            man["round2_data"][rel] = sha256_file(p) if p.exists() else "MISSING"
        d = data_root / "confirm/design.json"
        if d.exists():
            man["round2_design"] = json.loads(d.read_text())
    OUT.write_text(json.dumps(man, indent=1))
    print(json.dumps({k: man[k] for k in man if k not in ("models", "round2_design")},
                     indent=1)[:2000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
