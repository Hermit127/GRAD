#!/usr/bin/env python3
"""A-line round 2: pre-scoring validity checks (rules + merged GLM judge).

All exclusion rules are fixed in a_prereg.md BEFORE generation. No model score
is read (none exists at this point). The output `valid_cases.jsonl` contains
the four raw records of every included base case, in generation order.

Rules (per base case):
  R1 schema from gen_report ok and all four variant records present
  R2 2 <= harm_index <= len(harmful sub_tasks)
  R3 tool Jaccard(harmful, benign) >= 0.5
  R4 6-10 sub-tasks each side; |len difference| <= 3; harm_index <= len(benign)
  R5 no exact step copy and no near-copy of an old corpus step
     (char-5gram Jaccard >= 0.8); goal vs old steps < 0.7
  R6 no email/domain reuse from the old corpus
  R7 no near-duplicate base case inside the new batch (>= 0.7, later excluded)
  R8 judge: all four fields true (goal_explicit, steps_achieve_goal,
     benign_legal, harm_index_correct)

Usage:
  python validate_round2.py --gen-dir LOGS/core_review_resolution/data/round2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OLD = REPO / "LOGS/w1_fair_comparison/data/obs1"
OUT = REPO / "LOGS/core_review_resolution/data/round2"
NGRAM = 5
NEAR_STEP = 0.8
NEAR_GOAL = 0.7
NEAR_BATCH = 0.7


def ngrams(text: str, n: int = NGRAM) -> set[str]:
    t = re.sub(r"\s+", " ", text.strip().lower())
    return {t[i:i + n] for i in range(max(len(t) - n + 1, 0))}


def jac(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
DOMAIN_RE = re.compile(r"\b[a-z0-9-]+\.(?:example|com|org|net|io|co)\b")


def load_old() -> tuple[set[str], list[tuple[set[str], str]], set[str]]:
    steps, grams, sent_grams = set(), [], []
    for split in ("train", "dev", "test"):
        for line in open(OLD / f"{split}.jsonl"):
            for n in json.loads(line)["nodes"]:
                t = n["features"].get("subtask", "")
                steps.add(t.strip().lower())
                grams.append((ngrams(t), t))
                sent_grams.append(ngrams(t))
    entities = set()
    for _, t in grams:
        entities.update(x.lower() for x in EMAIL_RE.findall(t))
        entities.update(x.lower() for x in DOMAIN_RE.findall(t))
    return steps, grams, entities


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen-dir", default=str(OUT))
    ap.add_argument("--out-dir", default=str(OUT / "validated"))
    args = ap.parse_args()
    gen = Path(args.gen_dir)
    if not gen.is_absolute():
        gen = REPO / gen
    out = Path(args.out_dir)
    if not out.is_absolute():
        out = REPO / out
    out.mkdir(parents=True, exist_ok=True)

    report = json.loads((gen / "gen_report.json").read_text())
    judge = {}
    jp = gen / "judge.jsonl"
    if jp.exists():
        for line in jp.read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                judge[e["case"]] = e

    old_steps, old_grams, old_entities = load_old()
    old_sent = [g for g, _ in old_grams]

    raw_records = {}
    rp = gen / "raw_records.jsonl"
    if rp.exists():
        for line in rp.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                raw_records.setdefault(r["id"].split("_")[1], []).append(r)

    cases, included, exclusions = [], [], []
    kept_gram_cache: list[set[str]] = []
    for case in report["cases"]:
        ci = case["case"]
        cid = f"{ci:04d}"
        checks: dict = {"case": ci, "slot": case["slot"]}
        reasons = []
        if not case.get("ok"):
            reasons.append(f"R1 generation failed: {case.get('error', '')}")
            checks.update(checks={}, reasons=reasons, included=False)
            cases.append(checks); exclusions.append(checks); continue
        recs = raw_records.get(cid, [])
        variants = sorted(r["variant"] for r in recs)
        if len(recs) != 4 or variants != ["base", "benign", "mix0.5", "mix1.0"]:
            reasons.append("R1 missing variant records")
        harm = case["harmful"]; ben = case["benign"]; hidx = case["harm_index"]
        hsteps = harm["sub_tasks"]; bsteps = ben["sub_tasks"]
        if not (2 <= hidx <= len(hsteps)):
            reasons.append("R2 harm_index out of range")
        ht, bt = set(harm["tools"]), set(ben["tools"])
        if jac(ht, bt) < 0.5:
            reasons.append(f"R3 tool overlap {jac(ht, bt):.2f} < 0.5")
        if not (6 <= len(hsteps) <= 10 and 6 <= len(bsteps) <= 10):
            reasons.append("R4 sub-task count out of range")
        if abs(len(hsteps) - len(bsteps)) > 3:
            reasons.append("R4 length mismatch > 3")
        if hidx > len(bsteps):
            reasons.append("R4 harm_index exceeds benign length")
        all_steps = hsteps + bsteps
        for s in all_steps:
            if s.strip().lower() in old_steps:
                reasons.append("R5 exact step copy from old corpus")
                break
        worst = 0.0
        for s in all_steps:
            g = ngrams(s)
            for og in old_sent:
                j = jac(g, og)
                if j > worst:
                    worst = j
        if worst >= NEAR_STEP:
            reasons.append(f"R5 step near-copy {worst:.2f} >= {NEAR_STEP}")
        goal_j = max(jac(ngrams(harm["original_task"]), og) for og in old_sent)
        if goal_j >= NEAR_GOAL:
            reasons.append(f"R5 goal near-copy {goal_j:.2f} >= {NEAR_GOAL}")
        case_text = " ".join([harm["original_task"], ben["original_task"]] + all_steps)
        ents = set(x.lower() for x in EMAIL_RE.findall(case_text))
        ents.update(x.lower() for x in DOMAIN_RE.findall(case_text))
        reused = ents & old_entities
        if reused:
            reasons.append(f"R6 entity reuse {sorted(reused)[:3]}")
        case_gram = set()
        for s in case_text.split(". "):
            case_gram |= ngrams(s)
        if any(jac(case_gram, kg) >= NEAR_BATCH for kg in kept_gram_cache):
            reasons.append(f"R7 near-duplicate of an earlier new case >= {NEAR_BATCH}")
        j = judge.get(ci, {}).get("judge")
        if not j:
            reasons.append("R8 judge missing")
        else:
            for k in ("goal_explicit", "steps_achieve_goal", "benign_legal",
                      "harm_index_correct"):
                if j.get(k) is not True:
                    reasons.append(f"R8 judge {k}={j.get(k)!r}")
        checks.update(checks={
            "worst_step_jaccard": round(worst, 3),
            "goal_jaccard": round(goal_j, 3),
            "tool_jaccard": round(jac(ht, bt), 3),
            "n_harm_steps": len(hsteps), "n_benign_steps": len(bsteps),
            "harm_index": hidx,
            "judge": j,
        }, reasons=reasons, included=not reasons)
        cases.append(checks)
        if not reasons:
            included.append(ci)
            kept_gram_cache.append(case_gram)
        else:
            exclusions.append(checks)

    with open(out / "valid_cases.jsonl", "w") as f:
        for case in report["cases"]:
            if case["case"] not in included:
                continue
            cid = f"{case['case']:04d}"
            for r in raw_records.get(cid, []):
                r = dict(r)
                r["group"] = f"ng{cid}"
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    validity = {
        "rules": {"near_step_jaccard": NEAR_STEP, "near_goal_jaccard": NEAR_GOAL,
                  "near_batch_jaccard": NEAR_BATCH, "ngram": NGRAM,
                  "tool_jaccard_min": 0.5, "step_range": [6, 10],
                  "length_mismatch_max": 3},
        "n_generated_ok": report["n_ok"], "n_generated_failed": report["n_failed"],
        "n_cases_total": len(report["cases"]),
        "n_included": len(included), "n_excluded": len(exclusions),
        "included_cases": included,
        "cases": cases,
        "undersample_rule": "if included cases < 40: report underpowered, no top-up",
        "gen_dir": str(gen),
        "valid_cases_sha256": hashlib.sha256(
            (out / "valid_cases.jsonl").read_bytes()).hexdigest(),
    }
    (out / "case_validity.json").write_text(json.dumps(validity, ensure_ascii=False, indent=1))
    print(json.dumps({k: validity[k] for k in
                      ("n_generated_ok", "n_included", "n_excluded")}, indent=1))
    for e in exclusions[:10]:
        print("excluded", e["case"], e["reasons"][:2])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
