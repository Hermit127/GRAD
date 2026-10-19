#!/usr/bin/env python3
"""Recompute archived paper outcomes without model inference or API calls."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compare(actual, expected, path="result"):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or actual.keys() != expected.keys():
            raise ValueError(f"{path}: different keys")
        for key in expected:
            compare(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{path}: different lengths")
        for i, (a, e) in enumerate(zip(actual, expected)):
            compare(a, e, f"{path}[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if abs(actual - expected) > 1e-10:
            raise ValueError(f"{path}: numerical mismatch")
    elif actual != expected:
        raise ValueError(f"{path}: mismatch")


def t3(output):
    module = load_module("grad_t3", "code/project/code/repro/final_table4/aggregate.py")
    rows = json.loads((ROOT / "evidence/t3/outcomes.json").read_text())["runs"]
    resolved = list(module.resolve(rows).values())
    if len(resolved) != 240:
        raise ValueError("Expected 240 distinct frozen cases")
    actual = module.build_table(resolved, output, "Recorded T3 closed-loop outcomes", {})
    # The export contains outcomes, not raw traces. Correct the provenance label.
    provenance_path = output / "cell_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    for cell in provenance.values():
        cell["source"] = "evidence/t3/outcomes.json (exported outcome fields)"
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    expected = json.loads((ROOT / "evidence/t3/expected.json").read_text())
    compare({key: actual[key] for key in expected}, expected)
    return {"task": "t3", "verified_cases": len(resolved), "verified_cells": len(actual["cells"])}


def w1(output):
    import numpy as np
    module = load_module("grad_w4", "code/project/code/repro/w1_final_confirmation/final_stats.py")
    expected = json.loads((ROOT / "evidence/w1/expected.json").read_text())
    thresholds = json.loads((ROOT / "evidence/w1/frozen_thresholds.json").read_text())
    meta = json.loads((ROOT / "evidence/w1/test_metadata.json").read_text())
    labels = np.array([row["label"] for row in meta])
    actual = {"auroc": {}, "operating_points": {}}
    runs = {}
    for arm in module.ARMS:
        values = []
        for seed in module.SEEDS:
            run = module.load_w1_run(ROOT / "evidence/w1/runs", arm, seed)
            if not np.array_equal(labels, run["test_labels"]):
                raise ValueError("Test label order mismatch")
            runs[arm, seed] = run
            values.append(module.auroc(labels, run["test_scores"]))
        actual["auroc"][arm] = {"per_seed": values, "seed_mean": float(np.mean(values)),
                                "seed_std": float(np.std(values))}
    compare(actual["auroc"], expected["auroc"])
    for target in module.TARGETS:
        key = f"dev_fpr{int(target * 100):02d}"
        actual["operating_points"][key] = {}
        for arm in module.ARMS:
            rows = []
            for seed in module.SEEDS:
                run = runs[arm, seed]
                frozen = thresholds[f"{arm}|{seed}"][key]["threshold"]
                computed = module.threshold_for_target_fpr(run["dev_labels"], run["dev_scores"], target)[0]
                if abs(frozen - computed) > 1e-9:
                    raise ValueError("Frozen development threshold mismatch")
                rows.append(module.operating_point(labels, run["test_scores"], frozen))
            means = {k: float(np.mean([r[k] for r in rows]))
                     for k in ("tpr", "fpr", "tp", "fp", "tn", "fn")}
            compare(means, expected["operating_points"][key]["arms"][arm]["seed_mean"])
            actual["operating_points"][key][arm] = means
    (output / "w1_recomputed.json").write_text(json.dumps(actual, indent=2) + "\n")
    return {"task": "w1", "verified_examples": len(labels), "verified_runs": len(runs),
            "auroc": {a: v["seed_mean"] for a, v in actual["auroc"].items()},
            "scope": "Seed AUROC and development-frozen operating points; bootstrap intervals not rerun"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["t3", "w1"], default="t3")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    output = args.output.resolve() / args.task
    if output == ROOT or ROOT in output.parents and output.relative_to(ROOT).parts[0] != "outputs":
        parser.error("Use outputs/ or a directory outside the checkout")
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "passed", **{"t3": t3, "w1": w1}[args.task](output)}
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
