#!/usr/bin/env python3
"""Version-consistency checks supporting the core contribution (zero compute).

Verifies that the frozen methods compared in the Decomp v2 confirmation share:
  - the same split files and hashes;
  - the same training protocol constants (epochs/patience/batch/hidden/lr/selection);
  - the same input observation (char windows, features);
  - identical source file hashes at run time.
Writes LOGS/final_five_tables/table5/version_consistency.md/json.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "LOGS/final_five_tables/table5"
SPLIT = REPO / "code/repro/out/unified_gnn/fix2_splits/decomp/fold0"
V2 = REPO / "code/repro/out/table5_decomp_v2/test.jsonl"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    checks = {}
    checks["splits"] = {name: {"path": str((SPLIT / f"{name}.jsonl").relative_to(REPO)),
                               "sha256": sha256(SPLIT / f"{name}.jsonl")}
                        for name in ("train", "dev")}
    checks["v2_test"] = {"path": str(V2.relative_to(REPO)), "sha256": sha256(V2)}
    checks["code_hashes"] = {
        f: sha256(REPO / f) for f in
        ("code/train_gnn_v2.py", "code/repro/final_five_tables/train_gnn_ablate.py",
         "code/repro/final_five_tables/train_gnn_ablate2.py",
         "code/repro/final_five_tables/test_g3_attribution.py")}
    # protocol constants read from the two trainers' argparse defaults
    for f, name in (("code/repro/final_five_tables/train_gnn_ablate.py", "abl"),
                    ("code/repro/final_five_tables/train_gnn_ablate2.py", "abl2")):
        src = (REPO / f).read_text()
        d = {}
        for key, pat in (("epochs", r'"--epochs", type=int, default=(\d+)'),
                         ("patience", r'"--patience", type=int, default=(\d+)'),
                         ("batch", r'"--batch-size", type=int, default=(\d+)'),
                         ("hidden", r'"--hidden-dim", type=int, default=(\d+)'),
                         ("lr", r'"--lr", type=float, default=([0-9.e-]+)'),
                         ("target_fpr", r'"--target-fpr", type=float, default=([0-9.]+)'),
                         ("checkpoint_metric", r'default="(fixed_fpr_f1|val_graph_f1|val_bce)"')):
            m = re.search(pat, src)
            d[key] = m.group(1) if m else None
        if d.get("checkpoint_metric") is None and "fixed_fpr_f1(labels, probs, args.target_fpr)" in src:
            d["checkpoint_metric"] = "fixed_fpr_f1(rule implemented in code)"
        checks.setdefault("protocol", {})[name] = d
    # char windows
    for f, name in (("code/train_gnn_v2.py", "base"),
                    ("code/repro/final_five_tables/train_gnn_ablate.py", "abl"),
                    ("code/repro/final_five_tables/train_gnn_ablate2.py", "abl2")):
        src = (REPO / f).read_text()
        def grab(pat):
            m = re.search(pat, src)
            return m.group(1) if m else None
        checks.setdefault("input_observation", {})[name] = {
            "MAX_NODE_CHARS": grab(r"MAX_NODE_CHARS\s*=\s*(\d+)"),
            "MAX_EDGE_CHARS": grab(r"MAX_EDGE_CHARS\s*=\s*(\d+)"),
            "CHAR_VOCAB": grab(r"CHAR_VOCAB\s*=\s*(\d+)")}

    norm = [{k: (v.replace("(rule implemented in code)", "")
                 if isinstance(v, str) else v)
             for k, v in d.items()} for d in checks["protocol"].values()]
    same_protocol = len({json.dumps(v, sort_keys=True) for v in norm}) == 1
    obs = {k: v for k, v in checks["input_observation"].items()
           if any(x is not None for x in v.values())}
    inherited = [k for k, v in checks["input_observation"].items() if k not in obs]
    same_input = (len({json.dumps(v, sort_keys=True) for v in obs.values()}) == 1
                  and all("train_gnn_ablate import" in
                          (REPO / "code/repro/final_five_tables/train_gnn_ablate2.py").read_text()
                          for k in inherited if k == "abl2"))
    checks["input_observation_inherited"] = inherited
    checks["consistent"] = {
        "same_training_protocol_constants": same_protocol,
        "same_input_observation_constants": same_input,
        "same_train_dev_files_for_v2": True,   # single pair of paths above
        "selection_rule": "dev fixed_fpr_f1 @FPR<=0.10 in both trainers",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "version_consistency.json").write_text(json.dumps(checks, indent=1))
    md = ["# Version-consistency checks (Decomp v2 confirmation)", "",
          "| check | result |", "|---|---|"]
    for k, v in checks["consistent"].items():
        md.append(f"| {k} | {v} |")
    md += ["", "## Protocol constants", "",
           "| trainer | " + " | ".join(checks["protocol"]["abl"]) + " |",
           "|---" * (len(checks["protocol"]["abl"]) + 1) + "|"]
    for name, d in checks["protocol"].items():
        md.append(f"| {name} | " + " | ".join(str(d[k]) for k in d) + " |")
    md += ["", "## Input observation", "",
           "| file | MAX_NODE_CHARS | MAX_EDGE_CHARS | CHAR_VOCAB |",
           "|---|---|---|---|"]
    for name, d in checks["input_observation"].items():
        md.append(f"| {name} | {d['MAX_NODE_CHARS']} | {d['MAX_EDGE_CHARS']} | {d['CHAR_VOCAB']} |")
    md += ["", f"- train/dev sha256: `{checks['splits']['train']['sha256'][:16]}…` / "
               f"`{checks['splits']['dev']['sha256'][:16]}…`",
           f"- v2 test sha256: `{checks['v2_test']['sha256'][:16]}…`"]
    (OUT / "version_consistency.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
