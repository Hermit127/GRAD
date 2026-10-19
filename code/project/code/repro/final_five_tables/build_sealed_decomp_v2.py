#!/usr/bin/env python3
"""Build the Decomp v2 confirmation corpus from never-converted DecomposedHarm rows.

Provenance: the local HuggingFace cache of YuehHanChen/DecomposedHarm holds
combined_{train,val,test}.csv. Only 1165 rows were ever converted
(decomp_graphs_*_full.jsonl). The unused rows of combined_val + combined_test
(334 rows: 167 harmful / 167 benign) are DISJOINT in task id from every converted
graph. This script converts exactly those rows with the original converter and
applies the original augmentation pipeline (benign pool from the ORIGINAL train
full file, taint propagation, benign injection at ratios 0.5/1.0, fixed seed 42).

Output: code/repro/out/table5_decomp_v2/{test.jsonl, manifest.json}
"""
from __future__ import annotations

import ast
import csv
import re
import glob
import hashlib
import json
import os
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO / "code/repro"))

from decomposed_to_graph import task_to_dict  # noqa: E402
from rebuild_dataset import (  # noqa: E402
    build_benign_pool, inject_benign_into_attack, propagate_taint_on_graph,
)

CSV_BASE = glob.glob(os.path.expanduser(
    "~/.cache/huggingface/hub/datasets--YuehHanChen--DecomposedHarm/snapshots/*"))[0]
DATA = REPO / "ref/dataset"
OUT = REPO / "code/repro/out/table5_decomp_v2"
SEED = 42


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_csv(name: str) -> list[dict]:
    with open(f"{CSV_BASE}/combined_{name}.csv") as f:
        return list(csv.DictReader(f))


def main() -> int:
    random.seed(SEED)
    used_ids = set()
    for split in ("train", "validation", "test"):
        for line in (DATA / f"decomp_graphs_{split}_full.jsonl").read_text().splitlines():
            if line.strip():
                used_ids.add(json.loads(line)["task_id"])
    # original_task prompts present in any converted row (strict no-overlap rule)
    def norm(t):
        return re.sub(r"\s+", " ", str(t)).strip().lower()
    used_prompts = set()
    for split in ("train", "validation", "test"):
        for line in (DATA / f"decomp_graphs_{split}_full.jsonl").read_text().splitlines():
            if line.strip():
                used_prompts.add(norm(json.loads(line)["task_id"]))
    # map converted ids -> prompts from the CSVs
    csv_prompt = {}
    for name in ("train", "val", "test"):
        for r in load_csv(name):
            csv_prompt[r["id"]] = norm(r["original_task"])
    used_prompts = {csv_prompt[i] for i in used_ids if i in csv_prompt}

    rows = []
    excluded_prompt_dups = []
    for r in load_csv("val") + load_csv("test"):
        if r["id"] in used_ids:
            continue
        if norm(r["original_task"]) in used_prompts:
            excluded_prompt_dups.append(r["id"])
            continue
        # the HF datasets loader yields None for empty CSV fields; replicate it so
        # benign rows get harm_index=-1 exactly like the converted corpus
        if str(r.get("harm_index", "")).strip() == "":
            r = {**r, "harm_index": None}
        # Some rows store the decomposition as a Python-list string (image-prompt
        # sub-population); the HF loader presented text rows as newline-separated
        # strings, so normalise lists to the same newline-separated form.
        dec = str(r.get("decomposition", ""))
        if dec.lstrip().startswith("["):
            try:
                items = ast.literal_eval(dec)
                if isinstance(items, (list, tuple)):
                    r = {**r, "decomposition": "\n".join(str(x) for x in items)}
            except Exception:
                pass
        rows.append(r)
    assert len(rows) == 333, len(rows)

    # original benign pool: nodes of ORIGINAL train full benign graphs
    train_full = [json.loads(l) for l in
                  (DATA / "decomp_graphs_train_full.jsonl").read_text().splitlines() if l.strip()]
    benign_pool = build_benign_pool(train_full)

    full = []
    for i, r in enumerate(rows):
        g = task_to_dict(r, i)
        if g:
            propagate_taint_on_graph(g)
            full.append(g)
    mixed = []
    for g in full:
        if g["label"] == 1:
            for ratio in (0.5, 1.0):
                mg = inject_benign_into_attack(dict(g), benign_pool, ratio)
                mg["task_id"] = mg["task_id"] + f"_r{ratio}"
                mixed.append(mg)
    result = full + mixed
    random.shuffle(result)

    OUT.mkdir(parents=True, exist_ok=True)
    out_jsonl = OUT / "test.jsonl"
    with open(out_jsonl, "w") as f:
        for g in result:
            f.write(json.dumps(g) + "\n")

    # leakage audit against every converted corpus in ref/dataset and fix2 splits
    conv_ids = set()
    content = set()
    for p in glob.glob(str(DATA / "*.jsonl")):
        for line in Path(p).read_text().splitlines():
            if line.strip():
                g = json.loads(line)
                conv_ids.add(str(g.get("task_id")))
                content.add(hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest())
    new_ids = {str(g["task_id"]) for g in result}
    new_content = {hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()
                   for g in result}
    base_ids = {str(r["id"]) for r in rows}
    manifest = {
        "name": "sealed_decomp_v2",
        "purpose": "one-shot confirmation data for frozen methods (never converted)",
        "source_csv": {"dir": CSV_BASE, "files": ["combined_val.csv", "combined_test.csv"],
                       "sha256": {f: sha256(Path(f"{CSV_BASE}/{f}"))
                                  for f in ("combined_val.csv", "combined_test.csv")}},
        "seed_shuffle": SEED,
        "n_rows_used": len(rows),
        "class_balance_rows": {"harmful": sum(r["label"] == "harmful" for r in rows),
                               "benign": sum(r["label"] == "benign" for r in rows)},
        "n_graphs_full": len(full),
        "n_graphs_mixed_total": len(result),
        "class_balance_graphs": {
            "harmful": sum(g["label"] == 1 for g in result),
            "benign": sum(g["label"] == 0 for g in result)},
        "excluded_prompt_duplicates": excluded_prompt_dups,
        "task_id_disjoint_from_all_converted": len(new_ids & conv_ids) == 0,
        "base_id_disjoint": len(base_ids & conv_ids) == 0,
        "content_hash_overlap": len(new_content & content),
        "sha256": sha256(out_jsonl),
        "note": "converted with code/decomposed_to_graph.py::task_to_dict and augmented "
                "with code/rebuild_dataset.py functions (same pipeline, same seed); "
                "benign pool from the original train full file only",
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
