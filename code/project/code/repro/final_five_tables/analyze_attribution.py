#!/usr/bin/env python3
"""G3 attribution: component ablations, group-clustered paired CIs, deployment cost.

Inputs:
  code/repro/out/table5_g3_attrib/{dev,legacy}/*.json   component ablations
  code/repro/out/table5_g3/{dev,confirm}/*.json          G3_p25
  code/repro/out/table5_ablation/runs/*.json             references (K0/structure/...)
  code/repro/out/table5_confirm/confirm/*_sealed_*.json  structure on held-out split
Outputs:
  LOGS/final_five_tables/table5/attribution.{json,md}
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
ATTR = REPO / "code/repro/out/table5_g3_attrib"
G3 = REPO / "code/repro/out/table5_g3"
REF = REPO / "code/repro/out/table5_ablation/runs"
ARCH = REPO / "code/repro/out/unified_gnn/runs"
SEALED_REF = REPO / "code/repro/out/table5_confirm/confirm"
OUT = REPO / "LOGS/final_five_tables/table5"
SEEDS = [211, 212, 213]
CELLS = {"decomp": ["fold0"], "mcp": [f"fold{k}" for k in range(5)],
         "mrt": [f"fold_{t}" for t in ("american_airlines_2", "api_key_calendar_agendas_2",
                                       "bank_transfer_2", "expenditure_accounting",
                                       "spam_filter_update")]}
ABLATIONS = {"G3": "G3_p25", "G3_no_ln": "G3_no_ln", "G3_no_aux": "G3_no_aux",
             "G3_no_moddrop": "G3_no_moddrop", "S3_reg": "S3_reg"}


def auroc(y, s):
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    pos, neg = y == 1, y == 0
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan")
    order = np.argsort(s); ss = s[order]
    ranks = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / 2)
                 / (pos.sum() * neg.sum()))


def group_of(ds, g):
    if ds == "decomp":
        return str(g.get("group") or g.get("task_id"))
    if ds == "mcp":
        try:
            import sys as _s
            _s.path.insert(0, str(REPO / "code/repro/static_big_table_v1"))
            import groups as _groups
            return str(_groups.group_mcp(g))
        except Exception:
            md = g.get("metadata", {}) or {}
            t = str(md.get("task") or g.get("task_id"))
            return t[:-7] if t.endswith("_benign") else t
    return str(g.get("task_id", "")).split("_transcripts_")[0]


def load_rows(path, split="test"):
    d = json.loads(path.read_text())
    return d.get("scores", {}).get(split, [])


def path_for(kind, label, ds, cell, seed):
    if kind == "attr":
        p = ATTR / "legacy" / f"{label}_{ds}_{cell}_s{seed}.json"
        if p.exists():
            return p
        return ATTR / "dev" / f"{label}_{ds}_{cell}_s{seed}.json"
    if kind == "g3_legacy":
        return G3 / "confirm" / f"{label}_{ds}_{cell}_s{seed}.json"
    if kind == "g3_dev":
        return G3 / "dev" / f"{label}_{ds}_{cell}_s{seed}.json"
    if kind == "ref":
        p = REF / f"{ds}_{cell}_{label}_s{seed}.json"
        if not p.exists() and ds == "mrt" and label == "full":
            p = ARCH / f"mrt_{cell.replace('fold_', '')}_s{seed}.json"
        return p
    raise ValueError(kind)


def seed_means(kind, label, ds, split="test"):
    out = {}
    for seed in SEEDS:
        cells = []
        for cell in CELLS[ds]:
            p = path_for(kind, label, ds, cell, seed)
            if not p.exists():
                continue
            rows = load_rows(p, split)
            if not rows:
                continue
            y = [r["label"] for r in rows]; s = [r["score"] for r in rows]
            a = auroc(y, s)
            npos = sum(y)
            cells.append((a, npos * (len(y) - npos)))
        if cells:
            out[seed] = float(np.average([a for a, _ in cells],
                                         weights=[w for _, w in cells]))
    return out


# ---------- group-clustered paired CI: G3 vs structure (existing runs) ----------

def clustered_ci(ds):
    """Pool each seed's test graphs (deduped), pair G3 vs structure, bootstrap groups."""
    pairs = []
    for seed in SEEDS:
        g3_rows, st_rows, groups, labels = {}, {}, {}, {}
        for cell in CELLS[ds]:
            if ds == "mcp":
                gpath = G3 / "confirm" / f"G3_p25_mcp_{cell}_s{seed}.json"
                spath = SEALED_REF / f"structure_only_mcp_{cell}_s{seed}.json"
                if not spath.exists():
                    spath = REF / f"mcp_{cell}_structure_only_s{seed}.json"
                graph_path = (REPO / "code/repro/out/unified_gnn/fix2_splits/mcp" /
                              cell / "test.jsonl")
            elif ds == "decomp":
                gpath = G3 / "confirm" / f"G3_p25_decomp_fold0_s{seed}.json"
                spath = REF / f"decomp_fold0_structure_only_s{seed}.json"
                graph_path = (REPO / "code/repro/out/unified_gnn/fix2_splits/decomp" /
                              "fold0" / "test.jsonl")
            else:
                gpath = G3 / "confirm" / f"G3_p25_mrt_{cell}_s{seed}.json"
                spath = REF / f"mrt_{cell}_structure_only_s{seed}.json"
                graph_path = REPO / "results/mrt_cross_task_v1" / cell / "admission.jsonl"
            if not (gpath.exists() and spath.exists()):
                continue
            graphs = [json.loads(l) for l in graph_path.read_text().splitlines() if l.strip()]
            gr = load_rows(gpath); sr = load_rows(spath)
            if len(gr) != len(graphs) or len(sr) != len(graphs):
                continue
            for g, rg, rs in zip(graphs, gr, sr):
                assert g["label"] == rg["label"] == rs["label"]
                tid = str(g.get("task_id"))
                if tid in g3_rows:
                    continue
                g3_rows[tid] = rg["score"]; st_rows[tid] = rs["score"]
                labels[tid] = int(g["label"])
                groups[tid] = group_of(ds, g)
        if not g3_rows:
            continue
        tids = sorted(g3_rows)
        ys = [labels[t] for t in tids]
        gs = np.array([g3_rows[t] for t in tids]); ss = np.array([st_rows[t] for t in tids])
        d0 = auroc(ys, gs) - auroc(ys, ss)
        rng = np.random.default_rng(0)
        by_group = defaultdict(list)
        for i, t in enumerate(tids):
            by_group[groups[t]].append(i)
        gl = sorted(by_group)
        deltas = []
        for _ in range(10000):
            pick = rng.choice(len(gl), len(gl), replace=True)
            idx = [i for gi in pick for i in by_group[gl[gi]]]
            yy = np.asarray(ys)[idx]
            if yy.sum() in (0, len(yy)):
                continue
            deltas.append(auroc(yy, gs[idx]) - auroc(yy, ss[idx]))
        lo, hi = np.percentile(deltas, [2.5, 97.5]) if deltas else (float("nan"), float("nan"))
        pairs.append({"seed": seed, "n_graphs": len(tids), "n_groups": len(gl),
                      "delta": float(d0), "ci95_group_cluster": [float(lo), float(hi)]})
    out = {"per_seed": pairs}
    if pairs:
        out["seed_level_mean_delta"] = float(np.mean([p["delta"] for p in pairs]))
        out["seed_level_std"] = float(np.std([p["delta"] for p in pairs]))
    return out


# ---------- deployment cost ----------

def deployment_cost():
    import torch  # local import keeps the analysis light
    import sys
    sys.path.insert(0, str(REPO / "code"))
    sys.path.insert(0, str(REPO / "code/repro/final_five_tables"))
    import train_gnn_ablate as abl
    import train_gnn_ablate2 as a2
    from torch_geometric.data import Batch

    graphs = [json.loads(l) for l in
              (REPO / "code/repro/out/unified_gnn/fix2_splits/mcp/fold0/test.jsonl")
              .read_text().splitlines() if l.strip()][:32]
    cfg = {"candidate": "full_ref", "edge_mode": "rebuilt", "edge_text": "copy",
           "edge_dropout": 0.0}
    data = [abl.graph_to_v2_data(g) for g in graphs]
    batch = Batch.from_data_list(data)
    full = abl.CausalCharGNN(22, len(abl.V2_EDGE_TYPES) + 1, hidden_dim=192)
    g3 = a2.ModalityNormGNN(22, len(abl.V2_EDGE_TYPES) + 1, hidden=192)
    full.eval(); g3.eval()
    import time

    def bench(model, fn, reps=5):
        with torch.no_grad():
            fn()  # warmup
            t0 = time.perf_counter()
            for _ in range(reps):
                fn()
            return (time.perf_counter() - t0) / reps * 1000 / len(graphs)

    def full_fwd():
        return abl.batch_logits(full, batch)

    def g3_fwd():
        return a2.batch_logits(g3, batch, {"candidate": "A_moddrop"}, training=False)

    # legacy behaviour: execute aux heads at eval too
    def g3_aux_fwd():
        return g3(batch, moddrop_p=0.0, training=True)

    out = {
        "batch_graphs": len(graphs),
        "params_full": sum(p.numel() for p in full.parameters()),
        "params_g3_total": sum(p.numel() for p in g3.parameters()),
        "params_g3_aux_heads": sum(p.numel() for m in
                                   (g3.aux_text_proj, g3.aux_text_attn, g3.aux_text_head,
                                    g3.aux_struct_proj, g3.aux_struct_attn, g3.aux_struct_head)
                                   for p in m.parameters()),
        "ms_per_graph_full": bench(full, full_fwd),
        "ms_per_graph_g3_eval_aux_skipped": bench(g3, g3_fwd),
        "ms_per_graph_g3_legacy_aux_executed": bench(g3, g3_aux_fwd),
        "note": "CPU timing, random weights, architecture-only; aux heads are not "
                "executed in eval mode (see test_g3_attribution.py)",
    }
    return out


def main() -> int:
    out = {"experiment_id": "table5-g3-attribution-20260919",
           "component_ablations": {"dev": {}, "legacy": {}},
           "vs_structure_group_clustered": {},
           "deployment": {}}
    for mode, split in (("dev", "val"), ("legacy", "test")):
        for ds in ("decomp", "mcp", "mrt"):
            out["component_ablations"][mode][ds] = {}
            for name, label in ABLATIONS.items():
                kind = "g3_dev" if (mode == "dev" and name == "G3") else (
                    "g3_legacy" if name == "G3" else "attr")
                vals = seed_means(kind, label, ds, split)
                if vals:
                    out["component_ablations"][mode][ds][name] = {
                        "per_seed": {str(k): round(v, 4) for k, v in vals.items()},
                        "mean": float(np.mean(list(vals.values())))}
            for ref in ("full", "structure_only"):
                vals = seed_means("ref", ref, ds, split)
                if vals:
                    out["component_ablations"][mode][ds][ref] = {
                        "per_seed": {str(k): round(v, 4) for k, v in vals.items()},
                        "mean": float(np.mean(list(vals.values())))}
    for ds in ("mcp", "decomp", "mrt"):
        out["vs_structure_group_clustered"][ds] = clustered_ci(ds)
    try:
        out["deployment"] = deployment_cost()
    except Exception as e:  # timing is best-effort, never blocks the report
        out["deployment"] = {"error": str(e)}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "attribution.json").write_text(json.dumps(out, indent=1, default=str))

    md = ["# G3 attribution, uncertainty, and deployment cost", "",
          "## 1. Component ablations (dev, test sealed)", "",
          "| Dataset | K0 full | G3 | G3 w/o LayerNorm | G3 w/o aux | G3 w/o moddrop | S3_reg (structure+reg) | structure-only |",
          "|---|---|---|---|---|---|---|---|"]
    def g(mode, ds, name):
        v = out["component_ablations"][mode][ds].get(name)
        return "--" if not v else f"{v['mean']:.4f}"
    for ds in ("decomp", "mcp", "mrt"):
        md.append(f"| {ds} | {g('dev', ds, 'full')} | {g('dev', ds, 'G3')} | "
                  f"{g('dev', ds, 'G3_no_ln')} | {g('dev', ds, 'G3_no_aux')} | "
                  f"{g('dev', ds, 'G3_no_moddrop')} | {g('dev', ds, 'S3_reg')} | "
                  f"{g('dev', ds, 'structure_only')} |")
    md += ["", "## 2. Component ablations (legacy test)", "",
           "| Dataset | K0 full | G3 | G3 w/o LayerNorm | G3 w/o aux | G3 w/o moddrop | S3_reg | structure-only |",
           "|---|---|---|---|---|---|---|---|"]
    for ds in ("decomp", "mcp", "mrt"):
        md.append(f"| {ds} | {g('legacy', ds, 'full')} | {g('legacy', ds, 'G3')} | "
                  f"{g('legacy', ds, 'G3_no_ln')} | {g('legacy', ds, 'G3_no_aux')} | "
                  f"{g('legacy', ds, 'G3_no_moddrop')} | {g('legacy', ds, 'S3_reg')} | "
                  f"{g('legacy', ds, 'structure_only')} |")
    md += ["", "## 3. G3 vs structure-only, group-clustered paired CI", "",
           "| Dataset | seed | n graphs | n groups | ΔAUROC | 95% CI (group bootstrap) |",
           "|---|---|---|---|---|---|"]
    for ds, d in out["vs_structure_group_clustered"].items():
        for p in d.get("per_seed", []):
            md.append(f"| {ds} | {p['seed']} | {p['n_graphs']} | {p['n_groups']} | "
                      f"{p['delta']:+.4f} | [{p['ci95_group_cluster'][0]:+.4f}, "
                      f"{p['ci95_group_cluster'][1]:+.4f}] |")
        if d.get("seed_level_mean_delta") is not None:
            note = ("  (only 5 task clusters - CI unreliable, treat as descriptive)"
                    if ds == "mrt" else "")
            md.append(f"| {ds} | seed mean | | | {d['seed_level_mean_delta']:+.4f} "
                      f"± {d['seed_level_std']:.4f} |{note} |")
    dep = out["deployment"]
    md += ["", "## 4. Deployment cost", ""]
    if "error" in dep:
        md.append(f"timing error: {dep['error']}")
    else:
        md += [f"- params: full {dep['params_full']:,}; G3 total {dep['params_g3_total']:,} "
               f"(aux heads {dep['params_g3_aux_heads']:,}, train-only).",
               f"- inference (same batch, CPU, architecture-only): full "
               f"{dep['ms_per_graph_full']:.2f} ms/graph; G3 eval (aux skipped) "
               f"{dep['ms_per_graph_g3_eval_aux_skipped']:.2f}; G3 with legacy "
               f"aux-at-eval {dep['ms_per_graph_g3_legacy_aux_executed']:.2f}.",
               f"- {dep['note']}"]
    (OUT / "attribution.md").write_text("\n".join(md) + "\n")
    print("wrote attribution.json/md")
    print("\n".join(md[:40]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
