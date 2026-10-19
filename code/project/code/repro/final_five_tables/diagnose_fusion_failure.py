#!/usr/bin/env python3
"""Fusion-failure diagnosis: why the semantic channel hurts on MCP/MRT.

PI directive 2026-09-21 (Discussion.md): stop sealing, diagnose why adding the
semantic channel makes the joint model worse than structure-only, before any
new fusion design.

Four questions, dev/train only, zero API/GPU:
  Q1 where do "structure right / fused wrong" samples concentrate
     (label, group, cell, size, truncation, text length, node-type composition,
      attack_category / MR task)?
  Q2 text channel input quality: truncation at MAX_NODE_CHARS=192, duplicates,
     per-graph truncated-node fraction;
  Q3 does the semantic channel corrupt structure ranking: pair-level flip rates
     and the G3 aux-head decomposition (s_s = structure aux, s_t = text aux);
  Q4 structure-channel strength: genuine or shortcut: trivial-feature LR / kNN
     probes on train->dev, rebuilt-edge low-level stats, group purity.

Inputs (stored runs, already-produced):
  code/repro/out/table5_g3/confirm/G3_p25_<ds>_<cell>_s<seed>.json   fused
  code/repro/out/table5_ablation/runs/<ds>_<cell>_<variant>_s<seed>.json
  split jsonl referenced by each run's config.val (mapped to local paths)
Outputs:
  LOGS/final_five_tables/table5/fusion_failure_diagnosis.{json,md}
Only scores['val'] is ever read; test arrays are not loaded.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code/repro/final_five_tables"))
from analyze_attribution import auroc, group_of, CELLS, SEEDS  # noqa: E402

REF = REPO / "code/repro/out/table5_ablation/runs"
G3 = REPO / "code/repro/out/table5_g3/confirm"
OUT = REPO / "LOGS/final_five_tables/table5"
MAX_NODE_CHARS = 192
KNN_K = 5

import train_gnn_ablate as tga  # noqa: E402  rebuild_v2_edges / node_text


def resolve_local(path_str: str) -> Path | None:
    p = str(path_str or "")
    for marker, base in (("/fix2_splits/", REPO / "code/repro/out/unified_gnn/fix2_splits"),
                         ("/mrt_cross_task_v1/", REPO / "results/mrt_cross_task_v1")):
        if marker in p:
            return base / p.split(marker, 1)[1]
    return None


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]


def val_rows(path: Path) -> tuple[dict, list[dict]]:
    d = json.loads(path.read_text())
    return d, d.get("scores", {}).get("val", [])


def rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def margins(labels, scores) -> np.ndarray:
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    out = np.full(len(scores), np.nan)
    for i, (y, s) in enumerate(zip(labels, scores)):
        opp = neg if y == 1 else pos
        if opp.size:
            above = float(np.mean(s > opp) + 0.5 * np.mean(s == opp))
            out[i] = above if y == 1 else 1.0 - above
    return out


def graph_feats(g: dict) -> dict:
    nodes = g.get("nodes", [])
    lens, texts, kw = [], [], 0
    for n in nodes:
        f = n.get("features") or {}
        s = str(f.get("subtask", "") or "")
        lens.append(len(s))
        if s:
            texts.append(s)
        if f.get("dangerous_pattern") or f.get("harm_step"):
            kw += 1
    n_nodes = len(nodes)
    n_chars = max(1, sum(lens))
    return {
        "n_nodes": n_nodes,
        "n_raw_edges": len(g.get("edges", [])),
        "text_len": sum(lens),
        "mean_node_chars": sum(lens) / max(1, n_nodes),
        "trunc_frac": sum(1 for L in lens if L > MAX_NODE_CHARS) / max(1, n_nodes),
        "kw_frac": kw / max(1, n_nodes),
        "dup_ratio": (len(texts) - len(set(texts))) / max(1, len(texts)),
        "n_chars": n_chars,
        "types": Counter(str(n.get("type", "")) for n in nodes),
    }


def edge_stats(g: dict) -> dict:
    try:
        es = tga.rebuild_v2_edges(g)
    except Exception:
        return {}
    n = len(es)
    if not n:
        return {"n_edges": 0, "seq_frac": 0.0, "data_flow_frac": 0.0, "mem_frac": 0.0,
                "mean_deg": 0.0, "max_deg": 0.0}
    tc = Counter(e["type"] for e in es)
    deg = Counter(e["source"] for e in es)
    degs = list(deg.values())
    return {
        "n_edges": n,
        "seq_frac": tc["sequence_next"] / n,
        "data_flow_frac": tc["data_flow"] / n,
        "mem_frac": (tc["memory_retrieve"] + tc["memory_store"]) / n,
        "mean_deg": float(np.mean(degs)),
        "max_deg": float(np.max(degs)),
    }


TRIVIAL_KEYS = ["n_nodes", "n_raw_edges", "text_len", "mean_node_chars",
                "trunc_frac", "kw_frac", "dup_ratio"]
EDGE_KEYS = ["n_edges", "seq_frac", "data_flow_frac", "mem_frac", "mean_deg", "max_deg"]


def collect_unit(ds: str, cell: str, seed: int) -> dict | None:
    g3p = G3 / f"G3_p25_{ds}_{cell}_s{seed}.json"
    sp = REF / f"{ds}_{cell}_structure_only_s{seed}.json"
    tp = REF / f"{ds}_{cell}_text_only_s{seed}.json"
    cp = REF / f"{ds}_{cell}_chrono_only_s{seed}.json"
    fp = REF / f"{ds}_{cell}_full_s{seed}.json"
    if not (g3p.exists() and sp.exists()):
        return None
    g3d, g3r = val_rows(g3p)
    split = resolve_local((g3d.get("config") or {}).get("val", ""))
    if split is None or not split.exists():
        return None
    graphs = load_jsonl(split)
    rows_by_tid: dict[str, dict] = {}
    for g, r in zip(graphs, g3r):
        assert int(g["label"]) == int(r["label"]), (ds, cell, seed, g.get("task_id"))
        tid = str(g.get("task_id"))
        feats = graph_feats(g)
        md = g.get("metadata") or {}
        rows_by_tid[tid] = {
            "tid": tid, "ds": ds, "cell": cell, "seed": seed,
            "label": int(g["label"]), "group": group_of(ds, g),
            "meta": {k: md.get(k) for k in ("attack_category", "task", "task_name") if md.get(k) is not None},
            "scores": {"fused": float(r["score"])},
            "aux": {"s_s": r.get("s_s"), "s_t": r.get("s_t")},
            **{k: v for k, v in feats.items() if k != "types"},
            "types": feats["types"],
            **edge_stats(g),
        }
    for label, path, key in (("struct", sp, "struct"), ("text", tp, "text"),
                             ("chrono", cp, "chrono"), ("full", fp, "full")):
        if not path.exists():
            continue
        _, rr = val_rows(path)
        if len(rr) != len(graphs):
            continue
        for g, r in zip(graphs, rr):
            tid = str(g.get("task_id"))
            if tid in rows_by_tid:
                rows_by_tid[tid]["scores"][key] = float(r["score"])
    rows = list(rows_by_tid.values())
    if len(rows) < 4:
        return None
    for k in ("s_s", "s_t"):
        vals = [r["aux"].get(k) for r in rows]
        if all(v is not None for v in vals):
            for r, v in zip(rows, vals):
                r["scores"][k] = float(v)
    y = [r["label"] for r in rows]
    for k in ("fused", "struct", "text", "chrono", "full", "s_s", "s_t"):
        vals = [r["scores"].get(k) for r in rows]
        if any(v is None for v in vals):
            continue
        mg = margins(y, vals)
        for r, m in zip(rows, mg):
            r[f"margin_{k}"] = m
    return {"ds": ds, "cell": cell, "seed": seed, "rows": rows,
            "auroc": {k: auroc(y, [r["scores"][k] for r in rows]) for k in rows[0]["scores"]}}


def pair_counts(rows: list[dict], key_a: str, key_b: str) -> dict:
    pos = [r for r in rows if r["label"] == 1 and key_a in r["scores"] and key_b in r["scores"]]
    neg = [r for r in rows if r["label"] == 0 and key_a in r["scores"] and key_b in r["scores"]]
    out = {"n_pairs": 0, "a_correct": 0, "b_correct": 0, "a_only": 0, "b_only": 0,
           "both": 0, "neither": 0}
    for p in pos:
        for n in neg:
            out["n_pairs"] += 1
            a_ok = p["scores"][key_a] > n["scores"][key_a]
            b_ok = p["scores"][key_b] > n["scores"][key_b]
            out["a_correct"] += a_ok
            out["b_correct"] += b_ok
            if a_ok and b_ok:
                out["both"] += 1
            elif a_ok:
                out["a_only"] += 1
            elif b_ok:
                out["b_only"] += 1
            else:
                out["neither"] += 1
    return out


def bucket(v, edges):
    for e in edges:
        if v <= e:
            return f"<= {e}"
    return f"> {edges[-1]}"


def main() -> None:
    units = []
    for ds in ("mcp", "decomp", "mrt"):
        for cell in CELLS[ds]:
            for seed in SEEDS:
                u = collect_unit(ds, cell, seed)
                if u is not None:
                    units.append(u)
    if not units:
        raise SystemExit("no units collected")

    report: dict = {"note": "dev-only; scores.test never read", "units": []}
    for u in units:
        rows = u["rows"]
        pc_struct_fused = pair_counts(rows, "struct", "fused")
        pc_fused_struct = pair_counts(rows, "fused", "struct")
        report["units"].append({
            "ds": u["ds"], "cell": u["cell"], "seed": u["seed"],
            "n": len(rows), "n_pos": sum(r["label"] for r in rows),
            "auroc": u["auroc"],
            "pairs_struct_vs_fused": {
                "n_pairs": pc_struct_fused["n_pairs"],
                "struct_correct": pc_struct_fused["a_correct"],
                "fused_wrong_where_struct_correct": pc_struct_fused["a_only"],
                "fused_correct_where_struct_wrong": pc_fused_struct["a_only"],
                "both_correct": pc_struct_fused["both"],
                "neither": pc_struct_fused["neither"],
            },
            "spearman": {
                "fused_struct": spearman([r["scores"]["fused"] for r in rows],
                                         [r["scores"]["struct"] for r in rows]),
                "fused_text": spearman([r["scores"]["fused"] for r in rows],
                                       [r["scores"]["text"] for r in rows]) if "text" in rows[0]["scores"] else None,
            },
        })

    by_ds: dict[str, list[dict]] = defaultdict(list)
    for u in units:
        by_ds[u["ds"]].append(u)

    summary: dict = {"pair_corruption": {}, "target_concentration": {},
                     "truncation": {}, "channel_override": {}, "shortcut": {},
                     "per_seed_delta": {}}

    for ds, us in by_ds.items():
        agg = {"n_pairs": 0, "struct_correct": 0, "corrupt": 0, "rescue": 0,
               "both_correct": 0, "neither": 0, "units": 0}
        per_seed_rates = defaultdict(list)
        for u in us:
            rows = u["rows"]
            pc_sf = pair_counts(rows, "struct", "fused")
            pc_fs = pair_counts(rows, "fused", "struct")
            agg["n_pairs"] += pc_sf["n_pairs"]
            agg["struct_correct"] += pc_sf["a_correct"]
            agg["corrupt"] += pc_sf["a_only"]
            agg["rescue"] += pc_fs["a_only"]
            agg["both_correct"] += pc_sf["both"]
            agg["neither"] += pc_sf["neither"]
            agg["units"] += 1
            if pc_sf["a_correct"]:
                per_seed_rates[u["seed"]].append(pc_sf["a_only"] / pc_sf["a_correct"])
        agg["corruption_rate"] = agg["corrupt"] / max(1, agg["struct_correct"])
        agg["rescue_rate"] = agg["rescue"] / max(1, agg["n_pairs"] - agg["struct_correct"])
        per_seed = {s: float(np.mean(v)) for s, v in per_seed_rates.items()}
        summary["per_seed_delta"][ds] = per_seed
        summary["pair_corruption"][ds] = agg

        # Q1: structure-right samples split by fused outcome
        tgt = [(u, r) for u in us for r in u["rows"]
               if r.get("margin_struct") is not None and r.get("margin_fused") is not None
               and r["margin_struct"] > 0.5 and r["margin_fused"] <= 0.5]
        comp = [(u, r) for u in us for r in u["rows"]
                if r.get("margin_struct") is not None and r.get("margin_fused") is not None
                and r["margin_struct"] > 0.5 and r["margin_fused"] > 0.5]
        base = len(tgt) / max(1, len(tgt) + len(comp))
        conc: dict = {"n_target": len(tgt), "n_complement": len(comp), "base_rate": base,
                      "strata": {}}

        def add_stratum(dim, value, target_hit):
            s = conc["strata"].setdefault(dim, {}).setdefault(str(value), {"target": 0, "complement": 0})
            s["target" if target_hit else "complement"] += 1

        def emit(u, r, hit):
            add_stratum("label", r["label"], hit)
            add_stratum("n_nodes", bucket(r["n_nodes"], [5, 10, 20]), hit)
            add_stratum("trunc_frac", bucket(r["trunc_frac"], [0.2, 0.5, 0.8]), hit)
            add_stratum("text_len", bucket(r["text_len"], [500, 1500, 3000]), hit)
            add_stratum("attack_category", (r["meta"].get("attack_category") or "n/a"), hit)
            add_stratum("task", (r["meta"].get("task") or r["meta"].get("task_name") or "n/a"), hit)
            add_stratum("cell", u["cell"], hit)
            add_stratum("group", r["group"], hit)

        for u, r in tgt:
            emit(u, r, True)
        for u, r in comp:
            emit(u, r, False)
        interesting = {}
        for dim, vals in conc["strata"].items():
            items = [{"value": v, **c, "rate": c["target"] / max(1, c["target"] + c["complement"])}
                     for v, c in vals.items()]
            items = [it for it in items if it["target"] + it["complement"] >= 5]
            items.sort(key=lambda it: -it["rate"])
            interesting[dim] = items[:8]
        conc["strata"] = interesting
        summary["target_concentration"][ds] = conc

        # Q4 part: node-type composition difference on target vs complement
        type_diff = {}
        all_types = Counter()
        for u, r in tgt + comp:
            all_types.update(r["types"])
        for t in [t for t, _ in all_types.most_common(12)]:
            mt = np.mean([r["types"].get(t, 0) for _, r in tgt]) if tgt else np.nan
            mc = np.mean([r["types"].get(t, 0) for _, r in comp]) if comp else np.nan
            type_diff[t] = {"target_mean": float(mt), "complement_mean": float(mc),
                            "diff": float(mt - mc)}
        summary["target_concentration"][ds]["node_type_diff"] = type_diff

        # Q2: truncation / noise
        node_lens = [L for u in us for r in u["rows"] for L in [r["mean_node_chars"]]]
        trunc_fracs = [r["trunc_frac"] for u in us for r in u["rows"]]
        dup = [r["dup_ratio"] for u in us for r in u["rows"]]
        tgt_trunc = [r["trunc_frac"] for _, r in tgt]
        comp_trunc = [r["trunc_frac"] for _, r in comp]
        summary["truncation"][ds] = {
            "n_graphs": len(trunc_fracs),
            "mean_node_chars": float(np.mean(node_lens)),
            "median_node_chars": float(np.median(node_lens)),
            "mean_trunc_frac": float(np.mean(trunc_fracs)),
            "median_trunc_frac": float(np.median(trunc_fracs)),
            "mean_dup_ratio": float(np.mean(dup)),
            "target_mean_trunc": float(np.mean(tgt_trunc)) if tgt_trunc else None,
            "complement_mean_trunc": float(np.mean(comp_trunc)) if comp_trunc else None,
        }

        # Q3b: aux-head decomposition (G3 stored s_s/s_t)
        ovr = {"n_target": len(tgt), "n_with_aux": 0, "s_s_right": 0, "s_t_right": 0}
        sp_fused_s, sp_fused_t = [], []
        for u in us:
            for r in u["rows"]:
                if r.get("margin_s_s") is not None:
                    ovr["n_with_aux"] += 1
            rows = u["rows"]
            if "s_s" in rows[0]["scores"]:
                sp_fused_s.append(spearman([r["scores"]["fused"] for r in rows],
                                           [r["scores"]["s_s"] for r in rows]))
                sp_fused_t.append(spearman([r["scores"]["fused"] for r in rows],
                                           [r["scores"]["s_t"] for r in rows]))
        for _, r in tgt:
            ms, mt = r.get("margin_s_s"), r.get("margin_s_t")
            if ms is not None and ms > 0.5:
                ovr["s_s_right"] += 1
            if mt is not None and mt > 0.5:
                ovr["s_t_right"] += 1
        ovr["spearman_fused_s_s"] = float(np.nanmean(sp_fused_s)) if sp_fused_s else None
        ovr["spearman_fused_s_t"] = float(np.nanmean(sp_fused_t)) if sp_fused_t else None
        summary["channel_override"][ds] = ovr

        # Q4: shortcut probes on the frozen frozen splits
        summary["shortcut"][ds] = shortcut_probe(ds, us)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "fusion_failure_diagnosis.json").write_text(json.dumps({"summary": summary, "units": report}, indent=2))
    write_md(summary)
    print("wrote", OUT / "fusion_failure_diagnosis.json")


def shortcut_probe(ds: str, us: list[dict]) -> dict:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    res = {"per_cell": [], "group_purity": []}
    for u in us:
        split = resolve_local((json.loads((G3 / f"G3_p25_{ds}_{u['cell']}_s{u['seed']}.json").read_text())
                               .get("config") or {}).get("val", ""))
        if split is None or not split.exists():
            continue
        train_path = split.parent / "train.jsonl"
        if not train_path.exists():
            continue
        if any(c["cell"] == u["cell"] for c in res["per_cell"]):
            continue
        train_g = load_jsonl(train_path)
        Xtr = np.array([[graph_feats(g)[k] for k in TRIVIAL_KEYS] +
                        [edge_stats(g).get(k, 0.0) for k in EDGE_KEYS] for g in train_g])
        ytr = np.array([int(g["label"]) for g in train_g])
        dev_rows = u["rows"]
        Xdv = np.array([[r[k] for k in TRIVIAL_KEYS] + [r.get(k, 0.0) for k in EDGE_KEYS] for r in dev_rows])
        ydv = [r["label"] for r in dev_rows]
        scaler = StandardScaler().fit(Xtr)
        Xtr_s, Xdv_s = scaler.transform(Xtr), scaler.transform(Xdv)
        lr = LogisticRegression(max_iter=2000, class_weight="balanced")
        lr.fit(Xtr_s, ytr)
        lr_auc = auroc(ydv, lr.predict_proba(Xdv_s)[:, 1])
        d = np.linalg.norm(Xdv_s[:, None, :] - Xtr_s[None, :, :], axis=2)
        nn = np.argsort(d, axis=1)[:, :KNN_K]
        knn_score = ytr[nn].mean(axis=1)
        knn_auc = auroc(ydv, knn_score)
        row = {"cell": u["cell"], "n_train": len(train_g), "n_dev": len(dev_rows),
               "struct_auroc": u["auroc"].get("struct"), "fused_auroc": u["auroc"].get("fused"),
               "trivial_lr_auroc": float(lr_auc), "knn_auroc": float(knn_auc)}
        res["per_cell"].append(row)
        by = defaultdict(set)
        for r in dev_rows:
            by[r["group"]].add(r["label"])
        res["group_purity"].append({"cell": u["cell"], "n_groups": len(by),
                                    "purity": sum(1 for v in by.values() if len(v) == 1) / max(1, len(by))})
    if res["per_cell"]:
        w = np.array([c["n_dev"] for c in res["per_cell"]], dtype=float)
        for k in ("struct_auroc", "fused_auroc", "trivial_lr_auroc", "knn_auroc"):
            vals = np.array([c[k] if c[k] is not None else np.nan for c in res["per_cell"]])
            res[f"weighted_{k}"] = float(np.nansum(vals * w) / np.nansum(w * ~np.isnan(vals)))
        res["mean_purity"] = float(np.mean([p["purity"] for p in res["group_purity"]]))
    return res


def fmt_table(rows: list[list], header: list[str]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def write_md(summary: dict) -> None:
    L = ["# Fusion-failure diagnosis (dev only, zero API/GPU)",
         "",
         "PI directive 2026-09-21: why does adding the semantic channel make the joint model worse",
         "than structure-only on MCP/MRT? Scores are `scores.val` from stored runs; `scores.test` was never read.",
         "Fused = G3_p25 (MCP/MRT) or G3_p25 / K0_full (Decomp); structure = `structure_only` runs.",
         "Margins are within-cell pairwise ranks: margin > 0.5 = correctly ranked against the majority",
         "of the opposite class (ties counted half; negatives use the mirrored direction).",
         ""]
    L.append("## Pair-level ranking corruption (Q3)")
    L.append("")
    rows = []
    for ds, a in summary["pair_corruption"].items():
        rows.append([ds, a["n_pairs"], a["struct_correct"], a["corrupt"],
                     f"{a['corruption_rate']:.3f}", a["rescue"],
                     f"{a['rescue_rate']:.3f}"])
    L.append(fmt_table(rows, ["dataset", "pairs", "struct correct", "struct-only-correct (fused flipped)",
                              "corruption rate", "fused-only-correct (rescue)", "rescue rate"]))
    L.append("")
    L.append("Per-seed corruption rate (structure-correct pairs flipped by fused):")
    L.append("")
    for ds, ps in summary["per_seed_delta"].items():
        L.append(f"- {ds}: " + ", ".join(f"s{s}={v:.3f}" for s, v in sorted(ps.items())))
    L.append("")
    L.append("## Structure-right sample concentration (Q1)")
    L.append("")
    for ds, c in summary["target_concentration"].items():
        L.append(f"### {ds}: target={c['n_target']} vs complement={c['n_complement']} (base target rate {c['base_rate']:.3f})")
        L.append("")
        for dim in ("label", "n_nodes", "trunc_frac", "text_len", "attack_category", "task", "cell"):
            items = c["strata"].get(dim) or []
            if not items:
                continue
            rows = [[it["value"], it["target"], it["complement"], f"{it['rate']:.3f}"] for it in items]
            L.append(f"**{dim}** (target rate vs base):")
            L.append("")
            L.append(fmt_table(rows, [dim, "target", "complement", "target rate"]))
            L.append("")
        td = c.get("node_type_diff") or {}
        if td:
            top = sorted(td.items(), key=lambda kv: -abs(kv[1]["diff"]))[:6]
            L.append("Top node-type composition differences (mean count target - complement):")
            L.append("")
            L.append(fmt_table([[t, f"{v['target_mean']:.2f}", f"{v['complement_mean']:.2f}", f"{v['diff']:+.2f}"]
                                for t, v in top],
                               ["node type", "target mean", "complement mean", "diff"]))
            L.append("")
    L.append("## Text-channel input quality (Q2)")
    L.append("")
    rows = []
    for ds, t in summary["truncation"].items():
        rows.append([ds, t["n_graphs"], f"{t['mean_node_chars']:.0f}", f"{t['median_node_chars']:.0f}",
                     f"{t['mean_trunc_frac']:.3f}", f"{t['median_trunc_frac']:.3f}",
                     f"{t['mean_dup_ratio']:.3f}",
                     "n/a" if t["target_mean_trunc"] is None else f"{t['target_mean_trunc']:.3f}",
                     "n/a" if t["complement_mean_trunc"] is None else f"{t['complement_mean_trunc']:.3f}"])
    L.append(fmt_table(rows, ["dataset", "graphs", "mean node chars", "median node chars",
                              "mean trunc frac", "median trunc frac", "mean dup ratio",
                              "target trunc", "complement trunc"]))
    L.append("")
    L.append("## Channel override decomposition, G3 aux heads (Q3)")
    L.append("")
    rows = []
    for ds, o in summary["channel_override"].items():
        rows.append([ds, o["n_target"], o["n_with_aux"], o["s_s_right"], o["s_t_right"],
                     o["spearman_fused_s_s"], o["spearman_fused_s_t"]])
    L.append(fmt_table(rows, ["dataset", "n target", "n with aux", "s_s right on target",
                              "s_t right on target", "spearman(fused, s_s)", "spearman(fused, s_t)"]))
    L.append("")
    L.append("## Structure-channel shortcut probes (Q4)")
    L.append("")
    for ds, s in summary["shortcut"].items():
        if not s.get("per_cell"):
            continue
        L.append(f"### {ds}: mean group purity {s.get('mean_purity', float('nan')):.2f}")
        L.append("")
        rows = [[c["cell"], c["n_train"], c["n_dev"],
                 "n/a" if c["struct_auroc"] is None else f"{c['struct_auroc']:.3f}",
                 "n/a" if c["fused_auroc"] is None else f"{c['fused_auroc']:.3f}",
                 f"{c['trivial_lr_auroc']:.3f}", f"{c['knn_auroc']:.3f}"] for c in s["per_cell"]]
        L.append(fmt_table(rows, ["cell", "n train", "n dev", "structure AUROC", "fused AUROC",
                                  "trivial LR AUROC", f"kNN(k={KNN_K}) AUROC"]))
        L.append("")
        L.append(f"Weighted: structure {s.get('weighted_struct_auroc', float('nan')):.3f}, "
                 f"fused {s.get('weighted_fused_auroc', float('nan')):.3f}, "
                 f"trivial LR {s.get('weighted_trivial_lr_auroc', float('nan')):.3f}, "
                 f"kNN {s.get('weighted_knn_auroc', float('nan')):.3f}")
        L.append("")
    L.append("## Reading / limits")
    L.append("")
    L.append("- Corruption rate = share of dev pairs that structure-only ranks correctly but the fused model flips;")
    L.append("  rescue rate = share of pairs structure-only gets wrong that the fused model fixes.")
    L.append("- Targets are dev-only, small per cell (MCP fold dev n=31). Per-seed rates are reported to avoid")
    L.append("  treating retrained seeds as independent.")
    L.append("- `s_s`/`s_t` are G3 auxiliary-head scores stored by an earlier remote revision; treat the")
    L.append("  decomposition as descriptive, not as a deployable two-head model.")
    L.append("- Trivial LR/kNN probes use train->dev only, features = graph size/edge counts/text length/")
    L.append("  truncation/keyword/duplicate ratios + rebuilt-edge low-level stats. They are shortcut ceilings,")
    L.append("  not baselines for the paper.")
    (OUT / "fusion_failure_diagnosis.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
