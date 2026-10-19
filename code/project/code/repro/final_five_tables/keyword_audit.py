#!/usr/bin/env python3
"""Phase A/B keyword diagnostics for the GRAD joint encoder (zero API).

Phase A: stability of full vs no_keyword_features (Decomp 0.958 vs 0.963):
  - config identity audit, parameter/dimension accounting;
  - seed-level fold-weighted deltas from raw per-graph scores;
  - paired bootstrap over TEST GRAPHS (test-sample uncertainty only), per seed;
  - explicit note: 3 seeds, folds are correlated, CIs are test-sampling only.

Phase B (train/dev only): keyword rules, label association overall and by
source/template, cross-source heterogeneity, disagreement cases between the
frozen full and no-keyword arms, and keyword-block scales.

Writes LOGS/final_five_tables/table5/keyword_audit.{json,md}.
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code"))
RUNS = REPO / "code/repro/out/table5_ablation/runs"
SPLIT = REPO / "code/repro/out/unified_gnn/fix2_splits"
MRT = REPO / "results/mrt_cross_task_v1"
OUT = REPO / "LOGS/final_five_tables/table5"
SEEDS = [211, 212, 213]
DATASETS = {
    "decomp": ["fold0"],
    "mcp": [f"fold{k}" for k in range(5)],
    "mrt": [f"fold_{t}" for t in ("american_airlines_2", "api_key_calendar_agendas_2",
                                  "bank_transfer_2", "expenditure_accounting",
                                  "spam_filter_update")],
}
REF = {"decomp": ["full", "text_only", "structure_only", "chrono_only"],
       "mcp": ["full", "text_only", "structure_only", "chrono_only"],
       "mrt": ["full", "text_only", "structure_only", "chrono_only"]}
ARMS = ["full", "no_keyword_features"]


def load(path: Path) -> dict:
    return json.loads(path.read_text())


ARCH = REPO / "code/repro/out/unified_gnn/runs"


def run_path(ds, cell, variant, seed):
    p = RUNS / f"{ds}_{cell}_{variant}_s{seed}.json"
    if p.exists():
        return p
    if variant == "full":
        # MRT full was not re-run in the freeze queue; the archived GPU sweep is
        # the protocol-identical reference.
        if ds == "mrt":
            q = ARCH / f"{ds}_{cell.replace('fold_', '')}_s{seed}.json"
            if q.exists():
                return q
        if ds == "decomp":
            q = ARCH / f"decomp_f0_s{seed}.json"
            if q.exists():
                return q
    return p


def scores(path: Path):
    d = load(path)
    rows = d.get("scores", {}).get("test", [])
    return [(int(r["label"]), float(r["score"])) for r in rows], d


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


def boot_delta(y, s1, s2, n_boot=10000, seed=0):
    """Paired bootstrap over test graphs; returns (delta, ci_lo, ci_hi)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y); s1 = np.asarray(s1); s2 = np.asarray(s2)
    n = len(y)
    d0 = auroc(y, s1) - auroc(y, s2)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yy = y[idx]
        if yy.sum() == 0 or yy.sum() == n:
            continue
        deltas.append(auroc(yy, s1[idx]) - auroc(yy, s2[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(d0), float(lo), float(hi)


# ---------------- Phase A ----------------

def phase_a() -> dict:
    out = {"config_audit": {}, "paired": {}}
    # config audit on one representative run per variant
    base = load(run_path("decomp", "fold0", "full", 211))
    kw = load(run_path("decomp", "fold0", "no_keyword_features", 211))
    diff = {k: (base["config"].get(k), kw["config"].get(k))
            for k in set(base["config"]) | set(kw["config"])
            if base["config"].get(k) != kw["config"].get(k)}
    out["config_audit"] = {
        "differing_fields": diff,
        "params_full": base["ablation"]["model_params"],
        "params_no_keyword": kw["ablation"]["model_params"],
        "node_base_dim_full": base["ablation"]["node_base_dim"],
        "node_base_dim_no_keyword": kw["ablation"]["node_base_dim"],
        "removal_mode": "dimension deletion (22 -> 13); no zero-padding",
        "untouched_channels": ["permission vector (12)", "turn position (1)",
                               "node text CharCNN", "edge type one-hot + confidence",
                               "edge rationale text"],
        "keywords_in_graph_construction":
            "CONTENT_KEYWORDS are used only for x_base[:,13:22]; the edge builder "
            "uses separate regexes over node text (DATA_FLOW_RE/CONTROL_RE/SINK_RE) "
            "and node types, so removing input keywords does NOT remove keyword-based "
            "edge construction",
    }
    for ds, cells in DATASETS.items():
        entry = {"per_seed": {}, "per_fold": {}, "seed_level_delta": None}
        seed_deltas = {}
        for seed in SEEDS:
            cell_vals = []
            for cell in cells:
                p1, p2 = run_path(ds, cell, "full", seed), run_path(ds, cell, "no_keyword_features", seed)
                if not p1.exists() or not p2.exists():
                    continue
                rows1, d1 = scores(p1)
                rows2, d2 = scores(p2)
                y1 = [r[0] for r in rows1]; s1 = [r[1] for r in rows1]
                y2 = [r[0] for r in rows2]; s2 = [r[1] for r in rows2]
                assert y1 == y2, (p1, "label mismatch")
                a1, a2 = auroc(y1, s1), auroc(y2, s2)
                npos = sum(y1); w = npos * (len(y1) - npos)
                d0, lo, hi = boot_delta(y1, s2, s1, seed=seed)
                entry["per_fold"][f"s{seed}_{cell}"] = {
                    "full": a1, "no_keyword": a2, "delta": a2 - a1,
                    "boot95_test": [lo, hi], "weight": w, "n": len(y1),
                }
                cell_vals.append((a2 - a1, w))
            if cell_vals:
                seed_deltas[seed] = float(np.average([d for d, _ in cell_vals],
                                                     weights=[w for _, w in cell_vals]))
        if seed_deltas:
            entry["seed_level_delta"] = {
                "per_seed": {str(k): round(v, 5) for k, v in seed_deltas.items()},
                "mean": float(np.mean(list(seed_deltas.values()))),
                "std": float(np.std(list(seed_deltas.values()))),
                "all_same_sign": (all(v > 0 for v in seed_deltas.values())
                                  or all(v < 0 for v in seed_deltas.values())),
            }
        out["paired"][ds] = entry
    return out


# ---------------- Phase B ----------------

def keyword_defs() -> dict:
    from train_gnn import CONTENT_KEYWORDS
    return {f"group_{i}": kws for i, kws in enumerate(CONTENT_KEYWORDS)}


def trigger_counts(g: dict) -> Counter:
    from train_gnn import content_features
    txt = " ".join(str((n.get("features") or {}).get("subtask", ""))
                   for n in g.get("nodes", []))
    feat = content_features(txt)
    return Counter({f"group_{i}": int(v) for i, v in enumerate(feat) if v})


def phase_b() -> dict:
    from train_gnn import CONTENT_KEYWORDS
    out = {"definitions": keyword_defs(), "associations": {}, "disagreement": {},
           "scales": {}}
    for ds in ("decomp", "mcp"):
        cells = DATASETS[ds]
        rows = defaultdict(list)   # keyword group -> (label, source)
        per_keyword = defaultdict(lambda: {"pos_with": 0, "n_with": 0,
                                           "pos_without": 0, "n_without": 0})
        sources = defaultdict(Counter)
        for cell in cells:
            for split in ("train", "dev"):
                p = SPLIT / ds / cell / f"{split}.jsonl"
                for line in p.read_text().splitlines():
                    if not line.strip():
                        continue
                    g = json.loads(line)
                    label = int(g["label"])
                    src = str(g.get("metadata", {}).get("config")
                              or g.get("metadata", {}).get("task")
                              or g.get("group") or g.get("task_id"))
                    trig = trigger_counts(g)
                    for grp in (f"group_{i}" for i in range(len(CONTENT_KEYWORDS))):
                        has = 1 if trig.get(grp, 0) > 0 else 0
                        rows[grp].append((label, has, src))
                        per_keyword[grp]["n_with"] += has
                        per_keyword[grp]["pos_with"] += label * has
                        per_keyword[grp]["n_without"] += 1 - has
                        per_keyword[grp]["pos_without"] += label * (1 - has)
                        if has:
                            sources[grp][(src, label)] += 1
        assoc = {}
        for grp, vals in rows.items():
            n = len(vals)
            npos = sum(l for l, _, _ in vals)
            nneg = n - npos
            trig = sum(h for _, h, _ in vals)
            trig_pos = sum(h for l, h, _ in vals if l == 1)
            trig_neg = sum(h for l, h, _ in vals if l == 0)
            assoc[grp] = {
                "trigger_rate_all": trig / n,
                "trigger_rate_label1": trig_pos / max(npos, 1),
                "trigger_rate_label0": trig_neg / max(nneg, 1),
            }
            # cross-source heterogeneity: P(label=1 | trigger) per source
            per_src = defaultdict(lambda: [0, 0])
            for lab, has, src in vals:
                if has:
                    per_src[src][0] += lab
                    per_src[src][1] += 1
            rates = {s: v[0] / v[1] for s, v in per_src.items() if v[1] >= 3}
            assoc[grp]["n_sources_with_trigger"] = len(rates)
            assoc[grp]["src_rate_min"] = min(rates.values()) if rates else None
            assoc[grp]["src_rate_max"] = max(rates.values()) if rates else None
            assoc[grp]["src_rate_std"] = float(np.std(list(rates.values()))) if rates else None
        out["associations"][ds] = assoc
        out["disagreement"][ds] = disagreement(ds)
        # keyword block scales on dev graphs
        vals = []
        for cell in cells:
            p = SPLIT / ds / cell / "dev.jsonl"
            for line in p.read_text().splitlines():
                if line.strip():
                    g = json.loads(line)
                    from train_gnn import content_features
                    txt = " ".join(str((n.get("features") or {}).get("subtask", ""))
                                   for n in g.get("nodes", []))
                    vals.append(content_features(txt))
        arr = np.asarray(vals, dtype=float)
        out["scales"][ds] = {
            "dim_mean": arr.mean(axis=0).round(4).tolist(),
            "dim_std": arr.std(axis=0).round(4).tolist(),
            "dim_max": arr.max(axis=0).tolist(),
            "fraction_nonzero": (arr > 0).mean(axis=0).round(4).tolist(),
        }
    return out


def disagreement(ds: str) -> dict:
    """Full vs no-keyword disagreement on dev (both arms' dev scores)."""
    out = {}
    agg = defaultdict(lambda: {"both_right": 0, "both_wrong": 0, "full_only": 0,
                               "nokw_only": 0, "n": 0, "kw_full_only": 0,
                               "kw_nokw_only": 0})
    for cell in DATASETS[ds]:
        for seed in SEEDS:
            p1, p2 = run_path(ds, cell, "full", seed), run_path(ds, cell, "no_keyword_features", seed)
            if not (p1.exists() and p2.exists()):
                continue
            d1, d2 = load(p1), load(p2)
            v1 = {r["label"]: None for r in []}
            rows1 = d1["scores"]["val"]; rows2 = d2["scores"]["val"]
            if len(rows1) != len(rows2):
                continue
            for r1, r2 in zip(rows1, rows2):
                assert r1["label"] == r2["label"]
                c1 = int((1 if r1["score"] >= 0.5 else 0) == r1["label"])
                c2 = int((1 if r2["score"] >= 0.5 else 0) == r2["label"])
                key = cell
                agg[key]["n"] += 1
                if c1 and c2:
                    agg[key]["both_right"] += 1
                elif not c1 and not c2:
                    agg[key]["both_wrong"] += 1
                elif c1 and not c2:
                    agg[key]["full_only"] += 1
                else:
                    agg[key]["nokw_only"] += 1
    out["per_fold"] = {k: dict(v) for k, v in agg.items()}
    return out


def main() -> int:
    report = {"phase_a": phase_a(), "phase_b": phase_b()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "keyword_audit.json").write_text(json.dumps(report, indent=1, default=str))

    md = ["# Keyword audit (Phase A/B, zero API)", "",
          "## A. full vs no_keyword_features — stable gap check", ""]
    ca = report["phase_a"]["config_audit"]
    md += [f"- params: full {ca['params_full']:,} vs no-keyword {ca['params_no_keyword']:,} "
           f"(node base {ca['node_base_dim_full']} -> {ca['node_base_dim_no_keyword']}, "
           f"{ca['removal_mode']}).",
           f"- config fields differing besides the variant: "
           f"{list(ca['differing_fields'].keys()) or 'none'}.",
           f"- untouched: {', '.join(ca['untouched_channels'])}.",
           f"- {ca['keywords_in_graph_construction']}", ""]
    for ds, entry in report["phase_a"]["paired"].items():
        sd = entry.get("seed_level_delta")
        if not sd:
            continue
        md += [f"### {ds}",
               f"- seed-level delta (no-keyword minus full): {sd['per_seed']} "
               f"(mean {sd['mean']:+.4f}, std {sd['std']:.4f}, all same sign: "
               f"{sd['all_same_sign']})",
               "", "| fold | full | no-keyword | delta | boot95 (test graphs) | n |",
               "|---|---|---|---|---|---|"]
        for cell, d in entry["per_fold"].items():
            md.append(f"| {cell} | {d['full']:.4f} | {d['no_keyword']:.4f} | "
                      f"{d['delta']:+.4f} | [{d['boot95_test'][0]:+.4f}, "
                      f"{d['boot95_test'][1]:+.4f}] | {d['n']} |")
        md.append("")
    md += ["## B. Keyword rules and associations (train/dev)", "",
           "| group | trigger rule (all substrings) |", "|---|---|"]
    for grp, kws in report["phase_b"]["definitions"].items():
        md.append(f"| {grp} | `{', '.join(kws)}` |")
    md += [""]
    for ds, assoc in report["phase_b"]["associations"].items():
        md += [f"### {ds} trigger rates and cross-source spread", "",
               "| group | rate(all) | rate(label=1) | rate(label=0) | sources | "
               "P(l=1\\|trigger) min..max |", "|---|---|---|---|---|---|"]
        for grp, a in assoc.items():
            md.append(f"| {grp} | {a['trigger_rate_all']:.3f} | "
                      f"{a['trigger_rate_label1']:.3f} | {a['trigger_rate_label0']:.3f} | "
                      f"{a['n_sources_with_trigger']} | "
                      f"{a['src_rate_min'] if a['src_rate_min'] is None else round(a['src_rate_min'],2)}"
                      f"..{a['src_rate_max'] if a['src_rate_max'] is None else round(a['src_rate_max'],2)} |")
        md.append("")
    md += ["## C. Frozen-arm disagreement on dev", ""]
    for ds, entry in report["phase_b"]["disagreement"].items():
        md += [f"### {ds}", "",
               "| fold | n | both right | both wrong | full only right | no-kw only right |",
               "|---|---|---|---|---|---|"]
        for cell, v in entry["per_fold"].items():
            md.append(f"| {cell} | {v['n']} | {v['both_right']} | {v['both_wrong']} | "
                      f"{v['full_only']} | {v['nokw_only']} |")
        md.append("")
    md += ["## D. Keyword block scales (dev)", ""]
    for ds, s in report["phase_b"]["scales"].items():
        md.append(f"### {ds}")
        md.append(f"- mean: {s['dim_mean']}")
        md.append(f"- std: {s['dim_std']}")
        md.append(f"- fraction nonzero: {s['fraction_nonzero']}")
        md.append("")
    (OUT / "keyword_audit.md").write_text("\n".join(md) + "\n")
    print("wrote keyword_audit.json/md")
    print(json.dumps(report["phase_a"]["paired"], indent=1)[:2500])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
