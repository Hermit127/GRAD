#!/usr/bin/env python3
"""G3_p25_raw confirmation vs K0 and structure-only (legacy + sealed)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
CONF = REPO / "code/repro/out/table5_g3/confirm"
REF = REPO / "code/repro/out/table5_ablation/runs"
ARCH = REPO / "code/repro/out/unified_gnn/runs"
OUT = REPO / "LOGS/final_five_tables/table5"
SEEDS = [211, 212, 213]
CELLS = {"decomp": ["fold0"], "mcp": [f"fold{k}" for k in range(5)],
         "mrt": [f"fold_{t}" for t in ("american_airlines_2", "api_key_calendar_agendas_2",
                                       "bank_transfer_2", "expenditure_accounting",
                                       "spam_filter_update")]}
import sys
LABEL = sys.argv[1] if len(sys.argv) > 1 else "G3_p25_raw"
OUT_NAME = "g3_confirmation" if LABEL == "G3_p25_raw" else "g3_confirmation_" + LABEL


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


def series(ds: str, source: str, variant: str | None = None):
    vals = {}
    for seed in SEEDS:
        items = []
        for cell in CELLS[ds]:
            if source == "g3":
                p = CONF / f"{LABEL}_{ds}_{cell}_s{seed}.json"
            else:
                p = REF / f"{ds}_{cell}_{variant}_s{seed}.json"
                if not p.exists() and ds == "mrt" and variant == "full":
                    p = ARCH / f"mrt_{cell.replace('fold_', '')}_s{seed}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            rows = d["scores"]["test"]
            a = auroc([r["label"] for r in rows], [r["score"] for r in rows])
            npos = sum(r["label"] for r in rows)
            items.append((a, npos * (len(rows) - npos)))
        if items:
            vals[seed] = float(np.average([a for a, _ in items],
                                          weights=[w for _, w in items]))
    return vals


def sealed_series(source: str, variant: str | None = None):
    vals = {}
    for seed in SEEDS:
        if source == "g3":
            p = CONF / f"{LABEL}_sealed_sealed_s{seed}.json"
        else:
            p = (REPO / "code/repro/out/table5_confirm/confirm" /
                 f"{variant}_sealed_sealed_s{seed}.json")
            if not p.exists():
                p = (REPO / "code/repro/out/table5_candidates/sealed_cpu" /
                     f"{variant}_s{seed}.json")
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        rows = d["scores"]["test"]
        vals[seed] = auroc([r["label"] for r in rows], [r["score"] for r in rows])
    return vals


def summary(vals):
    return {"mean": float(np.mean(list(vals.values()))) if vals else None,
            "std": float(np.std(list(vals.values()))) if vals else None,
            "per_seed": {str(k): round(v, 4) for k, v in vals.items()}}


def main() -> int:
    out = {"candidate": LABEL, "legacy": {}, "sealed": {}, "acceptance": {}}
    res = {}
    for ds in ("decomp", "mcp", "mrt"):
        g3 = summary(series(ds, "g3"))
        k0 = summary(series(ds, "k0", "full"))
        st = summary(series(ds, "ref", "structure_only"))
        res[ds] = {LABEL: g3, "K0": k0, "structure_only": st,
                   "delta_vs_K0": (None if g3["mean"] is None or k0["mean"] is None
                                   else g3["mean"] - k0["mean"]),
                   "delta_vs_structure": (None if g3["mean"] is None or st["mean"] is None
                                          else g3["mean"] - st["mean"])}
    g3s = summary(sealed_series("g3"))
    k0s = summary(sealed_series("ref", "full"))
    sts = summary(sealed_series("ref", "structure_only"))
    out["legacy"] = res
    out["sealed"] = {LABEL: g3s, "K0": k0s, "structure_only": sts,
                     "delta_vs_K0": (None if g3s["mean"] is None or k0s["mean"] is None
                                     else g3s["mean"] - k0s["mean"])}
    legacy = [res[ds] for ds in ("decomp", "mcp", "mrt")]
    means_g3 = [r[LABEL]["mean"] for r in legacy]
    means_k0 = [r["K0"]["mean"] for r in legacy]
    acc = {
        "a_no_legacy_regression_gt_0.01": all(
            (g is None or k is None) or g >= k - 0.01 for g, k in zip(means_g3, means_k0)),
        "b_legacy_mean_above_K0": float(np.mean(means_g3)) > float(np.mean(means_k0)),
        "c_sealed_not_below_K0_by_0.01": (g3s["mean"] is None or k0s["mean"] is None
                                          or g3s["mean"] >= k0s["mean"] - 0.01),
        "d_mcp_not_below_structure_by_0.005": (res["mcp"]["delta_vs_structure"] is not None
                                               and res["mcp"]["delta_vs_structure"] >= -0.005),
        "legacy_means": {"G3": means_g3, "K0": means_k0},
    }
    acc["accepted"] = all(acc[k] for k in ("a_no_legacy_regression_gt_0.01",
                                           "b_legacy_mean_above_K0",
                                           "c_sealed_not_below_K0_by_0.01",
                                           "d_mcp_not_below_structure_by_0.005"))
    out["acceptance"] = acc
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{OUT_NAME}.json").write_text(json.dumps(out, indent=1))

    md = [f"# G3 held-out evaluation ({LABEL}, test read once; reused split, not independent)", "",
          f"| Dataset | {LABEL} | K0 | structure | Δ vs K0 | Δ vs structure |",
          "|---|---|---|---|---|---|"]
    for ds in ("decomp", "mcp", "mrt"):
        r = res[ds]
        md.append(f"| {ds} legacy | {r[LABEL]['mean']:.4f}±{r[LABEL]['std']:.4f} "
                  f"| {r['K0']['mean']:.4f} | {r['structure_only']['mean']:.4f} "
                  f"| {r['delta_vs_K0']:+.4f} | {r['delta_vs_structure']:+.4f} |")
    md.append(f"| held-out group eval (reused) | {g3s['mean']:.4f}±{g3s['std']:.4f} | "
              f"{k0s['mean']:.4f} | {sts['mean']:.4f} | {g3s['mean']-k0s['mean']:+.4f} | "
              f"{g3s['mean']-sts['mean']:+.4f} |")
    md += ["", "## Acceptance rule", ""]
    for k in ("a_no_legacy_regression_gt_0.01", "b_legacy_mean_above_K0",
              "c_sealed_not_below_K0_by_0.01", "d_mcp_not_below_structure_by_0.005"):
        md.append(f"- {k}: **{acc[k]}**")
    md.append(f"- **accepted: {acc['accepted']}**")
    (OUT / f"{OUT_NAME}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
