#!/usr/bin/env python3
"""P0 checkpoint inventory (server-side, read-only).

PI 2026-09-21 P0: list server checkpoints with path, hash, model config, seed and
selection basis, so the closed-loop/CL work knows what can be reused. GRAD-512
confirm runs are audited separately; this scans the whole repo for *.pt/*.pth/*.ckpt.

Outputs:
  LOGS/final_five_tables/table5/checkpoint_inventory.{json,md}
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "LOGS/final_five_tables/table5"
META_NAMES = ("results.json", "report.json", "config.json", "run.json", "manifest.json")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sidecar_meta(d: Path) -> dict:
    for name in META_NAMES:
        p = d / name
        if p.exists():
            try:
                m = json.loads(p.read_text())
            except Exception:
                continue
            if not isinstance(m, dict):
                continue
            cfg = m.get("config") or m.get("cfg") or {}
            return {"sidecar": str(p.relative_to(REPO)),
                    "seed": m.get("seed") or cfg.get("seed"),
                    "candidate": m.get("candidate") or cfg.get("candidate") or cfg.get("variant"),
                    "selection": (m.get("candidate_meta") or {}).get("selection")
                    or m.get("selection_basis") or m.get("checkpoint_metric"),
                    "keys": sorted(m.keys())[:12]}
    return {}


def main() -> None:
    groups = defaultdict(lambda: {"files": [], "bytes": 0, "newest": 0.0})
    t0 = time.time()
    n = 0
    for root, dirs, files in os.walk(REPO):
        if "/.git" in root:
            continue
        for f in files:
            if not f.endswith((".pt", ".pth", ".ckpt")):
                continue
            p = Path(root) / f
            st = p.stat()
            g = groups[root]
            g["files"].append({"path": str(p.relative_to(REPO)), "bytes": st.st_size,
                               "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(st.st_mtime)),
                               "sha256": sha256_file(p)})
            g["bytes"] += st.st_size
            g["newest"] = max(g["newest"], st.st_mtime)
            n += 1
            if n % 250 == 0:
                print(f"hashed {n} ({time.time() - t0:.0f}s)", flush=True)
    out = {"n_files": n, "n_dirs": len(groups),
           "total_bytes": sum(g["bytes"] for g in groups.values()),
           "groups": {}}
    for root, g in sorted(groups.items(), key=lambda kv: -kv[1]["bytes"]):
        meta = sidecar_meta(Path(root))
        out["groups"][str(Path(root).relative_to(REPO))] = {
            "n_files": len(g["files"]), "bytes": g["bytes"],
            "newest": time.strftime("%Y-%m-%d %H:%M", time.localtime(g["newest"])),
            "meta": meta, "files": g["files"]}
    (OUT / "checkpoint_inventory.json").write_text(json.dumps(out, indent=2))
    write_md(out)
    print(f"done: {n} files, {out['total_bytes'] / 1e9:.2f} GB, {time.time() - t0:.0f}s")


def write_md(out: dict) -> None:
    L = ["# Server checkpoint inventory (P0)", "",
         f"{out['n_files']} checkpoint files in {out['n_dirs']} directories, "
         f"{out['total_bytes'] / 1e9:.2f} GB. Full per-file manifest (path/sha256/mtime) in "
         "`checkpoint_inventory.json`.", "",
         "| directory | files | MB | newest | seed | candidate | sidecar |",
         "|---|---|---|---|---|---|---|"]
    for d, g in list(out["groups"].items())[:40]:
        m = g["meta"]
        L.append(f"| {d} | {g['n_files']} | {g['bytes'] / 1e6:.1f} | {g['newest']} "
                 f"| {m.get('seed', '')} | {m.get('candidate', '')} | {m.get('sidecar', '')} |")
    L.append("")
    L.append("> GRAD-512 confirmation runs (`code/repro/out/grad_repair/confirm/`) saved **no** "
             "checkpoints (verified in the P0 operating-point audit), so closed-loop reuse of "
             "exactly those models requires retraining with `--save-checkpoint`.")
    (OUT / "checkpoint_inventory.md").write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
