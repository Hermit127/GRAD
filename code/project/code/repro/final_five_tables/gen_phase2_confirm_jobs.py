#!/usr/bin/env python3
"""Generate Phase 2 confirmation jobs: <label> <dataset> <cell> <seed>.

Candidates: B_gate, A_moddrop_0.1 (test evaluated; no dev-only).
Sealed-split references: full, structure_only, text_only, chrono_only.
Legacy MCP/decomp/mrt references already exist in the replication queue.
"""
from __future__ import annotations

import sys
from pathlib import Path

SEEDS = [211, 212, 213]
MCP = [f"fold{k}" for k in range(5)]
MRT = [f"fold_{t}" for t in ("american_airlines_2", "api_key_calendar_agendas_2",
                             "bank_transfer_2", "expenditure_accounting",
                             "spam_filter_update")]
CANDIDATES = ["B_gate", "A_moddrop_0.1"]
REFS = ["full", "structure_only", "text_only", "chrono_only"]


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "code/repro/out/table5_candidates/confirm_jobs.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for label in CANDIDATES:
        for cell in MCP:
            for s in SEEDS:
                lines.append(f"{label} mcp {cell} {s}")
        for s in SEEDS:
            lines.append(f"{label} decomp fold0 {s}")
        for cell in MRT:
            for s in SEEDS:
                lines.append(f"{label} mrt {cell} {s}")
    for label in CANDIDATES + REFS:
        for s in SEEDS:
            lines.append(f"{label} sealed sealed {s}")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} jobs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
