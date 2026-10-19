#!/usr/bin/env python3
"""Generate keyword-experiment GPU jobs: <label> <dataset> <cell> <seed>.

Priority: instrumentation (diagnostic) -> Decomp K dev -> MRT K dev -> MCP K dev
(local CPU already covers MCP; GPU re-run is for hardware consistency).
"""
from __future__ import annotations

import sys
from pathlib import Path

SEEDS = [211, 212, 213]
FOLDS = ["fold0", "fold1"]
MRT = [f"fold_{t}" for t in ("american_airlines_2", "api_key_calendar_agendas_2",
                             "bank_transfer_2", "expenditure_accounting",
                             "spam_filter_update")]
CANDIDATES = ["K2_kwnorm", "K3_kwdrop25", "K3_kwdrop50"]


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "code/repro/out/table5_keyword/jobs_keyword.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = ["instr decomp fold0 211", "instr decomp fold0 212"]
    for label in CANDIDATES:
        for s in SEEDS:
            lines.append(f"{label} decomp fold0 {s}")
    for label in CANDIDATES:
        for cell in MRT:
            for s in SEEDS:
                lines.append(f"{label} mrt {cell} {s}")
    for label in CANDIDATES:
        for fold in FOLDS:
            for s in SEEDS:
                lines.append(f"{label} mcp {fold} {s}")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} jobs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
