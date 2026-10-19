#!/usr/bin/env python3
"""Generate G3 jobs: <label> <dataset> <cell> <seed> (dev and confirmation)."""
from __future__ import annotations
import sys
from pathlib import Path
SEEDS = [211, 212, 213]
CFG = ["G3_p25", "G3_p50", "G3_p25_raw", "G3_p50_raw"]
MRT = [f"fold_{t}" for t in ("american_airlines_2", "api_key_calendar_agendas_2",
                             "bank_transfer_2", "expenditure_accounting", "spam_filter_update")]
def main() -> int:
    mode = sys.argv[2] if len(sys.argv) > 2 else "dev"
    out = Path(sys.argv[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if mode == "dev":
        for label in CFG:
            for fold in ("fold0", "fold1"):
                for s in SEEDS:
                    lines.append(f"{label} mcp {fold} {s}")
            for s in SEEDS:
                lines.append(f"{label} decomp fold0 {s}")
    else:
        label = sys.argv[3] if len(sys.argv) > 3 else CFG[0]
        for label in [label]:
            for cell in [f"fold{k}" for k in range(5)]:
                for s in SEEDS:
                    lines.append(f"{label} mcp {cell} {s}")
            for s in SEEDS:
                lines.append(f"{label} decomp fold0 {s}")
            for cell in MRT:
                for s in SEEDS:
                    lines.append(f"{label} mrt {cell} {s}")
            for s in SEEDS:
                lines.append(f"{label} sealed sealed {s}")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} jobs, mode={mode})")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
