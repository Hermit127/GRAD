#!/usr/bin/env python3
"""Attribution jobs: <label> <dataset> <cell> <seed>. Mode dev or legacy."""
from __future__ import annotations
import sys
from pathlib import Path
SEEDS = [211, 212, 213]
LABELS = ["G3_no_ln", "G3_no_aux", "G3_no_moddrop", "S3_reg"]
MRT = [f"fold_{t}" for t in ("american_airlines_2", "api_key_calendar_agendas_2",
                             "bank_transfer_2", "expenditure_accounting", "spam_filter_update")]
def main() -> int:
    out = Path(sys.argv[1]); mode = sys.argv[2] if len(sys.argv) > 2 else "dev"
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    if mode == "dev":
        for label in LABELS:
            for fold in ("fold0", "fold1"):
                for s in SEEDS:
                    lines.append(f"{label} mcp {fold} {s}")
            for s in SEEDS:
                lines.append(f"{label} decomp fold0 {s}")
    else:
        prio = {"decomp": 0, "mrt": 1, "mcp": 2}
        jobs = []
        for label in LABELS:
            for s in SEEDS:
                jobs.append((prio["decomp"], f"{label} decomp fold0 {s}"))
            for cell in MRT:
                for s in SEEDS:
                    jobs.append((prio["mrt"], f"{label} mrt {cell} {s}"))
            for cell in [f"fold{k}" for k in range(5)]:
                for s in SEEDS:
                    jobs.append((prio["mcp"], f"{label} mcp {cell} {s}"))
        jobs.sort(key=lambda kv: kv[0])
        lines = [j for _, j in jobs]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} jobs, mode={mode})")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
