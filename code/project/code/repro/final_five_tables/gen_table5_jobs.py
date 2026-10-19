#!/usr/bin/env python3
"""Generate the Table 5 GPU job list in priority order (server or local).

Priority: MRT variants first (only GPU can run them), then MCP variants,
then decomp variants, then same-harness `full` runs, then the appendix variant.
MRT `full` is intentionally absent: the archived GPU full runs are reused
(same protocol/hardware), per PI reuse rule.
"""
from __future__ import annotations

import sys
from pathlib import Path

MAIN = ["text_only", "structure_only", "chrono_only",
        "no_relation_types", "no_keyword_features"]
SEEDS = [211, 212, 213]
MRT_FOLDS = ["fold_american_airlines_2", "fold_api_key_calendar_agendas_2",
             "fold_bank_transfer_2", "fold_expenditure_accounting",
             "fold_spam_filter_update"]
MCP_FOLDS = [f"fold{k}" for k in range(5)]


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "code/repro/out/table5_ablation/jobs_table5.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    def add(ds, cells, variants, seeds):
        for cell in cells:
            for v in variants:
                for s in seeds:
                    lines.append(f"{ds} {cell} {v} {s}")

    add("mrt", MRT_FOLDS, MAIN, SEEDS)
    add("mcp", MCP_FOLDS, MAIN, SEEDS)
    add("decomp", ["fold0"], MAIN, SEEDS)
    add("mcp", MCP_FOLDS, ["full"], SEEDS)
    add("decomp", ["fold0"], ["full"], SEEDS)
    add("mrt", MRT_FOLDS, ["no_type_semantics"], SEEDS)
    add("mcp", MCP_FOLDS, ["no_type_semantics"], SEEDS)
    add("decomp", ["fold0"], ["no_type_semantics"], SEEDS)

    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} jobs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
