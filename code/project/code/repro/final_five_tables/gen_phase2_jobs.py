#!/usr/bin/env python3
"""Generate the Phase 2 dev job list (MCP fold0/fold1, seeds 211/212).

Format: <candidate_label> <fold> <seed>
Candidate labels map to trainer flags in run_phase2_gpu.sh.
Priority: A(lambda 0.1), A(lambda 0.3), B_gate, C_raw_rel, C_raw_rel_drop, A_moddrop.
"""
from __future__ import annotations

import sys
from pathlib import Path

FOLDS = ["fold0", "fold1"]
SEEDS = [211, 212]
CANDIDATES = ["A_norm_aux_0.1", "A_norm_aux_0.3", "B_gate",
              "C_raw_rel", "C_raw_rel_drop", "A_moddrop_0.1", "full_ref"]


def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "code/repro/out/table5_candidates/phase2_jobs.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{c} {f} {s}" for c in CANDIDATES for f in FOLDS for s in SEEDS]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} jobs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
