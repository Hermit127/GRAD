#!/usr/bin/env python3
"""12 jobs: <method> <seed> for the Decomp v2 one-shot confirmation."""
from __future__ import annotations
import sys
from pathlib import Path
METHODS = ["K0_full", "G3_p25", "structure_only", "S3_reg"]
SEEDS = [211, 212, 213]
def main() -> int:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "code/repro/out/table5_decomp_v2/jobs.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{m} {s}" for m in METHODS for s in SEEDS]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out} ({len(lines)} jobs)")
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
