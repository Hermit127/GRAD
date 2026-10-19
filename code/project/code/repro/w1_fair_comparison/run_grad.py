#!/usr/bin/env python3
"""W1 wrapper for the frozen GRAD arm trainer with optional device override."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

_FT = Path(__file__).resolve().parents[1] / "final_five_tables"
sys.path.insert(0, str(_FT))
import train_grad_repair as t  # noqa: E402

if os.environ.get("W1_DEVICE"):
    t.DEVICE = torch.device(os.environ["W1_DEVICE"])

if __name__ == "__main__":
    raise SystemExit(t.main())
