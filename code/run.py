#!/usr/bin/env python3
"""Run a bundled source script from the portable project root."""
import os
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parent / 'project'
if len(sys.argv) < 2:
    raise SystemExit('Usage: python run.py code/repro/<script>.py [arguments]')
script = (ROOT / sys.argv[1]).resolve()
if not script.is_relative_to(ROOT) or not script.is_file() or script.suffix != '.py':
    raise SystemExit('Choose a .py entry point inside project/.')
os.chdir(ROOT)
os.environ['GRAD_BASE'] = str(ROOT)
sys.path[:0] = [str(script.parent), str(ROOT/'code'), str(ROOT/'code/repro')]
sys.argv = [str(script), *sys.argv[2:]]
runpy.run_path(str(script), run_name='__main__')
