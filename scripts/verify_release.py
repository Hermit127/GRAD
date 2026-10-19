#!/usr/bin/env python3
"""Check frozen research sources, Python syntax and public artifact boundaries."""
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def main():
    errors = []
    manifest = json.loads((ROOT / "docs/frozen_source_manifest.json").read_text())
    for row in manifest:
        p = ROOT / row["path"]
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != row["sha256"]:
            errors.append("Frozen source changed: " + row["path"])
    count = 0
    for p in ROOT.rglob("*.py"):
        if any(x in p.parts for x in (".venv", ".git", "__pycache__")):
            continue
        try:
            ast.parse(p.read_text())
            count += 1
        except SyntaxError:
            errors.append("Invalid Python: " + str(p.relative_to(ROOT)))
    for p in ROOT.rglob("*"):
        if not p.is_file() or any(x in p.parts for x in (".git", ".venv", "outputs", "__pycache__")):
            continue
        rel = str(p.relative_to(ROOT))
        # Public snapshot excludes model binaries and credential files.
        if p.suffix in (".pt", ".pth", ".ckpt", ".pem") or (p.name.startswith(".env") and p.name != ".env.example"):
            errors.append("Non-public artifact: " + rel)
        if p.suffix in (".py", ".md", ".json", ".yml", ".txt"):
            text = p.read_text(errors="replace")
            patterns = (r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----",
                        r"gh[pousr]_[A-Za-z0-9]{30,}", r"github_pat_[A-Za-z0-9_]{50,}")
            if any(re.search(pattern, text) for pattern in patterns):
                errors.append("Credential pattern (value withheld): " + rel)
    report = {"frozen_sources": len(manifest), "parsed_python_files": count, "errors": errors}
    print(json.dumps(report, indent=2))
    return bool(errors)


if __name__ == "__main__":
    raise SystemExit(main())
