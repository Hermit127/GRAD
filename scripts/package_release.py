#!/usr/bin/env python3
"""Create a source archive using Git's tracked/unignored file inventory."""
import argparse
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("Write the archive outside the source checkout")
    names = subprocess.check_output(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT)
    files = sorted(set(name.decode() for name in names.split(b"\0") if name))
    if not files:
        parser.error("No source files found; initialize a Git checkout first")
    subprocess.run([sys.executable, str(ROOT / "scripts/verify_release.py")], check=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in files:
            path = ROOT / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"Unexpected source entry: {name}")
            archive.write(path, "GRAD/" + name)
    print(f"Archived {len(files)} files to {output}")


if __name__ == "__main__":
    main()
