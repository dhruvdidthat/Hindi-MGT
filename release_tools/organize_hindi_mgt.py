"""Build an independent paper workspace from the audited CODE tree (Python 3.10+)."""
import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

TOOLS = Path(__file__).resolve().parent

def digest(path):
    total = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            total.update(chunk)
    return total.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--apply", action="store_true", help="copy after verification")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((TOOLS / "server_inputs.json").read_text())
    errors = []
    for name, expected in manifest.items():
        source = root / name
        if not source.resolve().is_relative_to(root) or not source.is_file():
            errors.append(f"Missing or outside root: {name}")
        elif digest(source) != expected:
            errors.append(f"Changed since audit: {name}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Verified {len(manifest)} files.")
    if not args.apply:
        return
    target = root / "Hindi_MGT_paper_release"
    if target.exists():
        raise SystemExit(f"Release already exists: {target}")
    with tempfile.TemporaryDirectory(prefix=".mgt-build-", dir=root) as temporary:
        stage = Path(temporary) / "release"
        for name, expected in manifest.items():
            dest = stage / "CODE" / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, dest)
            if digest(dest) != expected:
                raise RuntimeError(f"Copy failed verification: {name}")
        shutil.copytree(TOOLS, stage / "reproduction", ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copyfile(TOOLS / "README.md", stage / "README.md")
        os.rename(stage, target)
    print(target)

if __name__ == "__main__":
    main()
