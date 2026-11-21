"""Verify the final local delivery against its SHA-256 inventory."""

import argparse
import json
from pathlib import Path
from .common import ROOT, digest


def verify_release(manifest):
    document = json.loads(Path(manifest).read_text())
    failures = []
    for name, expected in document["files"].items():
        path = ROOT / name
        if not path.is_file() or digest(path) != expected:
            failures.append(name)
    result = {
        "status": "passed" if not failures else "failed",
        "files_checked": len(document["files"]),
        "changed_or_missing": failures,
        "scope": "Listed release files; caches, temporary files and the manifest itself are excluded.",
    }
    print(json.dumps(result, indent=2))
    if failures:
        raise SystemExit(1)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(ROOT / "audit/release_manifest.json"))
    verify_release(parser.parse_args().manifest)
