"""Check distributed files and formatting provenance without importing the study."""

import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_path(relative: str) -> Path:
    path = ROOT / relative
    if Path(relative).is_absolute() or ".." in Path(relative).parts:
        raise ValueError(f"Unsafe manifest path: {relative}")
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing file or unexpected symlink: {relative}")
    if not path.resolve().is_relative_to(ROOT):
        raise ValueError(f"Path escapes repository: {relative}")
    return path


def check_manifest(name: str) -> int:
    manifest = json.loads(checked_path(name).read_text())
    for relative, expected in manifest["files"].items():
        if digest(checked_path(relative)) != expected:
            raise ValueError(f"Hash mismatch in {name}: {relative}")
    return len(manifest["files"])


def syntax_tree(path: Path) -> str:
    return ast.dump(ast.parse(path.read_text()), include_attributes=False)


def main() -> None:
    counts = {
        name: check_manifest(name)
        for name in (
            "repository_manifest.json",
            "portable_input_manifest.json",
            "revision_input_manifest.json",
        )
    }
    formatting = json.loads(
        checked_path("provenance/repository_formatting.json").read_text()
    )
    for relative, record in formatting["files"].items():
        original = checked_path(record["original_copy"])
        current = checked_path(relative)
        if digest(original) != record["original_sha256"]:
            raise ValueError(f"Original source changed: {relative}")
        if digest(current) != record["repository_sha256"]:
            raise ValueError(f"Formatted source changed: {relative}")
        original_tree = syntax_tree(original)
        if (
            hashlib.sha256(original_tree.encode()).hexdigest()
            != record["original_ast_sha256"]
        ):
            raise ValueError(f"Original syntax tree differs: {relative}")
        if original_tree != syntax_tree(current):
            raise ValueError(f"Formatting changed the syntax tree: {relative}")
    for relative in (
        "expected_scientific_outputs.json",
        "revision/expected_outputs.json",
    ):
        if digest(checked_path(relative)) != digest(
            checked_path(f"provenance/reviewer_bundle/{relative}")
        ):
            raise ValueError(f"Expected scientific hashes changed: {relative}")
    print(
        json.dumps(
            {
                "status": "passed",
                "manifest_entries": counts,
                "equivalent_syntax_trees": len(formatting["files"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, KeyError, SyntaxError) as error:
        print(f"Repository verification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
