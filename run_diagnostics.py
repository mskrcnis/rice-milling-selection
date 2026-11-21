"""Run the frozen post hoc diagnostics and verify their archived outputs."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = "revision/outputs"
DIAGNOSTICS = (
    ("actions", "action_diagnostic.py"),
    ("breeder", "breeder_figure.py"),
    ("rounding", "rounding_diagnostic.py"),
)


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one input or generated artifact."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs() -> None:
    """Reject missing or modified revision inputs before running diagnostics."""
    manifest = json.loads(
        (REPOSITORY_ROOT / "revision_input_manifest.json").read_text()
    )
    for relative_path, expected_hash in manifest["files"].items():
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Required revision input is missing: {path}")
        if file_sha256(path) != expected_hash:
            raise ValueError(f"Revision input hash differs: {relative_path}")


def diagnostic_command(kind: str, script: str, output: Path) -> list[str]:
    """Use the current interpreter and the original diagnostic CLI arguments."""
    command = [
        sys.executable,
        str(REPOSITORY_ROOT / "revision/src" / script),
        "--source-project",
        str(REPOSITORY_ROOT / "study"),
        "--output-dir",
        str(output / kind),
    ]
    if kind == "rounding":
        command.extend(
            [
                "--plan",
                str(
                    REPOSITORY_ROOT / "revision/protocol/rounding_diagnostic_plan.json"
                ),
                "--lock",
                str(
                    REPOSITORY_ROOT / "revision/protocol/rounding_diagnostic_lock.json"
                ),
            ]
        )
    return command


def run_diagnostics(output: str) -> None:
    """Run each diagnostic once and compare every expected artifact hash."""
    verify_inputs()
    out = (REPOSITORY_ROOT / output).resolve()
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Use an empty diagnostic output directory: {out}")

    for kind, script in DIAGNOSTICS:
        try:
            subprocess.run(diagnostic_command(kind, script, out), check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"The {kind} diagnostic failed with exit status {error.returncode}. "
                f"Partial output, if any, remains in {out / kind}."
            ) from error

    expected = json.loads(
        (REPOSITORY_ROOT / "revision/expected_outputs.json").read_text()
    )
    checks = {
        name: (out / name).is_file() and file_sha256(out / name) == expected_hash
        for name, expected_hash in expected.items()
    }
    record = {
        "status": "passed" if all(checks.values()) else "failed",
        "identical_outputs": sum(checks.values()),
        "checks": checks,
        "scope": (
            "Frozen post hoc diagnostic plans, original scientific policies unchanged"
        ),
    }
    (out / "portable_diagnostic_verification.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2))
    failed = [name for name, matches in checks.items() if not matches]
    if failed:
        raise ValueError(f"Diagnostic outputs differ for: {', '.join(failed)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            "Empty diagnostic output directory, relative to the repository "
            f"(default: {DEFAULT_OUTPUT})."
        ),
    )
    args = parser.parse_args()
    run_diagnostics(args.output)


if __name__ == "__main__":
    main()
