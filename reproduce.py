"""Refit the frozen study pipeline and compare outputs with the archived results."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent
STUDY_ROOT = REPOSITORY_ROOT / "study"
DEFAULT_OUTPUT = "study/outputs/reproduction"


def file_sha256(path: Path) -> str:
    """Return the digest used by the portable input and output manifests."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs() -> dict[str, str]:
    """Check every portable input before creating a new run directory."""
    manifest_path = REPOSITORY_ROOT / "portable_input_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected_files = manifest["files"]
    for relative_path, expected_hash in expected_files.items():
        path = REPOSITORY_ROOT / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Required portable input is missing: {path}")
        if file_sha256(path) != expected_hash:
            raise ValueError(f"Portable input hash differs: {relative_path}")
    return expected_files


def prepare_output_directory(output: str) -> Path:
    """Resolve output relative to the repository and refuse to reuse a run."""
    path = (REPOSITORY_ROOT / output).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f"Choose an empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def compare_outputs(output: Path, expected: dict[str, str]) -> dict[str, bool]:
    """Treat missing artifacts as failed comparisons, alongside changed hashes."""
    return {
        name: (output / name).is_file() and file_sha256(output / name) == expected_hash
        for name, expected_hash in expected.items()
    }


def run_reproduction(output: str) -> None:
    """Run fitting, replay, independent verification, and archived sensitivities."""
    # Import the bundled study explicitly; --help needs only the standard library.
    sys.path.insert(0, str(STUDY_ROOT))
    import joblib

    from src.analysis import aggregate, plot_final
    from src.availability_report import run_availability_report
    from src.common import freeze, now, write_json
    from src.diagnostics import run_diagnostics
    from src.experiment import (
        evaluate,
        fit_menu,
        initial_predictions,
        load_key,
        read_public,
    )
    from src.models import MODEL_MENU
    from src.regenerate_tables import verify_all_tables
    from src.sensitivity import run_sensitivities
    from src.verify import verify_run
    from src.verify_tables import verify_tables

    input_files = verify_inputs()
    out = prepare_output_directory(output)
    started = now()
    config = json.loads(
        (STUDY_ROOT / "protocol/selected_configuration.json").read_text()
    )
    write_json(out / "selected_configuration.json", config)

    print(
        "Verified portable inputs; fitting original fixed historical models.",
        flush=True,
    )
    development = read_public(STUDY_ROOT / "derived/cohort/development_public.csv")
    development_key = load_key(
        STUDY_ROOT / "private/development_labels.csv", development.row_id
    )
    models = fit_menu(development, development_key, MODEL_MENU, out)
    joblib.dump(models, out / "frozen_models.joblib")

    public = read_public(STUDY_ROOT / "derived/cohort/evaluation_public.csv")
    initial_predictions(public, models, config, out)
    inputs = [
        REPOSITORY_ROOT / name
        for name in input_files
        if name.startswith("study/") and "/outputs/" not in name
    ]
    frozen_predictions = [
        out / name
        for name in (
            "selected_configuration.json",
            "frozen_models.joblib",
            "frozen_initial_predictions.csv",
            "frozen_initial_batch_plans.json",
        )
    ]
    freeze(
        inputs + frozen_predictions,
        out / "evaluation_freeze.json",
        {
            "scope": (
                "Portable reproducibility execution after original outcome exposure; "
                "not a new blind evaluation. Scientific pipeline and derived inputs "
                "verified against included hashes. Original raw-source/code locks "
                "retained separately as provenance."
            ),
            "training_max_year": 2016,
        },
    )

    print("Running all original fixed policies.", flush=True)
    evaluation_key = load_key(STUDY_ROOT / "private/evaluation_key.csv", public.row_id)
    results = evaluate(public, evaluation_key, models, config, out)

    print("Independently reconstructing events and tables.", flush=True)
    # This replay has a new execution freeze, not the original blind-analysis lock.
    verify_run(out, analysis_lock=out / "not_an_original_analysis_lock.json")
    aggregate(results, config, out)
    verify_tables(out)
    verify_all_tables(out)
    plot_final(out, config)
    run_diagnostics(out)
    run_availability_report(out)
    run_sensitivities(
        out / "evaluation_events.jsonl.gz",
        STUDY_ROOT / "derived/cohort/evaluation_public.csv",
        STUDY_ROOT / "derived/cohort/all_public_metadata.csv",
        STUDY_ROOT / "derived/uruguay/private_outcomes.csv",
        STUDY_ROOT / "derived/arkansas/private_outcomes.csv",
        STUDY_ROOT / "derived/arkansas/public_features.csv",
        out / "evaluation_freeze.json",
        out / "sensitivity",
    )

    expected = json.loads(
        (REPOSITORY_ROOT / "expected_scientific_outputs.json").read_text()
    )
    checks = compare_outputs(out, expected)
    write_json(
        out / "portable_reproduction_verification.json",
        {
            "status": "passed" if all(checks.values()) else "failed",
            "started_utc": started,
            "finished_utc": now(),
            "comparisons": checks,
            "original_expected_hashes": expected,
            "input_manifest_sha256": file_sha256(
                REPOSITORY_ROOT / "portable_input_manifest.json"
            ),
            "scope": (
                "Fresh fit and replay with the original scientific pipeline "
                "and inputs. "
                "No revised models, cohorts, budgets or scientific endpoint."
            ),
        },
    )
    failed = [name for name, matches in checks.items() if not matches]
    if failed:
        raise ValueError(f"Portable execution differs for: {', '.join(failed)}")
    print(
        json.dumps(
            {
                "status": "passed",
                "identical_original_scientific_outputs": len(checks),
                "events": len(results),
                "output": str(out),
            },
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help=(
            "Empty run directory, relative to the repository "
            f"(default: {DEFAULT_OUTPUT})."
        ),
    )
    args = parser.parse_args()
    run_reproduction(args.output)


if __name__ == "__main__":
    main()
