"""Download original source files separately and verify their recorded hashes."""

import argparse
import hashlib
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent
URUGUAY_FILES = {"README.md", "Phenotypes.txt", "Trials.txt", "Lines.txt"}
DOWNLOAD_TIMEOUT_SECONDS = 120


@dataclass(frozen=True)
class SourceRecord:
    """A source version identified by its saved path, download URL, and digest."""

    relative_path: str
    url: str
    expected_sha256: str


def source_records(program: str) -> list[SourceRecord]:
    """Select the same four Uruguay files and downloaded Arkansas PDFs as before."""
    records = []
    if program in {"uruguay", "all"}:
        manifest = json.loads(
            (REPOSITORY_ROOT / "study/audit/uruguay_source_manifest.json").read_text()
        )
        records.extend(
            SourceRecord(item["path"], item["mirror_url"], item["expected_sha256"])
            for item in manifest
            if item.get("name") in URUGUAY_FILES
        )
    if program in {"arkansas", "all"}:
        manifest = json.loads(
            (REPOSITORY_ROOT / "study/raw/arkansas/source_manifest.json").read_text()
        )
        records.extend(
            SourceRecord(item["file"], item["url"], item["sha256"])
            for item in manifest
            if item.get("downloaded") and item.get("file", "").endswith(".pdf")
        )
    return records


def download_source(url: str) -> bytes:
    """Retrieve bytes without accepting a different version after a failed hash."""
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Rice archive reproducibility source retrieval"},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=DOWNLOAD_TIMEOUT_SECONDS
        ) as response:
            return response.read()
    except (urllib.error.URLError, TimeoutError) as error:
        raise RuntimeError(f"Could not retrieve source {url}: {error}") from error


def acquire_source(record: SourceRecord, destination: Path) -> dict[str, str]:
    """Reuse verified local files; write a new download only after verification."""
    path = destination / record.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    already_present = path.exists()
    data = path.read_bytes() if already_present else download_source(record.url)
    actual_hash = hashlib.sha256(data).hexdigest()
    if actual_hash != record.expected_sha256:
        raise ValueError(
            "Source version differs; inspect rather than substitute: "
            f"{record.url} (expected {record.expected_sha256}, got {actual_hash})"
        )
    if not already_present:
        path.write_bytes(data)
    print(f"Verified {record.relative_path}", flush=True)
    return {
        "path": record.relative_path,
        "url": record.url,
        "sha256": actual_hash,
    }


def acquire_sources(output: str, program: str) -> None:
    """Acquire the requested program's sources in a caller-selected directory."""
    destination = Path(output).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    checked = [
        acquire_source(record, destination) for record in source_records(program)
    ]
    (destination / "acquisition_verification.json").write_text(
        json.dumps(checked, indent=2) + "\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        required=True,
        help="Separate directory for downloaded source files.",
    )
    parser.add_argument(
        "--program", choices=["uruguay", "arkansas", "all"], default="all"
    )
    args = parser.parse_args()
    acquire_sources(args.output, args.program)


if __name__ == "__main__":
    main()
