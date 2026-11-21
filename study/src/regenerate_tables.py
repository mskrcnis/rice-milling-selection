"""Regenerate every aggregate table using independently reconstructed event metrics."""

import json
import shutil
import tempfile
from pathlib import Path
import pandas as pd
from .analysis import aggregate
from .common import ROOT, digest, write_json, verify_freeze


def verify_all_tables(out):
    out = Path(out).resolve()
    verify_freeze(out / "evaluation_freeze.json")
    summary = json.loads((out / "verification/verification_summary.json").read_text())
    source = out / "verification/regenerated_per_pool_results.csv"
    if (
        summary["status"] != "passed"
        or digest(source) != summary["regenerated_results_sha256"]
    ):
        raise ValueError("Independent event reconstruction unavailable or changed")
    config = json.loads((out / "selected_configuration.json").read_text())
    (ROOT / "tmp").mkdir(exist_ok=True)
    checked = {}
    with tempfile.TemporaryDirectory(
        prefix="event_table_regeneration_", dir=ROOT / "tmp"
    ) as temp:
        temp = Path(temp)
        shutil.copyfile(
            out / "evaluation_events.jsonl.gz", temp / "evaluation_events.jsonl.gz"
        )
        aggregate(pd.read_csv(source), config, temp)
        original = {p.name for p in (out / "tables").glob("*.csv")}
        generated = {p.name for p in (temp / "tables").glob("*.csv")}
        if original != generated:
            raise ValueError("Aggregate table set differs")
        for name in sorted(original):
            a = pd.read_csv(out / "tables" / name)
            b = pd.read_csv(temp / "tables" / name)
            pd.testing.assert_frame_equal(
                a, b, check_exact=False, rtol=1e-10, atol=1e-8
            )
            checked[name] = {
                "rows": len(a),
                "columns": len(a.columns),
                "source_sha256": digest(out / "tables" / name),
            }
    result = {
        "status": "passed",
        "tables": checked,
        "table_count": len(checked),
        "absolute_tolerance": 1e-8,
        "relative_tolerance": 1e-10,
        "scope": "Every aggregate table regenerated from independently reconstructed per-pool event metrics. Six core tables additionally have a separately coded statistical reconstruction in table_verification.json.",
        "event_metrics_sha256": digest(source),
    }
    write_json(out / "verification/all_tables_event_regeneration.json", result)
    return result


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    a = p.parse_args()
    print(json.dumps(verify_all_tables(a.out), indent=2))
