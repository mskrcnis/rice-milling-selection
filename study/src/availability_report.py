"""Post-freeze public-metadata reporting; does not construct a replay cohort."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, digest, verify_freeze, write_json


GROUPINGS = {
    "source": ["program"],
    "source_year": ["program", "year"],
    "source_site": ["program", "site"],
    "source_stage": ["program", "stage"],
    "source_class": ["program", "source_class"],
    "source_year_site": ["program", "year", "site"],
    "source_year_stage": ["program", "year", "stage"],
    "source_year_class": ["program", "year", "source_class"],
    "source_year_site_stage_class": [
        "program",
        "year",
        "site",
        "stage",
        "source_class",
    ],
    "source_pool_context": ["program", "pool_context_stratum"],
    "source_year_pool_context": ["program", "year", "pool_context_stratum"],
    "source_grain_report": ["program", "grain_report_id"],
    "source_milling_report": ["program", "milling_report_id"],
    "source_year_grain_report": ["program", "year", "grain_report_id"],
    "source_year_site_grain_report": ["program", "year", "site", "grain_report_id"],
    "source_year_site_milling_report": ["program", "year", "site", "milling_report_id"],
    "source_pool": ["program", "year", "site", "pool_id"],
}
STATUSES = ["included_eligible", "field_observed_missing_answer", "other_excluded"]
FEATURE_UNITS = {
    "field_yield": "kg/ha; source-specific moisture basis, not cross-program harmonized",
    "yield_rank": "fractional rank in all compatible observed-field entries",
    "yield_rel": "yield divided by all compatible observed-field mean",
    "height": "cm",
    "lodging_percent": "percentage points",
    "days_to_flowering": "days",
    "n_field_replicates": "reported field replicate count",
}


def _bool(series):
    values = series.astype(str).str.lower()
    if not values.isin(["true", "false", "1", "0"]).all():
        raise ValueError("Invalid frozen boolean: " + str(series.name))
    return values.isin(["true", "1"])


def _label(value):
    if pd.isna(value):
        return "[missing_metadata]"
    if (
        isinstance(value, (int, float, np.integer, np.floating))
        and float(value).is_integer()
    ):
        return str(int(value))
    return str(value)


def prepare_public_metadata(metadata):
    required = {
        "row_id",
        "candidate_id",
        "pool_id",
        "program",
        "year",
        "site",
        "stage",
        "source_class",
        "stratum",
        "observed_field",
        "paired",
        "eligible",
        "pool_ge10",
        "field_yield",
        "yield_rank",
        "yield_rel",
    }
    if not required.issubset(metadata):
        raise ValueError(
            "Missing public metadata fields: " + str(sorted(required - set(metadata)))
        )
    frame = metadata.copy()
    if frame.row_id.isna().any() or frame.row_id.duplicated().any():
        raise ValueError("Missing or repeated candidate-pool row ID")
    if frame[["pool_id", "candidate_id"]].duplicated().any():
        raise ValueError("Repeated candidate inside source pool")
    if frame[["pool_id", "candidate_id", "program", "year"]].isna().any().any():
        raise ValueError("Missing source identity")
    for name in ["observed_field", "paired", "eligible", "pool_ge10"]:
        frame[name] = _bool(frame[name])
    if (frame.paired & ~frame.observed_field).any():
        raise ValueError("Frozen paired answer lacks a valid field yield")
    if not frame.eligible.eq(frame.paired & frame.pool_ge10).all():
        raise ValueError("Frozen eligibility flags are internally inconsistent")
    valid_yield = np.isfinite(frame.field_yield) & frame.field_yield.gt(0)
    if not frame.observed_field.eq(valid_yield).all():
        raise ValueError("Frozen observed-field flag differs from valid yield")
    if frame.loc[frame.observed_field, ["yield_rank", "yield_rel"]].isna().any().any():
        raise ValueError("Observed-field record lacks frozen public normalization")
    if not frame.loc[frame.observed_field, "yield_rank"].between(0, 1).all():
        raise ValueError("Frozen public rank is outside fractional scale")
    if (
        frame.groupby("pool_id")[["program", "year", "site"]]
        .nunique(dropna=False)
        .gt(1)
        .any()
        .any()
    ):
        raise ValueError("Source pool has conflicting public identities")
    contexts = (
        frame.loc[frame.eligible]
        .groupby("pool_id")
        .stratum.agg(lambda x: sorted(set(x)))
    )
    if contexts.map(len).gt(1).any():
        raise ValueError("Frozen eligible pool crosses analysis strata")
    frame["pool_context_stratum"] = frame.pool_id.map(
        contexts.map(lambda x: x[0])
    ).fillna("no_eligible_pool")
    frame["availability_status"] = np.select(
        [frame.eligible, frame.observed_field & ~frame.paired],
        STATUSES[:2],
        default=STATUSES[2],
    )
    for name in ["site", "stage", "source_class"]:
        frame[name] = frame[name].map(_label)
    for kind in ["grain", "milling"]:
        column = kind + "_source_report"
        values = (
            frame[column] if column in frame else pd.Series(np.nan, index=frame.index)
        )
        frame[kind + "_report_id"] = values.map(_label)
        uruguay = frame.program.eq("uruguay")
        if "source_doi" in frame:
            frame.loc[uruguay, kind + "_report_id"] = frame.loc[
                uruguay, "source_doi"
            ].map(_label)
    for feature in FEATURE_UNITS:
        if feature not in frame:
            frame[feature] = np.nan
        frame[feature] = pd.to_numeric(frame[feature], errors="raise")
        frame.loc[~np.isfinite(frame[feature]), feature] = np.nan
    frame.loc[~frame.observed_field, ["field_yield", "yield_rank", "yield_rel"]] = (
        np.nan
    )
    frame["missing_answer_with_field"] = frame.observed_field & ~frame.paired
    frame["other_excluded"] = frame.availability_status.eq(STATUSES[2])
    return frame


def availability_tables(metadata):
    frame = prepare_public_metadata(metadata)
    coverage = []
    attributes = []
    for dimension, keys in GROUPINGS.items():
        groups = frame.groupby(keys, sort=True, dropna=False)
        table = groups.agg(
            candidate_pool_records=("row_id", "size"),
            distinct_source_candidate_ids=("candidate_id", "nunique"),
            pools=("pool_id", "nunique"),
            site_labels=("site", "nunique"),
            harvest_years=("year", "nunique"),
            field_observed_records=("observed_field", "sum"),
            paired_observed_records=("paired", "sum"),
            included_eligible_records=("eligible", "sum"),
            field_observed_missing_answer_records=("missing_answer_with_field", "sum"),
            other_excluded_records=("other_excluded", "sum"),
        ).reset_index()
        table.insert(0, "grouping", dimension)
        table["paired_fraction_of_observed_field"] = (
            table.paired_observed_records
            / table.field_observed_records.replace(0, np.nan)
        )
        table["paired_fraction_of_source_records"] = (
            table.paired_observed_records / table.candidate_pool_records
        )
        table["eligible_fraction_of_source_records"] = (
            table.included_eligible_records / table.candidate_pool_records
        )
        if (
            not (
                table.included_eligible_records
                + table.field_observed_missing_answer_records
                + table.other_excluded_records
            )
            .eq(table.candidate_pool_records)
            .all()
        ):
            raise ValueError("Availability statuses do not partition source records")
        if int(table.candidate_pool_records.sum()) != len(frame):
            raise ValueError("A grouping lost source rows")
        coverage.append(table)
        comparison_keys = keys + ["availability_status"]
        comparison = frame.groupby(comparison_keys, sort=True, dropna=False)
        counts = comparison.agg(
            candidate_pool_records=("row_id", "size"), pools=("pool_id", "nunique")
        ).reset_index()
        for feature, unit in FEATURE_UNITS.items():
            stats = (
                comparison[feature]
                .agg(
                    nonmissing_count="count",
                    mean="mean",
                    median="median",
                    standard_deviation="std",
                    minimum="min",
                    maximum="max",
                )
                .reset_index()
            )
            result = counts.merge(
                stats, on=comparison_keys, how="left", validate="one_to_one"
            )
            result.insert(0, "grouping", dimension)
            result["feature"] = feature
            result["unit"] = unit
            result["missing_count"] = (
                result.candidate_pool_records - result.nonmissing_count
            )
            result["weighting"] = (
                "candidate-pool records; descriptive, not equal-pool performance"
            )
            attributes.append(result)
    identity = [
        "row_id",
        "pool_id",
        "candidate_id",
        "program",
        "year",
        "site",
        "stage",
        "source_class",
        "stratum",
        "pool_context_stratum",
        "grain_report_id",
        "milling_report_id",
        "observed_field",
        "paired",
        "eligible",
        "pool_ge10",
        "availability_status",
    ]
    candidate_status = frame[identity].copy()
    return (
        pd.concat(coverage, ignore_index=True),
        pd.concat(attributes, ignore_index=True),
        candidate_status,
    )


def run_availability_report(out):
    out = Path(out).resolve()
    freeze_path = out / "evaluation_freeze.json"
    metadata_path = ROOT / "derived/cohort/all_public_metadata.csv"
    freeze = json.loads(freeze_path.read_text())
    verify_freeze(freeze_path)
    if str(metadata_path.resolve()) not in {
        str(Path(p).resolve()) for p in freeze["files"]
    }:
        raise ValueError("Public availability metadata absent from evaluation freeze")
    paths = [freeze_path, metadata_path, Path(__file__).resolve()]
    before = {str(p): digest(p) for p in paths}
    metadata = pd.read_csv(
        metadata_path,
        dtype={"row_id": str, "pool_id": str, "candidate_id": str},
        low_memory=False,
    )
    coverage, attributes, status = availability_tables(metadata)
    if {str(p): digest(p) for p in paths} != before:
        raise ValueError("Availability-report inputs changed while reading")
    destination = out / "diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    tables = {
        "availability_by_group.csv": coverage,
        "availability_field_attributes.csv": attributes,
        "availability_candidate_status.csv": status,
    }
    for filename, table in tables.items():
        table.to_csv(destination / filename, index=False)
    manifest = {
        "status": "passed",
        "scope": "Post-freeze reporting completion from public metadata only; no new cohort, policy, outcome, or endpoint",
        "timing": "Reporting helper added after initial evaluation; not claimed to be pre-frozen scientific code",
        "input_sha256": before,
        "output_sha256": {
            str(destination / name): digest(destination / name) for name in tables
        },
        "rows": {name: len(table) for name, table in tables.items()},
        "source_records": len(metadata),
        "status_counts": status.groupby(["program", "availability_status"])
        .size()
        .reset_index(name="records")
        .to_dict("records"),
        "groupings": GROUPINGS,
        "definitions": {
            "included_eligible": "Frozen eligible flag, across every frozen training/evaluation/secondary stratum; not synonymous with primary external eligibility",
            "field_observed_missing_answer": "Frozen valid observed-field flag and no frozen paired valid answer",
            "other_excluded": "All remaining source candidate-pool records, including missing/invalid field yield and paired records in too-small pools",
            "pool_context_stratum": "Existing eligible rows identify the original pool stratum for missing-answer peers; no_eligible_pool otherwise; never changes eligibility",
            "normalization": "Existing yield_rank and yield_rel from all compatible observed-field entries are used unchanged, without ranking the measured-answer subset",
            "field_yield": "Summaries include valid observed field yields only; original frozen file is unchanged; native moisture bases are not harmonized",
            "report_id": "Original grain_source_report/milling_source_report identifier; Uruguay uses original source DOI. Missing report metadata remains explicit",
            "milling_report_grouping": "A provenance summary; missing answers often lack a milling report by construction, so this grouping is not an unbiased source-availability denominator",
            "weighting": "Candidate-pool records. Distinct source codes and repeated years do not create independent lines or experiments",
            "absent_status_groups": "Rows are emitted for represented status groups; absent groups have zero source records and no estimable field summary",
            "inference": "Descriptive availability and field comparisons; no hypothesis tests, policy retuning, or missing-outcome imputation",
        },
        "validation": [
            "Full run freeze verified before metadata read",
            "Metadata file explicitly covered by freeze",
            "Unique candidate-pool identities and frozen boolean consistency",
            "Every grouping preserves all source records",
            "Three status groups exhaust source records",
            "Input hashes unchanged throughout computation",
        ],
    }
    write_json(destination / "availability_report_manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = run_availability_report(args.out)
    print(
        json.dumps(
            {
                "status": result["status"],
                "source_records": result["source_records"],
                "rows": result["rows"],
            },
            indent=2,
        )
    )
