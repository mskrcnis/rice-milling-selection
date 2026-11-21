from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT, write_json


def truth(values):
    return values.astype(str).str.lower().isin(["true", "1", "yes"])


def harmonize_public():
    frames = []
    for program in ["uruguay", "arkansas"]:
        path = ROOT / "derived" / program / "public_features.csv"
        frame = pd.read_csv(
            path, dtype={"row_id": str, "pool_id": str, "candidate_id": str}
        )
        frame["program"] = program
        frame["source_class"] = (
            program + ":" + frame.grain_class.fillna("unknown").astype(str)
        )
        frame["grain_class"] = "unharmonized"
        if program == "uruguay":
            frame["observed_answer"] = truth(frame.eligible_observed)
            frame["compromised"] = False
            frame["exposed"] = False
            frame["height"] = frame.get("plant_height_cm", np.nan)
            frame["n_field_replicates"] = frame.replicate_count
        else:
            frame["observed_answer"] = truth(frame.paired_outcome_available)
            frame["compromised"] = truth(frame.pool_compromised)
            frame["exposed"] = truth(frame.previously_seen)
            frame["stage"] = frame.trial_stage
            frame["n_field_replicates"] = frame.replicate_count
        if (
            frame.row_id.duplicated().any()
            or frame[["pool_id", "candidate_id"]].duplicated().any()
        ):
            raise ValueError("Candidate identity collision in " + program)
        frame["observed_field"] = np.isfinite(frame.field_yield) & frame.field_yield.gt(
            0
        )
        frame["yield_rel"] = np.nan
        frame["yield_rank"] = np.nan
        valid = frame.observed_field
        field = frame.loc[valid]
        frame.loc[valid, "yield_rel"] = field.field_yield / field.groupby(
            "pool_id"
        ).field_yield.transform("mean")
        frame.loc[valid, "yield_rank"] = field.groupby("pool_id").field_yield.rank(
            method="average", pct=True
        )
        frame["paired"] = frame.observed_answer & frame.observed_field
        frame["paired_pool_n"] = (
            frame.groupby("pool_id").paired.transform("sum").astype(int)
        )
        frame["pool_ge10"] = frame.paired_pool_n.ge(10)
        frame["eligible"] = frame.paired & frame.pool_ge10
        frame["stratum"] = "ineligible"
        if program == "uruguay":
            frame.loc[frame.eligible & frame.year.le(2011), "stratum"] = (
                "initial_training"
            )
            frame.loc[frame.eligible & frame.year.between(2012, 2016), "stratum"] = (
                "development"
            )
            frame.loc[frame.eligible & frame.year.between(2017, 2020), "stratum"] = (
                "uruguay_temporal"
            )
        else:
            frame.loc[frame.eligible & frame.year.le(2016), "stratum"] = (
                "arkansas_pre_cutoff"
            )
            mask = frame.eligible & frame.year.ge(2017)
            frame.loc[mask & ~frame.compromised & ~frame.exposed, "stratum"] = (
                "arkansas_external"
            )
            frame.loc[mask & frame.compromised, "stratum"] = "arkansas_compromised"
            frame.loc[mask & frame.exposed, "stratum"] = "arkansas_previously_seen"
        frames.append(frame)
    all_public = pd.concat(frames, ignore_index=True).sort_values(
        ["program", "year", "pool_id", "row_id"]
    )
    if all_public.row_id.duplicated().any():
        raise ValueError("Cross-program row identity collision")
    out = ROOT / "derived/cohort"
    out.mkdir(exist_ok=True)
    all_public.to_csv(out / "all_public_metadata.csv", index=False)
    columns = [
        "row_id",
        "pool_id",
        "candidate_id",
        "program",
        "site",
        "year",
        "stage",
        "source_class",
        "grain_class",
        "field_yield",
        "yield_rel",
        "yield_rank",
        "n_field_replicates",
        "stratum",
        "compromised",
        "exposed",
        "eligible",
    ]
    selected = all_public.loc[all_public.eligible, columns].copy()
    selected.to_csv(out / "public_candidates.csv", index=False)
    selected.loc[selected.stratum.isin(["initial_training", "development"])].to_csv(
        out / "development_public.csv", index=False
    )
    selected.loc[
        selected.stratum.isin(
            [
                "uruguay_temporal",
                "arkansas_external",
                "arkansas_compromised",
                "arkansas_previously_seen",
            ]
        )
    ].to_csv(out / "evaluation_public.csv", index=False)
    all_public[
        [
            "row_id",
            "pool_id",
            "candidate_id",
            "program",
            "site",
            "year",
            "stratum",
            "eligible",
        ]
    ].to_csv(ROOT / "protocol/split_manifest.csv", index=False)
    coverage = (
        all_public.groupby(["program", "year", "stratum"], dropna=False)
        .agg(
            rows=("row_id", "size"),
            pools=("pool_id", "nunique"),
            candidates=("candidate_id", "nunique"),
            sites=("site", "nunique"),
            field_observed=("observed_field", "sum"),
            paired=("paired", "sum"),
            eligible=("eligible", "sum"),
        )
        .reset_index()
    )
    coverage.to_csv(out / "cohort_flow.csv", index=False)
    field_comparison = (
        all_public[all_public.observed_field]
        .groupby(["program", "year", "paired"])
        .agg(
            candidates=("row_id", "size"),
            pools=("pool_id", "nunique"),
            mean_field_yield=("field_yield", "mean"),
            median_field_yield=("field_yield", "median"),
            mean_yield_rank=("yield_rank", "mean"),
        )
        .reset_index()
    )
    field_comparison.to_csv(out / "observed_answer_selection.csv", index=False)
    selected["returning_line"] = False
    for program in selected.program.unique():
        for year in sorted(selected.loc[selected.program.eq(program), "year"].unique()):
            prior = set(
                selected.loc[
                    selected.program.eq(program) & selected.year.lt(year),
                    "candidate_id",
                ]
            )
            mask = selected.program.eq(program) & selected.year.eq(year)
            selected.loc[mask, "returning_line"] = selected.loc[
                mask, "candidate_id"
            ].isin(prior)
    selected[["row_id", "candidate_id", "program", "year", "returning_line"]].to_csv(
        out / "line_history_status.csv", index=False
    )
    summary = (
        selected.groupby("stratum")
        .agg(
            rows=("row_id", "size"),
            pools=("pool_id", "nunique"),
            sites=("site", "nunique"),
            years=("year", "nunique"),
            lines=("candidate_id", "nunique"),
        )
        .to_dict("index")
    )
    write_json(out / "metadata_summary.json", summary)
    return summary


def partition_keys():
    public = pd.read_csv(
        ROOT / "derived/cohort/public_candidates.csv", dtype={"row_id": str}
    )
    records = []
    for program in ["uruguay", "arkansas"]:
        source = pd.read_csv(
            ROOT / "derived" / program / "private_outcomes.csv", dtype={"row_id": str}
        )
        if source.row_id.duplicated().any():
            raise ValueError("Repeated private-key identity")
        labels = source[["row_id", "hr_percent"]].rename(
            columns={"hr_percent": "hr_pct"}
        )
        joined = public[public.program.eq(program)][["row_id", "stratum"]].merge(
            labels, on="row_id", how="left", validate="one_to_one"
        )
        if joined.hr_pct.isna().any() or not joined.hr_pct.between(0, 100).all():
            raise ValueError("Invalid measured private key")
        records.append(joined)
    labels = pd.concat(records, ignore_index=True).sort_values("row_id")
    train = labels[labels.stratum.isin(["initial_training", "development"])]
    evaluation = labels[
        labels.stratum.isin(
            [
                "uruguay_temporal",
                "arkansas_external",
                "arkansas_compromised",
                "arkansas_previously_seen",
            ]
        )
    ]
    train[["row_id", "hr_pct"]].to_csv(
        ROOT / "private/development_labels.csv", index=False
    )
    evaluation[["row_id", "hr_pct"]].to_csv(
        ROOT / "private/evaluation_key.csv", index=False
    )
    labels[["row_id", "hr_pct"]].to_csv(
        ROOT / "private/all_measured_keys.csv", index=False
    )
    return {
        "development_rows": len(train),
        "evaluation_rows": len(evaluation),
        "all_rows": len(labels),
        "scope": "Separate curation operation; no numeric labels printed. Model development reads development_labels.csv only.",
    }
