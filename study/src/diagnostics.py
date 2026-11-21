"""Descriptive post-freeze diagnostics; retains original whole decision pools."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, digest, verify_freeze, write_json
from .replay import ordered_ids, score_event

PRIMARY_BUDGET = 0.25
METRICS = ["regret", "regret_pct", "top_k_recall", "best_entry_found", "tests"]
EVENT_ID = ["pool_id", "policy", "seed", "budget_fraction"]
POOL_ID = ["stratum", "program", "year", "site", "pool_id", "policy"]
HISTORY_RATES = [
    "testing_rate",
    "selection_rate",
    "oracle_recovery_rate",
    "shortlist_share",
    "oracle_share",
]
HISTORY_COUNTS = [
    "candidate_count",
    "tested_count",
    "shortlist_count",
    "oracle_topk_count",
    "oracle_tested_count",
    "oracle_shortlisted_count",
]
CALIBRATION_BINS = ["[0,20)", "[20,40)", "[40,60)", "[60,80)", "[80,100]"]
HISTORY_DEFINITION = "Same-program coded candidate present in an earlier eligible archive year; not necessarily in model training"


def _boolean(series):
    tokens = series.astype(str).str.lower()
    if not tokens.isin(["true", "false", "1", "0"]).all():
        raise ValueError("Invalid metadata boolean: " + str(series.name))
    return tokens.isin(["true", "1"])


def _unique_rows(frame, name):
    if frame.row_id.isna().any() or frame.row_id.duplicated().any():
        raise ValueError("Missing or duplicate row identity: " + name)
    return frame.set_index("row_id", drop=False)


def pool_descriptors(metadata):
    _unique_rows(metadata, "public metadata")
    required = {"pool_id", "stage", "source_class", "observed_field", "paired"}
    if not required.issubset(metadata):
        raise ValueError("Missing composition/coverage metadata")
    frame = metadata.copy()
    frame["observed_field"] = _boolean(frame.observed_field)
    frame["paired"] = _boolean(frame.paired)
    if (frame.paired & ~frame.observed_field).any():
        raise ValueError("Paired answer without field yield")
    output = []
    for pool, group in frame.groupby("pool_id", sort=True):
        field = group.loc[group.observed_field]
        measured = group.loc[group.paired]
        classes = sorted(field.source_class.fillna("unknown").astype(str).unique())
        replay_classes = sorted(
            measured.source_class.fillna("unknown").astype(str).unique()
        )
        stages = sorted(group.stage.fillna("unknown").astype(str).unique())
        fraction = len(measured) / len(field) if len(field) else np.nan
        output.append(
            {
                "pool_id": pool,
                "source_candidate_count": len(group),
                "field_candidate_count": len(field),
                "paired_candidate_count": len(measured),
                "paired_fraction_of_field_candidates": fraction,
                "coverage_stratum": (
                    "no_observed_field"
                    if not len(field)
                    else (
                        "complete"
                        if len(measured) == len(field)
                        else (
                            "90_to_less_than_100_percent"
                            if fraction >= 0.9
                            else "below_90_percent"
                        )
                    )
                ),
                "stage_composition": "|".join(stages),
                "source_class_composition": "|".join(classes),
                "source_class_count": len(classes),
                "class_composition_kind": (
                    "single_class" if len(classes) == 1 else "mixed_class"
                ),
                "replay_source_class_composition": "|".join(replay_classes),
            }
        )
    return pd.DataFrame(output)


def composition_diagnostics(results, metadata):
    required = set(EVENT_ID + POOL_ID + METRICS + ["n", "k"])
    if not required.issubset(results):
        raise ValueError("Missing completed result fields")
    if results.duplicated(EVENT_ID).any():
        raise ValueError("Duplicate completed result identity")
    primary = results.loc[results.budget_fraction.eq(PRIMARY_BUDGET)].copy()
    if primary.empty:
        raise ValueError("No primary-budget results")
    descriptors = pool_descriptors(metadata)
    rows = []
    for identity, group in primary.groupby(POOL_ID, sort=True, dropna=False):
        if group.n.nunique() != 1 or group.k.nunique() != 1:
            raise ValueError(
                "Seed repeats have incompatible pool size or shortlist size"
            )
        row = dict(zip(POOL_ID, identity))
        row.update({name: group[name].mean() for name in METRICS})
        row.update(
            n=int(group.n.iloc[0]),
            k=int(group.k.iloc[0]),
            budget_fraction=PRIMARY_BUDGET,
            seed_repetitions=group.seed.nunique(),
        )
        rows.append(row)
    pools = pd.DataFrame(rows).merge(
        descriptors, on="pool_id", how="left", validate="many_to_one"
    )
    if (
        pools.paired_candidate_count.isna().any()
        or not pools.n.eq(pools.paired_candidate_count).all()
    ):
        raise ValueError("Result pool differs from source paired universe")
    summaries = []
    for dimension in [
        "stage_composition",
        "source_class_composition",
        "class_composition_kind",
        "coverage_stratum",
    ]:
        keys = ["stratum", "program", "policy", dimension]
        for identity, group in pools.groupby(keys, sort=True, dropna=False):
            row = dict(zip(keys[:-1], identity[:-1]))
            row.update(
                dimension=dimension,
                level=identity[-1],
                pools=group.pool_id.nunique(),
                sites=group.site.nunique(),
                years=group.year.nunique(),
                replay_candidates=int(group.n.sum()),
                budget_fraction=PRIMARY_BUDGET,
                weighting="equal whole pools after averaging policy seeds",
                interpretation="descriptive composition stratum; no new decision subpools",
            )
            row.update({name: group[name].mean() for name in METRICS})
            summaries.append(row)
    return pools, pd.DataFrame(summaries)


def prediction_metrics(predictions):
    required = {
        "row_id",
        "pool_id",
        "model",
        "observed_hr",
        "mean_hr",
        "stratum",
        "program",
        "year",
        "site",
    }
    if not required.issubset(predictions):
        raise ValueError("Missing prediction diagnostic fields")
    if predictions.duplicated(["row_id", "model"]).any():
        raise ValueError("Duplicate candidate/model prediction")
    frame = predictions.loc[~predictions.model.eq("yield")].copy()
    if frame.empty:
        raise ValueError(
            "No fitted recovery predictions after excluding yield surrogate"
        )
    for column in ["observed_hr", "mean_hr"]:
        if (
            not np.isfinite(frame[column]).all()
            or not frame[column].between(0, 100).all()
        ):
            raise ValueError(
                "Recovery prediction/answer is missing or outside percentage scale"
            )
    frame["error"] = frame.mean_hr - frame.observed_hr
    frame["squared_error"] = frame.error**2
    frame["absolute_error"] = frame.error.abs()
    frame["bin_index"] = np.minimum((frame.mean_hr // 20).astype(int), 4)
    metrics = []
    calibration = []
    for identity, group in frame.groupby(["stratum", "program", "model"], sort=True):
        base = dict(zip(["stratum", "program", "model"], identity))
        pool_errors = group.groupby("pool_id").agg(
            mse=("squared_error", "mean"), mae=("absolute_error", "mean")
        )
        metrics.append(
            {
                **base,
                "candidate_rows": len(group),
                "pools": group.pool_id.nunique(),
                "sites": group.site.nunique(),
                "years": group.year.nunique(),
                "rmse_pp": float(np.sqrt(group.squared_error.mean())),
                "mae_pp": group.absolute_error.mean(),
                "bias_predicted_minus_observed_pp": group.error.mean(),
                "pool_equal_rmse_pp": float(np.sqrt(pool_errors.mse.mean())),
                "pool_equal_mae_pp": pool_errors.mae.mean(),
                "weighting": "candidate rows; additional pool-equal metrics supplied",
                "interpretation": "prediction diagnostic only; repeated years/lines are not independent replicates",
            }
        )
        for bin_index in range(5):
            part = group.loc[group.bin_index.eq(bin_index)]
            calibration.append(
                {
                    **base,
                    "prediction_bin": CALIBRATION_BINS[bin_index],
                    "candidate_rows": len(part),
                    "pools": part.pool_id.nunique(),
                    "mean_predicted_hr_pct": part.mean_hr.mean(),
                    "mean_observed_hr_pct": part.observed_hr.mean(),
                    "observed_minus_predicted_pp": -part.error.mean(),
                    "rmse_pp": (
                        float(np.sqrt(part.squared_error.mean()))
                        if len(part)
                        else np.nan
                    ),
                    "bin_source": "fixed predeclared 20-percentage-point bands",
                    "interpretation": "descriptive calibration means; no fitted recalibration or interval claim",
                }
            )
    return pd.DataFrame(metrics), pd.DataFrame(calibration)


def _observed_pools(predictions, metadata, history):
    if predictions.duplicated(["row_id", "model"]).any():
        raise ValueError("Duplicate prediction identity")
    if predictions.groupby("row_id").observed_hr.nunique(dropna=False).gt(1).any():
        raise ValueError("Models disagree about the observed source answer")
    public = _unique_rows(metadata, "public metadata")
    historical = _unique_rows(history, "line history")
    labels = (
        predictions[["row_id", "observed_hr"]]
        .drop_duplicates("row_id")
        .set_index("row_id")
    )
    if not set(labels.index).issubset(public.index) or not set(labels.index).issubset(
        historical.index
    ):
        raise ValueError("Prediction row missing public metadata or prior-line status")
    frame = public.loc[labels.index].copy()
    for column in ["candidate_id", "program", "year"]:
        if (
            not frame[column]
            .astype(str)
            .eq(historical.loc[frame.index, column].astype(str))
            .all()
        ):
            raise ValueError("Line-history identity mismatch: " + column)
    frame["returning_line"] = _boolean(historical.loc[frame.index, "returning_line"])
    frame["observed_hr"] = labels.observed_hr
    if (
        not np.isfinite(frame.observed_hr).all()
        or not frame.observed_hr.between(0, 100).all()
    ):
        raise ValueError("Missing observed recovery for line-history oracle")
    for column in ["pool_id", "program", "stratum", "year", "site"]:
        joined = predictions[["row_id", column]].merge(
            frame[["row_id", column]].reset_index(drop=True),
            on="row_id",
            how="left",
            suffixes=("_prediction", "_source"),
            validate="many_to_one",
        )
        if (
            not joined[column + "_prediction"]
            .astype(str)
            .eq(joined[column + "_source"].astype(str))
            .all()
        ):
            raise ValueError("Prediction/public metadata mismatch: " + column)
    return {
        str(pool): group.reset_index(drop=True)
        for pool, group in frame.groupby("pool_id", sort=True)
    }


def line_history_diagnostics(events, predictions, metadata, history, results):
    pools = _observed_pools(predictions, metadata, history)
    primary = results.loc[results.budget_fraction.eq(PRIMARY_BUDGET)]
    if primary.duplicated(EVENT_ID).any():
        raise ValueError("Duplicate primary result")
    lookup = primary.set_index(EVENT_ID)
    seen = set()
    records = []
    for event in events:
        if event["budget_fraction"] != PRIMARY_BUDGET:
            continue
        identity = tuple(event[key] for key in EVENT_ID)
        if identity in seen or identity not in lookup.index:
            raise ValueError("Event/result identity mismatch")
        seen.add(identity)
        if str(event["pool_id"]) not in pools:
            raise ValueError("Event lacks observed candidate pool")
        frame = pools[str(event["pool_id"])]
        for column in ["program", "stratum", "year", "site"]:
            if not frame[column].astype(str).eq(str(event[column])).all():
                raise ValueError("Event metadata mismatch: " + column)
        key = dict(zip(frame.row_id, frame.observed_hr))
        measured = dict(zip(frame.row_id, frame.field_yield * frame.observed_hr / 100))
        scored = score_event(frame, key, event)
        if (
            event["shortlist"]
            != ordered_ids(event["tested_ids"], measured)[: event["k"]]
        ):
            raise ValueError("Invalid measured final shortlist")
        saved = lookup.loc[identity]
        for metric in METRICS:
            if not np.isclose(
                scored[metric], saved[metric], equal_nan=True, rtol=1e-10, atol=1e-10
            ):
                raise ValueError("Result differs from event recomputation: " + metric)
        oracle = set(ordered_ids(measured, measured)[: event["k"]])
        tested = set(event["tested_ids"])
        selected = set(event["shortlist"])
        for returning in [False, True]:
            ids = set(frame.loc[frame.returning_line.eq(returning), "row_id"])
            oracle_group = oracle & ids
            records.append(
                {
                    **{key: event[key] for key in POOL_ID + ["seed"]},
                    "budget_fraction": PRIMARY_BUDGET,
                    "history_group": "returning" if returning else "new",
                    "candidate_count": len(ids),
                    "tested_count": len(tested & ids),
                    "shortlist_count": len(selected & ids),
                    "oracle_topk_count": len(oracle_group),
                    "oracle_tested_count": len(oracle_group & tested),
                    "oracle_shortlisted_count": len(oracle_group & selected),
                    "testing_rate": len(tested & ids) / len(ids) if ids else np.nan,
                    "selection_rate": len(selected & ids) / len(ids) if ids else np.nan,
                    "oracle_recovery_rate": (
                        len(oracle_group & selected) / len(oracle_group)
                        if oracle_group
                        else np.nan
                    ),
                    "shortlist_share": len(selected & ids) / event["k"],
                    "oracle_share": len(oracle_group) / event["k"],
                    "history_definition": HISTORY_DEFINITION,
                }
            )
    if seen != set(lookup.index):
        raise ValueError("Primary results do not all have corresponding events")
    raw = pd.DataFrame(records)
    if raw.empty:
        raise ValueError("No primary line-history events")
    group_keys = POOL_ID + ["history_group"]
    averages = (
        raw.groupby(group_keys, sort=True, dropna=False)[HISTORY_COUNTS + HISTORY_RATES]
        .mean()
        .reset_index()
    )
    seed_counts = (
        raw.groupby(group_keys, sort=True, dropna=False)
        .seed.nunique()
        .reset_index(name="seed_repetitions")
    )
    averages = averages.merge(seed_counts, on=group_keys, validate="one_to_one")
    averages["budget_fraction"] = PRIMARY_BUDGET
    summaries = []
    for identity, group in averages.groupby(
        ["stratum", "program", "policy", "history_group"], sort=True
    ):
        row = dict(zip(["stratum", "program", "policy", "history_group"], identity))
        row.update(
            pools=group.pool_id.nunique(),
            sites=group.site.nunique(),
            years=group.year.nunique(),
            pools_with_history_candidates=int(group.candidate_count.gt(0).sum()),
            pools_with_oracle_members=int(group.oracle_topk_count.gt(0).sum()),
            budget_fraction=PRIMARY_BUDGET,
            history_definition=HISTORY_DEFINITION,
            interpretation="descriptive original-pool oracle membership; no class or history decision subpools",
        )
        row.update(
            {
                name + "_sum_after_seed_average": group[name].sum()
                for name in HISTORY_COUNTS
            }
        )
        row.update(
            {name + "_equal_pool_mean": group[name].mean() for name in HISTORY_RATES}
        )
        denom = group.oracle_topk_count.sum()
        row["oracle_member_weighted_recovery"] = (
            group.oracle_shortlisted_count.sum() / denom if denom else np.nan
        )
        summaries.append(row)
    return raw, averages, pd.DataFrame(summaries)


def run_diagnostics(out):
    out = Path(out)
    locked = out / "evaluation_freeze.json"
    metadata_path = ROOT / "derived/cohort/all_public_metadata.csv"
    history_path = ROOT / "derived/cohort/line_history_status.csv"
    manifest = json.loads(locked.read_text())
    verify_freeze(locked)
    frozen = {str(Path(p).resolve()) for p in manifest["files"]}
    if not {str(metadata_path.resolve()), str(history_path.resolve())}.issubset(frozen):
        raise ValueError(
            "Public composition/history files absent from evaluation freeze"
        )
    paths = [
        locked,
        metadata_path,
        history_path,
        out / "per_pool_results.csv",
        out / "prediction_diagnostics.csv",
        out / "evaluation_events.jsonl.gz",
    ]
    before = {str(p.resolve()): digest(p) for p in paths}
    results = pd.read_csv(out / "per_pool_results.csv")
    predictions = pd.read_csv(out / "prediction_diagnostics.csv", dtype={"row_id": str})
    metadata = pd.read_csv(metadata_path, dtype={"row_id": str}, low_memory=False)
    history = pd.read_csv(history_path, dtype={"row_id": str})
    pool_table, strata_table = composition_diagnostics(results, metadata)
    metric_table, calibration_table = prediction_metrics(predictions)
    with gzip.open(out / "evaluation_events.jsonl.gz", "rt") as stream:
        events = (json.loads(line) for line in stream)
        line_raw, line_pool, line_summary = line_history_diagnostics(
            events, predictions, metadata, history, results
        )
    if {str(p.resolve()): digest(p) for p in paths} != before:
        raise ValueError("Diagnostic inputs changed while reading")
    destination = out / "diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    tables = {
        "pool_composition_primary.csv": pool_table,
        "pool_stratified_primary.csv": strata_table,
        "prediction_metrics.csv": metric_table,
        "prediction_calibration.csv": calibration_table,
        "line_history_event_primary.csv": line_raw,
        "line_history_pool_primary.csv": line_pool,
        "line_history_summary_primary.csv": line_summary,
    }
    for name, frame in tables.items():
        frame.to_csv(destination / name, index=False)
    record = {
        "input_hashes": before,
        "output_hashes": {name: digest(destination / name) for name in tables},
        "outputs_are_descriptive": True,
        "new_decision_subpools": False,
        "primary_budget": PRIMARY_BUDGET,
        "prediction_model_excluded": "yield: fixed100 recovery used solely to rank paddy yield",
        "calibration_bins": CALIBRATION_BINS,
        "policy_actions_changed": False,
        "random_seed_handling": "average repetitions within original pool before summary",
        "intervals_or_sampling_uncertainty": "none estimated",
        "history_definition": HISTORY_DEFINITION,
        "postfreeze_output_inputs": [
            "per_pool_results.csv",
            "prediction_diagnostics.csv",
            "evaluation_events.jsonl.gz",
        ],
    }
    write_json(destination / "diagnostics_manifest.json", record)
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    run_diagnostics(parser.parse_args().out)
