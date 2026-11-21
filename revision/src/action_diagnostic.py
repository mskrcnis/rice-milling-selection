"""Post hoc audit of frozen decisions; never fits or modifies the original study."""

from __future__ import annotations
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd

PRIMARY = ["uruguay_temporal", "arkansas_external"]
INPUTS = [
    "outputs/run_a/evaluation_events.jsonl.gz",
    "outputs/run_a/frozen_public_candidates.csv",
    "outputs/run_a/frozen_initial_predictions.csv",
    "outputs/run_a/selected_configuration.json",
    "outputs/run_a/per_pool_results.csv",
    "outputs/run_a/evaluation_freeze.json",
    "derived/uruguay/private_outcomes.csv",
    "derived/arkansas/private_outcomes.csv",
    "derived/cohort/all_public_metadata.csv",
    "protocol/split_manifest.csv",
]
CONTRASTS = [
    ("two_batch", "no_update"),
    ("two_batch", "mixed"),
    ("mixed", "yield"),
    ("adaptive_grid", "adaptive_grid_no_update"),
    ("adaptive_grid", "trees_20"),
    ("adaptive_grid_no_update", "trees_20"),
    ("adaptive_grid", "mixed"),
    ("trees_20", "mixed"),
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return pd.read_csv(path, float_precision="round_trip", low_memory=False)


def tie(row_id):
    return hashlib.sha256(("rice-milling-tie-v1|" + row_id).encode()).hexdigest()


def order(ids, values):
    return sorted(ids, key=lambda i: (-values[i], tie(i)))


def prepare_paths(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError("Output must be outside the read-only source project")
    output.mkdir(parents=True, exist_ok=True)
    return source, output


def candidates(source):
    public = read(source / "outputs/run_a/frozen_public_candidates.csv")
    public = public[public.stratum.isin(PRIMARY)].copy()
    answers = pd.concat(
        [
            read(source / f"derived/{p}/private_outcomes.csv")[["row_id", "hr_percent"]]
            for p in ["uruguay", "arkansas"]
        ],
        ignore_index=True,
    )
    assert (
        not answers.row_id.duplicated().any() and not public.row_id.duplicated().any()
    )
    frame = public.merge(answers, on="row_id", validate="one_to_one")
    assert len(frame) == len(public) and frame.hr_percent.notna().all()
    frame["index_value"] = frame.field_yield * frame.hr_percent / 100
    return frame


def read_events(source):
    wanted = {p for pair in CONTRASTS for p in pair}
    events = {}
    with gzip.open(source / "outputs/run_a/evaluation_events.jsonl.gz", "rt") as fh:
        for line in fh:
            x = json.loads(line)
            if (
                x["stratum"] in PRIMARY
                and x["policy"] in wanted
                and (x["budget_fraction"] in [0.25, 0.4] or x["policy"] == "yield")
            ):
                key = (x["pool_id"], x["policy"], x["budget_fraction"])
                assert key not in events
                events[key] = x
    return events


def score(group, event):
    ids = group.row_id.tolist()
    values = group.set_index("row_id").index_value.to_dict()
    answer = group.set_index("row_id").hr_percent.to_dict()
    n = len(ids)
    k = math.ceil(0.1 * n)
    budget = min(n, max(k, math.ceil(event["budget_fraction"] * n)))
    queried = event["tested_ids"]
    assert len(queried) == len(set(queried)) == budget
    assert event["n"] == n and event["k"] == k and event["budget"] == budget
    assert queried == [i for b in event["events"] for i in b["selected_ids"]]
    assert set(queried) <= set(ids)
    for batch in event["events"]:
        for i, value in batch["revealed_hr_pct"].items():
            assert math.isclose(value, answer[i], abs_tol=1e-10, rel_tol=1e-12)
    oracle = order(ids, values)[:k]
    chosen = order(queried, values)[:k]
    assert chosen == event["shortlist"], (
        event["pool_id"],
        event["policy"],
        chosen,
        event["shortlist"],
    )
    oracle_mean = math.fsum(values[i] for i in oracle) / k
    selected_mean = math.fsum(values[i] for i in chosen) / k
    regret = oracle_mean - selected_mean
    if abs(regret) < 1e-11:
        regret = 0.0
    return dict(
        n=n,
        k=k,
        tests=budget,
        actual_testing_fraction=budget / n,
        oracle_mean=oracle_mean,
        selected_mean=selected_mean,
        regret=regret,
        regret_pct=100 * regret / oracle_mean,
        top_k_recall=len(set(oracle) & set(chosen)) / k,
        best_entry_found=int(oracle[0] in chosen),
    )


def ranks(group, event, prediction, config):
    if len(event["events"]) != 2:
        return {"before": None, "after": None, "count": None}
    ids = group.row_id.tolist()
    positions = {x: i for i, x in enumerate(ids)}
    mean = prediction.reindex(ids).to_numpy()
    assert np.isfinite(mean).all()
    fy = group.field_yield.to_numpy()
    base = dict(zip(ids, fy * mean / 100))
    first = event["events"][0]["selected_ids"]
    remaining = [x for x in ids if x not in first]
    before = order(remaining, base)
    if event["policy"] in ["two_batch", "adaptive_grid"]:
        shrinkage = (
            config["secondary_adaptive"]["shrinkage"]
            if event["policy"] == "adaptive_grid"
            else config["shrinkage"]
        )
        at = np.array([positions[x] for x in first])
        rel = group.yield_rel.to_numpy()
        cls = group.grain_class.to_numpy()
        w = np.exp(-0.5 * ((rel[:, None] - rel[at][None, :]) / 0.25) ** 2)
        w *= np.where(cls[:, None] == cls[at][None, :], 1.0, 0.25)
        paid = event["events"][0]["revealed_hr_pct"]
        residual = np.array([paid[x] for x in first]) - mean[at]
        updated = np.clip(
            mean + (w @ residual) / (float(shrinkage) + w.sum(axis=1)), 0, 100
        )
    else:
        updated = mean
    after = order(remaining, dict(zip(ids, fy * updated / 100)))
    count = sum(a != b for a, b in zip(before, after))
    assert count == event["untested_rank_positions_changed"], (
        event["pool_id"],
        event["policy"],
        count,
        event["untested_rank_positions_changed"],
    )
    assert (
        after[: len(event["events"][1]["selected_ids"])]
        == event["events"][1]["selected_ids"]
    )
    return {"before": before, "after": after, "count": count}


def subset(frame):
    return np.where(
        frame.program.eq("uruguay"),
        "Uruguay 2017–2020",
        np.where(frame.year.le(2020), "Arkansas 2017–2020", "Arkansas 2021–2025"),
    )


def coverage(source, frame):
    full = read(source / "derived/cohort/all_public_metadata.csv")
    full = full[full.pool_id.isin(frame.pool_id)].copy()
    for c in ["observed_field", "paired"]:
        full[c] = full[c].astype(str).str.lower().eq("true")
    rows = []
    for pool, g in full.groupby("pool_id", sort=True):
        eligible = frame[frame.pool_id.eq(pool)]
        row = dict(
            pool_id=pool,
            program=g.program.iloc[0],
            year=int(g.year.iloc[0]),
            site=g.site.iloc[0],
            all_source_candidate_rows=len(g),
            observed_field_rows=int(g.observed_field.sum()),
            paired_entries=len(eligible),
        )
        assert int(g.paired.sum()) == len(eligible)
        row["coverage"] = row["paired_entries"] / row["observed_field_rows"]
        rows.append(row)
    pools = pd.DataFrame(rows)
    pools["period"] = subset(pools)
    summary = []
    for label, d in list(pools.groupby("program")) + list(pools.groupby("period")):
        summary.append(
            dict(
                group=label,
                pools=len(d),
                years=d.year.nunique(),
                sites=d.site.nunique(),
                all_source_candidate_rows=int(d.all_source_candidate_rows.sum()),
                observed_field_rows=int(d.observed_field_rows.sum()),
                paired_entries=int(d.paired_entries.sum()),
                missing_answers_among_observed_field=int(
                    (d.observed_field_rows - d.paired_entries).sum()
                ),
                incomplete_pools=int(d.coverage.lt(1).sum()),
                mean_pool_coverage=d.coverage.mean(),
                minimum_pool_coverage=d.coverage.min(),
                overall_entry_coverage=d.paired_entries.sum()
                / d.observed_field_rows.sum(),
            )
        )
    return pools, pd.DataFrame(summary)


def main(source_project, output_dir):
    source, out = prepare_paths(source_project, output_dir)
    before = {p: sha(source / p) for p in INPUTS}
    frame = candidates(source)
    groups = {p: g for p, g in frame.groupby("pool_id", sort=True)}
    events = read_events(source)
    predictions = read(source / "outputs/run_a/frozen_initial_predictions.csv")
    predictors = {
        m: g.set_index("row_id").mean_hr for m, g in predictions.groupby("model")
    }
    config = json.loads(
        (source / "outputs/run_a/selected_configuration.json").read_text()
    )
    assert config["best_single_batch"] == "mixed"
    recorded = read(source / "outputs/run_a/per_pool_results.csv").set_index(
        ["pool_id", "policy", "budget_fraction", "seed"]
    )
    checked = {}
    rank = {}
    score_rows = []
    for key, event in events.items():
        group = groups[key[0]]
        checked[key] = score(group, event)
        original = recorded.loc[(*key, 0)]
        for metric, value in checked[key].items():
            assert math.isclose(
                value, float(original[metric]), rel_tol=1e-10, abs_tol=1e-9
            ), (key, metric, value, original[metric])
        if event["budget_fraction"] in [0.25, 0.4]:
            rank[key] = ranks(group, event, predictors[event["model"]], config)
        score_rows.append(
            dict(
                pool_id=key[0],
                policy=key[1],
                budget_fraction=key[2],
                program=event["program"],
                stratum=event["stratum"],
                year=event["year"],
                site=event["site"],
                **checked[key],
            )
        )
    scores = pd.DataFrame(score_rows).sort_values(
        ["program", "pool_id", "budget_fraction", "policy"]
    )
    scores["period"] = subset(scores)
    scores.to_csv(out / "independently_recomputed_metrics.csv", index=False)
    records = []
    for pool, group in groups.items():
        for fraction in [0.25, 0.4]:
            for a, b in CONTRASTS:
                ka, kb = (pool, a, fraction), (pool, b, fraction)
                ea, eb = events[ka], events[kb]
                ra, rb = rank[ka], rank[kb]
                both_two = len(ea["events"]) == len(eb["events"]) == 2
                matched = (
                    both_two
                    and ea["events"][0]["selected_ids"]
                    == eb["events"][0]["selected_ids"]
                )
                if (a, b) in [
                    ("two_batch", "no_update"),
                    ("adaptive_grid", "adaptive_grid_no_update"),
                ]:
                    assert matched
                va, vb = checked[ka], checked[kb]
                delta = va["regret_pct"] - vb["regret_pct"]
                qset = set(ea["tested_ids"]) != set(eb["tested_ids"])
                sset = set(ea["shortlist"]) != set(eb["shortlist"])
                r = dict(
                    pool_id=pool,
                    program=ea["program"],
                    year=ea["year"],
                    site=ea["site"],
                    budget_fraction=fraction,
                    policy=a,
                    comparator=b,
                    n=ea["n"],
                    k=ea["k"],
                    budget=ea["budget"],
                    two_batch_comparison_applicable=both_two,
                    first_batch_order_matched=matched if both_two else None,
                    second_batch_comparison_applicable=matched,
                    second_batch_order_changed=(
                        (
                            ea["events"][1]["selected_ids"]
                            != eb["events"][1]["selected_ids"]
                        )
                        if matched
                        else None
                    ),
                    second_batch_set_changed=(
                        (
                            set(ea["events"][1]["selected_ids"])
                            != set(eb["events"][1]["selected_ids"])
                        )
                        if matched
                        else None
                    ),
                    policy_own_feedback_rank_positions_changed=ra["count"],
                    matched_untested_rank_positions_different=(
                        sum(x != y for x, y in zip(ra["after"], rb["after"]))
                        if matched
                        else None
                    ),
                    query_order_changed=ea["tested_ids"] != eb["tested_ids"],
                    queried_set_changed=qset,
                    queried_entries_replaced=len(
                        set(ea["tested_ids"]) - set(eb["tested_ids"])
                    ),
                    shortlist_order_changed=ea["shortlist"] != eb["shortlist"],
                    shortlist_set_changed=sset,
                    shortlisted_entries_replaced=len(
                        set(ea["shortlist"]) - set(eb["shortlist"])
                    ),
                    policy_regret_pct=va["regret_pct"],
                    comparator_regret_pct=vb["regret_pct"],
                    regret_difference_pp=delta,
                    measured_regret_changed=abs(delta) > 1e-10,
                    equal_regret_different_queries=abs(delta) <= 1e-10 and qset,
                    equal_regret_different_shortlist=abs(delta) <= 1e-10 and sset,
                    policy_tested_ids=json.dumps(ea["tested_ids"]),
                    comparator_tested_ids=json.dumps(eb["tested_ids"]),
                    policy_shortlist=json.dumps(ea["shortlist"]),
                    comparator_shortlist=json.dumps(eb["shortlist"]),
                )
                records.append(r)
    comparison = pd.DataFrame(records)
    comparison.to_csv(out / "pool_action_comparisons.csv", index=False)
    summary = []
    booleans = [
        "query_order_changed",
        "queried_set_changed",
        "shortlist_order_changed",
        "shortlist_set_changed",
        "measured_regret_changed",
        "equal_regret_different_queries",
        "equal_regret_different_shortlist",
    ]
    for key, d in comparison.groupby(
        ["program", "budget_fraction", "policy", "comparator"], sort=True
    ):
        r = dict(zip(["program", "budget_fraction", "policy", "comparator"], key))
        r["pools"] = len(d)
        own = d.policy_own_feedback_rank_positions_changed
        applicable = d.second_batch_comparison_applicable
        r.update(
            own_rank_comparison_pools=int(own.notna().sum()),
            pools_with_own_feedback_rank_change=(
                int(own.gt(0).sum()) if own.notna().any() else None
            ),
            second_batch_applicable_pools=int(applicable.sum()),
            pools_with_matched_untested_rank_difference=(
                int(d.matched_untested_rank_positions_different.gt(0).sum())
                if applicable.any()
                else None
            ),
            pools_with_second_batch_order_change=(
                int(d.loc[applicable, "second_batch_order_changed"].sum())
                if applicable.any()
                else None
            ),
            pools_with_second_batch_set_change=(
                int(d.loc[applicable, "second_batch_set_changed"].sum())
                if applicable.any()
                else None
            ),
            mean_regret_difference_pp=d.regret_difference_pp.mean(),
        )
        r.update({f"pools_with_{c}": int(d[c].sum()) for c in booleans})
        summary.append(r)
    pd.DataFrame(summary).to_csv(
        out / "action_comparison_summary.csv", index=False, na_rep="NA"
    )
    comparison[
        (comparison.policy.eq("adaptive_grid"))
        & (comparison.comparator.eq("adaptive_grid_no_update"))
        & comparison.measured_regret_changed
    ].to_csv(out / "secondary_feedback_regret_exceptions.csv", index=False)
    headline = []
    yield_scores = scores[scores.policy.eq("yield")]
    for label, d in list(yield_scores.groupby("program")) + list(
        yield_scores.groupby("period")
    ):
        for fraction, g in d.groupby("budget_fraction"):
            headline.append(
                dict(
                    group=label,
                    budget_fraction=fraction,
                    pools=len(g),
                    entries=int(g.n.sum()),
                    years=g.year.nunique(),
                    sites=g.site.nunique(),
                    shortlist_entries=int(g.k.sum()),
                    queried_entries=int(g.tests.sum()),
                    mean_recall=g.top_k_recall.mean(),
                    mean_regret_pct=g.regret_pct.mean(),
                    mean_best_entry_found=g.best_entry_found.mean(),
                    pools_best_entry_found=int(g.best_entry_found.sum()),
                    mean_actual_testing_fraction=g.actual_testing_fraction.mean(),
                    entry_weighted_testing_fraction=g.tests.sum() / g.n.sum(),
                    minimum_actual_testing_fraction=g.actual_testing_fraction.min(),
                    maximum_actual_testing_fraction=g.actual_testing_fraction.max(),
                )
            )
    hd = pd.DataFrame(headline)
    hd.to_csv(out / "headline_and_period_metrics.csv", index=False)
    crossing = (
        hd[hd.mean_recall.ge(0.9)]
        .sort_values("budget_fraction")
        .groupby("group")
        .first()
        .reset_index()
    )
    crossing.to_csv(out / "first_measured_grid_crossing.csv", index=False)
    cp, cs = coverage(source, frame)
    cp.to_csv(out / "pool_coverage.csv", index=False)
    cs.to_csv(out / "coverage_summary.csv", index=False)
    for p, h in before.items():
        assert sha(source / p) == h
    manifest = {
        "status": "pass",
        "scope": "post_hoc_descriptive_action_and_independent_metric_audit",
        "source_inputs": before,
        "code_sha256": sha(Path(__file__)),
        "primary_pools": len(groups),
        "primary_candidates": len(frame),
        "events_independently_scored": len(events),
        "pool_policy_contrasts": len(comparison),
        "regret_equality_tolerance_pp": 1e-10,
        "original_files_changed": False,
        "static_second_batch": "not_applicable",
        "outputs": {p.name: sha(p) for p in sorted(out.glob("*.csv"))},
    }
    plan = (
        Path(__file__).resolve().parents[1]
        / "protocol/action_and_breeder_diagnostic_plan.md"
    )
    if plan.exists():
        manifest["diagnostic_plan_sha256"] = sha(plan)
    (out / "verification.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                k: manifest[k]
                for k in [
                    "status",
                    "primary_pools",
                    "primary_candidates",
                    "events_independently_scored",
                    "pool_policy_contrasts",
                ]
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    main(args.source_project, args.output_dir)
