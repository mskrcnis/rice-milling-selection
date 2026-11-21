"""Independently reconstruct aggregate tables and a worked decision example.

No production analysis, replay, experiment or model functions are imported.
Real evaluation files are opened only when verify_tables is explicitly called,
after a passing event verification and its bound input hashes are checked.
"""

import argparse
from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .common import digest, verify_freeze, write_json


METRICS = (
    "regret",
    "regret_pct",
    "top_k_recall",
    "best_entry_found",
    "tests",
    "actual_testing_fraction",
    "top_ten_count",
    "top_ten_recall",
)
PAIRED_METRICS = ("regret_pct", "regret", "top_k_recall", "best_entry_found")
POOL_KEYS = ("stratum", "program", "year", "site", "pool_id", "budget_fraction")
SEED_KEYS = (
    "stratum",
    "program",
    "year",
    "site",
    "pool_id",
    "policy",
    "budget_fraction",
)
ABSOLUTE_TOLERANCE = 1e-9
BOOTSTRAP_SEED = 1731
BOOTSTRAP_REPETITIONS = 4000


def require(condition, message):
    if not condition:
        raise ValueError("Independent table verification: " + message)


def finite_values(values):
    result = []
    for value in values:
        value = float(value)
        if math.isnan(value):
            continue
        require(math.isfinite(value), "infinite metric encountered")
        result.append(value)
    return result


def mean(values):
    values = finite_values(values)
    return math.fsum(values) / len(values) if values else float("nan")


def groups(records, fields):
    result = defaultdict(list)
    for row in records:
        result[tuple(row[f] for f in fields)].append(row)
    return result


def interval(records, metric):
    by_year = defaultdict(list)
    for row in records:
        value = float(row[metric])
        if not math.isnan(value):
            require(math.isfinite(value), "infinite bootstrap input")
            by_year[int(row["year"])].append(value)
    years = sorted(by_year)
    if len(years) < 2:
        return float("nan"), float("nan"), len(years)
    totals = np.asarray([math.fsum(by_year[y]) for y in years], dtype=float)
    counts = np.asarray([len(by_year[y]) for y in years], dtype=int)
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    sampled_years = generator.integers(
        0, len(years), size=(BOOTSTRAP_REPETITIONS, len(years))
    )
    replicates = np.sum(totals[sampled_years], axis=1) / np.sum(
        counts[sampled_years], axis=1
    )
    lower, upper = np.quantile(replicates, [0.025, 0.975], method="linear")
    return float(lower), float(upper), len(years)


def reconstruct_tables(regenerated, config):
    source = regenerated.to_dict("records")
    require(source, "empty independently regenerated results")
    averaged = []
    for token, records in sorted(groups(source, SEED_KEYS).items()):
        require(
            len({int(row["seed"]) for row in records}) == len(records),
            "duplicate seed within a policy/pool",
        )
        row = dict(zip(SEED_KEYS, token))
        for metric in METRICS + ("rank_positions_changed",):
            row[metric] = mean(record[metric] for record in records)
        averaged.append(row)
    summary = []
    for token, records in sorted(
        groups(averaged, ("stratum", "policy", "budget_fraction")).items()
    ):
        row = dict(zip(("stratum", "policy", "budget_fraction"), token))
        row.update(
            pools=len({r["pool_id"] for r in records}),
            years=len({r["year"] for r in records}),
            sites=len({r["site"] for r in records}),
        )
        for metric in METRICS:
            row[metric] = mean(r[metric] for r in records)
            row[metric + "_lower"], row[metric + "_upper"], _ = interval(
                records, metric
            )
        row["uncertainty"] = (
            "descriptive harvest-year block bootstrap; recurring sites and temporal shift remain limitations"
        )
        summary.append(row)

    by_policy = defaultdict(dict)
    for row in averaged:
        token = tuple(row[k] for k in POOL_KEYS)
        require(
            token not in by_policy[row["policy"]], "duplicate averaged pool identity"
        )
        by_policy[row["policy"]][token] = row

    def paired_records(adaptive, comparator):
        a, b = by_policy[adaptive], by_policy[comparator]
        require(
            a and set(a) == set(b),
            "unmatched pools for " + adaptive + " versus " + comparator,
        )
        rows = []
        for token in sorted(a):
            for metric in PAIRED_METRICS:
                left, right = float(a[token][metric]), float(b[token][metric])
                rows.append(
                    {
                        **dict(zip(POOL_KEYS, token)),
                        "comparator": comparator,
                        "metric": metric,
                        "adaptive": left,
                        "comparator_value": right,
                        "difference": left - right,
                    }
                )
        return rows

    comparators = list(
        dict.fromkeys(
            [
                config["best_single_batch"],
                "no_update",
                "exploit_only",
                "yield",
                "mixed",
                "class_history",
            ]
        )
    )
    paired = [
        row
        for comparator in comparators
        for row in paired_records("two_batch", comparator)
    ]
    paired_summary = []
    fields = ("stratum", "comparator", "metric", "budget_fraction")
    for token, records in sorted(groups(paired, fields).items()):
        row = dict(zip(fields, token))
        lower, upper, years = interval(records, "difference")
        row.update(
            pools=len(records),
            years=years,
            sites=len({r["site"] for r in records}),
            difference=mean(r["difference"] for r in records),
            lower=lower,
            upper=upper,
            negative_favors_adaptive=row["metric"].startswith("regret"),
            interpretation="descriptive; no operational superiority margin has been established",
        )
        paired_summary.append(row)
    annual = []
    annual_keys = ("stratum", "year", "comparator", "metric", "budget_fraction")
    for token, records in sorted(groups(paired, annual_keys).items()):
        annual.append(
            {
                **dict(zip(annual_keys, token)),
                "pools": len({r["pool_id"] for r in records}),
                "difference": mean(r["difference"] for r in records),
            }
        )

    tables = {
        "pool_seed_averages.csv": (pd.DataFrame(averaged), SEED_KEYS),
        "policy_summary.csv": (
            pd.DataFrame(summary),
            ("stratum", "policy", "budget_fraction"),
        ),
        "paired_pool_differences.csv": (
            pd.DataFrame(paired),
            POOL_KEYS + ("comparator", "metric"),
        ),
        "paired_summary.csv": (pd.DataFrame(paired_summary), fields),
        "paired_by_year.csv": (pd.DataFrame(annual), annual_keys),
    }
    if "secondary_adaptive" in config:
        secondary = []
        comparators = list(
            dict.fromkeys(
                [config["best_single_batch"], "adaptive_grid_no_update", "yield"]
            )
        )
        for comparator in comparators:
            differences = paired_records("adaptive_grid", comparator)
            for token, records in sorted(
                groups(differences, ("stratum", "metric", "budget_fraction")).items()
            ):
                row = dict(zip(("stratum", "metric", "budget_fraction"), token))
                lower, upper, years = interval(records, "difference")
                row.update(
                    policy="adaptive_grid",
                    comparator=comparator,
                    pools=len(records),
                    years=years,
                    difference=mean(r["difference"] for r in records),
                    lower=lower,
                    upper=upper,
                    status="development_selected_secondary; primary conclusions remain separate",
                )
                secondary.append(row)
        tables["secondary_adaptive_paired.csv"] = (
            pd.DataFrame(secondary),
            ("stratum", "policy", "comparator", "metric", "budget_fraction"),
        )
    return tables


def compare_table(expected, actual, keys, filename):
    require(
        set(expected.columns) == set(actual.columns),
        "column schema differs in " + filename,
    )
    require(len(expected) == len(actual), "row count differs in " + filename)
    require(
        not actual.duplicated(list(keys)).any(), "duplicate row identity in " + filename
    )
    expected_rows = {tuple(r[k] for k in keys): r for r in expected.to_dict("records")}
    actual_rows = {tuple(r[k] for k in keys): r for r in actual.to_dict("records")}
    require(
        set(expected_rows) == set(actual_rows), "row identities differ in " + filename
    )
    count_columns = {"pools", "years", "sites"}
    maximum_error = 0.0
    for token, wanted in expected_rows.items():
        observed = actual_rows[token]
        for column, value in wanted.items():
            other = observed[column]
            if column in count_columns:
                require(
                    float(other).is_integer() and int(other) == value,
                    "count mismatch in " + filename + ":" + column,
                )
            elif isinstance(value, (bool, np.bool_)):
                require(
                    isinstance(other, (bool, np.bool_)) and bool(other) == bool(value),
                    "boolean interpretation mismatch in " + filename,
                )
            elif isinstance(value, str):
                require(
                    other == value,
                    "text metadata mismatch in " + filename + ":" + column,
                )
            else:
                a, b = float(value), float(other)
                if math.isnan(a) or math.isnan(b):
                    require(
                        math.isnan(a) and math.isnan(b),
                        "undefined metric changed in " + filename + ":" + column,
                    )
                else:
                    difference = abs(a - b)
                    maximum_error = max(maximum_error, difference)
                    require(
                        math.isfinite(difference) and difference <= ABSOLUTE_TOLERANCE,
                        "numeric mismatch in "
                        + filename
                        + ":"
                        + column
                        + " at "
                        + repr(token),
                    )
    return {
        "rows": len(expected),
        "columns": len(expected.columns),
        "max_absolute_error": maximum_error,
        "identity_columns": list(keys),
    }


def value_rank(values):
    return sorted(
        values,
        key=lambda i: (
            -values[i],
            hashlib.sha256(("rice-milling-tie-v1|" + i).encode()).hexdigest(),
        ),
    )


def bound_path(summary, names):
    matches = [Path(p) for p in summary["input_sha256"] if Path(p).name in names]
    require(
        len(matches) == 1,
        "cannot uniquely locate bound input: " + ",".join(sorted(names)),
    )
    return matches[0]


def worked_pool(out, regenerated, event_summary):
    public_path = bound_path(event_summary, {"evaluation_public.csv", "public.csv"})
    key_path = bound_path(event_summary, {"evaluation_key.csv", "key.csv"})
    public = pd.read_csv(
        public_path, dtype={"row_id": str, "pool_id": str, "candidate_id": str}
    )
    key = (
        pd.read_csv(key_path, dtype={"row_id": str})
        .set_index("row_id")
        .hr_pct.to_dict()
    )
    available = regenerated[
        (regenerated.policy == "two_batch") & (regenerated.budget_fraction == 0.25)
    ]
    require(not available.empty, "no primary-budget pool available for worked example")
    priority = {"uruguay_temporal": 0, "arkansas_external": 1}
    stratum = sorted(available.stratum.unique(), key=lambda x: (priority.get(x, 2), x))[
        0
    ]
    choices = available[available.stratum == stratum].copy()
    median_size = float(choices.n.median())
    chosen_row = min(
        choices.to_dict("records"),
        key=lambda r: (abs(r["n"] - median_size), str(r["pool_id"])),
    )
    pool = str(chosen_row["pool_id"])
    candidates = public[public.pool_id == pool].sort_values("row_id").copy()
    n = len(candidates)
    k = math.ceil(n / 10)
    budget = min(n, max(k, math.ceil(0.25 * n)))
    values = {
        row.row_id: float(row.field_yield) * float(key[row.row_id]) / 100.0
        for row in candidates.itertuples()
    }
    rank = value_rank(values)
    oracle = rank[:k]
    event_path = bound_path(event_summary, {"evaluation_events.jsonl.gz"})
    selected_event = None
    with gzip.open(event_path, "rt") as handle:
        for line in handle:
            event = json.loads(line)
            if (
                str(event["pool_id"]) == pool
                and event["policy"] == "two_batch"
                and event["budget_fraction"] == 0.25
            ):
                require(selected_event is None, "duplicate worked-example event")
                selected_event = event
    require(selected_event is not None, "worked-example event not found")
    tested = [i for batch in selected_event["events"] for i in batch["selected_ids"]]
    require(
        len(tested) == budget and len(set(tested)) == budget,
        "worked-example budget inconsistency",
    )
    selected = [i for i in rank if i in set(tested)][:k]
    require(
        selected == selected_event["shortlist"], "worked-example shortlist mismatch"
    )
    query_metadata = {}
    for batch in selected_event["events"]:
        for i in batch["selected_ids"]:
            query_metadata[i] = (len(query_metadata) + 1, int(batch["batch"]))
    oracle_total = math.fsum(values[i] for i in oracle)
    selected_total = math.fsum(values[i] for i in selected)
    oracle_mean, selected_mean = oracle_total / k, selected_total / k
    regret = oracle_mean - selected_mean
    percentage = 100 * regret / oracle_mean if oracle_mean > 0 else float("nan")
    recall = len(set(selected) & set(oracle)) / k
    for column, value in [
        ("oracle_mean", oracle_mean),
        ("selected_mean", selected_mean),
        ("regret", regret),
        ("regret_pct", percentage),
        ("top_k_recall", recall),
    ]:
        recorded = float(chosen_row[column])
        require(
            (math.isnan(value) and math.isnan(recorded))
            or abs(recorded - value) <= ABSOLUTE_TOLERANCE,
            "worked-example arithmetic differs from independent per-pool result: "
            + column,
        )
    audit_rows = []
    for row in candidates.itertuples():
        order, batch = query_metadata.get(row.row_id, (np.nan, np.nan))
        audit_rows.append(
            {
                "row_id": row.row_id,
                "candidate_id": row.candidate_id,
                "field_yield": float(row.field_yield),
                "hr_pct": key[row.row_id],
                "output_index": values[row.row_id],
                "oracle_rank": rank.index(row.row_id) + 1,
                "query_order": order,
                "batch": batch,
                "oracle_top_k": row.row_id in oracle,
                "tested": row.row_id in query_metadata,
                "shortlisted": row.row_id in selected,
            }
        )
    destination = out / "verification"
    candidate_file = destination / "worked_pool_candidates.csv"
    pd.DataFrame(audit_rows).sort_values("oracle_rank").to_csv(
        candidate_file, index=False, float_format="%.17g"
    )
    info = {r["row_id"]: r for r in audit_rows}

    def number(value):
        return "undefined" if math.isnan(float(value)) else f"{float(value):.12g}"

    def safe(value):
        return str(value).replace("|", "\\|").replace("\n", " ")

    lines = [
        "# Independently worked primary-budget pool",
        "",
        f'Pool `{safe(pool)}`; stratum `{safe(stratum)}`; policy `two_batch`; harvest year {int(chosen_row["year"])}.',
        "",
        "Selection used metadata only: prefer the Uruguay temporal stratum, then choose a pool closest to its median candidate count, with pool ID as the tie-break. This is an auditable worked example, not proof of population representativeness.",
        "",
        f"n={n}; k=ceil(0.10×{n})={k}; B=ceil(0.25×{n})={budget}. Each candidate output is field yield × HR/100, a source-basis output index. No cross-program moisture harmonization is implied.",
        "",
        "The following rows are in paid query order. HR appears here only after the frozen evaluation. Displayed numbers are rounded; the companion CSV retains17 significant digits.",
        "",
        "| Query | Batch | Candidate / row | Field yield | HR (%) | Yield × HR/100 | Oracle top k | Shortlisted |",
        "|---:|---:|---|---:|---:|---:|:---:|:---:|",
    ]
    for i in tested:
        r = info[i]
        lines.append(
            f'| {int(r["query_order"])} | {int(r["batch"])} | {safe(r["candidate_id"])} / `{safe(i)}` | {number(r["field_yield"])} | {number(r["hr_pct"])} | {number(r["output_index"])} | {"yes" if r["oracle_top_k"] else "no"} | {"yes" if r["shortlisted"] else "no"} |'
        )
    lines += [
        "",
        "Oracle top-k entries (including any not queried):",
        "",
        "| Oracle rank | Candidate / row | Output index | Queried |",
        "|---:|---|---:|:---:|",
    ]
    for i in oracle:
        r = info[i]
        lines.append(
            f'| {r["oracle_rank"]} | {safe(r["candidate_id"])} / `{safe(i)}` | {number(values[i])} | {"yes" if i in query_metadata else "no"} |'
        )
    lines += [
        "",
        "Arithmetic:",
        "",
        "- Oracle mean = ("
        + " + ".join(number(values[i]) for i in oracle)
        + f")/{k} = {number(oracle_mean)}.",
        "- Selected mean = ("
        + " + ".join(number(values[i]) for i in selected)
        + f")/{k} = {number(selected_mean)}.",
        f"- Shortlist regret = {number(oracle_mean)} − {number(selected_mean)} = {number(regret)}.",
        f"- Relative regret =100×regret/oracle mean = {number(percentage)}%.",
        f"- Top-k recall = {len(set(selected) & set(oracle))}/{k} = {number(recall)}.",
        f"- Best entry recovered = {int(oracle[0] in selected)}; entries tested ={budget}/{n}.",
        "",
        "Oracle and tested-only shortlist use descending output and the frozen SHA-256 tie rule. The shortlist is independently reconstructed from paid results. The complete candidate table is [worked_pool_candidates.csv](worked_pool_candidates.csv).",
        "",
    ]
    report_path = destination / "worked_pool_audit.md"
    report_path.write_text("\n".join(lines))
    return {
        "pool_id": pool,
        "stratum": stratum,
        "year": int(chosen_row["year"]),
        "n": n,
        "k": k,
        "budget": budget,
        "selection_rule": "Preferred temporal stratum; nearest median candidate count; lexical pool-ID tie-break; no outcome selection.",
        "arithmetic_passed": True,
        "report": str(report_path),
        "candidate_table": str(candidate_file),
        "report_sha256": digest(report_path),
        "candidate_table_sha256": digest(candidate_file),
    }


def verify_tables(out):
    out = Path(out).resolve()
    destination = out / "verification"
    event_summary_path = destination / "verification_summary.json"
    require(
        event_summary_path.is_file(),
        "passing independent event verification is required first",
    )
    event_summary = json.loads(event_summary_path.read_text())
    require(event_summary.get("status") == "passed", "event verification did not pass")
    regenerated_path = destination / "regenerated_per_pool_results.csv"
    require(
        digest(regenerated_path) == event_summary["regenerated_results_sha256"],
        "regenerated per-pool table changed",
    )
    freeze_path = out / "evaluation_freeze.json"
    frozen_count = verify_freeze(freeze_path)
    for name, expected_hash in event_summary["input_sha256"].items():
        require(
            digest(name) == expected_hash, "event-verification input changed: " + name
        )
    config_path = out / "selected_configuration.json"
    require(
        str(config_path) in event_summary["input_sha256"],
        "configuration not bound by event verifier",
    )
    config = json.loads(config_path.read_text())
    regenerated = pd.read_csv(
        regenerated_path,
        dtype={
            "pool_id": str,
            "policy": str,
            "site": str,
            "program": str,
            "stratum": str,
        },
    )
    expected_tables = reconstruct_tables(regenerated, config)
    checked = {}
    expected_directory = destination / "regenerated_tables"
    expected_directory.mkdir(exist_ok=True)
    for filename, (expected, identity) in expected_tables.items():
        actual_path = out / "tables" / filename
        require(actual_path.is_file(), "required aggregate table missing: " + filename)
        actual = pd.read_csv(actual_path)
        checked[filename] = compare_table(expected, actual, identity, filename)
        expected_path = expected_directory / filename
        expected.to_csv(expected_path, index=False)
        checked[filename].update(
            actual_sha256=digest(actual_path), regenerated_sha256=digest(expected_path)
        )
    example = worked_pool(out, regenerated, event_summary)
    report = {
        "status": "passed",
        "tables": checked,
        "table_count": len(checked),
        "absolute_numeric_tolerance": ABSOLUTE_TOLERANCE,
        "relative_numeric_tolerance": 0.0,
        "bootstrap": {
            "method": "equal-pool statistic after harvest-year resampling, preserving each sampled year pool count",
            "replicates": BOOTSTRAP_REPETITIONS,
            "seed": BOOTSTRAP_SEED,
            "generator": "PCG64",
            "quantiles": [0.025, 0.975],
            "quantile_method": "linear",
            "fewer_than_two_valid_years": "undefined interval; point estimate retained",
        },
        "seed_handling": "average random seeds within each pool before policy means, contrasts and bootstrap",
        "effect_direction": "adaptive minus comparator; negative regret differences favor adaptive",
        "independence": "No imports from analysis/replay/experiment/models; group means, differences and bootstrap reconstructed separately.",
        "event_verification_summary_sha256": digest(event_summary_path),
        "regenerated_per_pool_sha256": digest(regenerated_path),
        "verifier_source_sha256": digest(__file__),
        "frozen_run_files_rechecked": frozen_count,
        "worked_pool": example,
        "additional_tables_not_verified_by_this_module": sorted(
            p.name for p in (out / "tables").glob("*.csv") if p.name not in checked
        ),
    }
    write_json(destination / "table_verification.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = verify_tables(args.out)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
