from pathlib import Path
import gzip
import json
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "rice-milling-selection-v1"
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
from .common import ROOT, write_json

METRICS = [
    "regret",
    "regret_pct",
    "top_k_recall",
    "best_entry_found",
    "tests",
    "actual_testing_fraction",
    "top_ten_count",
    "top_ten_recall",
]


def pool_average(results):
    identity = [
        "stratum",
        "program",
        "year",
        "site",
        "pool_id",
        "policy",
        "budget_fraction",
    ]
    return (
        results.groupby(identity, dropna=False)[METRICS + ["rank_positions_changed"]]
        .mean()
        .reset_index()
    )


def year_interval(frame, column, seed=1731, repetitions=4000):
    frame = frame.loc[frame[column].notna()]
    groups = frame.groupby("year")[column].agg(["sum", "count"])
    if len(groups) < 2:
        return (np.nan, np.nan, len(groups))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), size=(repetitions, len(groups)))
    values = groups["sum"].to_numpy()[draws].sum(axis=1) / groups["count"].to_numpy()[
        draws
    ].sum(axis=1)
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high), len(groups)


def aggregate(results, config, out):
    out = Path(out)
    tables = out / "tables"
    tables.mkdir(exist_ok=True)
    averaged = pool_average(results)
    averaged.to_csv(tables / "pool_seed_averages.csv", index=False)
    summaries = []
    for (stratum, policy, fraction), part in averaged.groupby(
        ["stratum", "policy", "budget_fraction"], sort=True
    ):
        row = {
            "stratum": stratum,
            "policy": policy,
            "budget_fraction": fraction,
            "pools": part.pool_id.nunique(),
            "years": part.year.nunique(),
            "sites": part.site.nunique(),
        }
        for metric in METRICS:
            row[metric] = part[metric].mean()
            low, high, _ = year_interval(part, metric)
            row[metric + "_lower"] = low
            row[metric + "_upper"] = high
        row["uncertainty"] = (
            "descriptive harvest-year block bootstrap; recurring sites and temporal shift remain limitations"
        )
        summaries.append(row)
    summary = pd.DataFrame(summaries)
    summary.to_csv(tables / "policy_summary.csv", index=False)
    pairs = []
    benchmarks = [
        config["best_single_batch"],
        "no_update",
        "exploit_only",
        "yield",
        "mixed",
        "class_history",
    ]
    for comparator in dict.fromkeys(benchmarks):
        a = averaged[averaged.policy.eq("two_batch")]
        b = averaged[averaged.policy.eq(comparator)]
        keys = ["stratum", "program", "year", "site", "pool_id", "budget_fraction"]
        joined = a.merge(
            b, on=keys, suffixes=("_adaptive", "_comparator"), validate="one_to_one"
        )
        for metric in ["regret_pct", "regret", "top_k_recall", "best_entry_found"]:
            for _, row in joined.iterrows():
                pairs.append(
                    {
                        **{key: row[key] for key in keys},
                        "comparator": comparator,
                        "metric": metric,
                        "adaptive": row[metric + "_adaptive"],
                        "comparator_value": row[metric + "_comparator"],
                        "difference": row[metric + "_adaptive"]
                        - row[metric + "_comparator"],
                    }
                )
    paired = pd.DataFrame(pairs)
    paired.to_csv(tables / "paired_pool_differences.csv", index=False)
    contrasts = []
    for (stratum, comparator, metric, fraction), part in paired.groupby(
        ["stratum", "comparator", "metric", "budget_fraction"]
    ):
        low, high, years = year_interval(part, "difference")
        contrasts.append(
            {
                "stratum": stratum,
                "comparator": comparator,
                "metric": metric,
                "budget_fraction": fraction,
                "pools": len(part),
                "years": years,
                "sites": part.site.nunique(),
                "difference": part.difference.mean(),
                "lower": low,
                "upper": high,
                "negative_favors_adaptive": metric.startswith("regret"),
                "interpretation": "descriptive; no operational superiority margin has been established",
            }
        )
    pd.DataFrame(contrasts).to_csv(tables / "paired_summary.csv", index=False)
    contemporaneous = paired[
        paired.stratum.eq("arkansas_external") & paired.year.ge(2021)
    ]
    contemporary_rows = []
    for (comparator, metric, fraction), part in contemporaneous.groupby(
        ["comparator", "metric", "budget_fraction"]
    ):
        low, high, years = year_interval(part, "difference")
        contemporary_rows.append(
            {
                "stratum": "arkansas_contemporaneous_2021_2025",
                "comparator": comparator,
                "metric": metric,
                "budget_fraction": fraction,
                "pools": len(part),
                "years": years,
                "difference": part.difference.mean(),
                "lower": low,
                "upper": high,
            }
        )
    pd.DataFrame(contemporary_rows).to_csv(
        tables / "contemporaneous_external_sensitivity.csv", index=False
    )
    annual = (
        paired.groupby(["stratum", "year", "comparator", "metric", "budget_fraction"])
        .agg(pools=("pool_id", "nunique"), difference=("difference", "mean"))
        .reset_index()
    )
    annual.to_csv(tables / "paired_by_year.csv", index=False)
    sensitivity = []
    primary = paired[paired.budget_fraction.eq(0.25) & paired.metric.eq("regret_pct")]
    for (stratum, comparator), part in primary.groupby(["stratum", "comparator"]):
        for dimension in ["year", "site"]:
            for omitted in sorted(part[dimension].unique()):
                remaining = part[part[dimension].ne(omitted)]
                sensitivity.append(
                    {
                        "stratum": stratum,
                        "comparator": comparator,
                        "omitted_dimension": dimension,
                        "omitted_value": omitted,
                        "remaining_pools": len(remaining),
                        "difference": remaining.difference.mean(),
                    }
                )
    pd.DataFrame(sensitivity).to_csv(tables / "leave_block_out.csv", index=False)
    target = []
    for (stratum, pool_id, policy), part in averaged.groupby(
        ["stratum", "pool_id", "policy"]
    ):
        reached = part[part.top_k_recall.ge(0.9)].sort_values("budget_fraction")
        target.append(
            {
                "stratum": stratum,
                "pool_id": pool_id,
                "policy": policy,
                "target_recall": 0.9,
                "first_observed_grid_budget": (
                    reached.budget_fraction.iloc[0] if len(reached) else np.nan
                ),
                "actual_fraction_at_target": (
                    reached.actual_testing_fraction.iloc[0] if len(reached) else np.nan
                ),
            }
        )
    pd.DataFrame(target).to_csv(tables / "budget_target.csv", index=False)
    feedback = (
        averaged[averaged.policy.eq("two_batch")]
        .groupby(["stratum", "budget_fraction"])
        .agg(
            pools=("pool_id", "nunique"),
            pools_with_ranking_change=(
                "rank_positions_changed",
                lambda x: int((x > 0).sum()),
            ),
            mean_changed_rank_positions=("rank_positions_changed", "mean"),
        )
        .reset_index()
    )
    feedback.to_csv(tables / "feedback_rank_changes.csv", index=False)
    with gzip.open(out / "evaluation_events.jsonl.gz", "rt") as handle:
        event_lookup = {
            (e["pool_id"], e["policy"], e["budget_fraction"]): e
            for line in handle
            if (e := json.loads(line))["policy"] in ("two_batch", "no_update")
        }
    action_rows = []
    for (pool_id, policy, fraction), event in event_lookup.items():
        if policy != "two_batch":
            continue
        other = event_lookup[(pool_id, "no_update", fraction)]
        initial_identical = (
            event["events"][0]["selected_ids"] == other["events"][0]["selected_ids"]
        )
        if not initial_identical:
            raise ValueError("Feedback ablation first batch differs")
        action_rows.append(
            {
                "pool_id": pool_id,
                "stratum": event["stratum"],
                "year": event["year"],
                "site": event["site"],
                "budget_fraction": fraction,
                "first_batch_identical": initial_identical,
                "tested_members_changed": len(
                    set(event["tested_ids"]) ^ set(other["tested_ids"])
                )
                // 2,
                "shortlist_members_changed": len(
                    set(event["shortlist"]) ^ set(other["shortlist"])
                )
                // 2,
                "untested_rank_positions_changed": event[
                    "untested_rank_positions_changed"
                ],
            }
        )
    pd.DataFrame(action_rows).to_csv(
        tables / "feedback_action_changes.csv", index=False
    )
    if "secondary_adaptive" in config:
        secondary_rows = []
        for comparator in dict.fromkeys(
            [config["best_single_batch"], "adaptive_grid_no_update", "yield"]
        ):
            keys = ["stratum", "program", "year", "site", "pool_id", "budget_fraction"]
            joined = averaged[averaged.policy.eq("adaptive_grid")].merge(
                averaged[averaged.policy.eq(comparator)],
                on=keys,
                suffixes=("_adaptive", "_comparator"),
                validate="one_to_one",
            )
            for metric in ["regret_pct", "regret", "top_k_recall", "best_entry_found"]:
                joined["difference"] = (
                    joined[metric + "_adaptive"] - joined[metric + "_comparator"]
                )
                for (stratum, fraction), part in joined.groupby(
                    ["stratum", "budget_fraction"]
                ):
                    low, high, years = year_interval(part, "difference")
                    secondary_rows.append(
                        {
                            "stratum": stratum,
                            "policy": "adaptive_grid",
                            "comparator": comparator,
                            "metric": metric,
                            "budget_fraction": fraction,
                            "pools": len(part),
                            "years": years,
                            "difference": part.difference.mean(),
                            "lower": low,
                            "upper": high,
                            "status": "development_selected_secondary; primary conclusions remain separate",
                        }
                    )
        pd.DataFrame(secondary_rows).to_csv(
            tables / "secondary_adaptive_paired.csv", index=False
        )
    return summary, pd.DataFrame(contrasts)


def plot_development(out):
    out = Path(out)
    (out / "figures").mkdir(exist_ok=True)
    frame = pd.read_csv(out / "development_per_pool.csv")
    averaged = pool_average(frame)
    rows = []
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for name, color in [("yield", "#2b6a99"), ("random", "#888888")]:
        part = averaged[averaged.policy.eq(name)]
        curve = []
        for fraction, group in part.groupby("budget_fraction"):
            lo, hi, years = year_interval(group, "top_k_recall")
            row = {
                "policy": name,
                "budget_fraction": fraction,
                "recall": group.top_k_recall.mean(),
                "lower": lo,
                "upper": hi,
                "pools": group.pool_id.nunique(),
                "years": years,
                "sites": group.site.nunique(),
            }
            rows.append(row)
            curve.append(row)
        d = pd.DataFrame(curve)
        ax.plot(
            d.budget_fraction * 100,
            d.recall,
            marker="o",
            color=color,
            label=(
                "Paddy yield first"
                if name == "yield"
                else "Random (20 seeds averaged per pool)"
            ),
        )
        ax.fill_between(
            d.budget_fraction * 100, d.lower, d.upper, color=color, alpha=0.15
        )
    ax.axhline(
        0.9, color="black", ls=":", lw=1, label="Illustrative 90% recovery target"
    )
    ax.set(
        xlabel="Design budget (% of observed-answer candidates)",
        ylabel="Oracle top-k recovery",
        ylim=(0, 1.03),
        title=f"Development only: {frame.pool_id.nunique()} pools, {frame.year.nunique()} harvest years",
    )
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "figures/development_baseline_curve.png", dpi=220)
    fig.savefig(out / "figures/development_baseline_curve.svg", metadata={"Date": None})
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out / "development_baseline_curve.csv", index=False)
    primary = averaged[averaged.policy.eq("yield") & averaged.budget_fraction.eq(0.25)]
    write_json(
        out / "development_baseline_diagnosis.json",
        {
            "pools": len(primary),
            "years": primary.year.nunique(),
            "yield_first_recall_at_25pct": primary.top_k_recall.mean(),
            "maximum_possible_recall_gain_at_25pct": 1 - primary.top_k_recall.mean(),
            "yield_first_regret_pct_at_25pct": primary.regret_pct.mean(),
            "scope": "Development only. Full information gives recall1 and regret0 by definition; remaining headroom is not an attainable adaptive benefit.",
        },
    )


def plot_final(out, config):
    out = Path(out)
    figures = out / "figures"
    figures.mkdir(exist_ok=True)
    summary = pd.read_csv(out / "tables/policy_summary.csv")
    primary_strata = ["uruguay_temporal", "arkansas_external"]
    colors = {
        "two_batch": "#157a6e",
        "no_update": "#d17b32",
        config["best_single_batch"]: "#3e5caa",
        "yield": "#666666",
        "random": "#aaaaaa",
    }
    labels = {
        "two_batch": "Two batches + feedback",
        "no_update": "Same first batch, no update",
        config["best_single_batch"]: "Locked best one batch ("
        + config["best_single_batch"]
        + ")",
        "yield": "Paddy yield first",
        "random": "Random",
    }
    for metric, ylabel in [
        ("regret_pct", "Shortlist regret (% of oracle mean)"),
        ("top_k_recall", "Oracle top-k recovery"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
        for ax, stratum in zip(axes, primary_strata):
            for policy in dict.fromkeys(
                [
                    "two_batch",
                    "no_update",
                    config["best_single_batch"],
                    "yield",
                    "random",
                ]
            ):
                d = summary[
                    summary.stratum.eq(stratum) & summary.policy.eq(policy)
                ].sort_values("budget_fraction")
                if d.empty:
                    continue
                ax.plot(
                    d.budget_fraction * 100,
                    d[metric],
                    marker="o",
                    ms=4,
                    lw=1.5,
                    label=labels[policy],
                    color=colors[policy],
                )
                if policy in ["two_batch", config["best_single_batch"]]:
                    ax.fill_between(
                        d.budget_fraction * 100,
                        d[metric + "_lower"],
                        d[metric + "_upper"],
                        color=colors[policy],
                        alpha=0.1,
                    )
            d = summary[summary.stratum.eq(stratum)]
            ax.set_title(
                f"{stratum.replace('_',' ').title()}\n{int(d.pools.max())} pools; {int(d.years.max())} years"
            )
            ax.set_xlabel("Design budget (%)")
            ax.spines[["top", "right"]].set_visible(False)
        axes[0].set_ylabel(ylabel)
        handles, legend = axes[0].get_legend_handles_labels()
        fig.legend(
            handles, legend, loc="lower center", ncol=3, frameon=False, fontsize=8
        )
        fig.tight_layout(rect=(0, 0.13, 1, 1))
        fig.savefig(figures / f"policy_budget_{metric}.png", dpi=220)
        fig.savefig(figures / f"policy_budget_{metric}.svg", metadata={"Date": None})
        plt.close(fig)
    annual = pd.read_csv(out / "tables/paired_by_year.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, stratum in zip(axes, primary_strata):
        for comparator, color, label in [
            (config["best_single_batch"], "#3e5caa", "vs locked one batch"),
            ("no_update", "#d17b32", "vs no-update ablation"),
        ]:
            d = annual[
                annual.stratum.eq(stratum)
                & annual.budget_fraction.eq(0.25)
                & annual.metric.eq("regret_pct")
                & annual.comparator.eq(comparator)
            ]
            ax.plot(d.year, d.difference, marker="o", color=color, label=label)
            for _, row in d.iterrows():
                ax.annotate(
                    str(int(row.pools)),
                    (row.year, row.difference),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                    color=color,
                )
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title(stratum.replace("_", " ").title())
        ax.set_xlabel("Harvest year")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(
        "Adaptive minus comparator regret (percentage points)\nNegative favors adaptive; labels show pool counts"
    )
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "paired_feedback_by_year.png", dpi=220)
    fig.savefig(figures / "paired_feedback_by_year.svg", metadata={"Date": None})
    plt.close(fig)
