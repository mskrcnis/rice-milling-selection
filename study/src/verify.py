"""Independent verification of the locked milling-selection replay.

This module deliberately does not import replay, experiment, models or scoring.
It reconstructs actions and arithmetic from public fields, frozen predictions,
the selected configuration and the answers actually paid for at each batch.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT, digest, verify_freeze, write_json


class VerificationError(ValueError):
    pass


IDENTITY = ["pool_id", "policy", "seed", "budget_fraction"]
SPECIAL = {
    "two_batch",
    "no_update",
    "exploit_only",
    "adaptive_grid",
    "adaptive_grid_no_update",
}
ATOL = 1e-8
RTOL = 1e-10


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def close(actual, expected, message):
    try:
        ok = np.allclose(
            np.asarray(actual, dtype=float),
            np.asarray(expected, dtype=float),
            rtol=RTOL,
            atol=ATOL,
            equal_nan=True,
        )
    except (TypeError, ValueError):
        ok = False
    require(ok, message)


def integer(value, label):
    require(
        not isinstance(value, (bool, np.bool_)),
        label + ": boolean is not an integer count",
    )
    try:
        number = float(value)
    except (ValueError, TypeError):
        raise VerificationError(label + ": invalid integer") from None
    require(
        math.isfinite(number) and number == math.floor(number),
        label + ": invalid integer",
    )
    return int(number)


def fraction_key(value):
    value = float(value)
    require(math.isfinite(value) and 0 < value <= 1, "Invalid budget fraction")
    return round(value, 12)


def identity(event):
    return (
        str(event["pool_id"]),
        str(event["policy"]),
        integer(event["seed"], "seed"),
        fraction_key(event["budget_fraction"]),
    )


def tied_rank(values, ids=None):
    return sorted(
        values if ids is None else ids,
        key=lambda i: (
            -float(values[i]),
            hashlib.sha256(("rice-milling-tie-v1|" + str(i)).encode()).hexdigest(),
        ),
    )


def decision_size(n, fraction):
    k = math.ceil(n / 10)
    return k, min(n, max(k, math.ceil(float(fraction) * n)))


def adaptive_specifications(config):
    primary = {
        "model": config["adaptive_model"],
        "uncertainty_fraction": float(config["uncertainty_fraction"]),
        "shrinkage": float(config["shrinkage"]),
    }
    specs = {
        "two_batch": {**primary, "update": True},
        "no_update": {**primary, "update": False},
        "exploit_only": {**primary, "uncertainty_fraction": 0.0, "update": True},
    }
    if "secondary_adaptive" in config:
        secondary = config["secondary_adaptive"]
        require(
            isinstance(secondary, dict)
            and secondary.get("status") == "development_selected_secondary",
            "Secondary adaptive policy must be declared development-selected secondary",
        )
        require(
            {"model", "uncertainty_fraction", "shrinkage"} <= set(secondary),
            "Incomplete secondary adaptive configuration",
        )
        settings = {
            "model": secondary["model"],
            "uncertainty_fraction": float(secondary["uncertainty_fraction"]),
            "shrinkage": float(secondary["shrinkage"]),
        }
        specs["adaptive_grid"] = {**settings, "update": True}
        specs["adaptive_grid_no_update"] = {**settings, "update": False}
    for policy, settings in specs.items():
        require(
            math.isfinite(settings["uncertainty_fraction"])
            and 0 <= settings["uncertainty_fraction"] <= 1
            and math.isfinite(settings["shrinkage"])
            and settings["shrinkage"] > 0,
            "Invalid adaptive configuration: " + policy,
        )
    return specs


def independent_initial(group, prediction, count, uncertainty):
    ids = group.row_id.tolist()
    y = group.field_yield.to_numpy(float)
    mu = prediction.mean_hr.to_numpy(float)
    sd = prediction.sd_hr.to_numpy(float)
    output = dict(zip(ids, y * mu / 100.0))
    if uncertainty == 0:
        chosen = tied_rank(output)[:count]
        return chosen, [output[i] for i in chosen]
    n_exploit = max(1, math.floor(count * (1 - uncertainty)))
    chosen = tied_rank(output)[:n_exploit]
    selected_scores = [output[i] for i in chosen]
    k = math.ceil(len(ids) / 10)
    boundary = sorted(output.values(), reverse=True)[k - 1]
    spread = y * sd / 100.0
    z = (np.asarray([output[i] for i in ids]) - boundary) / np.maximum(spread, 1e-12)
    information = spread * np.exp(-(z**2) / 2.0) / math.sqrt(2 * math.pi)
    rel = group.yield_rel.to_numpy(float)
    classes = group.grain_class.fillna("unknown").astype(str).to_numpy()
    positions = {i: p for p, i in enumerate(ids)}
    while len(chosen) < count:
        chosen_positions = [positions[i] for i in chosen]
        distance = (rel[:, None] - rel[chosen_positions][None, :]) / 0.25
        matching = classes[:, None] == classes[chosen_positions][None, :]
        redundancy = np.sum(matching * np.exp(-(distance**2) / 2.0), axis=1)
        acquisition = information / (1.0 + redundancy)
        available = {i: acquisition[p] for p, i in enumerate(ids) if i not in chosen}
        selected = tied_rank(available)[0]
        chosen.append(selected)
        selected_scores.append(available[selected])
    return chosen, selected_scores


def independent_second(group, prediction, first_ids, answers, shrinkage, update):
    ids = group.row_id.tolist()
    mu = prediction.mean_hr.to_numpy(float)
    output = dict(zip(ids, group.field_yield.to_numpy(float) * mu / 100.0))
    position = {i: p for p, i in enumerate(ids)}
    untested = [i for i in ids if i not in first_ids]
    relative = group.yield_rel.to_numpy(float)
    classes = group.grain_class.fillna("unknown").astype(str).to_numpy()
    yields = group.field_yield.to_numpy(float)
    q = np.asarray([position[i] for i in first_ids], dtype=int)
    residuals = np.asarray([answers[i] - mu[position[i]] for i in first_ids])
    updated_scores = {}
    for candidate in untested:
        p = position[candidate]
        corrected = mu[p]
        if update:
            distance = (relative[p] - relative[q]) / 0.25
            same = classes[q] == classes[p]
            weights = np.exp(-(distance**2) / 2.0) * np.where(same, 1.0, 0.25)
            corrected += float(
                np.sum(weights * residuals) / (shrinkage + np.sum(weights))
            )
            corrected = min(100.0, max(0.0, corrected))
        updated_scores[candidate] = yields[p] * corrected / 100.0
    before = tied_rank(output, untested)
    after = tied_rank(updated_scores)
    return after, updated_scores, sum(a != b for a, b in zip(before, after))


def prepare_inputs(public, key, predictions, plans, config):
    public = public.copy()
    required = {
        "row_id",
        "pool_id",
        "candidate_id",
        "field_yield",
        "yield_rel",
        "yield_rank",
        "grain_class",
    }
    require(required <= set(public), "Missing public verification fields")
    for name in ["row_id", "pool_id", "candidate_id"]:
        require(public[name].notna().all(), "Missing public identity: " + name)
        public[name] = public[name].astype(str)
    require(not public.row_id.duplicated().any(), "Duplicate public row identity")
    require(
        not public[["pool_id", "candidate_id"]].duplicated().any(),
        "Duplicate candidate within pool",
    )
    require(len(public) > 0, "Empty evaluation cohort")
    require(
        np.isfinite(
            public[["field_yield", "yield_rel", "yield_rank"]].to_numpy(float)
        ).all(),
        "Nonfinite public yield feature",
    )
    require(
        (public.field_yield > 0).all() and (public.yield_rel > 0).all(),
        "Nonpositive public yield",
    )
    require(
        public.yield_rank.between(0, 1, inclusive="right").all(),
        "Invalid public yield percentile",
    )
    public["grain_class"] = public.grain_class.fillna("unknown").astype(str)
    if isinstance(key, pd.DataFrame):
        require({"row_id", "hr_pct"} <= set(key), "Missing private-key columns")
        require(
            not key.row_id.astype(str).duplicated().any(),
            "Duplicate private-key identity",
        )
        key = dict(zip(key.row_id.astype(str), key.hr_pct))
    else:
        key = {str(i): v for i, v in dict(key).items()}
    require(
        set(key) == set(public.row_id),
        "Private key and public cohort identities differ",
    )
    require(
        all(np.isfinite(float(v)) and 0 <= float(v) <= 100 for v in key.values()),
        "Invalid or missing measured recovery",
    )
    key = {i: float(v) for i, v in key.items()}
    predictions = predictions.copy()
    require(
        {"row_id", "pool_id", "model", "mean_hr", "sd_hr"} <= set(predictions),
        "Missing frozen-prediction columns",
    )
    for name in ["row_id", "pool_id", "model"]:
        predictions[name] = predictions[name].astype(str)
    require(
        not predictions[["row_id", "model"]].duplicated().any(),
        "Duplicate frozen prediction",
    )
    require(
        np.isfinite(predictions[["mean_hr", "sd_hr"]].to_numpy(float)).all(),
        "Nonfinite frozen prediction",
    )
    require(
        predictions.mean_hr.between(0, 100).all() and (predictions.sd_hr >= 0).all(),
        "Invalid frozen recovery prediction",
    )
    models = sorted(predictions.model.unique())
    require(models and "yield" in models, "Missing yield baseline predictions")
    require(
        not set(models) & (SPECIAL | {"random"}), "Reserved policy used as model name"
    )
    specs = adaptive_specifications(config)
    require(
        all(settings["model"] in models for settings in specs.values())
        and config["best_single_batch"] in models,
        "Selected model not frozen",
    )
    budgets = [fraction_key(f) for f in config["budgets"]]
    seeds = [integer(s, "random seed") for s in config["random_seeds"]]
    require(
        len(set(budgets)) == len(budgets) and 1.0 in budgets,
        "Duplicate budgets or missing full-budget check",
    )
    require(
        seeds and len(set(seeds)) == len(seeds) and min(seeds) >= 0,
        "Invalid random seed list",
    )
    groups = {
        str(p): g.sort_values("row_id").reset_index(drop=True)
        for p, g in public.groupby("pool_id")
    }
    cached = {}
    expected_prediction_rows = set()
    for pool, group in groups.items():
        for name in ["year", "site", "program", "stratum", "stage"]:
            if name in group:
                require(
                    group[name].nunique(dropna=False) == 1,
                    "Mixed metadata in pool: " + name,
                )
        for model in models:
            part = predictions[
                (predictions.pool_id == pool) & (predictions.model == model)
            ]
            require(
                set(part.row_id) == set(group.row_id),
                "Incomplete frozen prediction grid",
            )
            cached[pool, model] = (
                part.set_index("row_id").loc[group.row_id].reset_index()
            )
            expected_prediction_rows.update((i, model) for i in group.row_id)
    require(
        set(zip(predictions.row_id, predictions.model)) == expected_prediction_rows,
        "Extra frozen prediction identities",
    )
    plan_lookup = {}
    for plan in plans:
        key_plan = (
            str(plan["pool_id"]),
            str(plan["policy"]),
            fraction_key(plan["budget_fraction"]),
        )
        require(key_plan not in plan_lookup, "Duplicate frozen batch plan")
        plan_lookup[key_plan] = plan
    expected_plans = {
        (pool, policy, fraction)
        for pool in groups
        for policy in models + sorted(specs)
        for fraction in budgets
    }
    require(
        set(plan_lookup) == expected_plans, "Incomplete or extra frozen batch-plan grid"
    )
    initial = {}
    for (pool, policy, fraction), plan in plan_lookup.items():
        group = groups[pool]
        _, b = decision_size(len(group), fraction)
        model = specs[policy]["model"] if policy in specs else policy
        require(plan["model"] == model, "Frozen plan references wrong predictor")
        prediction = cached[pool, model]
        if policy in specs and b > 1:
            u = specs[policy]["uncertainty_fraction"]
            ids, scores = independent_initial(group, prediction, math.ceil(b / 2), u)
        else:
            output = dict(
                zip(
                    group.row_id,
                    group.field_yield.to_numpy(float)
                    * prediction.mean_hr.to_numpy(float)
                    / 100.0,
                )
            )
            ids = tied_rank(output)[:b]
            scores = [output[i] for i in ids]
        require(
            plan["initial_batch_ids"] == ids,
            "Frozen initial plan is inconsistent with prediction/configuration",
        )
        require(len(plan["initial_scores"]) == len(scores), "Wrong frozen score count")
        close(plan["initial_scores"], scores, "Frozen initial plan scores differ")
        initial[pool, policy, fraction] = (ids, scores)
    return groups, key, cached, models, budgets, seeds, initial


def verify_records(public, key, predictions, plans, events, results, config):
    groups, key, cached, models, budgets, seeds, initial = prepare_inputs(
        public, key, predictions, plans, config
    )
    specs = adaptive_specifications(config)
    paired_policies = [("two_batch", "no_update")]
    if "adaptive_grid" in specs:
        paired_policies.append(("adaptive_grid", "adaptive_grid_no_update"))
    paired_names = {name for pair in paired_policies for name in pair}
    expected = {
        (p, policy, 0, f)
        for p in groups
        for policy in models + sorted(specs)
        for f in budgets
    }
    expected |= {
        (p, "random", seed, f) for p in groups for seed in seeds for f in budgets
    }
    seen = set()
    rows = []
    first_batches = {}
    n_queries = n_full = 0
    for event in events:
        token = identity(event)
        require(
            token in expected and token not in seen,
            "Unexpected or duplicate evaluation event: " + repr(token),
        )
        seen.add(token)
        pool, policy, seed, fraction = token
        group = groups[pool]
        ids = group.row_id.tolist()
        n = len(ids)
        k, b = decision_size(n, fraction)
        require(
            integer(event["n"], "n") == n
            and integer(event["k"], "k") == k
            and integer(event["budget"], "budget") == b,
            "Incorrect event candidate, shortlist or budget count",
        )
        model = (
            specs[policy]["model"]
            if policy in specs
            else "yield" if policy == "random" else policy
        )
        require(event["model"] == model, "Event uses a different predictor than frozen")
        metadata = {}
        for name in ["year", "site", "program", "stratum", "stage"]:
            if name in group:
                value = (
                    integer(group.iloc[0][name], name)
                    if name == "year"
                    else str(group.iloc[0][name])
                )
                require(
                    event.get(name) == value,
                    "Event metadata does not match public cohort: " + name,
                )
                metadata[name] = value
        if policy == "random":
            pool_number = int(hashlib.sha256(pool.encode()).hexdigest()[:8], 16)
            generator = np.random.Generator(
                np.random.PCG64(np.random.SeedSequence([seed, pool_number]))
            )
            random_score = dict(zip(ids, generator.random(n)))
            chosen = tied_rank(random_score)[:b]
            scores = [random_score[i] for i in chosen]
        else:
            chosen, scores = initial[pool, policy, fraction]
        batches = event["events"]
        need_two = policy in specs and b > 1
        require(len(batches) == (2 if need_two else 1), "Wrong number of query batches")
        history = []
        paid = {}
        changed = 0
        for index, batch in enumerate(batches):
            require(
                integer(batch["batch"], "batch") == index + 1, "Invalid batch numbering"
            )
            require(
                batch["revealed_ids_before"] == history,
                "Invalid revealed-answer history",
            )
            if index == 1:
                rank, second_scores, changed = independent_second(
                    group,
                    cached[pool, model],
                    history,
                    paid,
                    specs[policy]["shrinkage"],
                    specs[policy]["update"],
                )
                chosen = rank[: b - len(history)]
                scores = [second_scores[i] for i in chosen]
            require(
                batch["selected_ids"] == chosen,
                "Query actions differ from independently reconstructed policy",
            )
            require(
                len(chosen) == len(set(chosen)) and not set(chosen) & set(history),
                "Repeated query identity",
            )
            require(set(chosen) <= set(ids), "Query outside candidate pool")
            require(
                len(batch["selection_scores"]) == len(chosen),
                "Wrong logged acquisition score count",
            )
            close(batch["selection_scores"], scores, "Logged acquisition scores differ")
            require(
                set(batch["revealed_hr_pct"]) == set(chosen),
                "Returned answer identities differ from queried IDs",
            )
            for i in chosen:
                close(
                    batch["revealed_hr_pct"][i],
                    key[i],
                    "Queried answer differs from measured key",
                )
                paid[i] = key[i]
            history += chosen
            require(
                integer(batch["remaining_budget"], "remaining budget")
                == b - len(history),
                "Incorrect remaining budget",
            )
        require(
            history == event["tested_ids"] and len(history) == b,
            "Tested entries differ from query log",
        )
        require(
            integer(event["untested_rank_positions_changed"], "rank changes")
            == changed,
            "Incorrect untested rank-change count",
        )
        if policy in paired_names:
            first_batches[pool, fraction, policy] = batches[0]
        values = {i: float(y) * key[i] / 100.0 for i, y in zip(ids, group.field_yield)}
        oracle = tied_rank(values)[:k]
        shortlist = tied_rank(values, history)[:k]
        require(
            event["shortlist"] == shortlist,
            "Final shortlist is not the measured best tested k under fixed tie rule",
        )
        oracle_mean = sum(values[i] for i in oracle) / k
        selected_mean = sum(values[i] for i in shortlist) / k
        regret = oracle_mean - selected_mean
        require(regret >= -ATOL, "Computed negative regret")
        regret = max(0.0, regret)
        recall = len(set(oracle) & set(shortlist)) / k
        top_ten_count = (
            len(set(tied_rank(values)[:10]) & set(history))
            if n >= 10 and b >= 10
            else np.nan
        )
        row = dict(
            pool_id=pool,
            policy=policy,
            seed=seed,
            budget_fraction=fraction,
            n=n,
            k=k,
            tests=b,
            actual_testing_fraction=b / n,
            oracle_mean=oracle_mean,
            selected_mean=selected_mean,
            regret=regret,
            regret_pct=100 * regret / oracle_mean if oracle_mean > 0 else np.nan,
            top_k_recall=recall,
            best_entry_found=int(oracle[0] in shortlist),
            top_ten_count=top_ten_count,
            top_ten_recall=top_ten_count / 10,
            rank_positions_changed=changed,
            **metadata
        )
        rows.append(row)
        n_queries += b
        if fraction == 1.0:
            require(
                regret == 0 and recall == 1 and row["best_entry_found"] == 1,
                "Full-budget oracle identity failed",
            )
            n_full += 1
    require(seen == expected, "Evaluation event grid is incomplete")
    for pool in groups:
        for fraction in budgets:
            for adaptive, no_update in paired_policies:
                a = first_batches[pool, fraction, adaptive]
                d = first_batches[pool, fraction, no_update]
                contrast = adaptive + " versus " + no_update
                require(
                    a["selected_ids"] == d["selected_ids"],
                    "Adaptive and no-update first batches differ: " + contrast,
                )
                close(
                    a["selection_scores"],
                    d["selection_scores"],
                    "Adaptive/no-update first-batch scores differ: " + contrast,
                )
                require(
                    a["revealed_hr_pct"] == d["revealed_hr_pct"],
                    "Adaptive/no-update paid answers differ: " + contrast,
                )
    regenerated = pd.DataFrame(rows).sort_values(IDENTITY).reset_index(drop=True)
    observed = results.copy()
    require(
        set(regenerated) <= set(observed), "Saved per-pool table lacks required columns"
    )
    require(
        len(observed) == len(regenerated), "Saved per-pool result row count differs"
    )
    observed_tokens = [identity(r) for r in observed.to_dict("records")]
    require(
        len(set(observed_tokens)) == len(observed_tokens)
        and set(observed_tokens) == expected,
        "Saved per-pool table has duplicate, missing or extra identities",
    )
    lookup = dict(zip(observed_tokens, observed.to_dict("records")))
    for row in regenerated.to_dict("records"):
        other = lookup[identity(row)]
        for column, value in row.items():
            if column in {"pool_id", "policy", "site", "program", "stratum", "stage"}:
                require(
                    str(other[column]) == str(value),
                    "Saved result metadata differs: " + column,
                )
            else:
                close(other[column], value, "Saved result metric differs: " + column)
    summary = dict(
        status="passed",
        event_count=len(rows),
        pool_count=len(groups),
        candidate_count=sum(len(g) for g in groups.values()),
        query_revelations_checked=n_queries,
        full_budget_events_checked=n_full,
        paired_first_batches_checked=len(groups) * len(budgets) * len(paired_policies),
        models_checked=models,
        paired_first_batches_by_comparison={
            a + "__vs__" + d: len(groups) * len(budgets) for a, d in paired_policies
        },
        secondary_adaptive_checked="adaptive_grid" in specs,
        budget_fractions=budgets,
        random_seed_count=len(seeds),
        numeric_absolute_tolerance=ATOL,
        numeric_relative_tolerance=RTOL,
        additional_result_columns_not_scored=sorted(set(observed) - set(regenerated)),
        independence="No imports from replay, experiment, models or scoring; actions and metrics independently reconstructed.",
        limitation="Verifies replay consistency conditional on frozen inputs; does not certify source extraction or prospective assay validity.",
    )
    return regenerated, summary


def read_events(path):
    with gzip.open(path, "rt") as handle:
        for line_number, line in enumerate(handle, 1):
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise VerificationError(
                    "Invalid event JSON line " + str(line_number)
                ) from exc


def check_prediction_diagnostics(path, predictions, key):
    diagnostics = pd.read_csv(path, dtype={"row_id": str, "pool_id": str, "model": str})
    require(
        not diagnostics[["row_id", "model"]].duplicated().any(),
        "Duplicate prediction diagnostics",
    )
    frozen = predictions.set_index(["row_id", "model"]).sort_index()
    recorded = diagnostics.set_index(["row_id", "model"]).sort_index()
    require(
        recorded.index.equals(frozen.index),
        "Prediction diagnostics do not match frozen grid",
    )
    require(
        recorded.pool_id.tolist() == frozen.pool_id.tolist(),
        "Prediction diagnostic pools changed",
    )
    close(
        recorded.mean_hr,
        frozen.mean_hr,
        "Evaluation means differ from frozen predictions",
    )
    close(
        recorded.sd_hr,
        frozen.sd_hr,
        "Evaluation uncertainty differs from frozen predictions",
    )
    close(
        recorded.observed_hr,
        [key[i] for i, _ in recorded.index],
        "Prediction diagnostics use a different answer key",
    )
    return len(recorded)


def verify_run(
    out,
    public_path=None,
    key_path=None,
    config_path=None,
    freeze_path=None,
    analysis_lock=None,
    write_outputs=True,
):
    out = Path(out).resolve()
    public_path = Path(
        public_path or ROOT / "derived/cohort/evaluation_public.csv"
    ).resolve()
    key_path = Path(key_path or ROOT / "private/evaluation_key.csv").resolve()
    config_path = Path(config_path or out / "selected_configuration.json").resolve()
    freeze_path = Path(freeze_path or out / "evaluation_freeze.json").resolve()
    predictions_path = out / "frozen_initial_predictions.csv"
    plans_path = out / "frozen_initial_batch_plans.json"
    events_path = out / "evaluation_events.jsonl.gz"
    results_path = out / "per_pool_results.csv"
    require(
        freeze_path.is_file(), "Missing evaluation freeze; answer key has not been read"
    )
    frozen_count = verify_freeze(freeze_path)
    run_manifest = json.loads(freeze_path.read_text())
    bound = {str(Path(p).resolve()) for p in run_manifest["files"]}
    for path in [config_path, predictions_path, plans_path]:
        require(str(path) in bound, "Run freeze does not bind " + str(path))
    lock_path = Path(analysis_lock or ROOT / "protocol/analysis_lock.json").resolve()
    lock_count = 0
    if lock_path.exists():
        lock_count = verify_freeze(lock_path)
        bound |= {
            str(Path(p).resolve()) for p in json.loads(lock_path.read_text())["files"]
        }
    for path in [public_path, key_path]:
        require(str(path) in bound, "No verified freeze binds " + str(path))
    public = pd.read_csv(
        public_path, dtype={"row_id": str, "pool_id": str, "candidate_id": str}
    )
    key = pd.read_csv(key_path, dtype={"row_id": str})
    predictions = pd.read_csv(
        predictions_path, dtype={"row_id": str, "pool_id": str, "model": str}
    )
    config = json.loads(config_path.read_text())
    plans = json.loads(plans_path.read_text())
    results = pd.read_csv(results_path, dtype={"pool_id": str, "policy": str})
    regenerated, summary = verify_records(
        public, key, predictions, plans, read_events(events_path), results, config
    )
    diagnostics_path = out / "prediction_diagnostics.csv"
    if diagnostics_path.exists():
        summary["prediction_diagnostics_checked"] = check_prediction_diagnostics(
            diagnostics_path, predictions, dict(zip(key.row_id, key.hr_pct))
        )
    checked = [
        public_path,
        key_path,
        config_path,
        predictions_path,
        plans_path,
        events_path,
        results_path,
        freeze_path,
    ]
    if lock_path.exists():
        checked.append(lock_path)
    summary["input_sha256"] = {str(p): digest(p) for p in checked}
    summary["frozen_run_files_checked"] = frozen_count
    summary["frozen_analysis_files_checked"] = lock_count
    if write_outputs:
        destination = out / "verification"
        destination.mkdir(exist_ok=True)
        regenerated.to_csv(
            destination / "regenerated_per_pool_results.csv", index=False
        )
        summary["regenerated_results_sha256"] = digest(
            destination / "regenerated_per_pool_results.csv"
        )
        write_json(destination / "verification_summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--public", type=Path)
    parser.add_argument("--key", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--analysis-lock", type=Path)
    args = parser.parse_args()
    summary = verify_run(
        args.out, args.public, args.key, args.config, args.freeze, args.analysis_lock
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
