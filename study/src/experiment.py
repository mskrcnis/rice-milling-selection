import gzip
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .common import ROOT, write_json
from .models import RecoveryModel, MODEL_MENU
from .replay import (
    public_view,
    QueryGate,
    Policy,
    budget,
    replay,
    score_event,
    first_batch,
    rank_scores,
    ordered_ids,
)

BUDGETS = [0.10, 0.20, 0.25, 0.40, 0.60, 1.0]
RANDOM_SEEDS = list(range(20))


def read_public(path):
    frame = pd.read_csv(
        path, dtype={"row_id": str, "pool_id": str, "candidate_id": str}
    )
    return frame


def load_key(path, ids):
    frame = pd.read_csv(path, dtype={"row_id": str})
    if frame.row_id.duplicated().any():
        raise ValueError("Repeated private-key identity")
    key = frame.set_index("row_id").hr_pct
    result = key.reindex(ids)
    if result.isna().any():
        raise ValueError("Missing measured answer")
    return result.to_dict()


def fit_menu(frame, key, names, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fitted = {}
    attempts = []
    for name in names:
        kind, parameter = MODEL_MENU[name]
        try:
            fitted[name] = RecoveryModel(kind, parameter).fit(frame, key)
            attempts.append(
                {"name": name, "status": "fitted", **fitted[name].diagnostics}
            )
        except Exception as exc:
            attempts.append({"name": name, "status": "failed", "error": repr(exc)})
            write_json(out / "model_attempts.json", attempts)
            raise
        write_json(out / "model_attempts.json", attempts)
    return fitted


def evaluate_pool(frame, key, policy, mean, sd, fraction, metadata):
    k, maximum = budget(len(frame), fraction)
    gate = QueryGate(key, frame.row_id.tolist(), maximum)
    event = replay(frame, gate, policy, mean, sd, fraction)
    event.update(metadata)
    result = score_event(frame, key, event)
    result.update(metadata)
    return event, result


def save_events(path, events):
    with gzip.GzipFile(filename=str(path), mode="wb", mtime=0) as handle:
        for event in events:
            handle.write(
                (json.dumps(event, sort_keys=True, allow_nan=False) + "\n").encode()
            )


def develop(public, key, development_years, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    events = []
    prediction_rows = []
    fits = []
    for year in development_years:
        train = public[public.year < year]
        test = public[public.year == year]
        if train.empty or test.empty:
            raise ValueError("Empty rolling development split")
        models = fit_menu(train, key, MODEL_MENU, out / f"fit_{year}")
        fits.append(
            {
                "held_out_year": int(year),
                "training_max_year": int(train.year.max()),
                "training_rows": len(train),
                "training_ids": sorted(train.row_id),
                "held_out_ids": sorted(test.row_id),
            }
        )
        print("Development year", year, "pools", test.pool_id.nunique(), flush=True)
        for pool_id, group in test.groupby("pool_id", sort=True):
            group = public_view(group)
            metadata = {
                "year": int(year),
                "site": str(group.site.iloc[0]),
                "program": str(group.program.iloc[0]),
                "stratum": "development",
            }
            pool_key = {i: key[i] for i in group.row_id}
            predictions = {name: model.predict(group) for name, model in models.items()}
            for name, (mean, sd) in predictions.items():
                for i in group.row_id:
                    prediction_rows.append(
                        {
                            "row_id": i,
                            "pool_id": pool_id,
                            "year": year,
                            "model": name,
                            "mean_hr": float(mean[i]),
                            "sd_hr": float(sd[i]),
                        }
                    )
                fractions = (
                    BUDGETS if name in ("yield", "class_history", "mixed") else [0.25]
                )
                for fraction in fractions:
                    event, result = evaluate_pool(
                        group,
                        pool_key,
                        Policy(name, "static"),
                        mean,
                        sd,
                        fraction,
                        metadata,
                    )
                    events.append(event)
                    results.append(result)
                if name not in ("yield", "class_history"):
                    for uncertainty in [0.25, 0.5, 0.75]:
                        for shrinkage in [2.0, 8.0]:
                            policy = Policy(
                                f"adapt__{name}__u{uncertainty}__s{shrinkage}",
                                "two_batch",
                                uncertainty,
                                shrinkage,
                            )
                            event, result = evaluate_pool(
                                group, pool_key, policy, mean, sd, 0.25, metadata
                            )
                            events.append(event)
                            results.append(result)
            for fraction in BUDGETS:
                for seed in RANDOM_SEEDS:
                    mean, sd = predictions["yield"]
                    event, result = evaluate_pool(
                        group,
                        pool_key,
                        Policy("random", "random", seed=seed),
                        mean,
                        sd,
                        fraction,
                        metadata,
                    )
                    events.append(event)
                    results.append(result)
    metrics = pd.DataFrame(results)
    metrics.to_csv(out / "development_per_pool.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(
        out / "development_predictions.csv", index=False
    )
    save_events(out / "development_events.jsonl.gz", events)
    write_json(out / "rolling_splits.json", fits)
    primary = (
        metrics[metrics.budget_fraction.eq(0.25)].groupby("policy").regret_pct.mean()
    )
    static_names = list(MODEL_MENU)
    best_static = sorted(static_names, key=lambda name: (primary[name], name))[0]
    predictive = [
        name for name in static_names if name not in ("yield", "class_history")
    ]
    best_predictive = sorted(predictive, key=lambda name: (primary[name], name))[0]
    adaptive_names = [
        name
        for name in primary.index
        if name.startswith("adapt__" + best_predictive + "__")
    ]
    winner = sorted(adaptive_names, key=lambda name: (primary[name], name))[0]
    _, model_name, u, s = winner.split("__")
    secondary_winner = sorted(
        [name for name in primary.index if name.startswith("adapt__")],
        key=lambda name: (primary[name], name),
    )[0]
    _, secondary_model, secondary_u, secondary_s = secondary_winner.split("__")
    config = {
        "secondary_adaptive": {
            "model": secondary_model,
            "uncertainty_fraction": float(secondary_u[1:]),
            "shrinkage": float(secondary_s[1:]),
            "status": "development_selected_secondary",
        },
        "best_single_batch": best_static,
        "adaptive_model": model_name,
        "uncertainty_fraction": float(u[1:]),
        "shrinkage": float(s[1:]),
        "selection_metric": "equal-pool mean percentage regret at25% budget on rolling-origin development",
        "model_selection": primary.to_dict(),
        "budgets": BUDGETS,
        "random_seeds": RANDOM_SEEDS,
        "test_training_information": "fixed pre-evaluation historical training set; no annual refit using test labels",
    }
    write_json(out / "selected_configuration.json", config)
    return config


def policies(config):
    items = [(Policy(name, "static"), name) for name in MODEL_MENU]
    items += [
        (
            Policy(
                "two_batch",
                "two_batch",
                config["uncertainty_fraction"],
                config["shrinkage"],
            ),
            config["adaptive_model"],
        ),
        (
            Policy(
                "no_update",
                "two_batch",
                config["uncertainty_fraction"],
                config["shrinkage"],
                False,
            ),
            config["adaptive_model"],
        ),
        (
            Policy("exploit_only", "two_batch", 0.0, config["shrinkage"]),
            config["adaptive_model"],
        ),
    ]
    if "secondary_adaptive" in config:
        secondary = config["secondary_adaptive"]
        items += [
            (
                Policy(
                    "adaptive_grid",
                    "two_batch",
                    secondary["uncertainty_fraction"],
                    secondary["shrinkage"],
                ),
                secondary["model"],
            ),
            (
                Policy(
                    "adaptive_grid_no_update",
                    "two_batch",
                    secondary["uncertainty_fraction"],
                    secondary["shrinkage"],
                    False,
                ),
                secondary["model"],
            ),
        ]
    items += [(Policy("random", "random", seed=seed), "yield") for seed in RANDOM_SEEDS]
    return items


def initial_predictions(public, models, config, out):
    out = Path(out)
    rows = []
    plans = []
    for pool_id, group in public.groupby("pool_id", sort=True):
        group = public_view(group)
        cached = {name: model.predict(group) for name, model in models.items()}
        for name, (mean, sd) in cached.items():
            for i in group.row_id:
                rows.append(
                    {
                        "row_id": i,
                        "pool_id": pool_id,
                        "model": name,
                        "mean_hr": float(mean[i]),
                        "sd_hr": float(sd[i]),
                    }
                )
        for fraction in BUDGETS:
            for policy, model_name in policies(config):
                if policy.kind == "random":
                    continue
                mean, sd = cached[model_name]
                k, maximum = budget(len(group), fraction)
                if policy.kind == "two_batch" and maximum > 1:
                    chosen, values = first_batch(
                        group,
                        mean.to_numpy(),
                        sd.to_numpy(),
                        int(np.ceil(maximum / 2)),
                        policy.uncertainty_fraction,
                    )
                else:
                    score = rank_scores(group, mean.to_numpy())
                    chosen = ordered_ids(group.row_id, score)[:maximum]
                    values = [score[i] for i in chosen]
                plans.append(
                    {
                        "pool_id": pool_id,
                        "policy": policy.name,
                        "budget_fraction": fraction,
                        "model": model_name,
                        "initial_batch_ids": chosen,
                        "initial_scores": values,
                    }
                )
    pd.DataFrame(rows).to_csv(out / "frozen_initial_predictions.csv", index=False)
    write_json(out / "frozen_initial_batch_plans.json", plans)


def evaluate(public, key, models, config, out):
    out = Path(out)
    events = []
    results = []
    diagnostics = []
    lookup = public.set_index("row_id")
    for pool_id, group_raw in public.groupby("pool_id", sort=True):
        group = public_view(group_raw)
        info = group_raw.iloc[0]
        metadata = {
            "year": int(info.year),
            "site": str(info.site),
            "program": str(info.program),
            "stratum": str(info.stratum),
            "stage": str(info.get("stage", "unknown")),
        }
        pool_key = {i: key[i] for i in group.row_id}
        predicted = {name: model.predict(group) for name, model in models.items()}
        for name, (mean, sd) in predicted.items():
            for i in group.row_id:
                diagnostics.append(
                    {
                        "row_id": i,
                        "pool_id": pool_id,
                        "model": name,
                        "observed_hr": pool_key[i],
                        "mean_hr": float(mean[i]),
                        "sd_hr": float(sd[i]),
                        **metadata,
                    }
                )
        for fraction in BUDGETS:
            for policy, model_name in policies(config):
                mean, sd = predicted[model_name]
                event, result = evaluate_pool(
                    group, pool_key, policy, mean, sd, fraction, metadata
                )
                event["model"] = model_name
                events.append(event)
                results.append(result)
    save_events(out / "evaluation_events.jsonl.gz", events)
    pd.DataFrame(results).to_csv(out / "per_pool_results.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(out / "prediction_diagnostics.csv", index=False)
    return pd.DataFrame(results)
