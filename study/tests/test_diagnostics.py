import copy
import gzip
import json

import numpy as np
import pandas as pd
import pytest

from src.common import freeze
from src.replay import Policy, QueryGate, replay, score_event
import src.diagnostics as diagnostics


def fixture_data():
    public = pd.DataFrame(
        {
            "row_id": [f"r{i:02}" for i in range(20)],
            "candidate_id": [f"c{i:02}" for i in range(20)],
            "pool_id": "p",
            "program": "uruguay",
            "site": "s",
            "year": 2020,
            "stratum": "temporal",
            "stage": "E2",
            "grain_class": "unharmonized",
            "source_class": ["source:A"] * 10 + ["source:B"] * 10,
            "field_yield": np.arange(1000.0, 1020.0),
            "observed_field": True,
            "paired": True,
        }
    )
    key = dict(zip(public.row_id, np.arange(50.0, 70.0)))
    events = []
    results = []
    for seed in [0, 1]:
        e = replay(
            public,
            QueryGate(key, list(key), 5),
            Policy("random", "random", seed=seed),
            np.repeat(60.0, 20),
            np.repeat(1.0, 20),
            0.25,
        )
        meta = {"program": "uruguay", "site": "s", "year": 2020, "stratum": "temporal"}
        e.update(meta)
        r = score_event(public, key, e)
        r.update(meta)
        events.append(e)
        results.append(r)
    predictions = []
    for model in ["yield", "ridge_1"]:
        for _, r in public.iterrows():
            predictions.append(
                {
                    "row_id": r.row_id,
                    "pool_id": "p",
                    "model": model,
                    "observed_hr": key[r.row_id],
                    "mean_hr": 100.0 if model == "yield" else key[r.row_id] + 1,
                    "sd_hr": 2,
                    "program": "uruguay",
                    "site": "s",
                    "year": 2020,
                    "stratum": "temporal",
                }
            )
    history = public[["row_id", "candidate_id", "program", "year"]].copy()
    history["returning_line"] = [False] * 10 + [True] * 10
    return public, pd.DataFrame(predictions), history, pd.DataFrame(results), events


def test_mixed_classes_do_not_create_decision_pools_or_seed_replication():
    public, pred, hist, results, events = fixture_data()
    pool, strata = diagnostics.composition_diagnostics(results, public)
    assert (
        len(pool) == 1 and pool.iloc[0].n == 20 and pool.iloc[0].seed_repetitions == 2
    )
    assert pool.iloc[0].class_composition_kind == "mixed_class"
    assert pool.iloc[0].source_class_composition == "source:A|source:B"
    assert strata.pools.eq(1).all()
    assert len(strata) == 4


def test_missingness_denominator_is_public_field_candidate_universe():
    public, *_ = fixture_data()
    extra = public.iloc[:2].copy()
    extra.row_id = ["x", "y"]
    extra.paired = False
    extra.loc[extra.row_id.eq("y"), "observed_field"] = False
    row = diagnostics.pool_descriptors(pd.concat([public, extra])).iloc[0]
    assert row.source_candidate_count == 22 and row.field_candidate_count == 21
    assert row.paired_candidate_count == 20
    assert row.paired_fraction_of_field_candidates == pytest.approx(20 / 21)


def test_prediction_metrics_exclude_yield_surrogate_and_preserve_scale():
    _, pred, *_ = fixture_data()
    metrics, calibration = diagnostics.prediction_metrics(pred)
    assert set(metrics.model) == {"ridge_1"}
    assert metrics.iloc[0].rmse_pp == pytest.approx(1.0)
    assert metrics.iloc[0].mae_pp == pytest.approx(1.0)
    assert metrics.iloc[0].bias_predicted_minus_observed_pp == pytest.approx(1.0)
    assert len(calibration) == 5 and calibration.candidate_rows.sum() == 20
    assert (
        calibration.loc[calibration.candidate_rows.gt(0), "observed_minus_predicted_pp"]
        .eq(-1)
        .all()
    )


def test_calibration_100_percent_stays_in_last_bin():
    _, pred, *_ = fixture_data()
    pred.loc[pred.model.eq("ridge_1"), "mean_hr"] = 100.0
    _, cal = diagnostics.prediction_metrics(pred)
    assert cal.loc[cal.prediction_bin.eq("[80,100]"), "candidate_rows"].iloc[0] == 20


def test_history_counts_use_whole_pool_oracle_and_preserve_events():
    public, pred, hist, results, events = fixture_data()
    before = copy.deepcopy(events)
    raw, pool, summary = diagnostics.line_history_diagnostics(
        iter(events), pred, public, hist, results
    )
    assert events == before
    assert len(raw) == 4 and len(pool) == 2 and len(summary) == 2
    for _, g in raw.groupby("seed"):
        assert g.candidate_count.sum() == 20 and g.tested_count.sum() == 5
        assert g.shortlist_count.sum() == 2 and g.oracle_topk_count.sum() == 2
    assert pool.seed_repetitions.eq(2).all()
    assert raw.loc[raw.history_group.eq("new"), "oracle_recovery_rate"].isna().all()
    assert (
        summary.loc[summary.history_group.eq("new"), "pools_with_oracle_members"].iloc[
            0
        ]
        == 0
    )


def test_history_missing_identity_is_not_silently_treated_as_new():
    public, pred, hist, results, events = fixture_data()
    with pytest.raises(ValueError, match="prior-line status"):
        diagnostics.line_history_diagnostics(
            events, pred, public, hist.iloc[1:], results
        )


def test_models_must_agree_on_observed_answer():
    public, pred, hist, results, events = fixture_data()
    pred.loc[0, "observed_hr"] += 1
    with pytest.raises(ValueError, match="disagree"):
        diagnostics.line_history_diagnostics(events, pred, public, hist, results)


def test_event_result_discrepancy_is_rejected():
    public, pred, hist, results, events = fixture_data()
    results.loc[0, "regret_pct"] += 1
    with pytest.raises(ValueError, match="event recomputation"):
        diagnostics.line_history_diagnostics(events, pred, public, hist, results)


def test_duplicate_results_are_rejected():
    public, pred, hist, results, events = fixture_data()
    with pytest.raises(ValueError, match="Duplicate"):
        diagnostics.composition_diagnostics(
            pd.concat([results, results.iloc[:1]]), public
        )


def write_synthetic_files(tmp_path, monkeypatch):
    public, pred, hist, results, events = fixture_data()
    root = tmp_path / "root"
    cohort = root / "derived/cohort"
    cohort.mkdir(parents=True)
    out = root / "run"
    out.mkdir()
    public.to_csv(cohort / "all_public_metadata.csv", index=False)
    hist.to_csv(cohort / "line_history_status.csv", index=False)
    pred.to_csv(out / "prediction_diagnostics.csv", index=False)
    results.to_csv(out / "per_pool_results.csv", index=False)
    with gzip.open(out / "evaluation_events.jsonl.gz", "wt") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    freeze(
        [cohort / "all_public_metadata.csv", cohort / "line_history_status.csv"],
        out / "evaluation_freeze.json",
        {"synthetic": True},
    )
    monkeypatch.setattr(diagnostics, "ROOT", root)
    return root, out


def test_freeze_verified_before_any_dataframe_read(tmp_path, monkeypatch):
    root, out = write_synthetic_files(tmp_path, monkeypatch)
    (root / "derived/cohort/line_history_status.csv").write_text("changed")

    def forbidden(*args, **kwargs):
        raise AssertionError("Unexpected dataframe read before freeze verification")

    monkeypatch.setattr(pd, "read_csv", forbidden)
    with pytest.raises(ValueError, match="Frozen input changed"):
        diagnostics.run_diagnostics(out)


def test_synthetic_pipeline_repeats_byte_identically(tmp_path, monkeypatch):
    _, out = write_synthetic_files(tmp_path, monkeypatch)
    first = diagnostics.run_diagnostics(out)
    second = diagnostics.run_diagnostics(out)
    assert first == second
    assert len(first["output_hashes"]) == 7
    assert first["outputs_are_descriptive"] and not first["new_decision_subpools"]
    assert not first["policy_actions_changed"]
