"""Synthetic integration and corruption tests; no real cohort/key is read.

Production functions generate fixtures only. Verification arithmetic is supplied
exclusively by src.verify, with separate hand-calculated numerical checks below.
"""

import copy
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common import freeze
from src.replay import Policy, QueryGate, public_view, replay, score_event
from src.verify import VerificationError, verify_records, verify_run


def synthetic_bundle(n=12, pools=2, zero=False, secondary=False):
    config = {'adaptive_model': 'ridge_10', 'best_single_batch': 'yield',
              'uncertainty_fraction': .5, 'shrinkage': 2., 'budgets': [.1, .25, 1.],
              'random_seeds': [0, 5]}
    if secondary:
        config['secondary_adaptive'] = {'model': 'trees_20', 'uncertainty_fraction': .25, 'shrinkage': 8.,
                                        'status': 'development_selected_secondary'}
    public_parts = []
    predictions = []
    plans = []
    events = []
    results = []
    key = {}
    for p in range(pools):
        raw = pd.DataFrame({'row_id': [f'p{p}r{i:03d}' for i in range(n)], 'pool_id': f'p{p}',
                            'candidate_id': [f'c{i:03d}' for i in range(n)], 'site': f's{p}',
                            'program': 'synthetic', 'stratum': 'test', 'stage': 'fixture', 'year': 2030 + p,
                            'field_yield': np.linspace(4000, 9000, n),
                            'grain_class': ['long' if i % 2 else 'medium' for i in range(n)]})
        group = public_view(raw)
        for name in ['stratum', 'stage']:
            group[name] = raw[name].iloc[0]
        public_parts.append(group)
        ids = group.row_id.tolist()
        observed = [50., 25.] if n == 2 else 58 + 20 * np.sin(np.arange(n) * 1.7 + p)
        if zero:
            observed = np.zeros(n)
        key.update(dict(zip(ids, observed)))
        means = {'yield': pd.Series(100., index=ids),
                 'ridge_10': pd.Series(59 + 3 * np.cos(np.arange(n) * .6), index=ids)}
        sds = {'yield': pd.Series(0., index=ids), 'ridge_10': pd.Series(7., index=ids)}
        if secondary:
            means['trees_20'] = pd.Series(57 + 8 * np.sin(np.arange(n) * .5), index=ids)
            sds['trees_20'] = pd.Series(9., index=ids)
        for model in means:
            predictions.extend({'row_id': i, 'pool_id': f'p{p}', 'model': model,
                                'mean_hr': float(means[model][i]), 'sd_hr': float(sds[model][i])} for i in ids)
        policies = [(Policy(name, 'static'), name) for name in means]
        policies += [(Policy('two_batch', 'two_batch', .5, 2.), 'ridge_10'),
                     (Policy('no_update', 'two_batch', .5, 2., False), 'ridge_10'),
                     (Policy('exploit_only', 'two_batch', 0., 2.), 'ridge_10')]
        if secondary:
            policies += [(Policy('adaptive_grid', 'two_batch', .25, 8.), 'trees_20'),
                         (Policy('adaptive_grid_no_update', 'two_batch', .25, 8., False), 'trees_20')]
        policies += [(Policy('random', 'random', seed=s), 'yield') for s in config['random_seeds']]
        metadata = {name: raw.iloc[0][name] for name in ['year', 'site', 'program', 'stratum', 'stage']}
        metadata['year'] = int(metadata['year'])
        for fraction in config['budgets']:
            b = min(n, max(int(np.ceil(.1 * n)), int(np.ceil(fraction * n))))
            for policy, model in policies:
                event = replay(group, QueryGate(key, ids, b), policy, means[model], sds[model], fraction)
                event.update(metadata)
                result = score_event(group, key, event)
                result.update(metadata)
                event['model'] = model
                events.append(event)
                results.append(result)
                if policy.name != 'random':
                    plans.append({'pool_id': f'p{p}', 'policy': policy.name, 'budget_fraction': fraction,
                                  'model': model, 'initial_batch_ids': list(event['events'][0]['selected_ids']),
                                  'initial_scores': list(event['events'][0]['selection_scores'])})
    return dict(public=pd.concat(public_parts, ignore_index=True), key=key,
                predictions=pd.DataFrame(predictions), plans=plans, events=events,
                results=pd.DataFrame(results), config=config)


def run(bundle):
    return verify_records(**bundle)


def test_independent_reconstruction_matches_synthetic_production():
    regenerated, summary = run(synthetic_bundle())
    assert len(regenerated) == 42
    assert summary['paired_first_batches_checked'] == 6
    assert summary['full_budget_events_checked'] == 14
    assert summary['candidate_count'] == 24


def test_secondary_grid_complete_reconstruction_and_both_feedback_pairs():
    table, summary = run(synthetic_bundle(secondary=True))
    assert len(table) == 60
    assert summary['paired_first_batches_checked'] == 12
    assert summary['full_budget_events_checked'] == 20
    assert summary['secondary_adaptive_checked'] is True
    assert summary['paired_first_batches_by_comparison'] == {
        'two_batch__vs__no_update': 6, 'adaptive_grid__vs__adaptive_grid_no_update': 6}


@pytest.mark.parametrize('field,value', [('model', 'ridge_10'), ('uncertainty_fraction', .75),
                                       ('shrinkage', 2.), ('status', 'test_selected')])
def test_secondary_settings_cannot_change_after_plans(field, value):
    data = synthetic_bundle(secondary=True)
    data['config']['secondary_adaptive'][field] = value
    with pytest.raises(VerificationError):
        run(data)


def test_secondary_no_update_cannot_reuse_primary_first_batch():
    data = synthetic_bundle(secondary=True)
    secondary = next(e for e in data['events'] if e['policy'] == 'adaptive_grid_no_update' and e['budget_fraction'] == .25)
    primary = next(e for e in data['events'] if e['policy'] == 'no_update' and e['budget_fraction'] == .25)
    secondary['events'][0] = copy.deepcopy(primary['events'][0])
    with pytest.raises(VerificationError):
        run(data)


def test_secondary_missing_counterpart_rejected():
    data = synthetic_bundle(secondary=True)
    data['events'] = [e for e in data['events'] if e['policy'] != 'adaptive_grid_no_update']
    with pytest.raises(VerificationError, match='grid is incomplete'):
        run(data)


def test_secondary_policies_require_explicit_configuration():
    data = synthetic_bundle(secondary=True)
    del data['config']['secondary_adaptive']
    with pytest.raises(VerificationError, match='batch-plan grid'):
        run(data)


def test_hand_calculated_two_entry_output_and_regret():
    table, _ = run(synthetic_bundle(n=2, pools=1))
    assert set(table.oracle_mean) == {2250.}
    assert set(table.selected_mean) <= {2000., 2250.}
    assert set(table.regret) <= {0., 250.}
    for _, row in table.iterrows():
        assert row.regret_pct == pytest.approx(0. if row.selected_mean == 2250. else 100 / 9)
    assert table[table.budget_fraction == 1].top_k_recall.eq(1).all()


def test_zero_oracle_retains_undefined_relative_regret():
    table, _ = run(synthetic_bundle(zero=True))
    assert table.regret_pct.isna().all()
    assert table.regret.eq(0).all()


def test_order_independent_table_and_prediction_alignment():
    data = synthetic_bundle()
    for field in ['public', 'predictions', 'results']:
        data[field] = data[field].sample(frac=1, random_state=17)
    data['events'] = list(reversed(data['events']))
    data['plans'] = list(reversed(data['plans']))
    run(data)


@pytest.mark.parametrize('metric', ['oracle_mean', 'selected_mean', 'regret', 'regret_pct', 'top_k_recall',
                                  'best_entry_found', 'tests', 'n', 'k', 'actual_testing_fraction',
                                  'rank_positions_changed', 'top_ten_count', 'top_ten_recall'])
def test_corrupted_saved_metrics_are_rejected(metric):
    data = synthetic_bundle()
    index = data['results'][metric].first_valid_index()
    data['results'].loc[index, metric] += 1
    with pytest.raises(VerificationError):
        run(data)


@pytest.mark.parametrize('field,value', [('n', 13), ('k', 5), ('budget', 4), ('model', 'yield'),
                                       ('untested_rank_positions_changed', 999), ('year', 3000)])
def test_corrupted_event_counters_and_metadata_rejected(field, value):
    data = synthetic_bundle()
    event = next(e for e in data['events'] if e['policy'] == 'two_batch' and e['budget_fraction'] == .25)
    event[field] = value
    with pytest.raises(VerificationError):
        run(data)


@pytest.mark.parametrize('corruption', ['answer', 'history', 'remaining', 'score', 'batch_number', 'repeat', 'untested_shortlist', 'wrong_shortlist'])
def test_corrupt_query_history_and_decisions_rejected(corruption):
    data = synthetic_bundle()
    event = next(e for e in data['events'] if e['policy'] == 'two_batch' and e['budget_fraction'] == .25)
    first = event['events'][0]
    second = event['events'][1]
    if corruption == 'answer':
        first['revealed_hr_pct'][first['selected_ids'][0]] += .01
    elif corruption == 'history':
        second['revealed_ids_before'] = []
    elif corruption == 'remaining':
        first['remaining_budget'] += 1
    elif corruption == 'score':
        second['selection_scores'][0] += .1
    elif corruption == 'batch_number':
        second['batch'] = 4
    elif corruption == 'repeat':
        second['selected_ids'] = [first['selected_ids'][0]]
    elif corruption == 'untested_shortlist':
        untested = sorted(set(data['public'][data['public'].pool_id == event['pool_id']].row_id) - set(event['tested_ids']))
        event['shortlist'][0] = untested[0]
    else:
        event['shortlist'] = list(reversed(event['shortlist']))
    with pytest.raises(VerificationError):
        run(data)


@pytest.mark.parametrize('field', ['plans', 'predictions', 'events', 'results'])
def test_incomplete_grid_rejected(field):
    data = synthetic_bundle()
    data[field] = data[field][:-1]
    with pytest.raises(VerificationError):
        run(data)


def test_initial_frozen_plan_cannot_be_rewritten_consistently_with_event():
    data = synthetic_bundle()
    plan = next(p for p in data['plans'] if p['policy'] == 'two_batch' and p['budget_fraction'] == .25)
    event = next(e for e in data['events'] if e['policy'] == 'two_batch' and e['budget_fraction'] == .25)
    plan['initial_batch_ids'] = list(reversed(plan['initial_batch_ids']))
    event['events'][0]['selected_ids'] = list(plan['initial_batch_ids'])
    with pytest.raises(VerificationError, match='Frozen initial plan'):
        run(data)


def test_frozen_prediction_change_cannot_pass_plan_check():
    data = synthetic_bundle()
    data['predictions'].loc[data['predictions'].model == 'ridge_10', 'mean_hr'] += 10
    with pytest.raises(VerificationError, match='Frozen initial plan'):
        run(data)


def test_duplicate_key_and_missing_answers_rejected():
    data = synthetic_bundle()
    records = pd.DataFrame({'row_id': list(data['key']), 'hr_pct': list(data['key'].values())})
    data['key'] = pd.concat([records, records.iloc[:1]])
    with pytest.raises(VerificationError, match='Duplicate private'):
        run(data)
    data['key'] = records.copy()
    data['key'].loc[0, 'hr_pct'] = np.nan
    with pytest.raises(VerificationError, match='Invalid or missing'):
        run(data)


def write_synthetic_run(tmp_path):
    data = synthetic_bundle(pools=1)
    public_path = tmp_path / 'public.csv'
    key_path = tmp_path / 'key.csv'
    config_path = tmp_path / 'selected_configuration.json'
    prediction_path = tmp_path / 'frozen_initial_predictions.csv'
    plan_path = tmp_path / 'frozen_initial_batch_plans.json'
    data['public'].to_csv(public_path, index=False)
    pd.DataFrame({'row_id': list(data['key']), 'hr_pct': list(data['key'].values())}).to_csv(key_path, index=False)
    data['predictions'].to_csv(prediction_path, index=False)
    config_path.write_text(json.dumps(data['config']))
    plan_path.write_text(json.dumps(data['plans']))
    data['results'].to_csv(tmp_path / 'per_pool_results.csv', index=False)
    with gzip.open(tmp_path / 'evaluation_events.jsonl.gz', 'wt') as f:
        for e in data['events']:
            f.write(json.dumps(e) + '\n')
    freeze_path = tmp_path / 'evaluation_freeze.json'
    freeze([public_path, key_path, config_path, prediction_path, plan_path], freeze_path, {'synthetic': True})
    return dict(out=tmp_path, public_path=public_path, key_path=key_path, config_path=config_path,
                freeze_path=freeze_path, analysis_lock=tmp_path / 'absent_global_lock.json')


def test_end_to_end_synthetic_files_and_manifest(tmp_path):
    paths = write_synthetic_run(tmp_path)
    summary = verify_run(**paths)
    assert summary['status'] == 'passed'
    assert summary['frozen_run_files_checked'] == 5
    assert (tmp_path / 'verification/regenerated_per_pool_results.csv').is_file()
    assert (tmp_path / 'verification/verification_summary.json').is_file()


def test_changed_frozen_input_rejected_before_key_read(tmp_path, monkeypatch):
    paths = write_synthetic_run(tmp_path)
    paths['config_path'].write_text(paths['config_path'].read_text() + ' ')
    def never_read(*args, **kwargs):
        pytest.fail('CSV input opened before freeze validation')
    monkeypatch.setattr(pd, 'read_csv', never_read)
    with pytest.raises(ValueError, match='Frozen input changed'):
        verify_run(**paths)


def test_missing_freeze_rejected_before_key_read(tmp_path, monkeypatch):
    paths = write_synthetic_run(tmp_path)
    paths['freeze_path'].unlink()
    monkeypatch.setattr(pd, 'read_csv', lambda *a, **k: pytest.fail('CSV opened without freeze'))
    with pytest.raises(VerificationError, match='Missing evaluation freeze'):
        verify_run(**paths)


def test_unbound_key_rejected_before_read(tmp_path, monkeypatch):
    paths = write_synthetic_run(tmp_path)
    document = json.loads(paths['freeze_path'].read_text())
    document['files'].pop(str(paths['key_path'].resolve()))
    paths['freeze_path'].write_text(json.dumps(document))
    monkeypatch.setattr(pd, 'read_csv', lambda *a, **k: pytest.fail('CSV opened with unbound answer key'))
    with pytest.raises(VerificationError, match='No verified freeze binds'):
        verify_run(**paths)


def test_verifier_has_no_production_arithmetic_imports():
    import ast
    import src.verify
    parsed = ast.parse(Path(src.verify.__file__).read_text())
    modules = [n.module for n in ast.walk(parsed) if isinstance(n, ast.ImportFrom)]
    assert not set(modules) & {'replay', 'experiment', 'models', 'scoring', 'src.replay', 'src.experiment'}
