"""Post-freeze, fixed-query endpoint and source-precision sensitivities.

This module never runs policies, fits models, or creates new outcome observations.
The filesystem entry point verifies the run freeze before reading answer tables.
"""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .common import digest, verify_freeze, write_json
from .replay import ordered_ids, score_event, tie_key

KG_HA_PER_BU_AC = 45 * .45359237 / .40468564224
EVENT_KEYS = ('pool_id', 'policy', 'seed', 'budget_fraction', 'program', 'stratum',
              'year', 'site', 'n', 'k', 'budget')
PRODUCT_DEFINITIONS = ('matched_product_of_means', 'output_mean_of_matched_plot_products')


def _identity(event):
    return {key: event.get(key) for key in EVENT_KEYS}


def _indexed(frame, name):
    if 'row_id' not in frame or frame.row_id.isna().any() or frame.row_id.duplicated().any():
        raise ValueError('Missing or repeated row identity in ' + name)
    return frame.set_index('row_id', drop=False)


def _primary_validation(public, private, event):
    rows = _indexed(public, 'public pool')
    answers = _indexed(private, 'private source')
    if rows.pool_id.nunique() != 1 or str(rows.pool_id.iloc[0]) != str(event['pool_id']):
        raise ValueError('Event/public pool mismatch')
    if not set(rows.index).issubset(answers.index):
        raise ValueError('Missing observed answer identity')
    answers = answers.loc[rows.index]
    if not np.isfinite(rows.field_yield).all() or not rows.field_yield.gt(0).all():
        raise ValueError('Invalid or missing public field yield')
    for column in ['program', 'year', 'site', 'stratum']:
        if column in rows and column in event and not rows[column].astype(str).eq(str(event[column])).all():
            raise ValueError('Event/public metadata mismatch: '+column)
    if 'pool_id' in answers and not answers.pool_id.astype(str).eq(str(event['pool_id'])).all():
        raise ValueError('Private/public pool mismatch')
    hr = pd.to_numeric(answers.hr_percent, errors='raise')
    if not np.isfinite(hr).all() or not hr.between(0, 100).all():
        raise ValueError('Invalid or missing observed recovery')
    primary = score_event(public, hr.to_dict(), event)
    values = (rows.field_yield * hr / 100).to_dict()
    if event['shortlist'] != ordered_ids(event['tested_ids'], values)[:event['k']]:
        raise ValueError('Recorded shortlist differs from measured tested top-k')
    return rows, answers, values, primary


def _score_values(values, event):
    ids = list(values)
    if any(not np.isfinite(v) or v < 0 for v in values.values()):
        raise ValueError('Sensitivity requires observed finite nonnegative products')
    k = event['k']
    oracle = ordered_ids(ids, values)[:k]
    chosen = ordered_ids(event['tested_ids'], values)[:k]
    om = float(np.mean([values[i] for i in oracle]))
    sm = float(np.mean([values[i] for i in chosen]))
    regret = max(0., om-sm)
    return {'oracle_mean': om, 'selected_mean': sm, 'regret': regret,
            'regret_pct': regret/om*100 if om > 0 else np.nan,
            'top_k_recall': len(set(oracle) & set(chosen))/k,
            'oracle_ids': json.dumps(oracle), 'shortlist_ids': json.dumps(chosen),
            'selected_entries_changed_from_primary': len(set(chosen)-set(event['shortlist']))}


def product_definition_sensitivity(public, private, event):
    """Reselect measured top-k within the SAME recorded tested set for each endpoint."""
    rows, answers, values, primary = _primary_validation(public, private, event)
    missing_columns = [c for c in PRODUCT_DEFINITIONS + ('matched_plot_count',) if c not in answers]
    if missing_columns:
        complete = pd.Series(False, index=rows.index)
        reason = 'matched_product_fields_unavailable'
    else:
        complete = answers.matched_plot_count.gt(0)
        for column in PRODUCT_DEFINITIONS:
            complete &= np.isfinite(answers[column]) & answers[column].ge(0)
        reason = 'not_all_replay_candidates_have_observed_matched_products'
    common = {**_identity(event), 'queries_unchanged': True,
              'matched_candidates': int(complete.sum()),
              'missing_matched_candidates': int((~complete).sum()),
              'primary_regret_pct': primary['regret_pct'],
              'output_unit': 'Uruguay reported kg/ha proxy; moisture unspecified',
              'interpretation': 'fixed-query endpoint sensitivity, not a rerun of adaptive actions'}
    result = []
    for definition in PRODUCT_DEFINITIONS:
        record = {**common, 'definition': definition,
                  'status': 'available' if complete.all() else 'unavailable',
                  'unavailable_reason': '' if complete.all() else reason,
                  **dict.fromkeys(['oracle_mean', 'selected_mean', 'regret', 'regret_pct',
                                   'top_k_recall', 'selected_entries_changed_from_primary'], np.nan),
                  'oracle_ids': '', 'shortlist_ids': ''}
        if complete.all():
            record.update(_score_values(answers[definition].to_dict(), event))
        result.append(record)
    return result


def output_intervals(yield_bu_ac, hr_percent):
    """Closed nearest-integer rounding envelopes; endpoints clipped to physical bounds."""
    y = np.asarray(yield_bu_ac, dtype=float)
    h = np.asarray(hr_percent, dtype=float)
    if y.shape != h.shape or not np.isfinite(y).all() or not np.isfinite(h).all():
        raise ValueError('Missing or misaligned rounding inputs')
    if (y <= 0).any() or (h < 0).any() or (h > 100).any():
        raise ValueError('Invalid physical rounding inputs')
    if not np.allclose(y, np.round(y), atol=1e-10, rtol=0) or not np.allclose(h, np.round(h), atol=1e-10, rtol=0):
        raise ValueError('Source precision differs from prespecified integer units')
    return (np.maximum(0., y-.5)*KG_HA_PER_BU_AC*np.maximum(0., h-.5)/100,
            (y+.5)*KG_HA_PER_BU_AC*np.minimum(100., h+.5)/100)


def _outranks(value_a, id_a, value_b, id_b):
    return value_a > value_b or (value_a == value_b and tie_key(id_a) < tie_key(id_b))


def rank_interval_summary(point, lower, upper, k):
    """Conservative independent rectangular bounds with the frozen deterministic tie rule."""
    ids = list(point)
    if set(ids) != set(lower) or set(ids) != set(upper) or not 1 <= k <= len(ids):
        raise ValueError('Invalid interval identities or k')
    for i in ids:
        if not all(np.isfinite(x) for x in (point[i], lower[i], upper[i])) or not 0 <= lower[i] <= point[i] <= upper[i]:
            raise ValueError('Invalid output interval')
    top = ordered_ids(ids, point)[:k]
    outsider = [i for i in ids if i not in top]
    pairs = [(i, j) for i in top for j in outsider if _outranks(upper[j], j, lower[i], i)]
    ranks = []
    for i in ids:
        best = 1 + sum(_outranks(lower[j], j, upper[i], i) for j in ids if j != i)
        worst = 1 + sum(_outranks(upper[j], j, lower[i], i) for j in ids if j != i)
        ranks.append({'row_id': i, 'point_output': point[i], 'output_lower': lower[i],
                      'output_upper': upper[i], 'best_possible_rank': best,
                      'worst_possible_rank': worst, 'point_top_k': i in top,
                      'guaranteed_top_k': worst <= k, 'possible_top_k': best <= k})
    return {'membership_may_change': bool(pairs), 'ambiguous_boundary_pairs': len(pairs),
            'selected_candidates_at_risk': len({i for i, _ in pairs}),
            'outside_candidates_possible': len({j for _, j in pairs}),
            'point_top_k_ids': json.dumps(top)}, ranks


def rounding_interval_sensitivity(public, private, source_public, event):
    rows, answers, point, _ = _primary_validation(public, private, event)
    source = _indexed(source_public, 'Arkansas public source')
    if not set(rows.index).issubset(source.index):
        raise ValueError('Missing source precision identity')
    source = source.loc[rows.index]
    y = pd.to_numeric(source.field_yield_bu_ac, errors='raise')
    if not np.allclose(y*KG_HA_PER_BU_AC, rows.field_yield, atol=1e-8, rtol=1e-12):
        raise ValueError('Arkansas field-yield conversion differs from audited source')
    lo, hi = output_intervals(y, answers.hr_percent)
    lower = dict(zip(rows.index, lo)); upper = dict(zip(rows.index, hi))
    oracle, intervals = rank_interval_summary(point, lower, upper, event['k'])
    tested = event['tested_ids']
    shortlist, _ = rank_interval_summary({i: point[i] for i in tested},
                                       {i: lower[i] for i in tested},
                                       {i: upper[i] for i in tested}, event['k'])
    record = {**_identity(event), 'status': 'available', 'queries_unchanged': True,
              'yield_rounding_halfunit_bu_ac': .5, 'hr_rounding_halfunit_pp': .5,
              'interval_interpretation': 'source-rounding envelope, not a confidence interval',
              **{'oracle_'+key: val for key, val in oracle.items()},
              **{'tested_shortlist_'+key: val for key, val in shortlist.items()}}
    for row in intervals:
        row.update({'pool_id': event['pool_id'], 'program': event.get('program'),
                    'year': event.get('year'), 'output_unit': 'kg/ha at 12% rough-rice moisture'})
    return record, intervals


def paired_coverage(all_public):
    needed = {'pool_id', 'row_id', 'observed_field', 'paired'}
    if not needed.issubset(all_public):
        raise ValueError('Missing public paired-coverage fields')
    _indexed(all_public, 'all public metadata')
    frame = all_public.copy()
    for column in ['observed_field', 'paired']:
        tokens = frame[column].astype(str).str.lower()
        if not tokens.isin(['true', 'false', '1', '0']).all():
            raise ValueError('Invalid public coverage boolean')
        frame[column] = tokens.isin(['true', '1'])
    if (frame.paired & ~frame.observed_field).any():
        raise ValueError('Paired candidate lacks observed field yield')
    records = []
    for pool, group in frame.groupby('pool_id', sort=True):
        nfield = int(group.observed_field.sum()); npaired = int(group.paired.sum())
        fraction = npaired/nfield if nfield else np.nan
        records.append({'pool_id': pool, 'source_candidate_count': len(group),
                        'observed_field_candidate_count': nfield,
                        'paired_candidate_count': npaired,
                        'paired_fraction_of_field_candidates': fraction,
                        'coverage_stratum': ('no_observed_field' if not nfield else
                                             'complete' if npaired == nfield else
                                             '90_to_less_than_100_percent' if fraction >= .9 else
                                             'below_90_percent')})
    return pd.DataFrame(records)


def _check_freeze(freeze_path, required_paths):
    manifest = json.loads(Path(freeze_path).read_text())
    verify_freeze(freeze_path)
    frozen = {str(Path(p).resolve()) for p in manifest['files']}
    missing = {str(Path(p).resolve()) for p in required_paths} - frozen
    if missing:
        raise ValueError('Sensitivity inputs absent from run freeze: ' + ', '.join(sorted(missing)))


def run_sensitivities(events_path, evaluation_public_path, all_public_metadata_path,
                      uruguay_private_path, arkansas_private_path, arkansas_source_public_path,
                      run_freeze_path, out_dir):
    """Read real outputs only after validation; the event log is POST-freeze evidence."""
    required = [evaluation_public_path, all_public_metadata_path, uruguay_private_path,
                arkansas_private_path, arkansas_source_public_path]
    _check_freeze(run_freeze_path, required)
    paths = [Path(events_path), Path(run_freeze_path)] + [Path(p) for p in required]
    before = {str(p.resolve()): digest(p) for p in paths}
    public = pd.read_csv(evaluation_public_path, dtype={'row_id': str}, low_memory=False)
    all_public = pd.read_csv(all_public_metadata_path, dtype={'row_id': str}, low_memory=False)
    source_private = {'uruguay': pd.read_csv(uruguay_private_path, dtype={'row_id': str}),
                      'arkansas': pd.read_csv(arkansas_private_path, dtype={'row_id': str})}
    private = {}
    for program, source in source_private.items():
        _indexed(source, program+' private source')
        private[program] = {str(key): group for key, group in source.groupby('pool_id', sort=True)}
    ar_source = pd.read_csv(arkansas_source_public_path, dtype={'row_id': str}, low_memory=False)
    _indexed(ar_source, 'Arkansas source public')
    ar_public = {str(key): group for key, group in ar_source.groupby('pool_id', sort=True)}
    _indexed(public, 'evaluation public')
    pools = {str(key): group for key, group in public.groupby('pool_id', sort=True)}
    coverage = paired_coverage(all_public).set_index('pool_id')
    product_rows = []; rounding_rows = []; interval_rows = []; coverage_rows = []
    seen = set(); interval_pools = set(); count = 0
    opener = gzip.open if str(events_path).endswith('.gz') else open
    with opener(events_path, 'rt') as stream:
        for line in stream:
            event = json.loads(line); count += 1
            identity = tuple(event.get(key) for key in ('pool_id', 'policy', 'seed', 'budget_fraction'))
            if identity in seen:
                raise ValueError('Duplicate event identity')
            seen.add(identity)
            pool = str(event['pool_id']); program = event['program']
            if pool not in pools or program not in private:
                raise ValueError('Unknown event pool or program')
            frame = pools[pool]
            if not frame.program.eq(program).all():
                raise ValueError('Event/public program mismatch')
            if pool not in private[program]:
                raise ValueError('Missing private source pool')
            pool_private = private[program][pool]
            _, _, _, primary = _primary_validation(frame, pool_private, event)
            if pool not in coverage.index or int(coverage.loc[pool, 'paired_candidate_count']) != len(frame):
                raise ValueError('Replay pool differs from public paired candidate universe')
            coverage_rows.append({**_identity(event), **coverage.loc[pool].to_dict(),
                                  **{c: primary[c] for c in ['regret_pct', 'top_k_recall', 'tests']}})
            if program == 'uruguay':
                product_rows.extend(product_definition_sensitivity(frame, pool_private, event))
            else:
                record, intervals = rounding_interval_sensitivity(frame, pool_private, ar_public[pool], event)
                rounding_rows.append(record)
                if pool not in interval_pools:
                    interval_rows.extend(intervals); interval_pools.add(pool)
    if not count:
        raise ValueError('No completed evaluation events')
    if {identity[0] for identity in seen} != set(pools):
        raise ValueError('Evaluation event log does not cover every public replay pool')
    if {str(p.resolve()): digest(p) for p in paths} != before:
        raise ValueError('Sensitivity source changed while reading')
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    tables = {'product_definition_sensitivity.csv': product_rows,
              'rounding_interval_sensitivity.csv': rounding_rows,
              'rounding_candidate_intervals.csv': interval_rows,
              'paired_coverage_sensitivity.csv': coverage_rows}
    for filename, rows in tables.items():
        pd.DataFrame(rows, columns=None if rows else list(EVENT_KEYS)+['status']).to_csv(out/filename, index=False)
    manifest = {'run_freeze_sha256': before[str(Path(run_freeze_path).resolve())],
                'postfreeze_eventlog_sha256': before[str(Path(events_path).resolve())],
                'eventlog_was_prefrozen': False, 'completed_events': count,
                'input_hashes': before, 'output_hashes': {name: digest(out/name) for name in tables},
                'query_sets_changed': False, 'policy_retuning': False,
                'sampling_uncertainty_estimated': False,
                'interpretation': 'fixed-query endpoint and deterministic source-rounding sensitivity'}
    write_json(out/'sensitivity_manifest.json', manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ['events', 'evaluation-public', 'all-public-metadata', 'uruguay-private',
                 'arkansas-private', 'arkansas-source-public', 'run-freeze', 'out']:
        parser.add_argument('--'+flag, required=True)
    a = parser.parse_args()
    run_sensitivities(a.events, a.evaluation_public, a.all_public_metadata, a.uruguay_private,
                      a.arkansas_private, a.arkansas_source_public, a.run_freeze, a.out)


if __name__ == '__main__':
    main()
