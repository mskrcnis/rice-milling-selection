"""Post hoc Arkansas rounding diagnostics; the source project is read-only.

Exact fixed-query envelopes are distinct from joint feasible scenario results.
No original model, policy, cohort, budget, source file or result is changed.
"""
import argparse
import gzip
import hashlib
import io
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

REVISION = Path(__file__).resolve().parents[1]
FACTOR = 45 * .45359237 / .40468564224
METRICS = ('recall_pct', 'relative_regret_pct', 'native_regret_kg_ha')
COHORTS = ('all_62', 'historical_19', 'contemporary_43')
TOLERANCE = 1e-8


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def hash_rank(identifier):
    return hashlib.sha256(('rice-milling-tie-v1|' + str(identifier)).encode()).hexdigest()


def rank(scores):
    # Columns have already been put in SHA order; stable sorting preserves ties.
    return np.argsort(-np.asarray(scores), axis=-1, kind='stable')


def scalar_metrics(yield_bu, recovery, k, budget, fixed=None):
    n = len(yield_bu)
    indices = list(range(n))
    products = [float(yield_bu[i]) * float(recovery[i]) for i in indices]
    oracle = sorted(indices, key=lambda i: (-products[i], i))[:k]
    queried = list(fixed) if fixed is not None else sorted(indices, key=lambda i: (-float(yield_bu[i]), i))[:budget]
    selected = sorted(queried, key=lambda i: (-products[i], i))[:k]
    oracle_mean = math.fsum(products[i] for i in oracle) / k
    selected_mean = math.fsum(products[i] for i in selected) / k
    require(oracle_mean > 0, 'Undefined positive-oracle requirement')
    return {'recall_pct': 100 * len(set(oracle) & set(selected)) / k,
            'relative_regret_pct': 100 * (oracle_mean - selected_mean) / oracle_mean,
            'native_regret_kg_ha': (oracle_mean - selected_mean) * FACTOR / 100,
            'oracle': oracle, 'queried': queried, 'selected': selected}


def vector_metrics(yield_bu, recovery, k, budget, fixed=None, nominal=None):
    y = np.atleast_2d(yield_bu)
    h = np.atleast_2d(recovery)
    values = y * h
    oracle = rank(values)[:, :k]
    queries = rank(y)[:, :budget] if fixed is None else np.broadcast_to(fixed, (len(y), budget))
    q_values = np.take_along_axis(values, queries, axis=1)
    local_rank = rank(q_values)[:, :k]
    selected = np.take_along_axis(queries, local_rank, axis=1)
    # The query-order columns are not SHA-sorted. Explicit secondary ordering
    # makes ties use global SHA rank for the final selected identities.
    selected = np.empty((len(y), k), dtype=int)
    order = np.lexsort((queries, -q_values), axis=1)[:, :k]
    selected[:] = np.take_along_axis(queries, order, axis=1)
    oracle_mean = np.take_along_axis(values, oracle, axis=1).sum(axis=1) / k
    selected_mean = np.take_along_axis(values, selected, axis=1).sum(axis=1) / k
    difference = oracle_mean - selected_mean
    require(np.all(oracle_mean > 0), 'Nonpositive scenario oracle')
    require(np.all(difference >= -1e-9), 'Negative regret beyond floating-point noise')
    difference = np.maximum(difference, 0.)
    query_membership = np.zeros(values.shape, dtype=bool)
    np.put_along_axis(query_membership, queries, True, axis=1)
    recall = np.take_along_axis(query_membership, oracle, axis=1).sum(axis=1) * 100 / k
    result = {'recall_pct': recall, 'relative_regret_pct': difference * 100 / oracle_mean,
              'native_regret_kg_ha': difference * FACTOR / 100}
    if nominal is not None:
        result['query_changed_pools'] = np.any(np.sort(queries, axis=1) != np.sort(nominal['queried']), axis=1).astype(float)
        result['oracle_changed_pools'] = np.any(np.sort(oracle, axis=1) != np.sort(nominal['oracle']), axis=1).astype(float)
        result['shortlist_changed_pools'] = np.any(np.sort(selected, axis=1) != np.sort(nominal['selected']), axis=1).astype(float)
    return result


def check_scalar(y, h, k, budget, vector, selected_draws, fixed=None):
    largest = 0.
    for index in selected_draws:
        scalar = scalar_metrics(y[index], h[index], k, budget, fixed)
        for metric in METRICS:
            error = abs(float(vector[metric][index]) - scalar[metric])
            require(error < TOLERANCE, f'Scalar/vector mismatch {metric}: {error}')
            largest = max(largest, error)
    return largest


def synthetic_extrema_check():
    # Enumerate every vertex independently for small boxes, including ties.
    cases = [(np.array([2., 3., 4., 5.]), np.array([40., 50., 60., 70.]), 1, [0, 2]),
             (np.array([3., 3., 4., 4.]), np.array([50., 50., 50., 50.]), 2, [0, 1, 3])]
    checked = 0
    for y, h, k, queried in cases:
        lower = (y - .5) * (h - .5)
        upper = (y + .5) * (h + .5)
        mask = np.array([i in queried for i in range(len(y))])
        bad = np.where(mask, lower, upper)
        good = np.where(mask, upper, lower)
        def metric_from_values(values):
            return scalar_metrics(values, np.ones(len(y)), k, len(queried), queried)
        adverse, favorable = metric_from_values(bad), metric_from_values(good)
        for bits in itertools.product([0, 1], repeat=len(y)):
            point = metric_from_values(np.where(bits, upper, lower))
            require(adverse['recall_pct'] <= point['recall_pct'] <= favorable['recall_pct'], 'Recall corner proof check failed')
            for metric in ['relative_regret_pct', 'native_regret_kg_ha']:
                require(favorable[metric] - 1e-9 <= point[metric] <= adverse[metric] + 1e-9, 'Regret corner proof check failed')
            checked += 1
    return checked


def verify_inputs(source, plan, lock, plan_path):
    require(sha(plan_path) == lock['plan_sha256'], 'Diagnostic plan changed')
    require(sha(__file__) == lock['code_sha256'], 'Diagnostic implementation differs from pre-computation lock')
    original = Path(plan['original_root'])
    rebased = {}
    for old_name, expected in plan['input_sha256'].items():
        relative = Path(old_name).relative_to(original)
        current = source / relative
        require(sha(current) == expected, 'Source input changed: ' + str(relative))
        rebased[str(relative)] = expected
    return rebased


def load_inputs(source, plan):
    public = pd.read_csv(source/'derived/cohort/evaluation_public.csv')
    public = public[public.stratum.eq('arkansas_external')].copy()
    private = pd.read_csv(source/'private/evaluation_key.csv')
    native = pd.read_csv(source/'derived/arkansas/public_features.csv')
    frame = public.merge(private, on='row_id', validate='one_to_one').merge(
        native[['row_id', 'field_yield_bu_ac']], on='row_id', validate='one_to_one')
    require(len(frame) == 1611 and frame.pool_id.nunique() == 62, 'Original external cohort differs')
    require(np.allclose(frame.field_yield, frame.field_yield_bu_ac * FACTOR, atol=1e-8, rtol=0), 'Source yield conversion mismatch')
    require(np.all(frame.hr_pct == np.round(frame.hr_pct)) and np.all(frame.field_yield_bu_ac == np.round(frame.field_yield_bu_ac)), 'Half-unit assumptions do not match integer source precision')
    require(frame.hr_pct.between(0, 100).all() and frame.field_yield_bu_ac.gt(.5).all(), 'Invalid source outcome range')
    events = {}
    eligible = set(frame.pool_id)
    with gzip.open(source/'outputs/run_a/evaluation_events.jsonl.gz', 'rt') as handle:
        for line in handle:
            event = json.loads(line)
            if event['pool_id'] in eligible and event['policy'] == 'yield':
                token = (event['pool_id'], event['budget_fraction'])
                require(token not in events, 'Duplicate original yield-first event')
                events[token] = event
    require(len(events) == 62 * 6, 'Incomplete original yield-first events')
    archived = pd.read_csv(source/'outputs/run_a/per_pool_results.csv')
    archived = archived[archived.policy.eq('yield')].set_index(['pool_id', 'budget_fraction'])
    coverage = pd.read_csv(source/'outputs/run_a/sensitivity/paired_coverage_sensitivity.csv')
    coverage = coverage[coverage.stratum.eq('arkansas_external') & coverage.policy.eq('yield') & coverage.budget_fraction.eq(.25)]
    require(len(coverage) == 62, 'Coverage table differs')
    reference = pd.read_csv(source/'outputs/run_a/tables/policy_summary.csv')
    reference = reference[reference.stratum.eq('uruguay_temporal') & reference.policy.eq('yield')].set_index('budget_fraction')
    return frame, events, archived, coverage, reference


def run(source, output, plan_path, lock_path):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(not output.is_relative_to(source), 'Diagnostic output must be outside the original source project')
    plan = json.loads(Path(plan_path).read_text())
    lock = json.loads(Path(lock_path).read_text())
    bindings = verify_inputs(source, plan, lock, plan_path)
    if output.exists():
        require(not any(output.iterdir()), 'Use an empty output directory; existing diagnostic outputs are preserved')
    output.mkdir(parents=True, exist_ok=True)
    synthetic_vertices = synthetic_extrema_check()
    frame, events, archived, coverage, reference = load_inputs(source, plan)
    fractions = plan['budgets']
    draws = plan['joint_perturbation']['draws']
    all_metrics = list(METRICS) + ['query_changed_pools', 'oracle_changed_pools', 'shortlist_changed_pools']
    random_sums = {c: {m: np.zeros((draws, len(fractions))) for m in all_metrics} for c in COHORTS}
    scenario_names = [f'global_y{ys:+d}_h{hs:+d}' for ys, hs in itertools.product([-1, 1], repeat=2)]
    scenario_names += [f'target_{f:g}_{direction}' for f in fractions for direction in ['adverse', 'favorable']]
    stress_sums = {c: {m: np.zeros((len(scenario_names), len(fractions))) for m in all_metrics} for c in COHORTS}
    pool_rows, numerical_rows = [], []
    check_count = 0
    maximum_error = 0.
    for pool, raw in frame.groupby('pool_id', sort=True):
        raw = raw.assign(sha_tie=raw.row_id.map(hash_rank)).sort_values('sha_tie').reset_index(drop=True)
        ids = raw.row_id.tolist()
        y, h = raw.field_yield_bu_ac.to_numpy(float), raw.hr_pct.to_numpy(float)
        n, k = len(raw), math.ceil(.1 * len(raw))
        year = int(raw.year.iloc[0])
        cohorts = ['all_62', 'historical_19' if year < 2021 else 'contemporary_43']
        original_queries = {}
        stress_y, stress_h = [], []
        for ys, hs in itertools.product([-1, 1], repeat=2):
            stress_y.append(y + ys * .5); stress_h.append(np.clip(h + hs * .5, 0, 100))
        for fraction in fractions:
            budget = min(n, max(k, math.ceil(fraction * n)))
            event = events[(pool, fraction)]
            queried = [ids.index(identifier) for identifier in event['tested_ids']]
            require(queried == rank(y)[:budget].tolist(), 'Archived query order differs from frozen yield/SHA ordering')
            original_queries[fraction] = queried
            mask = np.isin(np.arange(n), queried)
            for adverse in [True, False]:
                direction = np.where(mask, -1 if adverse else 1, 1 if adverse else -1)
                stress_y.append(y + direction * .5)
                stress_h.append(np.clip(h + direction * .5, 0, 100))
        stress_y, stress_h = np.asarray(stress_y), np.asarray(stress_h)
        seed_text = f'rounding-diagnostic-v1|{plan["joint_perturbation"]["master_seed"]}|{pool}'
        seed = int.from_bytes(hashlib.sha256(seed_text.encode()).digest()[:16], 'little')
        random = np.random.Generator(np.random.PCG64(seed)).uniform(-.5, .5, size=(draws, n, 2))
        random_y, random_h = y + random[:, :, 0], np.clip(h + random[:, :, 1], 0, 100)
        for j, fraction in enumerate(fractions):
            queried = original_queries[fraction]
            budget = len(queried)
            nominal = scalar_metrics(y, h, k, budget, queried)
            original = archived.loc[(pool, fraction)]
            native_recall_delta = nominal['recall_pct'] - original.top_k_recall * 100
            native_regret_delta = nominal['relative_regret_pct'] - original.regret_pct
            numerical_rows.append({'pool_id': pool, 'budget_fraction': fraction,
                                   'native_product_minus_archived_recall_pp': native_recall_delta,
                                   'native_product_minus_archived_regret_pp': native_regret_delta})
            # Preserve the archived reference even if unit-conversion float ties
            # differ from exact native half-unit products in this diagnostic.
            nominal['oracle'] = [ids.index(i) for i in rank_original_event_oracle(raw, k)]
            nominal['selected'] = [ids.index(i) for i in events[(pool, fraction)]['shortlist']]
            row = {'pool_id': pool, 'year': year, 'site': raw.site.iloc[0], 'cohort': cohorts[1],
                   'n': n, 'k': k, 'budget_fraction': fraction, 'B': budget, 'actual_fraction': budget/n,
                   'archived_recall_pct': float(original.top_k_recall * 100),
                   'archived_relative_regret_pct': float(original.regret_pct),
                   'archived_native_regret_kg_ha': float(original.regret)}
            for corner_index, direction in [(4 + 2*j, 'worst'), (5 + 2*j, 'best')]:
                cy, ch = stress_y[corner_index:corner_index+1], stress_h[corner_index:corner_index+1]
                metrics = vector_metrics(cy, ch, k, budget, queried)
                maximum_error = max(maximum_error, check_scalar(cy, ch, k, budget, metrics, [0], queried))
                check_count += 1
                for metric in METRICS: row[direction+'_'+metric] = float(metrics[metric][0])
            pool_rows.append(row)
            joint = vector_metrics(random_y, random_h, k, budget, nominal=nominal)
            sample = plan['independent_checks']['joint_draw_indices_zero_based']
            maximum_error = max(maximum_error, check_scalar(random_y, random_h, k, budget, joint, sample))
            check_count += len(sample)
            stress = vector_metrics(stress_y, stress_h, k, budget, nominal=nominal)
            maximum_error = max(maximum_error, check_scalar(stress_y, stress_h, k, budget, stress, range(len(stress_y))))
            check_count += len(stress_y)
            if fraction == 1:
                require(np.all(joint['recall_pct'] == 100) and np.all(joint['relative_regret_pct'] == 0), 'Joint full-budget identity failed')
                require(np.all(stress['recall_pct'] == 100) and np.all(stress['relative_regret_pct'] == 0), 'Stress full-budget identity failed')
            for cohort in cohorts:
                for metric in all_metrics:
                    random_sums[cohort][metric][:, j] += joint[metric]
                    stress_sums[cohort][metric][:, j] += stress[metric]
        print(f'Completed {pool}: {n} candidates', flush=True)
    fixed = pd.DataFrame(pool_rows)
    fixed.to_csv(output/'fixed_query_pool_bounds.csv', index=False, float_format='%.17g')
    pd.DataFrame(numerical_rows).to_csv(output/'native_float_tie_audit.csv', index=False, float_format='%.17g')
    budget_rows, uniform_rows, stress_rows, crossing_rows = [], [], [], []
    scenario_rows = []
    for cohort in COHORTS:
        part = fixed if cohort == 'all_62' else fixed[fixed.cohort.eq(cohort)]
        pool_count = part.pool_id.nunique()
        subset = coverage[coverage.pool_id.isin(part.pool_id)]
        for metric in METRICS:
            random_sums[cohort][metric] /= pool_count
            stress_sums[cohort][metric] /= pool_count
        for j, fraction in enumerate(fractions):
            group = part[part.budget_fraction.eq(fraction)]
            uy = reference.loc[fraction]
            record = {'cohort': cohort, 'budget_fraction': fraction, 'pools': pool_count,
                      'measured_entries': int(group.n.sum()), 'observed_field_entries': int(subset.observed_field_candidate_count.sum()),
                      'total_tests': int(group.B.sum()), 'mean_actual_fraction': float(group.actual_fraction.mean()),
                      'pooled_actual_fraction': float(group.B.sum()/group.n.sum()),
                      'min_actual_fraction': float(group.actual_fraction.min()), 'max_actual_fraction': float(group.actual_fraction.max()),
                      'uruguay_fixed_recall_pct': float(uy.top_k_recall*100), 'uruguay_fixed_regret_pct': float(uy.regret_pct)}
            for metric in METRICS:
                record['archived_'+metric] = float(group['archived_'+metric].mean())
                record['fixed_query_lower_'+metric] = float(group[('worst_' if metric == 'recall_pct' else 'best_')+metric].mean())
                record['fixed_query_upper_'+metric] = float(group[('best_' if metric == 'recall_pct' else 'worst_')+metric].mean())
            budget_rows.append(record)
            for metric in all_metrics:
                values = random_sums[cohort][metric][:, j]
                q = np.quantile(values, [.025, .5, .975])
                uniform_rows.append({'cohort': cohort, 'budget_fraction': fraction, 'metric': metric,
                                     'pools': pool_count, 'scenario_draws': draws, 'scenario_mean': float(values.mean()),
                                     'scenario_minimum': float(values.min()), 'scenario_p025': float(q[0]),
                                     'scenario_median': float(q[1]), 'scenario_p975': float(q[2]),
                                     'scenario_maximum': float(values.max()),
                                     'interpretation': 'Uniform rounding-scenario summary; not a confidence interval or guaranteed joint bound'})
            for i, name in enumerate(scenario_names):
                stress_rows.append({'cohort': cohort, 'scenario': name, 'budget_fraction': fraction,
                                    **{m: float(stress_sums[cohort][m][i, j]) for m in all_metrics}})
            scenario_rows.append(pd.DataFrame({'cohort': cohort, 'draw': np.arange(draws), 'budget_fraction': fraction,
                                               **{m: random_sums[cohort][m][:, j] for m in all_metrics}}))
        candidates = [r for r in budget_rows if r['cohort'] == cohort]
        original_cross = next(r['budget_fraction'] for r in candidates if r['archived_recall_pct'] >= 90)
        possible = next(r['budget_fraction'] for r in candidates if r['fixed_query_upper_recall_pct'] >= 90)
        guaranteed = next(r['budget_fraction'] for r in candidates if r['fixed_query_lower_recall_pct'] >= 90)
        for kind, values in [('uniform', random_sums[cohort]['recall_pct']), ('deterministic', stress_sums[cohort]['recall_pct'])]:
            earliest = np.asarray(fractions)[np.argmax(values >= 90, axis=1)]
            for fraction in fractions:
                crossing_rows.append({'cohort': cohort, 'scenario_family': kind, 'first_grid_fraction': fraction,
                                      'scenario_count': int(np.sum(earliest == fraction)), 'total_scenarios': len(earliest),
                                      'archived_crossing': original_cross, 'fixed_query_possible_crossing': possible,
                                      'fixed_query_guaranteed_crossing': guaranteed})
    pd.DataFrame(budget_rows).to_csv(output/'cohort_budget_summary.csv', index=False, float_format='%.17g')
    pd.DataFrame(uniform_rows).to_csv(output/'joint_uniform_summary.csv', index=False, float_format='%.17g')
    pd.DataFrame(stress_rows).to_csv(output/'joint_deterministic_scenarios.csv', index=False, float_format='%.17g')
    pd.DataFrame(crossing_rows).to_csv(output/'grid_crossing_summary.csv', index=False)
    with (output/'joint_uniform_draws.csv.gz').open('wb') as binary:
        with gzip.GzipFile(filename='', fileobj=binary, mode='wb', mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8', newline='') as text:
                pd.concat(scenario_rows, ignore_index=True).to_csv(text, index=False, float_format='%.17g')
    numerical = pd.DataFrame(numerical_rows)
    verification = {'status': 'passed', 'scalar_scenario_budget_checks': check_count,
                    'scalar_metric_checks': check_count * len(METRICS), 'maximum_absolute_metric_discrepancy': maximum_error,
                    'absolute_tolerance': TOLERANCE, 'synthetic_rectangle_vertices': synthetic_vertices,
                    'original_query_sets_checked': len(pool_rows),
                    'native_vs_archived_recall_nonzero_rows': int(numerical.native_product_minus_archived_recall_pp.abs().gt(1e-9).sum()),
                    'native_vs_archived_max_regret_difference_pp': float(numerical.native_product_minus_archived_regret_pp.abs().max()),
                    'full_budget_identity_checked': True, 'same_yield_realization_for_queries_and_scoring': True,
                    'same_realization_across_all_budgets': True}
    write_json(output/'independent_verification.json', verification)
    verify_inputs(source, plan, lock, plan_path)
    manifest = {'post_hoc': True, 'status': 'complete', 'plan_sha256': sha(plan_path), 'code_sha256': sha(__file__),
                'input_sha256_relative_to_source': bindings,
                'output_sha256': {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()},
                'original_files_modified': False,
                'scope': 'Sharp fixed-query pointwise metric envelopes and feasible joint perturbation scenarios; no guaranteed joint policy bounds or sampling confidence intervals'}
    write_json(output/'rounding_diagnostic_manifest.json', manifest)
    return manifest


def rank_original_event_oracle(raw, k):
    values = raw.field_yield.to_numpy(float) * raw.hr_pct.to_numpy(float) / 100
    return raw.row_id.iloc[rank(values)[:k]].tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-project', type=Path, default=REVISION.parent/'paper1_rice_milling_selection')
    parser.add_argument('--output-dir', type=Path, default=REVISION/'outputs/rounding')
    parser.add_argument('--plan', type=Path, default=REVISION/'protocol/rounding_diagnostic_plan.json')
    parser.add_argument('--lock', type=Path, default=REVISION/'protocol/rounding_diagnostic_lock.json')
    args = parser.parse_args()
    print(json.dumps(run(args.source_project, args.output_dir, args.plan, args.lock), indent=2))


if __name__ == '__main__':
    main()
