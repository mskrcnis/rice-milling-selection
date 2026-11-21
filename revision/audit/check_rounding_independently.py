"""Independent scalar check of predeclared rounding draws and cohort summaries.

Does not import rounding_diagnostic or any original project calculation module.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


def main(source, diagnostic, destination):
    fields = pd.read_csv(source/'derived/cohort/evaluation_public.csv')
    fields = fields[fields.stratum.eq('arkansas_external')]
    native = pd.read_csv(source/'derived/arkansas/public_features.csv').set_index('row_id')
    key = pd.read_csv(source/'private/evaluation_key.csv').set_index('row_id').hr_pct.to_dict()
    indices = [0, 1, 2, 9, 99, 999, 9999, 19999]
    fractions = [.1, .2, .25, .4, .6, 1.]
    sums = defaultdict(list)
    factor = 45 * .45359237 / .40468564224
    scalar_pool_decisions = 0
    for pool, group in fields.groupby('pool_id', sort=True):
        tie = {i: hashlib.sha256(('rice-milling-tie-v1|' + i).encode()).hexdigest() for i in group.row_id}
        ids = sorted(tie, key=tie.get)
        seed = int.from_bytes(hashlib.sha256(('rounding-diagnostic-v1|20261120|' + pool).encode()).digest()[:16], 'little')
        offsets = np.random.Generator(np.random.PCG64(seed)).uniform(-.5, .5, size=(20000, len(ids), 2))
        for draw in indices:
            y = {i: float(native.loc[i, 'field_yield_bu_ac']) + float(offsets[draw, j, 0]) for j, i in enumerate(ids)}
            h = {i: min(100, max(0, float(key[i]) + float(offsets[draw, j, 1]))) for j, i in enumerate(ids)}
            v = {i: y[i] * h[i] for i in ids}
            k = math.ceil(len(ids) / 10)
            full = sorted(ids, key=lambda i: (-v[i], tie[i]))[:k]
            full_mean = math.fsum(v[i] for i in full) / k
            y_order = sorted(ids, key=lambda i: (-y[i], tie[i]))
            for fraction in fractions:
                b = min(len(ids), max(k, math.ceil(len(ids) * fraction)))
                queried = y_order[:b]
                selected = sorted(queried, key=lambda i: (-v[i], tie[i]))[:k]
                selected_mean = math.fsum(v[i] for i in selected) / k
                values = {'recall_pct': len(set(full) & set(selected)) * 100 / k,
                          'relative_regret_pct': (full_mean - selected_mean) * 100 / full_mean,
                          'native_regret_kg_ha': (full_mean - selected_mean) * factor / 100}
                for cohort in ['all_62', 'historical_19' if int(group.year.iloc[0]) < 2021 else 'contemporary_43']:
                    for metric, value in values.items():
                        sums[(cohort, draw, fraction, metric)].append(value)
                scalar_pool_decisions += 1
    actual = pd.read_csv(diagnostic/'joint_uniform_draws.csv.gz')
    actual = actual[actual.draw.isin(indices)].set_index(['cohort', 'draw', 'budget_fraction'])
    largest = 0.
    for (cohort, draw, fraction, metric), values in sums.items():
        expected = math.fsum(values) / len(values)
        error = abs(expected - actual.loc[(cohort, draw, fraction), metric])
        if error > 1e-8:
            raise ValueError((cohort, draw, fraction, metric, expected, error))
        largest = max(largest, error)
    result = {'status': 'passed', 'scalar_pool_budget_decisions': scalar_pool_decisions,
              'aggregate_metric_checks': len(sums), 'maximum_absolute_discrepancy': largest,
              'absolute_tolerance': 1e-8, 'draw_indices_zero_based': indices,
              'independence': 'Python dictionaries, sorted IDs and math.fsum; no imports from diagnostic or original replay code',
              'input_draw_table_sha256': hashlib.sha256((diagnostic/'joint_uniform_draws.csv.gz').read_bytes()).hexdigest()}
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-project', type=Path, required=True)
    parser.add_argument('--diagnostic-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.source_project, args.diagnostic_dir, args.output)
