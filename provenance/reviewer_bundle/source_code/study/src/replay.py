from dataclasses import dataclass
import hashlib
import math
import numpy as np
import pandas as pd

PUBLIC_COLUMNS = ('row_id', 'pool_id', 'candidate_id', 'site', 'year', 'program',
                  'field_yield', 'grain_class', 'stage', 'height', 'days_to_flowering',
                  'yield_rel', 'yield_rank', 'n_field_replicates', 'source_class')


def public_view(frame):
    required = {'row_id', 'pool_id', 'candidate_id', 'field_yield', 'grain_class'}
    if not required.issubset(frame.columns):
        raise ValueError('Missing public fields: ' + str(required - set(frame.columns)))
    if frame.row_id.duplicated().any() or frame[['pool_id', 'candidate_id']].duplicated().any():
        raise ValueError('Duplicate candidate or query identity')
    result = frame.loc[:, [c for c in PUBLIC_COLUMNS if c in frame]].copy()
    if not np.isfinite(result.field_yield).all() or (result.field_yield <= 0).any():
        raise ValueError('Field yield must be observed, finite and positive')
    result['grain_class'] = result.grain_class.fillna('unknown').astype(str)
    if ('yield_rel' in result) != ('yield_rank' in result):
        raise ValueError('Both precomputed public yield features must be supplied together')
    if 'yield_rel' not in result:
        means = result.groupby('pool_id').field_yield.transform('mean')
        result['yield_rel'] = result.field_yield / means
        result['yield_rank'] = result.groupby('pool_id').field_yield.rank(method='average', pct=True)
    if not np.isfinite(result[['yield_rel','yield_rank']]).all().all() or (result.yield_rel <= 0).any() or not result.yield_rank.between(0,1,inclusive='right').all():
        raise ValueError('Invalid precomputed public yield features')
    return result.sort_values('row_id').reset_index(drop=True)


def tie_key(candidate_id):
    return hashlib.sha256(('rice-milling-tie-v1|' + str(candidate_id)).encode()).hexdigest()


def ordered_ids(ids, values):
    return sorted(ids, key=lambda i: (-float(values[i]), tie_key(i)))


def budget(n, fraction):
    if n < 1 or not 0 < fraction <= 1:
        raise ValueError('Invalid pool or budget')
    k = math.ceil(.10 * n)
    return k, min(n, max(k, math.ceil(fraction * n)))


class QueryGate:
    def __init__(self, key, candidate_ids, maximum):
        if len(set(candidate_ids)) != len(candidate_ids):
            raise ValueError('Repeated query identities')
        self.__key = {i: float(key[i]) for i in candidate_ids}
        if any(not np.isfinite(v) or v < 0 or v > 100 for v in self.__key.values()):
            raise ValueError('Milling answers must be observed percentages in [0,100]')
        self.maximum = maximum
        self.revealed = {}
        self.events = []
        self.closed = False

    def query_batch(self, ids, scores, batch_number):
        if self.closed or len(ids) != len(set(ids)) or set(ids) & set(self.revealed):
            raise ValueError('Closed replay or repeated query')
        if not set(ids).issubset(self.__key) or len(self.revealed) + len(ids) > self.maximum:
            raise ValueError('Unknown query or budget exceeded')
        if len(scores) != len(ids):
            raise ValueError('One pre-reveal score per selection is required')
        before = list(self.revealed)
        answers = {i: self.__key[i] for i in ids}
        self.events.append({'batch': batch_number, 'selected_ids': list(ids),
                            'selection_scores': [float(x) for x in scores],
                            'revealed_ids_before': before, 'revealed_hr_pct': answers,
                            'remaining_budget': self.maximum-len(before)-len(ids)})
        self.revealed.update(answers)
        return answers

    def finalize(self, shortlist):
        if self.closed or len(shortlist) != len(set(shortlist)) or not set(shortlist).issubset(self.revealed):
            raise ValueError('Final shortlist must contain tested entries only')
        if len(shortlist) != math.ceil(.1*len(self.__key)) or len(self.revealed) != self.maximum:
            raise ValueError('Wrong shortlist size or unspent testing budget')
        self.closed = True
        return {'events': self.events, 'shortlist': list(shortlist),
                'tested_ids': list(self.revealed), 'budget': self.maximum}


@dataclass(frozen=True)
class Policy:
    name: str
    kind: str
    uncertainty_fraction: float = .5
    shrinkage: float = 4.
    update: bool = True
    seed: int = 0


def rank_scores(frame, mean):
    return dict(zip(frame.row_id, frame.field_yield.to_numpy() * np.asarray(mean) / 100))


def first_batch(frame, mean, sd, count, uncertainty_fraction):
    ids = frame.row_id.tolist()
    mu = rank_scores(frame, mean)
    if uncertainty_fraction == 0:
        chosen = ordered_ids(ids, mu)[:count]
        return chosen, [mu[i] for i in chosen]
    exploit_count = max(1, int(math.floor(count * (1-uncertainty_fraction))))
    chosen = ordered_ids(ids, mu)[:exploit_count]
    scores = [mu[i] for i in chosen]
    k = math.ceil(.1 * len(ids))
    threshold = sorted(mu.values(), reverse=True)[k-1]
    std = frame.field_yield.to_numpy() * np.asarray(sd) / 100
    z = (np.array([mu[i] for i in ids]) - threshold) / np.maximum(std, 1e-12)
    information = std * np.exp(-.5*z*z) / math.sqrt(2*math.pi)
    rel = frame.yield_rel.to_numpy()
    classes = frame.grain_class.to_numpy()
    similarity = (classes[:,None] == classes[None,:]) * np.exp(-.5*((rel[:,None]-rel[None,:])/.25)**2)
    positions = {i:j for j,i in enumerate(ids)}
    while len(chosen) < count:
        available = [i for i in ids if i not in chosen]
        redundancy = similarity[:,[positions[j] for j in chosen]].sum(axis=1)
        values = information/(1+redundancy)
        value = {i:values[positions[i]] for i in available}
        selected = ordered_ids(available, value)[0]
        chosen.append(selected)
        scores.append(value[selected])
    return chosen, scores


def feedback_update(frame, mean, answers, shrinkage):
    baseline = dict(zip(frame.row_id, mean))
    rows = frame.set_index('row_id');chosen=list(answers)
    rel=frame.yield_rel.to_numpy();selected_rel=rows.loc[chosen,'yield_rel'].to_numpy()
    classes=frame.grain_class.to_numpy();selected_classes=rows.loc[chosen,'grain_class'].to_numpy()
    same_class=classes[:,None]==selected_classes[None,:]
    weights=np.where(same_class,1.,.25)*np.exp(-.5*((rel[:,None]-selected_rel[None,:])/.25)**2)
    residual=np.array([answers[i]-baseline[i] for i in chosen])
    correction=(weights@residual)/(shrinkage+weights.sum(axis=1))
    values=np.clip(np.asarray(mean)+correction,0,100)
    updated={i:float(v) if i not in answers else baseline[i] for i,v in zip(frame.row_id,values)}
    return updated


def replay(frame, gate, policy, mean, sd, fraction):
    mean = mean if isinstance(mean,pd.Series) else pd.Series(mean,index=frame.row_id)
    sd = sd if isinstance(sd,pd.Series) else pd.Series(sd,index=frame.row_id)
    frame = public_view(frame)
    if frame.pool_id.nunique() != 1:
        raise ValueError('One decision pool is required')
    ids = frame.row_id.tolist()
    k, maximum = budget(len(frame), fraction)
    if gate.maximum != maximum:
        raise ValueError('Policy and gate budgets differ')
    if len(mean) != len(frame) or len(sd) != len(frame) or set(mean.index) != set(ids) or set(sd.index) != set(ids):
        raise ValueError('Prediction alignment mismatch')
    mean = mean.reindex(ids).to_numpy(float)
    sd = sd.reindex(ids).to_numpy(float)
    if not np.isfinite(mean).all() or not np.isfinite(sd).all() or np.any(sd < 0):
        raise ValueError('Invalid predictions')
    predictions = dict(zip(ids, mean))
    scores = rank_scores(frame, mean)
    rank_changed = 0
    if policy.kind == 'random':
        pool_seed = int(hashlib.sha256(str(frame.pool_id.iloc[0]).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(np.random.SeedSequence([policy.seed, pool_seed]))
        random_scores = dict(zip(ids, rng.random(len(ids))))
        selected = ordered_ids(ids, random_scores)[:maximum]
        gate.query_batch(selected, [random_scores[i] for i in selected], 1)
    elif policy.kind == 'static' or maximum == 1:
        selected = ordered_ids(ids, scores)[:maximum]
        gate.query_batch(selected, [scores[i] for i in selected], 1)
    elif policy.kind == 'two_batch':
        n_first = math.ceil(maximum/2)
        initial, initial_scores = first_batch(frame, mean, sd, n_first, policy.uncertainty_fraction)
        answers = gate.query_batch(initial, initial_scores, 1)
        untested = [i for i in ids if i not in initial]
        updated = feedback_update(frame, mean, answers, policy.shrinkage) if policy.update else predictions
        next_scores = dict(zip(ids, frame.field_yield.to_numpy()*np.array([updated[i] for i in ids])/100))
        before_rank = ordered_ids(untested, scores)
        after_rank = ordered_ids(untested, next_scores)
        rank_changed = sum(i != j for i, j in zip(before_rank, after_rank))
        second = after_rank[:maximum-n_first]
        gate.query_batch(second, [next_scores[i] for i in second], 2)
    else:
        raise ValueError('Unknown policy')
    yields = frame.set_index('row_id').field_yield.to_dict()
    measured = {i: yields[i]*hr/100 for i, hr in gate.revealed.items()}
    shortlist = ordered_ids(gate.revealed, measured)[:k]
    event = gate.finalize(shortlist)
    event.update({'pool_id': str(frame.pool_id.iloc[0]), 'policy': policy.name,
                  'seed': policy.seed, 'n': len(ids), 'k': k, 'budget_fraction': fraction,
                  'untested_rank_positions_changed': rank_changed})
    return event


def score_event(frame, full_key, event):
    ids = set(frame.row_id)
    tested = event['tested_ids']; selected = event['shortlist']
    k,maximum = budget(len(frame),event['budget_fraction'])
    if (event['n'] != len(frame) or event['k'] != k or event['budget'] != maximum or
        len(tested) != maximum or len(set(tested)) != len(tested) or
        len(selected) != k or len(set(selected)) != k or not set(tested).issubset(ids)):
        raise ValueError('Malformed replay counts or identities')
    if not set(selected).issubset(tested):
        raise ValueError('Untested shortlist entry')
    history = []
    for batch in event['events']:
        if batch['revealed_ids_before'] != history or len(batch['selected_ids']) != len(batch['selection_scores']):
            raise ValueError('Inconsistent query history')
        if set(batch['revealed_hr_pct']) != set(batch['selected_ids']):
            raise ValueError('Inconsistent returned answers')
        history += batch['selected_ids']
        if batch['remaining_budget'] != maximum-len(history):
            raise ValueError('Inconsistent remaining budget')
        for i,v in batch['revealed_hr_pct'].items():
            if float(v) != float(full_key[i]):
                raise ValueError('Queried answer differs from private source key')
    if history != tested:
        raise ValueError('Tested IDs differ from query events')
    yields = frame.set_index('row_id').field_yield.to_dict()
    values = {i: yields[i]*float(full_key[i])/100 for i in yields}
    rank = ordered_ids(values, values)
    k = event['k']
    oracle = rank[:k]
    selected = event['shortlist']
    oracle_mean = float(np.mean([values[i] for i in oracle]))
    selected_mean = float(np.mean([values[i] for i in selected]))
    regret = oracle_mean-selected_mean
    if regret < -1e-8:
        raise ValueError('Selection beats the full-information oracle')
    return {'pool_id': event['pool_id'], 'policy': event['policy'], 'seed': event['seed'],
            'budget_fraction': event['budget_fraction'], 'n': event['n'], 'k': k,
            'tests': len(event['tested_ids']), 'actual_testing_fraction': len(event['tested_ids'])/event['n'],
            'oracle_mean': oracle_mean, 'selected_mean': selected_mean,
            'regret': max(0., regret), 'regret_pct': 100*max(0., regret)/oracle_mean if oracle_mean > 0 else np.nan,
            'top_k_recall': len(set(oracle)&set(selected))/k,
            'best_entry_found': int(rank[0] in selected),
            'top_ten_count': len(set(rank[:10])&set(event['tested_ids'])) if event['n'] >= 10 and event['budget'] >= 10 else np.nan,
            'top_ten_recall': len(set(rank[:10])&set(event['tested_ids']))/10 if event['n'] >= 10 and event['budget'] >= 10 else np.nan,
            'rank_positions_changed': event['untested_rank_positions_changed']}
