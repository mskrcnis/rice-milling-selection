import numpy as np
import pandas as pd
import pytest
from src.replay import public_view, QueryGate, Policy, budget, replay, score_event
from src.models import RecoveryModel


def example(n=20,year=2020):
    return public_view(pd.DataFrame({'row_id':[f'r{i:03d}' for i in range(n)],
        'pool_id':'p1','candidate_id':[f'c{i}' for i in range(n)],'site':'s1',
        'year':year,'field_yield':np.linspace(4000,9000,n),
        'grain_class':['long','medium']*(n//2)+(['long'] if n%2 else [])}))


def test_forbidden_features_not_public():
    frame=example();frame['hr_pct']=60;frame['total_milling_pct']=70
    frame['future_cultivar_mean']=55;frame['chalkiness']=3
    view=public_view(frame)
    assert not set(['hr_pct','total_milling_pct','future_cultivar_mean','chalkiness'])&set(view)


def test_duplicates_rejected():
    frame=example();frame.loc[1,'candidate_id']=frame.loc[0,'candidate_id']
    with pytest.raises(ValueError):public_view(frame)


@pytest.mark.parametrize('n,fraction,expected',[(100,.25,(10,25)),(11,.1,(2,2)),(1,.1,(1,1)),(31,1,(4,31))])
def test_budget_rounding(n,fraction,expected):
    assert budget(n,fraction)==expected


def test_missing_answers_never_imputed():
    with pytest.raises(ValueError):QueryGate({'a':float('nan')},['a'],1)


def test_repeated_or_excess_queries_rejected():
    gate=QueryGate({'a':60,'b':55},['a','b'],1)
    gate.query_batch(['a'],[1],1)
    with pytest.raises(ValueError):gate.query_batch(['a'],[1],2)
    with pytest.raises(ValueError):gate.query_batch(['b'],[1],2)


def test_untested_shortlist_rejected():
    gate=QueryGate({'a':60,'b':55},['a','b'],1)
    gate.query_batch(['a'],[1],1)
    with pytest.raises(ValueError):gate.finalize(['b'])


def test_identical_first_batch_and_unrevealed_invariance():
    frame=example();ids=frame.row_id.tolist();key=dict(zip(ids,np.linspace(30,80,len(ids))))
    mean=np.full(len(ids),60.);sd=np.full(len(ids),10.)
    adaptive=replay(frame,QueryGate(key,ids,5),Policy('adaptive','two_batch'),mean,sd,.25)
    ablation=replay(frame,QueryGate(key,ids,5),Policy('no_update','two_batch',update=False),mean,sd,.25)
    assert adaptive['events'][0]['selected_ids']==ablation['events'][0]['selected_ids']
    untested=set(ids)-set(adaptive['tested_ids'])
    changed=key.copy()
    for i in untested:changed[i]=100-key[i]
    repeated=replay(frame,QueryGate(changed,ids,5),Policy('adaptive','two_batch'),mean,sd,.25)
    assert repeated==adaptive
    first=set(adaptive['events'][0]['selected_ids'])
    for i in set(ids)-first:changed[i]=100-key[i]
    check=replay(frame,QueryGate(changed,ids,5),Policy('adaptive','two_batch'),mean,sd,.25)
    assert check['events'][0]==adaptive['events'][0]


@pytest.mark.parametrize('kind',['static','two_batch','random'])
def test_full_budget_oracle(kind):
    frame=example();ids=frame.row_id.tolist();key={i:60. for i in ids}
    event=replay(frame,QueryGate(key,ids,len(ids)),Policy(kind,kind),np.full(len(ids),60.),np.full(len(ids),10.),1.)
    score=score_event(frame,key,event)
    assert score['regret']==0 and score['top_k_recall']==1 and score['best_entry_found']==1


def test_exact_whole_kernel_denominator():
    frame=example(2);ids=frame.row_id.tolist();key={ids[0]:50.,ids[1]:25.}
    event=replay(frame,QueryGate(key,ids,2),Policy('yield','static'),np.ones(2)*100,np.ones(2),1)
    score=score_event(frame,key,event)
    assert score['oracle_mean']==2250


def test_models_reject_future_training_and_accept_new_site():
    frame=example();key=dict(zip(frame.row_id,np.linspace(40,70,len(frame))))
    model=RecoveryModel('ridge',10).fit(frame,key)
    with pytest.raises(ValueError):model.predict(frame)
    future=frame.copy();future['year']=2021;future['row_id']='new_'+future.row_id;future['site']='unseen'
    mean,sd=model.predict(future)
    assert np.isfinite(mean).all() and np.isfinite(sd).all()


def test_policy_one_candidate_fallback():
    frame=example(1);ids=frame.row_id.tolist();key={ids[0]:60}
    event=replay(frame,QueryGate(key,ids,1),Policy('a','two_batch'),[50],[10],.25)
    assert len(event['events'])==1


def test_precomputed_public_features_survive_answer_restriction():
    frame=example();subset=frame.iloc[::2].copy()
    view=public_view(subset)
    assert np.array_equal(view.yield_rel,subset.yield_rel)
    assert np.array_equal(view.yield_rank,subset.yield_rank)


def test_prediction_ids_survive_row_permutation():
    frame=example();ids=frame.row_id.tolist();key=dict(zip(ids,np.linspace(30,80,len(ids))))
    mean=pd.Series(np.linspace(40,60,len(ids)),index=ids);sd=pd.Series(10.,index=ids)
    first=replay(frame,QueryGate(key,ids,5),Policy('a','two_batch'),mean,sd,.25)
    shuffled=frame.sample(frac=1,random_state=5)
    second=replay(shuffled,QueryGate(key,ids,5),Policy('a','two_batch'),mean,sd,.25)
    assert first==second


def test_duplicate_shortlist_rejected():
    ids=[str(i) for i in range(20)];gate=QueryGate(dict.fromkeys(ids,60),ids,5)
    gate.query_batch(ids[:5],[1]*5,1)
    with pytest.raises(ValueError):gate.finalize([ids[0],ids[0]])


def test_scorer_rejects_forged_counts():
    frame=example();ids=frame.row_id.tolist();key=dict.fromkeys(ids,60)
    event=replay(frame,QueryGate(key,ids,5),Policy('a','static'),np.ones(20)*60,np.ones(20),.25)
    event['k']=3
    with pytest.raises(ValueError):score_event(frame,key,event)
