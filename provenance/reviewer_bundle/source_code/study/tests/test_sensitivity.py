import copy
import gzip
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common import freeze
from src.replay import QueryGate, Policy, replay, ordered_ids
from src.sensitivity import (KG_HA_PER_BU_AC, output_intervals, paired_coverage,
    product_definition_sensitivity, rank_interval_summary,
    rounding_interval_sensitivity, run_sensitivities)


def example(program='uruguay', n=20, fraction=.25):
    ids=[program+str(i) for i in range(n)]
    y=np.arange(100,100+n,dtype=float)
    frame=pd.DataFrame({'row_id':ids,'candidate_id':ids,'pool_id':program+'_pool',
        'program':program,'site':'synthetic','year':2020,'grain_class':'unknown',
        'stratum':'synthetic','field_yield':y*KG_HA_PER_BU_AC})
    private=pd.DataFrame({'row_id':ids,'pool_id':program+'_pool','hr_percent':60.,
        'matched_plot_count':2,'matched_product_of_means':y*30,
        'output_mean_of_matched_plot_products':y[::-1]*30})
    event=replay(frame,QueryGate(dict.fromkeys(ids,60.),ids,max(2,int(np.ceil(n*fraction)))),
        Policy('yield','static'),np.full(n,60.),np.ones(n),fraction)
    event.update(program=program,year=2020,site='synthetic',stratum='synthetic')
    source=frame.assign(field_yield_bu_ac=y)
    return frame,private,event,source


def test_endpoint_comparison_keeps_query_set_and_reselects():
    frame,private,event,_=example();before=copy.deepcopy(event)
    records=product_definition_sensitivity(frame,private,event)
    assert event==before
    assert len(records)==2 and all(r['status']=='available' for r in records)
    assert all(set(json.loads(r['shortlist_ids'])).issubset(event['tested_ids']) for r in records)
    assert records[0]['shortlist_ids']!=records[1]['shortlist_ids']
    assert records[0]['oracle_ids']!=records[1]['oracle_ids']
    assert records[0]['regret']==0 and records[1]['regret']>0


def test_unmatched_untested_candidate_makes_entire_pool_unavailable():
    frame,private,event,_=example()
    untested=next(i for i in frame.row_id if i not in event['tested_ids'])
    private.loc[private.row_id.eq(untested),'output_mean_of_matched_plot_products']=np.nan
    records=product_definition_sensitivity(frame,private,event)
    assert all(r['status']=='unavailable' for r in records)
    assert all(r['missing_matched_candidates']==1 for r in records)
    assert all(np.isnan(r['regret']) for r in records)


def test_zero_matched_count_never_imputed():
    frame,private,event,_=example();private.loc[0,'matched_plot_count']=0
    assert product_definition_sensitivity(frame,private,event)[0]['status']=='unavailable'


def test_full_budget_each_endpoint_uses_own_oracle():
    frame,private,event,_=example(fraction=1)
    for record in product_definition_sensitivity(frame,private,event):
        assert record['regret']==0 and record['top_k_recall']==1


def test_event_query_answer_validation():
    frame,private,event,_=example();event['events'][0]['revealed_hr_pct'][event['tested_ids'][0]]=61
    with pytest.raises(ValueError,match='Queried answer'):
        product_definition_sensitivity(frame,private,event)


def test_nonoptimal_recorded_shortlist_rejected():
    frame,private,event,_=example();event['shortlist']=event['tested_ids'][-2:]
    with pytest.raises(ValueError,match='Recorded shortlist'):
        product_definition_sensitivity(frame,private,event)


def test_rounding_units_and_physical_bounds():
    lower,upper=output_intervals([100,100],[0,100])
    assert lower[0]==0
    assert upper[0]==pytest.approx(100.5*KG_HA_PER_BU_AC*.005)
    assert upper[1]==pytest.approx(100.5*KG_HA_PER_BU_AC)
    assert lower[1]==pytest.approx(99.5*KG_HA_PER_BU_AC*.995)
    with pytest.raises(ValueError,match='precision'):
        output_intervals([100.1],[60])


def test_interval_ambiguity_stable_and_overlapping():
    stable,_=rank_interval_summary({'a':10,'b':5},{'a':9,'b':4},{'a':11,'b':6},1)
    assert not stable['membership_may_change']
    ambiguous,_=rank_interval_summary({'a':10,'b':9},{'a':8,'b':8},{'a':11,'b':12},1)
    assert ambiguous['membership_may_change'] and ambiguous['ambiguous_boundary_pairs']==1


def test_zero_width_ties_use_frozen_tiebreak():
    points={'a':1.,'b':1.,'c':1.}
    record,ranks=rank_interval_summary(points,points,points,1)
    assert not record['membership_may_change']
    assert all(r['best_possible_rank']==r['worst_possible_rank'] for r in ranks)


def test_rank_bounds_cover_all_endpoint_permutations():
    lower={'a':1.,'b':2.,'c':3.};upper={'a':4.,'b':3.,'c':5.}
    point={i:(lower[i]+upper[i])/2 for i in lower}
    _,ranks=rank_interval_summary(point,lower,upper,1);bounds={r['row_id']:r for r in ranks}
    for choices in itertools.product([0,1],repeat=3):
        values={i:[lower[i],upper[i]][v] for i,v in zip(lower,choices)}
        order=ordered_ids(values,values)
        for i in order:
            assert bounds[i]['best_possible_rank']<=order.index(i)+1<=bounds[i]['worst_possible_rank']


def test_rounding_checks_conversion_and_keeps_event():
    frame,private,event,source=example('arkansas');before=copy.deepcopy(event)
    record,intervals=rounding_interval_sensitivity(frame,private,source,event)
    assert event==before and record['queries_unchanged'] and len(intervals)==len(frame)
    assert record['yield_rounding_halfunit_bu_ac']==.5
    frame.loc[0,'field_yield']*=1.1
    with pytest.raises(ValueError,match='conversion'):
        rounding_interval_sensitivity(frame,private,source,event)


def test_public_paired_coverage_denominator():
    source=pd.DataFrame({'row_id':['a','b','c'],'pool_id':'p',
                         'observed_field':[True,True,False],'paired':[True,False,False]})
    record=paired_coverage(source).iloc[0]
    assert record.paired_fraction_of_field_candidates==.5
    assert record.coverage_stratum=='below_90_percent'
    source.loc[2,'paired']=True
    with pytest.raises(ValueError,match='lacks observed field'):
        paired_coverage(source)


def synthetic_files(tmp_path):
    uy,uk,ue,_=example();ar,ak,ae,source=example('arkansas')
    public=pd.concat([uy,ar],ignore_index=True)
    all_public=public.assign(observed_field=True,paired=True)
    paths={name:tmp_path/(name+'.csv') for name in ['public','all_public','uk','ak','ar_source']}
    for name,frame in [('public',public),('all_public',all_public),('uk',uk),('ak',ak),('ar_source',source)]:
        frame.to_csv(paths[name],index=False)
    events=tmp_path/'events.jsonl.gz'
    with gzip.open(events,'wt') as f:
        for event in [ue,ae]:f.write(json.dumps(event)+'\n')
    locked=tmp_path/'freeze.json';freeze(list(paths.values()),locked,{'test_fixture':'synthetic only'})
    args=[events,paths['public'],paths['all_public'],paths['uk'],paths['ak'],paths['ar_source'],locked]
    return paths,args


def test_freeze_violation_prevents_dataframe_reads(tmp_path,monkeypatch):
    paths,args=synthetic_files(tmp_path)
    paths['uk'].write_text('changed after freeze')
    def forbidden(*args,**kwargs):raise AssertionError('Read before freeze verification')
    monkeypatch.setattr(pd,'read_csv',forbidden)
    with pytest.raises(ValueError,match='Frozen input changed'):
        run_sensitivities(*args,tmp_path/'output')


def test_unfrozen_source_rejected_before_dataframe_reads(tmp_path,monkeypatch):
    paths,args=synthetic_files(tmp_path)
    manifest=json.loads(Path(args[-1]).read_text());manifest['files'].pop(str(paths['uk'].resolve()))
    Path(args[-1]).write_text(json.dumps(manifest))
    def forbidden(*args,**kwargs):raise AssertionError('Read before freeze coverage verification')
    monkeypatch.setattr(pd,'read_csv',forbidden)
    with pytest.raises(ValueError,match='absent from run freeze'):
        run_sensitivities(*args,tmp_path/'output')


def test_synthetic_end_to_end_repeatability(tmp_path):
    _,args=synthetic_files(tmp_path)
    first=run_sensitivities(*args,tmp_path/'one')
    second=run_sensitivities(*args,tmp_path/'two')
    assert first['output_hashes']==second['output_hashes']
    assert first['completed_events']==2 and not first['eventlog_was_prefrozen']
    assert not first['query_sets_changed'] and not first['policy_retuning']
    product=pd.read_csv(tmp_path/'one/product_definition_sensitivity.csv')
    assert len(product)==2 and product.status.eq('available').all()
    rounding=pd.read_csv(tmp_path/'one/rounding_interval_sensitivity.csv')
    assert len(rounding)==1
