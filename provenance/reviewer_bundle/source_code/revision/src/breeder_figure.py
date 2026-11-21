"""Post hoc pool-equal rank display of frozen full-testing reference entries."""
from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
import numpy as np
import pandas as pd
from action_diagnostic import candidates, order, prepare_paths, read, sha, subset

SOURCES=['outputs/run_a/frozen_public_candidates.csv','outputs/run_a/per_pool_results.csv',
         'derived/cohort/all_public_metadata.csv','derived/uruguay/private_outcomes.csv',
         'derived/arkansas/private_outcomes.csv','outputs/run_a/evaluation_freeze.json']
COLORS={'uruguay':'#126B8B','arkansas':'#AF5429'}
LABELS={'uruguay':'Uruguay, 2017–2020','arkansas':'Arkansas, 2017–2025'}

def ecdf(d,column):
    pools=d.pool_id.nunique()
    weights=1/(pools*d.k.to_numpy())
    z=pd.DataFrame({'x':d[column].to_numpy(),'weight':weights})
    sums=z.groupby('x',sort=True).weight.sum()
    end=4. if column=='rank_relative_to_25_percent_query_count' else 1.
    x=np.r_[0,sums.index.to_numpy(),end]
    y=np.r_[0,sums.cumsum().to_numpy(),1.]
    assert abs(float(sums.sum())-1)<1e-12
    return x,y

def describe(d,column,kind,label):
    x,y=ecdf(d,column)
    return pd.DataFrame({'view':kind,'group':label,'x':x,'pool_equal_cumulative_proportion':y,
                         'pools':d.pool_id.nunique(),'reference_entries':len(d)})

def main(source_project,output_dir):
    source,out=prepare_paths(source_project,output_dir)
    before={p:sha(source/p) for p in SOURCES}
    frame=candidates(source)
    all_public=read(source/'derived/cohort/all_public_metadata.csv')
    all_public=all_public[all_public.pool_id.isin(frame.pool_id)]
    field=all_public[all_public.observed_field.astype(str).str.lower().eq('true')].copy()
    field['independent_ascending_rank']=field.groupby('pool_id').field_yield.rank(method='average',pct=True)
    field_idx=field.set_index('row_id')
    field_counts=field.groupby('pool_id').size().to_dict()
    original=read(source/'outputs/run_a/per_pool_results.csv')
    original=original[original.policy.eq('yield')&original.budget_fraction.eq(.25)].set_index('pool_id')
    rows=[]; pools=[]
    for pool,g in frame.groupby('pool_id',sort=True):
        ids=g.row_id.tolist(); n=len(g);k=math.ceil(.1*n);B=math.ceil(.25*n)
        oracle=order(ids,g.set_index('row_id').index_value.to_dict())[:k]
        rank={i:r for r,i in enumerate(order(ids,g.set_index('row_id').field_yield.to_dict()),1)}
        indexed=g.set_index('row_id')
        covered=sum(rank[i]<=B for i in oracle)
        assert abs(covered/k-original.loc[pool,'top_k_recall'])<1e-12
        assert abs(B/n-original.loc[pool,'actual_testing_fraction'])<1e-12
        for i in oracle:
            row=indexed.loc[i]
            original_pct=float(row.yield_rank)
            assert abs(original_pct-float(field_idx.loc[i,'independent_ascending_rank']))<1e-12
            rows.append(dict(row_id=i,pool_id=pool,program=row.program,year=int(row.year),site=row.site,
                             n=n,n_original_observed_field=field_counts[pool],k=k,B25=B,
                             field_yield_rank_eligible=rank[i],eligible_rank_fraction=rank[i]/n,
                             rank_relative_to_25_percent_query_count=rank[i]/B,
                             original_all_field_ascending_average_rank_percentile=original_pct,
                             original_all_field_upper_tail_fraction=1-original_pct,
                             queried_by_yield_at_25_percent=rank[i]<=B))
        pools.append(dict(pool_id=pool,program=g.program.iloc[0],year=int(g.year.iloc[0]),site=g.site.iloc[0],
                          n=n,n_original_observed_field=field_counts[pool],k=k,B25=B,
                          reference_entries_queried=covered,recall=covered/k,actual_testing_fraction=B/n))
    points=pd.DataFrame(rows);points['period']=subset(points)
    pool_data=pd.DataFrame(pools);pool_data['period']=subset(pool_data)
    N=pool_data.groupby('program').size().to_dict()
    points['program_pool_equal_weight']=[1/(N[p]*k) for p,k in zip(points.program,points.k)]
    points.to_csv(out/'reference_entry_field_yield_ranks.csv',index=False)
    pool_data.to_csv(out/'pool_reference_coverage.csv',index=False)
    curves=[]
    for label,d in list(points.groupby('program'))+list(points.groupby('period')):
        curves.append(describe(d,'rank_relative_to_25_percent_query_count','quarter_budget_rank',label))
        curves.append(describe(d,'eligible_rank_fraction','eligible_rank_fraction',label))
        curves.append(describe(d,'original_all_field_upper_tail_fraction','original_all_field_upper_tail_fraction',label))
    for (program,year),d in points.groupby(['program','year']):
        curves.append(describe(d,'rank_relative_to_25_percent_query_count','quarter_budget_rank_year',f'{program} {year}'))
    pd.concat(curves,ignore_index=True).to_csv(out/'pool_equal_ecdf_data.csv',index=False)
    summaries=[]
    for label,d in list(pool_data.groupby('program'))+list(pool_data.groupby('period')):
        summaries.append(dict(group=label,pools=len(d),years=d.year.nunique(),eligible_entries=int(d.n.sum()),
                              original_observed_field_entries=int(d.n_original_observed_field.sum()),
                              reference_entries=int(d.k.sum()),reference_entries_queried=int(d.reference_entries_queried.sum()),
                              mean_pool_recall=d.recall.mean(),
                              pooled_entry_recall=d.reference_entries_queried.sum()/d.k.sum(),
                              mean_actual_testing_fraction=d.actual_testing_fraction.mean()))
    pd.DataFrame(summaries).to_csv(out/'rank_display_summary.csv',index=False)
    os.environ['MPLCONFIGDIR']=str(out/'.matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9.5,'axes.spines.top':False,'axes.spines.right':False,
                         'svg.hashsalt':'rice-milling-revision-breeder-v1','savefig.facecolor':'white'})
    fig,axes=plt.subplots(1,2,figsize=(10,4.25),sharex=True,sharey=True)
    for ax,program in zip(axes,['uruguay','arkansas']):
        p=points[points.program.eq(program)]; q=pool_data[pool_data.program.eq(program)]
        for year,d in p.groupby('year'):
            x,y=ecdf(d,'rank_relative_to_25_percent_query_count')
            ax.step(x,100*y,where='post',color=COLORS[program],alpha=.24,lw=.8)
        x,y=ecdf(p,'rank_relative_to_25_percent_query_count')
        ax.step(x,100*y,where='post',color=COLORS[program],lw=2.3,label='Equal-pool mean')
        ax.axvline(1,color='#555555',ls='--',lw=1)
        ax.plot(1,100*q.recall.mean(),'o',color=COLORS[program],ms=5)
        ax.annotate(f'{100*q.recall.mean():.2f}% at quarter budget',xy=(1,100*q.recall.mean()),
                    xytext=(1.32,100*q.recall.mean()-17),arrowprops={'arrowstyle':'-','lw':.7},fontsize=9)
        ax.set_title(f'{LABELS[program]}\n{len(q)} pools · {int(q.n.sum()):,} eligible entries',loc='left',fontweight='bold')
        ax.set_xlim(0,4);ax.set_ylim(0,102);ax.set_xticks([0,1,2,3,4]);ax.set_yticks([0,25,50,75,100])
        ax.grid(axis='y',alpha=.15)
        ax.set_xlabel('Field-yield rank ÷ quarter-budget query count')
    axes[0].set_ylabel('Full-testing top-decile entries recovered (%)')
    fig.text(.5,.014,'Dashed line: exact integer-rounded 25% budget. Thin curves: individual harvest years; no uncertainty bands.',ha='center',fontsize=8.5)
    fig.tight_layout(rect=(0,.045,1,1))
    for ext in ['png','svg','pdf']:
        kw={'metadata':{'Date':None}} if ext=='svg' else ({'metadata':{'CreationDate':None,'ModDate':None}} if ext=='pdf' else {})
        fig.savefig(out/f'breeder_field_yield_ranks.{ext}',dpi=300,**kw)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4.0),sharey=True)
    for program,d in points.groupby('program'):
        for ax,col in zip(axes,['eligible_rank_fraction','original_all_field_upper_tail_fraction']):
            x,y=ecdf(d,col);ax.step(100*x,100*y,where='post',lw=2,color=COLORS[program],label=LABELS[program])
    axes[0].set_title('Eligible observed-answer denominator',loc='left',fontweight='bold')
    axes[0].set_xlabel('Descending field-yield rank / eligible count (%)')
    axes[1].set_title('Original all-field feature denominator',loc='left',fontweight='bold')
    axes[1].set_xlabel('100 × (1 − ascending average-rank percentile)')
    axes[0].set_ylabel('Full-testing top-decile entries accumulated (%)')
    for ax in axes:
        ax.set_xlim(0,100);ax.set_ylim(0,102);ax.grid(axis='y',alpha=.15)
        ax.set_xticks([0,25,50,75,100]);ax.legend(frameon=False,loc='lower right',fontsize=8)
    fig.text(.5,.015,'Rank denominators differ when milling answers are missing. Average ties in the original feature do not define query order.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.045,1,1))
    fig.savefig(out/'breeder_denominator_comparison.png',dpi=300)
    fig.savefig(out/'breeder_denominator_comparison.svg',metadata={'Date':None});plt.close(fig)
    fig,ax=plt.subplots(figsize=(6.5,4.0))
    ar=points[points.program.eq('arkansas')]
    for label,d in ar.groupby('period'):
        x,y=ecdf(d,'rank_relative_to_25_percent_query_count')
        ax.step(x,100*y,where='post',lw=2,label=f'{label}: {d.pool_id.nunique()} pools')
    ax.axvline(1,color='#555555',ls='--',lw=1)
    ax.set(xlim=(0,4),ylim=(0,102),xlabel='Field-yield rank ÷ quarter-budget query count',
           ylabel='Full-testing top-decile entries recovered (%)')
    ax.set_title('Arkansas source-coverage periods — descriptive',loc='left',fontweight='bold')
    ax.legend(frameon=False,loc='lower right');ax.grid(axis='y',alpha=.15)
    fig.tight_layout();fig.savefig(out/'breeder_arkansas_periods.png',dpi=300)
    fig.savefig(out/'breeder_arkansas_periods.svg',metadata={'Date':None});plt.close(fig)
    for p,h in before.items():assert sha(source/p)==h
    manifest={'status':'pass','scope':'post_hoc_prespecified_descriptive_display','source_inputs':before,
              'code_sha256':sha(Path(__file__)),'helper_code_sha256':sha(Path(__file__).with_name('action_diagnostic.py')),
              'pools':len(pool_data),'eligible_entries':len(frame),'reference_entries':len(points),
              'pool_equal_curves_not_candidate_equal':True,'original_all_field_rank_independently_checked':True,
              'quarter_budget_curve_matches_archived_recall':True,'no_uncertainty_intervals':True,
              'outputs':{p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.suffix in ['.csv','.png','.svg','.pdf']}}
    plan=Path(__file__).resolve().parents[1]/'protocol/action_and_breeder_diagnostic_plan.md'
    if plan.exists():manifest['diagnostic_plan_sha256']=sha(plan)
    (out/'verification.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps({k:manifest[k] for k in ['status','pools','eligible_entries','reference_entries']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-project',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();main(args.source_project,args.output_dir)
