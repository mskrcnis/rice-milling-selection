"""Presentation of already frozen, independently checked contrasts; no policy fitting."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['svg.hashsalt']='rice-milling-report-v1'
matplotlib.rcParams['svg.fonttype']='none'
import matplotlib.pyplot as plt


def paired_figure(out):
    out=Path(out)
    primary=pd.read_csv(out/'tables/paired_summary.csv')
    secondary=pd.read_csv(out/'tables/secondary_adaptive_paired.csv')
    primary=primary[primary.budget_fraction.eq(.25)&primary.metric.eq('regret_pct')].copy()
    secondary=secondary[secondary.budget_fraction.eq(.25)&secondary.metric.eq('regret_pct')].copy()
    primary['policy']='two_batch'
    rows=pd.concat([primary,secondary],ignore_index=True)
    comparisons=[('two_batch','mixed','Primary: vs locked static'),('two_batch','no_update','Primary: feedback vs no update'),
                 ('adaptive_grid','mixed','Secondary: vs locked static'),('adaptive_grid','adaptive_grid_no_update','Secondary: feedback vs no update')]
    fig,axes=plt.subplots(1,2,figsize=(12.2,4.7),sharey=True,sharex=True)
    for ax,stratum,title in zip(axes,['uruguay_temporal','arkansas_external'],['Uruguay temporal evaluation','Arkansas independent program']):
        for y,(policy,comparator,label) in enumerate(comparisons):
            part=rows[rows.stratum.eq(stratum)&rows.policy.eq(policy)&rows.comparator.eq(comparator)]
            if len(part)!=1:raise ValueError('Missing or duplicate paired effect')
            r=part.iloc[0];color='#176f7b' if policy=='two_batch' else '#a75e21'
            ax.hlines(y,r.lower,r.upper,color=color,lw=2)
            ax.plot(r.difference,y,'o',color=color,ms=6)
            ax.annotate(f'{r.difference:+.3f} [{r.lower:+.3f}, {r.upper:+.3f}]',(r.difference,y),
                        xytext=(0,9),textcoords='offset points',ha='center',fontsize=8,color=color)
        ax.axvline(0,color='#777777',lw=.8,ls='--')
        ax.set_title(f'{title}\n{int(r.pools)} pools; {int(r.years)} harvest years',fontsize=11)
        ax.set_xlabel('Adaptive minus comparator regret (percentage points)\nNegative favors adaptive',fontsize=9)
        ax.spines[['top','right','left']].set_visible(False)
        ax.tick_params(axis='y',length=0)
        ax.set_ylim(3.65,-.65)
        ax.margins(x=.3)
    axes[0].set_yticks(range(4),[x[2] for x in comparisons],fontsize=9)
    fig.suptitle('25% design budget: effects and descriptive 95% harvest-year intervals',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.93))
    target=out/'figures';target.mkdir(exist_ok=True)
    fig.savefig(target/'paired_effects.png',dpi=300)
    fig.savefig(target/'paired_effects.svg',metadata={'Date':None})
    plt.close(fig)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True)
    paired_figure(parser.parse_args().out)


def secondary_budget_figure(out):
    out=Path(out)
    data=pd.read_csv(out/'tables/secondary_adaptive_paired.csv')
    data=data[data.comparator.eq('adaptive_grid_no_update')&data.metric.eq('regret_pct')]
    fig,axes=plt.subplots(1,2,figsize=(10.7,4.5),sharey=True)
    for ax,stratum,label in zip(axes,['uruguay_temporal','arkansas_external'],['Uruguay: 188 pools, 4 years','Arkansas: 62 pools, 9 years']):
        d=data[data.stratum.eq(stratum)].sort_values('budget_fraction')
        ax.plot(d.budget_fraction*100,d.difference,marker='o',color='#a75e21',lw=1.5)
        ax.fill_between(d.budget_fraction*100,d.lower,d.upper,color='#a75e21',alpha=.17)
        ax.axhline(0,color='#777777',ls='--',lw=.8)
        ax.axvline(25,color='#999999',ls=':',lw=.8)
        ax.set_title(label);ax.set_xlabel('Design testing budget (%)');ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Secondary adaptive minus no-update regret (pp)\nNegative favors paid feedback')
    fig.suptitle('Secondary tree policy: feedback effects across all fixed budgets',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.94))
    fig.savefig(out/'figures/secondary_feedback_budget.png',dpi=300)
    fig.savefig(out/'figures/secondary_feedback_budget.svg',metadata={'Date':None})
    plt.close(fig)
