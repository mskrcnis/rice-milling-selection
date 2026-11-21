"""Generate the decision report and anonymous manuscript from verified results."""
import argparse
import html
import json
import shutil
import re
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from .common import ROOT,digest,write_json,verify_freeze
from .bibliography import escape as basic_escape,create_references
from .report_figures import paired_figure,secondary_budget_figure

STRATA=['uruguay_temporal','arkansas_external']
NAMES={'uruguay_temporal':'Uruguay temporal','arkansas_external':'Arkansas external'}
POLICIES={'yield':'Paddy yield first','mixed':'Locked static mixed model','two_batch':'Primary two batches',
          'no_update':'Primary no update','trees_20':'Static extra trees (leaf 20)',
          'adaptive_grid':'Secondary two batches','adaptive_grid_no_update':'Secondary no update','random':'Random (20 seeds)'}


def clean_prose(text):
    text=re.sub(r'(?<=[A-Za-z])(?=\d)|(?<=\d)(?=[A-Za-z])',' ',str(text))
    text=re.sub(r'(?<=%)(?=[A-Za-z])',' ',text)
    text=re.sub(r'(?<=[A-Za-z])(?=[+-]\d)',' ',text)
    text=re.sub(r'(?<=[A-Za-z]),(?=\d)',', ',text)
    return text.replace('CC 0','CC0')


def escape(text):
    return basic_escape(clean_prose(text))


def sensitivity_text(out,destination):
    product=pd.read_csv(out/'sensitivity/product_definition_sensitivity.csv')
    p=product[product.stratum.eq('uruguay_temporal')&product.policy.eq('two_batch')&product.budget_fraction.eq(.25)]
    if not p.status.eq('available').all():raise ValueError('Unavailable matched products need explicit report revision')
    values=p.groupby('definition').regret_pct.mean()
    rounding=pd.read_csv(out/'sensitivity/rounding_interval_sensitivity.csv')
    r=rounding[rounding.stratum.eq('arkansas_external')&rounding.policy.eq('two_batch')&rounding.budget_fraction.eq(.25)]
    ambiguous=int(r.oracle_membership_may_change.sum());shortlist=int(r.tested_shortlist_membership_may_change.sum())
    pools=pd.read_csv(out/'tables/pool_seed_averages.csv')
    a=pools[pools.stratum.eq('arkansas_external')&pools.policy.eq('two_batch')&pools.budget_fraction.eq(.25)]
    early=a[a.year.lt(2021)];recent=a[a.year.ge(2021)]
    text=(f'All188 Uruguay temporal pools had complete matched-product answers. With the same recorded queried sets, mean primary relative regret was'
          f'{p.primary_regret_pct.mean():.6f}% under the original product-of-means,'
          f'{values["matched_product_of_means"]:.6f}% under the matched-observation product-of-means, and'
          f'{values["output_mean_of_matched_plot_products"]:.6f}% under mean matched-plot products. '
          'The original-to-matched-means contrast reflects restricting to paired plot observations; the matched-means-to-mean-products contrast isolates aggregation on the same observations. Every candidate had matched answers, although some original replicates lacked one trait. The primary feedback/static equality persisted under the alternative endpoints. These sensitivities reselect the measured shortlist and oracle; they do not refit policies or claim unchanged biological ranks. '
          f'In Arkansas, rounding envelopes allowed the oracle top-k membership to change in{ambiguous}/{len(r)} primary pools, and the tested shortlist in{shortlist}/{len(r)}. '
          'These are conservative source-precision envelopes, not sampling intervals. '
          f'Primary relative regret was{early.regret_pct.mean():.6f}% in{len(early)} historical2017-2020 pools and'
          f'{recent.regret_pct.mean():.6f}% in{len(recent)} contemporaneous2021-2025 pools. Paid feedback added no primary-budget value in either subset. '
          'Calendar, location, cultivar and answer-coverage differences are confounded, so this contrast cannot be attributed solely to missingness. '
          'Whole-pool stage/class-composition, measured-answer coverage, leave-one-site/year, new/returning-line recovery and prediction-calibration diagnostics are supplied as descriptive companion tables.')
    d=pd.read_csv(out/'diagnostics/prediction_metrics.csv')
    u=d[d.stratum.eq('uruguay_temporal')&d.model.eq('mixed')].iloc[0]
    a=d[d.stratum.eq('arkansas_external')&d.model.eq('mixed')].iloc[0]
    text += (f' The locked mixed recovery model had candidate-weighted RMSE{u.rmse_pp:.3f} percentage points in Uruguay and{a.rmse_pp:.3f} in Arkansas,'
             f' with prediction-minus-observation bias{u.bias_predicted_minus_observed_pp:+.3f} and{a.bias_predicted_minus_observed_pp:+.3f} points. Good ranking utility does not imply calibrated recovery forecasts.')
    (destination/'sensitivity_results.tex').write_text(escape(text)+'\n')
    return clean_prose(text)


def cell(value,digits=3):
    return 'NA' if pd.isna(value) else f'{float(value):.{digits}f}'


def latex_table(headers,rows,caption,label):
    lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{'+escape(caption)+'}',r'\label{'+label+'}',
           r'\begin{tabular}{'+'l'*len(headers)+'}',r'\toprule',' & '.join(escape(x) for x in headers)+r' \\',r'\midrule']
    lines += [' & '.join(escape(x) for x in row)+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    return '\n'.join(lines)


def build_report(out,compile_pdf=True):
    out=Path(out).resolve()
    verify_freeze(out/'evaluation_freeze.json')
    verification=json.loads((out/'verification/verification_summary.json').read_text())
    tables_verified=json.loads((out/'verification/table_verification.json').read_text())
    if verification['status']!='passed' or tables_verified['status']!='passed':raise ValueError('Verification did not pass')
    paths=[out/'tables'/f for f in ['policy_summary.csv','paired_summary.csv','secondary_adaptive_paired.csv','feedback_action_changes.csv','contemporaneous_external_sensitivity.csv']]
    summary,paired,secondary,actions,modern=[pd.read_csv(p) for p in paths]
    metadata=json.loads((ROOT/'derived/cohort/metadata_summary.json').read_text())
    def policy(stratum,name='yield',fraction=.25):
        return summary[summary.stratum.eq(stratum)&summary.policy.eq(name)&summary.budget_fraction.eq(fraction)].iloc[0]
    def effect(stratum,comparator,secondary_policy=False):
        d=secondary if secondary_policy else paired
        return d[d.stratum.eq(stratum)&d.comparator.eq(comparator)&d.budget_fraction.eq(.25)&d.metric.eq('regret_pct')].iloc[0]
    ur=policy('uruguay_temporal');ar=policy('arkansas_external')
    sec_ur=effect('uruguay_temporal','adaptive_grid_no_update',True)
    sec_ar=effect('arkansas_external','mixed',True)
    targets={s:summary[summary.stratum.eq(s)&summary.policy.eq('yield')&summary.top_k_recall.ge(.9)].budget_fraction.min() for s in STRATA}
    abstract=(f'We replayed a postharvest decision in which field yield was known and only tested entries could enter a final shortlist. '
        f'Historical Uruguay records supported rolling development in2012-2016, followed by188 temporal pools (7,568 candidate entries,2017-2020) and62 unseen Arkansas pools (1,611 entries,2017-2025). '
        f'The shortlist contained ceil(10% of observed-answer candidates); the primary testing budget was ceil(25%). '
        f'At that budget, yield-first recovered{100*ur.top_k_recall:.2f}% of oracle top entries in Uruguay and{100*ar.top_k_recall:.2f}% in Arkansas, '
        f'with mean relative regret{ur.regret_pct:.3f}% and{ar.regret_pct:.3f}%, respectively. '
        f'The primary two-batch policy matched its development-selected static benchmark and identical-first-batch no-update ablation in every primary pool. '
        f'A development-selected tree secondary increased Uruguay regret by{sec_ur.difference:.4f} percentage points versus its own no-update policy and gave no feedback gain in Arkansas at25% testing. '
        f'Yield-first first attained90% mean recall at the measured{100*targets[STRATA[0]]:.0f}% and{100*targets[STRATA[1]]:.0f}% budget scenarios, respectively. '
        f'These data support a strong static rule in Uruguay and a substantial transfer gap in top-entry recovery, with no feedback benefit at the primary25% budget; sparse secondary effects at other budgets require separate interpretation. '
        f'Inference is conditional on measured answers and a source-basis output index; Uruguay moisture harmonization and prospective laboratory utility remain unresolved.')
    main=(f'At the25% design budget, paddy-yield-first achieved{100*ur.top_k_recall:.2f}% oracle top-k recovery across188 Uruguay pools '
          f'and{100*ar.top_k_recall:.2f}% across62 unseen Arkansas pools. Relative shortlist regret was{ur.regret_pct:.3f}% and{ar.regret_pct:.3f}%. '
          'The primary adaptive, matched no-update and locked static policies had equal primary-budget regret in every pool. '
          'This is an observed identity of the tested policies, not an equivalence claim about all adaptive methods.')
    feedback=(f'At the25% budget, the secondary tree policy had{policy(STRATA[0],"adaptive_grid").regret_pct:.3f}% regret in Uruguay versus'
              f'{policy(STRATA[0],"adaptive_grid_no_update").regret_pct:.3f}% without feedback: a difference of{sec_ur.difference:+.4f} percentage points '
              f'(descriptive95% year-block interval{sec_ur.lower:+.4f} to{sec_ur.upper:+.4f}). '
              f'Its Arkansas difference versus the locked static benchmark was{sec_ar.difference:+.4f} points '
              f'({sec_ar.lower:+.4f} to{sec_ar.upper:+.4f}), but the static tree and no-update policies shared this gain. '
              'The external difference therefore provides no evidence of value from paid feedback.')
    b40=secondary[secondary.stratum.eq('arkansas_external')&secondary.comparator.eq('adaptive_grid_no_update')&secondary.metric.eq('regret_pct')&secondary.budget_fraction.eq(.4)].iloc[0]
    pool_results=pd.read_csv(out/'tables/pool_seed_averages.csv')
    q=pool_results[pool_results.stratum.eq('arkansas_external')&pool_results.budget_fraction.eq(.4)].pivot(index='pool_id',columns='policy',values='regret_pct')
    changed=int(((q.adaptive_grid-q.adaptive_grid_no_update).abs()>1e-10).sum())
    feedback += (f' At the secondary40% budget in Arkansas, tree-based feedback reduced relative regret by{-b40.difference:.4f} percentage points'
                 f' (adaptive-minus-no-update interval{b40.lower:+.4f} to{b40.upper:+.4f}), entirely from{changed}/62 pools. This budget-specific secondary finding does not replace the primary-budget result or establish a stable adaptive benefit.')
    budget=(f'The first measured budget attaining at least90% equal-pool mean oracle recall was{100*targets[STRATA[0]]:.0f}% in Uruguay '
            f'and{100*targets[STRATA[1]]:.0f}% in Arkansas. This threshold is illustrative, was fixed before evaluation, and is assessed on the measured grid only. '
            f'At the25% design budget, mean actual fractions were{100*ur.actual_testing_fraction:.2f}% and{100*ar.actual_testing_fraction:.2f}% because entry counts are rounded upward. '
            'These are entry-testing scenarios, not measured assay, time or financial savings.')
    limits=('The target is conditional on candidates with observed milling answers. Uruguay temporal coverage was7,568/7,596 observed-field entries. '
            'Arkansas historical2017-2020 coverage was447/551 across19 pools; contemporary2021-2025 coverage was1,164/1,164 across43 pools. '
            'The two sources do not have a verified common moisture basis. Uruguay results use a source-basis output index supported by same-program recovery definitions; '
            'within-pool relative regret cancels common scaling but not unrecorded candidate-specific measurement differences. '
            'Year-block intervals are descriptive, with only four Uruguay evaluation years and recurring sites across years.')
    repeat_path=ROOT/'audit/clean_repeat_verification.json'
    repeat=json.loads(repeat_path.read_text()) if repeat_path.exists() else None
    repeated=bool(repeat and repeat.get('status')=='pass')
    verification_text=(f'The independent reconstruction checked{verification["event_count"]:,} policy events across{verification["pool_count"]} primary and secondary pools, '
                       f'{verification["query_revelations_checked"]:,} paid entry reveals, and{verification["full_budget_events_checked"]:,} full-budget events. '
                       f'All full-budget policies had zero regret and complete oracle recovery. Six primary aggregate tables, including descriptive year-block intervals, '
                       'were independently reconstructed with absolute tolerance0.000000001. All11 aggregate tables were also regenerated from the independently reconstructed event metrics. The source extraction and targeted test audits are retained. '
                       +('Two fresh historical fits/replays reproduced all compared scientific artifacts byte-for-byte. ' if repeated else 'The second fresh-fit reproducibility check is still in progress. '))
    readiness=('A coherent negative-result computational manuscript is supported. It is not submission-ready: the authors must review the agronomic interpretation, '
               'resolve or explicitly retain the archive-wide Uruguay unit assumption, and supply authorship, affiliations, funding and conflict declarations. '
               'No journal submission, new laboratory experiment or prospective operational validation has occurred. No claim of a first sequential-selection method is made.')
    policy_rows=[]
    for s in STRATA:
        for name,label in POLICIES.items():
            r=policy(s,name)
            policy_rows.append([NAMES[s],label,cell(r.regret_pct),cell(100*r.top_k_recall,2),cell(100*r.best_entry_found,2)])
    comparison_rows=[]
    for s in STRATA:
        for is_secondary,comparator,label in [(False,'mixed','Primary vs static'),(False,'no_update','Primary feedback'),
                                              (True,'mixed','Secondary vs static'),(True,'adaptive_grid_no_update','Secondary feedback')]:
            r=effect(s,comparator,is_secondary)
            comparison_rows.append([NAMES[s],label,cell(r.difference,4),f'[{cell(r.lower,4)}, {cell(r.upper,4)}]'])
    budget_rows=[]
    for f in [.1,.2,.25,.4,.6,1.]:
        u=policy(STRATA[0],fraction=f);a=policy(STRATA[1],fraction=f)
        budget_rows.append([f'{100*f:.0f}',cell(100*u.top_k_recall,2),cell(u.regret_pct),cell(100*a.top_k_recall,2),cell(a.regret_pct)])
    cohort_rows=[]
    for s,label,period in [('initial_training','Initial Uruguay fitting','1997-2011'),('development','Rolling Uruguay development','2012-2016'),
                           ('uruguay_temporal','Uruguay temporal evaluation','2017-2020'),('arkansas_external','Arkansas unseen evaluation','2017-2025'),
                           ('arkansas_compromised','Arkansas compromised','2022,2024,2025'),('arkansas_previously_seen','Arkansas previously seen','2025')]:
        m=metadata[s];cohort_rows.append([label,period,str(m['pools']),f"{m['rows']:,}",str(m['sites'])])
    report=out/'report';report.mkdir(exist_ok=True)
    sections=[('Decision and strongest finding',main),('Value of feedback',feedback),('How much testing',budget),
              ('Evidence boundary',limits),('Verification',verification_text),('Submission-readiness assessment',readiness)]
    style='body{font:17px/1.55 Georgia,serif;color:#18333d;max-width:1080px;margin:48px auto;padding:0 28px}h1,h2{font-family:Arial,sans-serif}h1{font-size:30px}h2{margin-top:30px;font-size:21px}table{border-collapse:collapse;font:14px/1.4 Arial,sans-serif;width:100%;margin:20px 0}th,td{border-bottom:1px solid #c4d1d5;text-align:left;padding:7px}th{background:#edf3f3}img{max-width:100%}a{color:#176f7b}code{font-size:13px}footer{font-size:13px;border-top:1px solid #ccd;padding-top:15px}'
    page=['<!doctype html><html lang="en"><meta charset="utf-8"><title>Rice milling selection: decision report</title><style>'+style+'</style><body>',
          '<h1>Which rice lines should be sent for milling tests?</h1><p>Executed retrospective study | 20 November 2026 | two public source programs</p>']
    for heading,text in sections:page += ['<h2>'+heading+'</h2><p>'+html.escape(clean_prose(text))+'</p>']
    for heading,headers,rows in [('Primary-budget results',['Evaluation','Policy','Regret (%)','Top-k recall (%)','Best entry found (%)'],policy_rows),
                                 ('Paired effects',['Evaluation','Contrast','Difference (pp)','Descriptive 95% interval'],comparison_rows),
                                 ('Yield-first budget curve',['Design budget (%)','Uruguay recall (%)','Uruguay regret (%)','Arkansas recall (%)','Arkansas regret (%)'],budget_rows)]:
        page += ['<h2>'+heading+'</h2>',pd.DataFrame(rows,columns=headers).to_html(index=False,escape=True)]
        if heading=='Paired effects':
            page += ['<p>Adaptive minus comparator relative regret; negative values favor adaptive. All contrasts in this table use the 25% design budget.</p>']
    page += ['<h2>Figures and reproducibility</h2><img src="../figures/paired_effects.png" alt="Paired effects with descriptive year intervals">',
             '<p><a href="../manuscript/rice_milling_selection.pdf">Complete anonymous manuscript</a> | <a href="../figures/policy_budget_top_k_recall.svg">Policy recovery curves (SVG)</a> | <a href="../figures/policy_budget_regret_pct.svg">Policy regret curves (SVG)</a></p>',
             '<pre><code>./run.sh --offline --output outputs/reproduction</code></pre>',
             '<p>Run from /home/mskr/Research/Agro/Precision Farming/paper1_rice_milling_selection. Source provenance, protocol and full result tables are available in the project directory.</p>',
             '<footer>Primary evidence: <a href="https://doi.org/10.5061/dryad.x69p8czn8">Uruguay Dryad dataset</a>; <a href="https://zenodo.org/records/8222856">checksum-exact mirror</a>; <a href="https://uaex.uada.edu/farm-ranch/crops-commercial-horticulture/rice/">official Arkansas rice archive</a>. Source reviewers saw outcomes; policy development was outcome-separated until the first freeze.</footer></body></html>']
    (report/'decision_report.html').write_text('\n'.join(page))
    paired_figure(out)
    secondary_budget_figure(out)
    destination=out/'manuscript';destination.mkdir(exist_ok=True)
    create_references(destination)
    draft=(ROOT/'manuscript/methods_draft.tex').read_text()
    intro_methods,limitations=draft.split(r'\section{Discussion: scope and limitations}',1)
    tex=[r'\documentclass[11pt,a4paper]{article}',r'\usepackage[margin=24mm]{geometry}',r'\usepackage[T1]{fontenc}',
         r'\usepackage[utf8]{inputenc}',r'\usepackage{lmodern,microtype,amsmath,booktabs,graphicx,float,placeins}',
         r'\usepackage[colorlinks=true,linkcolor=teal,citecolor=teal,urlcolor=teal]{hyperref}',r'\usepackage{xcolor}',
         r'\setlength{\parskip}{4pt}',r'\setlength{\parindent}{0pt}',r'\setcounter{topnumber}{3}',r'\renewcommand{\topfraction}{0.9}',
         r'\title{Which rice lines should be sent for milling tests?\\Budget-constrained selection across historical rice trials}',
         r'\author{Anonymous computational draft for author review}',r'\date{20 November 2026}',r'\begin{document}',r'\maketitle',
         r'\begin{abstract}',escape(abstract),r'\end{abstract}',r'\noindent\textbf{Keywords:} rice; milling recovery; selective phenotyping; sequential testing; retrospective replay; transfer.',intro_methods,
         r'\section{Results}',r'\subsection{Cohort and development baseline}',
         escape('The91,636 Uruguay source rows contained80,805 source records with jointly observed yield and head recovery (not a count of distinct physical plots). Candidate aggregation produced38,480 candidate groups;34,423 eligible candidates in915 pools met the measured-answer and minimum-panel rules. Final historical fitting used26,855 candidate entries in727 pools through2016. Arkansas screening covered final annual reports2011-2025 and retained2,583 source candidate rows in97 pools; the primary transfer cohort contained62 unseen, noncompromised pools. Table1 summarizes the principal partitions.'),
         latex_table(['Partition','Harvest years','Pools','Entries','Site labels'],cohort_rows,'Cohort partitions. Entries are candidate-pool records; site labels are not necessarily distinct physical fields. Source-specific missing-answer and pre-cutoff records remain audited.','tab:cohort'),
         escape('On the273 development pools, yield-first at25% testing recovered93.322% of oracle top entries with0.283700% mean relative regret. The mixed, ridge and yield-only rankings tied. The primary locked choice was the mixed model with uncertainty fraction0.25 and shrinkage2. The secondary joint-grid choice was extra trees with minimum leaf20, uncertainty fraction0.25 and shrinkage8, with development regret0.283187%. This small selected development difference did not establish an adaptive benefit.'),
         r'\subsection{Primary comparison and paid feedback}',escape(main),escape(feedback),
         latex_table(['Evaluation','Policy','Regret(%)','Recall(%)','Best(%)'],policy_rows,'Primary25% design budget. Regret is percentage of each pool oracle mean. Recall and best-entry finding are percentages. Random seeds are averaged inside each pool.','tab:primary'),
         latex_table(['Evaluation','Contrast','Difference(pp)','95% interval'],comparison_rows,'Adaptive minus comparator relative regret. Negative favors adaptive. Intervals resample harvest years and are descriptive; zeros can reflect identical actions. Secondary comparisons remain secondary.','tab:paired'),
         r'\subsection{Budget dependence and transfer}',escape(budget),
         escape(f'At25% testing, the yield-only mean native-source regret was{ur.regret:.3f} Uruguay index units and{ar.regret:.3f}kg/ha in Arkansas. No cross-program comparison of these absolute units is intended. The chance of finding the measured best entry was{100*ur.best_entry_found:.2f}% and{100*ar.best_entry_found:.2f}%, respectively. The high Uruguay recall therefore did not transfer at the same testing fraction.'),
         latex_table(['Budget(%)','UR recall(%)','UR regret(%)','AR recall(%)','AR regret(%)'],budget_rows,'Yield-first testing on the fixed budget grid. Connecting plotted points does not estimate an untested budget. The90% mean-recall target is illustrative.','tab:budget'),
         r'\subsection{Measurement coverage and robustness}',escape(limits),
         r'\input{sensitivity_results.tex}',r'\subsection{Reproducibility}',escape(verification_text),
         r'\FloatBarrier',r'\section{Discussion}',
         escape('The most direct result is the absence of useful paid feedback at the primary25% budget under the frozen policies. The primary mixed-model predictor produced a yield-consistent ranking, and its first-batch updates did not alter untested ranks at the primary budget. Exact ties are consequently an algorithmic observation in these pools, not evidence that all possible adaptive laboratory designs are equivalent. The more flexible secondary tree policy changed some ranks, but its only primary-budget Uruguay feedback effect was adverse; its small external static-ranking advantage was shared by the no-update policy.'),
         escape('A simple rule can be strong in one program while still needing many more tests in another. Yield-first reached high top-entry recovery in Uruguay, but the same budget missed a larger share of the Arkansas oracle shortlist. Whole-kernel-output regret and oracle recall answer different questions: entries close in output can produce modest regret despite lower identity recovery. No stakeholder-approved practical margin or assay-cost study was available, so the observed percentage differences should not be converted into financial or operational claims.'),limitations,
         r'\section{Conclusion}',escape('For these measured-answer cohorts, two-batch feedback did not improve the primary-budget shortlist beyond the frozen static or matched no-update policies. Yield-first was a strong retrospective screen in Uruguay, while Arkansas required a larger measured budget to reach the same illustrative mean-recall target. Prospective utility and broader adaptive designs remain open questions.'),
         r'\section*{Data and code availability}',escape('All observations originate from existing public datasets and reports. Uruguay source files are CC0 and were obtained from the checksum-exact Dryad-community mirror; Arkansas report redistribution terms are not assumed to match that license. The local reproducibility package contains public acquisition scripts, source hashes, extraction audits, immutable splits, policy events, independent verification and pinned Python dependencies. No public code repository has been uploaded as part of this work.'),
         r'\href{https://doi.org/10.5061/dryad.x69p8czn8}{Uruguay dataset}; \href{https://zenodo.org/records/8222856}{exact mirror}; \href{https://uaex.uada.edu/farm-ranch/crops-commercial-horticulture/rice/}{official Arkansas source archive}.',
         r'\section*{Author-review status}',escape(readiness),r'{\small\raggedright\setlength{\parskip}{0pt}\input{references.tex}}',r'\section*{Figures}']
    figures=[('secondary_feedback_budget.png','All secondary tree-policy feedback contrasts on the fixed grid. Shading is the descriptive95% year-block interval. A sparse Arkansas benefit appears at40%; it is a secondary budget-specific result, and does not change the primary25% comparison.'),('development_baseline_curve.png','Development-only yield-first and random baseline.273 pools in2012-2016; shading shows descriptive95% harvest-year intervals. Random testing averages20 fixed seeds within pool.'),
             ('policy_budget_top_k_recall.png','Oracle top-k recovery in temporal Uruguay and independent Arkansas evaluation. The shortlist and measured-answer candidate sets are fixed within each pool. Shading shows descriptive year-block intervals for the primary adaptive and locked static policies; overlapping curves indicate identical results.'),
             ('policy_budget_regret_pct.png','Relative whole-kernel-output shortlist regret on the fixed budget grid. Absolute source units are not pooled. Random testing illustrates headroom; the paired figure resolves smaller adaptive-static effects.'),
             ('paired_effects.png','Primary and secondary adaptive effects at25% testing. Negative relative-regret differences favor adaptation. Zero primary effects reflect unchanged decisions; the secondary external gain is shared by its no-update policy. Intervals are descriptive year-block resamples.')]
    figures=figures[1:]+figures[:1]
    shutil.copyfile(ROOT/'outputs/development/figures/development_baseline_curve.png',out/'figures/development_baseline_curve.png')
    shutil.copyfile(ROOT/'outputs/development/figures/development_baseline_curve.svg',out/'figures/development_baseline_curve.svg')
    for name,caption in figures:tex += [r'\begin{figure}[H]\centering',r'\includegraphics[width=\textwidth]{../figures/'+name+'}',r'\caption{'+escape(caption)+'}',r'\end{figure}']
    tex += [r'\end{document}']
    (destination/'rice_milling_selection.tex').write_text('\n\n'.join(tex)+'\n')
    sensitivity_text(out,destination)
    if compile_pdf:
        with (destination/'build.log').open('w') as log:
            for _ in range(2):subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','rice_milling_selection.tex'],cwd=destination,stdout=log,stderr=subprocess.STDOUT,check=True)
    write_json(report/'report_inputs.json',{'results_verified':True,'original_clean_repeat_verified':repeated,
        'input_sha256':{str(p):digest(p) for p in paths+[out/'evaluation_freeze.json',out/'verification/verification_summary.json',out/'verification/table_verification.json',ROOT/'manuscript/methods_draft.tex',destination/'sensitivity_results.tex']},
        'generated_tables':'All numerical manuscript tables are generated directly from frozen-run CSVs; narrative scope follows the locked protocol.',
        'presentation_source_sha256':digest(__file__)})
    return report/'decision_report.html'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--no-pdf',action='store_true');a=p.parse_args()
    print(build_report(a.out,not a.no_pdf))
