# Methods and interpretation

The [analysis protocol](../study/protocol/analysis_protocol.md) and [selected configuration](../study/protocol/selected_configuration.json) contain the full decisions. This summary does not amend them.

## Target and budgets

A candidate is one source-coded line or cultivar in a compatible trial, site and harvest year. Pools are source-defined trials, not artificial class-specific subsets. Eligible pools have at least ten candidates with positive field yield and observed recovery in 0–100%.

```text
index = mean field yield × mean head-rice recovery / 100
k = ceil(0.10 × n)
B = min(n, max(k, ceil(f × n)))
```

The reference shortlist contains the highest-index `k` eligible candidates. The primary design fraction `f` is 25%; secondary fractions are 10%, 20%, 40%, 60% and 100%. Actual testing is `B/n`, which can exceed the nominal fraction.

Policies see field-yield features and only queried answers. Two-batch policies choose the entire first batch before seeing its answers, then apply a fixed feedback rule. The no-update comparison uses the same model, budget and first batch. Exact ties use the preserved hash of each pool-specific row identifier.

## Fitting and evaluation

Uruguay 1997–2011 supports initial fitting; rolling origins in 2012–2016 select settings. Year 2002 is absent. Final models fit once through 2016. Uruguay 2017–2020 is temporal evaluation; Arkansas 2017–2025 is primary transfer evaluation. Documented compromised and previously seen pools remain separate.

Core predictors use relative field yield and within-pool yield percentile, computed over all compatible field-observed entries before the recovery-answer restriction. Source grain and germplasm labels are not asserted to be harmonized market classes.

Historical fitting uses candidate rows; evaluation weights pools equally. Random-policy seeds are averaged within pool. Percentage regret compares selected and full-testing mean indices within the same pool. Recall counts recovered reference entries, not pools finding the single best candidate.

The primary two-batch policy did not improve matched no-update decisions at 25%. This is specific to that policy, information set and budget. Secondary models and other budgets remain in the full results and do not replace the primary comparison. No universal claim that feedback is ineffective follows.

## Post hoc diagnostics

Plans under `revision/protocol/` were written after original outcome exposure and before the new diagnostic calculations. They preserve original policies, cohorts, endpoints and budgets.

Action checks distinguish ranks, second-batch decisions, query order and membership, shortlist membership and regret. Rounding checks distinguish sharp fixed-query pointwise extrema from joint feasible scenarios that recompute queries and outcomes together. The 20,000 uniform scenarios and 16 deterministic scenarios are conditional numerical explorations, not new observations or an estimated sampling distribution. Scenario percentiles are not confidence intervals; the reported Arkansas 60% grid crossing is not universally rounding-stable.

## Limits

- Results condition on observed milling answers; missing answers are not imputed and excluded candidates' utility is unidentified.
- Queries count candidate-level answers, not physical assays, costs or time saved.
- Uruguay retains a source-basis interpretation. Absolute output is not pooled across programs or presented as moisture-harmonized mass.
- Four Uruguay evaluation years and recurring sites limit independence. Harvest-year intervals are descriptive, not prospective guarantees.
- Source reviewers saw outcomes. Original model-development separation was not whole-team blinding; later runs check reproducibility.
- Replay does not identify genetic causal effects, farmer risk or expert agronomic endorsement.
