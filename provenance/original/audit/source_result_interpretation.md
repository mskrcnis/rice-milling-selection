# Independent source and result interpretation

Review date: 20 November 2026. Outcome separation was explicitly released only after `outputs/run_a/evaluation_freeze.json` existed. Its SHA-256 is `a8bc5b3621171f33fde57730653e1cd4f55fc52fef1bd46d703a4755ec09dc65`; this review independently verified all 32 bound files before opening results. No scientific code, fitted model, protocol, source file, or selection action was changed.

Status: **complete**. Primary result tables, event logs, all four sensitivity CSVs, and their manifest were reviewed. Independent calculations agree with the completed pipeline sensitivity outputs. The sensitivity manifest covers 50,886 completed replay events, records unchanged queries and no policy retuning, and has SHA-256 `a6d502dfa3e3daa0d900ee950f8a10bfb93ea38601fa11e75ae275fe133f6f1e`. All four sensitivity-output hashes verify. The first run completed on 20 November 2026 at 17:26:54 UTC.

## Main finding and its scope

At the nominal 25% testing budget, the primary two-batch, no-update, mixed-model, and paddy-yield-first policies have identical results in both primary cohorts. Mean percentage regret is **0.226086% in 188 Uruguay pools** (7,568 candidates, four harvest years, four location labels) and **2.597324% in 62 Arkansas pools** (1,611 candidates, nine harvest years, 16 source location labels). Mean oracle top-k recall is 93.9269% and 67.7419%, respectively. Random testing has mean regret 8.678948% and 8.382929% under the same budget, with its 20 seeds averaged within each pool.

The primary feedback ablation is an identity of observed actions: no untested ranks, tested sets, or final shortlists change in any primary pool at 25%, and the recorded primary rank-change count is zero at every prespecified budget. The point contrast and its resampling interval are therefore exactly zero. This is strong evidence that this frozen adaptive implementation added no value in these replays; it is not a general equivalence bound for all adaptive policies or proof that milling measurements contain no useful information.

The secondary, development-selected tree policy does not rescue a feedback claim. In Uruguay, feedback increases mean regret by 0.007850 percentage points relative to its identical-first-batch no-update version. In Arkansas its mean regret is 2.519653%, 0.077672 percentage points below the locked mixed benchmark, but exactly the same as the secondary no-update version. The external difference is consequently attributable to the frozen non-feedback selection design, not paid feedback. Independent inspection of 25% event logs finds secondary rank changes in 140/188 Uruguay and 26/62 Arkansas pools, but changed tested sets in only seven and one pools, and a changed final shortlist in only one and zero pools, respectively. Rank changes alone are not demonstrated selection utility.

The near-oracle Uruguay output does not mean universal recovery of the best identities. At 25%, only 151/188 Uruguay pools and 25/62 Arkansas pools attain at least 90% top-k recall. Because of ceiling rules, average actual testing fractions are 25.4499% and 26.4060%, not exactly 25%. Mean yield-first recall first exceeds 90% at the observed 20% nominal grid point in Uruguay and the 60% grid point in Arkansas; this is a grid-level descriptive statement, not a guaranteed minimum budget for an individual trial.

## Endpoint definition sensitivity

All 188 Uruguay temporal pools have observed matched products for every replay candidate, so this sensitivity requires no outcome imputation or additional pool exclusion. The following values were independently recomputed from the private candidate summaries and the already recorded tested IDs, using the frozen SHA-256 tie convention. Each endpoint reselects the best k within the same tested set and defines its own full-information oracle.

| Uruguay endpoint, fixed queried entries | Primary mean regret (%) | Mean top-k recall | Pools with changed oracle top-k | Pools with changed tested shortlist |
|---|---:|---:|---:|---:|
| All-observed marginal mean yield × marginal mean HR / 100 | 0.226086 | 0.939269 | Reference | Reference |
| Product of marginal means on matched source records only | 0.248006 | 0.927301 | 20/188 | 15/188 |
| Mean of matched source-record yield × HR / 100 products | 0.243815 | 0.930227 | 30/188 | 23/188 |

The two matched definitions separate effects of the observation subset from effects of within-record yield–recovery covariance. They are not interchangeable measurements. The original primary equality among two-batch, no-update, mixed, and yield-first persists because those methods queried the same entries. The sensitivity supports the qualitative negative finding **conditional on the fixed queried sets**. It does not rerun adaptive decisions, refit prediction models, or establish that a differently defined reveal would produce the same future actions.

Uruguay's archive specifies kg/ha and whole-milled-kernel percentages but does not explicitly establish an archive-wide rough-rice denominator or a common moisture basis. Same-program primary methods support the denominator assumption; the saved metadata accurately states that limitation. The manuscript should retain “source-basis whole-kernel-output index” for Uruguay, rather than imply a directly verified, moisture-harmonized physical yield. Within-pool percentage regret cancels a common pool-wide multiplicative conversion, but cannot correct unknown entry-specific moisture or denominator differences. Arkansas uses verified 45 lb/bushel and 12% rough-rice moisture. Its endpoint is also a product of published cultivar means; plotwise product means cannot be reconstructed there.

## Coverage and historical versus contemporary Arkansas evidence

| Evaluated cohort | Pools | Entries with observed field yield | Entries with paired answers |
|---|---:|---:|---:|
| Uruguay temporal | 188 | 7,596 | 7,568 |
| Arkansas historical tables, 2017–2020 | 19 | 551 | 447 |
| Arkansas contemporary tables, 2021–2025 | 43 | 1,164 | 1,164 |
| Arkansas previously seen, separate | 1 | 29 | 29 |
| Arkansas compromised, separate | 6 | 175 | 175 |

There are 180 complete paired-answer pools in the Uruguay temporal cohort; seven have 90% to less than 100% coverage and one has less than 90% coverage. The 180 complete Uruguay pools contain 7,137 candidates and have primary mean regret 0.236134%; the eight incomplete pools have zero point regret on their observed-answer subsets. All 104 absent milling answers within eligible Arkansas primary pools occur in the historical 2017–2020 subset. Every one of those 19 pools has below 90% coverage; the lowest coverage is 42.4%. Annual field tables establish the original candidate list, but later historical milling tables preserve selected cultivars. This prevents claiming that the replay describes all original candidate entries. The source workflow correctly retained unpaired field entries for public yield normalization before restricting the scored answer universe.

Absolute performance is materially different across these source periods. Yield-first/primary mean regret is **0.758902% in the 19 historical pools** versus **3.409650% in the 43 contemporary pools**; mean top-k recall is 82.4561% versus 61.2403%. The primary adaptive-minus-static and feedback contrasts remain zero in both subsets, including the fully paired contemporary cohort. The more favorable historical result must not be highlighted without its selective-label qualification. Coverage, calendar period, candidate composition, and location changes are confounded; these results do not identify missingness as the cause of the period difference.

The entirely previously seen 2025 NERREC pool has zero primary regret but is not fresh confirmation. The six compromised pools likewise remain a separate descriptive stratum. Location counts are counts of source labels or station/county codes; the files do not certify independently geolocated physical fields across all years.

## Arkansas integer rounding and ties

The source reports integer bu/ac and integer HR percentage points. Under the prespecified nearest-integer ±0.5-unit rectangular envelopes, an independent reconstruction identifies potential oracle top-k membership changes in **39/62 primary external pools**: 14/19 historical and 25/43 contemporary pools. The tested primary shortlist may change in **27/62 pools**, and the secondary tree shortlist in 29/62. Closed endpoint ties were handled using the frozen deterministic hash rule; ignoring touching interval endpoints would undercount ambiguity by one pool.

These are conditional source-precision envelopes, not measured laboratory error distributions or confidence intervals. They show that exact rank identity is often unresolved by displayed precision. They do not, by themselves, quantify a worst-case change in mean regret or establish whether the small secondary contrast would reverse sign. No such paired-regret interval is claimed. Because the primary policies query identical sets, their equality under a shared endpoint is not evidence of precise knowledge of each entry's rank.

## Manuscript-critical boundaries

- Lead with a strong simple yield rule in Uruguay, weaker identity recovery in Arkansas, and no benefit from this frozen feedback implementation. Do not claim a new generally superior adaptive method.
- Distinguish percentage points of relative regret from percent relative improvement; the secondary Arkansas contrast is −0.077672 percentage points, not a demonstrated feedback gain.
- Show the 2021–2025 complete-contemporary sensitivity with its absolute regret and recall, not just its zero paired contrast.
- Treat four-year Uruguay and nine-year Arkansas harvest-block intervals as descriptive. Recurring locations, year shifts, selective answer availability, and only one external program limit generalization. An interval of [0,0] for identical actions is not a practical-equivalence precision claim.
- Report entry results queried. Neither physical assay counts, monetary savings, laboratory time, nor deployable prospective utility were measured. Repeated source rows do not establish independent milling assays; the source repetition flag includes unknown block IDs.
- Keep the fixed-query nature of product-definition and rounding checks explicit. Neither is a rerun of policy development or a validation of latent biological truth.

Primary evidence paths: `outputs/run_a/tables/policy_summary.csv`, `paired_summary.csv`, `contemporaneous_external_sensitivity.csv`, `secondary_adaptive_paired.csv`, `feedback_action_changes.csv`, `pool_seed_averages.csv`, and `outputs/run_a/evaluation_events.jsonl.gz`. Sensitivity evidence is in `outputs/run_a/sensitivity/product_definition_sensitivity.csv`, `rounding_interval_sensitivity.csv`, `rounding_candidate_intervals.csv`, `paired_coverage_sensitivity.csv`, and `sensitivity_manifest.json`. Coverage was independently reconstructed from `derived/cohort/all_public_metadata.csv` and the frozen evaluation candidate list. Endpoint and rounding checks used the frozen source outcomes and original published-precision fields; no new outcomes were created. Exact tested-set equality among the four primary policies was separately checked in all 250 primary pools at 25%.
