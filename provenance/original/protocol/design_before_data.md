# Milling-selection replay: design before comparative evaluation

Created 20 November 2026. This is an implementation design record; the final protocol and exact metadata-based split manifest will be hashed before comparative test evaluation. No final test or external policy performance has been examined. Acquisition reviewers see source outcomes; policy development remains separated from them.

## Observable decision

The candidate is one distinct entry in one compatible trial, site and harvest year. Repeated checks and field replicates do not create additional candidates. A query reveals one measured candidate-level head-rice-recovery result. The target is the product of candidate mean paddy mass yield and mean percentage head recovery from original rough-rice mass. This product of published means is not automatically the mean of plotwise products, a causal treatment effect, physical assay count or farmer-specific risk. Matched Uruguay plots, if available, support the separately labelled product-definition sensitivity. Missing milling outcomes never become imputed answers.

Each pool receives equal weight. Primary eligibility requires at least ten distinct candidates with measured paired values. Damaged, mixed, pooled-across-site, planting-date and incompatible experiments are retained in a disposition ledger; conservative damage exclusions are made from source annotations before performance evaluation. The previously illustrated 2025 NERREC/Poinsett Arkansas pool is separately exposed evidence.

## Budgets, ranking and outcomes

For n candidates, k=ceil(0.10n), B=min(n,max(k,ceil(fn))). Primary f=0.25; secondary f=0.10,0.20,0.40,0.60,1.00. First-batch size is ceil(B/2), with a one-batch fallback for B=1. The final shortlist is the measured best k entries among queried candidates. Exact value ties use the deterministic SHA-256 order of `rice-milling-tie-v1|row_id`. This convention is shared by the oracle and all policies. Source-resolution uncertainty near rank k will be a separate sensitivity; it will not select a preferred tie convention.

Primary loss is oracle shortlist mean minus policy shortlist mean, with the difference also divided by the positive oracle mean and multiplied by100. Smaller regret is better; adaptive-minus-static and adaptive-minus-no-update differences below zero favor adaptation. Secondary outcomes are recall of oracle top k, best-entry recovery, tested-entry counts, and recall/count of oracle top ten among tests when n and B permit. The illustrative recovery target is90% top-k recall on the measured budget grid; no interpolation or operational savings claim is made.

## Historical information

All histories and model fits for a harvest year use strictly earlier years, with no sequential revelation across pools within the same year. Same-year laboratory ordering is unknown. Pool identifiers, candidate replicates and shared samples cannot cross a split. Model feature construction includes only pre-milling public information. The primary transferable feature set is within-pool relative yield, within-pool yield percentile, and grain class when its cross-source meaning is verified; missing/unseen classes receive explicit fallback. Relative yield and its rank can use all known field yields in the current pool. Cultivar histories, pedigree, weather, post-milling traits, total recovery and future cultivar descriptions are excluded from the core feature set.

## Policies and tuning

Static comparators are random testing with20 fixed seeds (0–19), paddy yield, paddy yield times historical grain-class recovery, a Gaussian random-pool-intercept recovery model, ridge regression with alpha1,10,100, and extra trees with128 trees and minimum leaf size5 or20. The mixed model is fitted by restricted marginal likelihood, with residual and between-pool scales estimated from earlier observations. Prediction for a new pool uses the fixed effects and includes historical candidate/pool uncertainty in the exploration score. This is an uncertainty heuristic, not a claimed calibrated probability interval. All transformations are fitted using historical training candidates only. Tree seeds are fixed at91021.

The strongest one-batch comparator minimizes average pool-level percentage regret at25% budget on rolling-origin development years. Ties use alphabetical model name. The adaptive predictor is the strongest development model among mixed, ridge and trees; a yield-only winner remains the primary one-batch benchmark if that is what development shows. No external comparison selects a model.

For adaptive batch one, a fraction of candidates is selected by predicted whole-kernel output and the remainder by a boundary uncertainty score. Uncertainty fractions0.25,0.50,0.75 and residual shrinkage2 or8 form the development-only grid. Let m_i=y_i mu_i/100, s_i=y_i sigma_i/100 and t be the kth largest m. Boundary uncertainty is s_i phi((m_i-t)/s_i), divided by one plus similarity to already chosen entries. Similarity uses matching grain class and a Gaussian kernel on within-pool relative yield with bandwidth0.25. At least one exploitation selection is retained. All first-batch IDs are selected before any of its answers are revealed.

For an untested candidate j, the feedback correction is

`delta_j = sum_i w_ij (HR_i - mu_i) / (lambda + sum_i w_ij)`.

The sum uses only paid first-batch answers. Kernel weight is `exp(-0.5*((yield_rel_i-yield_rel_j)/0.25)^2)`, multiplied by1 for matching grain class and0.25 otherwise. New or sparse classes thus borrow weakly from other tested entries; a missing class is an explicit unknown level. Updated recovery is clipped to the physical0–100 range. Batch two takes the largest updated expected outputs. The critical no-update policy shares identical first-batch selections, predictor and budget, with delta fixed to zero. The exploit-only first-batch variant removes uncertainty sampling. Rank changes of untested entries are recorded. Only the final choice among tested entries uses their measured recovery.

## Separation, verification and interpretation

The policy API sees only allowlisted public fields and a query gate. The full key is available only to scoring after the shortlist is fixed. Event logs retain ordered batch IDs, pre-reveal scores, previously revealed IDs, returned answers and remaining budgets. A clean repeat must reproduce policy events and result tables. Tests must prove that unqueried answers cannot alter prior actions, no-update first batches match, repeated queries fail, and full budget attains the oracle under the fixed tie rule.

Paired contrasts are computed within identical pools and budgets. Harvest-year block resampling describes uncertainty; repeated random seeds are averaged within each pool and never counted as independent trials. With fewer than five harvest-year blocks, intervals are explicitly descriptive and no precision-based general conclusion follows. Site dominance, stage/class, observed-answer availability, damage exclusions, rounding and new/returning candidates are reported without selecting favorable subgroups. Practical superiority margins and budgets are illustrative, not stakeholder-approved laboratory costs. Sequential selection and temporal validation already exist in prior art; any contribution must be supported by this decision-specific feedback ablation and independent-program evidence.
