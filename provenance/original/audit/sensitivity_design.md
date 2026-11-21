# Fixed-query sensitivity design

Prespecified on 2026-11-20 before scientific code freeze and before the sensitivity developer read real evaluation outcomes. Only synthetic fixtures were used to implement and test this helper. The earlier Uruguay source extraction was outcome-exposed; this is not a claim of source-author or extractor blinding.

## Input and freeze contract

The callable is `src.sensitivity.run_sensitivities(events_path, evaluation_public_path, all_public_metadata_path, uruguay_private_path, arkansas_private_path, arkansas_source_public_path, run_freeze_path, out_dir)`.

Expected inputs are the completed `evaluation_events.jsonl.gz`, cohort `evaluation_public.csv` and `all_public_metadata.csv`, each program's source `private_outcomes.csv`, and Arkansas source `public_features.csv`. The run freeze uses the `src.common.freeze` schema: `files` maps absolute paths to SHA-256 digests. Every public/private source input above must be covered. The helper verifies the whole freeze and source coverage **before** opening data frames or the event log. It then hashes all inputs on entry and again before output. It does not fit a model, execute a policy, request a new assay, or retune settings.

The completed event log is generated after the frozen policies have made paid reveals. Its hash is recorded as a **post-freeze** input; it is not described as a pre-frozen outcome file. Every event must agree with its public pool, program/year/site/stratum, observed private answers, ordered query history, budget, unique tested IDs and measured top-k shortlist under the original objective. The log must cover every public evaluation pool. Repeated `(pool_id,policy,seed,budget_fraction)` events are rejected.

## Uruguay product definition

For each completed Uruguay event, keep exactly its recorded tested IDs. Require **every candidate in its original replay pool**, including untested candidates, to have a positive matched-plot count and finite observed values for both:

- `matched_product_of_means`: mean GY on jointly measured plots times mean PHR on those same plots divided by100.
- `output_mean_of_matched_plot_products`: mean of GY times PHR/100 on the same jointly measured plots.

The archive prep calculates those fields from observed source plots. The comparison separates the effect of products-of-means versus mean-of-products on one matched subset. The original objective uses all available candidate-level GY and PHR means, so the original event's regret is retained as context and is not mislabeled as the matched endpoint.

Within each endpoint, choose the best k **already tested** entries using the existing `src.replay.ordered_ids` tie rule, then score against that endpoint's own full-pool oracle. There are no new queries and no counterfactual recalculation of the adaptive second batch. If any replay candidate lacks either matched endpoint, both endpoint rows are explicitly unavailable and endpoint metrics are NaN. The helper never restricts the oracle to an easier matched subset, drops an inconvenient candidate, or imputes a result. Output uses the source's kg/ha proxy; unknown Uruguay moisture and archivewide denominator limitations remain.

## Arkansas source rounding

The Arkansas extractor confirmed on 2026-11-20 that selected individual-year grain cells are integer bushels/acre and HR/TR cells are integer percentage points, with no finer precision or substituted units. The audited conversion is45lb/bushel and12% source moisture; the helper independently checks the source bu/ac value against the cohort kg/ha field.

For the reported pair `(y,h)`, the deterministic envelope is:

`lower = max(0,y-0.5) × kg_per_ha_per_bu_ac × max(0,h-0.5)/100`

`upper = (y+0.5) × kg_per_ha_per_bu_ac × min(100,h+0.5)/100`.

Here `kg_per_ha_per_bu_ac = 45 × 0.45359237 / 0.40468564224`. Noninteger inputs fail explicitly rather than receiving an assumed precision. These are closed nearest-unit rounding envelopes, not estimates of field/laboratory measurement error, sampling intervals, or independent new observations.

For the point-value oracle and the point-value best tested shortlist separately, the helper asks whether an outsider's upper bound can outrank an insider's lower bound. Equality uses the frozen hash tie rule. It records membership-change possibility, the number of potentially crossing pairs, selected entries at risk, and possible entering entries. For each Arkansas candidate it also records conservative best/worst ranks in the full pool and whether top-k membership is guaranteed or possible. Bounds use a rectangular error set without assuming source rounding errors are independent random draws; they are conservative possibility diagnostics. No uniform perturbations, simulated observations, policy action changes, or sampling-uncertainty claims are introduced.

## Public missingness strata

The candidate denominator is the source pool's entries with observed field yield, and the numerator is entries with both observed field yield and an observed answer. Public `observed_field` and `paired` flags are required and checked. A pool is tagged `complete`, `90_to_less_than_100_percent`, `below_90_percent`, or `no_observed_field`. The replay pool must exactly match the public paired-candidate count. Each frozen event's original regret and top-k recall are attached to its coverage stratum for later summaries. This is a descriptive selection diagnostic; it does not establish performance on unmeasured entries or repair missing outcomes.

## Output schemas and execution

Output directory is `<run-output>/sensitivity/`:

- `product_definition_sensitivity.csv`: event identity (`pool_id,policy,seed,budget_fraction,program,stratum,year,site,n,k,budget`), endpoint definition, availability/missing-match counts, unchanged-query flag, original regret, endpoint oracle/selected means, regret and relative regret, recall, endpoint oracle/shortlist IDs, number of changed shortlist entries, units and interpretation.
- `rounding_interval_sensitivity.csv`: event identity, half-unit assumptions, oracle and tested-shortlist membership-ambiguity diagnostics, interpretation and unchanged-query flag.
- `rounding_candidate_intervals.csv`: one row per Arkansas candidate/pool, point/lower/upper output, best/worst possible rank, point/guaranteed/possible top-k membership and unit basis. This table is not duplicated across policy events.
- `paired_coverage_sensitivity.csv`: event identity, source/field/paired candidate denominators, paired fraction, coverage stratum and original regret/recall/tests.
- `sensitivity_manifest.json`: run-freeze hash, completed post-freeze event-log hash, input/output hashes, completed-event count, and explicit flags that queries and policy settings were unchanged and sampling uncertainty was not estimated.

Callable integration is preferred. The standalone interface is `python -m src.sensitivity --events ... --evaluation-public ... --all-public-metadata ... --uruguay-private ... --arkansas-private ... --arkansas-source-public ... --run-freeze ... --out ...`.

Tests in `tests/test_sensitivity.py` use only synthetic values. They cover fixed queries, separate endpoint oracles, incomplete matched outcomes, full-budget zero regret, source-answer validation, invalid shortlists, rounding units/bounds, exact tie behavior, brute-force endpoint rank containment, unit-conversion mismatch, paired-coverage denominators, refusal to read data before successful freeze verification, and deterministic output replay. No physical assay count, money, or laboratory-time savings is computed.
