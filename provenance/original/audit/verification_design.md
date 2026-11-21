# Independent event-log verification design

The verifier in `src/verify.py` imports neither the replay, experiment, model nor scoring modules. It independently computes budgets, SHA-256 tie ordering, random-policy selections, initial uncertainty/diversity acquisitions, paid residual corrections, second-batch rankings, tested-only shortlists, oracle output, regret and recall. The frozen predictions are inputs; no model is refitted or selected. Public inputs are used for initial actions, and only the first batch's paid answers enter reconstruction of later actions.

Before reading any answer CSV, the CLI invokes `common.verify_freeze` on the per-run evaluation freeze and, when present, the global analysis lock. The run freeze must bind selected configuration, initial predictions and batch plans. The public cohort and hidden answer key must be bound by one of these verified freezes. This checks immutable input consistency; it does not itself establish that model fitting preceded outcome access. The acquisition/modeling separation and timestamped project protocol provide that evidence.

Every pool must have the complete frozen model/prediction and nonrandom initial-plan grid. Every configured budget must have all static policies, the adaptive/no-update/exploit-only policies and every declared random seed. The verifier rejects duplicate candidates, prediction rows, events or table identities, missing/nonfinite answers, malformed counts, wrong models, metadata mismatches, queries outside a pool, repeated queries, inconsistent reveal histories, incorrect paid answers, acquisition-score changes, unspent/exceeded budgets, incorrect final shortlists and any discrepancy in independently regenerated metrics. Full-budget policies must attain exactly zero regret and complete top-k/best-entry recovery. Adaptive and no-update policies must have the same first batch, scores and paid answers.

The optional pre-test secondary analysis is activated only by `config.secondary_adaptive`, with status `development_selected_secondary`, a frozen model, uncertainty fraction and shrinkage. It adds `adaptive_grid` and `adaptive_grid_no_update` to every pool/budget and independently checks their model-specific plans, acquisition scores, updates and matched first batch. The original primary pair remains required. This development-driven extension is a separate secondary comparison; its small development improvement is not evidence of test superiority, and whichever policy performs better in the final evaluation must not be retrospectively promoted to primary. Both compare against the same locked strongest-static benchmark. Yield-only's constant100 is a rank-preserving surrogate, not an honest recovery forecast; its event consistency is verified without interpreting recovery calibration.

The numerical comparison uses absolute tolerance1e-8 and relative tolerance1e-10; identity/order/count checks are exact. Undefined percentage regret for a zero oracle and unavailable top-ten metrics remain undefined. If prediction diagnostics exist, their means and uncertainty must equal the frozen prediction grid, and their observed values must match the key. Additional result columns outside the explicitly reconstructed schema are listed as unverified rather than silently certified.

Outputs are written under `<run>/verification/`: `regenerated_per_pool_results.csv` and `verification_summary.json`, including checked denominators, input hashes and tolerance. These are independent verification artifacts; the verifier does not overwrite the experiment's saved results, policy events or frozen inputs.

Synthetic tests generate integration fixtures using the production replay but never use its scoring functions inside the verifier. A separate hand-calculated two-entry check anchors the output arithmetic:4000×50%=2000 and9000×25%=2250, so selecting the former gives250 regret and100/9 percent relative regret. Corruption tests alter metrics, counters, metadata, paid answers, query histories, acquisition scores, batch numbers, shortlist identities, frozen plans/predictions and complete-grid coverage. Tests also enforce frozen-input validation before any CSV reading. All fixtures are synthetic; no real private outcomes were opened while implementing this verifier.

After the model owner releases the outcome freeze, run:

```bash
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m src.verify \
  --out outputs/run_a \
  --freeze outputs/run_a/evaluation_freeze.json \
  --config outputs/run_a/selected_configuration.json
```

Defaults are `derived/cohort/evaluation_public.csv`, `private/evaluation_key.csv` and `protocol/analysis_lock.json`; CLI overrides are available. A second fresh run gets its own run directory and freeze. Final source extraction, pooled summaries, uncertainty intervals and publication claims require their separate audits; this verifier certifies the logged decision replay conditional on its bound inputs.
