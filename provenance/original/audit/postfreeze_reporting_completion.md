# Post-freeze reporting and verification completion

The scientific configuration, outcome definitions, candidate cohorts, splits, prediction models, replay policies, scoring, aggregation, prespecified diagnostics and sensitivities were not changed after the original evaluation freeze. The first and second runs retain their original frozen source snapshots and all bound inputs. The final release re-verifies those snapshots and the earlier development, cohort and analysis locks.

After initial evaluation, reporting work added or completed `report.py`, `report_figures.py`, `bibliography.py`, the manuscript text, and source-linked HTML/PDF delivery. The report explicitly retains the secondary 40% Arkansas tree-feedback improvement from one of 62 pools. Presentation changes corrected spacing, effect-direction labels, primary-budget scope and final repeat status; they did not change numerical scientific results.

`availability_report.py` completes the requested source/year/site/stage/class availability and included/excluded public-field summaries from already frozen public metadata. It does not open outcomes, change eligibility or select a new endpoint. Both runs produce identical availability CSVs.

`regenerate_tables.py` checks all 11 aggregate CSVs by applying the frozen aggregation implementation to independently reconstructed event metrics. This is distinct from the six core tables' separately coded statistical reconstruction in `verify_tables.py`. Both checks pass for both runs. The additional check does not create new scientific evidence.

`reproduce.py` was strengthened to require successful hash-bound event and aggregate verification, compare symmetric artifact sets, and compare fitted-model bytes. It checks 45 identical artifacts from the two fresh fits, including the event logs, results, predictions, fitted models, tables, figures, diagnostics and sensitivity CSVs. PDF timestamps and run-specific manifests are intentionally outside that byte-identity claim. The two PDFs instead have identical LaTeX sources and extracted page text, clean compilation and 11 pages each.

The operational driver `run.py` now invokes the added all-table check and availability helper, as well as the completed report builder. The original scientific runs completed before these final reporting integrations; the added helpers were executed separately on both saved runs. Their actual outputs were included in the final repeat check. Integration imports compile, and the README distinguishes a fresh full replay from report-only regeneration. No third full scientific fit is claimed.

The final PDF was rendered and inspected; `final_artifact_qa.json` records the checks and artifact hashes. The release remains an anonymous manuscript for author and agronomic review. No submission, public upload, author contact, new experiment or prospective laboratory validation occurred.
