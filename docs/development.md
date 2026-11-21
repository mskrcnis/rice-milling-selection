# Development and provenance

## Preserved layers

1. **Original study:** protocols, locks, splits and audits remain under `provenance/original/`; scientific references remain under `study/outputs/run_a/`. Historical absolute paths and hashes describe that execution and are not rewritten to describe this checkout.
2. **Portable reviewer package:** separate wrappers reproduced the original replay and post hoc diagnostics from bundled numeric inputs. Its original manifests remain under `provenance/reviewer_bundle/`. This was a rerun after outcome exposure, not a new blinded evaluation.
3. **This repository:** formatting and packaging improve inspectability. [Formatting provenance](../provenance/repository_formatting.json) records source and repository hashes and Python abstract-syntax-tree comparisons. New input manifests describe this repository's file bytes, separately from the historical locks.

`portable_input_manifest.json` and `revision_input_manifest.json` bind their runners' inputs; `repository_manifest.json` records repository integrity. Expected scientific and diagnostic hashes remain numerical reproduction references. The rounding execution lock is newly dated for this formatted repository; its original pre-computation lock remains under `provenance/reviewer_bundle/revision/protocol/`. It does not claim that repository packaging occurred before the original results.

[Recorded omissions](../provenance/repository_omissions.json) identify three large archived outputs that the fresh replay regenerates and a manuscript-only checker. The original archive and manifests preserve their hashes. The retained `study/outputs/run_a/` tree is therefore a reference subset, not a complete historical working directory. Use the root runners for the supported portable workflow; historical report-building helpers can require omitted manuscript or output files.

## Changes and validation

Formatting is not intended to alter analysis behavior. Abstract-syntax-tree equality checks Python structure; tests and fresh artifact comparisons provide separate execution evidence. Neither alone certifies scientific assumptions or source extraction.

Keep scientific changes separate from formatting. Changes to eligibility, units, models, information access, budgets or endpoints need a documented analysis change and new results, not silent replacement of expected hashes. Retain failed comparisons while investigating discrepancies.

Install `requirements-dev.txt` for the pinned Black formatter; `pyproject.toml` excludes archived code copies from formatting. Run `python scripts/verify_repository.py` to check file hashes and the 31 source syntax-tree comparisons. Expected hashes must be updated only as part of an explicitly reviewed new release, never automatically by a test.

GitHub Actions runs integrity checks and the portable tests on pushes and pull requests. Its manual `full_replay` option runs the full computation and diagnostics.

Use the commands in [reproducibility.md](reproducibility.md) for tests and full execution. Raw-PDF extraction checks require separately acquired sources and a scratch copy.

## Assistance and review

Development of code and documentation used AI assistance. Automated tests, repeated executions and separately recorded source/content checks provide validation evidence; they are distinct from human authorship and expert review. The repository does not claim human-only authorship or completed agronomic consultation.

No author list, publication acceptance, contribution statement or repository-wide license has been invented. Software citation metadata should be added after authorship and release details are confirmed. Source attribution remains in [DATA_TERMS.md](../DATA_TERMS.md).
