# Data and acquisition

## Included observations

The replay uses public rice-trial measurements. Candidate metadata and outcomes remain separate to preserve the original information-access design. Names such as `private_outcomes.csv` identify outcome keys, not confidential human data.

| Source | Included evidence | Provenance |
| --- | --- | --- |
| Uruguay INIA archive | Four original files, candidate features, aggregated outcomes and source-row links | [Uruguay manifest](../study/audit/uruguay_source_manifest.json) |
| Arkansas reports | Candidate features, extracted answers and numeric-cell locations; no full PDFs | [Arkansas manifest](../study/raw/arkansas/source_manifest.json) |
| Cohorts | Pool/candidate identifiers, split metadata, paired availability and outcome keys | `study/derived/cohort/`, `study/protocol/`, `study/private/` |

The Uruguay source is [Dryad DOI 10.5061/dryad.x69p8czn8](https://doi.org/10.5061/dryad.x69p8czn8), version 248051. Acquisition used the checksum-matched [Dryad-community mirror](https://zenodo.org/records/8222856). ReData was not substituted: its anonymized columns did not establish the required trait definitions.

Arkansas candidates are anchored to contemporaneous field-entry lists. A later historical table supplies an answer only when candidate, site and year are explicit; unresolved aliases remain missing. Screening and exclusion records remain under `provenance/original/audit/`.

Both programs have `private_outcomes.csv` under their respective `study/derived/` directories. Public replay features are in the cohort's `development_public.csv` and `evaluation_public.csv`; Arkansas also retains its program-specific `public_features.csv`. Arkansas `private_all_extracted_cells.csv` records report, page, table and raw cell fields. `study/derived/cohort/all_public_metadata.csv` includes records outside the paired-answer analysis.

## Units and aggregation

Uruguay reports kg/ha and percentage head recovery. Same-program methods support a rough-rice denominator for specified trials, but archive-wide moisture and recordwise denominator conventions remain unconfirmed. Its yield-times-recovery value remains a source-basis index.

Arkansas reports use a 45-lb rice bushel, yield adjusted to 12% moisture, and head recovery relative to original rough-rice mass. Extracted yield and recovery cells are integer bushels per acre and integer percentage points. Half-unit rounding intervals describe reported precision, not sampling or laboratory error.

The primary index multiplies separate candidate means. The preserved Uruguay sensitivity compares that definition with mean matched plotwise products only where the required matched records are complete. Source record and block counts do not identify physical assay counts.

## Optional retrieval

The replay is offline after dependencies are installed. To retrieve recorded public sources separately:

```bash
.venv/bin/python acquire_sources.py \
  --output /absolute/path/to/source-downloads --program all
```

Use `--program uruguay` or `--program arkansas` for one source. The downloader retains recorded relative paths, verifies archived hashes and rejects changed versions. URLs and host availability can change; consult the manifest's final URL or DOI if retrieval fails. The included reference executions used bundled numeric inputs, not a fresh download of every source.

## Optional raw-extraction audit

Use a scratch copy of `study/`, place acquired files at their recorded relative paths, and install Poppler's `pdftotext` alongside the pinned Python environment. Preparation writes derived outputs; do not run it in the reference checkout used for hash-checked reproduction.

From the scratch study directory, inspect and run the preserved entry points `src/prepare_uruguay.py` and `src/prepare_arkansas.py`, then the relevant extraction tests with `PYTHONPATH=.`. The default repository run excludes all 16 tests in `tests/test_arkansas_extraction.py` because the full reports are intentionally absent.

See [DATA_TERMS.md](../DATA_TERMS.md) before redistributing source material. A public URL does not establish a redistribution license.
