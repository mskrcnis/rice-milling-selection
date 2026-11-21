# Rice milling selection

Reproducible research software for selecting rice entries for milling measurement after field yield is known. Fixed testing policies are replayed on public Uruguay and Arkansas trial records to measure recovery of entries with high yield-times-head-recovery values.

The primary two-batch policy did **not** improve its matched no-feedback decisions at the 25% design budget. This repository preserves that result, the full policy comparisons and separately identified post hoc diagnostics. A query is a reported candidate-level answer, not a measured laboratory assay, cost or time saving.

## Quick start

Use Linux, Bash and Python 3.12 on a CPU. No GPU, R or LaTeX installation is needed for the computational replay.

```bash
git clone https://github.com/mskrcnis/rice-milling-selection.git
cd rice-milling-selection
bash run_reviewer.sh
```

The runner creates `.venv` and installs [pinned dependencies](requirements.lock) if necessary. Installation requires internet access; the replay uses included numeric data. To use an existing Python 3.12 environment containing those exact dependencies:

```bash
REVIEWER_PYTHON=/absolute/path/to/python bash run_reviewer.sh
```

The workflow runs 97 unit and synthetic tests, freshly fits historical models, replays fixed policies, reconstructs results from events, and compares **18 original scientific artifacts plus 27 post hoc diagnostic artifacts** with recorded hashes. Sixteen tests requiring omitted Arkansas source PDFs are excluded from the default run. The earlier portable replay took about ten minutes on its validation workstation, plus diagnostic time; runtime depends on hardware.

The [release validation records](validation/README.md) document a successful fresh replay and a separate clean-environment test run.

Success is recorded in:

- `study/outputs/reproduction/portable_reproduction_verification.json`
- `revision/outputs/portable_diagnostic_verification.json`

Both output directories must be empty before starting. Use a fresh checkout for another full run. See [reproduction details](docs/reproducibility.md) for individual commands and environment limits.

## Study boundaries

The observed-answer index is mean field yield multiplied by mean head recovery, divided by 100. Missing milling answers are not imputed. Final models use Uruguay history through 2016; temporal evaluation uses Uruguay 2017–2020 and primary external evaluation uses Arkansas 2017–2025.

Design fractions are 10%, 20%, 25%, 40%, 60% and 100%. Query counts are rounded up within each pool, so actual fractions can be larger. Evaluation weights trial pools equally; historical fitting uses candidate rows. Source-basis assumptions, the observed-answer restriction, recurring sites and limited years constrain interpretation. Absolute output is not pooled across programs.

Read [methods and limitations](docs/methods.md) before interpreting results. A rerun reproduces an already observed study; it is not a new blinded evaluation.

## Repository map

| Path | Purpose |
| --- | --- |
| `study/src/`, `study/tests/` | Analysis implementation and tests; repository formatting is recorded separately |
| `study/derived/`, `study/private/` | Candidate metadata and separated outcome tables |
| `study/raw/` | Four original Uruguay files and Arkansas acquisition metadata |
| `study/protocol/` | Analysis protocol, fixed configuration and split records |
| `study/outputs/run_a/` | Subset of recorded original reference outputs used by diagnostics |
| `revision/src/`, `revision/protocol/` | Post hoc diagnostics and their dated plans |
| `revision/reference_outputs/` | Recorded diagnostic outputs |
| `provenance/` | Historical locks, reviewer-package records and formatting evidence |
| `repository_manifest.json` | Repository file-integrity record |

Names containing `private` preserve computational outcome separation. These are public rice-trial observations, not confidential participant records.

## Data, provenance and reuse

- [Data and acquisition](docs/data.md): source identifiers, extraction evidence and optional PDF retrieval.
- [Development](docs/development.md): original, portable and repository provenance; formatting checks and AI assistance.
- [Source terms](DATA_TERMS.md): the Uruguay archive's CC0 scope and limits on redistribution.

Full Arkansas reports and third-party papers are not redistributed. No repository-wide software license or paper authorship metadata is asserted here; a source dataset's CC0 dedication does not license the whole repository.
