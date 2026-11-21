# Reproducing the analysis

## Full run

Use CPython 3.12 on Linux with [requirements.lock](../requirements.lock). The recorded package environment is [environment.json](../environment.json). The runner sets one numerical-library thread and `PYTHONHASHSEED=0`; no GPU is used.

```bash
bash run_reviewer.sh
```

This runs:

1. The 97 unit and synthetic tests that do not require omitted PDFs.
2. Input-hash checks, fresh historical fits, a new execution freeze and all fixed original policies.
3. Event-based reconstruction of results and tables, plus preserved descriptive and endpoint sensitivities.
4. Comparison of 18 designated original artifacts with [expected_scientific_outputs.json](../expected_scientific_outputs.json).
5. Separate post hoc diagnostics and comparison of 27 artifacts with [revision/expected_outputs.json](../revision/expected_outputs.json).

Both verification JSON files named in the README must report `passed`. These artifact counts are designated hash comparisons, not total generated files. Full-budget reference identity, matching first batches and outcome-access accounting are also checked.

Default output directories `study/outputs/reproduction/` and `revision/outputs/` must be empty. Use a fresh checkout for another full run. Artifacts under `study/outputs/run_a/` and `revision/reference_outputs/` are recorded references, not freshly generated results.

## Individual commands

After installing the pinned dependencies:

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 PYTHONHASHSEED=0
PYTHONPATH=study .venv/bin/python -m pytest -q study/tests \
  --ignore=study/tests/test_arkansas_extraction.py
.venv/bin/python reproduce.py --output study/outputs/reproduction
.venv/bin/python run_diagnostics.py --output revision/outputs
```

Each Python runner accepts its own output path. Arguments supplied to `run_reviewer.sh` are forwarded to `reproduce.py`; diagnostics still use their default directory. One custom `--output` does not change both locations.

## Integrity and interpretation

Repository manifests distinguish current file bytes from historical locks. A hash mismatch is a failed check: inspect the named file, environment and provenance rather than replacing expected hashes to make a run pass. Byte identity was validated in the recorded environment and is not promised for arbitrary Python or numerical-library versions.

The default command excludes 16 Arkansas extraction tests because their source PDFs are omitted. See [raw-source audit instructions](data.md#optional-raw-extraction-audit). This exclusion does not mean those tests were rerun in the portable package.

A successful execution demonstrates reproduction of designated computations. It does not independently confirm source measurements, establish prospective performance, add biological replication or create a new blinded study. Source and extraction evidence are separate from computational tests.
