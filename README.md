# Transported causal net benefit with survival data

This repository accompanies the manuscript *Transfer learning for causal net
benefit with survival data*. It contains the code used for the simulation study
and temporal MIMIC-IV case analysis. The primary implementation is a
cross-fitted one-step estimator of target-population causal net benefit (CNB)
with right-censored survival outcomes.

## What is included

* `src/simulation_engine.py` implements the simulation data-generating
  mechanisms, calibration density-ratio estimation, survival learners, and
  cross-fitted one-step estimator used for the manuscript simulations.
* `scripts/run_paper_simulations.py` runs the source-size, event-model, and
  overlap-sensitivity configurations underlying Tables 1--3. It has a short
  smoke-test mode and a 500-repetition paper mode.
* `src/mimic_temporal_analysis.py` implements the five-fold temporal transport
  analysis, including event-time, censoring, treatment, and density-ratio
  nuisance fits, EIF inference, and full-refit bootstrap.
* `scripts/prepare_mimic_analysis_inputs.py` locks analysis-ready source and
  target files, creates missingness indicators, and verifies that the target
  file contains covariates only.
* `src/relabel_phenylephrine_a1.py` converts the computational coding
  (norepinephrine = 1) to the manuscript contrast, phenylephrine minus
  norepinephrine.

No MIMIC-IV row-level data, patient identifiers, model checkpoints, or
row-level analysis results are included in this repository.

## Software environment

Create the environment from the repository root:

```bash
conda env create -f environment.yml
conda activate transported-cnb
```

The code was run with Python 3.13, NumPy 2.3.5, pandas 2.3.3,
scikit-learn 1.9.0, scikit-survival 0.28.0, and SciPy 1.16.3. Exact package
versions are recorded in each case-analysis execution manifest.

## Reproducing the simulations

Run a two-repetition implementation check:

```bash
python scripts/run_paper_simulations.py --mode smoke
```

Run the manuscript configurations (500 Monte Carlo repetitions per
configuration):

```bash
python scripts/run_paper_simulations.py --mode paper --reps 500
```

The paper mode writes separate result directories for the source-size analysis,
the four-model comparison, and each overlap configuration under
`results/simulations/`. It is computationally intensive and is intended for a
multi-core workstation or computing cluster.

## Reproducing the MIMIC-IV analysis

### Data access

MIMIC-IV is a credentialed-access database distributed through PhysioNet.
Researchers must obtain access and comply with its data-use agreement before
using the code. This repository does not redistribute MIMIC-IV data.

The temporal design uses adult ICU stays with an eligible first continuous
phenylephrine or norepinephrine infusion within 24 hours of ICU admission. The
source period is 2014--2019 and the target period is 2020--2022. The source
file contains baseline covariates, treatment, follow-up, and the event
indicator; the target file contains baseline covariates only. Detailed input
definitions are in `data/mimic_temporal/README.md`.

### Prepare the locked analysis files

After constructing the cohort from credentialed MIMIC-IV data, prepare two
local parquet files following the data contract and run:

```bash
python scripts/prepare_mimic_analysis_inputs.py \
  --source path/to/source_prelock.parquet \
  --target path/to/target_prelock.parquet
```

This creates the non-versioned files in `data/mimic_temporal/` that are read by
the estimator, together with a SHA-256 manifest. The target file is rejected
if it contains treatment, outcome, follow-up, death, or discharge fields.

### Estimate and report

```bash
# Point estimate, EIF inference, and nuisance-model sensitivity analyses
python scripts/run_mimic_analysis.py --stage point --workers 2

# Full-refit stratified bootstrap
python scripts/run_mimic_analysis.py --stage bootstrap --reps 500 --workers 4

# Re-express the contrast as phenylephrine minus norepinephrine
python scripts/run_mimic_analysis.py --stage relabel

# Create balance and descriptive reporting tables
python src/make_mimic_temporal_tables.py
```

The raw computational treatment convention is `A=1` for norepinephrine and
`A=0` for phenylephrine. The manuscript reports the sign-reversed contrast,
phenylephrine minus norepinephrine. This relabeling changes the sign of the
estimate and reverses confidence-interval endpoints; fitted nuisance models,
weights, and standard errors are unchanged.

## Reproducibility and reporting

* Random seeds, software versions, tuning constants, data hashes, fold counts,
  and output diagnostics are saved in `execution_manifest.json`.
* The case analysis does not read target treatment, outcome, or follow-up data.
* The `results/` directory is ignored by Git. Only aggregated tables and
  figures derived under the MIMIC-IV data-use agreement should be shared after
  local institutional review.

## License

The source code is released under the MIT License. The MIMIC-IV data are not
covered by this license and remain governed by the PhysioNet data-use agreement.

## Citation

Please cite the accompanying manuscript. A DOI and final bibliographic record
will be added after publication.
