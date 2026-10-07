# MIMIC-IV analysis input contract

This directory must remain empty in the public repository. It is populated
locally by `scripts/prepare_mimic_analysis_inputs.py` after the researcher has
obtained credentialed MIMIC-IV access and constructed the cohort.

## Required pre-lock source columns

`stay_id`, `subject_id`, `hadm_id`, `A`, `U_days`, `Delta`, and the baseline
covariates below:

```
anchor_age, female, bicarbonate, creatinine, hemoglobin, lactate, pH,
platelets, potassium, sodium, arterial_dbp, arterial_sbp, heart_rate,
resp_rate, map_any, low_bp_signal, consciousness_any, gcs_any, rass_any,
prior_any_pressor
```

`A=1` denotes norepinephrine and `A=0` denotes phenylephrine in the
computational input. `U_days` is follow-up from the first eligible infusion,
and `Delta=1` denotes in-hospital death by 14 days. Live discharge before day
14 is treated as right censoring; observation through day 14 is administrative
censoring.

## Required pre-lock target columns

`stay_id`, `subject_id`, `hadm_id`, and exactly the baseline covariates listed
above. The target file must not contain treatment, outcome, follow-up, death,
or discharge columns.

## Covariate timing and temporal populations

Use the first eligible continuous infusion initiated within 24 hours of ICU
admission as time zero. Time-varying laboratory and physiologic covariates are
the last values with chart time and store time before time zero, within the
preceding six hours. Measurement-availability indicators use the same window;
the low-blood-pressure indicator is any recorded systolic blood pressure below
90 mmHg. The temporal source population is 2014--2019, and the target
population is 2020--2022 after anchor-year alignment.

The preparation script creates the following local-only files:

* `source_X_A_U_Delta.parquet`
* `target_X_only.parquet`
* `formal_analysis_counts.csv`
* `00_data_lock_manifest.json`
