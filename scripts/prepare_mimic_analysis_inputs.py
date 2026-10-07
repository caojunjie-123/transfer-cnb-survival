"""Lock analysis-ready files for the temporal MIMIC-IV transport analysis.

The script expects cohort files that have already been constructed from
credentialed MIMIC-IV data using the cohort and time-zero definitions in
``data/mimic_temporal/README.md``.  It never estimates a treatment effect.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


BASE_FEATURES = [
    "anchor_age", "female", "bicarbonate", "creatinine", "hemoglobin",
    "lactate", "pH", "platelets", "potassium", "sodium", "arterial_dbp",
    "arterial_sbp", "heart_rate", "resp_rate", "map_any", "low_bp_signal",
    "consciousness_any", "gcs_any", "rass_any", "prior_any_pressor",
]
IDENTIFIERS = ["subject_id", "hadm_id", "stay_id"]
SOURCE_REQUIRED = [*IDENTIFIERS, "A", "U_days", "Delta", *BASE_FEATURES]
TARGET_REQUIRED = [*IDENTIFIERS, *BASE_FEATURES]
FORBIDDEN_TARGET_FIELDS = {
    "a", "u_days", "delta", "drug", "deathtime", "dischtime",
    "hospital_expire_flag", "death_14d_in_hospital", "followup_days",
    "outcome", "event", "censoring",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", required=True, type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "mimic_temporal",
    )
    args = parser.parse_args()

    source = pd.read_parquet(args.source)
    target = pd.read_parquet(args.target)
    require_columns(source, SOURCE_REQUIRED, "Source file")
    require_columns(target, TARGET_REQUIRED, "Target file")

    forbidden = sorted(FORBIDDEN_TARGET_FIELDS & {str(x).lower() for x in target.columns})
    if forbidden:
        raise ValueError(
            "Target file must contain covariates only; forbidden fields found: "
            f"{forbidden}"
        )
    if source.stay_id.duplicated().any() or target.stay_id.duplicated().any():
        raise ValueError("Each input must have one row per ICU stay.")
    if set(source.stay_id) & set(target.stay_id):
        raise ValueError("Source and target stay IDs must not overlap.")
    if not source.A.isin([0, 1]).all() or not source.Delta.isin([0, 1]).all():
        raise ValueError("Source A and Delta must be binary 0/1 variables.")
    if source.U_days.isna().any() or (source.U_days <= 0).any():
        raise ValueError("Source U_days must be positive and non-missing.")

    missing_features = [
        feature for feature in BASE_FEATURES
        if source[feature].isna().any() or target[feature].isna().any()
    ]
    indicator_features = [f"{feature}__missing" for feature in missing_features]
    for feature in missing_features:
        source[f"{feature}__missing"] = source[feature].isna().astype(int)
        target[f"{feature}__missing"] = target[feature].isna().astype(int)

    source = source[[*IDENTIFIERS, "A", "U_days", "Delta", *BASE_FEATURES, *indicator_features]].copy()
    target = target[[*IDENTIFIERS, *BASE_FEATURES, *indicator_features]].copy()
    source = source.sort_values("stay_id").reset_index(drop=True)
    target = target.sort_values("stay_id").reset_index(drop=True)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = output_dir / "source_X_A_U_Delta.parquet"
    target_path = output_dir / "target_X_only.parquet"
    source.to_parquet(source_path, index=False)
    target.to_parquet(target_path, index=False)

    counts = pd.DataFrame(
        [
            {
                "role": "source",
                "n": len(source),
                "events_14d": int(source.Delta.sum()),
                "n_A1_norepinephrine": int(source.A.sum()),
                "n_A0_phenylephrine": int((1 - source.A).sum()),
            },
            {
                "role": "target",
                "n": len(target),
                "events_14d": "not_in_target_file",
                "n_A1_norepinephrine": "not_in_target_file",
                "n_A0_phenylephrine": "not_in_target_file",
            },
        ]
    )
    counts.to_csv(output_dir / "formal_analysis_counts.csv", index=False)

    manifest = {
        "design": "MIMIC-IV temporal transport: source 2014-2019 to target 2020-2022",
        "endpoint": "14-day in-hospital death after first eligible continuous vasopressor infusion",
        "censoring_rule": "alive hospital discharge before day 14; administrative truncation at day 14",
        "timestamp_quality_rule": "Exclude only if hospital_expire_flag=1 and resolved_death_time<t0",
        "n_source": len(source),
        "n_target": len(target),
        "source_events_14d": int(source.Delta.sum()),
        "source_treatment_counts": {
            "A1_norepinephrine": int(source.A.sum()),
            "A0_phenylephrine": int((1 - source.A).sum()),
        },
        "base_features": BASE_FEATURES,
        "missingness_indicator_features": indicator_features,
        "target_treatment_outcome_or_censoring_in_target_file": False,
        "source_file_sha256": sha256(source_path),
        "target_file_sha256": sha256(target_path),
        "estimation_run": False,
    }
    (output_dir / "00_data_lock_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
