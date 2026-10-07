"""Relabel the already-computed point estimates with phenylephrine as A=1.

Swapping binary treatment labels changes the sign of the contrast and reverses
confidence-interval endpoints; it does not change the underlying analysis.
"""
import os
from pathlib import Path
import json
import pandas as pd

ROOT = Path(os.environ.get("CNB_REPO_ROOT", Path(__file__).resolve().parents[1]))
SRC = ROOT / os.environ.get("MIMIC_TEMPORAL_RESULTS_DIR", "results/mimic_temporal")
OUT = ROOT / os.environ.get("MIMIC_TEMPORAL_RELABEL_DIR", "results/mimic_temporal_relabelled")
OUT.mkdir(exist_ok=True)

df = pd.read_csv(SRC / "primary_and_model_sensitivity_results.csv")
for c in ["target_plugin", "source_eif_correction", "transported_cnb", "source_only_cnb"]:
    df[c] = -df[c]
for lo, hi in [("wald_ci_low", "wald_ci_high"), ("source_only_ci_low", "source_only_ci_high")]:
    old_lo, old_hi = df[lo].copy(), df[hi].copy()
    df[lo], df[hi] = -old_hi, -old_lo
df.to_csv(OUT / "primary_and_model_sensitivity_results_relabelled.csv", index=False)

# Apply the same sign reversal to a completed full-refit bootstrap interval.
bootstrap_path = SRC / "bootstrap_full_refit_interval.csv"
if bootstrap_path.exists():
    bootstrap = pd.read_csv(bootstrap_path)
    if {"percentile_ci_low", "percentile_ci_high"}.issubset(bootstrap.columns):
        old_low = bootstrap["percentile_ci_low"].copy()
        bootstrap["percentile_ci_low"] = -bootstrap["percentile_ci_high"]
        bootstrap["percentile_ci_high"] = -old_low
    bootstrap.to_csv(OUT / "bootstrap_full_refit_interval_relabelled.csv", index=False)

meta = {
    "relabelled": True,
    "A1": "phenylephrine",
    "A0": "norepinephrine",
    "source_A1_n": 3899,
    "source_A0_n": 2357,
    "relationship_to_original": "All treatment contrasts and confidence-interval endpoints are sign-reversed; standard errors and weights are unchanged.",
    "bootstrap_interval_relabelled": bootstrap_path.exists(),
}
(OUT / "treatment_label_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
print(df.to_string(index=False))
