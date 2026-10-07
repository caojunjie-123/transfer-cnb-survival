"""Create manuscript Table 1, covariate-shift diagnostics, and a consistency audit.

All diagnostics are computed from the frozen final analysis files.  Target
treatment and outcome fields are never read.  The source transport ratios are
the final fitted ratios saved with the primary two-sample contributions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("CNB_REPO_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / os.environ.get("MIMIC_TEMPORAL_DATA_DIR", "data/mimic_temporal")
RESULTS = ROOT / os.environ.get("MIMIC_TEMPORAL_RESULTS_DIR", "results/mimic_temporal")
RELABEL = ROOT / os.environ.get("MIMIC_TEMPORAL_RELABEL_DIR", "results/mimic_temporal_relabelled")
TABLES = ROOT / "results" / "tables"
FIGURES = ROOT / "results" / "figures"
TABLES.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 8,
    }
)

LABELS_CN = {
    "anchor_age": "年龄",
    "female": "女性",
    "bicarbonate": "碳酸氢根",
    "creatinine": "肌酐",
    "hemoglobin": "血红蛋白",
    "lactate": "乳酸",
    "pH": "pH",
    "platelets": "血小板",
    "potassium": "钾",
    "sodium": "钠",
    "arterial_dbp": "动脉舒张压",
    "arterial_sbp": "动脉收缩压",
    "heart_rate": "心率",
    "resp_rate": "呼吸频率",
    "map_any": "MAP记录可得",
    "low_bp_signal": "低血压信号（SBP<90）",
    "consciousness_any": "意识记录可得",
    "gcs_any": "GCS记录可得",
    "rass_any": "RASS记录可得",
    "prior_any_pressor": "t0前升压药暴露",
    "bicarbonate__missing": "碳酸氢根缺失",
    "creatinine__missing": "肌酐缺失",
    "hemoglobin__missing": "血红蛋白缺失",
    "lactate__missing": "乳酸缺失",
    "pH__missing": "pH缺失",
    "platelets__missing": "血小板缺失",
    "potassium__missing": "钾缺失",
    "sodium__missing": "钠缺失",
    "arterial_dbp__missing": "动脉舒张压缺失",
    "arterial_sbp__missing": "动脉收缩压缺失",
    "heart_rate__missing": "心率缺失",
    "resp_rate__missing": "呼吸频率缺失",
}

LABELS_EN = {
    "anchor_age": "Age",
    "female": "Female",
    "bicarbonate": "Bicarbonate",
    "creatinine": "Creatinine",
    "hemoglobin": "Hemoglobin",
    "lactate": "Lactate",
    "pH": "pH",
    "platelets": "Platelets",
    "potassium": "Potassium",
    "sodium": "Sodium",
    "arterial_dbp": "Arterial DBP",
    "arterial_sbp": "Arterial SBP",
    "heart_rate": "Heart rate",
    "resp_rate": "Respiratory rate",
    "map_any": "MAP recorded",
    "low_bp_signal": "Low BP signal",
    "consciousness_any": "Consciousness recorded",
    "gcs_any": "GCS recorded",
    "rass_any": "RASS recorded",
    "prior_any_pressor": "Prior pressor before t0",
}

BINARY = {
    "female", "map_any", "low_bp_signal", "consciousness_any", "gcs_any",
    "rass_any", "prior_any_pressor",
}


def smd(source: np.ndarray, target: np.ndarray, weights: np.ndarray | None = None) -> float:
    pooled = np.sqrt((np.var(source, ddof=1) + np.var(target, ddof=1)) / 2)
    if pooled == 0 or not np.isfinite(pooled):
        return np.nan
    source_mean = np.average(source, weights=weights) if weights is not None else np.mean(source)
    return float((np.mean(target) - source_mean) / pooled)


def fmt_cont(values: pd.Series, variable: str) -> str:
    x = values.loc[values.notna()].astype(float)
    if len(x) == 0:
        return "—"
    mean, sd = x.mean(), x.std(ddof=1)
    digits = 2 if variable == "pH" else 1
    return f"{mean:.{digits}f} ({sd:.{digits}f})"


def fmt_binary(values: pd.Series) -> str:
    x = values.loc[values.notna()].astype(float)
    if len(x) == 0:
        return "—"
    n = int((x == 1).sum())
    return f"{n:,} ({100 * n / len(x):.1f}%)"


def build_final_balance(source: pd.DataFrame, target: pd.DataFrame, features: list[str], ratios: pd.Series) -> pd.DataFrame:
    combined = pd.concat([source[features], target[features]], ignore_index=True).copy()
    for c in features:
        if c.endswith("__missing"):
            continue
        if combined[c].isna().any():
            combined[c] = combined[c].fillna(combined[c].median())
    s = combined.iloc[: len(source)]
    t = combined.iloc[len(source) :]
    rows = []
    w = ratios.to_numpy(float)
    for c in features:
        sv = s[c].to_numpy(float)
        tv = t[c].to_numpy(float)
        rows.append(
            {
                "variable": c,
                "SMD_before": smd(sv, tv),
                "SMD_after": smd(sv, tv, w),
            }
        )
    return pd.DataFrame(rows)


def make_table1(source: pd.DataFrame, target: pd.DataFrame, features: list[str], balance: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in features:
        is_binary = c in BINARY
        rows.append(
            {
                "Variable": LABELS_CN.get(c, c),
                "Field": c,
                "Source (n=6,256)": fmt_binary(source[c]) if is_binary else fmt_cont(source[c], c),
                "Target (n=1,248)": fmt_binary(target[c]) if is_binary else fmt_cont(target[c], c),
                "Source missing (%)": round(100 * source[c].isna().mean(), 1),
                "Target missing (%)": round(100 * target[c].isna().mean(), 1),
                "SMD before": balance.loc[balance.variable.eq(c), "SMD_before"].iloc[0],
                "SMD after": balance.loc[balance.variable.eq(c), "SMD_after"].iloc[0],
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TABLES / "table1_covariates.csv", index=False, encoding="utf-8-sig")
    (TABLES / "table1_covariates.md").write_text(
        "# Table 1. Baseline pre-treatment covariates\n\n"
        "Continuous variables are mean (SD); binary variables are n (%). "
        "SMD is the target-minus-source standardized mean difference; SMD after "
        "uses the final source transport ratios. Missingness is reported separately.\n\n"
        + out.to_markdown(index=False, floatfmt=".3f")
        + "\n",
        encoding="utf-8",
    )
    return out


def make_balance_figure(balance: pd.DataFrame, base_features: list[str], indicator_features: list[str]) -> None:
    # prior_any_pressor is constant after the pre-t0 exposure exclusion and
    # therefore has no estimable SMD; omit it from the visual panel.
    base = balance.loc[balance.variable.isin(base_features) & balance.variable.ne("prior_any_pressor")].copy()
    miss = balance.loc[balance.variable.isin(indicator_features)].copy()
    panels = [(base, "A  Baseline covariates"), (miss, "B  Missingness indicators")]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 8.4), sharex=True)
    xlim = max(0.20, float(np.nanmax(np.abs(balance[["SMD_before", "SMD_after"]].to_numpy()))) * 1.15)
    for ax, (panel, title) in zip(axes, panels):
        panel = panel.assign(order=panel.SMD_before.abs()).sort_values("order")
        y = np.arange(len(panel))
        for yi, (_, row) in enumerate(panel.iterrows()):
            ax.plot([row.SMD_before, row.SMD_after], [yi, yi], color="#9AA5B1", linewidth=1.0, zorder=1)
        ax.scatter(panel.SMD_before, y, s=24, color="#52606D", zorder=2)
        ax.scatter(panel.SMD_after, y, s=28, color="#1F77B4", zorder=3)
        ax.axvline(0, color="black", linewidth=0.7)
        ax.axvline(0.10, color="#B7791F", linestyle="--", linewidth=0.7)
        ax.axvline(-0.10, color="#B7791F", linestyle="--", linewidth=0.7)
        def pretty_label(v: str) -> str:
            if v.endswith("__missing"):
                base_name = v[: -len("__missing")]
                return f"{LABELS_EN.get(base_name, base_name)} missing"
            return LABELS_EN.get(v, v)

        ax.set_yticks(y, [pretty_label(v) for v in panel.variable])
        ax.invert_yaxis()
        ax.set_xlim(-xlim, xlim)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Standardized mean difference")
        ax.grid(axis="x", alpha=0.18, linewidth=0.6)
    axes[0].legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#52606D", markeredgecolor="#52606D", label="Before weighting"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#1F77B4", markeredgecolor="#1F77B4", label="After weighting"),
        ],
        loc="upper right",
        frameon=False,
    )
    fig.suptitle("Source–target covariate balance", y=0.995, fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(FIGURES / "mimic_temporal_covariate_balance.png", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "mimic_temporal_covariate_balance.tiff", dpi=600, bbox_inches="tight")
    fig.savefig(FIGURES / "mimic_temporal_covariate_balance.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "mimic_temporal_covariate_balance.svg", bbox_inches="tight")
    plt.close(fig)
    (FIGURES / "mimic_temporal_covariate_balance_caption.md").write_text(
        "# Figure: Source–target covariate balance\n\n"
        "Panel A shows standardized mean differences for the 20 prespecified "
        "pre-treatment covariates; panel B shows the corresponding missingness "
        "indicators. Gray points are unweighted and blue points are after final "
        "source-to-target transport weighting. Dashed lines mark |SMD|=0.10.\n",
        encoding="utf-8",
    )


def make_main_results() -> pd.DataFrame:
    result = pd.read_csv(RELABEL / "primary_and_model_sensitivity_results_relabelled.csv")
    labels = {
        "primary_super_learner": "Survival Super Learner",
        "censor_A_only": "Censoring A-only",
        "linear_cox": "Linear Cox",
        "rsf": "RSF",
        "gbsa": "GBSA",
    }
    result["Model"] = result.variant.map(labels)
    out = result[["Model", "n_source", "n_target", "transported_cnb", "two_sample_eif_se", "wald_ci_low", "wald_ci_high", "source_only_cnb", "ratio_ess_fraction", "ratio_q99", "ratio_max"]].copy()
    out.to_csv(TABLES / "main_results_table.csv", index=False, encoding="utf-8-sig")
    (TABLES / "main_results_table.md").write_text(
        "# Main transported CNB results\n\n"
        + out.to_markdown(index=False, floatfmt=".4f")
        + "\n",
        encoding="utf-8",
    )
    return out


def consistency_audit(source: pd.DataFrame, target: pd.DataFrame, balance: pd.DataFrame, main_results: pd.DataFrame) -> dict:
    counts = pd.read_csv(DATA / "formal_analysis_counts.csv").set_index("role")
    flow_path = FIGURES / "mimic_temporal_cohort_flow_counts.csv"
    flow = pd.read_csv(flow_path) if flow_path.exists() else None
    bootstrap_path = RESULTS / "bootstrap_full_refit_interval.csv"
    bootstrap = pd.read_csv(bootstrap_path).iloc[0] if bootstrap_path.exists() else None
    target_outcome_cols = sorted(set(target.columns) & {"A", "U_days", "Delta", "event", "censor"})
    primary = main_results.loc[main_results.Model.eq("Survival Super Learner")].iloc[0]
    checks = {
        "source_n_matches_formal_data": int(len(source)) == int(counts.loc["source", "n"]),
        "target_n_matches_formal_data": int(len(target)) == int(counts.loc["target", "n"]),
        "source_events_match": int(source.Delta.sum()) == int(counts.loc["source", "events_14d"]),
        "target_outcome_treatment_absent": not target_outcome_cols,
        "flow_counts_available_or_not_requested": flow is None or (
            int(flow.loc[flow.stage.eq("Final source after timestamp audit"), "count"].iloc[0]) == len(source)
            and int(flow.loc[flow.stage.eq("Final target X-only analysis cohort"), "count"].iloc[0]) == len(target)
        ),
        "bootstrap_500_successful_or_not_requested": bootstrap is None or (
            int(bootstrap.requested_replicates) == 500
            and int(bootstrap.successful_replicates) == 500
            and int(bootstrap.failed_replicates) == 0
        ),
        "primary_bootstrap_reproduces_point_sign_or_not_requested": bootstrap is None or (
            np.sign(float(primary.transported_cnb)) == -np.sign(float(bootstrap.percentile_ci_low))
        ),
        "transport_balance_gate": float(balance.SMD_after.abs().max()) < 0.10,
        "figure_outputs_present": all((FIGURES / f"mimic_temporal_covariate_balance.{ext}").exists() for ext in ["png", "tiff", "pdf", "svg"]),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    report = {
        "checks": checks,
        "all_pass": bool(all(checks.values())),
        "source_n": int(len(source)),
        "target_n": int(len(target)),
        "source_events": int(source.Delta.sum()),
        "transport_max_abs_smd_before": float(balance.SMD_before.abs().max()),
        "transport_max_abs_smd_after": float(balance.SMD_after.abs().max()),
        "primary_transported_cnb_presentation": float(primary.transported_cnb),
        "primary_source_only_cnb_presentation": float(primary.source_only_cnb),
        "bootstrap_percentile_ci_presentation": (
            [float(-bootstrap.percentile_ci_high), float(-bootstrap.percentile_ci_low)]
            if bootstrap is not None else None
        ),
    }
    (RESULTS / "final_consistency_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# Final consistency audit (2026-09-19)", "", f"All checks passed: **{report['all_pass']}**", "", "| Check | Pass |", "|---|---:|"]
    lines += [f"| {key} | {'PASS' if value else 'FAIL'} |" for key, value in checks.items()]
    lines += ["", f"Final source/target: {len(source):,}/{len(target):,}; source 14-day events: {int(source.Delta.sum()):,}.", f"Final transport balance max |SMD|: {balance.SMD_after.abs().max():.4f}."]
    (RESULTS / "final_consistency_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> None:
    manifest = json.loads((DATA / "00_data_lock_manifest.json").read_text(encoding="utf-8"))
    source = pd.read_parquet(DATA / "source_X_A_U_Delta.parquet")
    target = pd.read_parquet(DATA / "target_X_only.parquet")
    contribution = pd.read_parquet(RESULTS / "primary_two_sample_contributions.parquet")
    ratio_map = contribution.loc[contribution.role.eq("source"), ["stay_id", "ratio"]].set_index("stay_id")["ratio"]
    ratios = source.stay_id.map(ratio_map)
    if ratios.isna().any():
        raise RuntimeError("Final source transport ratios do not cover every source stay.")
    base_features = [c for c in manifest["base_features"] if c in source.columns]
    indicator_features = [c for c in manifest["missingness_indicator_features"] if c in source.columns]
    all_features = base_features + indicator_features
    balance = build_final_balance(source, target, all_features, ratios)
    balance.to_csv(TABLES / "transport_balance_final.csv", index=False, encoding="utf-8-sig")
    table1_features = [c for c in base_features if c != "prior_any_pressor"]
    table1 = make_table1(source, target, table1_features, balance)
    table1_en = table1.copy()
    table1_en["Variable"] = table1_en["Field"].map(lambda c: LABELS_EN.get(c, c))
    table1_en.to_csv(TABLES / "table1_covariates_en.csv", index=False, encoding="utf-8-sig")
    (TABLES / "table1_covariates_en.md").write_text(
        "# Table 1. Baseline pre-treatment covariates\n\n"
        "Continuous variables are mean (SD); binary variables are n (%). "
        "SMD is the target-minus-source standardized mean difference; SMD after "
        "uses the final source transport ratios. Missingness is reported separately.\n\n"
        + table1_en.to_markdown(index=False, floatfmt=".3f")
        + "\n",
        encoding="utf-8",
    )
    make_balance_figure(balance, base_features, indicator_features)
    main_results = make_main_results()
    audit = consistency_audit(source, target, balance, main_results)
    print(f"Table 1 rows: {len(table1)}")
    print(f"Final balance max |SMD| before/after: {balance.SMD_before.abs().max():.4f}/{balance.SMD_after.abs().max():.4f}")
    print(f"Consistency audit all_pass={audit['all_pass']}")


if __name__ == "__main__":
    main()
