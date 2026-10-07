"""Entry point for the MIMIC-IV temporal transport analysis."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "mimic_temporal_analysis.py"
RELABEL = ROOT / "src" / "relabel_phenylephrine_a1.py"


def command(stage: str, reps: int, workers: int) -> list[str]:
    if stage == "relabel":
        return [sys.executable, str(RELABEL)]
    return [
        sys.executable,
        str(ENGINE),
        "--stage", stage,
        "--reps", str(reps),
        "--workers", str(workers),
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("validate", "point", "bootstrap", "relabel"), default="point")
    parser.add_argument("--reps", type=int, default=500)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    if args.stage != "relabel":
        required = ROOT / "data" / "mimic_temporal" / "00_data_lock_manifest.json"
        if not required.exists():
            raise FileNotFoundError(
                "Run prepare_mimic_analysis_inputs.py with credentialed local data first. "
                f"Missing: {required}"
            )
    env = os.environ.copy()
    env["CNB_REPO_ROOT"] = str(ROOT)
    env.setdefault("MIMIC_TEMPORAL_DATA_DIR", "data/mimic_temporal")
    env.setdefault("MIMIC_TEMPORAL_RESULTS_DIR", "results/mimic_temporal")
    subprocess.run(command(args.stage, args.reps, args.workers), env=env, check=True)


if __name__ == "__main__":
    main()
