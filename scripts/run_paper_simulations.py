"""Run the simulation configurations reported in manuscript Tables 1--3.

``--mode smoke`` is an implementation check. ``--mode paper`` uses the
manuscript sample sizes and requested number of Monte Carlo repetitions.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ENGINE = ROOT / "src" / "simulation_engine.py"
OUTPUT = ROOT / "results" / "simulations"


def run(label: str, output_dir: Path, reps: int, n_source: int, learners: str, overlap: float) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(ENGINE),
        "--label", label,
        "--reps", str(reps),
        "--rep-start", "0",
        "--n-s", str(n_source),
        "--n-t", "400",
        "--folds", "5",
        "--intervals", "128",
        "--s-learners", learners,
        "--scenarios", "linear,strong_nonlinear",
        "--ratio-learners", "calibration",
        "--source-mode", "shifted",
        "--overlap-strength", str(overlap),
        "--weight-cap", "8",
        "--calibration-basis", "linear",
        "--calibration-ridge", "0.05",
        "--calibration-ridge-rate", "fixed",
        "--rsf-jobs", "1",
        "--rsf-trees", "300",
        "--rsf-min-leaf", "15",
        "--rsf-prediction-mode", "survival",
        "--gbsa-trees", "150",
        "--gbsa-learning-rate", "0.04",
        "--gbsa-max-depth", "2",
        "--gbsa-min-leaf", "15",
        "--stack-inner-folds", "2",
        "--output-dir", str(output_dir),
    ]
    env = os.environ.copy()
    # This is needed on some Windows installations that load both MKL and
    # PyTorch OpenMP runtimes. It does not change random seeds or estimands.
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("smoke", "paper"), default="smoke")
    parser.add_argument("--reps", type=int, default=None)
    args = parser.parse_args()

    reps = args.reps if args.reps is not None else (2 if args.mode == "smoke" else 500)
    if reps < 1:
        raise ValueError("--reps must be positive")

    # Table 1: Linear Cox source-size analysis.
    for n_source in (1200, 2400, 3600, 4000):
        run(
            label=f"table1_nS{n_source}_r{reps}",
            output_dir=OUTPUT / "table1_source_size" / f"nS{n_source}",
            reps=reps,
            n_source=n_source,
            learners="linear_cox",
            overlap=1.0,
        )

    # Table 2: fixed-size comparison of the four event-time learners.
    run(
        label=f"table2_models_r{reps}",
        output_dir=OUTPUT / "table2_model_comparison",
        reps=reps,
        n_source=2400,
        learners="linear_cox,rsf,gbsa,survival_stack",
        overlap=1.0,
    )

    # Table 3: overlap sensitivity using the prespecified survival stack.
    for value, name in ((0.5, "better"), (1.0, "reference"), (1.5, "poorer")):
        run(
            label=f"table3_{name}_overlap_r{reps}",
            output_dir=OUTPUT / "table3_overlap" / name,
            reps=reps,
            n_source=2400,
            learners="survival_stack",
            overlap=value,
        )


if __name__ == "__main__":
    main()
