"""Run static checks before making the code repository public."""

from __future__ import annotations

import json
import py_compile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    "README.md",
    "LICENSE",
    "CITATION.cff",
    "environment.yml",
    "src/simulation_engine.py",
    "src/mimic_temporal_analysis.py",
    "scripts/prepare_mimic_analysis_inputs.py",
    "scripts/run_paper_simulations.py",
    "scripts/run_mimic_analysis.py",
    "data/mimic_temporal/README.md",
]


def main() -> None:
    missing = [item for item in REQUIRED if not (ROOT / item).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required repository files: {missing}")
    for path in (ROOT / "src").glob("*.py"):
        py_compile.compile(str(path), doraise=True)
    for path in (ROOT / "scripts").glob("*.py"):
        py_compile.compile(str(path), doraise=True)
    json.loads((ROOT / "config" / "paper_configuration.json").read_text(encoding="utf-8"))
    print("Repository static validation passed.")


if __name__ == "__main__":
    main()
