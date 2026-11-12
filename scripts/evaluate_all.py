import argparse

from src.evaluation.runner import run_all
from src.utils.config import SimulationConfig

p = argparse.ArgumentParser(description="Per-sample metrics M1-M6 for every run with a checkpoint.")
p.add_argument("--final", action="store_true",
               help="evaluate test ID, OOD and diagnostic sets (final evaluation only)")
p.add_argument("--runs", default="runs")
p.add_argument("--out", default="results/metrics")
p.add_argument("--only", default=None, help="evaluate only runs whose name contains this text")
a = p.parse_args()
run_all(SimulationConfig.from_yaml(), a.runs, a.out, a.final, a.only)
