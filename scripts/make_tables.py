import argparse

from src.evaluation.tables import build

p = argparse.ArgumentParser(description="Tables, statistics and figures from results/metrics.")
p.add_argument("--metrics", default="results/metrics")
p.add_argument("--out", default="results/tables")
p.add_argument("--tuning", default="results/tuning.json")
p.add_argument("--bootstrap", type=int, default=10_000)
a = p.parse_args()
print(build(a.metrics, a.out, a.tuning, a.bootstrap))
