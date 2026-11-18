import argparse

from src.evaluation.controls import build

p = argparse.ArgumentParser(description="Descriptive tables for the control experiments.")
p.add_argument("--metrics", default="results/metrics")
p.add_argument("--out", default="results/tables")
p.add_argument("--tags", nargs="+", default=["S4", "S2", "S1", "S3", "X1", "X2", "X4"])
a = p.parse_args()
build(a.metrics, a.out, tuple(a.tags))
print(open(f"{a.out}/controls.md", encoding="utf-8").read())
