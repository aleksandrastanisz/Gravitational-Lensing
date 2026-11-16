import argparse

from src.evaluation.figures import plot_curves, plot_reconstructions
from src.utils.config import SimulationConfig

p = argparse.ArgumentParser(description="Validation curves and example reconstructions.")
p.add_argument("--set", default="val", help="val, test_id, ood/<axis>_L<level> or diag/Kneg_L<level>")
p.add_argument("--n", type=int, default=2000)
p.add_argument("--seed", type=int, default=42)
p.add_argument("--variants", nargs="+", default=["V0", "V3"])
p.add_argument("--indices", nargs="+", type=int, default=None,
               help="sample indices; default: 10th, 50th and 90th percentile of the first variant's M1")
p.add_argument("--final", action="store_true", help="required for any set other than val")
p.add_argument("--out", default="results/figures")
p.add_argument("--skip-curves", action="store_true")
a = p.parse_args()

if not a.skip_curves:
    print(plot_curves(out_dir=a.out))
print(plot_reconstructions(SimulationConfig.from_yaml(), a.set, a.n, a.seed, tuple(a.variants), a.indices,
                           out_dir=a.out, final=a.final))
