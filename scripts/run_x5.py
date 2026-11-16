import argparse

from src.fitting.parametric import N_STARTS, STEPS, X5_SAMPLES, X5_SETS, run_set, summarize
from src.utils.config import SimulationConfig

p = argparse.ArgumentParser(description="X5: parametric reference fit (PROTOCOL §9).")
p.add_argument("--sets", nargs="+", default=list(X5_SETS),
               help="default: the registered X5 sets; 'val' is allowed for a dry run")
p.add_argument("--n-samples", type=int, default=X5_SAMPLES)
p.add_argument("--starts", type=int, default=N_STARTS)
p.add_argument("--steps", type=int, default=STEPS)
p.add_argument("--batch", type=int, default=5, help="samples fitted in parallel")
p.add_argument("--final", action="store_true", help="required for any set other than val")
p.add_argument("--out", default="results/x5")
p.add_argument("--summary-only", action="store_true")
a = p.parse_args()

if not a.summary_only:
    cfg = SimulationConfig.from_yaml()
    for s in a.sets:
        run_set(cfg, s, out_root=a.out, final=a.final, n_samples=a.n_samples, batch=a.batch,
                n_starts=a.starts, steps=a.steps)
print(summarize(a.out, sets=a.sets))
