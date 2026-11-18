"""Control X4 (PROTOCOL §9): variant R = L_sup + lambda_R * L_R, L_R = mean((laplacian psi)^2) on the
grid interior. Run because H1 is supported.

Grid rule fixed in advance: lambda_R such that lambda_R * L_R(true psi of the training set) equals 1%,
10% and 100% of the final training L_sup of the selected V0 tuning run. The training set is the first
N = 2 000 samples used in tuning. lambda_R is tuned at N = 2 000, seed 100, on val_select, with lr and
wd of the tuned V0 (as V3 inherits them), then 5 seeds (42-46). Results in results/tuning_x4.json.
"""
import json
from pathlib import Path

import numpy as np
import yaml

from src.training.trainer import TrainConfig, run

N, TUNE_SEED, SEEDS, FRACTIONS = 2_000, 100, range(42, 47), (0.01, 0.1, 1.0)
EXTENT, GRID = 3.0, 128


def curvature_of_truth(path="data/train/psi.npy", n=N, chunk=250):
    """Mean over the first n training samples of mean((laplacian psi)^2) on indices 2..125 (no padding needed)."""
    psi, d, acc = np.load(path, mmap_mode="r"), 2.0 * EXTENT / (GRID - 1), 0.0
    for i in range(0, n, chunk):
        p = np.array(psi[i:i + chunk, 0], dtype=np.float64)
        lap = (p[:, 2:, 1:-1] + p[:, :-2, 1:-1] + p[:, 1:-1, 2:] + p[:, 1:-1, :-2] - 4 * p[:, 1:-1, 1:-1]) / d ** 2
        acc += (lap[:, 1:-1, 1:-1] ** 2).mean(axis=(1, 2)).sum()
    return acc / n


def lambda_grid(tuned, runs_root="runs"):
    v0 = tuned["V0"]
    hist = json.loads((Path(runs_root) / f"tune_lr{v0['lr']:g}_wd{v0['wd']:g}_V0_N{N}_s{TUNE_SEED}" / "history.json").read_text())
    l_sup, l_r = hist[-1]["train_sup"], curvature_of_truth()
    return float(l_sup), float(l_r), [float(f * l_sup / l_r) for f in FRACTIONS]


if __name__ == "__main__":
    tuned = yaml.safe_load(open("configs/tuned.yaml"))
    hp = dict(lr=tuned["V0"]["lr"], wd=tuned["V0"]["wd"])
    l_sup, l_r, grid = lambda_grid(tuned)
    print(f"final training L_sup of the selected V0 tuning run {l_sup:.5f}; L_R of the true psi {l_r:.5f}")
    rows = []
    for f, lam in zip(FRACTIONS, grid):
        meta = run(TrainConfig("R", N, TUNE_SEED, **hp, lambda_r=lam, tag=f"X4tune_f{f:g}", save_ckpt=False))
        rows.append(dict(fraction=f, lambda_r=lam, val_select=meta["best"]["val_select"],
                         step=meta["best"]["step"], seconds=meta["seconds"]))
        print("->", rows[-1], flush=True)
    best = min(rows, key=lambda r: r["val_select"])
    print(f"== R: selected lambda_R {best['lambda_r']:.3g} (fraction {best['fraction']:g})", flush=True)
    Path("results/tuning_x4.json").write_text(json.dumps(dict(l_sup_final_v0=l_sup, l_r_truth=l_r, runs=rows,
                                                              selected=best), indent=1))
    for seed in SEEDS:
        run(TrainConfig("R", N, seed, **hp, lambda_r=best["lambda_r"], tag="X4"))
