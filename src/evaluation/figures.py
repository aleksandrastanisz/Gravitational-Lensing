"""Figures: validation curves and example reconstructions with residual maps.

Validation curves need only numpy and matplotlib. Reconstructions need torch and a trained
checkpoint. Test, OOD and diagnostic sets can be drawn only after the final evaluation has
been run (marker file written by scripts/evaluate_all.py --final).
"""
import json
from pathlib import Path

import numpy as np

VARIANTS = ("V0", "V1", "V2", "V3")
SEEDS = tuple(range(42, 47))
QUANTILES = (0.1, 0.5, 0.9)      # examples are chosen by rule, not by eye
KAPPA_LIM = 1.0                  # kappa = 0.5 at the Einstein radius; the core saturates by design
PULL_LIM = 5.0
SRC_ZOOM = 1.0                   # half-width [arcsec] of the displayed source-plane window


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def run_dir(runs_root, v, n, s, tag=""):
    return Path(runs_root) / f"{tag + '_' if tag else ''}{v}_N{n}_s{s}"


# ------------------------------------------------------------------ validation curves

def load_histories(runs_root, n, tag=""):
    """variant -> list over seeds of history lists, for runs that have finished."""
    out = {}
    for v in VARIANTS:
        hs = [json.loads(p.read_text()) for s in SEEDS
              if (p := run_dir(runs_root, v, n, s, tag) / "history.json").exists()]
        if hs:
            out[v] = hs
    return out


def plot_curves(runs_root="runs", out_dir="results/figures", ns=(500, 2_000, 8_000),
                keys=("val_select", "val_src", "val_alpha")):
    """Mean over seeds (line) and min-max range (band) of the validation quantities per variant."""
    plt = _plt()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for key in keys:
        fig, axes = plt.subplots(1, len(ns), figsize=(4.2 * len(ns), 3.4), squeeze=False)
        drawn = False
        for ax, n in zip(axes[0], ns):
            for v, hs in load_histories(runs_root, n).items():
                hs = [h for h in hs if len(h) == len(hs[0]) and key in h[0]]
                if not hs:
                    continue
                steps = np.array([r["step"] for r in hs[0]])
                y = np.array([[r[key] for r in h] for h in hs])
                ax.plot(steps, y.mean(0), label=f"{v} ({len(hs)} seed{'' if len(hs) == 1 else 's'})")
                ax.fill_between(steps, y.min(0), y.max(0), alpha=0.2)
                drawn = True
            ax.set_yscale("log")
            ax.set_xlabel("optimizer step [10$^3$]")
            ax.xaxis.set_major_formatter(lambda x, _: f"{x / 1000:g}")
            ax.set_ylabel(key)
            ax.set_title(f"N = {n}")
            if ax.has_data():
                ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        if drawn:
            fig.savefig(out / f"curves_{key}.png", dpi=200)
            written.append(f"curves_{key}.png")
        plt.close(fig)
    return written


# ------------------------------------------------------------------ example reconstructions

def pick_indices(m1, quantiles=QUANTILES):
    """Indices of the samples at the given quantiles of the baseline M1 (deterministic rule)."""
    m1 = np.asarray(m1, dtype=float)
    order = np.argsort(m1, kind="stable")
    return [int(order[int(round(q * (len(m1) - 1)))]) for q in quantiles]


def check_access(set_name, final, metrics_root):
    """Only the validation set may be drawn before the final evaluation has been run."""
    if set_name == "val":
        return
    if not final:
        raise PermissionError(f"'{set_name}' is a test set: pass final=True (--final)")
    if not (Path(metrics_root) / "FINAL_EVALUATION.json").exists():
        raise PermissionError("final evaluation has not been run yet (no FINAL_EVALUATION.json); "
                              "run scripts/evaluate_all.py --final first")


def plot_reconstructions(cfg, set_name="val", n=2_000, seed=42, variants=("V0", "V3"), indices=None,
                         runs_root="runs", data_root="data", out_dir="results/figures",
                         metrics_root="results/metrics", final=False, device=None):
    """One figure per example: truth row and one row per variant.

    Columns: image (observed / re-lensed with A_model), source, convergence, deflection error
    with the M1 band outlined, and noise-normalized image residual.
    """
    check_access(set_name, final, metrics_root)
    import torch

    from src.data.store import NpyDataset
    from src.evaluation.metrics import ring_band, sample_metrics
    from src.evaluation.runner import KEYS, evaluate, load_model
    from src.physics.deflection import alpha_fields
    from src.physics.noise import estimate_sigma
    from src.physics.poisson import PoissonSolver2D
    from src.physics.ray_tracer import GravitationalRayTracer

    plt = _plt()
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(data_root) / set_name
    models = {v: load_model(run_dir(runs_root, v, n, seed), device) for v in variants}
    if indices is None:
        indices = pick_indices(evaluate(models[variants[0]], data_dir, cfg, device)["m1"])

    ds = NpyDataset(data_dir)
    tracer = GravitationalRayTracer(extent=cfg.extent, psf_fwhm=cfg.psf_fwhm)
    poisson = PoissonSolver2D(extent=cfg.extent).to(device)
    d = 2.0 * cfg.extent / (cfg.grid_size - 1)
    box = [-cfg.extent, cfg.extent, -cfg.extent, cfg.extent]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written = []

    for idx in indices:
        b = {k: torch.from_numpy(np.array(ds.a[k][idx:idx + 1])).to(device) for k in KEYS}
        band = ring_band(b["params"], cfg.grid_size, cfg.extent)[0].cpu().numpy()
        rows = []
        with torch.inference_mode():
            sigma = estimate_sigma(b["observed"], cfg.gain)
            for v in variants:
                psi, src = models[v](b["observed"])
                img = tracer(src, psi)
                ex, ey, _, _ = alpha_fields(psi, b["psi"], d)
                m = sample_metrics(psi, src, b, cfg.extent, cfg.psf_fwhm, cfg.gain)
                rows.append(dict(name=v, img=img, src=src, kappa=poisson(psi),
                                 err=torch.sqrt(ex ** 2 + ey ** 2)[None], pull=(img - b["observed"]) / sigma,
                                 m1=float(m["m1"][0]), m2=float(m["m2"][0])))
            true_kappa = poisson(b["psi"])
        a = lambda t: t.reshape(cfg.grid_size, cfg.grid_size).cpu().numpy()
        v_img = float(b["observed"].max())
        v_src = float(b["source"].max())
        v_err = max(float(np.percentile(a(r["err"]), 99)) for r in rows)

        fig, axes = plt.subplots(len(rows) + 1, 5, figsize=(15, 3.1 * (len(rows) + 1)), squeeze=False)
        def show(ax, z, extend="neither", **kw):
            fig.colorbar(ax.imshow(z, origin="lower", extent=box, **kw), ax=ax, fraction=0.046, extend=extend)

        def zoom(ax):
            ax.set_xlim(-SRC_ZOOM, SRC_ZOOM)
            ax.set_ylim(-SRC_ZOOM, SRC_ZOOM)

        kappa_kw = dict(cmap="RdBu_r", vmin=-KAPPA_LIM, vmax=KAPPA_LIM, extend="both")
        show(axes[0][0], a(b["observed"]), cmap="magma", vmin=0, vmax=v_img)
        show(axes[0][1], a(b["source"]), cmap="magma", vmin=0, vmax=v_src)
        zoom(axes[0][1])
        show(axes[0][2], a(true_kappa), **kappa_kw)
        axes[0][3].axis("off")
        axes[0][4].axis("off")
        axes[0][0].set_ylabel("truth")
        for ax, title in zip(axes[0], ["observed image", f"source (central ±{SRC_ZOOM:g}″)", "convergence κ"]):
            ax.set_title(title)
        axes[1][3].set_title("deflection error |Δα| [arcsec]\n(white: M1 band)")
        axes[1][4].set_title("image residual / σ̂")
        for i, r in enumerate(rows, start=1):
            show(axes[i][0], a(r["img"]), cmap="magma", vmin=0, vmax=v_img)
            show(axes[i][1], a(r["src"]), cmap="magma", vmin=0, vmax=v_src)
            zoom(axes[i][1])
            show(axes[i][2], a(r["kappa"]), **kappa_kw)
            show(axes[i][3], a(r["err"]), extend="max", cmap="viridis", vmin=0, vmax=v_err)
            axes[i][3].contour(band.astype(float), levels=[0.5], colors="w", linewidths=0.7,
                               extent=box, origin="lower")
            show(axes[i][4], a(r["pull"]), extend="both", cmap="RdBu_r", vmin=-PULL_LIM, vmax=PULL_LIM)
            axes[i][0].set_ylabel(f"{r['name']}\nM1 = {r['m1']:.3f}, M2 = {r['m2']:.3f}")
            axes[i][0].set_title("re-lensed with A_model", fontsize=9)
        fig.suptitle(f"{set_name}, sample {idx}, N = {n}, seed {seed}", y=1.0)
        fig.tight_layout()
        name = f"recon_{set_name.replace('/', '_')}_N{n}_s{seed}_i{idx}.png"
        fig.savefig(out / name, dpi=160)
        plt.close(fig)
        written.append(name)
    return written
