"""X5: maximum-likelihood fit of the generator's own parametric model through A_model (PROTOCOL §9).

The fitted family is the elliptical potential with core and shear plus a Sersic source, i.e. the
14 generator parameters without sigma0. Each sample is fitted from several random starts by Adam
on the mean squared pull, with sigma estimated from the observation; the start with the lowest
chi^2 is kept. Parameters are optimized in units of the training ranges and projected onto wide
physical bounds after every step. Starts and samples are independent: the loss is a sum of
per-start terms and Adam acts per element.
"""
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from src.data.generator import PARAM_ORDER, param_ranges
from src.physics.noise import estimate_sigma
from src.physics.operators import make_grid
from src.physics.profiles import elliptical_potential, sersic_source
from src.physics.ray_tracer import GravitationalRayTracer

FIT_ORDER = tuple(PARAM_ORDER[:14])          # everything except sigma0
N_STARTS = 8
STEPS = 1_500
LR = 0.02                                    # in units of the training range, cosine decay to zero
START_SEED_OFFSET = 7_000_000                # start generator seed = offset + data seed
SIGMA_FLOOR = 1e-4                           # same floor as L_ray and M5
INF = float("inf")
BOUNDS = {                                   # projection bounds; the angles are free
    "theta_e": (0.3, 2.5), "q_lens": (0.3, 1.0), "x0_lens": (-0.5, 0.5), "y0_lens": (-0.5, 0.5),
    "phi_lens": (-INF, INF), "gamma1": (-0.3, 0.3), "gamma2": (-0.3, 0.3),
    "amp": (0.01, 10.0), "r_eff": (0.03, 1.5), "n_src": (0.3, 6.0),
    "x0_src": (-1.5, 1.5), "y0_src": (-1.5, 1.5), "q_src": (0.2, 1.0), "phi_src": (-INF, INF),
}
X5_SETS = ("test_id", "ood/A2_L2", "ood/A3_L2", "ood/A4_L1")
X5_SAMPLES = 50
X5_FIRST_SEED = 200_000


def _ranges(cfg, device):
    r = param_ranges(cfg)
    lo = torch.tensor([r[k][0] for k in FIT_ORDER], dtype=torch.float32, device=device)
    hi = torch.tensor([r[k][1] for k in FIT_ORDER], dtype=torch.float32, device=device)
    blo = torch.tensor([BOUNDS[k][0] for k in FIT_ORDER], dtype=torch.float32, device=device)
    bhi = torch.tensor([BOUNDS[k][1] for k in FIT_ORDER], dtype=torch.float32, device=device)
    return lo, hi, blo, bhi


def draw_starts(seed, n_starts=N_STARTS):
    """Uniform starts in units of the training ranges, [n_starts, 14]; fixed by the data seed."""
    g = torch.Generator().manual_seed(START_SEED_OFFSET + int(seed))
    return torch.rand(n_starts, len(FIT_ORDER), generator=g, dtype=torch.float32)


def param_fields(p, cfg):
    """psi and source on the model grid for parameters p [M, 14]; both [M, 1, n, n]."""
    x, y = make_grid(cfg.grid_size, cfg.extent, device=p.device)
    v = {k: p[:, i, None, None] for i, k in enumerate(FIT_ORDER)}
    psi = elliptical_potential(x, y, v["theta_e"], v["q_lens"], v["x0_lens"], v["y0_lens"],
                               phi=v["phi_lens"], gamma1=v["gamma1"], gamma2=v["gamma2"],
                               core_radius=cfg.core_radius, slope=1.0)
    src = sersic_source(x, y, v["amp"], v["r_eff"], v["n_src"], v["x0_src"], v["y0_src"],
                        v["q_src"], phi=v["phi_src"])
    return psi[:, None], src[:, None]


def fit(obs, seeds, cfg, n_starts=N_STARTS, steps=STEPS, lr=LR, u0=None):
    """Fit observations obs [B, 1, n, n] whose data seeds are `seeds`.

    u0 [B * n_starts, 14] overrides the random starts (used only in tests).

    Returns the best-start parameters [B, 14], the final chi^2 of every start [B, S], the index of
    the best start [B], and the best-start psi and source [B, 1, n, n].
    """
    device, b = obs.device, obs.shape[0]
    lo, hi, blo, bhi = _ranges(cfg, device)
    w = hi - lo
    ulo, uhi = (blo - lo) / w, (bhi - lo) / w
    tracer = GravitationalRayTracer(extent=cfg.extent, psf_fwhm=cfg.psf_fwhm)
    sigma = estimate_sigma(obs, cfg.gain).clamp_min(SIGMA_FLOOR)
    obs_r, sigma_r = obs.repeat_interleave(n_starts, 0), sigma.repeat_interleave(n_starts, 0)

    def chi2(u):
        psi, src = param_fields(lo + u * w, cfg)
        return (((tracer(src, psi) - obs_r) / sigma_r) ** 2).flatten(1).mean(1)

    u = torch.cat([draw_starts(s, n_starts) for s in seeds]) if u0 is None else u0.clone()
    u = u.to(device).requires_grad_(True)
    opt = torch.optim.Adam([u], lr=lr)
    for t in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr * 0.5 * (1.0 + math.cos(math.pi * t / steps))
        opt.zero_grad(set_to_none=True)
        chi2(u).sum().backward()
        u.grad.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
        opt.step()
        with torch.no_grad():
            u.copy_(torch.minimum(torch.maximum(u, ulo), uhi))

    with torch.no_grad():
        c = chi2(u).view(b, n_starts)
        best = torch.nan_to_num(c, nan=INF).argmin(1)
        p = (lo + u * w).view(b, n_starts, -1)[torch.arange(b, device=device), best]
        psi, src = param_fields(p, cfg)
    return p, c, best, psi, src


def run_set(cfg, set_name, data_root="data", out_root="results/x5", metrics_root="results/metrics",
            final=False, n_samples=X5_SAMPLES, batch=5, n_starts=N_STARTS, steps=STEPS, device=None):
    """Fit the first n_samples of one set and store per-sample metrics, parameters and chi^2."""
    from src.data.store import NpyDataset
    from src.evaluation.figures import check_access
    from src.evaluation.metrics import sample_metrics
    from src.evaluation.runner import KEYS
    from src.training.trainer import _git_state

    check_access(set_name, final, metrics_root)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(data_root) / set_name
    first = json.loads((data_dir / "meta.json").read_text())["seed_first"]
    if set_name != "val" and first != X5_FIRST_SEED:
        raise RuntimeError(f"{set_name} starts at seed {first}, expected {X5_FIRST_SEED}")
    ds = NpyDataset(data_dir)
    acc, t0 = {}, time.time()
    for i in range(0, n_samples, batch):
        j = min(i + batch, n_samples)
        bt = {k: torch.from_numpy(np.array(ds.a[k][i:j])).to(device) for k in KEYS}
        p, c, best, psi, src = fit(bt["observed"], range(first + i, first + j), cfg, n_starts, steps)
        m = sample_metrics(psi, src, bt, cfg.extent, cfg.psf_fwhm, cfg.gain)
        m.update(params_fit=p.cpu(), chi2_starts=c.cpu(), best_start=best.cpu(),
                 params_true=bt["params"][:, :len(FIT_ORDER)].cpu(),
                 seed=torch.arange(first + i, first + j))
        for k, v in m.items():
            acc.setdefault(k, []).append(v.numpy())
        print(f"{set_name}: {j}/{n_samples} fitted, {time.time() - t0:.0f} s", flush=True)
    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    stem = set_name.replace("/", "__")
    np.savez(out / f"{stem}.npz", **{k: np.concatenate(v) for k, v in acc.items()})
    commit, dirty = _git_state()
    (out / f"{stem}.json").write_text(json.dumps(dict(
        set=set_name, n_samples=n_samples, n_starts=n_starts, steps=steps, lr=LR, order=FIT_ORDER,
        bounds={k: [str(a), str(b)] for k, (a, b) in BOUNDS.items()}, commit=commit, git_dirty=dirty,
        torch=torch.__version__, device=str(device), seconds=round(time.time() - t0)), indent=1))
    return out / f"{stem}.npz"


def summarize(out_root="results/x5", metrics_root="results/metrics", sets=X5_SETS,
              variants=("V0", "V3"), ns=(500, 2_000, 8_000), seeds=range(42, 47)):
    """Markdown table: median M1 and M2 of the fit and of the networks on the same samples.

    Network entries are the mean over training seeds of the per-seed median. The last column
    counts the starts that end within 5% of the best chi^2, as a measure of how multimodal the
    fit is.
    """
    rows = ["| set | method | M1 | M2 | median best χ² | starts within 5% of best |", "|---|---|---|---|---|---|"]
    for s in sets:
        stem = s.replace("/", "__")
        f = Path(out_root) / f"{stem}.npz"
        if not f.exists():
            continue
        z = np.load(f)
        n = len(z["m1"])
        c = np.nan_to_num(z["chi2_starts"], nan=np.inf)
        near = (c <= 1.05 * c.min(1, keepdims=True)).sum(1)
        rows.append(f"| {s} | parametric fit | {np.median(z['m1']):.4f} | {np.median(z['m2']):.4f} | "
                    f"{np.median(c.min(1)):.3f} | median {int(np.median(near))} of {c.shape[1]} |")
        for v in variants:
            for nn in ns:
                med = [(np.median(m["m1"][:n]), np.median(m["m2"][:n])) for sd in seeds
                       if (q := Path(metrics_root) / f"{v}_N{nn}_s{sd}" / f"{stem}.npz").exists()
                       for m in [np.load(q)]]
                if med:
                    m1, m2 = np.mean(med, 0)
                    rows.append(f"| {s} | {v}, N = {nn} ({len(med)} seed{'' if len(med) == 1 else 's'}) | {m1:.4f} | {m2:.4f} | | |")
    text = "\n".join(rows) + "\n"
    Path(out_root).mkdir(parents=True, exist_ok=True)
    (Path(out_root) / "summary.md").write_text(text, encoding="utf-8")
    return text
