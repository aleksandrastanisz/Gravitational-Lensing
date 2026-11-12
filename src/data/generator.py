from dataclasses import dataclass
import numpy as np
import torch
from src.physics.noise import true_sigma
from src.physics.operators import a_truth
from src.physics.profiles import (elliptical_potential, gaussian_clumps,
                                  plummer_potential, sersic_source)

PARAM_ORDER = ["theta_e", "q_lens", "x0_lens", "y0_lens", "phi_lens", "gamma1", "gamma2",
               "amp", "r_eff", "n_src", "x0_src", "y0_src", "q_src", "phi_src", "sigma0"]
OUT_ORDER = PARAM_ORDER + ["cx_src", "cy_src", "slope"]

AXIS_ID = {"A1_theta_e": 1, "A1_q": 2, "A1_shear": 3, "A1_src_pos": 4,
           "A2": 5, "A3": 6, "A4": 7, "A5": 8, "A6": 9}
N_LEVELS = {"A1_theta_e": 3, "A1_q": 3, "A1_shear": 3, "A1_src_pos": 3,
            "A2": 3, "A3": 3, "A4": 1, "A5": 3, "A6": 2}
DIAG_LEVELS = {"Kneg": 3}
MIRROR_OF = {"Kneg": "A3"}
FRAC = (0.25, 0.50, 1.00)
DELTA = (0.1, 0.2, 0.3)
B_RANGE = ((0.02, 0.05), (0.05, 0.12), (0.12, 0.25))
SIGMA0_OOD = (0.005, 0.05, 0.08)
PSF_OOD = (0.12, 0.15)


@dataclass(frozen=True)
class Shift:
    axis: str
    level: int


def all_shifts():
    return [Shift(a, l) for a, n in N_LEVELS.items() for l in range(1, n + 1)]


def diag_shifts():
    return [Shift(a, l) for a, n in DIAG_LEVELS.items() for l in range(1, n + 1)]


def param_ranges(c):
    b, s = c.lens_pos_bound, c.src_pos_bound
    return {
        "theta_e": (c.theta_e_min, c.theta_e_max), "q_lens": (c.q_lens_min, c.q_lens_max),
        "x0_lens": (-b, b), "y0_lens": (-b, b), "phi_lens": (0.0, np.pi),
        "gamma1": (c.shear_min, c.shear_max), "gamma2": (c.shear_min, c.shear_max),
        "amp": (c.amp_min, c.amp_max), "r_eff": (c.r_eff_min, c.r_eff_max),
        "n_src": (c.n_sersic_min, c.n_sersic_max),
        "x0_src": (-s, s), "y0_src": (-s, s), "q_src": (c.q_src_min, c.q_src_max),
        "phi_src": (0.0, np.pi), "sigma0": (c.noise_std_train_min, c.noise_std_train_max),
    }


def draw_base(seed, cfg):
    g = torch.Generator().manual_seed(seed)
    u = torch.rand(len(PARAM_ORDER), generator=g, dtype=torch.float64).numpy()
    z = torch.randn(1, cfg.grid_size, cfg.grid_size, generator=g)
    rng = param_ranges(cfg)
    p = {k: float(rng[k][0] + ui * (rng[k][1] - rng[k][0])) for k, ui in zip(PARAM_ORDER, u)}
    return p, dict(zip(PARAM_ORDER, u)), z


def _strip(u, lo, hi, f, side):
    w = hi - lo
    if side == "up":
        return hi + u * f * w
    if side == "down":
        return lo - u * f * w
    return hi + (2 * u - 1) * f * w if u >= 0.5 else lo - 2 * u * f * w


def apply_shift(p, u, cfg, shift, seed):
    p, slope, sub, clumps, psf = dict(p), 1.0, None, None, cfg.psf_fwhm
    if shift is None:
        return p, slope, sub, clumps, psf
    l, ax, rng_all = shift.level, shift.axis, param_ranges(cfg)
    ss = np.random.SeedSequence([seed, AXIS_ID[MIRROR_OF.get(ax, ax)], l])
    rng = np.random.default_rng(ss)
    f = FRAC[l - 1]

    def move(key, side):
        lo, hi = rng_all[key]
        p[key] = float(_strip(u[key], lo, hi, f, side))

    if ax == "A1_theta_e":
        move("theta_e", "up")
    elif ax == "A1_q":
        move("q_lens", "down")
    elif ax == "A1_shear":
        move("gamma1", "both"); move("gamma2", "both")
    elif ax == "A1_src_pos":
        move("x0_src", "both"); move("y0_src", "both")
    elif ax == "A2":
        d = DELTA[l - 1]
        slope = float(rng.uniform(1 - d, 1 + d))
    elif ax in ("A3", "Kneg"):
        b = float(rng.uniform(*B_RANGE[l - 1]))
        if ax == "Kneg":
            b = -b
        ang, rad = rng.uniform(0, 2 * np.pi), p["theta_e"] * rng.uniform(0.6, 1.2)
        sub = (b, float(rad * np.cos(ang)), float(rad * np.sin(ang)))
    elif ax == "A4":
        clumps = [(float(rng.uniform(0.5, 1.0)), float(rng.uniform(-0.4, 0.4)),
                   float(rng.uniform(-0.4, 0.4)), float(rng.uniform(0.08, 0.2))) for _ in range(3)]
    elif ax == "A5":
        p["sigma0"] = SIGMA0_OOD[l - 1]
    elif ax == "A6":
        psf = PSF_OOD[l - 1]
    return p, slope, sub, clumps, psf


def make_fns(p, cfg, slope, sub, clumps):
    def psi(x, y):
        out = elliptical_potential(x, y, p["theta_e"], p["q_lens"], p["x0_lens"], p["y0_lens"],
                                   phi=p["phi_lens"], gamma1=p["gamma1"], gamma2=p["gamma2"],
                                   core_radius=cfg.core_radius, slope=slope)
        if sub is not None:
            out = out + plummer_potential(x, y, sub[0], sub[1], sub[2], a=cfg.plummer_scale)
        return out

    def src(x, y):
        if clumps is not None:
            return gaussian_clumps(x, y, clumps)
        return sersic_source(x, y, p["amp"], p["r_eff"], p["n_src"],
                             p["x0_src"], p["y0_src"], p["q_src"], phi=p["phi_src"])
    return psi, src


def _src_center(p, clumps):
    if clumps is None:
        return p["x0_src"], p["y0_src"]
    w = np.array([a * s**2 for a, _, _, s in clumps])
    return (float(np.sum(w * [c[1] for c in clumps]) / w.sum()),
            float(np.sum(w * [c[2] for c in clumps]) / w.sum()))


def generate(seed, cfg, shift=None):
    p, u, z = draw_base(seed, cfg)
    p, slope, sub, clumps, psf = apply_shift(p, u, cfg, shift, seed)
    psi_fn, src_fn = make_fns(p, cfg, slope, sub, clumps)
    clean, psi, src = a_truth(psi_fn, src_fn, cfg.extent, cfg.grid_size, psf)
    sigma = true_sigma(clean[None], p["sigma0"], cfg.gain)[0]
    cx, cy = _src_center(p, clumps)
    vals = {**p, "cx_src": cx, "cy_src": cy, "slope": slope}
    return dict(observed=(clean + z * sigma).numpy(), clean_lensed=clean.numpy(),
                source=src.numpy(), psi=psi.numpy(), noise_map=sigma.numpy(),
                params=np.array([vals[k] for k in OUT_ORDER], dtype=np.float64))