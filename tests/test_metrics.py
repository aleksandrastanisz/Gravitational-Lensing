import math

import torch

from src.data.generator import generate
from src.evaluation.metrics import IDX, ring_band, sample_metrics
from src.physics.operators import make_grid
from src.utils.config import SimulationConfig

KEYS = ("observed", "clean_lensed", "source", "psi", "noise_map")
CFG = SimulationConfig.from_yaml()


def _batch(seeds=(0, 1)):
    ds = [generate(s, CFG) for s in seeds]
    b = {k: torch.stack([torch.from_numpy(d[k]) for d in ds]) for k in KEYS}
    b["params"] = torch.stack([torch.from_numpy(d["params"]) for d in ds])
    return b


def _xy():
    x, y = make_grid(CFG.grid_size, CFG.extent)
    return x[None, None], y[None, None]


def _m(pred_psi, pred_src, b):
    return sample_metrics(pred_psi, pred_src, b, CFG.extent, CFG.psf_fwhm, CFG.gain)


def test_perfect_prediction():
    b = _batch()
    m = _m(b["psi"], b["source"], b)
    for k in ("m1", "m1b", "m2", "m3", "m4", "m4b_centroid_mas", "m4b_flux"):
        assert m[k].abs().max() < 1e-5, (k, m[k])
    assert (m["m6"] == 0).all(), m["m6"]          # kappa >= 0 holds for the ID family
    assert ((m["m5_global"] > 0.5) & (m["m5_global"] < 2.0)).all(), m["m5_global"]


def test_m1b_removes_constant_deflection_offset():
    b = _batch()
    x, y = _xy()
    m = _m(b["psi"] + 0.05 * x - 0.02 * y, b["source"], b)
    assert (m["m1"] > 0.01).all(), m["m1"]
    assert m["m1b"].max() < 1e-3, m["m1b"]


def test_constant_in_psi_is_ignored():
    b = _batch()
    m = _m(b["psi"] + 3.0, b["source"], b)
    assert m["m3"].max() < 1e-4 and m["m1"].max() < 1e-3 and m["m4"].max() < 1e-3


def test_scaled_psi_gives_relative_error_of_scale():
    b = _batch()
    m = _m(1.1 * b["psi"], b["source"], b)
    for k in ("m1", "m3", "m4"):
        assert torch.allclose(m[k], torch.full_like(m[k], 0.1), atol=1e-3), (k, m[k])


def test_centroid_shift_in_mas():
    b = _batch(seeds=(0,))
    b["params"][:, IDX["cx_src"]], b["params"][:, IDX["cy_src"]] = 0.1, -0.2
    x, y = _xy()
    gauss = lambda cx: torch.exp(-0.5 * ((x - cx) ** 2 + (y + 0.2) ** 2) / 0.1 ** 2)
    m = _m(b["psi"], gauss(0.15), {**b, "source": gauss(0.10)})
    assert abs(m["m4b_centroid_mas"].item() - 50.0) < 1.0, m["m4b_centroid_mas"]
    assert abs(m["m4b_flux"].item()) < 1e-3, m["m4b_flux"]


def test_m6_counts_negative_convergence():
    b = _batch()
    x, y = _xy()
    m = _m(-(x ** 2 + y ** 2).expand_as(b["psi"]).clone(), b["source"], b)
    assert (m["m6"] == 1.0).all(), m["m6"]


def test_m5_empty_arc_mask_is_nan():
    b = _batch()
    m = _m(b["psi"], b["source"], {**b, "clean_lensed": torch.zeros_like(b["clean_lensed"])})
    assert torch.isnan(m["m5_arc"]).all() and torch.isfinite(m["m5_global"]).all()


def test_ring_band_area_for_circular_lens():
    p = torch.zeros(1, len(IDX), dtype=torch.float64)
    p[0, IDX["theta_e"]], p[0, IDX["q_lens"]] = 1.0, 1.0
    band = ring_band(p, CFG.grid_size, CFG.extent)
    pix = (2 * CFG.extent / (CFG.grid_size - 1)) ** 2
    expected = math.pi * (1.5 ** 2 - 0.5 ** 2) / pix
    assert abs(band.sum().item() / expected - 1) < 0.03, (band.sum().item(), expected)
