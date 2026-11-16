import numpy as np
import pytest
import torch

from src.data.generator import apply_shift, draw_base, make_fns
from src.fitting.parametric import (BOUNDS, FIT_ORDER, _ranges, draw_starts, fit, param_fields,
                                    run_set, summarize)
from src.physics.noise import true_sigma
from src.physics.operators import a_inverse_crime
from src.utils.config import SimulationConfig

CFG = SimulationConfig.from_yaml()


def _sample(seed):
    """Truth parameters and the A_model image, psi and source of a training-family sample."""
    p, u, _ = draw_base(seed, CFG)
    p, slope, sub, clumps, _ = apply_shift(p, u, CFG, None, seed)
    img, psi, src = a_inverse_crime(*make_fns(p, CFG, slope, sub, clumps), CFG.extent, CFG.grid_size,
                                    CFG.psf_fwhm)
    truth = torch.tensor([[p[k] for k in FIT_ORDER]], dtype=torch.float32)
    return p, truth, img[None], psi[None], src[None]


def test_param_fields_reproduce_the_generator():
    _, truth, _, psi, src = _sample(3)
    fpsi, fsrc = param_fields(truth, CFG)
    assert torch.allclose(fpsi, psi.float(), atol=1e-5)
    assert torch.allclose(fsrc, src.float(), atol=1e-5)


def test_starts_are_fixed_by_the_seed_and_inside_the_training_ranges():
    a, b = draw_starts(200_000), draw_starts(200_000)
    assert torch.equal(a, b) and a.shape == (8, 14)
    assert not torch.equal(a, draw_starts(200_001))
    assert a.min() >= 0.0 and a.max() <= 1.0


def test_fit_recovers_the_truth_from_nearby_starts_and_keeps_the_lowest_chi2():
    p, truth, img, _, _ = _sample(5)
    g = torch.Generator().manual_seed(0)
    obs = (img + torch.randn(img.shape, generator=g) * true_sigma(img, p["sigma0"], CFG.gain)).float()
    lo, hi, _, _ = _ranges(CFG, "cpu")
    u_true = (truth - lo) / (hi - lo)
    u0 = u_true + 0.05 * torch.randn(3, 14, generator=g)
    u0[2] = u_true[0] + 0.6                      # a deliberately poor start
    fitted, chi2, best, psi, src = fit(obs, [5], CFG, n_starts=3, steps=400, u0=u0)
    assert chi2.shape == (1, 3) and int(best[0]) == int(chi2[0].argmin())
    assert float(chi2[0].min()) < 1.2
    i = FIT_ORDER.index("theta_e")
    assert abs(float(fitted[0, i] - truth[0, i])) < 0.01
    assert psi.shape == src.shape == obs.shape


def test_fitted_parameters_stay_inside_the_bounds():
    _, truth, img, _, _ = _sample(7)
    u0 = torch.full((2, 14), 5.0)                # far outside: must be projected back
    u0[1] = -5.0
    fitted, _, _, _, _ = fit(img.float(), [7], CFG, n_starts=2, steps=3, u0=u0)
    for i, k in enumerate(FIT_ORDER):
        lo, hi = BOUNDS[k]
        assert lo - 1e-5 <= float(fitted[0, i]) <= hi + 1e-5 or not np.isfinite(lo)


def test_test_sets_are_locked_until_the_final_evaluation(tmp_path):
    with pytest.raises(PermissionError):
        run_set(CFG, "test_id", metrics_root=tmp_path, final=False)
    with pytest.raises(PermissionError):
        run_set(CFG, "ood/A3_L2", metrics_root=tmp_path, final=True)


def test_summary_compares_fit_and_networks_on_the_same_samples(tmp_path):
    x5, met = tmp_path / "x5", tmp_path / "metrics" / "V0_N2000_s42"
    x5.mkdir()
    met.mkdir(parents=True)
    np.savez(x5 / "test_id.npz", m1=np.full(4, 0.01), m2=np.full(4, 0.1),
             chi2_starts=np.array([[1.0, 1.01, 3.0]] * 4))
    np.savez(met / "test_id.npz", m1=np.r_[np.full(4, 0.05), np.full(96, 9.0)], m2=np.full(100, 0.3))
    text = summarize(x5, tmp_path / "metrics", sets=("test_id",))
    assert "| test_id | parametric fit | 0.0100 | 0.1000 | 1.000 | median 2 of 3 |" in text
    assert "| test_id | V0, N = 2000 (1 seed) | 0.0500 | 0.3000 | | |" in text
    assert (x5 / "summary.md").exists()
