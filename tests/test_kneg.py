import numpy as np
import torch

from src.data.generator import N_LEVELS, Shift, all_shifts, generate
from src.physics.poisson import PoissonSolver2D
from src.utils.config import SimulationConfig

CFG = SimulationConfig.from_yaml()
SEEDS = (200_000, 200_123, 200_499)


def test_kneg_is_exact_mirror_of_a3():
    for seed in SEEDS:
        base = generate(seed, CFG)["psi"]
        for lvl in (1, 2, 3):
            a3 = generate(seed, CFG, Shift("A3", lvl))
            kn = generate(seed, CFG, Shift("Kneg", lvl))
            np.testing.assert_allclose(kn["psi"] - base, -(a3["psi"] - base), atol=1e-5)
            assert np.array_equal(kn["source"], a3["source"])
            assert np.array_equal(kn["params"], a3["params"])


def test_kneg_outside_e1_axes():
    assert "Kneg" not in N_LEVELS
    assert all(s.axis != "Kneg" for s in all_shifts())


def test_kneg_produces_negative_convergence():
    solver = PoissonSolver2D(CFG.extent)
    frac = []
    for seed in range(200_000, 200_020):
        psi = torch.from_numpy(generate(seed, CFG, Shift("Kneg", 3))["psi"])[None]
        kappa = solver(psi)[..., 2:-2, 2:-2]
        frac.append(bool((kappa < 0).any()))
    assert np.mean(frac) > 0.9, np.mean(frac)
