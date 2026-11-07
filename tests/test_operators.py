import torch
from src.physics.operators import make_grid, a_truth


def test_grid_nodes_coincide():
    xf, _ = make_grid(255, 3.0)
    xc, _ = make_grid(128, 3.0)
    assert torch.allclose(xf[0, ::2], xc[0], atol=1e-6)


def test_a_truth_shape():
    psi = lambda x, y: torch.sqrt(x**2 + y**2 + 0.05**2)
    src = lambda x, y: torch.exp(-(x**2 + y**2) / 0.1)
    lensed, p, s = a_truth(psi, src)
    assert lensed.shape == p.shape == s.shape == (1, 128, 128)