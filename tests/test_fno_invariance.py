import torch
from src.models.fno_reconstructor import FNOLensingReconstructor
from src.physics.ray_tracer import GravitationalRayTracer


def test_fno_variable_grid_resolutions():
    model = FNOLensingReconstructor(modes=8, width=16)
    model.eval()

    resolutions = [64, 128, 192]
    batch_size = 2

    for res in resolutions:
        dummy_input = torch.randn(batch_size, 1, res, res)
        pred_psi, pred_src = model(dummy_input)

        assert pred_psi.shape == (batch_size, 1, res, res)
        assert pred_src.shape == (batch_size, 1, res, res)


def test_ray_tracer_gradient_stability():
    grid_size = 64
    ray_tracer = GravitationalRayTracer(extent=3.0)

    source = torch.rand(1, 1, grid_size, grid_size, requires_grad=True)
    psi = torch.rand(1, 1, grid_size, grid_size, requires_grad=True)

    lensed = ray_tracer(source, psi)
    loss = torch.sum(lensed)
    loss.backward()

    assert source.grad is not None
    assert psi.grad is not None
    assert not torch.isnan(source.grad).any()
    assert not torch.isnan(psi.grad).any()
    assert not torch.isinf(source.grad).any()
    assert not torch.isinf(psi.grad).any()