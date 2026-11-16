import torch

from src.models.fno_reconstructor import FNOLensingReconstructor
from src.models.unet_reconstructor import UNetLensingReconstructor, count_real_params


def test_parameter_count_is_within_ten_percent_of_the_fno():
    fno = count_real_params(FNOLensingReconstructor(modes=32, width=64, pad_ratio=0.25))
    unet = count_real_params(UNetLensingReconstructor())
    assert abs(fno - 67_300_000) < 100_000, fno
    assert unet == 70_029_634, unet
    assert abs(unet / fno - 1.0) < 0.10


def test_interface_matches_the_fno():
    torch.manual_seed(0)
    model = UNetLensingReconstructor(base=16).eval()
    x = torch.randn(2, 1, 128, 128)
    with torch.no_grad():
        psi, src = model(x)
    assert psi.shape == src.shape == (2, 1, 128, 128)
    assert torch.allclose(psi.mean((-2, -1)), torch.zeros(2, 1), atol=1e-5)
    assert (src > 0).all()
    assert torch.isfinite(psi).all() and torch.isfinite(src).all()


def test_gradients_reach_every_parameter():
    torch.manual_seed(0)
    model = UNetLensingReconstructor(base=16)
    psi, src = model(torch.randn(2, 1, 128, 128))
    (psi.pow(2).mean() + src.mean()).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    # the last psi bias has zero gradient by construction: the mean of psi is subtracted
    dead = [n for n, p in model.named_parameters() if float(p.grad.abs().sum()) == 0]
    assert dead in ([], ["psi_head.2.bias"]), dead
