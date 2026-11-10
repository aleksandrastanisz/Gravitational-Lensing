import torch
from src.losses.lensing_loss import LensingLoss, loss_config, sup_per_sample
from src.physics.operators import make_grid

D = 6.0 / 127


def _batch(b=2):
    x, y = make_grid(128, 3.0)
    psi = torch.sqrt(x**2 + y**2 + 0.05**2)[None, None].repeat(b, 1, 1, 1)
    src = torch.exp(-(x**2 + y**2) / 0.1)[None, None].repeat(b, 1, 1, 1)
    return dict(psi=psi, source=src, observed=src.clone() + 0.01 * torch.randn_like(src),
                noise_map=torch.full_like(src, 0.02))


def test_sup_zero_on_truth():
    b = _batch()
    assert sup_per_sample(b["psi"], b["source"], b["psi"], b["source"], D).max() < 1e-5


def test_pos_zero_for_convex_and_positive_for_concave():
    b = _batch()
    loss = LensingLoss(loss_config("V1", 0.0, 15.0))
    assert loss.pos(b["psi"]).item() < 1e-12
    assert loss.pos(-b["psi"]).item() > 0


def test_v0_skips_ray_and_pos():
    b = _batch()
    _, terms = LensingLoss(loss_config("V0", 0.02, 15.0))(b["psi"], b["source"], b)
    assert set(terms) == {"sup"}


def test_v3_gradients_finite():
    b = _batch()
    psi = b["psi"].clone().requires_grad_(True)
    src = b["source"].clone().requires_grad_(True)
    total, terms = LensingLoss(loss_config("V3", 0.02, 15.0))(psi, src, b)
    total.backward()
    assert set(terms) == {"sup", "ray", "pos"}
    assert torch.isfinite(psi.grad).all() and torch.isfinite(src.grad).all()


def test_warmup_zero_disables_physics():
    b = _batch()
    _, terms = LensingLoss(loss_config("V3", 0.02, 15.0))(b["psi"], b["source"], b, warm=0.0)
    assert set(terms) == {"sup"}