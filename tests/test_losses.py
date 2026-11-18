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

def test_v4_has_no_supervised_term_and_r_only_the_curvature_term():
    b = _batch()
    v4 = loss_config("V4", 0.02, 15.0)
    assert not v4.use_sup and v4.lambda_ray == 0.02 and v4.lambda_pos == 15.0 and v4.lambda_r == 0.0
    _, terms = LensingLoss(v4)(b["psi"], b["source"], b)
    assert set(terms) == {"ray", "pos"} and terms["pos"].item() < 1e-12
    r = loss_config("R", 0.02, 15.0, lambda_r=0.5)
    assert r.use_sup and r.lambda_ray == 0.0 and r.lambda_pos == 0.0 and r.lambda_r == 0.5
    _, terms = LensingLoss(r)(b["psi"], b["source"], b)
    assert set(terms) == {"sup", "r"}
    for v in ("V0", "V1", "V2", "V3", "V4"):
        assert loss_config(v, 0.02, 15.0, lambda_r=0.5).lambda_r == 0.0


def test_curvature_penalty_is_zero_for_a_plane_and_sign_blind():
    x, y = make_grid(128, 3.0)
    plane = (0.3 * x - 0.2 * y)[None, None]
    loss = LensingLoss(loss_config("R", 0.0, 0.0, lambda_r=1.0))
    assert loss.curv(plane).item() < 1e-6
    b = _batch()
    assert loss.curv(b["psi"]).item() > 0
    assert abs(loss.curv(b["psi"]).item() - loss.curv(-b["psi"]).item()) < 1e-6
