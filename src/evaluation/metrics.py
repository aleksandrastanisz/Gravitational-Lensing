"""Per-sample evaluation metrics M1-M6 (PROTOCOL §8 and §13).

All metrics are computed per sample on the 128x128 grid and returned as 1-D tensors
of length B. Aggregation over samples (median, nanmedian for M5) happens elsewhere.
"""
import torch

from src.data.generator import OUT_ORDER
from src.physics.deflection import alpha_fields, alpha_rel_l2, field_norm, rel_l2
from src.physics.noise import estimate_sigma
from src.physics.operators import make_grid
from src.physics.poisson import PoissonSolver2D
from src.physics.ray_tracer import GravitationalRayTracer

BAND_HALF_WIDTH = 0.5  # arcsec, M1/M1b: |r_ell - theta_E| < 0.5"
APERTURE_R = 1.0       # arcsec, M2/M4b: aperture around the true source centre
ARC_SNR = 3.0          # M5 arc mask: clean_lensed > 3 sigma_true
BORDER = 2             # M6 grid interior, same as L_pos
SIGMA_FLOOR = 1e-4     # same floor as L_ray
EPS = 1e-6
IDX = {k: i for i, k in enumerate(OUT_ORDER)}


def _param(params, key):
    return params[:, IDX[key]].to(torch.float32)[:, None, None]


def _grid(n, extent, device):
    x, y = make_grid(n, extent, device=device)
    return x[None], y[None]


def ring_band(params, n=128, extent=3.0, half_width=BAND_HALF_WIDTH):
    """Elliptical band around the true Einstein ring, [B, n, n] bool.

    r_ell uses the lens convention from profiles.elliptical_potential
    (rotation about (x0, y0) by phi, r = sqrt(q x'^2 + y'^2 / q)), without the core.
    """
    x, y = _grid(n, extent, params.device)
    x0, y0 = _param(params, "x0_lens"), _param(params, "y0_lens")
    phi, q, te = _param(params, "phi_lens"), _param(params, "q_lens"), _param(params, "theta_e")
    dx, dy = x - x0, y - y0
    c, s = torch.cos(phi), torch.sin(phi)
    xr, yr = dx * c + dy * s, -dx * s + dy * c
    r = torch.sqrt(q * xr ** 2 + yr ** 2 / q)
    return (r - te).abs() < half_width


def source_aperture(params, n=128, extent=3.0, radius=APERTURE_R):
    """Circular aperture of radius R around the true source centre (cx_src, cy_src), [B, n, n] bool."""
    x, y = _grid(n, extent, params.device)
    cx, cy = _param(params, "cx_src"), _param(params, "cy_src")
    return (x - cx) ** 2 + (y - cy) ** 2 < radius ** 2


def _m1b(pred_psi, true_psi, d, band):
    """M1 after subtracting the mean deflection-error vector in the band; denominator unchanged."""
    ex, ey, tx, ty = alpha_fields(pred_psi, true_psi, d)
    m = band.to(ex.dtype)
    cnt = m.sum((-2, -1)).clamp_min(1.0)
    ex = ex - ((ex * m).sum((-2, -1)) / cnt)[:, None, None]
    ey = ey - ((ey * m).sum((-2, -1)) / cnt)[:, None, None]
    return field_norm([ex, ey], band) / (field_norm([tx, ty], band) + EPS)


def _m3(pred_psi, true_psi):
    """relL2(psi) after removing the constant (grid mean) from both."""
    p = pred_psi - pred_psi.mean((-2, -1), keepdim=True)
    t = true_psi - true_psi.mean((-2, -1), keepdim=True)
    return rel_l2(p, t)


def _m4b(pred_src, true_src, aperture, n, extent):
    """Flux-weighted centroid error [mas] and relative flux error (pred/true - 1) in the aperture.

    Both centroids are computed from the images inside the same aperture.
    A prediction with zero flux in the aperture gives NaN centroid error.
    """
    x, y = _grid(n, extent, pred_src.device)
    m = aperture.to(pred_src.dtype)
    wt, wp = true_src.squeeze(1) * m, pred_src.squeeze(1) * m
    ft, fp = wt.sum((-2, -1)), wp.sum((-2, -1))
    cxt, cyt = (wt * x).sum((-2, -1)) / ft, (wt * y).sum((-2, -1)) / ft
    cxp, cyp = (wp * x).sum((-2, -1)) / fp, (wp * y).sum((-2, -1)) / fp
    cent = 1000.0 * torch.sqrt((cxp - cxt) ** 2 + (cyp - cyt) ** 2)
    nan = torch.full_like(cent, float("nan"))
    return torch.where(fp > 0, cent, nan), fp / ft - 1.0


def _chi2(model_img, obs, sigma, arc):
    """Unclamped chi^2: mean pull^2 on the arc mask (NaN when empty) and over the whole grid."""
    p2 = ((model_img - obs) / sigma.clamp_min(SIGMA_FLOOR)) ** 2
    arc = arc.to(p2.dtype)
    n_arc = arc.flatten(1).sum(1)
    on_arc = (p2 * arc).flatten(1).sum(1) / n_arc
    on_arc = torch.where(n_arc > 0, on_arc, torch.full_like(on_arc, float("nan")))
    return on_arc, p2.flatten(1).mean(1)


@torch.inference_mode()
def sample_metrics(pred_psi, pred_src, batch, extent=3.0, psf_fwhm=0.10, gain=500.0):
    """All §8 metrics for one batch. Tensors are [B, 1, n, n]; batch["params"] is [B, len(OUT_ORDER)].

    M5 always uses A_model with psf_fwhm (0.10" on every axis, including A6).
    """
    true_psi, true_src, params = batch["psi"], batch["source"], batch["params"]
    n = pred_psi.shape[-1]
    d = 2.0 * extent / (n - 1)
    band = ring_band(params, n, extent)
    ap = source_aperture(params, n, extent)

    cent, flux = _m4b(pred_src, true_src, ap, n, extent)

    tracer = GravitationalRayTracer(extent=extent, psf_fwhm=psf_fwhm)
    img = tracer(pred_src, pred_psi)
    obs = batch["observed"]
    arc = batch["clean_lensed"] > ARC_SNR * batch["noise_map"]
    m5_arc, m5_glob = _chi2(img, obs, estimate_sigma(obs, gain), arc)
    m5_arc_or, m5_glob_or = _chi2(img, obs, batch["noise_map"], arc)

    kappa = PoissonSolver2D(extent=extent).to(pred_psi.device)(pred_psi)
    kappa = kappa[..., BORDER:-BORDER, BORDER:-BORDER]

    out = dict(
        m1=alpha_rel_l2(pred_psi, true_psi, d, band),
        m1b=_m1b(pred_psi, true_psi, d, band),
        m2=rel_l2(pred_src, true_src, ap),
        m3=_m3(pred_psi, true_psi),
        m4=alpha_rel_l2(pred_psi, true_psi, d),
        m4b_centroid_mas=cent,
        m4b_flux=flux,
        m5_arc=m5_arc, m5_global=m5_glob,
        m5_arc_oracle=m5_arc_or, m5_global_oracle=m5_glob_or,
        m6=(kappa < 0).to(pred_psi.dtype).flatten(1).mean(1),
    )
    return {k: v.detach().float().cpu() for k, v in out.items()}
