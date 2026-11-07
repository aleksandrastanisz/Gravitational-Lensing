import torch
from src.physics.ray_tracer import GravitationalRayTracer


def make_grid(n: int, extent: float, device=None):
    ax = torch.linspace(-extent, extent, n, device=device)
    y, x = torch.meshgrid(ax, ax, indexing="ij")
    return x, y


def a_truth(psi_fn, src_fn, extent=3.0, grid=128, psf_fwhm=0.10):
    n_fine = 2 * (grid - 1) + 1
    xf, yf = make_grid(n_fine, extent)
    tracer = GravitationalRayTracer(extent=extent, psf_fwhm=psf_fwhm)
    with torch.no_grad():
        lensed = tracer(src_fn(xf, yf)[None, None], psi_fn(xf, yf)[None, None])
    lensed = lensed[0, :, ::2, ::2]
    xc, yc = make_grid(grid, extent)
    return lensed, psi_fn(xc, yc)[None], src_fn(xc, yc)[None]


def a_inverse_crime(psi_fn, src_fn, extent=3.0, grid=128, psf_fwhm=0.10):
    xc, yc = make_grid(grid, extent)
    psi, src = psi_fn(xc, yc)[None, None], src_fn(xc, yc)[None, None]
    tracer = GravitationalRayTracer(extent=extent, psf_fwhm=psf_fwhm)
    with torch.no_grad():
        return tracer(src, psi)[0], psi[0], src[0]