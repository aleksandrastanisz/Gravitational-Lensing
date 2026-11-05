import torch


def rotate_coords(x, y, x0, y0, phi):
    dx, dy = x - x0, y - y0
    if not isinstance(phi, torch.Tensor):
        phi = torch.as_tensor(phi, dtype=x.dtype, device=x.device)
    c, s = torch.cos(phi), torch.sin(phi)
    return dx * c + dy * s, -dx * s + dy * c


def elliptical_potential(x, y, theta_e, q=1.0, x0=0.0, y0=0.0, phi=0.0,
                         gamma1=0.0, gamma2=0.0, core_radius=0.05, slope=1.0):
    xr, yr = rotate_coords(x, y, x0, y0, phi)
    r = torch.sqrt(q * xr**2 + yr**2 / q + core_radius**2) #
    psi = theta_e ** (2.0 - slope) * r**slope / slope
    return psi + 0.5 * gamma1 * (x**2 - y**2) + gamma2 * x * y


def plummer_potential(x, y, b, xs, ys, a=0.1):
    return b * a * torch.log((x - xs) ** 2 + (y - ys) ** 2 + a**2)


def sersic_source(x, y, amp=1.0, r_eff=0.2, n=1.0, x0=0.0, y0=0.0, q=1.0, phi=0.0):
    xr, yr = rotate_coords(x, y, x0, y0, phi)
    r = torch.sqrt(q * xr**2 + yr**2 / q + 1e-8)
    bn = 1.9992 * n - 0.3271
    return amp * torch.exp(-bn * ((r / r_eff) ** (1.0 / n) - 1.0))


def gaussian_clumps(x, y, clumps):
    out = torch.zeros_like(x)
    for a, cx, cy, s in clumps:
        out = out + a * torch.exp(-0.5 * ((x - cx) ** 2 + (y - cy) ** 2) / s**2)
    return out