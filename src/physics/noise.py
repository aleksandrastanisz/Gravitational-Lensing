import torch
import torch.nn.functional as F


def true_sigma(clean, sigma0, gain=500.0):
    s0 = torch.as_tensor(sigma0, dtype=clean.dtype, device=clean.device).view(-1, 1, 1, 1)
    return torch.sqrt(s0**2 + clean.clamp(min=0.0) / gain)


def _mad_sigma(resid, mask=None):
    if mask is not None:
        resid = resid.masked_fill(~mask, float("nan"))
    med = torch.nanmedian(resid, dim=1).values[:, None]
    mad = torch.nanmedian((resid - med).abs(), dim=1).values
    return 1.4826 * mad / (8.0 / 9.0) ** 0.5


def estimate_sigma(obs, gain=500.0, k=3.0, min_pixels=500):
    resid = (obs - F.avg_pool2d(obs, 3, stride=1, padding=1, count_include_pad=False)).flatten(1)
    s_coarse = _mad_sigma(resid)
    bg = F.avg_pool2d(obs, 7, stride=1, padding=3, count_include_pad=False).flatten(1) < k * s_coarse[:, None] / 7.0
    s0 = torch.where(bg.sum(1) >= min_pixels, _mad_sigma(resid, bg), s_coarse).view(-1, 1, 1, 1)
    smooth = F.avg_pool2d(obs, 5, stride=1, padding=2, count_include_pad=False)
    return torch.sqrt(s0**2 + smooth.clamp(min=0.0) / gain)