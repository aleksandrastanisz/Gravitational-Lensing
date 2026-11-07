import torch
import torch.nn.functional as F


def true_sigma(clean, sigma0, gain=500.0):
    s0 = torch.as_tensor(sigma0, dtype=clean.dtype, device=clean.device).view(-1, 1, 1, 1)
    return torch.sqrt(s0**2 + clean.clamp(min=0.0) / gain)


def estimate_sigma(obs, gain=500.0):
    mean3 = F.avg_pool2d(obs, 3, stride=1, padding=1, count_include_pad=False)
    resid = (obs - mean3).flatten(1)
    med = resid.median(dim=1).values[:, None]
    mad = (resid - med).abs().median(dim=1).values
    s0 = 1.4826 * mad / (8.0 / 9.0) ** 0.5
    smooth = F.avg_pool2d(obs, 5, stride=1, padding=2, count_include_pad=False)
    s0 = s0.view(-1, 1, 1, 1)
    return torch.sqrt(s0**2 + smooth.clamp(min=0.0) / gain)