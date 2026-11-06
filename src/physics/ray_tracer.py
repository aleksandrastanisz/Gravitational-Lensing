import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GravitationalRayTracer(nn.Module):
    def __init__(self, extent: float = 3.0, psf_fwhm: float = 0.1):
        super().__init__()
        self.extent = extent
        self.psf_fwhm = psf_fwhm

    def forward(self, source: torch.Tensor, psi: torch.Tensor) -> torch.Tensor:
        if psi.dim() == 4:
            psi = psi.squeeze(1)

        h, w = psi.shape[-2], psi.shape[-1]
        dy = (2.0 * self.extent) / (h - 1)
        dx = (2.0 * self.extent) / (w - 1)

        grad_y, grad_x = torch.gradient(psi, spacing=(dy, dx), dim=(-2, -1))

        y, x = torch.meshgrid(
            torch.linspace(-self.extent, self.extent, h, device=psi.device, dtype=psi.dtype),
            torch.linspace(-self.extent, self.extent, w, device=psi.device, dtype=psi.dtype),
            indexing="ij"
        )

        beta_x = x - grad_x
        beta_y = y - grad_y

        norm_x = beta_x / self.extent
        norm_y = beta_y / self.extent
        grid = torch.stack((norm_x, norm_y), dim=-1)

        lensed_image = F.grid_sample(
            source,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True
        )

        if self.psf_fwhm > 0:
            sigma_px = (self.psf_fwhm / 2.354820045) / dx
            radius = max(math.ceil(3.0 * sigma_px), 1)
            coords = torch.arange(-radius, radius + 1, device=source.device, dtype=source.dtype)
            ky, kx = torch.meshgrid(coords, coords, indexing="ij")
            kernel = torch.exp(-0.5 * (kx ** 2 + ky ** 2) / (sigma_px ** 2))
            kernel = (kernel / kernel.sum()).unsqueeze(0).unsqueeze(0)
            padded = F.pad(lensed_image, (radius, radius, radius, radius), mode="replicate")
            lensed_image = F.conv2d(padded, kernel)

        return lensed_image