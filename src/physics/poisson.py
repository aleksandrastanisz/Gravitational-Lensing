import torch
import torch.nn as nn
import torch.nn.functional as F

class PoissonSolver2D(nn.Module):
    def __init__(self, extent: float = 3.0):
        super().__init__()
        self.extent = extent

        kernel = torch.tensor([
            [0.0, 1.0, 0.0],
            [1.0, -4.0, 1.0],
            [0.0, 1.0, 0.0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        self.register_buffer("kernel", kernel)

    def forward(self, psi: torch.Tensor) -> torch.Tensor:
        if psi.dim() == 3:
            psi = psi.unsqueeze(1)

        dx = (2.0 * self.extent) / (psi.shape[-1] - 1)
        scale = 1.0 / (dx ** 2)

        pad_left = 2.0 * psi[..., :, :1] - psi[..., :, 1:2]
        pad_right = 2.0 * psi[..., :, -1:] - psi[..., :, -2:-1]
        psi_x = torch.cat([pad_left, psi, pad_right], dim=-1)

        pad_top = 2.0 * psi_x[..., :1, :] - psi_x[..., 1:2, :]
        pad_bottom = 2.0 * psi_x[..., -1:, :] - psi_x[..., -2:-1, :]
        psi_padded = torch.cat([pad_top, psi_x, pad_bottom], dim=-2)

        laplacian = F.conv2d(psi_padded, self.kernel * scale, padding=0)
        kappa = 0.5 * laplacian
        return kappa