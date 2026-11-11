import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.spectral_conv import SpectralConv2d


class FNOLensingReconstructor(nn.Module):
    def __init__(self, modes: int = 32, width: int = 64, pad_ratio: float = 0.25):
        super().__init__()
        self.width = width
        self.pad_ratio = pad_ratio
        self.fc0 = nn.Conv2d(3, self.width, kernel_size=1)

        self.conv0 = SpectralConv2d(self.width, self.width, modes, modes)
        self.conv1 = SpectralConv2d(self.width, self.width, modes, modes)
        self.conv2 = SpectralConv2d(self.width, self.width, modes, modes)
        self.conv3 = SpectralConv2d(self.width, self.width, modes, modes)

        self.w0 = nn.Conv2d(self.width, self.width, kernel_size=1)
        self.w1 = nn.Conv2d(self.width, self.width, kernel_size=1)
        self.w2 = nn.Conv2d(self.width, self.width, kernel_size=1)
        self.w3 = nn.Conv2d(self.width, self.width, kernel_size=1)

        self.psi_head = nn.Sequential(
            nn.Conv2d(self.width, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )

        self.source_head = nn.Sequential(
            nn.Conv2d(self.width, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Softplus()
        )
        self.source_head[4].bias.data.fill_(-3.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, _, h, w = x.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, h, device=x.device),
            torch.linspace(-1, 1, w, device=x.device),
            indexing="ij"
        )
        grid = torch.stack((grid_x, grid_y), dim=0).unsqueeze(0).repeat(b, 1, 1, 1)
        x = torch.cat((x, grid), dim=1)
        x = self.fc0(x)

        pad_h = int(h * self.pad_ratio)
        pad_w = int(w * self.pad_ratio)
        x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="constant", value=0.0)

        x_pad = F.gelu(self.conv0(x_pad) + self.w0(x_pad))
        x_pad = F.gelu(self.conv1(x_pad) + self.w1(x_pad))
        x_pad = F.gelu(self.conv2(x_pad) + self.w2(x_pad))
        x_pad = self.conv3(x_pad) + self.w3(x_pad)

        x = x_pad[..., :h, :w]
        pred_psi = self.psi_head(x)
        pred_source = self.source_head(x)
        pred_psi = pred_psi - pred_psi.mean(dim=(-2, -1), keepdim=True)

        return pred_psi, pred_source