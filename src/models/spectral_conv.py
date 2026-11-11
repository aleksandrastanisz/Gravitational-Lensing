import math
import torch
import torch.nn as nn


class SpectralConv2d(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.modes1, self.modes2 = modes1, modes2
        scale = 1.0 / math.sqrt(in_channels)
        shape = (modes1 * modes2, in_channels, out_channels)
        self.weights1 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    @staticmethod
    def _mix(x, w):
        b, c, m1, m2 = x.shape
        xs = x.permute(2, 3, 0, 1).reshape(m1 * m2, b, c)
        return torch.bmm(xs, w).reshape(m1, m2, b, -1).permute(2, 3, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        m1, m2 = self.modes1, self.modes2
        xh = torch.fft.fft(torch.fft.rfft(x, dim=-1)[..., :m2], dim=-2)
        lo = self._mix(xh[:, :, :m1], self.weights1)
        hi = self._mix(xh[:, :, -m1:], self.weights2)
        mid = lo.new_zeros(b, self.out_channels, h - 2 * m1, m2)
        out = torch.fft.ifft(torch.cat([lo, mid, hi], dim=2), dim=-2)
        return torch.fft.irfft(out, n=w, dim=-1)