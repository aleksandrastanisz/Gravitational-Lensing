import math
import torch
import torch.nn as nn

from src.models.spectral_conv import SpectralConv2d


class OldSpectralConv2d(nn.Module):
    def __init__(self, i, o, m1, m2):
        super().__init__()
        self.o, self.m1, self.m2 = o, m1, m2
        s = 1.0 / math.sqrt(i)
        self.weights1 = nn.Parameter(s * torch.randn(i, o, m1, m2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(s * torch.randn(i, o, m1, m2, dtype=torch.cfloat))

    def forward(self, x):
        xf = torch.fft.rfft2(x)
        out = torch.zeros(x.shape[0], self.o, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat, device=x.device)
        out[:, :, :self.m1, :self.m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, :self.m1, :self.m2], self.weights1)
        out[:, :, -self.m1:, :self.m2] = torch.einsum("bixy,ioxy->boxy", xf[:, :, -self.m1:, :self.m2], self.weights2)
        return torch.fft.irfft2(out, s=(x.size(-2), x.size(-1)))


def test_new_spectral_layer_matches_old_forward_and_gradients():
    torch.manual_seed(0)
    b, c, o, h, w, m1, m2 = 2, 6, 5, 40, 36, 6, 6
    new, old = SpectralConv2d(c, o, m1, m2), OldSpectralConv2d(c, o, m1, m2)
    with torch.no_grad():
        for n_, o_ in ((new.weights1, old.weights1), (new.weights2, old.weights2)):
            o_.copy_(n_.reshape(m1, m2, c, o).permute(2, 3, 0, 1))
    x = torch.randn(b, c, h, w, requires_grad=True)
    x2 = x.detach().clone().requires_grad_(True)
    y_new, y_old = new(x), old(x2)
    assert torch.allclose(y_new, y_old, atol=1e-5)
    g = torch.randn_like(y_new)
    y_new.backward(g)
    y_old.backward(g)
    assert torch.allclose(x.grad, x2.grad, atol=1e-4)
    gw = new.weights1.grad.reshape(m1, m2, c, o).permute(2, 3, 0, 1)
    assert torch.allclose(gw, old.weights1.grad, atol=1e-4)