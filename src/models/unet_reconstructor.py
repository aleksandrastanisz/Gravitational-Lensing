"""U-Net reconstructor for control S2 (PROTOCOL §7).

Same interface as FNOLensingReconstructor: the observed image plus two coordinate channels go in,
(psi, source) come out, psi has zero mean and the source is positive. The two output heads are
identical to those of the FNO. With base = 96 the model has about 70.0 M real parameters, within
10% of the FNO (about 67.3 M, complex weights counted twice).
"""
import torch
import torch.nn as nn


def count_real_params(model: nn.Module) -> int:
    """Number of real parameters; a complex weight counts twice (PROTOCOL §7)."""
    return sum(p.numel() * (2 if p.is_complex() else 1) for p in model.parameters())


def _block(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1), nn.GroupNorm(8, c_out), nn.GELU(),
        nn.Conv2d(c_out, c_out, kernel_size=3, padding=1), nn.GroupNorm(8, c_out), nn.GELU(),
    )


class UNetLensingReconstructor(nn.Module):
    def __init__(self, base: int = 96, depth: int = 4):
        super().__init__()
        ch = [base * 2 ** i for i in range(depth + 1)]
        self.enc = nn.ModuleList([_block(3, ch[0])] + [_block(a, b) for a, b in zip(ch, ch[1:])])
        self.pool = nn.MaxPool2d(2)
        rev = ch[::-1]
        self.up = nn.ModuleList([nn.ConvTranspose2d(a, b, kernel_size=2, stride=2) for a, b in zip(rev, rev[1:])])
        self.dec = nn.ModuleList([_block(2 * b, b) for b in rev[1:]])

        self.psi_head = nn.Sequential(
            nn.Conv2d(base, 128, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(128, 1, kernel_size=1)
        )
        self.source_head = nn.Sequential(
            nn.Conv2d(base, 128, kernel_size=3, padding=1),
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

        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x if i == 0 else self.pool(x))
            skips.append(x)
        skips.pop()
        for up, dec in zip(self.up, self.dec):
            x = dec(torch.cat((up(x), skips.pop()), dim=1))

        pred_psi = self.psi_head(x)
        pred_source = self.source_head(x)
        pred_psi = pred_psi - pred_psi.mean(dim=(-2, -1), keepdim=True)
        return pred_psi, pred_source
