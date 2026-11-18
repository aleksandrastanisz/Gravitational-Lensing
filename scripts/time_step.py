"""Time per training step and peak GPU memory for one architecture (V3 loss, batch 16, real data).

Usage: python -m scripts.time_step --arch unet
"""
import argparse
import time

import torch

from src.data.store import NpyDataset
from src.losses.lensing_loss import LensingLoss, loss_config
from src.training.trainer import build_model
from src.models.unet_reconstructor import count_real_params

p = argparse.ArgumentParser()
p.add_argument("--arch", default="unet", choices=["fno", "unet"])
p.add_argument("--steps", type=int, default=30)
p.add_argument("--benchmark", action="store_true", help="let cuDNN pick the fastest convolution algorithms")
a = p.parse_args()

dev = torch.device("cuda")
torch.backends.cudnn.benchmark = a.benchmark
torch.manual_seed(0)
model = build_model(a.arch).to(dev)
loss = LensingLoss(loss_config("V3", 0.02, 15.0)).to(dev)
opt = torch.optim.AdamW(model.parameters(), 1e-3)
ds = NpyDataset("data/train", n=64)
b = {k: torch.stack([ds[i][k] for i in range(16)]).to(dev) for k in ds[0]}


def step():
    psi, src = model(b["observed"])
    l, _ = loss(psi, src, b, 1.0)
    opt.zero_grad(set_to_none=True)
    l.backward()
    opt.step()


for _ in range(5):
    step()
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
t = time.perf_counter()
for _ in range(a.steps):
    step()
torch.cuda.synchronize()
s = (time.perf_counter() - t) / a.steps
print(f"{a.arch}{' (cudnn.benchmark)' if a.benchmark else ''}: {count_real_params(model) / 1e6:.2f} M real parameters, {s:.3f} s/step, "
      f"peak GPU memory {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB, "
      f"12 500 steps = {12_500 * s / 60:.0f} min without evaluation")
