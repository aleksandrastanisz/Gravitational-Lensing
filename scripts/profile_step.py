import torch
from torch.profiler import ProfilerActivity, profile

from src.data.store import NpyDataset
from src.losses.lensing_loss import LensingLoss, loss_config
from src.models.fno_reconstructor import FNOLensingReconstructor

dev = torch.device("cuda")
torch.backends.cudnn.benchmark = True
model = FNOLensingReconstructor(32, 64, 0.25).to(dev)
loss = LensingLoss(loss_config("V3", 0.02, 15.0)).to(dev)
opt = torch.optim.AdamW(model.parameters(), 1e-3)
ds = NpyDataset("data/train", n=64)
b = {k: torch.stack([ds[i][k] for i in range(16)]).to(dev) for k in ds[0]}


def step():
    psi, src = model(b["observed"])
    l, _ = loss(psi, src, b, 1.0)
    opt.zero_grad()
    l.backward()
    opt.step()


for _ in range(5):
    step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    for _ in range(10):
        step()
    torch.cuda.synchronize()
print(prof.key_averages(group_by_input_shape=True).table(sort_by="self_cuda_time_total", row_limit=15))