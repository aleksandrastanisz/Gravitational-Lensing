import numpy as np, torch
from src.data.generator import generate
from src.losses.lensing_loss import LensingLoss, LossConfig
from src.physics.noise import estimate_sigma
from src.utils.config import SimulationConfig

cfg = SimulationConfig.from_yaml()
loss = LensingLoss(LossConfig(lambda_ray=1.0))
arc, glob = [], []
for s in range(200_000, 200_100):
    d = {k: torch.from_numpy(v)[None] for k, v in generate(s, cfg).items() if k != "params"}
    pull = (loss.tracer(d["source"], d["psi"]) - d["observed"]) / estimate_sigma(d["observed"], cfg.gain)
    mask = d["clean_lensed"] > 3 * d["noise_map"]
    arc.append(pull[mask].pow(2).mean().item()); glob.append(pull.pow(2).mean().item())
print(f"chi2 na prawdziwych (psi, source): łuk {np.median(arc):.3f}, globalnie {np.median(glob):.3f}")