"""Regenerates the first samples of the validation set with the installed PyTorch and compares
them with the stored arrays. Reports, per array, whether the match is bitwise and the largest
absolute difference. Uses validation data only.
"""
import json

import numpy as np
import torch

from src.data.generator import generate
from src.data.store import KEYS, NpyDataset
from src.utils.config import SimulationConfig

N = 50
cfg = SimulationConfig.from_yaml()
ds = NpyDataset("data/val")
meta = json.loads(open("data/val/meta.json").read())
print(f"stored with torch {meta['torch']}, regenerated with torch {torch.__version__}")
for k in [*KEYS, "params"]:
    stored = np.array(ds.a[k][:N])
    fresh = np.stack([generate(meta["seed_first"] + i, cfg)[k] for i in range(N)]).astype(stored.dtype)
    diff = float(np.abs(stored.astype(np.float64) - fresh.astype(np.float64)).max())
    print(f"{k:13s} bitwise {np.array_equal(stored, fresh)!s:5s} max |diff| {diff:.3e}  (max |value| {float(np.abs(stored).max()):.3g})")
