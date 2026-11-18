"""Control S2 (PROTOCOL §7, §9): U-Net instead of FNO, V0 against V3, N = 2 000, seeds 42-46.

Step 1: the learning rate of U-Net V0 is tuned on the grid of §7.1 (3 runs, weight decay from
configs/tuned.yaml, N = 2 000, seed 100, criterion val_select). Results go to results/tuning_s2.json.
Step 2: U-Net V0 and V3 with that learning rate; weight decay and loss weights from tuned.yaml.
Finished runs are skipped, so the script can be interrupted and started again.
"""
import json
from pathlib import Path

import yaml

from src.training.trainer import TrainConfig, run

N, TUNE_SEED, SEEDS = 2_000, 100, range(42, 47)
LR_GRID = (3e-4, 1e-3, 3e-3)

tuned = yaml.safe_load(open("configs/tuned.yaml"))
rows = []
for lr in LR_GRID:
    meta = run(TrainConfig("V0", N, TUNE_SEED, lr=lr, wd=tuned["V0"]["wd"], tag=f"S2tune_lr{lr:g}",
                           save_ckpt=False, arch="unet"))
    rows.append(dict(variant="V0", arch="unet", lr=lr, wd=tuned["V0"]["wd"], val_select=meta["best"]["val_select"],
                     step=meta["best"]["step"], seconds=meta["seconds"]))
    print("->", rows[-1], flush=True)
best = min(rows, key=lambda r: r["val_select"])
edge = "lower grid edge" if best["lr"] == min(LR_GRID) else "upper grid edge" if best["lr"] == max(LR_GRID) else "interior"
print(f"== U-Net V0: selected lr {best['lr']:g} ({edge})", flush=True)
Path("results").mkdir(exist_ok=True)
Path("results/tuning_s2.json").write_text(json.dumps(dict(runs=rows, selected_lr=best["lr"], position=edge), indent=1))

for seed in SEEDS:
    for v in ("V0", "V3"):
        hp = {k: x for k, x in tuned[v].items() if k != "lr"}
        run(TrainConfig(v, N, seed, lr=best["lr"], **hp, tag="S2", arch="unet"))
