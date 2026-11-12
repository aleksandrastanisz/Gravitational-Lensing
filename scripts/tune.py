import json
from pathlib import Path

import yaml

from src.training.trainer import TrainConfig, run

N, SEED = 2000, 100
LR, WD = [3e-4, 1e-3, 3e-3], [1e-4, 1e-3, 1e-2]
LRAY, LPOS = [0.005, 0.02, 0.1], [1, 15, 50]
GRIDS = {"lr": LR, "wd": WD, "lambda_ray": LRAY, "lambda_pos": LPOS}
rows = []


def trial(variant, **hp):
    tag = "tune_" + "_".join(f"{k}{v:g}" for k, v in hp.items())
    meta = run(TrainConfig(variant, N, SEED, tag=tag, save_ckpt=False, **hp))
    row = dict(variant=variant, **hp, val_select=meta["best"]["val_select"],
               step=meta["best"]["step"], seconds=meta["seconds"])
    rows.append(row)
    print("->", row, flush=True)
    return row


def stage(variant, grid, tuned_keys):
    best = min((trial(variant, **hp) for hp in grid), key=lambda r: r["val_select"])
    chosen = {k: best[k] for k in tuned_keys}
    where = {k: ("dolna granica siatki" if best[k] == min(GRIDS[k])
                 else "górna granica siatki" if best[k] == max(GRIDS[k]) else "wnętrze")
             for k in tuned_keys}
    print(f"== {variant}: wybrane {chosen}; położenie: {where}", flush=True)
    return best


b0 = stage("V0", [dict(lr=l, wd=w) for l in LR for w in WD], ["lr", "wd"])
lr0, wd0 = b0["lr"], b0["wd"]
b1 = stage("V1", [dict(lr=lr0, lambda_pos=p, wd=w) for p in LPOS for w in WD], ["lambda_pos", "wd"])
b2 = stage("V2", [dict(lr=lr0, lambda_ray=r, wd=w) for r in LRAY for w in WD], ["lambda_ray", "wd"])
b3 = stage("V3", [dict(lr=lr0, wd=wd0, lambda_ray=r, lambda_pos=p) for r in LRAY for p in LPOS],
           ["lambda_ray", "lambda_pos"])

pick = lambda b, keys: {k: b[k] for k in keys}
tuned = {"V0": pick(b0, ["lr", "wd"]),
         "V1": pick(b1, ["lr", "wd", "lambda_pos"]),
         "V2": pick(b2, ["lr", "wd", "lambda_ray"]),
         "V3": pick(b3, ["lr", "wd", "lambda_ray", "lambda_pos"])}
Path("configs/tuned.yaml").write_text(yaml.safe_dump(tuned, sort_keys=False))
Path("results").mkdir(exist_ok=True)
Path("results/tuning.json").write_text(json.dumps(rows, indent=1))
print("\nwybrane:", tuned)
print(f"łączny czas treningu: {sum(r['seconds'] for r in rows) / 3600:.1f} h")