"""Control X3 (PROTOCOL §9): one repeat of the selected V0 and V3 tuning runs (N = 2 000, seed 100).

Identical settings to the tuning runs, so any difference in val_select is run-to-run noise from
GPU non-determinism. Descriptive only; it does not change the tuning choice.
"""
import json
from pathlib import Path

import yaml

from src.training.trainer import TrainConfig, run

tuned = yaml.safe_load(open("configs/tuned.yaml"))
tuning = json.loads(Path("results/tuning.json").read_text())
rows = []
for v in ("V0", "V3"):
    hp = tuned[v]
    first = [r for r in tuning if r["variant"] == v and all(r.get(k) == x for k, x in hp.items())]
    assert len(first) == 1, f"{v}: expected one matching tuning run, found {len(first)}"
    meta = run(TrainConfig(v, 2000, 100, tag="X3", save_ckpt=False, **hp))
    a, b = first[0]["val_select"], meta["best"]["val_select"]
    rows.append(dict(variant=v, **hp, val_select_tuning=a, val_select_repeat=b,
                     relative_difference=b / a - 1.0, step_tuning=first[0]["step"],
                     step_repeat=meta["best"]["step"]))
    print(f"{v}: tuning {a:.4f}, repeat {b:.4f}, difference {100 * (b / a - 1):+.2f}%", flush=True)
Path("results/x3.json").write_text(json.dumps(rows, indent=1))
