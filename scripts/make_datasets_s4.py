"""Control S4 (PROTOCOL §9): every split regenerated with A_model as the observation operator.

Same seeds, parameters and noise draws as the E-main data; only the operator that renders the
clean image differs (128 grid instead of the 255 grid of A_truth). Written to data_s4/.
"""
import json
from functools import partial
from pathlib import Path

from src.data.generator import all_shifts, generate
from src.data.splits import TEST, TRAIN, VAL
from src.data.store import write_dataset
from src.physics.operators import a_inverse_crime
from src.utils.config import SimulationConfig

ROOT = "data_s4"
cfg = SimulationConfig.from_yaml()
jobs = [(name, seeds, None) for name, seeds in (("train", TRAIN), ("val", VAL), ("test_id", TEST))]
jobs += [(f"ood/{sh.axis}_L{sh.level}", TEST, sh) for sh in all_shifts()]
for name, seeds, sh in jobs:
    out = Path(ROOT) / name
    if (out / "meta.json").exists():
        print(f"{name}: exists, skipped", flush=True)
        continue
    write_dataset(out, seeds, cfg, partial(generate, shift=sh, operator=a_inverse_crime))
    meta = json.loads((out / "meta.json").read_text())
    meta["operator"] = "A_model (a_inverse_crime)"
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"{name}: done", flush=True)
