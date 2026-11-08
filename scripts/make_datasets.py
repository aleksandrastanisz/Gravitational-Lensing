from functools import partial
from src.data.generator import all_shifts, generate
from src.data.splits import TEST, TRAIN, VAL
from src.data.store import write_dataset
from src.utils.config import SimulationConfig

cfg = SimulationConfig.from_yaml()
for name, seeds in [("train", TRAIN), ("val", VAL), ("test_id", TEST)]:
    write_dataset(f"data/{name}", seeds, cfg, generate)
for sh in all_shifts():
    write_dataset(f"data/ood/{sh.axis}_L{sh.level}", TEST, cfg, partial(generate, shift=sh))