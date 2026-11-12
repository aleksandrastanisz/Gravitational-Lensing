from functools import partial
from src.data.generator import diag_shifts, generate
from src.data.splits import TEST
from src.data.store import write_dataset
from src.utils.config import SimulationConfig

cfg = SimulationConfig.from_yaml()
for sh in diag_shifts():
    write_dataset(f"data/diag/{sh.axis}_L{sh.level}", TEST, cfg, partial(generate, shift=sh))
