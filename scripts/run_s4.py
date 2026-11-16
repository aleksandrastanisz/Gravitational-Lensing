"""Control S4 (PROTOCOL §9): V0 and V3, N = 2 000, seeds 42-46, tuned configurations, data_s4/."""
import yaml

from src.training.trainer import TrainConfig, run

tuned = yaml.safe_load(open("configs/tuned.yaml"))
for seed in range(42, 47):
    for v in ("V0", "V3"):
        run(TrainConfig(v, 2000, seed, **tuned[v], data_root="data_s4", tag="S4"))
