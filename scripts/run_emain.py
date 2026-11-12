import yaml
from src.training.trainer import TrainConfig, run

tuned = yaml.safe_load(open("configs/tuned.yaml"))
for n in (2000, 500, 8000):
    for seed in range(42, 47):
        for v in ("V0", "V1", "V2", "V3"):
            run(TrainConfig(v, n, seed, **tuned[v]))