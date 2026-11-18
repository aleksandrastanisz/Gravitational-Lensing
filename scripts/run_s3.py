"""Control S3 (PROTOCOL §9): V2 and V3 with the true noise map (oracle sigma) in L_ray instead of
the estimate, N = 2 000, seeds 42-46, configurations from configs/tuned.yaml."""
import yaml

from src.training.trainer import TrainConfig, run

tuned = yaml.safe_load(open("configs/tuned.yaml"))
for seed in range(42, 47):
    for v in ("V2", "V3"):
        run(TrainConfig(v, 2_000, seed, **tuned[v], sigma_mode="oracle", tag="S3"))
