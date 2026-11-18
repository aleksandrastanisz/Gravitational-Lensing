"""Control X2 (PROTOCOL §9): V3 with L_sup on the first 2 000 training samples and L_ray, L_pos on
batches drawn from all 8 000 training observations, seeds 42-46, configuration from tuned.yaml."""
import yaml

from src.training.trainer import TrainConfig, run

v3 = yaml.safe_load(open("configs/tuned.yaml"))["V3"]
for seed in range(42, 47):
    run(TrainConfig("V3", 2_000, seed, **v3, phys_pool=8_000, tag="X2"))
