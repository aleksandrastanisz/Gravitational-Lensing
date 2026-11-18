"""Control X1 (PROTOCOL §9): V4 = L_ray + L_pos without labels, N = 2 000, seeds 42-46.
Learning rate, weight decay and loss weights from the tuned V3; checkpoint selected by val_select
(this uses the labelled validation set, as for every variant). V4 has no warm-up (see trainer.warm)."""
import yaml

from src.training.trainer import TrainConfig, run

v3 = yaml.safe_load(open("configs/tuned.yaml"))["V3"]
for seed in range(42, 47):
    run(TrainConfig("V4", 2_000, seed, **v3, tag="X1"))
