import numpy as np
import torch

from src.training.trainer import TrainConfig, batches, warm


class FakeDS:
    def __init__(self, n):
        self.n = n
        self.a = {"observed": np.arange(n, dtype=np.float32).reshape(n, 1, 1, 1),
                  "params": np.zeros((n, 18))}


def _ids(seed, n=500, batch=16, epochs=2):
    it = batches(FakeDS(n), batch, seed)
    return [next(it)["observed"].flatten().long().tolist() for _ in range(epochs * (n // batch))]


def test_batches_cover_epoch_without_repeats_and_depend_only_on_seed():
    a = _ids(42)
    flat = [i for b in a[:31] for i in b]
    assert len(flat) == len(set(flat)) == 31 * 16
    assert all(len(b) == 16 for b in a)
    assert a == _ids(42) and a != _ids(43)


def test_warmup_schedule():
    c = TrainConfig("V3", 2000, 42)
    assert warm(0, c) == 0.0 and warm(1500, c) == 0.0
    assert abs(warm(2250, c) - 0.5) < 1e-9 and warm(3000, c) == 1.0 and warm(12_000, c) == 1.0