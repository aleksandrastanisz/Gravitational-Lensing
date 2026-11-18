import numpy as np
import torch

from src.losses.lensing_loss import sup_per_sample
from src.models.fno_reconstructor import FNOLensingReconstructor
from src.training.trainer import TrainConfig, batches, process_git_state, val_select, warm


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

def test_val_select_total_matches_supervised_loss_and_components():
    torch.manual_seed(0)
    model = FNOLensingReconstructor(modes=32, width=64, pad_ratio=0.25)
    arrays = {"observed": torch.rand(4, 1, 128, 128), "psi": torch.rand(4, 1, 128, 128),
              "source": torch.rand(4, 1, 128, 128)}
    d = 6.0 / 127
    total, s, a = val_select(model, arrays, d, torch.device("cpu"), batch=3)
    assert model.training, "val_select must restore train mode"
    with torch.no_grad():
        model.eval()
        psi, src = model(arrays["observed"])
        ref = sup_per_sample(psi, src, arrays["psi"], arrays["source"], d).mean().item()
    assert abs(total - ref) < 1e-6 and abs(total - (s + 2 * a)) < 1e-6


def test_git_state_is_read_once_per_process():
    assert process_git_state() is process_git_state()


def test_build_model_keeps_the_fno_of_the_main_runs():
    from src.models.unet_reconstructor import UNetLensingReconstructor, count_real_params
    from src.training.trainer import build_model
    import pytest

    torch.manual_seed(0)
    a = build_model("fno")
    torch.manual_seed(0)
    b = FNOLensingReconstructor(modes=32, width=64, pad_ratio=0.25)
    assert all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))
    assert TrainConfig("V0", 2000, 42).arch == "fno"
    assert isinstance(build_model("unet"), UNetLensingReconstructor)
    assert count_real_params(build_model("unet")) == 70_029_634
    with pytest.raises(ValueError):
        build_model("resnet")


def test_runs_stored_before_the_arch_field_still_match_their_config():
    from dataclasses import asdict
    from src.training.trainer import same_config

    c = TrainConfig("V3", 2000, 42, lr=3e-3, wd=1e-3)
    old = asdict(c)
    del old["arch"]
    assert same_config(old, c) and same_config(asdict(c), c)
    assert not same_config(old, TrainConfig("V3", 2000, 42, lr=3e-3, wd=1e-3, arch="unet"))
    assert not same_config(dict(old, lr=1e-3), c)
    assert not same_config(dict(old, unknown=1), c)
