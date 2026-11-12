import numpy as np
import torch

from src.data.generator import generate
from src.data.store import write_dataset
from src.evaluation.runner import datasets, evaluate
from src.models.fno_reconstructor import FNOLensingReconstructor
from src.utils.config import SimulationConfig

CFG = SimulationConfig.from_yaml()


def test_non_final_mode_touches_only_val(tmp_path):
    for d in ("val", "test_id", "ood/A1_q_L1", "diag/Kneg_L1"):
        (tmp_path / d).mkdir(parents=True)
    assert set(datasets(tmp_path, final=False)) == {"val"}
    assert set(datasets(tmp_path, final=True)) == {"test_id", "ood/A1_q_L1", "diag/Kneg_L1"}


def test_evaluate_returns_one_value_per_sample(tmp_path):
    write_dataset(tmp_path / "val", range(3), CFG, generate)
    torch.manual_seed(0)
    model = FNOLensingReconstructor(modes=32, width=64, pad_ratio=0.25).eval()
    m = evaluate(model, tmp_path / "val", CFG, torch.device("cpu"), batch=2)
    assert {"m1", "m1b", "m2", "m3", "m4", "m5_arc", "m6"} <= set(m)
    for k, v in m.items():
        assert v.shape == (3,), k
    for k in ("m1", "m1b", "m2", "m3", "m4", "m5_global", "m6"):
        assert np.isfinite(m[k]).all(), k
