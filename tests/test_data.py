import numpy as np
import torch

from src.data.generator import OUT_ORDER, Shift, generate
from src.data.splits import TEST, TRAIN, VAL
from src.physics.noise import estimate_sigma, true_sigma
from src.utils.config import SimulationConfig

CFG = SimulationConfig.from_yaml()
IDX = {k: i for i, k in enumerate(OUT_ORDER)}


def test_seed_ranges_disjoint():
    s = [set(TRAIN), set(VAL), set(TEST)]
    assert not (s[0] & s[1]) and not (s[0] & s[2]) and not (s[1] & s[2])


def test_generate_bitwise_reproducible():
    a, b = generate(123, CFG), generate(123, CFG)
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_a1_pairing_and_direction():
    for seed in (200_000, 200_123, 200_499):
        base = generate(seed, CFG)["params"]
        for lvl in (1, 2, 3):
            sh = generate(seed, CFG, Shift("A1_theta_e", lvl))["params"]
            changed = np.flatnonzero(base != sh)
            assert list(changed) == [IDX["theta_e"]]
            assert sh[IDX["theta_e"]] > CFG.theta_e_max
            q = generate(seed, CFG, Shift("A1_q", lvl))["params"]
            assert q[IDX["q_lens"]] < CFG.q_lens_min


def test_a5_same_clean_image():
    a, b = generate(200_007, CFG), generate(200_007, CFG, Shift("A5", 3))
    assert np.array_equal(a["clean_lensed"], b["clean_lensed"])
    assert not np.array_equal(a["observed"], b["observed"])


def test_sigma_estimator():
    errs = []
    for s in range(200_000, 200_150):
        d = generate(s, CFG)
        s0 = d["params"][IDX["sigma0"]]
        sig_true = true_sigma(torch.from_numpy(d["clean_lensed"])[None], s0, CFG.gain)
        sig_hat = estimate_sigma(torch.from_numpy(d["observed"])[None], CFG.gain)
        errs.append(abs((sig_hat / sig_true).median().item() - 1.0))
    print(f"\nmedian: {np.median(errs):.5f}, p90: {np.percentile(errs, 90):.5f}")
    assert np.median(errs) < 0.05 and np.percentile(errs, 90) < 0.08