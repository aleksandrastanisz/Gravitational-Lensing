import math

import numpy as np
import pytest

from src.evaluation import stats as S

SEEDS = list(range(42, 47))
SETS = ["test_id"] + S.E1_SETS


def _fake(factor, n=500, jitter=0.0, rng_seed=0):
    """V0 lognormal per sample; V3 = V0 * factor(set) * exp(jitter * N(0,1)) per seed."""
    rng = np.random.default_rng(rng_seed)
    M = {"V0": {}, "V3": {}, "V1": {}, "V2": {}}
    for s in SEEDS:
        base = {a: np.exp(rng.normal(-1.0, 0.5, n)) for a in SETS}
        M["V0"][s] = base
        for v in ("V1", "V2", "V3"):
            M[v][s] = {a: base[a] * factor(a) * math.exp(jitter * rng.normal()) for a in SETS}
    return M


def test_t_distribution_df4():
    assert S.t_cdf_df4(0.0) == pytest.approx(0.5)
    assert S.t_cdf_df4(S.T975_DF4) == pytest.approx(0.975, abs=1e-9)
    assert S.t_pvalue(np.array([0.0, 0.1, -0.1, 0.05, -0.05])) == pytest.approx(1.0)


def test_holm_matches_hand_computation():
    adj = S.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj == pytest.approx({"a": 0.03, "b": 0.06, "c": 0.06})


def test_h1_supported_for_10_percent_gain():
    M = _fake(lambda a: 0.9, jitter=0.01)
    r = S.decide_h1(M, SEEDS, B=500)
    assert r["mean"] == pytest.approx(math.log(0.9), abs=0.02)
    assert r["supported"], r["criteria"]
    assert r["interval"][0] <= r["mean"] <= r["interval"][1]


def test_h1_not_supported_without_effect():
    M = _fake(lambda a: 1.0, jitter=0.02, rng_seed=1)
    r = S.decide_h1(M, SEEDS, B=500)
    assert not r["criteria"]["mean_le_threshold"] and not r["supported"]


def test_a1_enters_e1_as_one_axis():
    M = _fake(lambda a: 0.5 if a.startswith("ood/A1_") else 1.0)
    e = S.e1(M, "V3", SEEDS)
    assert np.allclose(e, math.log(0.5) / 6)


def test_h2_detects_effect_growing_as_n_falls():
    f = {500: 0.8, 2_000: 0.9, 8_000: 1.0}
    M_by_N = {n: _fake(lambda a, n=n: f[n], jitter=0.005, rng_seed=n) for n in S.NS}
    r = S.decide_h2(M_by_N, SEEDS)
    assert r["mean"] > 0 and r["supported"]


def test_h2_threshold_rejects_small_but_consistent_slope():
    f = {500: 0.99, 2_000: 0.995, 8_000: 1.0}
    M_by_N = {n: _fake(lambda a, n=n: f[n], jitter=0.0005, rng_seed=n) for n in S.NS}
    r = S.decide_h2(M_by_N, SEEDS)
    assert r["criteria"]["ci_excludes_zero"] and not r["criteria"]["mean_ge_threshold"]
    assert not r["supported"]
    assert abs(S.H2_THRESHOLD - 0.0185) < 1e-4


def test_h3_non_inferiority_margin():
    ok = S.decide_h3({"m1": _fake(lambda a: 1.02, jitter=0.005)}, SEEDS, B=500)
    bad = S.decide_h3({"m1": _fake(lambda a: 1.10, jitter=0.005)}, SEEDS, B=500)
    assert ok["supported"] and not bad["supported"]


def test_secondary_has_18_tests_and_holm():
    r = S.secondary(_fake(lambda a: 0.9, jitter=0.01), SEEDS)
    assert len(r) == 18 and all(v["p_holm"] >= v["p"] for v in r.values())


def test_wider_interval_governs():
    assert S.wider(-1, 1, -2, 2)[2] == "bootstrap" and S.wider(-2, 2, -1, 1)[2] == "t"
