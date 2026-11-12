"""Statistics and decision rules of PROTOCOL §10 (H1, H2, H3, secondary tests).

Input is per-sample metric arrays: M[variant][seed][set] -> np.ndarray of length n,
for one training-set size N. Sets are named as in runner.datasets ("test_id", "ood/A3_L2").
All effects are log ratios of medians over samples (negative = variant better than V0).
"""
import math
from pathlib import Path

import numpy as np

N_SEEDS = 5
T975_DF4 = 2.7764451051977987          # 97.5% quantile of Student t with 4 df
NS = (500, 2_000, 8_000)
REF_N = 2_000
H1_THRESHOLD = math.log(0.95)
H3_MARGIN = math.log(1.05)
B_DEFAULT = 10_000

# Moderate level of every E1 axis; A1 is the mean of its four sub-axes (§15 item 3).
E1_AXES = {
    "A1": ["ood/A1_theta_e_L2", "ood/A1_q_L2", "ood/A1_shear_L2", "ood/A1_src_pos_L2"],
    "A2": ["ood/A2_L2"], "A3": ["ood/A3_L2"], "A4": ["ood/A4_L1"],
    "A5": ["ood/A5_L2"], "A6": ["ood/A6_L1"],
}
E1_SETS = [s for sets in E1_AXES.values() for s in sets]
E1_WEIGHTS = np.array([1.0 / len(E1_AXES) / len(sets) for sets in E1_AXES.values() for _ in sets])


# ---------------------------------------------------------------- t distribution, df = 4

def t_cdf_df4(t):
    """Exact CDF of Student t with 4 degrees of freedom."""
    x = np.asarray(t, dtype=float) / np.sqrt(4.0 + np.asarray(t, dtype=float) ** 2)
    return 0.5 + 0.5 * x * (1.0 + 0.5 * (1.0 - x * x))


def t_interval(x):
    """Mean over seeds and its 95% t interval (df = 4)."""
    x = np.asarray(x, dtype=float)
    assert x.shape == (N_SEEDS,), f"expected {N_SEEDS} seeds, got {x.shape}"
    m, se = x.mean(), x.std(ddof=1) / math.sqrt(N_SEEDS)
    return m, m - T975_DF4 * se, m + T975_DF4 * se


def t_pvalue(x):
    """Two-sided one-sample t test of mean = 0 over seeds (df = 4)."""
    x = np.asarray(x, dtype=float)
    se = x.std(ddof=1) / math.sqrt(N_SEEDS)
    if se == 0:
        return 0.0 if x.mean() != 0 else 1.0
    return float(2.0 * (1.0 - t_cdf_df4(abs(x.mean() / se))))


def holm(pvalues):
    """Holm-adjusted p-values for a dict name -> p."""
    names = sorted(pvalues, key=pvalues.get)
    m, out, running = len(names), {}, 0.0
    for i, k in enumerate(names):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


# ---------------------------------------------------------------- effects

def log_ratio(M, variant, seeds, sets, ref="V0"):
    """d[s, k] = log median(variant) - log median(ref), shape [S, K]."""
    return np.array([[math.log(np.median(M[variant][s][a])) - math.log(np.median(M[ref][s][a]))
                      for a in sets] for s in seeds])


def axis_effects(M, variant, seeds):
    """Effect per E1 axis at the moderate level, shape [S, 6] (A1 averaged over sub-axes)."""
    return np.stack([log_ratio(M, variant, seeds, sets).mean(1) for sets in E1_AXES.values()], 1)


def e1(M, variant, seeds):
    """Primary endpoint E1[s]: mean over the six axes."""
    return axis_effects(M, variant, seeds).mean(1)


def bootstrap_seed_mean(M, variant, seeds, sets, weights, B=B_DEFAULT, rng_seed=0, chunk=200):
    """Percentile 95% interval of mean_s sum_k w_k d[s,k] under resampling of test samples.

    One draw of sample indices per replicate is shared by all seeds, both variants and all
    sets: the test set is common to them and OOD sets are paired with test ID by index.
    """
    X = np.stack([np.stack([np.stack([np.asarray(M[v][s][a], dtype=float) for a in sets])
                            for s in seeds]) for v in (variant, "V0")])     # [2, S, K, n]
    n = X.shape[-1]
    w = np.asarray(weights, dtype=float)
    rng = np.random.default_rng(rng_seed)
    out = []
    for start in range(0, B, chunk):
        idx = rng.integers(0, n, size=(min(chunk, B - start), n))           # [b, n]
        med = np.median(X[..., idx], axis=-1)                              # [2, S, K, b]
        d = np.log(med[0]) - np.log(med[1])                                # [S, K, b]
        out.append(np.einsum("skb,k->sb", d, w).mean(0))
    stats = np.concatenate(out)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def wider(t_lo, t_hi, b_lo, b_hi):
    """The wider of the t interval over seeds and the bootstrap interval (§10)."""
    if (b_hi - b_lo) > (t_hi - t_lo):
        return b_lo, b_hi, "bootstrap"
    return t_lo, t_hi, "t"


# ---------------------------------------------------------------- hypotheses

def decide_h1(M, seeds, B=B_DEFAULT):
    """H1 at N = 2 000 for M1: three criteria of §10."""
    e = e1(M, "V3", seeds)
    m, t_lo, t_hi = t_interval(e)
    b_lo, b_hi = bootstrap_seed_mean(M, "V3", seeds, E1_SETS, E1_WEIGHTS, B)
    lo, hi, which = wider(t_lo, t_hi, b_lo, b_hi)
    crit = dict(mean_le_threshold=bool(m <= H1_THRESHOLD),
                ci_excludes_zero=bool(hi < 0 or lo > 0),
                negative_in_4_of_5=bool((e < 0).sum() >= 4))
    return dict(E1=e.tolist(), mean=m, t_interval=[t_lo, t_hi], bootstrap_interval=[b_lo, b_hi],
                governing=which, interval=[lo, hi], criteria=crit, supported=all(crit.values()))


def decide_h2(M_by_N, seeds):
    """H2: per-seed slope of E1 on log N, mean slope with t interval; positive slope = effect grows as N falls."""
    x = np.log(np.array(NS, dtype=float))
    E = np.stack([e1(M_by_N[n], "V3", seeds) for n in NS], 1)               # [S, 3]
    slopes = np.array([np.polyfit(x, E[i], 1)[0] for i in range(len(seeds))])
    m, lo, hi = t_interval(slopes)
    return dict(E1_by_N={n: E[:, j].tolist() for j, n in enumerate(NS)}, slopes=slopes.tolist(),
                mean=m, interval=[lo, hi], supported=bool(lo > 0))


def decide_h3(M_by_metric, seeds, B=B_DEFAULT):
    """H3 at N = 2 000: non-inferiority of V3 vs V0 on test ID for M1 and M2."""
    out = {}
    for metric, M in M_by_metric.items():
        d = log_ratio(M, "V3", seeds, ["test_id"])[:, 0]
        m, t_lo, t_hi = t_interval(d)
        b_lo, b_hi = bootstrap_seed_mean(M, "V3", seeds, ["test_id"], [1.0], B)
        lo, hi, which = wider(t_lo, t_hi, b_lo, b_hi)
        out[metric] = dict(d=d.tolist(), mean=m, t_interval=[t_lo, t_hi],
                           bootstrap_interval=[b_lo, b_hi], governing=which,
                           upper=hi, non_inferior=bool(hi <= H3_MARGIN))
    return dict(by_metric=out, supported=all(v["non_inferior"] for v in out.values()))


def secondary(M, seeds, variants=("V1", "V2", "V3"), alpha=0.05):
    """Per-axis effects of V1, V2, V3 vs V0 with Holm correction over 18 tests."""
    p, est = {}, {}
    for v in variants:
        eff = axis_effects(M, v, seeds)
        for j, ax in enumerate(E1_AXES):
            key = f"{v}:{ax}"
            est[key] = t_interval(eff[:, j])
            p[key] = t_pvalue(eff[:, j])
    adj = holm(p)
    return {k: dict(mean=est[k][0], interval=[est[k][1], est[k][2]], p=p[k], p_holm=adj[k],
                    significant=bool(adj[k] < alpha)) for k in p}


# ---------------------------------------------------------------- loading

def load(root, metric, n, seeds, variants, sets, tag=""):
    """Per-sample metric arrays from results/metrics/<run>/<set>.npz."""
    M = {v: {s: {} for s in seeds} for v in variants}
    for v in variants:
        for s in seeds:
            run = f"{tag + '_' if tag else ''}{v}_N{n}_s{s}"
            for a in sets:
                M[v][s][a] = np.load(Path(root) / run / (a.replace("/", "__") + ".npz"))[metric]
    return M
