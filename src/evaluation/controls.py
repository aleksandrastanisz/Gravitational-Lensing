"""Descriptive analysis of the control experiments (PROTOCOL §9, last paragraphs).

For a control identified by its run tag, two quantities are reported per seed, with the mean over
seeds and the 95% t interval, for the median M1:

- relative to the E-main counterpart: ln m[control, v, s, a] - ln m[E-main, v, s, a] for the same
  variant, N and seed, on test ID and averaged over the E1 axes with the E1 weights;
- when the control has both V0 and V3 (S2, S4): E1 of V3 against V0 within the control, next to
  the E-main E1 of the same seeds, and their difference.

No hypothesis tests are made.
"""
import json
import math
from pathlib import Path

import numpy as np

from src.evaluation import stats as S
from src.evaluation.tables import SEEDS, md_table, run_name, set_file


def _log_median(root, run, a, metric):
    return math.log(float(np.median(np.load(set_file(root, run, a))[metric])))


def _interval(x):
    m, lo, hi = S.t_interval(x)
    return dict(per_seed=[float(v) for v in x], mean=float(m), interval=[float(lo), float(hi)])


def _weighted(root, run_a, run_b, metric):
    """Sum over E1 sets of w * (ln median(run_a) - ln median(run_b))."""
    return float(sum(w * (_log_median(root, run_a, a, metric) - _log_median(root, run_b, a, metric))
                     for a, w in zip(S.E1_SETS, S.E1_WEIGHTS)))


def control_summary(root, tag, n=S.REF_N, seeds=SEEDS, metric="m1"):
    """Summary dict for one control tag; variants without all seeds are listed as missing."""
    out = dict(tag=tag, n=n, metric=metric, vs_emain={}, missing=[])
    have = []
    for v in ("V0", "V1", "V2", "V3", "V4", "R"):
        runs = [run_name(v, n, s, tag) for s in seeds]
        present = [(Path(root) / r).exists() for r in runs]
        if not any(present):
            continue
        if not all(present):
            out["missing"].append(v)
            continue
        have.append(v)
        if not all((Path(root) / run_name(v, n, s)).exists() for s in seeds):
            continue                                    # no E-main counterpart (V4, R)
        out["vs_emain"][v] = dict(
            test_id=_interval([_log_median(root, run_name(v, n, s, tag), "test_id", metric)
                               - _log_median(root, run_name(v, n, s), "test_id", metric) for s in seeds]),
            e1_axes=_interval([_weighted(root, run_name(v, n, s, tag), run_name(v, n, s), metric)
                               for s in seeds]))
    if "V0" in have and "V3" in have:
        within = np.array([_weighted(root, run_name("V3", n, s, tag), run_name("V0", n, s, tag), metric)
                           for s in seeds])
        main = np.array([_weighted(root, run_name("V3", n, s), run_name("V0", n, s), metric) for s in seeds])
        out["e1"] = dict(control=_interval(within), emain=_interval(main), difference=_interval(within - main))
    return out


def _cell(r):
    return f"{r['mean']:+.4f} [{r['interval'][0]:+.4f}, {r['interval'][1]:+.4f}]"


def build(metrics_root="results/metrics", out_dir="results/tables", tags=("S4", "S2", "S1", "S3", "X2")):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    res = {t: control_summary(metrics_root, t) for t in tags}
    (out / "controls.json").write_text(json.dumps(res, indent=1))
    md = ["# Control experiments (descriptive, PROTOCOL §9)", "",
          "Log-ratios of the median M1, mean over seeds with the 95% t interval; negative = lower error.", "",
          "## Control run against its E-main counterpart (same variant, N and seed)", ""]
    rows = [[t, v, _cell(r["test_id"]), _cell(r["e1_axes"])]
            for t, c in res.items() for v, r in c["vs_emain"].items()]
    md += [md_table(["control", "variant", "test ID", "mean over E1 axes"], rows) if rows else "No control runs found.", ""]
    rows = [[t, _cell(c["e1"]["control"]), _cell(c["e1"]["emain"]), _cell(c["e1"]["difference"])]
            for t, c in res.items() if "e1" in c]
    if rows:
        md += ["## E1 (V3 against V0) within the control and in E-main", "",
               md_table(["control", "E1 in control", "E1 in E-main", "difference"], rows), ""]
    miss = [f"{t}: {', '.join(c['missing'])}" for t, c in res.items() if c["missing"]]
    if miss:
        md += ["Incomplete (not all seeds evaluated): " + "; ".join(miss), ""]
    (out / "controls.md").write_text("\n".join(md), encoding="utf-8")
    return res
