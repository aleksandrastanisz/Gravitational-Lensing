import csv
import json
import math
from pathlib import Path

import numpy as np

from src.evaluation import stats as S

VARIANTS = ("V0", "V1", "V2", "V3")
SEEDS = tuple(range(42, 47))
NAN_METRICS = {"m5_arc", "m5_arc_oracle", "m4b_centroid_mas"}
SHORT = {"test_id": "ID", "ood/A1_theta_e_L2": "A1 θ_E", "ood/A1_q_L2": "A1 q",
         "ood/A1_shear_L2": "A1 γ", "ood/A1_src_pos_L2": "A1 src", "ood/A2_L2": "A2",
         "ood/A3_L2": "A3", "ood/A4_L1": "A4", "ood/A5_L2": "A5", "ood/A6_L1": "A6"}
MAIN_SETS = ["test_id"] + S.E1_SETS


def run_name(v, n, s, tag=""):
    return f"{tag + '_' if tag else ''}{v}_N{n}_s{s}"


def set_file(root, run, a):
    return Path(root) / run / (a.replace("/", "__") + ".npz")


def _values(path, metric):
    if metric == "m4b_flux_abs":
        return np.abs(np.load(path)["m4b_flux"])
    return np.load(path)[metric]


def _median(x, metric):
    return float(np.nanmedian(x) if metric in NAN_METRICS else np.median(x))


def seed_medians(root, metric, v, n, a, tag=""):
    out = []
    for s in SEEDS:
        p = set_file(root, run_name(v, n, s, tag), a)
        out.append(_median(_values(p, metric), metric) if p.exists() else math.nan)
    return np.array(out)


def summarize(x):
    x = np.asarray(x, dtype=float)
    if np.isnan(x).any():
        return math.nan, math.nan
    m, _, hi = S.t_interval(x)
    return m, hi - m


def fmt(m, hw):
    return "—" if math.isnan(m) else f"{m:.4g} ± {hw:.2g}"


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    return "\n".join(lines + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows])


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def discover_sets(root):
    return sorted({p.stem.replace("__", "/") for p in Path(root).glob("*/*.npz")})


def hypotheses(root, B):
    out = {}
    try:
        M = S.load(root, "m1", S.REF_N, SEEDS, VARIANTS, MAIN_SETS)
        out["H1"] = S.decide_h1(M, SEEDS, B)
        out["secondary"] = S.secondary(M, SEEDS)
    except FileNotFoundError as e:
        out["H1"] = out["secondary"] = dict(missing=str(e))
    try:
        M_by_N = {n: S.load(root, "m1", n, SEEDS, ("V0", "V3"), S.E1_SETS) for n in S.NS}
        out["H2"] = S.decide_h2(M_by_N, SEEDS)
    except FileNotFoundError as e:
        out["H2"] = dict(missing=str(e))
    try:
        by_metric = {m: S.load(root, m, S.REF_N, SEEDS, ("V0", "V3"), ["test_id"]) for m in ("m1", "m2")}
        out["H3"] = S.decide_h3(by_metric, SEEDS, B)
    except FileNotFoundError as e:
        out["H3"] = dict(missing=str(e))
    return out


def metric_table(root, metric, n, sets, variants=VARIANTS):
    header = ["variant"] + [SHORT.get(a, a) for a in sets]
    rows, raw = [], []
    for v in variants:
        cells = [summarize(seed_medians(root, metric, v, n, a)) for a in sets]
        rows.append([v] + [fmt(*c) for c in cells])
        raw.append([v] + [x for c in cells for x in c])
    raw_header = ["variant"] + [f"{a}:{k}" for a in sets for k in ("mean", "halfwidth")]
    return header, rows, raw_header, raw


def ratio_table(root, n, sets):
    header = ["variant"] + [SHORT.get(a, a) for a in sets]
    rows = []
    for v in VARIANTS:
        ref = seed_medians(root, "m1", v, n, "test_id")
        rows.append([v] + [fmt(*summarize(seed_medians(root, "m1", v, n, a) / ref)) for a in sets])
    return header, rows


def dose_response(root, sets_found):
    axes = {}
    for a in sets_found:
        if a.startswith(("ood/", "diag/")) and "_L" in a:
            ax, lvl = a.rsplit("_L", 1)
            axes.setdefault(ax, []).append((int(lvl), a))
    rows = []
    for ax, lv in sorted(axes.items()):
        for v in VARIANTS:
            for n in S.NS:
                for lvl, a in [(0, "test_id")] + sorted(lv):
                    m, hw = summarize(seed_medians(root, "m1", v, n, a))
                    rows.append([ax, lvl, v, n, m, hw])
    return rows


def tuning_table(path):
    rows = json.loads(Path(path).read_text())
    out = {}
    for v in VARIANTS:
        r = [x for x in rows if x["variant"] == v]
        if not r:
            continue
        keys = [k for k in ("lr", "wd", "lambda_ray", "lambda_pos") if k in r[0]]
        best = min(r, key=lambda x: x["val_select"])
        out[v] = (["lr", "wd", "λ_ray", "λ_pos", "val_select", "best step", ""],
                  [[x.get("lr", ""), x.get("wd", ""), x.get("lambda_ray", ""), x.get("lambda_pos", ""),
                    f"{x['val_select']:.4f}", x["step"], "selected" if x is best else ""] for x in r])
    return out


def figures(rows, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "matplotlib not installed: figures skipped (data in dose_response.csv)"
    for ax in sorted({r[0] for r in rows}):
        fig, panel = plt.subplots(figsize=(4, 3))
        for v in VARIANTS:
            pts = [(r[1], r[4], r[5]) for r in rows if r[0] == ax and r[2] == v and r[3] == S.REF_N]
            if pts and not all(math.isnan(p[1]) for p in pts):
                x, y, e = zip(*pts)
                panel.errorbar(x, y, yerr=e, marker="o", capsize=3, label=v)
        panel.set_xlabel("level (0 = ID)")
        panel.set_ylabel("median M1")
        panel.set_title(f"{ax}, N = {S.REF_N}")
        panel.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(Path(out_dir) / f"dose_{ax.replace('/', '_')}.png", dpi=200)
        plt.close(fig)
    return "figures written"


def build(metrics_root="results/metrics", out_dir="results/tables", tuning="results/tuning.json", B=S.B_DEFAULT):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# Results", "", f"Mean over seeds {SEEDS[0]}–{SEEDS[-1]} of the per-seed median over samples, ± half-width of the 95% t interval.", ""]

    hyp = hypotheses(metrics_root, B)
    (out / "stats.json").write_text(json.dumps(hyp, indent=1, default=float))
    md += ["## Hypotheses (§10)", ""]
    for h in ("H1", "H2", "H3"):
        r = hyp[h]
        md.append(f"- **{h}:** " + ("missing data" if "missing" in r else ("supported" if r["supported"] else "not supported")))
    md.append("")

    found = discover_sets(metrics_root)
    for n in S.NS:
        h, rows, rh, raw = metric_table(metrics_root, "m1", n, MAIN_SETS)
        md += [f"## M1, N = {n}", "", md_table(h, rows), ""]
        write_csv(out / f"m1_N{n}.csv", rh, raw)

    h, rows = ratio_table(metrics_root, S.REF_N, S.E1_SETS)
    md += [f"## Generalization ratio R = M1[axis] / M1[ID], N = {S.REF_N}", "", md_table(h, rows), ""]

    for metric, title in [("m1b", "M1b"), ("m2", "M2"), ("m3", "M3"), ("m4", "M4"),
                          ("m4b_centroid_mas", "M4b centroid [mas]"), ("m4b_flux", "M4b flux (bias)"),
                          ("m4b_flux_abs", "M4b flux (|error|)")]:
        h, rows, rh, raw = metric_table(metrics_root, metric, S.REF_N, MAIN_SETS)
        md += [f"## {title}, N = {S.REF_N}", "", md_table(h, rows), ""]
        write_csv(out / f"{metric}_N{S.REF_N}.csv", rh, raw)

    md += ["## Physical consistency (favours variants that optimize it)", ""]
    for metric, title in [("m5_arc", "M5 χ² on the arc"), ("m5_global", "M5 χ² global"),
                          ("m5_arc_oracle", "M5 χ² on the arc, oracle σ"), ("m6", "M6 fraction κ < 0")]:
        h, rows, rh, raw = metric_table(metrics_root, metric, S.REF_N, MAIN_SETS)
        md += [f"### {title}, N = {S.REF_N}", "", md_table(h, rows), ""]
        write_csv(out / f"{metric}_N{S.REF_N}.csv", rh, raw)

    diag = [a for a in found if a.startswith("diag/")]
    if diag:
        md += ["## Diagnostic axis Kneg (outside E1)", ""]
        for metric, title in [("m1", "M1"), ("m6", "M6")]:
            h, rows, _, _ = metric_table(metrics_root, metric, S.REF_N, ["test_id"] + diag)
            md += [f"### {title}, N = {S.REF_N}", "", md_table(h, rows), ""]

    if "secondary" in hyp and "missing" not in hyp["secondary"]:
        rows = [[k, f"{v['mean']:.4f}", f"[{v['interval'][0]:.4f}, {v['interval'][1]:.4f}]",
                 f"{v['p']:.4f}", f"{v['p_holm']:.4f}", "yes" if v["significant"] else ""]
                for k, v in hyp["secondary"].items()]
        md += ["## Secondary tests (Holm over 18), N = 2 000", "",
               md_table(["comparison", "mean d", "95% t interval", "p", "p (Holm)", "significant"], rows), ""]

    dr = dose_response(metrics_root, found)
    write_csv(out / "dose_response.csv", ["axis", "level", "variant", "N", "mean_M1", "halfwidth"], dr)
    note = figures(dr, out)

    if Path(tuning).exists():
        md += ["## Tuning (§7.1, validation set, N = 2 000, seed 100)", ""]
        for v, (h, rows) in tuning_table(tuning).items():
            md += [f"### {v}", "", md_table(h, rows), ""]

    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    return note
