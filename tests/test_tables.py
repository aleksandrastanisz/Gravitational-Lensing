import json
import math

import numpy as np

from src.evaluation import stats as S
from src.evaluation.tables import SEEDS, VARIANTS, build, run_name, set_file

KEYS = ("m1", "m1b", "m1c", "mst_lambda", "m2", "m3", "m4", "m4b_centroid_mas", "m4b_flux", "m5_arc",
        "m5_global", "m5_arc_oracle", "m5_global_oracle", "m6")
SETS = ["test_id"] + S.E1_SETS + ["ood/A2_L1", "ood/A2_L3", "diag/Kneg_L1"]


def _fake_results(root, factor_v3=0.9):
    rng = np.random.default_rng(0)
    for n in S.NS:
        for s in SEEDS:
            base = {a: np.exp(rng.normal(-1.0, 0.5, 200)) for a in SETS}
            for v in VARIANTS:
                f = factor_v3 * math.exp(0.01 * rng.normal()) if v == "V3" else 1.0
                for a in SETS:
                    p = set_file(root, run_name(v, n, s), a)
                    p.parent.mkdir(parents=True, exist_ok=True)
                    vals = {k: base[a] * f for k in KEYS}
                    vals["m5_arc"] = vals["m5_arc"].copy()
                    vals["m5_arc"][0] = np.nan
                    vals["m4b_flux"] = rng.normal(0, 0.05, 200)
                    np.savez(p, **vals)


def test_build_writes_tables_and_decides_h1(tmp_path):
    _fake_results(tmp_path / "metrics")
    build(tmp_path / "metrics", tmp_path / "tables", tmp_path / "missing.json", B=200)
    st = json.loads((tmp_path / "tables" / "stats.json").read_text())
    assert st["H1"]["supported"] and abs(st["H1"]["mean"] - math.log(0.9)) < 0.02
    assert st["H3"]["supported"]
    assert len(st["secondary"]) == 18
    rep = (tmp_path / "tables" / "report.md").read_text(encoding="utf-8")
    assert "## M1, N = 2000" in rep and "Kneg" in rep and "—" not in rep.split("## Generalization")[0]
    assert (tmp_path / "tables" / "dose_response.csv").exists()
    assert (tmp_path / "tables" / "h1_e1_per_seed.png").exists()
    assert (tmp_path / "tables" / "h2_e1_vs_n.png").exists()


def test_missing_runs_are_reported_not_crashing(tmp_path):
    _fake_results(tmp_path / "metrics")
    set_file(tmp_path / "metrics", run_name("V3", 2_000, 44), "ood/A3_L2").unlink()
    build(tmp_path / "metrics", tmp_path / "tables", tmp_path / "missing.json", B=200)
    st = json.loads((tmp_path / "tables" / "stats.json").read_text())
    assert "missing" in st["H1"] and "missing" not in st["H3"]


def test_controls_compare_with_emain_counterparts(tmp_path):
    from src.evaluation.controls import build as build_controls

    root = tmp_path / "metrics"
    _fake_results(root, factor_v3=0.9)
    rng = np.random.default_rng(1)
    for s in SEEDS:                                   # control S4: V0 unchanged, V3 a further 20% lower
        for v, f in (("V0", 1.0), ("V3", 0.8)):
            for a in SETS:
                src = np.load(set_file(root, run_name(v, 2_000, s), a))
                p = set_file(root, run_name(v, 2_000, s, "S4"), a)
                p.parent.mkdir(parents=True, exist_ok=True)
                np.savez(p, **{k: src[k] * f for k in src.files})
    for a in SETS:                                    # control S1: only one seed present
        p = set_file(root, run_name("V3", 2_000, 42, "S1"), a)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, m1=rng.random(200))
    res = build_controls(root, tmp_path / "tables", tags=("S4", "S1", "S3"))
    s4 = res["S4"]
    assert abs(s4["vs_emain"]["V0"]["e1_axes"]["mean"]) < 1e-12
    assert abs(s4["vs_emain"]["V3"]["e1_axes"]["mean"] - math.log(0.8)) < 1e-9
    assert abs(s4["vs_emain"]["V3"]["test_id"]["mean"] - math.log(0.8)) < 1e-9
    assert abs(s4["e1"]["difference"]["mean"] - math.log(0.8)) < 1e-9
    assert abs(s4["e1"]["emain"]["mean"] - math.log(0.9)) < 0.02
    assert res["S1"]["missing"] == ["V3"] and res["S1"]["vs_emain"] == {}
    assert res["S3"]["vs_emain"] == {} and res["S3"]["missing"] == []
    text = (tmp_path / "tables" / "controls.md").read_text(encoding="utf-8")
    assert "| S4 | V3 |" in text and "Incomplete" in text and "S1: V3" in text


def test_variants_without_counterpart_are_compared_with_emain_v0_and_v3(tmp_path):
    from src.evaluation.controls import build as build_controls

    root = tmp_path / "metrics"
    _fake_results(root, factor_v3=0.9)
    for s in SEEDS:                                   # control X4: R is 5% better than V3 of the same seed
        for a in SETS:
            src = np.load(set_file(root, run_name("V3", 2_000, s), a))
            p = set_file(root, run_name("R", 2_000, s, "X4"), a)
            p.parent.mkdir(parents=True, exist_ok=True)
            np.savez(p, **{k: src[k] * 0.95 for k in src.files})
    res = build_controls(root, tmp_path / "tables", tags=("X4",))
    ref = res["X4"]["vs_reference"]["R"]
    assert abs(ref["V3"]["e1_axes"]["mean"] - math.log(0.95)) < 1e-9
    e1_v3 = S.e1(S.load(root, "m1", 2_000, SEEDS, ("V0", "V3"), S.E1_SETS), "V3", SEEDS).mean()
    assert abs(ref["V0"]["e1_axes"]["mean"] - (math.log(0.95) + e1_v3)) < 1e-9
    assert res["X4"]["vs_emain"] == {} and "e1" not in res["X4"]
    text = (tmp_path / "tables" / "controls.md").read_text(encoding="utf-8")
    assert "| X4 | R | V3 |" in text and "| X4 | R | V0 |" in text
