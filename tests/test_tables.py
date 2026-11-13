import json
import math

import numpy as np

from src.evaluation import stats as S
from src.evaluation.tables import SEEDS, VARIANTS, build, run_name, set_file

KEYS = ("m1", "m1b", "m2", "m3", "m4", "m4b_centroid_mas", "m4b_flux", "m5_arc",
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


def test_missing_runs_are_reported_not_crashing(tmp_path):
    _fake_results(tmp_path / "metrics")
    set_file(tmp_path / "metrics", run_name("V3", 2_000, 44), "ood/A3_L2").unlink()
    build(tmp_path / "metrics", tmp_path / "tables", tmp_path / "missing.json", B=200)
    st = json.loads((tmp_path / "tables" / "stats.json").read_text())
    assert "missing" in st["H1"] and "missing" not in st["H3"]
