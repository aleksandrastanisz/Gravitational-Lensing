import json

import numpy as np
import pytest

from src.evaluation.figures import check_access, pick_indices, plot_curves, run_dir


def test_pick_indices_follows_quantiles():
    m1 = np.array([0.5, 0.1, 0.9, 0.3, 0.7])
    assert pick_indices(m1, (0.0, 0.5, 1.0)) == [1, 0, 2]
    assert pick_indices(np.arange(101.0)[::-1]) == [90, 50, 10]


def test_test_sets_are_locked_until_final_evaluation(tmp_path):
    check_access("val", False, tmp_path)
    with pytest.raises(PermissionError):
        check_access("test_id", False, tmp_path)
    with pytest.raises(PermissionError):
        check_access("ood/A3_L2", True, tmp_path)
    (tmp_path / "FINAL_EVALUATION.json").write_text("{}")
    check_access("ood/A3_L2", True, tmp_path)


def test_curves_are_drawn_from_histories(tmp_path):
    pytest.importorskip("matplotlib")
    for v in ("V0", "V3"):
        for s in (42, 43):
            d = run_dir(tmp_path / "runs", v, 2_000, s)
            d.mkdir(parents=True)
            hist = [dict(step=500 * (i + 1), val_select=1.0 / (i + 1), val_src=0.6 / (i + 1),
                         val_alpha=0.2 / (i + 1)) for i in range(5)]
            (d / "history.json").write_text(json.dumps(hist))
    written = plot_curves(tmp_path / "runs", tmp_path / "fig")
    assert written == ["curves_val_select.png", "curves_val_src.png", "curves_val_alpha.png"]
    assert all((tmp_path / "fig" / w).stat().st_size > 1000 for w in written)
