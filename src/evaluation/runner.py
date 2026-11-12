"""Per-sample evaluation of trained checkpoints (PROTOCOL §6, §8).

Without `final`, only the validation set is evaluated. Test ID, OOD and diagnostic sets
are touched only in the final evaluation, which leaves an audit marker on first use.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.data.store import NpyDataset, verify_dataset
from src.evaluation.metrics import sample_metrics
from src.models.fno_reconstructor import FNOLensingReconstructor
from src.training.trainer import _git_state

KEYS = ("observed", "clean_lensed", "source", "psi", "noise_map", "params")


def datasets(root, final):
    """Name -> directory of the sets to evaluate. Non-final mode returns val only."""
    root = Path(root)
    if not final:
        return {"val": root / "val"}
    out = {"test_id": root / "test_id"}
    for sub in ("ood", "diag"):
        if (root / sub).exists():
            for d in sorted(p for p in (root / sub).iterdir() if p.is_dir()):
                out[f"{sub}/{d.name}"] = d
    return out


def load_model(run_dir, device):
    model = FNOLensingReconstructor(modes=32, width=64, pad_ratio=0.25).to(device)
    state = torch.load(Path(run_dir) / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.eval()


@torch.inference_mode()
def evaluate(model, data_dir, cfg, device, batch=50):
    """Per-sample metrics for one dataset: dict name -> np.ndarray of length n."""
    ds = NpyDataset(data_dir)
    acc = {}
    for i in range(0, len(ds), batch):
        b = {k: torch.from_numpy(np.array(ds.a[k][i:i + batch])).to(device) for k in KEYS}
        psi, src = model(b["observed"])
        for k, v in sample_metrics(psi, src, b, cfg.extent, cfg.psf_fwhm, cfg.gain).items():
            acc.setdefault(k, []).append(v.numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


def run_all(cfg, runs_root="runs", out_root="results/metrics", final=False, only=None, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    commit, dirty = _git_state()
    if final:
        marker = out_root / "FINAL_EVALUATION.json"
        if not marker.exists():
            marker.write_text(json.dumps(dict(started=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                                              commit=commit, git_dirty=dirty), indent=1))
    runs = sorted(r for r in Path(runs_root).iterdir()
                  if (r / "best.pt").exists() and (r / "meta.json").exists()
                  and (only is None or only in r.name))
    verified = {}
    for r in runs:
        meta = json.loads((r / "meta.json").read_text())
        dst_dir = out_root / r.name
        model = None
        for name, d in datasets(meta["train"]["data_root"], final).items():
            dst = dst_dir / (name.replace("/", "__") + ".npz")
            if dst.exists():
                continue
            if d not in verified:
                verified[d] = verify_dataset(d)
                if not verified[d]:
                    raise RuntimeError(f"checksum mismatch in {d}")
            model = model or load_model(r, device)
            dst_dir.mkdir(parents=True, exist_ok=True)
            np.savez(dst, **evaluate(model, d, cfg, device))
            print(f"{r.name} {name}: done", flush=True)
        if model is not None:
            (dst_dir / "eval_meta.json").write_text(json.dumps(
                dict(run_meta=meta, eval_commit=commit, eval_git_dirty=dirty, final=final,
                     torch=torch.__version__, device=str(device)), indent=1))
