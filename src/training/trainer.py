import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR

from src.data.store import NpyDataset
from src.losses.lensing_loss import LensingLoss, loss_config
from src.physics.deflection import alpha_rel_l2, rel_l2
from src.models.fno_reconstructor import FNOLensingReconstructor
from src.utils.config import SimulationConfig

REF_STEPS_PER_EPOCH = 8_000 // 16
TRAIN_KEYS = ("observed", "psi", "source", "noise_map")
VAL_KEYS = ("observed", "psi", "source")


@dataclass(frozen=True)
class TrainConfig:
    variant: str
    n_train: int
    seed: int
    lr: float = 1e-3
    wd: float = 1e-4
    lambda_ray: float = 0.02
    lambda_pos: float = 15.0
    steps: int = 12_500
    batch: int = 16
    eval_every: int = 500
    warm_start: int = 3 * REF_STEPS_PER_EPOCH
    warm_len: int = 3 * REF_STEPS_PER_EPOCH
    clip: float = 1.0
    chi2_mode: str = "huber"
    sigma_mode: str = "est"
    data_root: str = "data"
    out_root: str = "runs"
    tag: str = ""
    save_ckpt: bool = True


def warm(step: int, c: TrainConfig) -> float:
    return min(max((step - c.warm_start) / c.warm_len, 0.0), 1.0)


def to_dev(batch, device):
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def load_arrays(ds, keys=None):
    return {k: torch.from_numpy(np.array(ds.a[k][:ds.n])) for k in (keys or ds.a)}


def batches(ds, batch, seed, keys=None):
    assert ds.n >= batch, f"dataset size ({ds.n}) smaller than batch ({batch})"
    g = torch.Generator().manual_seed(seed)
    data = load_arrays(ds, keys)
    while True:
        perm = torch.randperm(ds.n, generator=g)
        for i in range(0, ds.n - batch + 1, batch):
            idx = perm[i:i + batch]
            yield {k: v[idx] for k, v in data.items()}


@torch.inference_mode()
def val_select(model, arrays, d, device, batch=50):
    """Selection criterion (§7) and its two components, all means over the validation set.

    The total is computed per sample exactly as sup_per_sample: relL2(source) + 2 relL2(alpha).
    """
    model.eval()
    n = len(next(iter(arrays.values())))
    tot, srcs, alphas = [], [], []
    for i in range(0, n, batch):
        b = to_dev({k: v[i:i + batch] for k, v in arrays.items()}, device)
        psi, src = model(b["observed"])
        s, a = rel_l2(src, b["source"]), alpha_rel_l2(psi, b["psi"], d)
        tot.append(s + 2.0 * a); srcs.append(s); alphas.append(a)
    model.train()
    return torch.cat(tot).mean().item(), torch.cat(srcs).mean().item(), torch.cat(alphas).mean().item()


def _git_state():
    try:
        h = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"],
                                             text=True).strip())
        return h, dirty
    except Exception:
        return None, None


_PROCESS_GIT = None


def process_git_state():
    """Git state read once, when the first run of this process starts (the code it executes)."""
    global _PROCESS_GIT
    if _PROCESS_GIT is None:
        _PROCESS_GIT = _git_state()
    return _PROCESS_GIT


def run_name(c: TrainConfig) -> str:
    return f"{c.tag + '_' if c.tag else ''}{c.variant}_N{c.n_train}_s{c.seed}"


def run(c: TrainConfig, device=None):
    out = Path(c.out_root) / run_name(c)
    if (out / "meta.json").exists():
        meta = json.loads((out / "meta.json").read_text())
        if meta["train"] != asdict(c):
            raise RuntimeError(f"{run_name(c)}: existing run has different config (delete {out} or change tag)")
        return meta
    out.mkdir(parents=True, exist_ok=True)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    commit, dirty = process_git_state()
    gpu = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    cfg = SimulationConfig.from_yaml()

    torch.manual_seed(c.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(c.seed)
    model = FNOLensingReconstructor(modes=32, width=64, pad_ratio=0.25).to(device)
    criterion = LensingLoss(
        loss_config(c.variant, c.lambda_ray, c.lambda_pos, chi2_mode=c.chi2_mode, sigma_mode=c.sigma_mode),
        extent=cfg.extent, psf_fwhm=cfg.psf_fwhm, gain=cfg.gain).to(device)
    opt = AdamW(model.parameters(), lr=c.lr, weight_decay=c.wd)
    sched = OneCycleLR(opt, max_lr=c.lr, total_steps=c.steps)

    data = batches(NpyDataset(f"{c.data_root}/train", n=c.n_train), c.batch, c.seed, TRAIN_KEYS)
    val_arrays = to_dev(load_arrays(NpyDataset(f"{c.data_root}/val"), VAL_KEYS), device)
    d = 2.0 * cfg.extent / (cfg.grid_size - 1)

    best, best_rec, hist, acc, n_acc = math.inf, None, [], {}, 0
    eval_sec, t0 = 0.0, time.perf_counter()
    model.train()
    for step in range(c.steps):
        b = to_dev(next(data), device)
        psi, src = model(b["observed"])
        loss, terms = criterion(psi, src, b, warm(step, c))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip)
        opt.step()
        sched.step()
        for k, v in terms.items():
            acc[k] = acc.get(k, 0) + v
        n_acc += 1

        if (step + 1) % c.eval_every == 0 or step + 1 == c.steps:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t_eval = time.perf_counter()
            vs, v_src, v_alpha = val_select(model, val_arrays, d, device)
            eval_sec += time.perf_counter() - t_eval
            if not math.isfinite(vs):
                raise RuntimeError(f"{run_name(c)}: val_select={vs} at step {step + 1}")
            sec = time.perf_counter() - t0
            rec = dict(step=step + 1, ref_epoch=(step + 1) / REF_STEPS_PER_EPOCH, val_select=vs, val_src=v_src, val_alpha=v_alpha,
                       sec=sec, sec_eval=eval_sec, lr=sched.get_last_lr()[0],
                       **{f"train_{k}": float(v) / n_acc for k, v in acc.items()})
            hist.append(rec)
            acc, n_acc = {}, 0
            print(f"{run_name(c)} step {step + 1}/{c.steps} val_select={vs:.4f} "
                  f"t={sec:.0f}s (eval {eval_sec:.0f}s)", flush=True)
            if vs < best:
                best, best_rec = vs, rec
                if c.save_ckpt:
                    torch.save(model.state_dict(), out / "best.pt")

    (out / "history.json").write_text(json.dumps(hist, indent=1))
    meta = dict(
        train=asdict(c), best=best_rec, seconds=time.perf_counter() - t0, eval_seconds=eval_sec,
        prior_sha256=hashlib.sha256(Path("configs/prior.yaml").read_bytes()).hexdigest(),
        commit=commit, git_dirty=dirty, torch=torch.__version__, device=str(device), gpu=gpu)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return meta