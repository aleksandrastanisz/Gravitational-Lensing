import hashlib, json
from pathlib import Path
import numpy as np
import torch
from numpy.lib.format import open_memmap
from torch.utils.data import Dataset
from src.data.generator import OUT_ORDER

KEYS = ["observed", "clean_lensed", "source", "psi", "noise_map"]


def _sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def write_dataset(out_dir, seeds, cfg, gen_fn, config_path="configs/prior.yaml"):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds); n, g = len(seeds), cfg.grid_size
    arrs = {k: open_memmap(out / f"{k}.npy", "w+", np.float32, (n, 1, g, g)) for k in KEYS}
    params = open_memmap(out / "params.npy", "w+", np.float64, (n, len(OUT_ORDER)))
    for i, s in enumerate(seeds):
        d = gen_fn(s, cfg)
        for k in KEYS:
            arrs[k][i] = d[k]
        params[i] = d["params"]
    for a in [*arrs.values(), params]:
        a.flush()
    meta = dict(seed_first=seeds[0], seed_last=seeds[-1], n=n, torch=torch.__version__,
                config_sha256=hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                files={p.name: _sha256(p) for p in sorted(out.glob("*.npy"))})
    (out / "meta.json").write_text(json.dumps(meta, indent=2))


def verify_dataset(d):
    meta = json.loads((Path(d) / "meta.json").read_text())
    return all(_sha256(Path(d) / f) == h for f, h in meta["files"].items())


class NpyDataset(Dataset):
    def __init__(self, d, n=None):
        self.a = {k: np.load(Path(d) / f"{k}.npy", mmap_mode="r") for k in [*KEYS, "params"]}
        self.n = n or len(self.a["observed"])

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        out = {k: torch.from_numpy(np.array(self.a[k][i])) for k in KEYS}
        out["params"] = torch.from_numpy(np.array(self.a["params"][i]))
        return out