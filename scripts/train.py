import argparse
from src.training.trainer import TrainConfig, run

p = argparse.ArgumentParser()
p.add_argument("--variant", required=True, choices=["V0", "V1", "V2", "V3"])
p.add_argument("--n", type=int, required=True)
p.add_argument("--seed", type=int, required=True)
p.add_argument("--steps", type=int, default=12_500)
p.add_argument("--eval_every", type=int, default=500)
p.add_argument("--lr", type=float, default=1e-3)
p.add_argument("--wd", type=float, default=1e-4)
p.add_argument("--lambda_ray", type=float, default=0.02)
p.add_argument("--lambda_pos", type=float, default=15.0)
p.add_argument("--tag", default="")
p.add_argument("--warm_start", type=int, default=1500)
p.add_argument("--warm_len", type=int, default=1500)
p.add_argument("--chi2_mode", default="huber", choices=["huber", "clamp"])
p.add_argument("--sigma_mode", default="est", choices=["est", "oracle"])
p.add_argument("--data_root", default="data")
a = p.parse_args()

meta = run(TrainConfig(a.variant, a.n, a.seed, a.lr, a.wd, a.lambda_ray, a.lambda_pos,
                       steps=a.steps, eval_every=a.eval_every,
                       warm_start=a.warm_start, warm_len=a.warm_len, tag=a.tag,
                       chi2_mode=a.chi2_mode, sigma_mode=a.sigma_mode, data_root=a.data_root))
print(meta["best"], f"{meta['seconds']:.0f}s")