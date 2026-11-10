from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.physics.deflection import alpha_rel_l2, rel_l2
from src.physics.noise import estimate_sigma
from src.physics.poisson import PoissonSolver2D
from src.physics.ray_tracer import GravitationalRayTracer


@dataclass(frozen=True)
class LossConfig:
    lambda_ray: float = 0.0
    lambda_pos: float = 0.0
    use_sup: bool = True
    chi2_mode: str = "huber"
    sigma_mode: str = "est"
    huber_c: float = 10.0
    clamp_max: float = 100.0


def sup_per_sample(pred_psi, pred_src, true_psi, true_src, d):
    return rel_l2(pred_src, true_src) + 2.0 * alpha_rel_l2(pred_psi, true_psi, d)


def pseudo_huber(pull, c):
    return 2.0 * c * c * (torch.sqrt(1.0 + (pull / c) ** 2) - 1.0)


class LensingLoss(nn.Module):
    def __init__(self, cfg: LossConfig, extent=3.0, psf_fwhm=0.10, gain=500.0):
        super().__init__()
        self.cfg, self.extent, self.gain = cfg, extent, gain
        self.tracer = GravitationalRayTracer(extent=extent, psf_fwhm=psf_fwhm)
        self.poisson = PoissonSolver2D(extent=extent)

    def _d(self, psi):
        return 2.0 * self.extent / (psi.shape[-1] - 1)

    def ray(self, pred_psi, pred_src, obs, noise_map):
        sigma = estimate_sigma(obs, self.gain) if self.cfg.sigma_mode == "est" else noise_map
        sigma = sigma.detach().clamp_min(1e-4)
        pull = (self.tracer(pred_src, pred_psi) - obs) / sigma
        if self.cfg.chi2_mode == "huber":
            return pseudo_huber(pull, self.cfg.huber_c).mean()
        return pull.pow(2).clamp(max=self.cfg.clamp_max).mean()

    def pos(self, pred_psi):
        kappa = self.poisson(pred_psi)
        return F.relu(-kappa[..., 2:-2, 2:-2]).pow(2).mean()

    def forward(self, pred_psi, pred_src, batch, warm: float = 1.0):
        c, terms, total = self.cfg, {}, pred_psi.new_zeros(())
        if c.use_sup:
            terms["sup"] = sup_per_sample(pred_psi, pred_src, batch["psi"], batch["source"],
                                          self._d(pred_psi)).mean()
            total = total + terms["sup"]
        if c.lambda_ray * warm > 0:
            terms["ray"] = self.ray(pred_psi, pred_src, batch["observed"], batch["noise_map"])
            total = total + c.lambda_ray * warm * terms["ray"]
        if c.lambda_pos * warm > 0:
            terms["pos"] = self.pos(pred_psi)
            total = total + c.lambda_pos * warm * terms["pos"]
        return total, {k: v.detach() for k, v in terms.items()}


VARIANTS = {"V0": (False, False), "V1": (False, True), "V2": (True, False), "V3": (True, True)}


def loss_config(variant, lambda_ray, lambda_pos, **kw):
    use_ray, use_pos = VARIANTS[variant]
    return LossConfig(lambda_ray if use_ray else 0.0, lambda_pos if use_pos else 0.0, **kw)