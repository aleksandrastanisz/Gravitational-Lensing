from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass(frozen=True)
class SimulationConfig:
    grid_size: int
    fine_grid_size: int
    extent: float
    gain: float
    psf_fwhm: float
    noise_std_eval: float
    noise_std_train_min: float
    noise_std_train_max: float
    theta_e_min: float
    theta_e_max: float
    q_lens_min: float
    q_lens_max: float
    shear_min: float
    shear_max: float
    lens_pos_bound: float
    core_radius: float
    amp_min: float
    amp_max: float
    r_eff_min: float
    r_eff_max: float
    n_sersic_min: float
    n_sersic_max: float
    src_pos_bound: float
    q_src_min: float
    q_src_max: float

    @classmethod
    def from_yaml(cls, path: str | Path = "configs/prior.yaml") -> "SimulationConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)