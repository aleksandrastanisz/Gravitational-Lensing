from dataclasses import dataclass


@dataclass(frozen=True)
class SimulationConfig:
    grid_size: int = 128
    fine_grid_size: int = 255
    extent: float = 3.0
    gain: float = 500.0
    psf_fwhm: float = 0.1
    noise_std_eval: float = 0.02
    noise_std_train_min: float = 0.01
    noise_std_train_max: float = 0.03
    theta_e_min: float = 0.8
    theta_e_max: float = 1.4
    q_lens_min: float = 0.71
    q_lens_max: float = 0.95
    shear_min: float = -0.08
    shear_max: float = 0.08
    lens_pos_bound: float = 0.1
    core_radius: float = 0.05
    amp_min: float = 0.8
    amp_max: float = 1.5
    r_eff_min: float = 0.15
    r_eff_max: float = 0.35
    n_sersic_min: float = 0.8
    n_sersic_max: float = 2.5
    src_pos_bound: float = 0.4
    q_src_min: float = 0.5
    q_src_max: float = 0.9