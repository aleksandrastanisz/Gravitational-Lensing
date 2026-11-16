import numpy as np
import torch

from src.data.generator import Shift, generate
from src.physics.operators import a_inverse_crime
from src.physics.ray_tracer import GravitationalRayTracer
from src.utils.config import SimulationConfig

CFG = SimulationConfig.from_yaml()
SHIFTS = (None, Shift("A2", 2), Shift("A3", 2), Shift("A4", 1), Shift("A5", 3), Shift("A6", 2))


def test_s4_differs_from_main_data_only_in_the_rendered_image():
    for sh in SHIFTS:
        a = generate(200_011, CFG, sh)
        b = generate(200_011, CFG, sh, operator=a_inverse_crime)
        for k in ("psi", "source", "params"):
            assert np.array_equal(a[k], b[k]), (sh, k)
        assert not np.array_equal(a["clean_lensed"], b["clean_lensed"])
        rel = np.linalg.norm(a["clean_lensed"] - b["clean_lensed"]) / np.linalg.norm(a["clean_lensed"])
        assert rel < 0.3, (sh, rel)


def test_s4_clean_image_is_exactly_a_model_of_the_stored_truth():
    for sh in SHIFTS:
        d = generate(200_011, CFG, sh, operator=a_inverse_crime)
        psf = 0.15 if sh is not None and sh.axis == "A6" else CFG.psf_fwhm
        tracer = GravitationalRayTracer(extent=CFG.extent, psf_fwhm=psf)
        img = tracer(torch.from_numpy(d["source"])[None], torch.from_numpy(d["psi"])[None])[0]
        np.testing.assert_allclose(d["clean_lensed"], img.numpy(), atol=1e-6)


def test_s4_uses_the_same_noise_draw_and_noise_model():
    a = generate(200_011, CFG)
    b = generate(200_011, CFG, operator=a_inverse_crime)
    za = (a["observed"] - a["clean_lensed"]) / a["noise_map"]
    zb = (b["observed"] - b["clean_lensed"]) / b["noise_map"]
    np.testing.assert_allclose(za, zb, atol=2e-3)
    sigma0 = b["params"][14]
    np.testing.assert_allclose(b["noise_map"] ** 2, sigma0 ** 2 + np.clip(b["clean_lensed"], 0, None) / CFG.gain,
                               rtol=1e-4)
