# PROTOCOL: Effect of physical constraints on generalization in gravitational-lensing reconstruction

**Version:** 1.0 (Frozen), 2026-11-04

**Status:** frozen before the first training run. Any subsequent modification requires an entry in §13 (date, reason, impact)

**Scope:** synthetic data only. Conclusions do not apply to real data (§12).

---

## 1. Research question

> Does adding physical constraints to the loss of a network that reconstructs the lens potential and the source improve reconstruction quality **outside the training distribution**, and does the effect grow when little training data is available?

We do not ask about speed or about comparison with parametric fitting. We study only the effect of the constraints, with architecture, data and training budget held fixed.

## 2. Definitions

**Physical constraints** (loss terms added to supervision):

| Symbol | Definition | Notes |
|---|---|---|
| `L_sup` | `relL2(source) + 2·relL2(α)`, where `α = ∇ψ` (central differences) | ψ is supervised only through α, so the constant in ψ is free |
| `L_ray` | mean robust pull: `2c²·(√(1 + (pull/c)²) − 1)`, where `pull = (I_recon − I_obs)/σ̂`, `c = 10` | σ̂ estimated from observation (`noise.py`), not clean image; asymptotic `pull²` for small values and `2c·\|pull\|` for outliers |
| `L_pos` | mean `ReLU(−κ)²`, `κ = ½∇²ψ`, grid interior excluding a 2-pixel border | **κ ≥ 0 holds for the entire data family (including all OOD axes), so the constraint is correct but consistent with the generator** (§11) |

**Variants** (one architecture, one dataset, one procedure):

| ID | Loss | Description |
|---|---|---|
| V0 | `L_sup` | supervised baseline |
| V1 | `L_sup + λ_pos·L_pos` | positivity only |
| V2 | `L_sup + λ_ray·L_ray` | ray-tracing consistency only |
| V3 | `L_sup + λ_ray·L_ray + λ_pos·L_pos` | full |

**Exploratory (not part of the hypotheses, §4):** V4 = `L_ray + L_pos` only, no labels.

**Generalization axis** is any change of the test data distribution relative to the training data (§6).

## 3. What is identifiable (degeneracies)

Before interpreting metrics we assume the following remain free:

1. **Constant in ψ.** Invisible in α, κ and the image. We remove it in the ψ metrics.
2. **Linear term of ψ** (constant deflection offset). Degenerate with a source translation: the source position error is interpreted together with the error of the constant component of α (metric M1b).
3. **Mass-sheet** (`α → λα + (1−λ)θ` with source rescaling). The training family does not contain it, so the network learns the generator's convention. We do not remove it in the metrics, but we record it in the methods as a limitation.
4. **Potential–source degeneracy** at low S/N. For this reason we also report image-plane metrics (M5) and do not assume that a good image fit implies a good ψ.

## 4. Hypotheses

Hypotheses are decided only by the rules in §10. Every result, including a null result, is reported.

- **H1 (primary).** V3 has a lower deflection error in the ring (M1) than V0, aggregated over the OOD axes (E1, §10).
- **H2 (interaction).** The size of the effect in H1 grows as the number of training samples decreases (N ∈ {500, 2000, 8000}).
- **H3 (non-inferiority).** On the in-distribution set (ID), V3 is not worse than V0 by more than 5% in M1 and M2.

Directional predictions, **exploratory** (no multiple-comparison correction, reported as observations):
- `L_ray` helps most under shifts that do not change the image–source relation (noise, part of the parametric axes) and may hurt under a mismatched operator (A6).
- `L_pos` helps under changes of profile shape and under substructure (A2, A3).

## 5. Physics and data

### 5.1 Forward operators
- **`A_truth`** (generates all observations): 255×255 grid (nodes coincide with every second node of the 128 grid), Gaussian PSF with FWHM 0.10″, subsampling `[::2]` to a 128×128 grid.
- **`A_model`** (used in `L_ray`): 128×128 grid, PSF 0.10″, bilinear interpolation.

Observations are never generated with the operator used in the loss. This rules out an *inverse crime* by construction. Sensitivity to this choice is checked in S4 (§9).

### 5.2 Training family (ID)
128×128 grid, `extent = 3.0″` (pixel step 47.24 mas), gain 500, noise `σ² = σ₀² + max(F,0)/gain`, `σ₀ ~ U(0.01, 0.03)` per sample.

| Parameter | Training range |
|---|---|
| θ_E | 0.8-1.4 |
| lens q | 0.71-0.95 |
| lens position x₀, y₀ | −0.1-0.1 |
| shear γ₁, γ₂ | −0.08-0.08 |
| lens angle | 0-π |
| profile slope s | 1.0 (SIE with core 0.05) |
| Sérsic source: amp | 0.8-1.5 |
| r_eff | 0.15-0.35 |
| n | 0.8-2.5 |
| source position | −0.4-0.4 |
| source q | 0.5-0.9 |

All ranges are stored in `configs/prior.yaml`. No script copies them by hand.

### 5.3 Datasets (finite, frozen, on disk)
Training data is **not generated online**. The training set is finite, because with an infinite data stream the benefit of the constraints is inherently underestimated.

| Dataset | Seed indices | Size | Notes |
|---|---|---|---|
| train | 0-7 999 | 8 000 | nested subsets: first 500, 2 000, 8 000 |
| val (ID) | 100 000-100 999 | 1 000 | the only set used for checkpoint selection and tuning |
| test ID | 200 000-200 499 | 500 | |
| test OOD, axis *k*, level *l* | 200 000-200 499 | 500 | paired with test ID: identical base seed and unit vector u, only axis mapping is altered |

An automated test checks that the index ranges are disjoint and that the datasets are bit-for-bit reproducible.

## 6. Generalization axes (OOD)

OOD sets are touched **only** in the final evaluation. They are not used for checkpoint selection, tuning or debugging methods.

| Axis | Change | Levels (mild / moderate / severe) |
|---|---|---|
| A1 parameters | extrapolation of one parameter: θ_E (upper), q (lower), shear (both sides), source position (both sides) | exceeding the range by 25% / 50% / 100% of the range width |
| A2 profile | slope `ψ ∝ r^s`, s ∈ [1−δ, 1+δ] | δ = 0.1 / 0.2 / 0.3 |
| A3 substructure | additional Plummer term with amplitude b near the ring | b ∈ [0.02, 0.05] / [0.05, 0.12] / [0.12, 0.25] |
| A4 source | 3 Gaussians instead of Sérsic | 1 level (fixed, enters E1) |
| A5 noise | `σ₀` outside the training range | 0.005 / 0.05 / 0.08 |
| A6 operator | A_truth with PSF 0.12″ instead of 0.10″ (mismatch with A_model) | PSF 0.12″ (moderate / L1, enters E1) / 0.15″ (severe / L2) |

For each axis we report a dose–response curve (metric vs level), not a single point.

## 7. Models and training

- **Architecture (main):** FNOLensingReconstructor (modes = 32, width = 64, pad_ratio = 0.25). We do not change it between variants.
- **Architecture (control, S2):** U-Net/CNN with a comparable number of parameters, V0 and V3 only, at N = 2 000. Checks whether the conclusions are specific to FNO.
- **Budget:** fixed number of optimizer steps for all N (12 500 steps, batch 16; for N = 8 000 this corresponds to about 25 epochs). Smaller N means more epochs.
- **Optimizer:** AdamW + OneCycleLR (parameters tuned in §7.1).
- **Warm-up of physical terms:** `warm(step) = clip((step − 1500) / 1500, 0, 1)`, corresponding to 0 for the first 1,500 steps (3 epochs at base N = 8,000, batch 16) and a linear ramp to 1.0 over steps 1,500–3,000 across all N.
- **Seeds:** 5 evaluation seeds (42-46). The seed controls initialization and data order. The same seed gives the same initialization in all variants (paired comparison).
- **Checkpoint selection:** one common criterion for all variants, `val_select = relL2(source) + 2·relL2(α)` on val (ID). The `L_ray` and `L_pos` terms do not enter the criterion.

### 7.1 Tuning (equal budget)
Tuning on **val (ID)**, N = 2 000, with a **separate seed 100** (outside the evaluation seeds). Each variant gets 9 configurations:

| Variant | Grid |
|---|---|
| V0 | lr ∈ {3e-4, 1e-3, 3e-3} × wd ∈ {1e-4, 1e-3, 1e-2} |
| V1 | λ_pos ∈ {1, 15, 50} × wd ∈ {1e-4, 1e-3, 1e-2} |
| V2 | λ_ray ∈ {0.005, 0.02, 0.1} × wd ∈ {1e-4, 1e-3, 1e-2} |
| V3 | λ_ray ∈ {0.005, 0.02, 0.1} × λ_pos ∈ {1, 15, 50} |

The selected configuration is fixed for all N and seeds. The aim is to ensure the baseline is not under-regularized relative to the variants with physical terms.

## 8. Metrics

All computed **per sample** (not as averages of batch means), on the 128×128 grid, with the ring ellipse matching the lens `r_ell`.

**Primary (neutral with respect to variants):**

| ID | Definition |
|---|---|
| M1 | `relL2(α)` in the band `|r_ell − θ_E| < 0.5″` (ellipse with the true lens parameters) |
| M2 | `relL2(source)` in an aperture of R = 1.0″ around the true source centre |

**Auxiliary (neutral):**

| ID | Definition |
|---|---|
| M1b | as M1, after subtracting the constant deflection component (removes the degeneracy with source translation) |
| M3 | `relL2(ψ)` after removing the constant (gauge-fixed) |
| M4 | `relL2(α)` over the full field (for comparison with M1) |
| M4b | source centroid error [mas] and flux error in the same aperture (no relative threshold) |

**Physical consistency (separate table, favours variants that optimized these quantities):**

| ID | Definition |
|---|---|
| M5 | unclamped `χ²`, separately on the arc mask (`F_clean > 3σ`) and globally, with σ estimated from the observation (oracle σ only as a control) |
| M6 | fraction of pixels with `κ < 0` in the grid interior |

**Generalization measure:** `R[v,a] = M1[v,a] / M1[v,ID]` (ratio of the error on the axis to the ID error) as well as the absolute `M1[v,a]`. We report both, because the gap `R` alone can shrink when ID performance worsens.

## 9. Experiments

| ID | Description | Role |
|---|---|---|
| E-main | V0-V3 × N ∈ {500, 2 000, 8 000} × 5 seeds × all OOD axes | confirmatory (H1, H2, H3) |
| S1 | V2 and V3 with clamped χ² (`chi2_mode = clamp`, max 100) instead of pseudo-Huber, N = 2 000 | sensitivity to the χ² clipping method |
| S2 | U-Net/CNN, V0 vs V3, N = 2 000 | sensitivity to architecture |
| S3 | oracle σ instead of estimated σ in `L_ray` (V2, V3) | sensitivity to noise |
| S4 | observations from `A_model` (the *inverse crime* variant), V0 vs V3, N = 2 000 | how much of the result depends on operator agreement |
| X1 | V4 (no labels), N = 2 000 | exploratory |
| X2 | semi-supervised extension: `L_ray` and `L_pos` on all 8 000 observations, `L_sup` on the first N | exploratory (mixes constraints with additional data) |

Number of trainings in E-main: 4 × 3 × 5 = 60 (+ 36 for tuning + controls).

## 10. Statistics and decision rules

**Units:**
- **test sample** level (n = 500 per set): variability of the test data,
- **seed** level (S = 5): variability of training.

**Estimation.** For variant `v`, seed `s`, set `a`: `m[v,s,a]` = median of M1 over samples. Effect relative to V0 on a log scale:

```
d[s,a] = log m[V3,s,a] − log m[V0,s,a]        (negative = V3 better)
```

**Primary endpoint E1** (a single test, no correction):

```
E1[s] = mean over OOD axes a ∈ {A1..A6, moderate level} of d[s,a]
```
For A4, the single fixed level is used; for A6, the moderate level corresponds to L1 (PSF 0.12″).
For N = 2 000 (reference size). H1 is considered supported when all of the following hold:
1. the mean of `E1[s]` over seeds is `≤ log(0.95)` (an improvement of at least 5%),
2. the 95% confidence interval for the mean (t distribution, 4 degrees of freedom) does not contain 0,
3. `E1[s] < 0` in at least 4 of 5 seeds.

**H2.** Slope of `E1` with respect to `log N` (regression over seeds), with a confidence interval. Supported when the interval does not contain 0 and the sign agrees with the hypothesis.

**H3.** Difference on ID `d[s,ID]` for M1 and M2: non-inferiority holds when the upper 95% CI bound is `≤ log(1.05)`.

**Individual axes and individual components (V1, V2 vs V0)** are secondary analyses: Holm correction over the family (6 axes × 3 comparisons), labelled as such in the report. They are not used to confirm H1.

**Additionally (always):** bootstrap interval over samples for `d` in each seed (95%), so that the contribution of test-set variability is visible. When the intervals over samples and over seeds disagree, the wider one governs.

**A null result is a result.** If H1 is not supported, the report states this and shows the distributions of `d`. The endpoint, the level and the set of axes must not be changed after looking at the results.

## 11. Threats to validity

| Threat | Handling |
|---|---|
| Constraints consistent with the generator (κ ≥ 0 always true in this family) | Described in the thesis. The conclusion concerns correct constraints; it does not study harm from incorrect ones. Optionally an axis with a component of κ < 0 (§15, item 4) |
| Inverse crime | `A_truth` ≠ `A_model` by construction; S4 measures the difference |
| Baseline tuning | Equal tuning budget (§7.1) |
| OOD leakage into tuning | OOD touched only in the final evaluation; checkpoint chosen on val (ID) |
| Narrow training family | Stated explicitly in the scope of conclusions (§12) |
| Different checkpoint selection criteria | Common `val_select` |
| Few seeds | 5 seeds; Student-t intervals show the uncertainty; no conclusions drawn from differences within the seed spread |
| Hidden dependencies: constant, mass-sheet | §3; M1b as a control |
| Finite training set vs online stream | Deliberate (§5.3); checking an online stream is out of scope |

## 12. Scope of conclusions

Conclusions concern: synthetic data from the family in §5.2, the six axes in §6, FNO and U-Net architectures, five seeds. They **do not concern** real data, other lens families or other loss functions. Label-free training (V4) is an exploration, not a result.

## 13. Deviations log (changes after freezing)

| Date | What changed | Reason | Impact on analysis |
|---|---|---|---|
| 2026-11-05 | Corrected OOD seeds from 300 000+ to 200 000-200 499 in §5.3 and clarified A1 extrapolation directions in §6 | Resolved specification logic error; exact pairing requires using identical base seeds as test ID | No impact on hypotheses; strictly required for paired sample evaluation |
| 2026-11-08 | Clarified moderate level mapping for A4 (fixed) and A6 (L1: PSF 0.12″) in §6 and §10 | Resolved ambiguity in composite endpoint E1 for axes with non-standard level counts | No impact on hypotheses; formalized prior to training runs |
| 2026-11-10 | Clarified pseudo-Huber formulation in §2 and explicit warm-up schedule formula in §7 | Resolved ambiguity in loss specification (Huber on pull scaled by 2c²; fixed step-based warmup across all N) | Preserves λ_ray scale consistency with clamp(max=100); deterministic optimizer schedule |
| 2026-11-11 | Refactored `SpectralConv2d` (weights reshaped to `[m1·m2, in, out]`; `modes`, `width`, `pad_ratio` unchanged) | Fewer memory copies and execution speedup via `torch.bmm` | None; identical mathematical function, forward and gradient equivalence verified by test |
| 2026-11-11 | Runtime estimate correction in §15 item 1 | Profiling and smoke tests measured ~0.32 s/step (~72 min/run instead of <8 min; ~116 h total on GTX 1660) | None; concerns computational schedule only, not hypotheses, decision rules, or metrics |

## 14. Reproducibility

- All parameters in `configs/*.yaml`. The report contains the config hash and the commit hash.
- Datasets stored on disk with a checksum. Tests: disjointness, reproducibility, agreement of the `A_truth` / `A_model` grids.
- `scripts/make_tables.py` reproduces every table and figure from the stored checkpoints, with no manual steps.
- For each checkpoint we record: variant, N, seed, configuration, `val_select` and epoch number.
- Weights in `SpectralConv2d` are flattened to shape `(modes1 * modes2, in_channels, out_channels)` for computational acceleration via `torch.bmm` instead of `einsum` (numerically equivalent, distinct `state_dict` key layout).

## 15. Decisions Prior to Freezing (Resolved)

1. **Seed count and compute feasibility:** Confirmed 5 seeds (42–46) and 60 training runs in E-main. On an NVIDIA GTX 1660, a single training run takes ~72 minutes (~0.32 s/step including evaluations every 500 steps), requiring ~72 hours total for E-main (~116 hours including the 36-run tuning phase). 5 seeds provide 4 degrees of freedom in the t-distribution ($t_{0.05} = 2.776$), ensuring statistical power.
2. **Significance thresholds:** The 5% improvement threshold (`ln ≤ -0.051`) and 5% non-inferiority bound (`ln ≤ +0.049`) in §10 are frozen.
3. **Axis A1 aggregation:** Axis A1 enters composite endpoint E1 as a single aggregated axis (unweighted mean of its four constituent parameter sweeps).
4. **Negative mass axis (κ < 0):** Added strictly as a standalone diagnostic control outside E1 to assess model behavior under violated prior assumptions.
5. **Detector gain:** Frozen at `gain = 500.0`, maintaining a realistic Poisson shot-noise regime in high-magnification arcs.
