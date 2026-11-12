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
| `L_pos` | mean `ReLU(−κ)²`, `κ = ½∇²ψ`, grid interior excluding a 2-pixel border | **κ ≥ 0 holds for the entire data family (including all OOD axes A1–A6), so the constraint is correct but consistent with the generator** (§11); the only exception is the diagnostic Kneg axis outside E1 (§15, item 4) |

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
- **`A_truth`** (generates all observations): 255×255 grid (every second node coincides with a node of the 128 grid), Gaussian PSF with FWHM 0.10″, subsampling `[::2]` to a 128×128 grid.
- **`A_model`** (used in `L_ray`): 128×128 grid, PSF 0.10″, bilinear interpolation.

Observations are never generated with the operator used in the loss. This rules out an *inverse crime* in the strict sense. The difference between the operators is small, however: both use the same ray tracer, the same PSF, bilinear interpolation and α from central differences of the sampled ψ. They differ only in resolution (255 vs 128) and in `A_truth` sampling the source analytically on the fine grid. The protection is therefore weak, and S4 (§9) measures how much of the result depends on it.

### 5.2 Training family (ID)
128×128 grid, `extent = 3.0″` (half-width of the field, i.e. ±3″; pixel step 47.24 mas), gain 500, noise `σ² = σ₀² + max(F,0)/gain`, `σ₀ ~ U(0.01, 0.03)` per sample.

| Parameter | Training range |
|---|---|
| θ_E | 0.8-1.4 |
| lens q | 0.71-0.95 |
| lens position x₀, y₀ | −0.1-0.1 |
| shear γ₁, γ₂ | −0.08-0.08 |
| lens angle | 0-π |
| profile slope s | 1.0 (isothermal elliptical potential `ψ = θ_E·r_ell`, core 0.05; not an elliptical mass distribution) |
| Sérsic source: amp | 0.8-1.5 |
| r_eff | 0.15-0.35 |
| n | 0.8-2.5 |
| source position | −0.4-0.4 |
| source q | 0.5-0.9 |
| source angle | 0-π |

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
| A1 parameters | extrapolation of one parameter group: θ_E (upper), q (lower), shear γ₁ and γ₂ together (both sides), source position x and y together (both sides) | values drawn uniformly from a band of width 25% / 50% / 100% of the range width just outside the range |
| A2 profile | slope `ψ = θ_E^(2−s)·r^s/s`, s ∈ [1−δ, 1+δ] | δ = 0.1 / 0.2 / 0.3 |
| A3 substructure | additional Plummer term with amplitude b near the ring (b = peak deflection [arcsec], core a = 0.1″) | b ∈ [0.02, 0.05] / [0.05, 0.12] / [0.12, 0.25] |
| A4 source | 3 Gaussians instead of Sérsic | 1 level (fixed, enters E1) |
| A5 noise | `σ₀` outside the training range | 0.005 / 0.05 / 0.08 |
| A6 operator | A_truth with PSF 0.12″ instead of 0.10″ (mismatch with A_model) | PSF 0.12″ (moderate / L1, enters E1) / 0.15″ (severe / L2) |

For each axis we report a dose–response curve (metric vs level), not a single point.

Generator details (`src/data/generator.py`):
- **A1:** each level draws the parameter uniformly from a band of width f·w just outside the training range (f = 0.25 / 0.5 / 1.0, w = range width). On two-sided axes the side is set by that component's base draw: u < 0.5 below the range, u ≥ 0.5 above it. All other parameters keep their test-ID values. A1 enters E1 as one axis (§15, item 3).
- **A2:** the normalization `θ_E^(2−s)` keeps the Einstein radius at θ_E for every s.
- **A3:** `ψ_sub = b·a·ln(|θ − θ_s|² + a²)` with a = 0.1″, so b is the peak deflection of the perturbation. The centre θ_s lies at distance θ_E·U(0.6, 1.2) from the grid centre (not the lens centre), at angle U(0, 2π).
- **A4:** three Gaussians with amplitude U(0.5, 1), position U(−0.4, 0.4)″ and width σ U(0.08, 0.2)″. The source centre used by M2 and M4b is the flux-weighted centroid (weights a·σ²).
- **Auxiliary draws** for A2–A4 come from `np.random.SeedSequence([seed, axis_id, level])`, so they do not consume the base stream (u, z) and the pairing with test ID is preserved.

## 7. Models and training

- **Architecture (main):** FNOLensingReconstructor (modes = 32, width = 64, pad_ratio = 0.25). We do not change it between variants.
- **Input and output:** the network sees only `observed` (plus two coordinate channels), not `noise_map`. The ψ output is centred (grid mean subtracted); the source output passes through Softplus, so it is non-negative.
- **Architecture (control, S2):** U-Net/CNN with a comparable number of parameters, V0 and V3 only, at N = 2 000. Checks whether the conclusions are specific to FNO.
- **Budget:** fixed number of optimizer steps for all N (12 500 steps, batch 16; for N = 8 000 this corresponds to about 25 epochs). Smaller N means more epochs.
- **Optimizer:** AdamW + OneCycleLR with PyTorch defaults (pct_start = 0.3, cosine annealing, div_factor = 25, final_div_factor = 10⁴) and max_lr = lr (tuned in §7.1); gradient norm clipped to 1.0.
- **Warm-up of physical terms:** `warm(step) = clip((step − 1500) / 1500, 0, 1)`, corresponding to 0 for the first 1,500 steps (3 epochs at base N = 8,000, batch 16) and a linear ramp to 1.0 over steps 1,500–3,000 across all N.
- **Seeds:** 5 evaluation seeds (42-46). The seed controls initialization and data order. The same seed gives the same initialization in all variants (paired comparison).
- **Data order:** each epoch is a fresh permutation of the N training samples (generator seeded with the run seed); the incomplete last batch of an epoch is dropped.
- **Checkpoint selection:** one common criterion for all variants, `val_select = relL2(source) + 2·relL2(α)` on val (ID). The `L_ray` and `L_pos` terms do not enter the criterion. Evaluated every 500 steps (25 evaluations per run); the best one is kept.

### 7.1 Tuning (equal budget)
Tuning on **val (ID)**, N = 2 000, with a **separate seed 100** (outside the evaluation seeds). Each variant gets 9 configurations:

| Variant | Grid |
|---|---|
| V0 | lr ∈ {3e-4, 1e-3, 3e-3} × wd ∈ {1e-4, 1e-3, 1e-2} |
| V1 | λ_pos ∈ {1, 15, 50} × wd ∈ {1e-4, 1e-3, 1e-2} |
| V2 | λ_ray ∈ {0.005, 0.02, 0.1} × wd ∈ {1e-4, 1e-3, 1e-2} |
| V3 | λ_ray ∈ {0.005, 0.02, 0.1} × λ_pos ∈ {1, 15, 50} |

V1–V3 take lr from the V0 choice; V3 also takes wd from V0. If the optimum lies on the edge of a grid, the grid is not extended; this is reported with the results.

The selected configuration is fixed for all N and seeds. The aim is to ensure the baseline is not under-regularized relative to the variants with physical terms.

## 8. Metrics

All computed **per sample** (not as averages of batch means), on the 128×128 grid, with the ring ellipse matching the lens `r_ell`.

**Primary (neutral with respect to variants):**

| ID | Definition |
|---|---|
| M1 | `relL2(α)` in the band `\|r_ell − θ_E\| < 0.5″` (ellipse with the true lens parameters) |
| M2 | `relL2(source)` in an aperture of R = 1.0″ around the true source centre (for high Sérsic n the central peak reaches up to ~100× the brightness at r_eff, so M2 is dominated by the central pixels) |

**Auxiliary (neutral):**

| ID | Definition |
|---|---|
| M1b | as M1, after subtracting the mean deflection-error vector in the band from the error (denominator unchanged; removes the degeneracy with source translation) |
| M3 | `relL2(ψ)` after removing the constant (gauge-fixed) |
| M4 | `relL2(α)` over the full field (for comparison with M1) |
| M4b | source centroid error [mas]: flux-weighted centroid of the prediction vs that of the true source image, both inside the M2 aperture; flux error `F_pred/F_true − 1` in the same aperture, reported as the median (bias) and the median of its absolute value (scatter) |

**Physical consistency (separate table, favours variants that optimized these quantities):**

| ID | Definition |
|---|---|
| M5 | unclamped `χ²` computed with `A_model` (PSF 0.10″ on every axis, including A6), separately on the arc mask (`clean_lensed > 3σ_true`) and globally, with σ estimated from the observation (oracle σ only as a control); an empty arc mask gives NaN, aggregated with nanmedian |
| M6 | fraction of pixels with strictly `κ < 0` (κ = 0 does not count) in the grid interior |

**Generalization measure:** `R[v,a] = M1[v,a] / M1[v,ID]` (ratio of the error on the axis to the ID error) as well as the absolute `M1[v,a]`. We report both, because the gap `R` alone can shrink when ID performance worsens.

## 9. Experiments

| ID | Description | Role |
|---|---|---|
| E-main | V0-V3 × N ∈ {500, 2 000, 8 000} × 5 seeds × all OOD axes | confirmatory (H1, H2, H3) |
| S1 | V2 and V3 with clamped χ² (`chi2_mode = clamp`, max 100) instead of pseudo-Huber, N = 2 000 | sensitivity to the χ² clipping method |
| S2 | U-Net/CNN, V0 vs V3, N = 2 000 | sensitivity to architecture |
| S3 | oracle σ instead of estimated σ in `L_ray` (V2, V3) | sensitivity to noise |
| S4 | observations from `A_model` (the *inverse crime* variant), V0 vs V3, N = 2 000; all splits (train, val, test ID, OOD) regenerated with `A_model` on the E-main seeds, configurations from `tuned.yaml` without separate tuning, seeds 42–46, data in `data_s4/`, runs tagged S4 | how much of the result depends on operator agreement |
| X1 | V4 (no labels), N = 2 000; λ_ray and λ_pos from the tuned V3; checkpoint selected by `val_select`, as for all variants (this uses the 1 000 labelled validation samples and is stated in the report) | exploratory |
| X2 | semi-supervised extension: `L_ray` and `L_pos` on all 8 000 observations, `L_sup` on the first N | exploratory (mixes constraints with additional data) |
| X3 | one repeat of the selected V0 and V3 tuning configurations (N = 2 000, seed 100) | scale of run-to-run noise from GPU non-determinism; descriptive only, does not change the tuning choice |
| X4 | variant R = `L_sup + λ_R·L_R`, `L_R = mean((∇²ψ)²)` on the grid interior (non-physical smoothness penalty on the curvature of ψ regardless of sign); λ_R from 3 values tuned at N = 2 000, seed 100, then 5 seeds (42–46); compared with V3 on the OOD axes as in §10 | exploratory; run **only if H1 is supported**; tests whether a gain of V3 reflects generic regularization. Grid fixed in advance: λ_R such that λ_R·L_R on the true ψ of the training set equals 1%, 10% and 100% of the final training `L_sup` of the selected V0 tuning run |

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

**H2.** Slope of `E1` with respect to `log N`, fitted separately in each seed (3 points per seed); the mean slope over seeds with a 95% t interval (4 degrees of freedom). Supported when the interval does not contain 0 and the sign agrees with the hypothesis.

**H3.** Difference on ID `d[s,ID]` for M1 and M2: non-inferiority holds when the upper 95% CI bound is `≤ log(1.05)`. Decided at N = 2 000.

**Individual axes and individual components (V1, V2 vs V0)** are secondary analyses: Holm correction over the family (6 axes × 3 comparisons = 18 tests), labelled as such in the report. They are not used to confirm H1.

**Additionally (always):** bootstrap interval over samples for `d` in each seed (95%), so that the contribution of test-set variability is visible. The two intervals compared are the t interval over seeds and the bootstrap interval of the seed mean; the wider one governs, for criterion 2 of H1 and for H3 (H2 uses the t interval). In each bootstrap replicate one set of test-sample indices is drawn and shared by all seeds, both variants and all sets, because the test set is common to them and OOD sets are paired with test ID; percentile interval, 10 000 replicates. Secondary tests are two-sided one-sample t tests over seeds (4 df) on the axis effect, Holm-adjusted at α = 0.05.

**A null result is a result.** If H1 is not supported, the report states this and shows the distributions of `d`. The endpoint, the level and the set of axes must not be changed after looking at the results.

## 11. Threats to validity

| Threat | Handling |
|---|---|
| Constraints consistent with the generator (κ ≥ 0 always true in this family) | Described in the thesis. The conclusion concerns correct constraints; it does not study harm from incorrect ones. Optionally an axis with a component of κ < 0 (§15, item 4) |
| Inverse crime | `A_truth` ≠ `A_model` by construction, but they differ only in resolution and source sampling (§5.1); S4 measures the effect |
| Baseline tuning | Equal tuning budget (§7.1) |
| OOD leakage into tuning | OOD touched only in the final evaluation; checkpoint chosen on val (ID) |
| Narrow training family | Stated explicitly in the scope of conclusions (§12) |
| Different checkpoint selection criteria | Common `val_select` |
| Few seeds | 5 seeds; Student-t intervals show the uncertainty; no conclusions drawn from differences within the seed spread |
| Hidden dependencies: constant, mass-sheet | §3; M1b as a control |
| Finite training set vs online stream | Deliberate (§5.3); checking an online stream is out of scope |
| Hyperparameters tuned only at N = 2 000 | Stated as a limitation of H2: the dependence of the effect on N may partly reflect λ, lr and wd being less well matched at N = 500 and 8 000 |
| Labelled validation set larger than the smallest training set | Checkpoint selection uses 1 000 labelled samples for every N; conclusions about scarce data concern the training set only (§12) |
| Physics terms acting as generic regularizers | The network overfits strongly (first tuning run: training `L_sup` 0.144 vs `val_select` 0.506), so a gain of V3 may reflect regularization in general rather than physics; the design does not separate the two. Stated in the discussion; V0 has its weight decay tuned (§7.1) |

## 12. Scope of conclusions

Conclusions concern: synthetic data from the family in §5.2, the six axes in §6, FNO and U-Net architectures, five seeds. They **do not concern** real data, other lens families or other loss functions. Label-free training (V4) is an exploration, not a result. The lens is an elliptical potential, not an elliptical mass distribution: its isodensity contours become dumbbell-shaped at low q, notably on the A1 q axis (down to q ≈ 0.47); κ stays positive. Images contain no lens-galaxy light. "Little training data" refers to the training set only: checkpoint selection always uses 1 000 labelled validation samples.

## 13. Deviations log (changes after freezing)

| Date | What changed | Reason | Impact on analysis |
|---|---|---|---|
| 2026-11-05 | Corrected OOD seeds from 300 000+ to 200 000-200 499 in §5.3 and clarified A1 extrapolation directions in §6 | Resolved specification logic error; exact pairing requires using identical base seeds as test ID | No impact on hypotheses; strictly required for paired sample evaluation |
| 2026-11-08 | Clarified moderate level mapping for A4 (fixed) and A6 (L1: PSF 0.12″) in §6 and §10 | Resolved ambiguity in composite endpoint E1 for axes with non-standard level counts | No impact on hypotheses; formalized prior to training runs |
| 2026-11-10 | Clarified pseudo-Huber formulation in §2 and explicit warm-up schedule formula in §7 | Resolved ambiguity in loss specification (Huber on pull scaled by 2c²; fixed step-based warmup across all N) | Preserves λ_ray scale consistency with clamp(max=100); deterministic optimizer schedule |
| 2026-11-11 | Refactored `SpectralConv2d` (weights reshaped to `[m1·m2, in, out]`; `modes`, `width`, `pad_ratio` unchanged) | Fewer memory copies and execution speedup via `torch.bmm` | None; identical mathematical function, forward and gradient equivalence verified by test |
| 2026-11-11 | Runtime estimate correction in §15 item 1 | Profiling and smoke tests measured ~0.32 s/step (~72 min/run instead of <8 min; ~116 h total on GTX 1660) | None; concerns computational schedule only, not hypotheses, decision rules, or metrics |
| 2026-11-12 | Fixed pipe characters in §8 (M1 row) | Added backslashes before vertical bars (`\|r_ell − θ_E\|`) to prevent Markdown table splitting | None; formatting fix only |
| 2026-11-12 | Clarified training vs. evaluation time breakdown in §15 item 1 | Separated pure training step time (~0.32 s/step) from validation evaluations (25 × ~12 s), which sum to ~72 min/run | None; wording precision in compute budget only |
| 2026-11-12 | OOD generation details in §6: A1 band semantics, A2 normalization, A3 Plummer (b = peak deflection, a = 0.1″, centre position), A4 Gaussians and flux-weighted centroid, auxiliary draws via `SeedSequence([seed, axis_id, level])` | Decisions taken before data generation but not logged; text now matches `generator.py` | None on hypotheses; describes the datasets as generated |
| 2026-11-12 | Metric definitions in §8: M1b (mean error vector in the band subtracted, denominator unchanged), M5 (`A_model` with PSF 0.10″ on every axis, arc mask `clean_lensed > 3σ_true`, empty mask = NaN + nanmedian), M6 (strict κ < 0) | Decisions taken when implementing the metrics, before any evaluation | Fixes metrics before results are seen |
| 2026-11-12 | Decision rules in §10: H2 as per-seed regression with t interval (4 df); H3 decided at N = 2 000; the "wider interval" compares the t interval over seeds with the bootstrap interval of the seed mean; Holm over 18 tests | Removes ambiguity in the decision rules | Rules fixed before evaluation |
| 2026-11-12 | Tuning procedure in §7.1: V1–V3 inherit lr from V0, V3 also wd; grid not extended when the optimum is on its edge | Matches `scripts/tune.py`; inheritance agreed before tuning | Fixed before the first tuning run |
| 2026-11-12 | S4 procedure in §9: all splits regenerated with `A_model` on E-main seeds, `tuned.yaml` without separate tuning, seeds 42–46, `data_s4/`, tag S4 | Makes the control experiment fully specified | None on E-main |
| 2026-11-12 | Kneg axis specified in §15 item 4 (mirror of A3 with b < 0, outside E1, `DIAG_LEVELS`; L1 borderline accepted); §2 note on κ ≥ 0 updated | Specifies the optional axis from §11 | Diagnostic only, outside E1 |
| 2026-11-12 | Descriptive clarifications in §5.1, §5.2, §7, §11 and §14: operator difference, `extent` as half-width, source angle range, network input and output, OneCycleLR defaults, gradient clipping at 1.0, dropped incomplete batch, evaluation every 500 steps, logging of step and reference epoch, GPU non-determinism | The text did not describe what the code does | None; descriptive only |
| 2026-11-12 | Wording of §15 item 3 aligned with the Polish version (mean of log-ratios d) | Language versions disagreed in precision | None |
| 2026-11-12 | Added the Zenodo DOI of the archived protocol to §14 | Public, immutable timestamp of the frozen protocol | None |
| 2026-11-12 | Lens model renamed from "SIE" to isothermal elliptical potential in §5.2; consequences added to §12 | The text misnamed the model implemented in `profiles.py`; code and data unchanged | None on results; limits the astrophysical interpretation, especially of the A1 q axis |
| 2026-11-12 | Three threats added to §11 (tuning only at N = 2 000, labelled validation set, physics terms as generic regularizers); note on M2 in §8 | Methodological review after the first completed tuning run (V0, lr 3e-4, wd 1e-4) | None on decision rules; limits interpretation of H1 and H2 |
| 2026-11-12 | Disclosure: test-ID seeds 200 000–200 149 were used without any model in `tests/test_data.py` and `scripts/check_chi2_truth.py` while developing the σ̂ estimator used in `L_ray`; smoke tests used seed 42 and the validation set | Transparency | Negligible; no model was evaluated on any test set before the final evaluation |
| 2026-11-12 | Disclosure: tuning runs record the git state at the start of each run, while the tuning process executes the code loaded at its start (commit c4585d4); later tuning runs may therefore show a later commit or a dirty flag | Provenance; training, loss and model code did not change after c4585d4 | None |
| 2026-11-12 | M4b definition fixed in §8 (centroid against the true image centroid in the aperture; signed relative flux error, median and median absolute value) | The protocol did not define the reference centroid or the flux error; decided before any evaluation | Fixes an auxiliary metric before results |
| 2026-11-12 | X1 (V4) specified in §9: λ from tuned V3, checkpoint by `val_select` | Keeps the common selection criterion of §7; the use of labelled validation data is disclosed | Exploratory only |
| 2026-11-12 | X3 added to §9: one repeat of the selected V0 and V3 tuning configurations | GPU non-determinism produced differences of ~0.01–0.015 in `val_select` between tuning runs mid-training | Descriptive; tuning choice unchanged |
| 2026-11-12 | X4 added to §9: conditional control with a non-physical smoothness penalty, grid rule fixed in advance | Addresses the threat "physics terms as generic regularizers" (§11); registered before any tuning result of V1–V3 or any test result | Exploratory; runs only if H1 is supported |
| 2026-11-12 | Bootstrap and interval rules specified in §10: indices shared across seeds, variants and sets (previously "within each seed"); percentile interval, 10 000 replicates; wider-of rule applies to H1 criterion 2 and H3; secondary tests are one-sample t tests with Holm at α = 0.05 | Implementation of §10 in `src/evaluation/stats.py`; resampling independently per seed would treat one common test set as five and understate test-set variability | Fixes the statistical procedure before any test result; implementation verified on synthetic data with known effects |

## 14. Reproducibility

- All parameters in `configs/*.yaml`. The report contains the config hash and the commit hash.
- Datasets stored on disk with a checksum. Tests: disjointness, reproducibility, agreement of the `A_truth` / `A_model` grids.
- `scripts/make_tables.py` reproduces every table and figure from the stored checkpoints, with no manual steps.
- For each checkpoint we record: variant, N, seed, configuration, `val_select`, step and reference epoch (step / 500, i.e. epochs at N = 8 000).
- Datasets and initialization are bit-for-bit reproducible; GPU training is not (non-deterministic FFT and `grid_sample` backward kernels), so repeated runs with the same seed may differ slightly.
- Weights in `SpectralConv2d` are flattened to shape `(modes1 * modes2, in_channels, out_channels)` for computational acceleration via `torch.bmm` instead of `einsum` (numerically equivalent, distinct `state_dict` key layout).
- The frozen protocol, with the §13 log up to 2026-11-12 and the code at that date, is archived on Zenodo before any test-set evaluation: [10.5281/zenodo.24164619](https://doi.org/10.5281/zenodo.24164619).

## 15. Decisions Prior to Freezing (Resolved)

1. **Seed count and compute feasibility:** Confirmed 5 seeds (42–46) and 60 training runs in E-main. On an NVIDIA GTX 1660, a single FNO training run of 12,500 steps takes ~72 minutes (12,500 steps at ~0.32 s/step plus 25 evaluations at ~12 s each), requiring ~72 hours total for E-main (~116 hours including the 36-run tuning phase). Originally estimated at 7–8 hours total. 5 seeds provide 4 degrees of freedom in the t-distribution ($t_{0.05} = 2.776$), ensuring statistical power.
2. **Significance thresholds:** The 5% improvement threshold (`ln ≤ -0.051`) and 5% non-inferiority bound (`ln ≤ +0.049`) in §10 are frozen.
3. **Axis A1 aggregation:** Axis A1 enters composite endpoint E1 as a single aggregated axis (arithmetic mean of the log-ratios d of its four sub-axes: θ_E, q, shear and source position).
4. **Negative mass axis (κ < 0):** Added strictly as a standalone diagnostic control outside E1 to assess model behavior under violated prior assumptions. Implemented as an exact mirror of A3 (same draw stream, a = 0.1″, sign of b reversed), kept outside E1 and outside `N_LEVELS` (separate `DIAG_LEVELS`). At L1 roughly half of the samples violate κ ≥ 0; this borderline level is accepted.
5. **Detector gain:** Frozen at `gain = 500.0`, maintaining a realistic Poisson shot-noise regime in high-magnification arcs.
