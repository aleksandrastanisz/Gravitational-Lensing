# PI-FNO: Physics-Informed Fourier Neural Operators for Gravitational Lensing Inversion

A rigorous benchmark evaluating the out-of-distribution (OOD) generalization of Fourier Neural Operators (FNO) for gravitational lensing inversion under physical constraints. The model solves the inverse problem by simultaneously reconstructing the 2D lens potential $\psi$ and unlensed source surface brightness $S$ from a single noisy observation $I_{\text{obs}}$.

---

## Overview

Inverting the lens equation $\beta = \theta - \nabla\psi(\theta)$ is ill-posed and susceptible to mathematical degeneracies (e.g., mass-sheet transformation, potential gauge shifts). Standard deep learning baselines overfit to training distributions and hallucinate unphysical structures under distribution shifts. 

This project assesses whether embedding explicit physical inductive biases—namely **ray-tracing consistency** and **mass positivity**—stabilizes neural inversion under controlled distributional shift without committing an *inverse crime*.

---

## Methodological Highlights

- **Operator Discretization (No Inverse Crime)**: 
  - Ground truth observations are synthesized via $A_{\text{truth}}$ on a fine $255 \times 255$ grid and sub-sampled (stride 2) to the observation grid $128 \times 128$.
  - The model loss uses $A_{\text{model}}$ operating natively at $128 \times 128$.
- **Strictly Paired OOD Datasets**:
  - Evaluation datasets across 6 OOD axes (parameter extrapolation, profile slope, Plummer subhalos, clumpy morphology, noise levels, and PSF mismatch) share identical latent random vectors $u$ with the ID test set.
  - Frozen storage via `numpy.memmap` validated with SHA-256 integrity manifests.
- **Degeneracy-Aware Evaluation Suite (M1–M6)**:
  - **M1 / M1b**: Annular deflection error along the Einstein ring, with M1b isolating curvature errors by removing dipole offset degeneracies.
  - **M2**: Aperture-bounded relative source error ($r \le 1.0''$).
  - **M3**: Gauge-invariant potential relative error (mean subtracted).
  - **M4b**: Source astrometric centroid error (mas) and total photometric flux recovery.
  - **M5**: Global and arc-masked unclipped $\chi^2$ residuals.
  - **M6**: Interior fraction of unphysical negative convergence ($\kappa = \frac{1}{2}\nabla^2\psi < 0$).

---

## Model Variants

All variants share an identical 2D FNO backbone and are trained under a fixed budget of 12,500 steps using AdamW and OneCycleLR with step-based warmup schedules.

| Variant | Supervised ($L_{\text{sup}}$) | Mass Positivity ($L_{\text{pos}}$) | Ray-Consistency ($L_{\text{ray}}$) | Objective |
| :---: | :---: | :---: | :---: | :--- |
| **V0** | Yes | No | No | Purely data-driven supervised baseline ($\alpha = \nabla\psi$ and $S$) |
| **V1** | Yes | Yes ($\lambda_{\text{pos}} = 15.0$) | No | Supervised + Poisson mass positivity constraint ($\kappa \ge 0$) |
| **V2** | Yes | No | Yes ($\lambda_{\text{ray}} = 0.02$) | Supervised + Ray-tracing pseudo-Huber consistency |
| **V3** | Yes | Yes ($\lambda_{\text{pos}} = 15.0$) | Yes ($\lambda_{\text{ray}} = 0.02$) | Full Physics-Informed FNO (PI-FNO) |
