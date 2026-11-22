# OncoVac Adenine (Gen 1)

**Physics-Coupled Molecular AI for Somatic Oncology Mutation Thermodynamics in Pure Differentiable JAX**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Built with JAX](https://img.shields.io/badge/Built%20with-JAX-orange.svg)](https://github.com/jax-ml/jax)
[![Status: Active Research](https://img.shields.io/badge/Status-Gen%201%20Active%20Training-yellow.svg)](#benchmark-status)
[![Website](https://img.shields.io/badge/Lab-nucleobaselabs.com-2B5A43.svg)](https://nucleobaselabs.com)

---

## Overview

Static structure predictors excel at wild-type equilibrium folds, yet living proteins are dynamic thermodynamic machines. A single somatic point mutation in a patient's tumor can subtly perturb side-chain packing, open cryptic binding pockets, shift conformational ensembles over time, and alter binding free energy ($\Delta\Delta G$)—signals that static single-snapshot predictions frequently miss.

**OncoVac Adenine** (originally initiated as *CancerFold*) is the flagship physics-coupled molecular AI architecture from **Nucleobase Labs** (Malang, Indonesia). Implemented end-to-end in pure differentiable **JAX**, it unites first-principles 3D statistical mechanics, open **Joltz-1** (Boltz-1 JAX port) biomolecular priors, **AlphaGenome** 1D regulatory dosage signals, and exact **Bennett Acceptance Ratio (BAR)** free-energy estimation in $\text{kcal mol}^{-1}$.

---

## Architecture — Four Coupled Layers

```text
Input (WT / Mutant Complex .pdb + Target Mutation + Optional Small-Molecule Ligand)
   │
   ▼
Layer 01 │ 3D Backbone Frames (N, Cα, C), Chiral Cβ* Geometry & Adaptive Debye–Hückel (ε = 8 ↔ 78.4)
   │
   ▼
Layer 02 │ Frozen Joltz-1 Pairformer Priors (token_s=384, token_z=128) + Conservative SE(3) U_NN(X)
   │
   ▼
Layer 03 │ Overdamped Langevin SDE at 310 K (jax.vmap / jax.lax.scan) + Metadynamics + Exact BAR
   │
   ▼
Layer 04 │ Bound–Unbound ΔΔG Cycles, Cryptic-Pocket Analytics (RMSF, P_open), & Ligand Ranking
```

1. **Layer 01 — Analytical 3D Physics & Adaptive Electrostatics:** Oriented `(N, Cα, C)` backbone frames, chiral `Cβ*` side-chain geometry, and coordination-burial adaptive Debye–Hückel electrostatics (`ε = 8 ↔ 78.4`).
2. **Layer 02 — Frozen Joltz-1 Prior & SE(3) Neural Potential:** Stop-gradient **Joltz-1** Pairformer representations (`token_s = 384`, `token_z = 128`) coupled with a conservative, $SE(3)$-invariant neural residual potential $U_{\text{NN}}(X)$ yielding exact forces $F = -\nabla H$.
3. **Layer 03 — Langevin SDE, Metadynamics & Exact BAR:** Parallel stochastic conformational sampling at `310 K` (`jax.vmap`, `jax.lax.scan`) with optional pocket-targeted Metadynamics and differentiable **Bennett Acceptance Ratio (BAR)** free-energy estimation enforcing exact $\text{WT} \leftrightarrow \text{Mut}$ thermodynamic antisymmetry.
4. **Layer 04 — Oncology Target & Drug-Selection Scope:** End-to-end bound–unbound thermodynamic cycles for mutation affinity shifts and small-molecule stabilizer ranking across oncogenic targets (`p53 Y220C`, `KRAS`, `EGFR`).

---

## Preliminary Verification (Mini-SKEMPI 2.0 Proof-of-Concept)

Before running the full 3-fold homology/complex-disjoint SKEMPI 2.0 suite, we verified end-to-end differentiability, zero-leakage splitting, and exact BAR antisymmetry on a **Mini-SKEMPI 2.0** subset (7 PDB complexes, 42 validated mutations; complex-disjoint split: 5 train PDBs [30 muts] vs. 2 unseen val PDBs [`1CBW`, `1CHO`, 12 muts]) on a single GPU (`JAX 0.11.1`):

| Stage | Val Pearson `r` | Val Spearman `ρ` | Val RMSE (`kcal/mol`) | Val MAE (`kcal/mol`) |
|---|---:|---:|---:|---:|
| Zero-Shot `H_phys` only | `-0.639` | `-0.042` | `3.864` | `1.952` |
| **Differentiable BAR (`H_phys + U_NN`, Best Val / Ep 01)** | **`+0.867`** | **`+0.902`** | **`1.600`** | **`1.199`** |
| Differentiable BAR (Ep 10, unregularized) | `+0.618` | `+0.916` | `2.928` | `2.465` |

* **Unseen-Complex Exact BAR Antisymmetry Audit:** $\max|\Delta\Delta G_{\text{fwd}} + \Delta\Delta G_{\text{rev}}| = 4.77 \times 10^{-7}$ (`float32` machine precision).
* **Scope Note:** These numbers represent an early 42-mutation proof-of-concept (where several complexes share protease–inhibitor fold homology) and show expected magnitude overfitting by Epoch 10 on 30 training mutations. Full publication benchmarks will be reported only after completing zero-leakage **3-fold homology/complex-disjoint SKEMPI 2.0 CV** with best-checkpoint selection and regularization.

---

## Benchmark Status

| Benchmark | Protocol & Metrics | Status |
|---|---|---|
| **Mini-SKEMPI 2.0 PoC** | Complex-disjoint sanity check (42 muts) & exact BAR antisymmetry (`4.77e-07`) | **Completed (PoC)** |
| **Full SKEMPI 2.0** | 3-fold homology/complex-disjoint CV · Overall & Per-Complex `r`, `ρ`, `RMSE`, exact BAR bias | **Active Training** |
| **p53 Y220C Cryptic Pocket** | Dynamic pocket opening at `310 K` vs. crystal structures (`PDB 2J1X / 2VUK`) | **Queued** |
| **Targeted Oncology Drug Panel** | Joltz-1 + BAR affinity ranking vs. experimental `Kd`, `IC50`, and `ΔTm` | **Queued** |
| **Ablation & Efficiency** | Component attribution (`H_phys`, adaptive `ε`, frozen Joltz-1 prior) & single-GPU JAX runtime | **Queued** |

---

## Attribution & License

- **OncoVac Adenine** source code is released under the **Apache License 2.0** for open scientific research.
- Built upon open **Joltz-1** (MIT Boltz-1 JAX port), **AlphaGenome**, and **JAX** research frameworks. See [`NOTICE`](NOTICE) for attribution details.
- **Nucleobase Labs** is an independent computational biophysics laboratory and is not affiliated with or endorsed by Google DeepMind.

---

## Maintainers

Developed at **[Nucleobase Labs](https://nucleobaselabs.com)** (Malang, Indonesia) by:
- **Yovan F. Maitlen** — Founder, Chief AI/Physics Architect & Neural Structure Lead
- **Joachim Fidelis** — Stochastic Dynamics & Simulation Infrastructure Lead
- **Celine A. Rosali** — Thermodynamics, Pocket Analytics & Validation Lead
