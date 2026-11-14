# CancerFold (by NeuroSon)

**Stochastic Physics-Guided Extension of AlphaFold for Modeling Somatic Cancer Mutation Ensembles in JAX**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Built with JAX](https://img.shields.io/badge/Built%20with-JAX-orange.svg)](https://github.com/jax-ml/jax)
[![Status: Active Research](https://img.shields.io/badge/Status-v0.1%20Architecture-yellow.svg)](#roadmap)

---

## Why CancerFold?

Standard AlphaFold pipelines excel at predicting static equilibrium folds driven by Multiple Sequence Alignments (MSAs). However, single somatic point mutations in cancer often become "invisible" to static predictions because one amino acid substitution is overwhelmed by evolutionary conservation in the MSA—causing wild-type and mutant predictions to collapse onto nearly identical rigid structures.

**CancerFold** addresses this blind spot by directly integrating a **JAX-native Stochastic Differential Equation (SDE) and physics-guided sampling engine** into the AlphaFold 2 structural prediction pipeline:
- **AlphaFold 2 Core (*Structural Prior*):** Supplies evolutionary and stereochemical folding context (`StructureModule` / `FoldIteration`).
- **NeuroSon Physics Engine (*Stochastic Exploration*):** Augments coordinate updates with an $SE(3)$-invariant history-repulsive bias ($U_{\text{rep}}$), local stereochemical constraints ($U_{\text{geom}}$), and annealed thermal noise ($\sigma(t)\,dW_t$) compiled end-to-end via `jax.lax.scan` and `jax.vmap`.

Instead of returning a single rigid snapshot, CancerFold generates **conformational ensembles** to quantify mutation-induced pocket opening, steric gatekeeper clashes, local flexibility shifts ($\Delta\text{RMSF}$), and effective thermodynamic stability shifts ($\Delta\Delta G_{\text{eff}}$).

---

## Architecture Overview

```text
Input (WT Sequence + Somatic Mutation + Optional Ligand SMILES)
   │
   ▼
Evolutionary Trunk (Lightweight MSA & Pair Representations)
   │
   ▼
Integrated AF2 + Neural SDE Core (JAX: StructureModule + U_rep + U_geom + σ(t)dW_t)
   │
   ▼
Thermodynamics & Drug-Pocket Analyzer (ΔRMSF, Pocket Geometry, Ensemble Ligand Scoring, ΔΔG_eff)
   │
   ▼
Outputs: Multi-Model Ensembles (.pdb), Energy Landscape Plots, & Lab-Benchmark Metrics
```

---

## Target Validation Scope

CancerFold is designed to capture conformational and thermodynamic shifts across four mutation archetypes where static predictions fall short, evaluated directly against published experimental data ($\Delta\Delta G_{\text{exp}}$, mutant PDB/NMR structures, and cellular $\text{IC}_{50}$ shifts):

1. **Cavity & Cryptic Pocket Mutations** (e.g., *TP53* Y220C)
2. **Gatekeeper / Steric Hindrance Mutations** (e.g., *EGFR* T790M)
3. **Allosteric / Conformational Switch Mutations** (e.g., *BRAF* V600E)
4. **Thermodynamic Folding Destabilization & Neutral Controls** (positive vs. benign mutation separation)

> *Note: Validation plots and ablation tables comparing Standard AF2 vs. Stochastic Noise vs. Full CancerFold against experimental lab measurements will be populated upon completion of Milestone `v0.3.0`.*

---

## Planned Interface (CLI & Python API)

```bash
# Example planned workflow (Target: v0.2.0+)
cancerfold predict \
  --sequence target_protein.fasta \
  --mutation "Y220C" \
  --num-samples 200 \
  --output results/tp53_y220c/
```

---

## Roadmap

- [ ] **`v0.1.0` — Physics & Geometry Core:** $SE(3)$-invariant $D_{ij}(X)$, differentiable $U_{\text{rep}}$ & $U_{\text{geom}}$, Euler–Maruyama SDE (`jax.lax.scan` + `jax.vmap`), and toy double-well ablation tests.
- [ ] **`v0.2.0` — Integrated AF2 Engine:** Direct embedding into AlphaFold 2 `StructureModule` / `FoldIteration` with parallel ensemble generation.
- [ ] **`v0.3.0` — Lab Benchmark Validation & Colab Demo:** Quantitative comparison against experimental $\Delta\Delta G$, PDB conformational shifts, $\text{IC}_{50}$ benchmarks, and 1-click Google Colab notebook.

---

## Attribution & License

- **CancerFold** code is released under the **Apache License 2.0** for open, non-commercial scientific research and collaboration.
- This project integrates and builds upon **AlphaFold 2** by Google DeepMind (code under **Apache 2.0**, model parameters under **CC BY 4.0**). See [`NOTICE`](NOTICE) for full attribution details.

---

## Maintainers

Developed by **Yovan F. Maitlen, Joachim Fidelis, Celine A. Rosali** & the **NeuroSon Computational Team**.
