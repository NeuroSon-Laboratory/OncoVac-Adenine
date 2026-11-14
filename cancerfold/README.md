<!-- Future banner logo: place your DeepMind-style banner in assets/cancerfold_banner.png and uncomment below -->
<!--
<p align="center">
  <img src="assets/cancerfold_banner.png" alt="CancerFold by NeuroSon" width="650"/>
</p>
-->

# CancerFold (by NeuroSon)

**Stochastic Physics-Guided Extension of AlphaFold for Modeling Somatic Cancer Mutation Ensembles in JAX**

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Built with JAX](https://img.shields.io/badge/Built%20with-JAX-orange.svg)](https://github.com/jax-ml/jax)
[![Status: v0.1.0](https://img.shields.io/badge/Status-v0.1.0%20Scaffold-yellow.svg)](#roadmap)

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

## Repository Structure

```text
cancerfold/
├── assets/                 # Logos and README architecture figures
├── benchmarks/             # Validation targets vs. experimental lab data (ΔΔG, PDB, IC50)
├── cancerfold/
│   ├── af2_core/           # Integrated AlphaFold 2 JAX modules & weight loaders
│   ├── analyzer/           # ΔRMSF, pocket accessibility, ligand scoring, & ΔΔG_eff
│   ├── mutation/           # Sequence mutation parser & active-site neighborhood masks
│   ├── physics/            # SE(3)-invariant D_ij, U_rep, U_geom, & Euler-Maruyama SDE
│   └── pipeline.py         # End-to-end unified runner
├── notebooks/              # Google Colab / Jupyter demos
└── tests/                  # Automated tests & toy ablation checks
```

---

## Target Validation Scope

CancerFold targets four mutation archetypes where static predictions fall short, benchmarked against published experimental measurements ($\Delta\Delta G_{\text{exp}}$, mutant PDB/NMR structures, and cellular $\text{IC}_{50}$ shifts):

1. **Cavity & Cryptic Pocket Mutations** (e.g., *TP53* Y220C)
2. **Gatekeeper / Steric Hindrance Mutations** (e.g., *EGFR* T790M)
3. **Allosteric / Conformational Switch Mutations** (e.g., *BRAF* V600E)
4. **Thermodynamic Folding Destabilization & Neutral Controls** (separating pathogenic shifts from benign mutations)

---

## Roadmap

- [x] **`v0.1.0-alpha` — Repository Architecture & Attribution Scaffold**
- [ ] **`v0.1.0` — Physics & Geometry Core:** $SE(3)$-invariant $D_{ij}(X)$, differentiable $U_{\text{rep}}$ & $U_{\text{geom}}$, Euler–Maruyama SDE (`jax.lax.scan` + `jax.vmap`), and toy double-well ablation tests.
- [ ] **`v0.2.0` — Integrated AF2 Engine:** Direct embedding into AlphaFold 2 `StructureModule` / `FoldIteration` with parallel ensemble generation.
- [ ] **`v0.3.0` — Lab Benchmark Validation & Colab Demo:** Quantitative correlation plots against experimental $\Delta\Delta G$, PDB shifts, and $\text{IC}_{50}$ benchmarks + 1-click Google Colab notebook.

---

## Attribution & License

- **CancerFold** source code is released under the **Apache License 2.0** for open, non-commercial scientific research and collaboration.
- This project integrates and builds upon **AlphaFold 2** by Google DeepMind (code under **Apache 2.0**, model parameters under **CC BY 4.0**). See [`NOTICE`](NOTICE) for full attribution details.

---

## Maintainers

Developed by **Yovan Fabiano Maitlen** & the **NeuroSon Computational Team**.
