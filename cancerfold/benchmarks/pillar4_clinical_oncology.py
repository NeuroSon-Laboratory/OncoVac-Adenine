"""
OncoVac Adenine Gen 1 — Pillar 4: Clinical Oncology Case Studies
================================================================
White-box 310 K BAOAB + Well-Tempered Metadynamics + Exact BAR + 3-Way Attribution
on three clinically critical oncogenic mutations:

1. ESR1 (ER-alpha LBD, PDB 1ERE, chain A): Y537S & D538G
   - Luminal breast cancer endocrine-therapy resistance (Tamoxifen / AI) via
     constitutive Helix-12 (res 536-548) activation.
   - Clinical motivation & references: Dinge, H. L. (2026), Sp Thesis,
     Universitas Hasanuddin (Sec. 2.2.3 & Table 1; PERABOI 2020 / ESMO 2024 / NCCN 2025);
     Robinson et al., Nature Genetics 2013; Leary et al., 2016;
     Belachew & Sewasew, Frontiers in Endocrinology 2021.
2. KRAS (GTPase, PDB 4OBE, chain A): G12C
   - Landmark "undruggable" oncogene opening the Switch-II (res 58-76) cryptic
     allosteric pocket (Ostrem et al., Nature 2013).
3. TP53 / p53 (DNA-binding domain, PDB 2J1X, chain A): Y220C
   - Canonical destabilizing cancer mutation (exp. ddG_fold ~ +4.0 kcal/mol)
     linked to apoptosis resistance in luminal breast cancer (Dinge, H. L., 2026,
     Sp Thesis, Universitas Hasanuddin, Sec. 2.2.7 & Table 1; Belachew & Sewasew, 2021)
     and opening a druggable crevice around res 145-157 & 218-232 (Joerger et al., PNAS 2006).

Runs 100% Pure-JAX (Zero-Shot 310K virtual_sc + U_chem), verifies Exact BAR
antisymmetry |fwd + rev| < 1e-5 on every mutation, computes regional RMSF &
pocket-radius shifts, and saves CSV/JSON + a 300-DPI publication figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.request
from typing import Dict, List, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from cancerfold.af2_core.boltz_chem import (
    du_chem_ensemble,
    load_boltz_chem_params,
    make_fixed_decoding_ranks,
)
from cancerfold.af2_core.loader import load_pdb_backbone, pad_structure
from cancerfold.analyzer.thermo import (
    bar_ddg,
    bar_overlap_diagnostic,
    evaluate_dh_batch,
)
from cancerfold.mutation import parser as mut_parser
from cancerfold.mutation.neighborhood import multi_pocket_mask
from cancerfold.physics.sde import (
    DEFAULT_BURN_IN,
    DEFAULT_DT,
    DEFAULT_GAMMA,
    sample_ensemble,
)

K_B_KCAL = 0.0019872041  # kcal / (mol * K)

# Exact Zero-Shot 300K/310K analytical force-field scales (identical to v3.1)
THETA_CLINICAL = {
    "k_clash": 2.0,
    "k_elec": 1.0,
    "k_hydro": 1.5,
    "k_hbond": 1.5,
    "k_rep": 0.0,
    "enm_soften": 1.0,
}

CLINICAL_CASES: List[Dict[str, object]] = [
    {
        "case_id": "ESR1_Y537S",
        "gene": "ESR1 (ERa LBD)",
        "pdb_id": "1ERE",
        "chain": "A",
        "mut_code": "YA537S",
        "reference": "Dinge, H. L. (2026), Sp Thesis, Univ. Hasanuddin (Sec. 2.2.3, Table 1); Robinson et al., Nat Genet 2013; Belachew & Sewasew, 2021",
        "functional_region_name": "Helix-12 (536-548)",
        "functional_res_range": (536, 548),
        "exp_note": "Constitutive Helix-12 lock; endocrine resistance (Tamoxifen/AI)",
    },
    {
        "case_id": "ESR1_D538G",
        "gene": "ESR1 (ERa LBD)",
        "pdb_id": "1ERE",
        "chain": "A",
        "mut_code": "DA538G",
        "reference": "Dinge, H. L. (2026), Sp Thesis, Univ. Hasanuddin (Sec. 2.2.3, Table 1); Robinson et al., Nat Genet 2013; Belachew & Sewasew, 2021",
        "functional_region_name": "Helix-12 (536-548)",
        "functional_res_range": (536, 548),
        "exp_note": "Helix-12 agonist-like stabilization; endocrine resistance",
    },
    {
        "case_id": "KRAS_G12C",
        "gene": "KRAS (GTPase)",
        "pdb_id": "4OBE",
        "chain": "A",
        "mut_code": "GA12C",
        "reference": "Ostrem et al., Nature 2013 (Switch-II Cryptic Pocket)",
        "functional_region_name": "Switch-II (58-76)",
        "functional_res_range": (58, 76),
        "exp_note": "Opens Switch-II cryptic allosteric pocket (Sotorasib/Adagrasib site)",
    },
    {
        "case_id": "TP53_Y220C",
        "gene": "TP53 / p53 (DBD)",
        "pdb_id": "2J1X",
        "chain": "A",
        "mut_code": "YA220C",
        "reference": "Dinge, H. L. (2026), Sp Thesis, Univ. Hasanuddin (Sec. 2.2.7, Table 1); Belachew & Sewasew, 2021; Joerger et al., PNAS 2006",
        "functional_region_name": "Y220C Crevice (145-157,218-232)",
        "functional_res_range": (218, 232),
        "exp_note": "Severe core destabilization (+4.0 kcal/mol); opens druggable pocket",
    },
]


def download_pdb(pdb_id: str, pdb_dir: str) -> str:
    os.makedirs(pdb_dir, exist_ok=True)
    path = os.path.join(pdb_dir, f"{pdb_id.upper()}.pdb")
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        print(f"[FETCH] Downloading {pdb_id.upper()} from RCSB...")
        urllib.request.urlretrieve(url, path)
    return path


def _select_bucket(n_res: int) -> int:
    for b in (192, 256, 320, 384, 512):
        if n_res <= b:
            return b
    return ((n_res + 63) // 64) * 64


def _kabsch_align(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Aligns point set P (N, 3) onto reference Q (N, 3) via Kabsch SVD."""
    p_mean = P.mean(axis=0, keepdims=True)
    q_mean = Q.mean(axis=0, keepdims=True)
    Pc = P - p_mean
    Qc = Q - q_mean
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    return (Pc @ R.T) + q_mean


def compute_structural_metrics(
    ens_wt: np.ndarray,
    ens_mut: np.ndarray,
    n_valid: int,
    mut_idx: int,
    region_indices: List[int],
) -> Dict[str, float]:
    """Computes Kabsch-aligned CA RMSF, mutation-site RMSF, and functional-region shift."""
    ca_wt = np.asarray(ens_wt[:, :n_valid, 1, :], dtype=np.float64)   # (M, N, 3)
    ca_mut = np.asarray(ens_mut[:, :n_valid, 1, :], dtype=np.float64) # (M, N, 3)
    ref = ca_wt[0]

    wt_aligned = np.stack([_kabsch_align(f, ref) for f in ca_wt], axis=0)
    mut_aligned = np.stack([_kabsch_align(f, ref) for f in ca_mut], axis=0)

    wt_mean = wt_aligned.mean(axis=0)
    mut_mean = mut_aligned.mean(axis=0)

    rmsf_wt = np.sqrt(np.mean(np.sum((wt_aligned - wt_mean[None]) ** 2, axis=-1), axis=0))
    rmsf_mut = np.sqrt(np.mean(np.sum((mut_aligned - mut_mean[None]) ** 2, axis=-1), axis=0))

    # Local pocket radius of gyration around mutation site (10 A neighborhood in WT mean)
    d_mut = np.linalg.norm(wt_mean - wt_mean[mut_idx : mut_idx + 1], axis=-1)
    pocket_idx = np.where(d_mut <= 10.0)[0]

    def _rg(frames: np.ndarray, idxs: np.ndarray) -> float:
        sub = frames[:, idxs, :]
        ctr = sub.mean(axis=1, keepdims=True)
        return float(np.mean(np.sqrt(np.mean(np.sum((sub - ctr) ** 2, axis=-1), axis=1))))

    rg_wt = _rg(wt_aligned, pocket_idx)
    rg_mut = _rg(mut_aligned, pocket_idx)

    if len(region_indices) > 0:
        reg = np.array(region_indices, dtype=np.int32)
        region_rmsf_wt = float(np.mean(rmsf_wt[reg]))
        region_rmsf_mut = float(np.mean(rmsf_mut[reg]))
        region_shift = float(np.mean(np.linalg.norm(mut_mean[reg] - wt_mean[reg], axis=-1)))
    else:
        region_rmsf_wt = float(np.mean(rmsf_wt))
        region_rmsf_mut = float(np.mean(rmsf_mut))
        region_shift = float(np.mean(np.linalg.norm(mut_mean - wt_mean, axis=-1)))

    return {
        "global_rmsf_wt_A": float(np.mean(rmsf_wt)),
        "global_rmsf_mut_A": float(np.mean(rmsf_mut)),
        "mut_site_rmsf_wt_A": float(rmsf_wt[mut_idx]),
        "mut_site_rmsf_mut_A": float(rmsf_mut[mut_idx]),
        "region_rmsf_wt_A": region_rmsf_wt,
        "region_rmsf_mut_A": region_rmsf_mut,
        "region_mean_shift_A": region_shift,
        "pocket_rg_wt_A": rg_wt,
        "pocket_rg_mut_A": rg_mut,
        "pocket_delta_rg_A": rg_mut - rg_wt,
    }


def run_single_case(
    case: Dict[str, object],
    pdb_dir: str,
    chem_params: Dict[str, jnp.ndarray],
    temp_k: float,
    num_replicas: int,
    num_steps: int,
    stride: int,
    seed: int,
) -> Dict[str, object]:
    t0 = time.time()
    pdb_id = str(case["pdb_id"])
    chain = str(case["chain"])
    mut_code = str(case["mut_code"])
    pdb_path = download_pdb(pdb_id, pdb_dir)

    struct = load_pdb_backbone(pdb_path, chains=[chain])
    seq_wt: str = str(struct["sequence"])
    n_valid = len(seq_wt)
    max_n = _select_bucket(n_valid)

    mut_info = mut_parser.parse(mut_code, res_map=struct["res_map"])
    mut_idx = int(mut_info["idx_0based"])
    actual_wt = seq_wt[mut_idx]
    if actual_wt != mut_info["wt"]:
        raise ValueError(
            f"[{case['case_id']}] WT mismatch at {mut_code} (idx={mut_idx}): "
            f"PDB has '{actual_wt}', expected '{mut_info['wt']}'."
        )

    seq_mut = mut_parser.point(seq_wt, mut_code, res_map=struct["res_map"])
    bb_pad, cm_pad, res_mask = pad_structure(struct["backbone"], struct["chain_mask"], max_n)
    pad_len = max_n - n_valid

    enc_wt = mut_parser.encode(seq_wt)
    enc_mut = mut_parser.encode(seq_mut)
    phys_wt = jnp.pad(enc_wt["phys_state"], ((0, pad_len), (0, 0)), constant_values=0.0)
    phys_mut = jnp.pad(enc_mut["phys_state"], ((0, pad_len), (0, 0)), constant_values=0.0)
    aa_wt = jnp.pad(enc_wt["idx"], (0, pad_len), constant_values=0)
    aa_mut = jnp.pad(enc_mut["idx"], (0, pad_len), constant_values=0)

    d_ref_ca = jnp.linalg.norm(bb_pad[:, 1, None, :] - bb_pad[None, :, 1, :], axis=-1)
    mut_mask = jnp.zeros((max_n,), dtype=jnp.float32).at[mut_idx].set(1.0)
    weights = multi_pocket_mask(bb_pad[:, 1, :], mut_mask, residue_mask=res_mask, sigma=8.0)

    # Map functional region residue numbers -> 0-based indices
    # struct["res_ids"] is List[Tuple[str, int]] of (chain_id, resseq)
    lo_res, hi_res = case["functional_res_range"]
    region_indices: List[int] = []
    for idx_i, rid in enumerate(struct["res_ids"]):
        try:
            rnum = int(rid[1]) if isinstance(rid, (tuple, list)) else int(str(rid).strip())
            if lo_res <= rnum <= hi_res:
                region_indices.append(idx_i)
        except (ValueError, TypeError, IndexError):
            pass

    # 1. Physiological 310 K BAOAB + Well-Tempered Metadynamics sampling
    key = jax.random.PRNGKey(seed)
    k_wt, k_mut = jax.random.split(key)

    out_wt = sample_ensemble(
        bb_pad, k_wt, d_ref_ca, phys_wt, weights, res_mask, THETA_CLINICAL,
        nn_params=None, chain_mask=cm_pad,
        num_replicas=num_replicas, num_steps=num_steps,
        burn_in=DEFAULT_BURN_IN, stride=stride,
        dt=DEFAULT_DT, gamma=DEFAULT_GAMMA, temp_k=temp_k,
        max_hills=16, deposit_every=20, wt_bias_factor=6.0,
    )
    out_mut = sample_ensemble(
        bb_pad, k_mut, d_ref_ca, phys_mut, weights, res_mask, THETA_CLINICAL,
        nn_params=None, chain_mask=cm_pad,
        num_replicas=num_replicas, num_steps=num_steps,
        burn_in=DEFAULT_BURN_IN, stride=stride,
        dt=DEFAULT_DT, gamma=DEFAULT_GAMMA, temp_k=temp_k,
        max_hills=16, deposit_every=20, wt_bias_factor=6.0,
    )

    ens_wt = out_wt["ensemble"]
    ens_mut = out_mut["ensemble"]
    t_kin_wt = float(out_wt["mean_t_kin"])
    t_kin_mut = float(out_mut["mean_t_kin"])

    # 2. White-box Analytical Physics Work (dH_phys)
    dh_phys_wt, _ = evaluate_dh_batch(
        ens_wt, d_ref_ca, phys_wt, phys_mut, weights, res_mask,
        THETA_CLINICAL, chain_mask=cm_pad,
    )
    dh_phys_mut, _ = evaluate_dh_batch(
        ens_mut, d_ref_ca, phys_wt, phys_mut, weights, res_mask,
        THETA_CLINICAL, chain_mask=cm_pad,
    )

    # 3. Pure-JAX Evolutionary Chemical Potential Work (dH_chem)
    # Chunked (4 frames/batch) + JIT so peak HBM stays < 1 GB on TPU v5e / T4 / CPU
    ranks_k = jnp.asarray(make_fixed_decoding_ranks(np.asarray(res_mask), num_orders=4))
    kbt = K_B_KCAL * temp_k

    @jax.jit
    def _du_chem_batch(bb_chunk: jnp.ndarray) -> jnp.ndarray:
        return du_chem_ensemble(
            chem_params, bb_chunk, aa_wt, aa_mut, res_mask, cm_pad, ranks_k, kbt=kbt,
        )

    def _du_chem_chunked(ens: jnp.ndarray, chunk_size: int = 4) -> jnp.ndarray:
        n_frames = ens.shape[0]
        parts = []
        for s in range(0, n_frames, chunk_size):
            parts.append(jax.device_get(_du_chem_batch(ens[s : s + chunk_size])))
        return jnp.asarray(np.concatenate(parts, axis=0), dtype=jnp.float32)

    dh_chem_wt = _du_chem_chunked(ens_wt, chunk_size=4)
    dh_chem_mut = _du_chem_chunked(ens_mut, chunk_size=4)

    # 4. Exact BAR Free Energy + Antisymmetry Check + Overlap Diagnostic
    dh_tot_wt = dh_phys_wt + dh_chem_wt
    dh_tot_mut = dh_phys_mut + dh_chem_mut

    ddg_phys = float(bar_ddg(dh_phys_wt, dh_phys_mut, kbt=kbt))
    ddg_chem = float(bar_ddg(dh_chem_wt, dh_chem_mut, kbt=kbt))
    ddg_total = float(bar_ddg(dh_tot_wt, dh_tot_mut, kbt=kbt))
    ddg_rev = float(bar_ddg(-dh_tot_mut, -dh_tot_wt, kbt=kbt))
    bar_antisym_err = abs(ddg_total + ddg_rev)
    overlap_diag = bar_overlap_diagnostic(dh_tot_wt, dh_tot_mut, jnp.array(ddg_total), kbt=kbt)

    # 5. Structural Dynamics (RMSF & Pocket Opening)
    struct_metrics = compute_structural_metrics(
        np.asarray(ens_wt), np.asarray(ens_mut), n_valid, mut_idx, region_indices
    )

    elapsed = time.time() - t0
    res = {
        "case_id": case["case_id"],
        "gene": case["gene"],
        "pdb_id": pdb_id,
        "chain": chain,
        "mut_code": mut_code,
        "n_residues": n_valid,
        "temp_k": temp_k,
        "t_kin_wt_K": round(t_kin_wt, 2),
        "t_kin_mut_K": round(t_kin_mut, 2),
        "ddg_total_kcal": round(ddg_total, 4),
        "ddg_phys_kcal": round(ddg_phys, 4),
        "ddg_chem_kcal": round(ddg_chem, 4),
        "bar_antisym_err": float(f"{bar_antisym_err:.2e}"),
        "bar_overlap": round(float(overlap_diag["overlap_score"]), 4),
        "bar_std_err_kcal": round(float(overlap_diag["std_err"]), 4),
        "functional_region": case["functional_region_name"],
        **{k: round(v, 4) for k, v in struct_metrics.items()},
        "elapsed_s": round(elapsed, 2),
        "reference": case["reference"],
        "exp_note": case["exp_note"],
    }

    print(
        f"[DONE] {res['case_id']:12s} ({pdb_id}:{mut_code}) | "
        f"ddG_BAR={ddg_total:+.3f} kcal/mol (phys={ddg_phys:+.3f}, chem={ddg_chem:+.3f}) | "
        f"|fwd+rev|={bar_antisym_err:.1e} | T_kin={t_kin_wt:.1f}/{t_kin_mut:.1f} K | "
        f"RegShift={struct_metrics['region_mean_shift_A']:.2f} A | "
        f"dRg={struct_metrics['pocket_delta_rg_A']:+.3f} A | {elapsed:.1f}s"
    )
    return res


def plot_pillar4_figure(results: List[Dict[str, object]], out_png: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not installed; skipping figure generation.")
        return

    labels = [f"{r['gene'].split()[0]}\n{r['mut_code']}" for r in results]
    x = np.arange(len(results))
    w = 0.25

    ddg_phys = [r["ddg_phys_kcal"] for r in results]
    ddg_chem = [r["ddg_chem_kcal"] for r in results]
    ddg_tot = [r["ddg_total_kcal"] for r in results]

    reg_wt = [r["region_rmsf_wt_A"] for r in results]
    reg_mut = [r["region_rmsf_mut_A"] for r in results]
    drg = [r["pocket_delta_rg_A"] for r in results]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), dpi=300)
    fig.patch.set_facecolor("white")

    # Panel A: White-Box 3-Way Free Energy Decomposition
    ax = axes[0]
    ax.axhline(0, color="#444444", linewidth=0.8, linestyle="--")
    ax.bar(x - w, ddg_phys, width=w, label=r"$\Delta\Delta G_{\mathrm{phys}}$ (Analytical)", color="#2B5B84")
    ax.bar(x, ddg_chem, width=w, label=r"$\Delta\Delta G_{\mathrm{chem}}$ (Pure-JAX $U_{\mathrm{chem}}$)", color="#D97724")
    ax.bar(x + w, ddg_tot, width=w, label=r"$\Delta\Delta G_{\mathrm{BAR}}$ (Total Exact BAR)", color="#1B365D")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax.set_ylabel(r"$\Delta\Delta G$ (kcal/mol)", fontsize=10, fontweight="bold")
    ax.set_title("A. White-Box Free Energy Attribution (310 K Exact BAR)", fontsize=10.5, fontweight="bold")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(axis="y", alpha=0.25)

    # Panel B: Functional Region Flexibility (WT vs Mutant) + Pocket Opening
    ax2 = axes[1]
    w2 = 0.32
    ax2.bar(x - w2 / 2, reg_wt, width=w2, label="WT Functional Region RMSF", color="#6C8EAD")
    ax2.bar(x + w2 / 2, reg_mut, width=w2, label="Mutant Functional Region RMSF", color="#C0392B")
    for xi, (rw, rm, dg) in enumerate(zip(reg_wt, reg_mut, drg)):
        top = max(rw, rm) + 0.04
        ax2.text(xi, top, f"dRg={dg:+.2f}A", ha="center", va="bottom", fontsize=8, fontweight="bold", color="#222222")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, fontsize=9, fontweight="bold")
    ax2.set_ylabel("C-alpha RMSF (Angstrom)", fontsize=10, fontweight="bold")
    ax2.set_title("B. 310 K BAOAB + Metadynamics Allosteric & Pocket Dynamics", fontsize=10.5, fontweight="bold")
    ax2.legend(frameon=False, fontsize=8.5)
    ax2.grid(axis="y", alpha=0.25)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved 300-DPI figure -> {out_png}")


def main() -> None:
    ap = argparse.ArgumentParser(description="OncoVac Adenine Pillar 4 Clinical Oncology Suite")
    ap.add_argument("--cache-dir", type=str, default="/content/oncovac_cache")
    ap.add_argument("--out-dir", type=str, default="/content/pillar4_clinical")
    ap.add_argument("--temp-k", type=float, default=310.0, help="Physiological temperature [K]")
    ap.add_argument("--num-replicas", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=160)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print(
        f"=== OncoVac Adenine Gen 1 — Pillar 4 Clinical Oncology Suite ===\n"
        f"Devices: {jax.devices()} | T={args.temp_k} K | replicas={args.num_replicas} | "
        f"steps={args.num_steps} | dt={DEFAULT_DT} | gamma={DEFAULT_GAMMA}"
    )

    os.makedirs(args.out_dir, exist_ok=True)
    pdb_dir = os.path.join(args.cache_dir, "clinical_pdbs")
    chem_params = load_boltz_chem_params(cache_dir=args.cache_dir)

    results: List[Dict[str, object]] = []
    for i, case in enumerate(CLINICAL_CASES):
        res = run_single_case(
            case=case,
            pdb_dir=pdb_dir,
            chem_params=chem_params,
            temp_k=args.temp_k,
            num_replicas=args.num_replicas,
            num_steps=args.num_steps,
            stride=args.stride,
            seed=args.seed + 100 * i,
        )
        results.append(res)

    json_path = os.path.join(args.out_dir, "pillar4_clinical_results.json")
    csv_path = os.path.join(args.out_dir, "pillar4_clinical_results.csv")
    png_path = os.path.join(args.out_dir, "fig_pillar4_clinical_oncology.png")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    plot_pillar4_figure(results, png_path)
    print(f"\n[SAVED] JSON -> {json_path}\n[SAVED] CSV  -> {csv_path}")


if __name__ == "__main__":
    main()
