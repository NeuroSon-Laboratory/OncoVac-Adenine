"""
OncoVac Adenine Gen 1 — Pillar 4B: Curated Clinical Drug Ranking (AutoDock Vina)
================================================================================
Focused blind ranking of curated clinical oncology drugs vs. FDA-approved
non-oncology negative decoys across the three Pillar 4 clinical pockets:

1. TP53 / p53 Y220C (PDB 2VUK, chain B, pocket P83):
   - Actives: PhiKan083, PK7088, Rezatapopt (PC14586)
   - Ref: Joerger et al., PNAS 2006; Dinge, H. L. (2026), Sp Thesis, Univ. Hasanuddin
2. ESR1 / ER-alpha LBD (PDB 3ERT, chain A, pocket OHT):
   - Actives: 4-Hydroxytamoxifen (SERM), Fulvestrant (SERD), Elacestrant (oral SERD)
   - Ref: Dinge, H. L. (2026), Sp Thesis, Univ. Hasanuddin; Robinson et al., Nat Genet 2013
3. KRAS G12C Switch-II Cryptic Pocket (PDB 6OIM, chain A, pocket MOV):
   - Actives: Sotorasib (AMG 510), Adagrasib (MRTX849)
   - Ref: Ostrem et al., Nature 2013

Negative Decoys (all 3 targets): Caffeine, Ibuprofen, Aspirin, Metformin.
All SMILES verified directly from PubChem; pocket centers extracted automatically
from the co-crystallized ligand coordinates in each RCSB PDB structure.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.request
from typing import Dict, List, Tuple

import numpy as np

# Exact PubChem-verified canonical SMILES (counterions stripped)
DECOYS: List[Dict[str, str]] = [
    {"name": "Caffeine",  "role": "Decoy", "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"},
    {"name": "Ibuprofen", "role": "Decoy", "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O"},
    {"name": "Aspirin",   "role": "Decoy", "smiles": "CC(=O)OC1=CC=CC=C1C(=O)O"},
    {"name": "Metformin", "role": "Decoy", "smiles": "CN(C)C(=N)N=C(N)N"},
]

TARGETS: List[Dict[str, object]] = [
    {
        "target_id": "TP53_Y220C",
        "title": "TP53 / p53 Y220C\n(PDB 2VUK, Crevice)",
        "pdb_id": "2VUK",
        "chain": "B",
        "lig_resname": "P83",
        "box_size": [20.0, 20.0, 20.0],
        "actives": [
            {"name": "PhiKan083",  "role": "Active", "smiles": "CCN1C2=C(C=C(C=C2)CNC)C3=CC=CC=C31"},
            {"name": "PK7088",     "role": "Active", "smiles": "CN1C=C(C(=N1)N2C=CC=C2)C3=CC=CC=C3"},
            {"name": "Rezatapopt", "role": "Active", "smiles": "CNC(=O)C1=CC(=C(C=C1)NCC#CC2=CC3=C(C=CC=C3N2CC(F)(F)F)NC4CCN(CC4F)C)OC"},
        ],
    },
    {
        "target_id": "ESR1_LBD",
        "title": "ESR1 / ERa LBD\n(PDB 3ERT, Helix-12)",
        "pdb_id": "3ERT",
        "chain": "A",
        "lig_resname": "OHT",
        "box_size": [22.0, 22.0, 22.0],
        "actives": [
            {"name": "4-OHT (Tamoxifen)", "role": "Active", "smiles": "CCC(=C(C1=CC=C(C=C1)O)C2=CC=C(C=C2)OCCN(C)C)C3=CC=CC=C3"},
            {"name": "Fulvestrant",       "role": "Active", "smiles": "CC12CCC3C(C1CCC2O)C(CC4=C3C=CC(=C4)O)CCCCCCCCCS(=O)CCCC(C(F)(F)F)(F)F"},
            {"name": "Elacestrant",       "role": "Active", "smiles": "CCNCCC1=CC=C(C=C1)CN(CC)C2=C(C=CC(=C2)OC)C3CCC4=C(C3)C=CC(=C4)O"},
        ],
    },
    {
        "target_id": "KRAS_G12C",
        "title": "KRAS G12C\n(PDB 6OIM, Switch-II)",
        "pdb_id": "6OIM",
        "chain": "A",
        "lig_resname": "MOV",
        "box_size": [22.0, 22.0, 22.0],
        "actives": [
            {"name": "Sotorasib", "role": "Active", "smiles": "CC1CN(CCN1C2=NC(=O)N(C3=NC(=C(C=C32)F)C4=C(C=CC=C4F)O)C5=C(C=CN=C5C(C)C)C)C(=O)C=C"},
            {"name": "Adagrasib", "role": "Active", "smiles": "CN1CCCC1COC2=NC3=C(CCN(C3)C4=CC=CC5=C4C(=CC=C5)Cl)C(=N2)N6CCN(C(C6)CC#N)C(=O)C(=C)F"},
        ],
    },
]

# AutoDock4 atom type mapping for standard protein heavy atoms (Vina ignores partial charges)
AROMATIC_ATOMS = {
    ("PHE", "CG"), ("PHE", "CD1"), ("PHE", "CD2"), ("PHE", "CE1"), ("PHE", "CE2"), ("PHE", "CZ"),
    ("TYR", "CG"), ("TYR", "CD1"), ("TYR", "CD2"), ("TYR", "CE1"), ("TYR", "CE2"), ("TYR", "CZ"),
    ("TRP", "CG"), ("TRP", "CD1"), ("TRP", "CD2"), ("TRP", "CE2"), ("TRP", "CE3"), ("TRP", "CZ2"), ("TRP", "CZ3"), ("TRP", "CH2"),
    ("HIS", "CG"), ("HIS", "CD2"), ("HIS", "CE1"),
}


def download_pdb(pdb_id: str, work_dir: str) -> str:
    os.makedirs(work_dir, exist_ok=True)
    path = os.path.join(work_dir, f"{pdb_id.upper()}.pdb")
    if not os.path.exists(path) or os.path.getsize(path) < 1000:
        url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
        urllib.request.urlretrieve(url, path)
    return path


def prep_receptor_pdbqt_and_center(
    pdb_path: str, chain: str, lig_resname: str, out_pdbqt: str
) -> List[float]:
    """Extracts clean chain protein atoms -> PDBQT and computes co-crystallized ligand centroid."""
    lig_coords: List[List[float]] = []
    pdbqt_lines: List[str] = []

    with open(pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = line[:6].strip()
            if rec == "HETATM" and line[21] == chain and line[17:20].strip() == lig_resname:
                alt = line[16]
                if alt in (" ", "A"):
                    lig_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
            elif rec == "ATOM" and line[21] == chain:
                alt = line[16]
                if alt not in (" ", "A"):
                    continue
                atom_name = line[12:16].strip()
                res_name = line[17:20].strip()
                elem = (line[76:78].strip() or atom_name[0]).upper()
                if elem == "H":
                    continue
                # Assign AutoDock4 atom type
                if elem == "C":
                    ad_type = "A" if (res_name, atom_name) in AROMATIC_ATOMS else "C"
                elif elem == "N":
                    ad_type = "NA" if (res_name == "HIS" and atom_name in ("ND1", "NE2")) else "N"
                elif elem == "O":
                    ad_type = "OA"
                elif elem == "S":
                    ad_type = "SA"
                else:
                    ad_type = elem[:2]
                # Standard PDBQT fixed-column format (cols 1-54 PDB, 55-66 occ/temp, 67-76 charge, 78-79 AD type)
                base = line[:54]
                pdbqt_lines.append(f"{base}  1.00  0.00     0.000 {ad_type:<2s}\n")

    if not lig_coords:
        raise ValueError(f"Ligand {lig_resname} on chain {chain} not found in {pdb_path}")

    with open(out_pdbqt, "w", encoding="utf-8") as out:
        out.writelines(pdbqt_lines)

    center = np.mean(np.array(lig_coords, dtype=np.float64), axis=0).tolist()
    return [round(c, 3) for c in center]


def smiles_to_pdbqt(smiles: str) -> str:
    """Converts canonical SMILES -> 3D ETKDGv3 + MMFF94 conformer -> Meeko PDBQT string."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    rc = AllChem.EmbedMolecule(mol, params)
    if rc != 0:
        AllChem.EmbedMolecule(mol, randomSeed=42, useRandomCoords=True)
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=300)

    prep = MoleculePreparation()
    mol_setups = prep.prepare(mol)
    pdbqt_str, is_ok, err_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
    if not is_ok:
        raise RuntimeError(f"Meeko failed: {err_msg}")
    return pdbqt_str


def compute_roc_auc(scores_lower_better: List[float], is_active: List[bool]) -> float:
    """Exact Mann-Whitney U ROC-AUC where more negative kcal/mol = stronger binder."""
    act = [s for s, a in zip(scores_lower_better, is_active) if a]
    dec = [s for s, a in zip(scores_lower_better, is_active) if not a]
    if not act or not dec:
        return float("nan")
    wins = 0.0
    for a in act:
        for d in dec:
            if a < d:
                wins += 1.0
            elif a == d:
                wins += 0.5
    return wins / (len(act) * len(dec))


def plot_pillar4b_figure(all_rows: List[Dict[str, object]], summary: List[Dict[str, object]], out_png: str) -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "legend.frameon": False,
    })
    NAVY, CRIMSON, GRAY = "#1B365D", "#B8332A", "#95A5A6"

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.6), dpi=300)
    for ax, tgt, summ in zip(axes, TARGETS, summary):
        tid = str(tgt["target_id"])
        rows = [r for r in all_rows if r["target_id"] == tid]
        # Sort strongest binder (most negative kcal/mol) at top
        rows = sorted(rows, key=lambda r: float(r["vina_affinity_kcal"]))
        names = [str(r["ligand"]) for r in rows]
        vals = [float(r["vina_affinity_kcal"]) for r in rows]
        cols = [NAVY if r["role"] == "Active" else GRAY for r in rows]

        y = np.arange(len(rows))
        bars = ax.barh(y, vals, height=0.62, color=cols, edgecolor="white", lw=0.6, zorder=3)
        ax.invert_yaxis()
        ax.axvline(0.0, color="#333333", lw=0.8)

        min_v = min(vals)
        for yi, val, r in zip(y, vals, rows):
            tag = f"{val:.2f}"
            ax.text(val - 0.18, yi, tag, va="center", ha="right", fontsize=7.8, fontweight="bold",
                    color=NAVY if r["role"] == "Active" else "#555555")

        ax.set_xlim(min_v - 1.8, 0.2)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8.2, fontweight="bold")
        ax.set_xlabel("AutoDock Vina Affinity (kcal/mol)", fontweight="bold")
        ax.set_title(
            f"{tgt['title']}\nROC-AUC = {summ['roc_auc']:.3f}  |  Δ = {summ['delta_active_vs_decoy_kcal']:+.2f} kcal/mol",
            loc="left", fontsize=9.2, fontweight="bold", pad=10,
        )
        ax.grid(axis="x", alpha=0.22, ls="--", zorder=0)

    # Shared clean legend
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor=NAVY, label="Clinical Oncology Active Drug"),
        Patch(facecolor=GRAY, label="FDA Non-Oncology Negative Decoy"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.05), fontsize=8.8, frameon=False)
    plt.tight_layout(w_pad=2.2)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.18)
    fig.savefig(os.path.splitext(out_png)[0] + ".pdf", bbox_inches="tight", pad_inches=0.18)
    plt.close(fig)
    print(f"[PLOT] Saved Pillar 4B figure -> {out_png}")


def main() -> None:
    from vina import Vina

    ap = argparse.ArgumentParser(description="OncoVac Adenine Pillar 4B Curated Clinical Drug Ranking")
    ap.add_argument("--out-dir", type=str, default="/content/pillar4b_docking")
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    work_dir = os.path.join(args.out_dir, "work")
    os.makedirs(work_dir, exist_ok=True)

    all_rows: List[Dict[str, object]] = []
    summary_rows: List[Dict[str, object]] = []

    print("=== OncoVac Adenine Gen 1 — Pillar 4B: Curated Clinical Drug Ranking (AutoDock Vina) ===")
    for tgt in TARGETS:
        t0 = time.time()
        tid = str(tgt["target_id"])
        pdb_path = download_pdb(str(tgt["pdb_id"]), work_dir)
        rec_pdbqt = os.path.join(work_dir, f"{tid}_receptor.pdbqt")
        center = prep_receptor_pdbqt_and_center(pdb_path, str(tgt["chain"]), str(tgt["lig_resname"]), rec_pdbqt)
        box = tgt["box_size"]
        print(f"\n[TARGET] {tid} ({tgt['pdb_id']}:{tgt['chain']}) | pocket={tgt['lig_resname']} center={center}")

        v = Vina(sf_name="vina", seed=args.seed, verbosity=0)
        v.set_receptor(rec_pdbqt)
        v.compute_vina_maps(center=center, box_size=box)

        ligands = list(tgt["actives"]) + DECOYS
        scores, flags = [], []
        for lig in ligands:
            lig_pdbqt = smiles_to_pdbqt(lig["smiles"])
            v.set_ligand_from_string(lig_pdbqt)
            v.dock(exhaustiveness=args.exhaustiveness, n_poses=5)
            aff = float(v.energies(n_poses=1)[0][0])
            scores.append(aff)
            flags.append(lig["role"] == "Active")
            all_rows.append({
                "target_id": tid,
                "pdb_id": tgt["pdb_id"],
                "ligand": lig["name"],
                "role": lig["role"],
                "vina_affinity_kcal": round(aff, 3),
            })
            print(f"  - {lig['name']:20s} [{lig['role']:6s}] : {aff:+.3f} kcal/mol")

        auc = compute_roc_auc(scores, flags)
        mean_act = float(np.mean([s for s, f in zip(scores, flags) if f]))
        mean_dec = float(np.mean([s for s, f in zip(scores, flags) if not f]))
        delta = mean_act - mean_dec
        summary_rows.append({
            "target_id": tid,
            "pdb_id": tgt["pdb_id"],
            "n_actives": sum(flags),
            "n_decoys": len(flags) - sum(flags),
            "mean_active_kcal": round(mean_act, 3),
            "mean_decoy_kcal": round(mean_dec, 3),
            "delta_active_vs_decoy_kcal": round(delta, 3),
            "roc_auc": round(auc, 4),
            "elapsed_s": round(time.time() - t0, 1),
        })
        print(f"[SUMMARY] {tid} | Active={mean_act:+.2f} vs Decoy={mean_dec:+.2f} (Δ={delta:+.2f} kcal/mol) | ROC-AUC={auc:.3f} | {time.time()-t0:.1f}s")

    with open(os.path.join(args.out_dir, "pillar4b_docking_results.json"), "w") as f:
        json.dump({"summary": summary_rows, "rankings": all_rows}, f, indent=2)
    with open(os.path.join(args.out_dir, "pillar4b_docking_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    plot_pillar4b_figure(all_rows, summary_rows, os.path.join(args.out_dir, "fig4b_curated_drug_ranking.png"))
    print(f"\n[DONE] Pillar 4B saved in {args.out_dir}")


if __name__ == "__main__":
    main()
