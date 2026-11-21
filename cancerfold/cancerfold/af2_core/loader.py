"""AlphaFold 2 parameter loader and model configuration helpers."""
"""PDB structure loader, static MAX_N padder, and multi-frame 3D trajectory exporter."""
from typing import Dict, List, Optional, Tuple
import numpy as np
import jax.numpy as jnp

AA3_TO_1: Dict[str, str] = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
AA1_TO_3: Dict[str, str] = {v: k for k, v in AA3_TO_1.items()}


def compute_chiral_cb_np(backbone_frame: np.ndarray) -> np.ndarray:
    """Reconstructs chiral L-amino acid pseudo-C_beta from (N, 3, 3) backbone [N, CA, C]."""
    n_pos = backbone_frame[:, 0, :]
    ca_pos = backbone_frame[:, 1, :]
    c_pos = backbone_frame[:, 2, :]
    b = ca_pos - n_pos
    c = c_pos - ca_pos
    a = np.cross(b, c)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca_pos


def load_pdb_backbone(
    pdb_path: str,
    chains: Optional[List[str]] = None,
) -> Dict[str, object]:
    """
    Parses a PDB file and extracts complete (N, CA, C) backbone frames.

    Returns:
      - backbone: (N, 3, 3) float32 array of [N, CA, C] coordinates [Angstrom]
      - sequence: 1-letter amino acid string of length N
      - chain_ids: List[str] of chain identifiers
      - chain_mask: (N,) int32 array mapping each chain to a unique integer ID
      - res_ids: List[Tuple[str, int]] of (chain_id, residue_number)
      - res_map: Dict[Tuple[str, int], int] mapping (chain_id, resseq) -> 0-based index
    """
    residues: Dict[Tuple[str, int], Dict[str, object]] = {}
    order: List[Tuple[str, int]] = []

    with open(pdb_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("ATOM  "):
                continue
            atom_name = line[12:16].strip()
            if atom_name not in ("N", "CA", "C"):
                continue
            altloc = line[16].strip()
            if altloc not in ("", "A"):
                continue

            resname = line[17:20].strip()
            if resname not in AA3_TO_1:
                continue

            chain_id = line[21].strip() or "A"
            if chains is not None and chain_id not in chains:
                continue

            resseq = int(line[22:26].strip())
            key = (chain_id, resseq)
            if key not in residues:
                residues[key] = {"aa": AA3_TO_1[resname], "atoms": {}}
                order.append(key)

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            residues[key]["atoms"][atom_name] = np.array([x, y, z], dtype=np.float32)

    coords_list: List[np.ndarray] = []
    seq_chars: List[str] = []
    chain_ids: List[str] = []
    res_ids: List[Tuple[str, int]] = []
    res_map: Dict[Tuple[str, int], int] = {}
    chain_to_int: Dict[str, int] = {}
    chain_ints: List[int] = []

    for key in order:
        atoms = residues[key]["atoms"]
        if "N" in atoms and "CA" in atoms and "C" in atoms:
            idx = len(coords_list)
            coords_list.append(np.stack([atoms["N"], atoms["CA"], atoms["C"]], axis=0))
            seq_chars.append(residues[key]["aa"])
            c_id = key[0]
            if c_id not in chain_to_int:
                chain_to_int[c_id] = len(chain_to_int)
            chain_ids.append(c_id)
            chain_ints.append(chain_to_int[c_id])
            res_ids.append(key)
            res_map[key] = idx

    if not coords_list:
        raise ValueError(f"No complete (N, CA, C) backbone residues found in {pdb_path}.")

    return {
        "backbone": jnp.asarray(np.stack(coords_list, axis=0), dtype=jnp.float32),
        "sequence": "".join(seq_chars),
        "chain_ids": chain_ids,
        "chain_mask": jnp.asarray(chain_ints, dtype=jnp.int32),
        "res_ids": res_ids,
        "res_map": res_map,
    }


def pad_structure(
    backbone: jnp.ndarray,
    chain_mask: jnp.ndarray,
    max_n: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pads backbone (N, 3, 3) and chain_mask (N,) to static MAX_N for JAX JIT stability."""
    n = backbone.shape[0]
    if n > max_n:
        raise ValueError(f"Structure length N={n} exceeds MAX_N={max_n}.")
    pad_len = max_n - n
    bb_pad = jnp.pad(backbone, ((0, pad_len), (0, 0), (0, 0)), mode="constant")
    cm_pad = jnp.pad(chain_mask, ((0, pad_len),), mode="constant", constant_values=-1)
    res_mask = jnp.concatenate(
        [jnp.ones((n,), dtype=jnp.float32), jnp.zeros((pad_len,), dtype=jnp.float32)]
    )
    return bb_pad, cm_pad, res_mask


def export_trajectory_pdb(
    output_path: str,
    trajectory_bb: np.ndarray,
    sequence: str,
    chain_ids: List[str],
    res_ids: List[Tuple[str, int]],
    b_factors: Optional[np.ndarray] = None,
    include_chiral_cb: bool = True,
) -> None:
    """
    Exports a (T, N, 3, 3) backbone trajectory to a multi-MODEL PDB file with optional
    chiral CB atoms and B-factor heatmaps for 3D rendering (PyMOL, ChimeraX, 3Dmol.js).
    """
    traj = np.asarray(trajectory_bb, dtype=np.float32)
    if traj.ndim == 3:
        traj = traj[None, ...]
    num_frames, n_res = traj.shape[0], len(sequence)

    if b_factors is not None:
        bf = np.asarray(b_factors, dtype=np.float32)
        if bf.ndim == 1:
            bf = np.broadcast_to(bf[None, :], (num_frames, n_res))
    else:
        bf = np.zeros((num_frames, n_res), dtype=np.float32)

    if include_chiral_cb:
        atom_names = ("N", "CA", "C", "CB")
        elements = ("N", "C", "C", "C")
    else:
        atom_names = ("N", "CA", "C")
        elements = ("N", "C", "C")

    with open(output_path, "w", encoding="utf-8") as f:
        for t in range(num_frames):
            if include_chiral_cb:
                cb_frame = compute_chiral_cb_np(traj[t])
                frame_atoms = np.concatenate([traj[t], cb_frame[:, None, :]], axis=1)
            else:
                frame_atoms = traj[t]

            f.write(f"MODEL     {t + 1:4d}\n")
            serial = 1
            for i in range(n_res):
                res3 = AA1_TO_3.get(sequence[i], "UNK")
                c_id, resseq = res_ids[i]
                b_val = float(np.clip(bf[t, i], -99.99, 999.99))
                for a_idx, (a_name, elem) in enumerate(zip(atom_names, elements)):
                    x, y, z = frame_atoms[i, a_idx]
                    f.write(
                        f"ATOM  {serial:5d}  {a_name:<3s} {res3:>3s} {c_id:1s}{resseq:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00{b_val:6.2f}          {elem:>2s}\n"
                    )
                    serial += 1
            f.write("ENDMDL\n")
        f.write("END\n")


# References:
#   - Standard PDB Atomic Coordinate Entry Format (ATOM, MODEL/ENDMDL, B-factor columns):
#     wwPDB Consortium (2012), Protein Data Bank Contents Guide, Version 3.30.
#   - Chiral L-amino acid tetrahedral pseudo-C_beta reconstruction:
#     Jumper et al. (2021), Nature 596:583-589 (AlphaFold 2 Supplementary Info).
#   - p53 Y220C cryptic pocket reference crystal structures (e.g., PDB 2J1X / 2VUK):
#     Joerger, Ang & Fersht (2006), PNAS 103(41):15056-15061.
