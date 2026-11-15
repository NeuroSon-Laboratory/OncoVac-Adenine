"""Multi-scale somatic mutation parser with dynamic thermal equilibrium and AlphaGenome coupling."""
import jax
import jax.numpy as jnp

AA_ORDER = "ARNDCQEGHILKMFPSTWYV"
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AA_ORDER)}


def parse(
    code: str,
    log2_fc: jnp.ndarray = jnp.array(0.0, dtype=jnp.float32),
    delta_psi: jnp.ndarray = jnp.array(0.0, dtype=jnp.float32),
    start_resnum: int = 1,
) -> dict:
    clean = code.strip().upper()
    wt_aa = clean[0]
    mut_aa = clean[-1]
    pos_1based = int(clean[1:-1])

    delta_phys = jnp.abs(AA_PHYSICS[AA_TO_IDX[mut_aa]] - AA_PHYSICS[AA_TO_IDX[wt_aa]])
    q_drive = 1.0 + jnp.sum(delta_phys) + jnp.abs(jnp.asarray(log2_fc, dtype=jnp.float32))

    t_ref = jnp.array(298.15, dtype=jnp.float32)   # IUPAC standard reference temperature (K)
    cp_water = jnp.array(0.018, dtype=jnp.float32) # Molar heat capacity of water (kcal/(mol*K), Privalov 1979)
    dt_span = jnp.array(12.0, dtype=jnp.float32)   # Characteristic ambient-to-physiological span (K)

    def temp_loss(t_k: jnp.ndarray) -> jnp.ndarray:
        heat_flux = q_drive * jnp.exp(-(t_k - t_ref) / dt_span)
        heat_sink = 0.5 * cp_water * (t_k - t_ref) ** 2
        return heat_flux + heat_sink

    grad_t = jax.grad(temp_loss)
    temp_k = jax.lax.fori_loop(0, 15, lambda _, t: t - 10.0 * grad_t(t), t_ref)

    r_gas = jnp.array(0.0019872, dtype=jnp.float32)
    kbt = r_gas * temp_k

    intact_splice = jnp.clip(1.0 + jnp.asarray(delta_psi, dtype=jnp.float32), 1e-6, 1.0)
    dosage = (2.0 ** jnp.asarray(log2_fc, dtype=jnp.float32)) * intact_splice * (t_ref / temp_k)
    ddg_dosage = -kbt * jnp.log(jnp.clip(dosage, 1e-6, 1e6))

    return {
        "wt": wt_aa,
        "mut": mut_aa,
        "pos_1based": pos_1based,
        "idx_0based": pos_1based - start_resnum,
        "temp_k": temp_k,
        "kbt": kbt,
        "dosage_ratio": dosage,
        "ddg_dosage": ddg_dosage,
    }


def point(wt_seq: str, code: str, start_resnum: int = 1) -> str:
    info = parse(code, start_resnum=start_resnum)
    idx = info["idx_0based"]
    if idx < 0 or idx >= len(wt_seq):
        raise ValueError(f"Position {info['pos_1based']} out of sequence bounds (length {len(wt_seq)})")
    if wt_seq[idx] != info["wt"]:
        raise ValueError(f"Expected '{info['wt']}' at position {info['pos_1based']}, got '{wt_seq[idx]}'")
    return wt_seq[:idx] + info["mut"] + wt_seq[idx + 1:]


# [charge_pH7.4, hydrophobicity, relative_volume, H_bonding]
AA_PHYSICS = jnp.array([
    [ 0.0,  1.8, 0.31, 0.0],  # A
    [ 1.0, -4.5, 0.85, 1.0],  # R
    [ 0.0, -3.5, 0.51, 1.0],  # N
    [-1.0, -3.5, 0.49, 1.0],  # D
    [ 0.0,  2.5, 0.44, 0.5],  # C
    [ 0.0, -3.5, 0.63, 1.0],  # Q
    [-1.0, -3.5, 0.61, 1.0],  # E
    [ 0.0, -0.4, 0.00, 0.0],  # G
    [ 0.5, -3.2, 0.68, 1.0],  # H
    [ 0.0,  4.5, 0.73, 0.0],  # I
    [ 0.0,  3.8, 0.73, 0.0],  # L
    [ 1.0, -3.9, 0.74, 1.0],  # K
    [ 0.0,  1.9, 0.71, 0.0],  # M
    [ 0.0,  2.8, 0.83, 0.0],  # F
    [ 0.0, -1.6, 0.49, 0.0],  # P
    [ 0.0, -0.8, 0.33, 1.0],  # S
    [ 0.0, -0.7, 0.48, 1.0],  # T
    [ 0.0, -0.9, 1.00, 1.0],  # W
    [ 0.0, -1.3, 0.86, 1.0],  # Y
    [ 0.0,  4.2, 0.59, 0.0],  # V
], dtype=jnp.float32)


def encode(seq: str) -> dict:
    clean = seq.strip().upper()
    for aa in clean:
        if aa not in AA_TO_IDX:
            raise ValueError(f"Unknown amino acid: '{aa}'")
    idx = jnp.array([AA_TO_IDX[aa] for aa in clean], dtype=jnp.int32)
    return {
        "idx": idx,
        "phys": AA_PHYSICS[idx],
    }


# References:
#   - Charge (pH 7.4): Nelson & Cox, Lehninger Principles of Biochemistry (His=+0.5 partial protonation)
#   - Hydropathy (exact): Kyte & Doolittle (1982), J. Mol. Biol. 157(1):105-132
#   - Normalized steric volume [0..1]: Zamyatnin (1972), Prog. Biophys. Mol. Biol. 24:107-123; Richards (1974)
#   - Side-chain H-bond capacity: McDonald & Thornton (1994), J. Mol. Biol. 238(5):777-793
#   - Molar heat capacity buffering & protein thermal stability: Privalov (1979), Adv. Protein Chem. 33:167-241
#   - Tumor microenvironment metabolic hyperthermia: Vaupel, Kallinowski & Okunieff (1989), Cancer Res. 49(23):6449-6465
#   - AlphaGenome regulatory variant expression & splicing (PSI): Avsec et al. (2025); Katz et al. (2010), Nature Methods
#   - Boltzmann chemical potential shift from dosage (dG = -RT ln(C_mut / C_wt)): Hill (1985), Springer
