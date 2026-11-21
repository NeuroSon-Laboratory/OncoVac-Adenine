"""Full-state analytical effective Hamiltonian H_alpha(X) with adaptive dielectric & zero-core ENM."""
from typing import Dict, Optional, Tuple
import jax
import jax.numpy as jnp

from cancerfold.physics.geometry import pairwise_dist

EPS_CORE = 8.0       # Buried protein interior dielectric constant
EPS_WATER = 78.4     # Bulk physiological water dielectric constant
KAPPA_DEBYE = 0.125  # Inverse Debye screening length at 150 mM NaCl [1 / Angstrom]
K_COULOMB = 332.0637 # Electrostatic conversion constant [kcal * Angstrom / (mol * e^2)]
EPS_SAFE = 1e-12     # Differentiable norm epsilon


def _safe_norm(vec: jnp.ndarray, axis: int = -1, keepdims: bool = False) -> jnp.ndarray:
    return jnp.sqrt(jnp.sum(vec ** 2, axis=axis, keepdims=keepdims) + EPS_SAFE)


def c2_envelope(dist: jnp.ndarray, r_cut: float) -> jnp.ndarray:
    """C^2-smooth quintic cutoff envelope: f(0)=1, f(r_cut)=f'(r_cut)=f''(r_cut)=0."""
    u = jnp.clip(dist / r_cut, 0.0, 1.0)
    return 1.0 - 10.0 * (u ** 3) + 15.0 * (u ** 4) - 6.0 * (u ** 5)


def virtual_cb(backbone: jnp.ndarray) -> jnp.ndarray:
    """Reconstructs chiral L-amino acid pseudo-C_beta position from rigid (N, CA, C) frame."""
    n_pos, ca_pos, c_pos = backbone[:, 0, :], backbone[:, 1, :], backbone[:, 2, :]
    b = ca_pos - n_pos
    c = c_pos - ca_pos
    a = jnp.cross(b, c)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca_pos


def coordination_burial(
    d_cb: jnp.ndarray,
    residue_mask: jnp.ndarray,
    r_burial: float = 10.5,
    n0: float = 8.0,
    dn: float = 2.5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Live C^2-smooth coordination number n_i(X) and burial fraction b_i(X) in (0, 1).
    Drops smoothly toward 0 (solvent-exposed) as pocket residues separate past ~10.5-11 A.
    """
    m = residue_mask.astype(jnp.float32)
    off_diag = 1.0 - jnp.eye(d_cb.shape[0], dtype=jnp.float32)
    f_burial = c2_envelope(d_cb, r_cut=r_burial) * (m[:, None] * m[None, :]) * off_diag
    n_i = jnp.sum(f_burial, axis=-1)
    b_i = jax.nn.sigmoid((n_i - n0) / dn) * m
    return n_i, b_i


def adaptive_inv_dielectric(b_i: jnp.ndarray) -> jnp.ndarray:
    """Smooth local inverse dielectric 1/eps_eff(b_i, b_j) interpolating core <-> bulk water."""
    inv_w = 1.0 / EPS_WATER
    inv_c = 1.0 / EPS_CORE
    return inv_w + (b_i[:, None] * b_i[None, :]) * (inv_c - inv_w)


def u_peptide_bonds(
    backbone: jnp.ndarray,
    residue_mask: jnp.ndarray,
    chain_mask: Optional[jnp.ndarray],
    k_pep: float = 50.0,
) -> jnp.ndarray:
    """Inter-residue covalent peptide geometry (Engh-Huber: C_i-N_{i+1}, CA_i-N_{i+1}, C_i-CA_{i+1})."""
    m = residue_mask.astype(jnp.float32)
    adj_valid = m[:-1] * m[1:]
    if chain_mask is not None:
        same_chain = (chain_mask[:-1] == chain_mask[1:]).astype(jnp.float32)
        adj_valid = adj_valid * same_chain

    c_prev, ca_prev = backbone[:-1, 2, :], backbone[:-1, 1, :]
    n_next, ca_next = backbone[1:, 0, :], backbone[1:, 1, :]

    d_cn = _safe_norm(n_next - c_prev, axis=-1)
    d_can = _safe_norm(n_next - ca_prev, axis=-1)
    d_cca = _safe_norm(ca_next - c_prev, axis=-1)

    e_pep = (
        (d_cn - 1.329) ** 2
        + 0.5 * (d_can - 2.425) ** 2
        + 0.5 * (d_cca - 2.441) ** 2
    )
    return 0.5 * k_pep * jnp.sum(adj_valid * e_pep)


def u_shared_scaffold(
    backbone: jnp.ndarray,
    d_ref_ca: jnp.ndarray,
    weights: jnp.ndarray,
    residue_mask: jnp.ndarray,
    theta: Dict[str, float],
    chain_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """
    Shared topological scaffold U_shared(X) = U_pep + U_ENM.
    Inside the mutation pocket core (w_i * w_j -> 1), ENM stiffness drops to 0 (enm_soften=1.0),
    acting strictly as a far-field fulcrum without biasing pocket opening or reverse symmetry.
    """
    u_pep = u_peptide_bonds(backbone, residue_mask, chain_mask, k_pep=theta.get("k_pep", 50.0))

    ca_pos = backbone[:, 1, :]
    d_ca = pairwise_dist(ca_pos)
    m = residue_mask.astype(jnp.float32)
    off_diag = 1.0 - jnp.eye(d_ca.shape[0], dtype=jnp.float32)
    pair_valid = (m[:, None] * m[None, :]) * off_diag

    soften = theta.get("enm_soften", 1.0)
    enm_cut = c2_envelope(d_ref_ca, r_cut=theta.get("r_enm", 12.0))
    k_ij = theta.get("k_enm", 1.0) * enm_cut * jnp.clip(1.0 - soften * (weights[:, None] * weights[None, :]), 0.0, 1.0)
    u_enm = 0.25 * jnp.sum(pair_valid * k_ij * (d_ca - d_ref_ca) ** 2)
    return u_pep + u_enm


def u_noncov_state(
    backbone: jnp.ndarray,
    phys_state: jnp.ndarray,
    residue_mask: jnp.ndarray,
    theta: Dict[str, float],
    use_adaptive_dielectric: bool = True,
) -> jnp.ndarray:
    """
    Full-state non-covalent potential U_noncov(X; phys_alpha) for exact BAR mirror symmetry.
    phys_state columns: [0: volume_norm, 1: charge, 2: hydrophobicity, 3: hbond_donor, 4: hbond_acc].
    """
    m = residue_mask.astype(jnp.float32)
    n = backbone.shape[0]
    off_diag = 1.0 - jnp.eye(n, dtype=jnp.float32)
    pair_valid = (m[:, None] * m[None, :]) * off_diag

    cb_pos = virtual_cb(backbone)
    d_cb = pairwise_dist(cb_pos)

    vol = phys_state[:, 0]
    q = phys_state[:, 1]
    hydro = phys_state[:, 2]
    donor = phys_state[:, 3]
    acc = phys_state[:, 4]

    # 1. Soft-core steric clash (symmetric per state)
    r_vdw = theta.get("r_vdw_base", 3.2) + theta.get("r_vdw_scale", 1.2) * (vol[:, None] + vol[None, :])
    overlap = jnp.maximum(0.0, r_vdw - d_cb)
    u_clash = 0.5 * theta.get("k_clash", 2.0) * jnp.sum(pair_valid * (overlap ** 2))

    # 2. Adaptive-dielectric Debye-Huckel electrostatics (long-range cutoff 16 A)
    if use_adaptive_dielectric:
        _, b_i = coordination_burial(d_cb, residue_mask)
        inv_eps = adaptive_inv_dielectric(b_i)
    else:
        inv_eps = 1.0 / theta.get("eps_fixed", 20.0)

    env_long = c2_envelope(d_cb, r_cut=theta.get("r_long", 16.0))
    d_safe = jnp.sqrt(d_cb ** 2 + 1.0)
    u_elec = 0.5 * theta.get("k_elec", 1.0) * K_COULOMB * jnp.sum(
        pair_valid * env_long * inv_eps * (q[:, None] * q[None, :]) * jnp.exp(-KAPPA_DEBYE * d_safe) / d_safe
    )

    # 3. Short-range hydrophobic & H-bond interactions (short-range cutoff 10 A)
    env_short = c2_envelope(d_cb, r_cut=theta.get("r_short", 10.0))
    u_hydro = -0.5 * theta.get("k_hydro", 0.5) * jnp.sum(
        pair_valid * env_short * (hydro[:, None] * hydro[None, :]) * jnp.exp(-0.5 * ((d_cb - 5.5) / 2.0) ** 2)
    )
    hb_pair = donor[:, None] * acc[None, :] + acc[:, None] * donor[None, :]
    u_hbond = -0.5 * theta.get("k_hbond", 0.8) * jnp.sum(
        pair_valid * env_short * hb_pair * jnp.exp(-0.5 * ((d_cb - 4.5) / 1.2) ** 2)
    )

    return u_clash + u_elec + u_hydro + u_hbond


def u_metadynamics_bias(
    backbone: jnp.ndarray,
    weights: jnp.ndarray,
    history_cb_D: jnp.ndarray,
    history_mask: jnp.ndarray,
    theta: Dict[str, float],
) -> jnp.ndarray:
    """Repulsive Gaussian hills in pocket distance space (active ONLY in Exploration Mode, k_rep > 0)."""
    k_rep = theta.get("k_rep", 0.0)
    cb_pos = virtual_cb(backbone)
    d_cb = pairwise_dist(cb_pos)
    w_pair = weights[:, None] * weights[None, :]
    w_norm = jnp.sum(w_pair) + 1e-6

    diff = d_cb[None, :, :] - history_cb_D
    msd = jnp.sum(w_pair[None, :, :] * (diff ** 2), axis=(1, 2)) / w_norm
    sigma_meta = theta.get("sigma_meta", 0.5)
    hills = history_mask.astype(jnp.float32) * jnp.exp(-0.5 * msd / (sigma_meta ** 2))
    return k_rep * jnp.sum(hills)


def state_physics_energy_and_force(
    backbone: jnp.ndarray,
    d_ref_ca: jnp.ndarray,
    phys_state: jnp.ndarray,
    weights: jnp.ndarray,
    residue_mask: jnp.ndarray,
    history_cb_D: jnp.ndarray,
    history_mask: jnp.ndarray,
    theta: Dict[str, float],
    chain_mask: Optional[jnp.ndarray] = None,
    use_adaptive_dielectric: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes full-state analytical energy H_phys(X; alpha) and conservative forces -grad H_phys."""
    def _u_fn(x_bb: jnp.ndarray) -> jnp.ndarray:
        u_sh = u_shared_scaffold(x_bb, d_ref_ca, weights, residue_mask, theta, chain_mask=chain_mask)
        u_nc = u_noncov_state(x_bb, phys_state, residue_mask, theta, use_adaptive_dielectric=use_adaptive_dielectric)
        u_meta = u_metadynamics_bias(x_bb, weights, history_cb_D, history_mask, theta)
        return u_sh + u_nc + u_meta

    u_val, grad_x = jax.value_and_grad(_u_fn)(backbone)
    return u_val, -grad_x


# References:
#   - Standard covalent peptide bond geometry (C-N, CA-N, C-CA):
#     Engh & Huber (1991), Acta Cryst. A 47(4):392-400.
#   - Chiral L-amino acid tetrahedral pseudo-C_beta reconstruction:
#     Jumper et al. (2021), Nature 596:583-589 (AlphaFold 2 Supplementary Info).
#   - Elastic Network Model (ENM) far-field topological scaffold:
#     Tirion (1996), Phys. Rev. Lett. 77(9):1905-1908;
#     Atilgan et al. (2001), Biophys. J. 80(1):505-515.
#   - C^2-smooth quintic radial cutoff envelope for conservative MD forces:
#     Gasteiger, Groß & Günnemann (2020), ICLR (DimeNet).
#   - Smooth coordination number & solvent burial exposure proxy:
#     Miyazawa & Jernigan (1985), Macromolecules 18(3):534-552;
#     Hamelryck (2005), Proteins 59(1):38-48.
#   - Debye-Huckel screening & effective protein interior vs. bulk water dielectric:
#     Debye & Huckel (1923), Phys. Z. 24:185-206;
#     Schutz & Warshel (2001), Proteins 44(4):400-417;
#     Warshel et al. (2006), Biochim. Biophys. Acta 1764(11):1647-1676.
#   - History-dependent repulsive Gaussian hills (Metadynamics, Exploration Mode only):
#     Laio & Parrinello (2002), PNAS 99(20):12562-12566.
#   - Exact thermodynamic antisymmetry requirement in ddG predictors:
#     Pucci et al. (2018), Briefings in Bioinformatics 19(4):599-611.
