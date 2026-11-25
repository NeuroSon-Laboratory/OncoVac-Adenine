"""Full-state analytical effective Hamiltonian H_alpha(X) with virtual_sc, adaptive dielectric & zero-core ENM."""
from typing import Dict, Optional, Tuple
import jax
import jax.numpy as jnp

from cancerfold.physics.geometry import pairwise_dist

EPS_CORE = 8.0       # Buried protein interior dielectric constant
EPS_WATER = 78.4     # Bulk physiological water dielectric constant
KAPPA_DEBYE = 0.125  # Inverse Debye screening length at 150 mM NaCl (~8.0 A) [1 / Angstrom]
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


def virtual_sc(backbone: jnp.ndarray, l_sc: jnp.ndarray) -> jnp.ndarray:
    """
    Reconstructs SE(3)-equivariant amino-acid-specific side-chain centroid from rigid (N, CA, C):
        r_sc,i(X; alpha_i) = r_{CA,i} + L_sc(alpha_i) * u_hat_i(X)
    where u_hat_i is the chiral L-amino-acid tetrahedral unit vector along CA -> CB.
    Captures true side-chain reach (Gly 0.00 A .. Lys 4.10 A / Arg 4.65 A) with 0% crystal X_0 leakage.
    """
    ca_pos = backbone[:, 1, :]
    cb_pos = virtual_cb(backbone)
    u_vec = cb_pos - ca_pos
    u_hat = u_vec / _safe_norm(u_vec, axis=-1, keepdims=True)
    return ca_pos + l_sc[:, None] * u_hat


def coordination_burial(
    d_cb: jnp.ndarray,
    residue_mask: jnp.ndarray,
    r_burial: float = 10.5,
    n0: float = 8.0,
    dn: float = 2.5,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Live C^2-smooth coordination number n_i(X) and burial fraction b_i(X) in (0, 1)."""
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
    """Full 6-bond Engh & Huber (1991) covalent backbone geometry."""
    m = residue_mask.astype(jnp.float32)
    n_pos, ca_pos, c_pos = backbone[:, 0, :], backbone[:, 1, :], backbone[:, 2, :]
    d_nca = _safe_norm(ca_pos - n_pos, axis=-1)
    d_cac = _safe_norm(c_pos - ca_pos, axis=-1)
    d_nc = _safe_norm(c_pos - n_pos, axis=-1)
    e_intra = (
        (d_nca - 1.458) ** 2
        + (d_cac - 1.525) ** 2
        + 0.5 * (d_nc - 2.462) ** 2
    )
    adj_valid = m[:-1] * m[1:]
    if chain_mask is not None:
        same_chain = ((chain_mask[:-1] == chain_mask[1:]) & (chain_mask[:-1] >= 0)).astype(jnp.float32)
        adj_valid = adj_valid * same_chain

    c_prev, ca_prev = backbone[:-1, 2, :], backbone[:-1, 1, :]
    n_next, ca_next = backbone[1:, 0, :], backbone[1:, 1, :]
    d_cn = _safe_norm(n_next - c_prev, axis=-1)
    d_can = _safe_norm(n_next - ca_prev, axis=-1)
    d_cca = _safe_norm(ca_next - c_prev, axis=-1)
    e_inter = (
        (d_cn - 1.329) ** 2
        + 0.5 * (d_can - 2.425) ** 2
        + 0.5 * (d_cca - 2.441) ** 2
    )
    return 0.5 * k_pep * (jnp.sum(m * e_intra) + jnp.sum(adj_valid * e_inter))


def u_shared_scaffold(
    backbone: jnp.ndarray,
    d_ref_ca: jnp.ndarray,
    weights: jnp.ndarray,
    residue_mask: jnp.ndarray,
    theta: Dict[str, float],
    chain_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Shared topological scaffold U_shared(X) = U_pep + U_ENM (zero-core inside pocket)."""
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
    chain_mask: Optional[jnp.ndarray] = None,
    use_adaptive_dielectric: bool = True,
) -> jnp.ndarray:
    """
    Full-state non-covalent potential U_noncov(X; phys_alpha) for exact BAR mirror symmetry.
    Accepts:
      - 6-column phys_state: [vol, charge, hydro, donor, acc, l_sc] -> uses AA-specific virtual_sc
      - 5-column phys_state: [vol, charge, hydro, donor, acc] -> legacy virtual_cb fallback
      - 4-column parser.py phys: [charge, hydro_KD, vol, hbond] -> legacy fallback
    """
    m = residue_mask.astype(jnp.float32)
    n = backbone.shape[0]
    off_diag = 1.0 - jnp.eye(n, dtype=jnp.float32)

    if chain_mask is not None:
        same_chain = ((chain_mask[:-1] == chain_mask[1:]) & (chain_mask[:-1] >= 0)).astype(jnp.float32) * (m[:-1] * m[1:])
    else:
        same_chain = m[:-1] * m[1:]
    adj_cov = jnp.diag(same_chain, k=1) + jnp.diag(same_chain, k=-1)
    pair_valid = (m[:, None] * m[None, :]) * off_diag * (1.0 - adj_cov)

    cb_pos = virtual_cb(backbone)
    d_cb = pairwise_dist(cb_pos)

    use_sc = False
    if phys_state.shape[-1] >= 6:
        vol = phys_state[:, 0]
        q = phys_state[:, 1]
        hydro = phys_state[:, 2]
        donor = phys_state[:, 3]
        acc = phys_state[:, 4]
        l_sc = phys_state[:, 5]
        use_sc = True
    elif phys_state.shape[-1] == 5:
        vol = phys_state[:, 0]
        q = phys_state[:, 1]
        hydro = phys_state[:, 2]
        donor = phys_state[:, 3]
        acc = phys_state[:, 4]
    elif phys_state.shape[-1] == 4:
        q = phys_state[:, 0]
        hydro = phys_state[:, 1] / 4.5
        vol = phys_state[:, 2]
        donor = phys_state[:, 3]
        acc = phys_state[:, 3]
    else:
        raise ValueError(f"phys_state must have 4, 5, or 6 feature columns, got shape {phys_state.shape}.")

    if use_sc:
        sc_pos = virtual_sc(backbone, l_sc)
        d_int = pairwise_dist(sc_pos)
        d_elec_floor_sq = 2.25  # 1.5 A smooth soft-core floor for side-chain centroids
        r_hydro_opt = 4.5
        r_hbond_opt = 3.8
    else:
        d_int = d_cb
        d_elec_floor_sq = 1.0
        r_hydro_opt = 5.5
        r_hbond_opt = 4.5

    # 1. Backbone-invariant steric clash on C_beta (prevents artificial rigid-chi1 clash)
    r_vdw = theta.get("r_vdw_base", 2.4) + theta.get("r_vdw_scale", 0.6) * (vol[:, None] + vol[None, :])
    overlap = jnp.maximum(0.0, r_vdw - d_cb)
    u_clash = 0.25 * theta.get("k_clash", 2.0) * jnp.sum(pair_valid * (overlap ** 2))

    # 2. Adaptive-dielectric Debye-Huckel electrostatics on side-chain centroids (cutoff 16 A)
    if use_adaptive_dielectric:
        _, b_i = coordination_burial(d_cb, residue_mask)
        inv_eps = adaptive_inv_dielectric(b_i)
    else:
        inv_eps = 1.0 / theta.get("eps_fixed", 20.0)

    env_long = c2_envelope(d_int, r_cut=theta.get("r_long", 16.0))
    d_safe = jnp.sqrt(d_int ** 2 + d_elec_floor_sq)
    u_elec = 0.5 * theta.get("k_elec", 1.0) * K_COULOMB * jnp.sum(
        pair_valid * env_long * inv_eps * (q[:, None] * q[None, :]) * jnp.exp(-KAPPA_DEBYE * d_safe) / d_safe
    )

    # 3. Short-range hydrophobic (h > 0 via ReLU) & H-bond interactions on side-chain centroids (cutoff 10 A)
    env_short = c2_envelope(d_int, r_cut=theta.get("r_short", 10.0))
    h_pos = jax.nn.relu(hydro)
    u_hydro = -0.5 * theta.get("k_hydro", 1.5) * jnp.sum(
        pair_valid * env_short * (h_pos[:, None] * h_pos[None, :]) * jnp.exp(-0.5 * ((d_int - r_hydro_opt) / 2.0) ** 2)
    )
    hb_pair = donor[:, None] * acc[None, :] + acc[:, None] * donor[None, :]
    u_hbond = -0.5 * theta.get("k_hbond", 1.5) * jnp.sum(
        pair_valid * env_short * hb_pair * jnp.exp(-0.5 * ((d_int - r_hbond_opt) / 1.2) ** 2)
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
        u_nc = u_noncov_state(
            x_bb, phys_state, residue_mask, theta,
            chain_mask=chain_mask, use_adaptive_dielectric=use_adaptive_dielectric,
        )
        u_meta = u_metadynamics_bias(x_bb, weights, history_cb_D, history_mask, theta)
        return u_sh + u_nc + u_meta

    u_val, grad_x = jax.value_and_grad(_u_fn)(backbone)
    return u_val, -grad_x


# References:
#   - Standard 6-bond covalent backbone & peptide geometry: Engh & Huber (1991), Acta Cryst. A 47(4):392-400.
#   - Chiral L-amino acid tetrahedral pseudo-C_beta: Jumper et al. (2021), Nature 596:583-589.
#   - Amino-acid-specific side-chain centroids (UNRES): Liwo et al. (1997), J. Comput. Chem. 18(7):849-873.
#   - Elastic Network Model (ENM): Tirion (1996), Phys. Rev. Lett. 77(9):1905-1908.
#   - C^2-smooth quintic cutoff envelope: Gasteiger, Gross & Gunnemann (2020), ICLR (DimeNet).
#   - Debye-Huckel screening & adaptive dielectric: Schutz & Warshel (2001), Proteins 44(4):400-417.
#   - Well-Tempered Metadynamics: Laio & Parrinello (2002), PNAS 99(20):12562-12566.
