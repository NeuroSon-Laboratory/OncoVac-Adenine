"""Differentiable 3D-oriented backbone/side-chain (U_geom) and pocket-targeted Metadynamics (U_rep) potentials."""
import jax
import jax.numpy as jnp

from cancerfold.physics.geometry import nonbonded_pair_mask, pairwise_dist, segment_dist


def virtual_cb(backbone: jnp.ndarray) -> jnp.ndarray:
    """Reconstructs 3D L-amino-acid pseudo-C_beta side-chain orientation from (N, CA, C)."""
    n_pos = backbone[:, 0, :]
    ca_pos = backbone[:, 1, :]
    c_pos = backbone[:, 2, :]
    b = ca_pos - n_pos
    c = c_pos - ca_pos
    a = jnp.cross(b, c)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca_pos


def u_geom_potential(
    backbone: jnp.ndarray,
    d_ref_ca: jnp.ndarray,
    phys_wt: jnp.ndarray,
    phys_mut: jnp.ndarray,
    weights: jnp.ndarray,
    theta: dict,
    chain_mask: jnp.ndarray = None,
) -> jnp.ndarray:
    n_pos = backbone[:, 0, :]
    ca_pos = backbone[:, 1, :]
    c_pos = backbone[:, 2, :]
    n = backbone.shape[0]

    # 1. Covalent 3D backbone triangle & peptide bonds (Engh & Huber 1991)
    d_n_ca = segment_dist(ca_pos, n_pos)
    d_ca_c = segment_dist(c_pos, ca_pos)
    d_n_c = segment_dist(c_pos, n_pos)
    u_intra = jnp.sum((d_n_ca - 1.458) ** 2 + (d_ca_c - 1.525) ** 2 + (d_n_c - 2.459) ** 2)

    cm = jnp.ones(n - 1, dtype=jnp.float32) if chain_mask is None else chain_mask
    d_c_n = segment_dist(n_pos[1:], c_pos[:-1])
    d_ca_n = segment_dist(n_pos[1:], ca_pos[:-1])
    d_c_ca = segment_dist(ca_pos[1:], c_pos[:-1])
    u_pep = jnp.sum(cm * ((d_c_n - 1.329) ** 2 + (d_ca_n - 2.425) ** 2 + (d_c_ca - 2.435) ** 2))
    u_bond = 0.5 * theta["k_bond"] * (u_intra + u_pep)

    # Upper-triangle non-bonded pairs (handles multi-chain interfaces via chain_mask)
    nb_mask = nonbonded_pair_mask(n, chain_mask=chain_mask)
    pair_w = weights[:, None] * weights[None, :]

    # 2. Gaussian ENM anchored to d_ref_ca, smoothly softened in pocket (Tirion 1996; Hinsen 1998)
    d_ca = pairwise_dist(ca_pos)
    enm_contact = jnp.exp(-0.5 * (d_ref_ca / theta["r_enm"]) ** 2)
    soften = jnp.clip(theta["soften"], 0.0, 0.95)
    elastic_k = theta["k_elastic"] * enm_contact * (1.0 - soften * pair_w)
    u_elastic = 0.5 * jnp.sum(nb_mask * elastic_k * (d_ca - d_ref_ca) ** 2)

    # 3. 3D side-chain steric clash: AF2 Overlap + Monotonic Soft-Core WCA Lennard-Jones
    cb = virtual_cb(backbone)
    d_cb = pairwise_dist(cb)
    vol = phys_mut[:, 2]
    d_min = theta["d_vdw"] + theta["k_vol"] * (vol[:, None] + vol[None, :])
    overlap = jax.nn.relu(d_min - d_cb)

    d_clamped = d_min - overlap
    soft_pad = (theta["sc_alpha"] * d_min) ** 2
    r_soft = jnp.sqrt(d_clamped ** 2 + soft_pad)
    r_cut = jnp.sqrt(d_min ** 2 + soft_pad)
    wca_core = ((r_cut / r_soft) ** 6 - 1.0) ** 2
    u_clash = jnp.sum(nb_mask * (0.5 * theta["k_clash"] * (overlap ** 2) + theta["eps_lj"] * wca_core))

    # 4. Non-covalent perturbation (Mutant - WT) in 3D cb space: Debye-Huckel, Hydrophobic (h>0), H-bond
    k_coulomb = 332.0637  # Universal Coulomb constant [kcal * A / (mol * e^2)]
    lambda_d = 3.04       # Physiological Debye length at ~150 mM salt (Honig & Nicholls 1995)
    d_elec = jnp.sqrt(d_cb ** 2 + theta["r_born"] ** 2)

    dq_pair = (phys_mut[:, 0:1] * phys_mut[:, 0:1].T) - (phys_wt[:, 0:1] * phys_wt[:, 0:1].T)
    dh_pair = (jax.nn.relu(phys_mut[:, 1:2]) * jax.nn.relu(phys_mut[:, 1:2]).T) - (
        jax.nn.relu(phys_wt[:, 1:2]) * jax.nn.relu(phys_wt[:, 1:2]).T
    )
    dhb_pair = (phys_mut[:, 3:4] * phys_mut[:, 3:4].T) - (phys_wt[:, 3:4] * phys_wt[:, 3:4].T)

    du_elec = (k_coulomb / theta["eps_r"]) * jnp.sum(
        nb_mask * pair_w * dq_pair * jnp.exp(-d_cb / lambda_d) / d_elec
    )
    du_hydro = -theta["k_hydro"] * jnp.sum(
        nb_mask * pair_w * dh_pair * jnp.exp(-0.5 * (d_cb / theta["r_hydro"]) ** 2)
    )
    du_hbond = -theta["k_hbond"] * jnp.sum(
        nb_mask * pair_w * dhb_pair * jnp.exp(-0.5 * (d_cb / theta["r_hbond"]) ** 2)
    )

    return u_bond + u_elastic + u_clash + du_elec + du_hydro + du_hbond


def u_rep_history(
    backbone: jnp.ndarray,
    history_cb_D: jnp.ndarray,
    history_mask: jnp.ndarray,
    weights: jnp.ndarray,
    theta: dict,
    chain_mask: jnp.ndarray = None,
) -> jnp.ndarray:
    """Pocket-targeted Metadynamics memory over H lightweight history windows via jax.lax.scan."""
    d_cb = pairwise_dist(virtual_cb(backbone))
    n = backbone.shape[0]
    nb_mask = nonbonded_pair_mask(n, chain_mask=chain_mask)
    pair_w = nb_mask * (weights[:, None] * weights[None, :])
    norm_w = jnp.maximum(jnp.sum(pair_w), 1e-8)

    def accum_hill(total_u: jnp.ndarray, item: tuple) -> tuple:
        d_hist, is_valid = item
        msd_pocket = jnp.sum(pair_w * (d_cb - d_hist) ** 2) / norm_w
        hill = theta["k_rep"] * is_valid * jnp.exp(-0.5 * msd_pocket / (theta["sigma_rep"] ** 2))
        return total_u + hill, None

    u_rep, _ = jax.lax.scan(accum_hill, jnp.array(0.0, dtype=jnp.float32), (history_cb_D, history_mask))
    return u_rep


def energy_and_force(
    backbone: jnp.ndarray,
    d_ref_ca: jnp.ndarray,
    phys_wt: jnp.ndarray,
    phys_mut: jnp.ndarray,
    weights: jnp.ndarray,
    history_cb_D: jnp.ndarray,
    history_mask: jnp.ndarray,
    theta: dict,
    chain_mask: jnp.ndarray = None,
) -> tuple:
    """Computes total physics energy and exact 3D forces F = -grad U on (N, CA, C)."""
    def total_energy(x_bb: jnp.ndarray) -> jnp.ndarray:
        return u_geom_potential(x_bb, d_ref_ca, phys_wt, phys_mut, weights, theta, chain_mask) + u_rep_history(
            x_bb, history_cb_D, history_mask, weights, theta, chain_mask
        )

    energy, grad_x = jax.value_and_grad(total_energy)(backbone)
    return energy, -grad_x


# References:
#   - Standard peptide covalent bond lengths & intra-residue N-CA-C geometry:
#     Engh & Huber (1991), Acta Crystallogr. A 47:392-400
#   - Tetrahedral pseudo-C_beta reconstruction & epsilon-regularized Euclidean distance:
#     Jumper et al. (2021), Nature 596:583-589 (AlphaFold 2, Supplementary Information)
#   - Elastic Network Model & Gaussian contact decay:
#     Tirion (1996), Phys. Rev. Lett. 77(9):1905-1908; Hinsen (1998), Proteins 33(3):417-429
#   - Repulsive Weeks-Chandler-Andersen potential & Soft-Core regularization:
#     Weeks, Chandler & Andersen (1971), J. Chem. Phys. 54(12):5237-5247
#     Beutler et al. (1994), Chem. Phys. Lett. 222(6):529-539
#   - Screened Debye-Huckel electrostatics & short-range Born regularization:
#     Honig & Nicholls (1995), Science 268(5214):1144-1149; Still et al. (1990), JACS 112(16):6127-6129
#   - Hydrophobic core attraction (h > 0) & side-chain H-bonding:
#     Kauzmann (1959), Adv. Protein Chem. 14:1-63; Kyte & Doolittle (1982), J. Mol. Biol. 157(1):105-132
#     McDonald & Thornton (1994), J. Mol. Biol. 238(5):777-793
#   - History-dependent repulsive Metadynamics potential (U_rep):
#     Laio & Parrinello (2002), PNAS 99(20):12562-12566
