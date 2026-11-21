"""Differentiable physics-embedded single/pair trunk with non-Abelian projection & adaptive refresh."""
from typing import Dict, Optional, Tuple
import jax
import jax.numpy as jnp

from cancerfold.physics.geometry import center, nonbonded_pair_mask, pairwise_dist
from cancerfold.physics.potentials import virtual_cb

NUM_AA = 20          # Canonical amino acids
NUM_PHYS_FEATS = 4   # [charge, hydrophobicity, volume, hbond] from parser.py
NUM_PHYS_PAIR = 5    # [elec, hydro, hbond, steric, enm_contact]
R_GAS = 0.0019872    # Universal gas constant [kcal / (mol * K)]
K_COULOMB = 332.0637 # Universal Coulomb constant [kcal * A / (mol * e^2)]
LAMBDA_D = 3.04      # Physiological Debye length at ~150 mM salt (Honig & Nicholls 1995)
BLOCK_K = 2          # Smallest non-Abelian real matrix algebra gl(2, R)
EPS_DIST = 1e-8      # Regularization against NaN gradients
EPS_NORM = 1e-5      # LayerNorm numerical stability


def _layer_norm(x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + EPS_NORM) + bias


def _glorot_uniform(key: jax.Array, shape: Tuple[int, ...]) -> jnp.ndarray:
    limit = jnp.sqrt(6.0 / (shape[-2] + shape[-1]))
    return jax.random.uniform(key, shape, dtype=jnp.float32, minval=-limit, maxval=limit)


def init_trunk_params(
    prng_key: jax.Array,
    c_af2_s: int = 384,  # Matches both AF2 Evoformer and Joltz-1 Pairformer token_s=384
    c_af2_z: int = 128,  # Matches both AF2 Evoformer and Joltz-1 Pairformer token_z=128
    c_s: int = 128,
    c_z: int = 64,
    num_rbf: int = 64,   # Standard 64-bin distogram over [2.0, 22.0] A
) -> Dict[str, jnp.ndarray]:
    """Initializes learnable trunk parameters (single-weight non-Abelian s2p requires c_z % 4 == 0)."""
    if c_z % (BLOCK_K * BLOCK_K) != 0:
        raise ValueError(f"c_z={c_z} must be divisible by {BLOCK_K * BLOCK_K} for gl(2, R) blocks.")

    k1, k2, k3 = jax.random.split(prng_key, 3)
    d_single_in = c_af2_s + NUM_AA + NUM_PHYS_FEATS + 1
    d_pair_in = c_af2_z + num_rbf + NUM_PHYS_PAIR + 2

    return {
        "w_single_in": _glorot_uniform(k1, (d_single_in, c_s)),
        "b_single_in": jnp.zeros((c_s,), dtype=jnp.float32),
        "ln_single_scale": jnp.ones((c_s,), dtype=jnp.float32),
        "ln_single_bias": jnp.zeros((c_s,), dtype=jnp.float32),
        "w_pair_in": _glorot_uniform(k2, (d_pair_in, c_z)),
        "b_pair_in": jnp.zeros((c_z,), dtype=jnp.float32),
        "ln_pair_scale": jnp.ones((c_z,), dtype=jnp.float32),
        "ln_pair_bias": jnp.zeros((c_z,), dtype=jnp.float32),
        # Single non-Abelian projection matrix (50% fewer params than w_left + w_right)
        "w_s2p": _glorot_uniform(k3, (c_s, c_z)),
    }


def distogram_rbf(d_ca: jnp.ndarray, num_rbf: int = 64) -> jnp.ndarray:
    """Fixed SE(3)-invariant Gaussian RBF distogram over [2.0, 22.0] A."""
    centers = jnp.linspace(2.0, 22.0, num_rbf, dtype=jnp.float32)
    width = (22.0 - 2.0) / max(num_rbf - 1, 1)
    diff = d_ca[..., None] - centers[None, None, :]
    return jnp.exp(-0.5 * (diff / width) ** 2)


def physics_pair_kernels(
    phys: jnp.ndarray,
    d_ca: jnp.ndarray,
    d_cb: jnp.ndarray,
    temp_k: float,
    theta: dict,
    chain_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """Dimensionless physical pair kernels (arcsinh(beta * U_ij)) matching potentials.py."""
    n = phys.shape[0]
    nb_upper = nonbonded_pair_mask(n, chain_mask=chain_mask)
    full_nb_mask = nb_upper + nb_upper.T

    beta = 1.0 / (R_GAS * temp_k)
    q = phys[:, 0:1]
    h = jax.nn.relu(phys[:, 1:2])
    vol = phys[:, 2]
    hb = phys[:, 3:4]

    # 1-4. Side-chain interactions in 3D C_beta* space (dimensionless beta * U_ij)
    d_elec = jnp.sqrt(d_cb ** 2 + theta["r_born"] ** 2)
    phi_elec = beta * (K_COULOMB / theta["eps_r"]) * (q * q.T) * jnp.exp(-d_cb / LAMBDA_D) / d_elec
    phi_hydro = -beta * theta["k_hydro"] * (h * h.T) * jnp.exp(-0.5 * (d_cb / theta["r_hydro"]) ** 2)
    phi_hbond = -beta * theta["k_hbond"] * (hb * hb.T) * jnp.exp(-0.5 * (d_cb / theta["r_hbond"]) ** 2)

    d_min = theta["d_vdw"] + theta["k_vol"] * (vol[:, None] + vol[None, :])
    overlap = jax.nn.relu(d_min - d_cb)
    phi_steric = beta * 0.5 * theta["k_clash"] * (overlap ** 2)

    # Smooth sign-preserving compression (linear for |beta*U| << 1, log-like for steric spikes)
    phi_elec = jnp.arcsinh(phi_elec)
    phi_hydro = jnp.arcsinh(phi_hydro)
    phi_hbond = jnp.arcsinh(phi_hbond)
    phi_steric = jnp.arcsinh(phi_steric)

    # 5. Backbone ENM contact kernel in C_alpha space (already dimensionless in [0, 1])
    phi_enm = jnp.exp(-0.5 * (d_ca / theta["r_enm"]) ** 2)

    kernels = jnp.stack([phi_elec, phi_hydro, phi_hbond, phi_steric, phi_enm], axis=-1)
    return kernels * full_nb_mask[..., None]


def build_representations(
    idx: jnp.ndarray,
    phys: jnp.ndarray,
    coords_ca: jnp.ndarray,
    weights: jnp.ndarray,
    af2_single: Optional[jnp.ndarray],
    af2_pair: Optional[jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    theta: dict,
    temp_k: float = 310.15,
    backbone_n_ca_c: Optional[jnp.ndarray] = None,
    chain_mask: Optional[jnp.ndarray] = None,
    num_rbf: int = 64,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Builds physics-embedded single (N, c_s) and non-Abelian pair (N, N, c_z) features."""
    n = idx.shape[0]
    c_z = params["w_s2p"].shape[1]
    num_blocks = c_z // (BLOCK_K * BLOCK_K)
    c_af2_s = params["w_single_in"].shape[0] - (NUM_AA + NUM_PHYS_FEATS + 1)
    c_af2_z = params["w_pair_in"].shape[0] - (num_rbf + NUM_PHYS_PAIR + 2)

    if af2_single is None:
        af2_single = jnp.zeros((n, c_af2_s), dtype=jnp.float32)
    if af2_pair is None:
        af2_pair = jnp.zeros((n, n, c_af2_z), dtype=jnp.float32)

    # SE(3)-centered distances: C_alpha for backbone/ENM, pseudo-C_beta* for side-chains
    if backbone_n_ca_c is not None:
        bb_c = center(backbone_n_ca_c)
        d_ca = pairwise_dist(bb_c[:, 1, :])
        d_cb = pairwise_dist(virtual_cb(bb_c))
    else:
        d_ca = pairwise_dist(center(coords_ca))
        d_cb = d_ca

    # 1. Single representation
    aa_onehot = jax.nn.one_hot(idx, NUM_AA, dtype=jnp.float32)
    single_raw = jnp.concatenate(
        [af2_single.astype(jnp.float32), aa_onehot, phys.astype(jnp.float32), weights[:, None]],
        axis=-1,
    )
    single_proj = jnp.einsum("ni,ij->nj", single_raw, params["w_single_in"]) + params["b_single_in"]
    single = _layer_norm(jax.nn.silu(single_proj), params["ln_single_scale"], params["ln_single_bias"])

    # 2. Pair representation with dimensionless physical kernels
    rbf_ij = distogram_rbf(d_ca, num_rbf=num_rbf)
    phys_ij = physics_pair_kernels(phys, d_ca, d_cb, temp_k, theta, chain_mask=chain_mask)
    w_prod = (weights[:, None] * weights[None, :])[..., None]
    w_mean = (0.5 * (weights[:, None] + weights[None, :]))[..., None]

    pair_raw = jnp.concatenate(
        [af2_pair.astype(jnp.float32), rbf_ij, phys_ij, w_prod, w_mean],
        axis=-1,
    )
    pair_proj = jnp.einsum("ijk,kl->ijl", pair_raw, params["w_pair_in"]) + params["b_pair_in"]

    # 3. Single-weight non-Abelian gl(2, R) injection: (I + M_i)(I + M_j) - I
    s_mat = jnp.einsum("ni,ij->nj", single, params["w_s2p"]).reshape(n, num_blocks, BLOCK_K, BLOCK_K)
    non_abelian = jnp.einsum("ibpq,jbqr->ijbpr", s_mat, s_mat) / jnp.sqrt(float(BLOCK_K))
    s2p = (s_mat[:, None, ...] + s_mat[None, :, ...] + non_abelian).reshape(n, n, c_z)

    pair = _layer_norm(jax.nn.silu(pair_proj + s2p), params["ln_pair_scale"], params["ln_pair_bias"])
    return single, pair


def pad_features(
    single: jnp.ndarray,
    pair: jnp.ndarray,
    max_n: int,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Static padding to MAX_N with explicit residue_mask for zero-retrace jax.jit."""
    n = single.shape[0]
    if n > max_n:
        raise ValueError(f"Sequence length N={n} exceeds static MAX_N={max_n}.")

    pad_n = max_n - n
    single_pad = jnp.pad(single, ((0, pad_n), (0, 0)), mode="constant", constant_values=0.0)
    pair_pad = jnp.pad(pair, ((0, pad_n), (0, pad_n), (0, 0)), mode="constant", constant_values=0.0)

    residue_mask = (jnp.arange(max_n) < n).astype(jnp.float32)
    pair_mask = residue_mask[:, None] * residue_mask[None, :]
    return single_pad * residue_mask[:, None], pair_pad * pair_mask[..., None], residue_mask


def run_trunk(
    idx: jnp.ndarray,
    phys: jnp.ndarray,
    coords_ca: jnp.ndarray,
    weights: jnp.ndarray,
    af2_single: Optional[jnp.ndarray],
    af2_pair: Optional[jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    theta: dict,
    max_n: int,
    temp_k: float = 310.15,
    backbone_n_ca_c: Optional[jnp.ndarray] = None,
    chain_mask: Optional[jnp.ndarray] = None,
    num_rbf: int = 64,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Static entry point matching Interface Contract Stage 7."""
    single, pair = build_representations(
        idx=idx,
        phys=phys,
        coords_ca=coords_ca,
        weights=weights,
        af2_single=af2_single,
        af2_pair=af2_pair,
        params=params,
        theta=theta,
        temp_k=temp_k,
        backbone_n_ca_c=backbone_n_ca_c,
        chain_mask=chain_mask,
        num_rbf=num_rbf,
    )
    return pad_features(single, pair, max_n=max_n)


def pocket_drift_drmsd(
    backbone_curr: jnp.ndarray,
    d_cb_anchor: jnp.ndarray,
    weights: jnp.ndarray,
    chain_mask: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    """SE(3)-invariant pocket-weighted C_beta* distance RMSD against the last trunk anchor."""
    n = backbone_curr.shape[0]
    d_cb_curr = pairwise_dist(virtual_cb(center(backbone_curr)))
    nb_mask = nonbonded_pair_mask(n, chain_mask=chain_mask)
    pair_w = nb_mask * (weights[:, None] * weights[None, :])
    norm_w = jnp.maximum(jnp.sum(pair_w), EPS_DIST)
    return jnp.sqrt(jnp.sum(pair_w * (d_cb_curr - d_cb_anchor) ** 2) / norm_w + EPS_DIST)


def maybe_refresh_trunk(
    backbone_curr: jnp.ndarray,
    d_cb_anchor: jnp.ndarray,
    single_cached: jnp.ndarray,
    pair_cached: jnp.ndarray,
    mask_cached: jnp.ndarray,
    idx: jnp.ndarray,
    phys: jnp.ndarray,
    weights: jnp.ndarray,
    af2_single: Optional[jnp.ndarray],
    af2_pair: Optional[jnp.ndarray],
    params: Dict[str, jnp.ndarray],
    theta: dict,
    max_n: int,
    temp_k: float = 310.15,
    chain_mask: Optional[jnp.ndarray] = None,
    num_rbf: int = 64,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Verlet-style adaptive refresh: recomputes trunk only if pocket dRMSD exceeds theta['r_trunk_tol']."""
    drmsd = pocket_drift_drmsd(backbone_curr, d_cb_anchor, weights, chain_mask=chain_mask)
    should_refresh = drmsd > theta["r_trunk_tol"]

    def _recompute(_):
        s_new, p_new, m_new = run_trunk(
            idx=idx,
            phys=phys,
            coords_ca=backbone_curr[:, 1, :],
            weights=weights,
            af2_single=af2_single,
            af2_pair=af2_pair,
            params=params,
            theta=theta,
            max_n=max_n,
            temp_k=temp_k,
            backbone_n_ca_c=backbone_curr,
            chain_mask=chain_mask,
            num_rbf=num_rbf,
        )
        d_cb_new = pairwise_dist(virtual_cb(center(backbone_curr)))
        return s_new, p_new, m_new, d_cb_new

    def _keep(_):
        return single_cached, pair_cached, mask_cached, d_cb_anchor

    return jax.lax.cond(should_refresh, _recompute, _keep, operand=None)


# References:
#   - Evoformer single/pair representations & 64-bin [2-22 A] distogram:
#     Jumper et al. (2021), Nature 596:583-589
#   - Screened Debye-Huckel electrostatics & Born regularization:
#     Honig & Nicholls (1995), Science 268(5214):1144-1149; Still et al. (1990), JACS 112(16):6127-6129
#   - Hydrophobic core (h > 0), side-chain H-bonding & Gaussian ENM contacts:
#     Kyte & Doolittle (1982), J. Mol. Biol. 157(1):105-132; McDonald & Thornton (1994), J. Mol. Biol. 238(5):777-793
#     Tirion (1996), Phys. Rev. Lett. 77(9):1905-1908; Hinsen (1998), Proteins 33(3):417-429
#   - Adaptive neighbor-list / conformation-triggered feature refresh:
#     Verlet (1967), Phys. Rev. 159(1):98-103
#   - Layer Normalization & Glorot uniform initialization:
#     Ba, Kiros & Hinton (2016), arXiv:1607.06450; Glorot & Bengio (2010), AISTATS 9:249-256
