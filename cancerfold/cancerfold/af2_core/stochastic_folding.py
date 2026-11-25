"""Pure-state SE(3)-invariant 2-hop Effective-Hamiltonian residual potential (Dual-Cutoff + Interface + Pocket H_eff)."""
from typing import Dict, Optional, Tuple
import jax
import jax.numpy as jnp

from cancerfold.physics.geometry import pairwise_dist
from cancerfold.physics.potentials import (
    adaptive_inv_dielectric,
    c2_envelope,
    coordination_burial,
    state_physics_energy_and_force,
    u_metadynamics_bias,
    u_noncov_state,
    u_shared_scaffold,
    virtual_cb,
)

K_SHELL = 12.0
R_SHORT = 10.0  # Kills folded trunk contact prior smoothly before 11 A
R_LONG = 16.0   # Preserves true long-range electrostatics / allosteric coupling
EPS_NORM = 1e-5
EPS_VEC = 1e-12
PRIOR_TARGET_RMS = 0.15
PRIOR_RMS_FLOOR = 0.05


def _safe_unit(vec: jnp.ndarray) -> jnp.ndarray:
    return vec / jnp.sqrt(jnp.sum(vec ** 2, axis=-1, keepdims=True) + EPS_VEC)


def _layer_norm(x: jnp.ndarray, scale: jnp.ndarray, bias: jnp.ndarray) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    return scale * (x - mean) / jnp.sqrt(var + EPS_NORM) + bias


def _soft_rms_scale(x: jnp.ndarray, target: float = PRIOR_TARGET_RMS, floor: float = PRIOR_RMS_FLOOR) -> jnp.ndarray:
    """
    Smooth per-token / per-pair RMS balancing: maps both fallback priors (std ~0.05) and
    LayerNormed Joltz-1 priors (std ~1.0) to a balanced ~0.10-0.15 scale matching 3D geometry & phys_state,
    while keeping padded zero vectors exactly 0.0 and preserving exact WT <-> Mut symmetry.
    """
    rms = jnp.sqrt(jnp.mean(x ** 2, axis=-1, keepdims=True) + floor ** 2)
    return (target / rms) * x


def _pad_phys_5(phys_state: Optional[jnp.ndarray], n: int) -> jnp.ndarray:
    """Normalizes biochemical node tensor to (n, 5): [vol, q, hydro, donor, acceptor]."""
    if phys_state is None:
        return jnp.zeros((n, 5), dtype=jnp.float32)
    p = phys_state[:n].astype(jnp.float32)
    d = p.shape[-1]
    if d == 5:
        return p
    if d > 5:
        return p[:, :5]
    return jnp.pad(p, ((0, 0), (0, 5 - d)), mode="constant")


def _glorot_uniform(key: jax.Array, shape: Tuple[int, ...], scale: float = 1.0) -> jnp.ndarray:
    limit = scale * jnp.sqrt(6.0 / (shape[-2] + shape[-1]))
    return jax.random.uniform(key, shape, dtype=jnp.float32, minval=-limit, maxval=limit)


def _live_rbf(dist: jnp.ndarray, num_rbf: int, r_min: float, r_max: float) -> jnp.ndarray:
    centers = jnp.linspace(r_min, r_max, num_rbf, dtype=jnp.float32)
    width = (r_max - r_min) / max(num_rbf - 1, 1)
    return jnp.exp(-0.5 * ((dist[..., None] - centers[None, None, :]) / width) ** 2)


def _orientation_scalars_7(backbone: jnp.ndarray, cb_pos: jnp.ndarray) -> jnp.ndarray:
    """
    7 SE(3)-invariant orientation scalars per residue pair (N, N, 7):
      - 4 side-chain scalars: u_i.u_j, u_i.r_ij, u_j.(-r_ij), (u_i x u_j).r_ij (chiral pseudoscalar)
      - 3 backbone tangent scalars: b_i.b_j, b_i.r_ij, b_j.(-r_ij) for beta-sandwich sheet geometry
    """
    n_pos, ca_pos, c_pos = backbone[:, 0, :], backbone[:, 1, :], backbone[:, 2, :]

    u_hat = _safe_unit(cb_pos - ca_pos)
    b_hat = _safe_unit(c_pos - n_pos)
    r_hat = _safe_unit(ca_pos[None, :, :] - ca_pos[:, None, :])  # r_j - r_i

    # 4 side-chain orientation scalars
    o1 = jnp.einsum("ik,jk->ij", u_hat, u_hat)
    o2 = jnp.einsum("ik,ijk->ij", u_hat, r_hat)
    o3 = -jnp.einsum("jk,ijk->ij", u_hat, r_hat)
    u_cross = jnp.cross(u_hat[:, None, :], u_hat[None, :, :])
    o4 = jnp.einsum("ijk,ijk->ij", u_cross, r_hat)

    # 3 backbone tangent scalars
    o5 = jnp.einsum("ik,jk->ij", b_hat, b_hat)
    o6 = jnp.einsum("ik,ijk->ij", b_hat, r_hat)
    o7 = -jnp.einsum("jk,ijk->ij", b_hat, r_hat)

    return jnp.stack([o1, o2, o3, o4, o5, o6, o7], axis=-1)


def init_folding_params(
    prng_key: jax.Array,
    c_s: int = 384,   # Matches full Joltz-1 Pairformer token_s=384 (zero dropped channels)
    c_z: int = 128,   # Matches full Joltz-1 Pairformer token_z=128 (zero dropped channels)
    c_h: int = 64,
    num_rbf_short: int = 12,
    num_rbf_long: int = 8,
) -> Dict[str, jnp.ndarray]:
    """
    Initializes parameters for the pure-state 2-hop SE(3) Effective-Hamiltonian neural residual:
      - Dedicated local biochemical + live total/interface burial + pocket branch (8-dim)
      - Balanced RMS-scaled Joltz-1 (c_s=384, c_z=128) -> c_h projection
      - Dual-Cutoff (10 A short contact / 16 A long dielectric) + 7 orientation scalars + is_inter_chain
      - 2-hop conservative message passing (2-body + 3-body many-body cooperativity)
    """
    keys = jax.random.split(prng_key, 14)
    d_short = 2 * num_rbf_short + 8  # CA_RBF + CB_RBF + 7 SE(3) orientation scalars + is_inter_chain
    d_long = num_rbf_long + 2        # Long-range CB_RBF + live 1/eps_eff(b_i, b_j) + is_inter_chain
    d_phys = 8                       # [vol, q, hydro, donor, acc, total burial b_i, inter_burial, pocket_w]

    return {
        "w_short_1": _glorot_uniform(keys[0], (d_short, c_h)),
        "b_short_1": jnp.zeros((c_h,), dtype=jnp.float32),
        "w_short_2": _glorot_uniform(keys[1], (c_h, c_h)),
        "b_short_2": jnp.zeros((c_h,), dtype=jnp.float32),
        "w_long_1": _glorot_uniform(keys[2], (d_long, c_h)),
        "b_long_1": jnp.zeros((c_h,), dtype=jnp.float32),
        "w_pair_proj": _glorot_uniform(keys[3], (c_z, c_h)),
        "b_pair_proj": jnp.zeros((c_h,), dtype=jnp.float32),
        # Balanced prior + explicit local physical-biochemical & interface node branches
        "w_single_val": _glorot_uniform(keys[4], (c_s, c_h)),
        "b_single_val": jnp.zeros((c_h,), dtype=jnp.float32),
        "w_single_self": _glorot_uniform(keys[5], (c_s, c_h)),
        "b_single_self": jnp.zeros((c_h,), dtype=jnp.float32),
        "w_phys_val": _glorot_uniform(keys[11], (d_phys, c_h)),
        "b_phys_val": jnp.zeros((c_h,), dtype=jnp.float32),
        "w_phys_self": _glorot_uniform(keys[12], (d_phys, c_h)),
        "b_phys_self": jnp.zeros((c_h,), dtype=jnp.float32),
        # Hop 1 (2-body pair interactions)
        "w_agg_1": _glorot_uniform(keys[6], (c_h, c_h)),
        "b_agg_1": jnp.zeros((c_h,), dtype=jnp.float32),
        "ln_h1_scale": jnp.ones((c_h,), dtype=jnp.float32),
        "ln_h1_bias": jnp.zeros((c_h,), dtype=jnp.float32),
        # Hop 2 (3-body many-body cooperativity residual)
        "w_val_2": _glorot_uniform(keys[7], (c_h, c_h)),
        "b_val_2": jnp.zeros((c_h,), dtype=jnp.float32),
        "w_agg_2": _glorot_uniform(keys[8], (c_h, c_h), scale=0.5),
        "b_agg_2": jnp.zeros((c_h,), dtype=jnp.float32),
        "ln_h2_scale": jnp.ones((c_h,), dtype=jnp.float32),
        "ln_h2_bias": jnp.zeros((c_h,), dtype=jnp.float32),
        # Effective-pocket per-residue energy readout (bounded small initial residual)
        "w_readout_1": _glorot_uniform(keys[9], (c_h, c_h)),
        "b_readout_1": jnp.zeros((c_h,), dtype=jnp.float32),
        "w_readout_2": _glorot_uniform(keys[10], (c_h, 1), scale=0.08),
        "energy_scale": jnp.array(0.25, dtype=jnp.float32),
    }


def single_state_nn_energy(
    backbone: jnp.ndarray,
    single: jnp.ndarray,
    pair: jnp.ndarray,
    residue_mask: jnp.ndarray,
    params: Dict[str, jnp.ndarray],
    phys_state: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
    chain_mask: Optional[jnp.ndarray] = None,
    r_short: float = R_SHORT,
    r_long: float = R_LONG,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Pure state function U_NN(X; s_alpha, z_alpha, phys_alpha, chain_mask) with Effective-Hamiltonian localization:
      - Dedicated 8-dim local biochemical + live total/interface burial + pocket branch
      - Explicit inter-chain binding interface indicator from 1D chain_mask (strictly 0 in single-chain unbound)
      - Smooth RMS-balanced frozen Joltz-1 (s, z) so global Pairformer channels never drown 3D geometry
      - Dual-Cutoff (10 A short contact / 16 A long dielectric) + 7 SE(3) orientation scalars
      - 2-hop conservative message passing for 3-body many-body cooperativity
      - Smooth continuous pocket envelope `weights` (Noisy-OR) suppressing distant global attention noise
    """
    n = backbone.shape[0]
    m = residue_mask[:n].astype(jnp.float32)
    pair_valid = (m[:, None] * m[None, :]) * (1.0 - jnp.eye(n, dtype=jnp.float32))

    if chain_mask is not None:
        if chain_mask.ndim == 1:
            cid = chain_mask[:n]
            same_chain = ((cid[:, None] == cid[None, :]) & (cid[:, None] >= 0)).astype(jnp.float32)
        else:
            same_chain = jnp.clip(chain_mask[:n, :n].astype(jnp.float32), 0.0, 1.0)
        is_inter = (1.0 - same_chain) * pair_valid
    else:
        is_inter = jnp.zeros((n, n), dtype=jnp.float32)

    if weights is not None:
        eff_mask = m * jnp.clip(weights[:n].astype(jnp.float32), 0.0, 1.0)
    else:
        eff_mask = m

    # Balance prior scale smoothly across both fallback (~0.05) and Joltz-1 (~1.0)
    s_bal = _soft_rms_scale(single[:n].astype(jnp.float32)) * m[:, None]
    z_bal = _soft_rms_scale(pair[:n, :n].astype(jnp.float32)) * pair_valid[..., None]

    ca_pos = backbone[:, 1, :]
    cb_pos = virtual_cb(backbone)
    d_ca = pairwise_dist(ca_pos)
    d_cb = pairwise_dist(cb_pos)

    # Live coordination burial b_i(X), smooth cross-chain interface burial, adaptive 1/eps_eff, & local biochem
    _, b_i = coordination_burial(d_cb, m)
    inv_eps = adaptive_inv_dielectric(b_i)
    inter_contacts = jnp.sum(c2_envelope(d_cb, r_short) * is_inter, axis=-1)
    inter_burial = jnp.tanh(inter_contacts / 6.0) * m

    phys_5 = _pad_phys_5(phys_state, n)
    phys_burial = jnp.concatenate(
        [phys_5, b_i[:, None], inter_burial[:, None], eff_mask[:, None]],
        axis=-1,
    ) * m[:, None]

    # Short-range channel (<= 10 A): kills folded trunk prior cleanly before 11 A
    num_rbf_short = (params["w_short_1"].shape[0] - 8) // 2
    orient_7 = _orientation_scalars_7(backbone, cb_pos)
    feat_short = jnp.concatenate(
        [
            _live_rbf(d_ca, num_rbf_short, 2.0, r_short),
            _live_rbf(d_cb, num_rbf_short, 2.0, r_short),
            orient_7,
            is_inter[..., None],
        ],
        axis=-1,
    )
    g_s = jax.nn.silu(jnp.einsum("ijk,kl->ijl", feat_short, params["w_short_1"]) + params["b_short_1"])
    w_s = jnp.einsum("ijk,kl->ijl", g_s, params["w_short_2"]) + params["b_short_2"]
    z_proj = jnp.einsum("ijk,kl->ijl", z_bal, params["w_pair_proj"]) + params["b_pair_proj"]
    e_short = jax.nn.silu(w_s + z_proj) * (c2_envelope(d_cb, r_short) * pair_valid)[..., None]

    # Long-range channel (<= 16 A): dielectric-modulated long-range coupling + interface awareness
    num_rbf_long = params["w_long_1"].shape[0] - 2
    feat_long = jnp.concatenate(
        [_live_rbf(d_cb, num_rbf_long, 6.0, r_long), inv_eps[..., None], is_inter[..., None]],
        axis=-1,
    )
    e_long = jax.nn.silu(
        jnp.einsum("ijk,kl->ijl", feat_long, params["w_long_1"]) + params["b_long_1"]
    ) * (c2_envelope(d_cb, r_long) * pair_valid)[..., None]

    e_total = e_short + e_long

    # Hop 1: balanced fusion of local biochemistry + interface burial + pocket + frozen Joltz-1 prior
    s_val_1 = jax.nn.silu(
        jnp.einsum("ni,ij->nj", phys_burial, params["w_phys_val"]) + params["b_phys_val"]
        + jnp.einsum("ni,ij->nj", s_bal, params["w_single_val"]) + params["b_single_val"]
    )
    s_self = (
        jnp.einsum("ni,ij->nj", phys_burial, params["w_phys_self"]) + params["b_phys_self"]
        + jnp.einsum("ni,ij->nj", s_bal, params["w_single_self"]) + params["b_single_self"]
    )

    msg_1 = jnp.einsum("ijh,jh->ih", e_total, s_val_1 * m[:, None]) / jnp.sqrt(K_SHELL)
    h_1 = _layer_norm(
        jax.nn.silu(s_self + jnp.einsum("ih,hk->ik", msg_1, params["w_agg_1"]) + params["b_agg_1"]),
        params["ln_h1_scale"],
        params["ln_h1_bias"],
    ) * m[:, None]

    # Hop 2: 3-body many-body cooperativity residual (strictly masked to active residues)
    v_2 = jax.nn.silu(jnp.einsum("ih,hk->ik", h_1, params["w_val_2"]) + params["b_val_2"]) * m[:, None]
    msg_2 = jnp.einsum("ijh,jh->ih", e_total, v_2) / jnp.sqrt(K_SHELL)
    h_2 = _layer_norm(
        h_1 + jax.nn.silu(jnp.einsum("ih,hk->ik", msg_2, params["w_agg_2"]) + params["b_agg_2"]),
        params["ln_h2_scale"],
        params["ln_h2_bias"],
    ) * m[:, None]

    # Effective-Hamiltonian pocket readout: focuses neural residual smoothly on mutation neighborhood
    u_hid = jax.nn.silu(jnp.einsum("ih,hk->ik", h_2, params["w_readout_1"]) + params["b_readout_1"])
    u_raw = jnp.squeeze(jnp.einsum("ik,kl->il", u_hid, params["w_readout_2"]), axis=-1)
    u_per_res = params["energy_scale"] * u_raw * eff_mask
    return jnp.sum(u_per_res), u_per_res


def full_state_energy(
    backbone: jnp.ndarray,
    d_ref_ca: jnp.ndarray,
    phys_state: jnp.ndarray,
    weights: jnp.ndarray,
    residue_mask: jnp.ndarray,
    history_cb_D: jnp.ndarray,
    history_mask: jnp.ndarray,
    theta: Dict[str, float],
    single: Optional[jnp.ndarray] = None,
    pair: Optional[jnp.ndarray] = None,
    nn_params: Optional[Dict[str, jnp.ndarray]] = None,
    chain_mask: Optional[jnp.ndarray] = None,
    use_adaptive_dielectric: bool = True,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Energy-only evaluation of H_alpha(X) = H_phys(X; alpha) + U_NN(X; alpha).
    Used inside BAR evaluation & Optax training to avoid unnecessary 2nd-order force differentiation.
    """
    u_sh = u_shared_scaffold(backbone, d_ref_ca, weights, residue_mask, theta, chain_mask=chain_mask)
    u_nc = u_noncov_state(
        backbone, phys_state, residue_mask, theta,
        chain_mask=chain_mask, use_adaptive_dielectric=use_adaptive_dielectric,
    )
    u_meta = u_metadynamics_bias(backbone, weights, history_cb_D, history_mask, theta)
    u_phys = u_sh + u_nc + u_meta

    if nn_params is None:
        u_nn = jnp.array(0.0, dtype=jnp.float32)
    else:
        u_nn, _ = single_state_nn_energy(
            backbone, single, pair, residue_mask, nn_params,
            phys_state=phys_state, weights=weights, chain_mask=chain_mask,
        )

    return u_phys + u_nn, {"u_phys": u_phys, "u_nn": u_nn, "u_shared": u_sh, "u_noncov": u_nc}


def full_state_energy_and_force(
    backbone: jnp.ndarray,
    d_ref_ca: jnp.ndarray,
    phys_state: jnp.ndarray,
    weights: jnp.ndarray,
    residue_mask: jnp.ndarray,
    history_cb_D: jnp.ndarray,
    history_mask: jnp.ndarray,
    theta: Dict[str, float],
    single: Optional[jnp.ndarray] = None,
    pair: Optional[jnp.ndarray] = None,
    nn_params: Optional[Dict[str, jnp.ndarray]] = None,
    chain_mask: Optional[jnp.ndarray] = None,
    use_adaptive_dielectric: bool = True,
    max_force: float = 500.0,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Evaluates full white-box state energy H_alpha(X) = H_phys(X; alpha) + U_NN(X; alpha)
    and conservative SE(3)-equivariant force F_alpha(X) = -grad_X H_alpha(X).

    Note: `single` and `pair` are frozen sequence-conditioned Joltz-1 priors (no static X_0 leakage);
    forces through U_NN differentiate cleanly wrt live coordinates `backbone` via RBF + 7 orientations +
    live total/interface burial + adaptive 1/eps_eff.
    """
    u_phys, f_phys = state_physics_energy_and_force(
        backbone,
        d_ref_ca,
        phys_state,
        weights,
        residue_mask,
        history_cb_D,
        history_mask,
        theta,
        chain_mask=chain_mask,
        use_adaptive_dielectric=use_adaptive_dielectric,
    )

    if nn_params is None:
        u_nn = jnp.array(0.0, dtype=jnp.float32)
        f_nn = jnp.zeros_like(f_phys)
    else:
        def _u_nn_fn(bb):
            val, _ = single_state_nn_energy(
                bb, single, pair, residue_mask, nn_params,
                phys_state=phys_state, weights=weights, chain_mask=chain_mask,
            )
            return val

        u_nn, grad_nn = jax.value_and_grad(_u_nn_fn)(backbone)
        f_nn = jnp.clip(-grad_nn, -max_force, max_force) * residue_mask[:, None, None]

    return u_phys + u_nn, f_phys + f_nn, {"u_phys": u_phys, "u_nn": u_nn}


# Mathematical & Scientific Grounding:
#   - Smooth C^2 short/long cutoffs & effective-Hamiltonian pocket localization:
#     Behler & Parrinello (2007), Phys. Rev. Lett. 98, 146401;
#     Batzner et al. (2022), Nature Communications 13, 2453 (NequIP).
#   - Conservative SE(3)-equivariant neural forces F = -grad_X U_NN(X):
#     Schutt et al. (2018), J. Chem. Phys. 148, 241722 (SchNet).
#   - SE(3)-invariant local residue frame orientation & inter-chain interface features:
#     Ingraham et al. (2019), NeurIPS (Generative Models for Graph-Based Protein Design);
#     Jumper et al. (2021), Nature 596, 583-589 (AlphaFold2).
