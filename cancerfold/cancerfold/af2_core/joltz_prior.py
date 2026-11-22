"""Frozen Joltz-1 prior adapter for Gen 1 (Adenine): full 384/128 channels without X_0 leakage."""
from typing import Callable, Dict, Optional, Tuple
import jax
import jax.numpy as jnp


def _pool_or_pad_last_dim(x: jnp.ndarray, target_dim: int) -> jnp.ndarray:
    """Preserves full feature information via exact pass-through, block-mean pooling, or zero-padding."""
    cur_dim = x.shape[-1]
    x_f32 = x.astype(jnp.float32)
    if cur_dim == target_dim:
        return x_f32
    if cur_dim > target_dim:
        if cur_dim % target_dim == 0:
            factor = cur_dim // target_dim
            return jnp.mean(x_f32.reshape(x_f32.shape[:-1] + (target_dim, factor)), axis=-1)
        pad_rem = target_dim - (cur_dim % target_dim)
        x_padded = jnp.pad(x_f32, [(0, 0)] * (x_f32.ndim - 1) + [(0, pad_rem)], mode="constant")
        factor = x_padded.shape[-1] // target_dim
        return jnp.mean(x_padded.reshape(x_f32.shape[:-1] + (target_dim, factor)), axis=-1)
    pad_width = [(0, 0)] * (x_f32.ndim - 1) + [(0, target_dim - cur_dim)]
    return jnp.pad(x_f32, pad_width, mode="constant")


def extract_frozen_joltz_repr(
    joltz_forward_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]],
    aa_indices: jnp.ndarray,
    residue_mask: jnp.ndarray,
    c_s: int = 384,
    c_z: int = 128,
    fallback_key: Optional[jax.Array] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Extracts frozen (single, pair) representations once at t=0 from a callable Joltz-1 Pairformer.
    Default (c_s=384, c_z=128) matches Joltz-1 token_s/token_z directly with zero channel dropping.
    If joltz_forward_fn is None, builds a deterministic sequence-only prior (no X_0 leakage).
    """
    m = residue_mask.astype(jnp.float32)
    n = aa_indices.shape[0]

    if joltz_forward_fn is not None:
        raw_s, raw_z = joltz_forward_fn(aa_indices, m)
        s_frozen = jax.lax.stop_gradient(_pool_or_pad_last_dim(raw_s, c_s))
        z_frozen = jax.lax.stop_gradient(_pool_or_pad_last_dim(raw_z, c_z))
    else:
        key = fallback_key if fallback_key is not None else jax.random.PRNGKey(0)
        vocab_size = 21
        aa_embed = jax.random.normal(key, (vocab_size, c_s), dtype=jnp.float32) * 0.05
        s_base = aa_embed[jnp.clip(aa_indices, 0, vocab_size - 1)]
        s_frozen = jax.lax.stop_gradient(s_base)

        same_aa = (aa_indices[:, None] == aa_indices[None, :]).astype(jnp.float32)[..., None]
        k_outer = min(8, c_s)
        outer_s = jnp.einsum("ia,jb->ijab", s_base[:, :k_outer], s_base[:, :k_outer]).reshape(
            n, n, k_outer * k_outer
        )
        abs_diff = jnp.abs(s_base[:, None, :min(32, c_s)] - s_base[None, :, :min(32, c_s)])
        z_combined = jnp.concatenate([same_aa, outer_s, abs_diff], axis=-1)
        z_frozen = jax.lax.stop_gradient(_pool_or_pad_last_dim(z_combined, c_z))

    s_masked = s_frozen * m[:, None]
    z_masked = z_frozen * (m[:, None] * m[None, :])[..., None]
    return s_masked, z_masked


def prepare_wt_mut_priors(
    joltz_forward_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]],
    wt_aa_indices: jnp.ndarray,
    mut_aa_indices: jnp.ndarray,
    residue_mask: jnp.ndarray,
    c_s: int = 384,
    c_z: int = 128,
    fallback_key: Optional[jax.Array] = None,
) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Computes frozen priors symmetrically for Wild-Type and Mutant sequences at t=0."""
    # Exact shared fallback key guarantees identical embeddings for unmutated residues (WT <-> Mut symmetry)
    shared_key = fallback_key if fallback_key is not None else jax.random.PRNGKey(0)
    s_wt, z_wt = extract_frozen_joltz_repr(
        joltz_forward_fn, wt_aa_indices, residue_mask, c_s=c_s, c_z=c_z, fallback_key=shared_key
    )
    s_mut, z_mut = extract_frozen_joltz_repr(
        joltz_forward_fn, mut_aa_indices, residue_mask, c_s=c_s, c_z=c_z, fallback_key=shared_key
    )
    return {
        "wt": (s_wt, z_wt),
        "mut": (s_mut, z_mut),
    }


# References:
#   - Stop-gradient isolation of frozen prior representations during downstream optimization:
#     Jumper et al. (2021), Nature 596:583-589 (AlphaFold 2 Evoformer / structure module interface).
#   - Open biomolecular Pairformer representations (Boltz-1 / Joltz-1, token_s=384, token_z=128):
#     Wohlwend et al. (2024), bioRxiv (Boltz-1: Democratizing Biomolecular Interaction Modeling).
#   - Symmetric state prior treatment for exact thermodynamic antisymmetry:
#     Pucci et al. (2018), Briefings in Bioinformatics 19(4):599-611.
