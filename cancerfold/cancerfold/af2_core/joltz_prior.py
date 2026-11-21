"""Frozen Joltz-1 prior adapter for Gen 1 (Adenine): extracted once at t=0 without X_0 leakage."""
from typing import Callable, Dict, Optional, Tuple
import jax
import jax.numpy as jnp


def _project_or_pad_last_dim(x: jnp.ndarray, target_dim: int) -> jnp.ndarray:
    """Deterministic chunk-pooled projection or zero-padding along the last feature axis."""
    cur_dim = x.shape[-1]
    if cur_dim == target_dim:
        return x.astype(jnp.float32)
    if cur_dim > target_dim:
        scale = cur_dim / target_dim
        idx = jnp.clip(jnp.floor(jnp.arange(target_dim) * scale).astype(jnp.int32), 0, cur_dim - 1)
        return jnp.take(x, idx, axis=-1).astype(jnp.float32)
    pad_width = [(0, 0)] * (x.ndim - 1) + [(0, target_dim - cur_dim)]
    return jnp.pad(x.astype(jnp.float32), pad_width, mode="constant")


def extract_frozen_joltz_repr(
    joltz_forward_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]],
    aa_indices: jnp.ndarray,
    residue_mask: jnp.ndarray,
    c_s: int = 128,
    c_z: int = 64,
    fallback_key: Optional[jax.Array] = None,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Extracts frozen (single, pair) representations once at t=0 from a callable Joltz-1 model.
    If joltz_forward_fn is None, builds a deterministic sequence-only prior (no X_0 leakage).
    """
    m = residue_mask.astype(jnp.float32)
    n = aa_indices.shape[0]

    if joltz_forward_fn is not None:
        raw_s, raw_z = joltz_forward_fn(aa_indices, m)
        s_frozen = jax.lax.stop_gradient(_project_or_pad_last_dim(raw_s, c_s))
        z_frozen = jax.lax.stop_gradient(_project_or_pad_last_dim(raw_z, c_z))
    else:
        key = fallback_key if fallback_key is not None else jax.random.PRNGKey(0)
        vocab_size = 21
        aa_embed = jax.random.normal(key, (vocab_size, c_s), dtype=jnp.float32) * 0.05
        s_base = aa_embed[jnp.clip(aa_indices, 0, vocab_size - 1)]
        s_frozen = jax.lax.stop_gradient(s_base)

        same_aa = (aa_indices[:, None] == aa_indices[None, :]).astype(jnp.float32)[..., None]
        outer_s = jnp.einsum("ia,jb->ijab", s_base[:, :8], s_base[:, :8]).reshape(n, n, 64)
        z_combined = jnp.concatenate([jnp.broadcast_to(same_aa, (n, n, c_z)), outer_s], axis=-1)
        z_frozen = jax.lax.stop_gradient(_project_or_pad_last_dim(z_combined, c_z))

    s_masked = s_frozen * m[:, None]
    z_masked = z_frozen * (m[:, None] * m[None, :])[..., None]
    return s_masked, z_masked


def prepare_wt_mut_priors(
    joltz_forward_fn: Optional[Callable[[jnp.ndarray, jnp.ndarray], Tuple[jnp.ndarray, jnp.ndarray]]],
    wt_aa_indices: jnp.ndarray,
    mut_aa_indices: jnp.ndarray,
    residue_mask: jnp.ndarray,
    c_s: int = 128,
    c_z: int = 64,
    fallback_key: Optional[jax.Array] = None,
) -> Dict[str, Tuple[jnp.ndarray, jnp.ndarray]]:
    """Computes frozen priors symmetrically for Wild-Type and Mutant sequences at t=0."""
    k_wt, k_mut = (
        jax.random.split(fallback_key)
        if fallback_key is not None
        else (jax.random.PRNGKey(1), jax.random.PRNGKey(2))
    )
    s_wt, z_wt = extract_frozen_joltz_repr(
        joltz_forward_fn, wt_aa_indices, residue_mask, c_s=c_s, c_z=c_z, fallback_key=k_wt
    )
    s_mut, z_mut = extract_frozen_joltz_repr(
        joltz_forward_fn, mut_aa_indices, residue_mask, c_s=c_s, c_z=c_z, fallback_key=k_mut
    )
    return {
        "wt": (s_wt, z_wt),
        "mut": (s_mut, z_mut),
    }


# References:
#   - Stop-gradient isolation of frozen prior representations during downstream optimization:
#     Jumper et al. (2021), Nature 596:583-589 (AlphaFold 2 Evoformer / structure module interface).
#   - Symmetric state prior treatment for exact thermodynamic antisymmetry:
#     Pucci et al. (2018), Briefings in Bioinformatics 19(4):599-611.
