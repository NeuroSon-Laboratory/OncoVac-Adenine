"""Smooth SE(3)-invariant multi-mutation pocket attenuation via Noisy-OR composition."""
from typing import Optional
import jax.numpy as jnp

from cancerfold.physics.geometry import pairwise_dist


def multi_pocket_mask(
    ca_coords: jnp.ndarray,
    mut_mask: jnp.ndarray,
    residue_mask: Optional[jnp.ndarray] = None,
    sigma: float = 8.0,
) -> jnp.ndarray:
    """
    Computes C^inf-smooth pocket weights w_i in [0, 1] for arbitrary single or multi-mutations:
        w_i = 1 - prod_k (1 - m_k * exp(-0.5 * (d_ik / sigma)^2))

    Args:
        ca_coords: (N, 3) C-alpha coordinates [Angstrom].
        mut_mask: (N,) binary mask (1.0 at mutated residues, 0.0 elsewhere).
        residue_mask: Optional (N,) valid-residue mask for static MAX_N padding.
        sigma: Spatial attenuation width [Angstrom].
    """
    d_ca = pairwise_dist(ca_coords)
    m_mut = mut_mask.astype(jnp.float32)
    if residue_mask is not None:
        m_res = residue_mask.astype(jnp.float32)
        m_mut = m_mut * m_res
    else:
        m_res = jnp.ones_like(m_mut)

    g_ik = jnp.exp(-0.5 * (d_ca / sigma) ** 2)
    one_minus = jnp.clip(1.0 - (m_mut[None, :] * g_ik), 0.0, 1.0)
    w_total = (1.0 - jnp.prod(one_minus, axis=-1)) * m_res
    return w_total


def pocket_mask(
    ca_coords: jnp.ndarray,
    mut_idx: int,
    residue_mask: Optional[jnp.ndarray] = None,
    sigma: float = 8.0,
) -> jnp.ndarray:
    """Backward-compatible wrapper for single-point mutations (identical to multi_pocket_mask)."""
    mut_mask = jnp.zeros((ca_coords.shape[0],), dtype=jnp.float32).at[mut_idx].set(1.0)
    return multi_pocket_mask(ca_coords, mut_mask, residue_mask=residue_mask, sigma=sigma)


# References:
#   - Noisy-OR bounded causal aggregation in [0, 1]:
#     Pearl (1988), Probabilistic Reasoning in Intelligent Systems, Morgan Kaufmann.
#   - Multi-mutation epistasis & SKEMPI 2.0 complex-level benchmark:
#     Jankauskaite et al. (2019), Bioinformatics 35(3):462-469.
