"""Continuous 3D mutation neighborhood attenuation from local packing density."""
from typing import Optional
import jax.numpy as jnp


def pocket_mask_from_point(
    coords: jnp.ndarray,
    anchor_xyz: jnp.ndarray,
    residue_mask: Optional[jnp.ndarray] = None,
    include_self: bool = False,
) -> dict:
    """Computes Gaussian spatial coupling weights w(d) around a 3D anchor point using K=12 shell."""
    diff = coords - anchor_xyz
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-8)
    k_target = 12 if include_self else 11

    if residue_mask is None:
        k_shell = jnp.minimum(k_target, jnp.maximum(dist.shape[0] - 1, 0))
        r_c = jnp.maximum(jnp.sort(dist)[k_shell], 1e-6)
        weights = jnp.exp(-0.5 * (dist / r_c) ** 2)
    else:
        m = residue_mask.astype(jnp.float32)
        n_valid = jnp.sum(m)
        k_shell = jnp.minimum(k_target, jnp.maximum(n_valid - 1, 0)).astype(jnp.int32)
        masked_dist = jnp.where(m > 0.5, dist, 1e9)
        r_c = jnp.maximum(jnp.sort(masked_dist)[k_shell], 1e-6)
        weights = jnp.exp(-0.5 * (dist / r_c) ** 2) * m

    return {
        "r_c": r_c,
        "weights": weights,
    }


def pocket_mask(
    coords: jnp.ndarray,
    mut_idx: int,
    residue_mask: Optional[jnp.ndarray] = None,
) -> dict:
    """Computes Gaussian spatial coupling weights w(d) centered on residue mut_idx (K=12 neighbors)."""
    return pocket_mask_from_point(
        coords=coords,
        anchor_xyz=coords[mut_idx],
        residue_mask=residue_mask,
        include_self=True,
    )


# References:
#   - First 3D coordination shell & local packing density in proteins (K = 12 close-packed neighbors):
#     Richards (1974), J. Mol. Biol. 82(1):1-14
#     Halle (2002), PNAS 99(3):1274-1279
#   - Continuous Gaussian distance attenuation in elastic protein networks:
#     Bahar, Atilgan & Erman (1997), Folding & Design 2(3):173-181
#   - Epsilon-regularized Euclidean norm sqrt(||dx||^2 + eps) for smooth JAX autodiff:
#     Jumper et al. (2021), Nature 596:583-589 (AlphaFold 2, Supplementary Information)
