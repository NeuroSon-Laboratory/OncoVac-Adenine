"""Continuous 3D mutation neighborhood attenuation from local packing density."""
import jax.numpy as jnp


def pocket_mask(coords: jnp.ndarray, mut_idx: int) -> dict:
    diff = coords - coords[mut_idx]
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-8)

    k_shell = jnp.minimum(12, dist.shape[0] - 1)
    r_c = jnp.sort(dist)[k_shell]
    weights = jnp.exp(-0.5 * (dist / r_c) ** 2)
    return {
        "r_c": r_c,
        "weights": weights,
    }


# References:
#   - First 3D coordination shell & local packing density in proteins (K = 12 close-packed neighbors):
#     Richards (1974), J. Mol. Biol. 82(1):1-14
#     Halle (2002), PNAS 99(3):1274-1279
#   - Continuous Gaussian distance attenuation in elastic protein networks:
#     Bahar, Atilgan & Erman (1997), Folding & Design 2(3):173-181
#   - Epsilon-regularized Euclidean norm sqrt(||dx||^2 + eps) for smooth JAX autodiff:
#     Jumper et al. (2021), Nature 596:583-589 (AlphaFold 2, Supplementary Information)
