"""SE(3)-invariant smoothed pairwise distance operators D_ij(X) in JAX."""
import jax
import jax.numpy as jnp


def pairwise_dist(
    coords: jnp.ndarray, 
    eps: float = 1e-8
) -> jnp.ndarray:
    
    """Computes smooth, SE(3)-invariant pairwise Euclidean distances D_ij(X) = sqrt(||x_i - x_j||^2 + eps)."""
    diff = coords[:, None, :] - coords[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)
    return jnp.sqrt(sq_dist + eps)


def cdist_smooth(
    coords_a: jnp.ndarray, 
    coords_b: jnp.ndarray, 
    eps: float = 1e-8
) -> jnp.ndarray:
    
    """Computes smooth pairwise distances D_ij(X, Y) between two 3D point sets (N, 3) and (M, 3)."""
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)
    return jnp.sqrt(sq_dist + eps)


def pairwise_unit_vectors(
    coords: jnp.ndarray,
    eps: float = 1e-8
) -> tuple:
    
    """Returns smoothed pairwise distances D_ij(X) and regularized unit direction vectors r_hat_ij."""
    diff = coords[:, None, :] - coords[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + eps)
    unit_vec = diff / dist[..., None]
    return dist, unit_vec


def distance_matrix_rmsd(
    d_pred: jnp.ndarray,
    d_ref: jnp.ndarray,
    pair_mask: jnp.ndarray = None,
    eps: float = 1e-8,
) -> jnp.ndarray:
    
    """Computes SE(3)-invariant distance RMSD (dRMSD) over upper-triangle non-adjacent residue pairs."""
    n = d_pred.shape[0]
    idx = jnp.arange(n)
    upper_non_adj = (idx[None, :] - idx[:, None] > 1).astype(jnp.float32)
    mask = upper_non_adj if pair_mask is None else upper_non_adj * pair_mask
    norm = jnp.maximum(jnp.sum(mask), eps)
    msd = jnp.sum(mask * (d_pred - d_ref) ** 2) / norm
    return jnp.sqrt(msd + eps)


# References:
#   - Epsilon-regularized Euclidean distance for differentiable structural modeling:
#     Jumper et al. (2021), Nature 596:583-589 (AlphaFold 2, Supplementary Information)
#   - SE(3) invariance of pairwise inter-atomic distance matrices:
#     Crippen & Havel (1988), Distance Geometry and Molecular Conformation, Research Studies Press
#   - Distance-matrix RMSD (dRMSD) for alignment-free protein structure comparison:
#     Holm & Sander (1993), J. Mol. Biol. 233(1):123-138
