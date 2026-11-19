"""SE(3)-invariant smoothed pairwise distance and backbone centering operators in JAX."""
import jax.numpy as jnp


def pairwise_dist(
    coords: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Computes smooth, SE(3)-invariant pairwise Euclidean distances D_ij(X) = sqrt(||x_i - x_j||^2 + eps)."""
    diff = coords[:, None, :] - coords[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)
    return jnp.sqrt(sq_dist + eps)


def cdist_smooth(
    coords_a: jnp.ndarray,
    coords_b: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Computes smooth pairwise distances D_ij(X, Y) between two 3D point sets (N, 3) and (M, 3)."""
    diff = coords_a[:, None, :] - coords_b[None, :, :]
    sq_dist = jnp.sum(diff ** 2, axis=-1)
    return jnp.sqrt(sq_dist + eps)


def segment_dist(
    coords_a: jnp.ndarray,
    coords_b: jnp.ndarray,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Computes regularized Euclidean distances between paired point arrays (..., 3)."""
    sq_dist = jnp.sum((coords_a - coords_b) ** 2, axis=-1)
    return jnp.sqrt(sq_dist + eps)


def pairwise_unit_vectors(
    coords: jnp.ndarray,
    eps: float = 1e-8,
) -> tuple:
    """Returns smoothed pairwise distances D_ij(X) and unit vectors r_hat_ij pointing from j to i (x_i - x_j)."""
    diff = coords[:, None, :] - coords[None, :, :]
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + eps)
    unit_vec = diff / dist[..., None]
    return dist, unit_vec


def center(coords: jnp.ndarray) -> jnp.ndarray:
    """Centers (N, 3) coordinates or (N, 3, 3) [N, CA, C] backbone so C-alpha center of mass sits at [0, 0, 0]."""
    if coords.ndim == 2 and coords.shape[-1] == 3:
        com = jnp.mean(coords, axis=0, keepdims=True)
        return coords - com
    if coords.ndim == 3 and coords.shape[1:] == (3, 3):
        ca_com = jnp.mean(coords[:, 1, :], axis=0, keepdims=True)
        return coords - ca_com[:, None, :]
    raise ValueError(f"Expected (N, 3) or (N, 3, 3) coordinates, got {coords.shape}")


def nonbonded_pair_mask(
    n: int,
    chain_mask: jnp.ndarray = None,
) -> jnp.ndarray:
    """Returns upper-triangle non-bonded pair mask (N, N), preserving inter-chain breaks where chain_mask == 0."""
    idx = jnp.arange(n)
    if chain_mask is None:
        return (idx[None, :] - idx[:, None] > 1).astype(jnp.float32)
    upper = (idx[None, :] - idx[:, None] > 0).astype(jnp.float32)
    covalent_adj = jnp.diag(chain_mask.astype(jnp.float32), k=1)
    return upper * (1.0 - covalent_adj)


def distance_matrix_rmsd(
    d_pred: jnp.ndarray,
    d_ref: jnp.ndarray,
    pair_mask: jnp.ndarray = None,
    chain_mask: jnp.ndarray = None,
    eps: float = 1e-8,
) -> jnp.ndarray:
    """Computes SE(3)-invariant distance RMSD (dRMSD) over upper-triangle non-bonded residue pairs."""
    n = d_pred.shape[0]
    base_mask = nonbonded_pair_mask(n, chain_mask=chain_mask)
    mask = base_mask if pair_mask is None else base_mask * pair_mask
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
