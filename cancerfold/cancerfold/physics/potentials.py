"""Differentiable history-repulsive (U_rep) and stereochemical (U_geom) potentials."""
from cancerfold.physics.geomerty import pairwise_dist

def u_geom_potential(
  coords: jnp.ndarray,
  d_ref: jnp.ndarray,
  phys: jnp.ndarray,
  weights: jnp.ndarray,
  theta: dict,
)-> jnp.ndarray:
  d_mat = pairwise_dist(coords)
  n = coords.shape[0]
  idx = jnp.arrange(n)

