"""Local active-site and mutation-centered radial neighborhood masks."""
import jax
import jax.numpy as jnp


def opt_radius(
    coords: jnp.ndarray,
    mut_idx: int,
    wt_phys: jnp.ndarray,
    mut_phys: jnp.ndarray,
    steps: int = 15,
    lr: float = 0.5,
) -> jnp.ndarray:
    diff = coords - coords[mut_idx]
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-8)

    delta = jnp.abs(mut_phys - wt_phys)
    weights = jnp.array([2.5, 0.3, 2.0, 0.5], dtype=jnp.float32)
    s_mut = 1.0 + jnp.sum(weights * delta)

    r_min = 8.0               # Minimum C-alpha shell cutoff (Angstrom, Kellogg et al. 2011)
    lambda_d = 3.04           # Debye electrostatic screening length (Angstrom, Honig & Nicholls 1995)
    kbt = 0.0019872 * 310.15  # Thermal energy RT at 310.15 K / 37 C body temp (kcal/mol)
    sigma_r = 1.5             # Radial shell fluctuation width (Angstrom, Bahar et al. 1997)
    k_spring = kbt / (sigma_r ** 2)

    def loss_fn(r: jnp.ndarray) -> jnp.ndarray:
        leak = s_mut * jnp.exp(-(r - r_min) / lambda_d)
        crowd = kbt * jnp.mean(jax.nn.sigmoid(r - dist))
        spring = 0.5 * k_spring * (r - r_min) ** 2
        return leak + crowd + spring

    grad_fn = jax.grad(loss_fn)
    r0 = jnp.array(10.0, dtype=jnp.float32)
    r_opt = jax.lax.fori_loop(0, steps, lambda _, r: r - lr * grad_fn(r), r0)
    return jnp.clip(r_opt, 8.0, 14.0)


def pocket_mask(coords: jnp.ndarray, mut_idx: int, radius: float = 10.0) -> jnp.ndarray:
    diff = coords - coords[mut_idx]
    dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-8)
    return dist <= radius


# References:
#   - Local 8-12 Angstrom repacking/mutation neighborhood shell:
#     Kellogg, Leaver-Fay & Baker (2011), Proteins 79(3):830-838 (Rosetta ddG)
#     Guerois, Nielsen & Serrano (2002), J. Mol. Biol. 320(2):369-387 (FoldX)
#   - Exponential distance screening of mutational/electrostatic perturbation (leak term):
#     Honig & Nicholls (1995), Science 268(5214):1144-1149
#   - Smooth sigmoid contact switching function for differentiable 3D coordination (crowd term):
#     Best, Hummer & Eaton (2013), PNAS 110(44):17874-17879
#   - Harmonic spring constant k = kB*T / sigma_r^2 from Gaussian Network Model equipartition:
#     Bahar, Atilgan & Erman (1997), Folding & Design 2(3):173-181
#   - Exact gradient optimization via automatic differentiation (jax.grad):
#     Bradbury et al. (2018), JAX: composable transformations of Python+NumPy programs
