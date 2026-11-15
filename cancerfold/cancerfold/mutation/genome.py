"""AlphaGenome regulatory variant coupling via Boltzmann chemical potential."""
import jax.numpy as jnp


def coupled_dosage(
    log2_fc: jnp.ndarray,
    temp_k: jnp.ndarray,
    delta_psi: jnp.ndarray = jnp.array(0.0, dtype=jnp.float32),
) -> dict:
    r_gas = jnp.array(0.0019872, dtype=jnp.float32)  # Universal gas constant R (kcal / (mol * K))
    kbt = r_gas * temp_k                             # Thermal energy R * T (kcal / mol)

    intact_splice = jnp.clip(1.0 + delta_psi, 1e-6, 1.0)
    dosage = (2.0 ** log2_fc) * intact_splice

    ddg_dosage = -kbt * jnp.log(jnp.clip(dosage, 1e-6, 1e6))
    return {
        "temp_k": temp_k,
        "kbt": kbt,
        "dosage_ratio": dosage,
        "ddg_dosage": ddg_dosage,
    }


# References:
#   - Sequence-to-function regulatory variant predictions (expression & splicing):
#     Avsec et al. (2025), AlphaGenome: advancing regulatory variant effect prediction
#   - Percent Spliced In (PSI / delta_PSI) isoform retention formulation:
#     Katz et al. (2010), Nature Methods 7(12):1009-1015
#   - Boltzmann chemical potential shift from macromolecular dosage ratio (dG = -RT ln(C_mut / C_wt)):
#     Hill (1985), Free Energy Transduction and Biochemical Cycle Kinetics, Springer
