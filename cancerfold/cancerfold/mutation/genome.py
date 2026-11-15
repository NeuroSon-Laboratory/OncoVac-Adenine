"""AlphaGenome regulatory variant coupling with dynamic temperature (Gibbs-Helmholtz)."""
import jax.numpy as jnp


def coupled_dosage(
    log2_fc: jnp.ndarray,
    temp_k: jnp.ndarray,
    delta_psi: jnp.ndarray = jnp.array(0.0, dtype=jnp.float32),
) -> dict:
    t_ref = jnp.array(298.15, dtype=jnp.float32)    # IUPAC standard reference temperature (K)
    r_gas = jnp.array(0.0019872, dtype=jnp.float32) # Universal gas constant R (kcal / (mol * K))
    kbt = r_gas * temp_k                            # Dynamic thermal energy at operating temperature T

    intact_splice = jnp.clip(1.0 + delta_psi, 1e-6, 1.0)
    thermal_factor = t_ref / temp_k
    dosage = (2.0 ** log2_fc) * intact_splice * thermal_factor

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
#   - Temperature-sensitive unfolding of cancer mutants (e.g., p53 Y220C) & protein stability curves:
#     Bullock et al. (1997), Nature Structural Biology 4(11):947-954
#     Becktel & Schellman (1987), Biopolymers 26(11):1859-1877
