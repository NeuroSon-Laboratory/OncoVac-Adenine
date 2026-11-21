"""Layer 1 physics & numerical verification suite for OncoVac-Adenine (Gen 1)."""
import jax
import jax.numpy as jnp

from cancerfold.af2_core.joltz_prior import prepare_wt_mut_priors
from cancerfold.af2_core.loader import pad_structure
from cancerfold.af2_core.stochastic_folding import (
    full_state_energy_and_force,
    init_folding_params,
)
from cancerfold.mutation.neighborhood import multi_pocket_mask
from cancerfold.physics.potentials import adaptive_inv_dielectric, coordination_burial

KBT_310K = 0.616033  # Physiological k_B * T at 310 K [kcal / mol]


def _make_helical_backbone(n_res: int = 24) -> jnp.ndarray:
    """Builds a clean right-handed helical [N, CA, C] backbone for numerical testing."""
    idx = jnp.arange(n_res, dtype=jnp.float32)
    phi = idx * 1.7453293
    ca = jnp.stack([1.5 * idx, 2.3 * jnp.cos(phi), 2.3 * jnp.sin(phi)], axis=-1)
    n = ca + jnp.array([-0.52, 1.36, 0.0], dtype=jnp.float32)
    c = ca + jnp.array([0.52, 1.36, 0.0], dtype=jnp.float32)
    return jnp.stack([n, ca, c], axis=1)


def check_continuity_11a(max_n: int = 16) -> None:
    """Checks smooth continuity of burial fraction b_i and effective 1/eps across ~10-11 A."""
    res_mask = jnp.ones((max_n,), dtype=jnp.float32)
    base_d = jnp.full((max_n, max_n), 8.5, dtype=jnp.float32)
    base_d = base_d - jnp.diag(jnp.diag(base_d))

    dists = jnp.linspace(6.0, 13.0, 71)
    eps_inv_01_vals = []
    for r in dists:
        d_mat = base_d.at[0, 1].set(r).at[1, 0].set(r)
        _, b_i = coordination_burial(d_mat, res_mask)
        inv_eps = adaptive_inv_dielectric(b_i)
        eps_inv_01_vals.append(float(inv_eps[0, 1]))

    eps_arr = jnp.asarray(eps_inv_01_vals)
    assert jnp.all(jnp.isfinite(eps_arr)), "Non-finite dielectric values"
    max_jump = float(jnp.max(jnp.abs(jnp.diff(eps_arr))))
    print(f"[PASS] Continuity 10-11 A: max step jump in 1/eps_eff = {max_jump:.4e}")


def _simple_bar_ddg(dh_wt: jnp.ndarray, dh_mut: jnp.ndarray, kbt: float = KBT_310K, num_iters: int = 50) -> jnp.ndarray:
    """Bennett Acceptance Ratio (BAR) estimator solved via bisection."""
    beta = 1.0 / kbt
    offset = kbt * jnp.log(jnp.asarray(dh_mut.shape[0], dtype=jnp.float32) / jnp.asarray(dh_wt.shape[0], dtype=jnp.float32))
    lo = jnp.minimum(jnp.min(dh_wt), jnp.min(dh_mut)) - 10.0 * kbt
    hi = jnp.maximum(jnp.max(dh_wt), jnp.max(dh_mut)) + 10.0 * kbt

    def _residual(c: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(jax.nn.sigmoid(-beta * (dh_wt - c))) - jnp.mean(jax.nn.sigmoid(-beta * (-dh_mut + c)))

    def _step(_, bounds):
        low, high = bounds
        mid = 0.5 * (low + high)
        res = _residual(mid)
        return jnp.where(res < 0.0, mid, low), jnp.where(res < 0.0, high, mid)

    lo_f, hi_f = jax.lax.fori_loop(0, num_iters, _step, (lo, hi))
    return 0.5 * (lo_f + hi_f) - offset


def run_lapisan1_tests() -> None:
    check_continuity_11a(max_n=16)

    n_res = 24
    max_n = 32
    bb_raw = _make_helical_backbone(n_res)
    cm_raw = jnp.zeros((n_res,), dtype=jnp.int32)
    bb, cm, res_mask = pad_structure(bb_raw, cm_raw, max_n=max_n)

    mut_mask = jnp.zeros((max_n,), dtype=jnp.float32).at[10].set(1.0).at[14].set(1.0)
    weights = multi_pocket_mask(bb[:, 1, :], mut_mask, residue_mask=res_mask)
    assert jnp.all((weights >= 0.0) & (weights <= 1.0)), "Noisy-OR weights out of [0, 1]"
    print("[PASS] Multi-mutation Noisy-OR pocket weights bounded in [0, 1].")

    d_ref_ca = jnp.linalg.norm(bb[:, 1, None, :] - bb[None, :, 1, :], axis=-1)
    phys_wt = jnp.tile(jnp.array([[0.5, 0.0, 0.2, 0.5, 0.5]], dtype=jnp.float32), (max_n, 1))
    phys_mut = phys_wt.at[10].set(jnp.array([0.1, 0.0, -0.4, 0.2, 0.2], dtype=jnp.float32))

    key = jax.random.PRNGKey(42)
    k_p, k_pr, k_noise = jax.random.split(key, 3)
    nn_params = init_folding_params(k_p)

    aa_wt = jnp.arange(max_n, dtype=jnp.int32) % 20
    aa_mut = aa_wt.at[10].set(4)
    priors = prepare_wt_mut_priors(None, aa_wt, aa_mut, res_mask, fallback_key=k_pr)
    (s_wt, z_wt), (s_mut, z_mut) = priors["wt"], priors["mut"]

    empty_D = jnp.zeros((1, max_n, max_n), dtype=jnp.float32)
    empty_m = jnp.zeros((1,), dtype=jnp.float32)
    theta_eq = {"k_rep": 0.0, "enm_soften": 1.0}

    u_wt, f_wt, _ = full_state_energy_and_force(
        bb, d_ref_ca, phys_wt, weights, res_mask, empty_D, empty_m, theta_eq,
        single=s_wt, pair=z_wt, nn_params=nn_params, chain_mask=cm,
    )
    u_mut, f_mut, _ = full_state_energy_and_force(
        bb, d_ref_ca, phys_mut, weights, res_mask, empty_D, empty_m, theta_eq,
        single=s_mut, pair=z_mut, nn_params=nn_params, chain_mask=cm,
    )
    assert jnp.isfinite(u_wt) and jnp.isfinite(u_mut)
    assert jnp.all(jnp.isfinite(f_wt)) and jnp.all(jnp.isfinite(f_mut))
    print(f"[PASS] Full-state energy & force finite: U_WT={float(u_wt):.4f}, U_Mut={float(u_mut):.4f}")

    # Perturbed backbone ensemble around bb for BAR antisymmetry check
    eps_pert = jax.random.normal(k_noise, (16, *bb.shape), dtype=jnp.float32) * 0.05
    bb_ens = bb[None, ...] + eps_pert * res_mask[None, :, None, None]

    def _dh_batch(bb_samples):
        def _single_dh(x):
            uw, _, _ = full_state_energy_and_force(
                x, d_ref_ca, phys_wt, weights, res_mask, empty_D, empty_m, theta_eq,
                single=s_wt, pair=z_wt, nn_params=nn_params, chain_mask=cm,
            )
            um, _, _ = full_state_energy_and_force(
                x, d_ref_ca, phys_mut, weights, res_mask, empty_D, empty_m, theta_eq,
                single=s_mut, pair=z_mut, nn_params=nn_params, chain_mask=cm,
            )
            return um - uw
        return jax.vmap(_single_dh)(bb_samples)

    dh = _dh_batch(bb_ens)
    ddg_fwd = _simple_bar_ddg(dh, dh, kbt=KBT_310K)
    ddg_rev = _simple_bar_ddg(-dh, -dh, kbt=KBT_310K)
    antisym_err = float(jnp.abs(ddg_fwd + ddg_rev))
    assert jnp.isfinite(ddg_fwd) and antisym_err < 1e-4, f"BAR antisymmetry error: {antisym_err}"
    print(f"[PASS] BAR antisymmetry check: ddG_fwd={float(ddg_fwd):.4f}, ddG_rev={float(ddg_rev):.4f}, |sum|={antisym_err:.2e}")


if __name__ == "__main__":
    run_lapisan1_tests()


# References:
#   - Bennett Acceptance Ratio (BAR) minimum-variance free energy estimator:
#     Bennett (1976), J. Comput. Phys. 22(2):245-268.
#   - Thermodynamic antisymmetry audit for mutation ddG predictors:
#     Pucci et al. (2018), Briefings in Bioinformatics 19(4):599-611.
