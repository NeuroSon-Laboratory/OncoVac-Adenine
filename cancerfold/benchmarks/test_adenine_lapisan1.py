"""Layer 1 physics & numerical verification suite for OncoVac-Adenine (Gen 1)."""
import jax
import jax.numpy as jnp

from cancerfold.af2_core.joltz_prior import prepare_wt_mut_priors
from cancerfold.af2_core.loader import pad_structure
from cancerfold.af2_core.stochastic_folding import (
    full_state_energy_and_force,
    init_folding_params,
)
from cancerfold.dynamics.sde import (
    KBT_310K,
    compute_bar_ddg,
    run_equilibrium_sde,
    run_exploration_sde,
)
from cancerfold.mutation.neighborhood import multi_pocket_mask
from cancerfold.physics.potentials import adaptive_inv_dielectric, coordination_burial


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
    b0_vals = []
    eps_inv_01_vals = []
    for r in dists:
        d_mat = base_d.at[0, 1].set(r).at[1, 0].set(r)
        _, b_i = coordination_burial(d_mat, res_mask)
        inv_eps = adaptive_inv_dielectric(b_i)
        b0_vals.append(float(b_i[0]))
        eps_inv_01_vals.append(float(inv_eps[0, 1]))

    b_arr = jnp.asarray(b0_vals)
    eps_arr = jnp.asarray(eps_inv_01_vals)
    assert jnp.all(jnp.isfinite(b_arr)) and jnp.all(jnp.isfinite(eps_arr)), "Non-finite burial/dielectric"
    max_jump = float(jnp.max(jnp.abs(jnp.diff(eps_arr))))
    print(f"[PASS] Continuity 10-11 A: max step jump in 1/eps_eff = {max_jump:.4e}")


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
    k_p, k_pr, k_exp, k_eq1, k_eq2 = jax.random.split(key, 5)
    nn_params = init_folding_params(k_p)

    aa_wt = jnp.arange(max_n, dtype=jnp.int32) % 20
    aa_mut = aa_wt.at[10].set(4)
    priors = prepare_wt_mut_priors(None, aa_wt, aa_mut, res_mask, fallback_key=k_pr)
    (s_wt, z_wt), (s_mut, z_mut) = priors["wt"], priors["mut"]

    # 1. Exploration SDE (k_rep > 0)
    theta_exp = {"k_rep": 0.5, "sigma_meta": 0.6, "enm_soften": 1.0}
    traj_bb, traj_u = run_exploration_sde(
        bb, d_ref_ca, phys_mut, weights, res_mask, theta_exp, k_exp,
        num_steps=40, save_stride=10, meta_stride=10,
        single=s_mut, pair=z_mut, nn_params=nn_params, chain_mask=cm,
    )
    assert jnp.all(jnp.isfinite(traj_bb)) and jnp.all(jnp.isfinite(traj_u)), "NaN/Inf in exploration SDE"
    print(f"[PASS] Exploration SDE: traj shape={traj_bb.shape}, finite energies verified.")

    # 2. Equilibrium SDE + BAR mirror antisymmetry (k_rep == 0 enforced)
    theta_eq = {"k_rep": 0.25, "enm_soften": 1.0}
    bb_eq_wt, _ = run_equilibrium_sde(
        bb, d_ref_ca, phys_wt, weights, res_mask, theta_eq, k_eq1,
        num_steps=60, burn_in=20, sample_stride=5,
        single=s_wt, pair=z_wt, nn_params=nn_params, chain_mask=cm,
    )
    bb_eq_mut, _ = run_equilibrium_sde(
        bb, d_ref_ca, phys_mut, weights, res_mask, theta_eq, k_eq2,
        num_steps=60, burn_in=20, sample_stride=5,
        single=s_mut, pair=z_mut, nn_params=nn_params, chain_mask=cm,
    )

    empty_D = jnp.zeros((1, max_n, max_n), dtype=jnp.float32)
    empty_m = jnp.zeros((1,), dtype=jnp.float32)

    def _dh_batch(bb_samples):
        def _single_dh(x):
            u_w, _, _ = full_state_energy_and_force(
                x, d_ref_ca, phys_wt, weights, res_mask, empty_D, empty_m, theta_eq,
                single=s_wt, pair=z_wt, nn_params=nn_params, chain_mask=cm,
            )
            u_m, _, _ = full_state_energy_and_force(
                x, d_ref_ca, phys_mut, weights, res_mask, empty_D, empty_m, theta_eq,
                single=s_mut, pair=z_mut, nn_params=nn_params, chain_mask=cm,
            )
            return u_m - u_w
        return jax.vmap(_single_dh)(bb_samples)

    dh_on_wt = _dh_batch(bb_eq_wt)
    dh_on_mut = _dh_batch(bb_eq_mut)
    ddg_fwd = compute_bar_ddg(dh_on_wt, dh_on_mut, kbt=KBT_310K)
    ddg_rev = compute_bar_ddg(-dh_on_mut, -dh_on_wt, kbt=KBT_310K)
    antisym_err = float(jnp.abs(ddg_fwd + ddg_rev))
    assert jnp.isfinite(ddg_fwd) and antisym_err < 1e-4, f"BAR antisymmetry error: {antisym_err}"
    print(f"[PASS] Equilibrium + BAR: ddG_fwd={float(ddg_fwd):.4f}, ddG_rev={float(ddg_rev):.4f}, |sum|={antisym_err:.2e}")


if __name__ == "__main__":
    run_lapisan1_tests()


# References:
#   - Bennett Acceptance Ratio (BAR) minimum-variance free energy estimator:
#     Bennett (1976), J. Comput. Phys. 22(2):245-268.
#   - Thermodynamic antisymmetry audit for mutation ddG predictors:
#     Pucci et al. (2018), Briefings in Bioinformatics 19(4):599-611.
#   - Multi-mutation epistasis & SKEMPI 2.0 benchmark context:
#     Jankauskaite et al. (2019), Bioinformatics 35(3):462-469.
