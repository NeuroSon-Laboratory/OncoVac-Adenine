"""Ensemble ΔΔG_eff estimator, ligand steric scoring, and lab benchmark metrics."""
"""Differentiable Bennett Acceptance Ratio (BAR), Bound-Unbound SKEMPI 2.0 cycle, and white-box diagnostics."""
import functools
from typing import Dict, Optional, Tuple
import jax
import jax.numpy as jnp

from cancerfold.af2_core.stochastic_folding import full_state_energy
from cancerfold.physics.sde import R_GAS, TEMP_PHYS

KBT_PHYS = R_GAS * TEMP_PHYS  # ~0.61633 kcal/mol at 310.15 K


def _solve_bar_root(
    dh_wt: jnp.ndarray,
    dh_mut: jnp.ndarray,
    kbt: float,
    num_iters: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solves the Bennett Acceptance Ratio (BAR) root c* such that:
        mean(sigmoid(-beta * (dh_wt - c*))) - mean(sigmoid(-beta * (-dh_mut + c*))) = 0
    where BOTH dh_wt and dh_mut are defined as Delta H(X) = H_Mut(X) - H_WT(X)
    evaluated on the WT ensemble and Mut ensemble, respectively.
    Returns (ddg = c* - offset, c*).
    """
    beta = 1.0 / kbt
    n_f = jnp.asarray(dh_wt.shape[0], dtype=jnp.float32)
    n_r = jnp.asarray(dh_mut.shape[0], dtype=jnp.float32)
    offset = kbt * jnp.log(n_r / n_f)

    margin = 12.0 * kbt + jnp.abs(offset)
    lo = jnp.minimum(jnp.min(dh_wt), jnp.min(dh_mut)) - margin
    hi = jnp.maximum(jnp.max(dh_wt), jnp.max(dh_mut)) + margin

    def _residual(c: jnp.ndarray) -> jnp.ndarray:
        f_term = jnp.mean(jax.nn.sigmoid(-beta * (dh_wt - c)))
        r_term = jnp.mean(jax.nn.sigmoid(-beta * (-dh_mut + c)))
        return f_term - r_term

    def _step(_, bounds):
        low, high = bounds
        mid = 0.5 * (low + high)
        res = _residual(mid)
        return jnp.where(res < 0.0, mid, low), jnp.where(res < 0.0, high, mid)

    lo_f, hi_f = jax.lax.fori_loop(0, num_iters, _step, (lo, hi))
    c_star = 0.5 * (lo_f + hi_f)
    return c_star - offset, c_star


@functools.partial(jax.custom_vjp, nondiff_argnums=(2, 3))
def bar_ddg(
    dh_wt: jnp.ndarray,
    dh_mut: jnp.ndarray,
    kbt: float = KBT_PHYS,
    num_iters: int = 60,
) -> jnp.ndarray:
    """
    Exact, thermodynamically antisymmetric, implicitly differentiable BAR estimator.
    Guarantees:
      - Exact WT <-> Mut antisymmetry: bar_ddg(-dh_mut, -dh_wt) == -bar_ddg(dh_wt, dh_mut)
      - Exact smooth gradients via Implicit Function Theorem for Optax training.
    """
    ddg, _ = _solve_bar_root(dh_wt, dh_mut, kbt, num_iters)
    return ddg


def _bar_fwd(
    dh_wt: jnp.ndarray,
    dh_mut: jnp.ndarray,
    kbt: float,
    num_iters: int,
):
    ddg, c_star = _solve_bar_root(dh_wt, dh_mut, kbt, num_iters)
    return ddg, (dh_wt, dh_mut, c_star)


def _bar_bwd(kbt: float, num_iters: int, res, g_out: jnp.ndarray):
    dh_wt, dh_mut, c_star = res
    beta = 1.0 / kbt
    n_f = float(dh_wt.shape[0])
    n_r = float(dh_mut.shape[0])

    # Implicit Function Theorem: d(c*)/d(w_i) = (f_i(1-f_i)/N_F) / [mean(f(1-f)) + mean(r(1-r))]
    f = jax.nn.sigmoid(-beta * (dh_wt - c_star))
    r = jax.nn.sigmoid(-beta * (-dh_mut + c_star))
    w_f = f * (1.0 - f) / n_f
    w_r = r * (1.0 - r) / n_r
    denom = jnp.sum(w_f) + jnp.sum(w_r)

    safe_denom = jnp.maximum(denom, 1e-12)
    grad_wt = jnp.where(denom > 1e-10, w_f / safe_denom, 0.5 / n_f)
    grad_mut = jnp.where(denom > 1e-10, w_r / safe_denom, 0.5 / n_r)
    return g_out * grad_wt, g_out * grad_mut


bar_ddg.defvjp(_bar_fwd, _bar_bwd)


def bar_overlap_diagnostic(
    dh_wt: jnp.ndarray,
    dh_mut: jnp.ndarray,
    ddg: jnp.ndarray,
    kbt: float = KBT_PHYS,
) -> Dict[str, jnp.ndarray]:
    """
    Computes phase-space overlap & asymptotic BAR statistical uncertainty (Shirts et al. 2003).
    `overlap_score` in [0, 1]: values near 0 warn that WT and Mut ensembles barely overlap.
    """
    beta = 1.0 / kbt
    n_f = jnp.asarray(dh_wt.shape[0], dtype=jnp.float32)
    n_r = jnp.asarray(dh_mut.shape[0], dtype=jnp.float32)
    c_star = ddg + kbt * jnp.log(n_r / n_f)

    f = jax.nn.sigmoid(-beta * (dh_wt - c_star))
    r = jax.nn.sigmoid(-beta * (-dh_mut + c_star))
    # Peak overlap occurs when f = r = 0.5 -> 4*f*(1-f) = 1.0
    overlap_score = 2.0 * (jnp.mean(f * (1.0 - f)) + jnp.mean(r * (1.0 - r)))

    inv_var = jnp.sum(f * (1.0 - f)) + jnp.sum(r * (1.0 - r))
    var_bar = (kbt ** 2) * (1.0 / jnp.maximum(inv_var, 1e-8) - (1.0 / n_f + 1.0 / n_r) / 4.0)
    std_err = jnp.sqrt(jnp.maximum(var_bar, 0.0))

    return {
        "overlap_score": overlap_score,
        "std_err": std_err,
        "dh_wt_mean": jnp.mean(dh_wt),
        "dh_mut_mean": jnp.mean(dh_mut),
    }


def evaluate_dh_batch(
    backbone_ens: jnp.ndarray,
    d_ref_ca: jnp.ndarray,
    phys_wt: jnp.ndarray,
    phys_mut: jnp.ndarray,
    weights: jnp.ndarray,
    residue_mask: jnp.ndarray,
    theta: Dict[str, float],
    single_wt: Optional[jnp.ndarray] = None,
    pair_wt: Optional[jnp.ndarray] = None,
    single_mut: Optional[jnp.ndarray] = None,
    pair_mut: Optional[jnp.ndarray] = None,
    nn_params: Optional[Dict[str, jnp.ndarray]] = None,
    chain_mask: Optional[jnp.ndarray] = None,
    use_adaptive_dielectric: bool = True,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Vectorized Delta H(X) = H_Mut(X) - H_WT(X) over an (M, MAX_N, 3, 3) equilibrium ensemble
    (zero Metadynamics hills: k_rep = 0.0), returning white-box physical vs. neural decomposition.
    """
    max_n = backbone_ens.shape[1]
    empty_D = jnp.zeros((1, max_n, max_n), dtype=jnp.float32)
    empty_m = jnp.zeros((1,), dtype=jnp.float32)
    theta_eq = dict(theta)
    theta_eq["k_rep"] = 0.0

    def _single_dh(x_bb: jnp.ndarray):
        u_wt, p_wt = full_state_energy(
            x_bb, d_ref_ca, phys_wt, weights, residue_mask, empty_D, empty_m, theta_eq,
            single=single_wt, pair=pair_wt, nn_params=nn_params,
            chain_mask=chain_mask, use_adaptive_dielectric=use_adaptive_dielectric,
        )
        u_mut, p_mut = full_state_energy(
            x_bb, d_ref_ca, phys_mut, weights, residue_mask, empty_D, empty_m, theta_eq,
            single=single_mut, pair=pair_mut, nn_params=nn_params,
            chain_mask=chain_mask, use_adaptive_dielectric=use_adaptive_dielectric,
        )
        return (
            u_mut - u_wt,
            p_mut["u_phys"] - p_wt["u_phys"],
            p_mut["u_nn"] - p_wt["u_nn"],
        )

    dh_tot, dh_phys, dh_nn = jax.vmap(_single_dh)(backbone_ens)
    return dh_tot, {"dh_phys": dh_phys, "dh_nn": dh_nn}


def binding_ddg_bar(
    complex_wt_ens: jnp.ndarray,
    complex_mut_ens: jnp.ndarray,
    d_ref_ca: jnp.ndarray,
    phys_wt: jnp.ndarray,
    phys_mut: jnp.ndarray,
    weights: jnp.ndarray,
    partner_a_mask: jnp.ndarray,
    partner_b_mask: jnp.ndarray,
    theta: Dict[str, float],
    single_wt: Optional[jnp.ndarray] = None,
    pair_wt: Optional[jnp.ndarray] = None,
    single_mut: Optional[jnp.ndarray] = None,
    pair_mut: Optional[jnp.ndarray] = None,
    nn_params: Optional[Dict[str, jnp.ndarray]] = None,
    chain_mask: Optional[jnp.ndarray] = None,
    unbound_a_wt_ens: Optional[jnp.ndarray] = None,
    unbound_a_mut_ens: Optional[jnp.ndarray] = None,
    unbound_b_wt_ens: Optional[jnp.ndarray] = None,
    unbound_b_mut_ens: Optional[jnp.ndarray] = None,
    temp_k: float = TEMP_PHYS,
    use_adaptive_dielectric: bool = True,
) -> Dict[str, jnp.ndarray]:
    """
    Exact SKEMPI 2.0 / Protein-Drug Bound-Unbound Thermodynamic Cycle via BAR:
        ddG_bind = dG_complex(WT -> Mut) - [dG_unbound_A(WT -> Mut) + dG_unbound_B(WT -> Mut)]

    If separate unbound ensembles are not supplied, evaluates the rigid-unbound leg on the
    complex ensembles masked strictly to Partner A and Partner B (inter-partner terms = 0).
    """
    kbt = R_GAS * temp_k
    complex_mask = jnp.clip(partner_a_mask + partner_b_mask, 0.0, 1.0)

    # 1. Bound complex leg
    dh_c_wt, parts_c_wt = evaluate_dh_batch(
        complex_wt_ens, d_ref_ca, phys_wt, phys_mut, weights, complex_mask, theta,
        single_wt=single_wt, pair_wt=pair_wt, single_mut=single_mut, pair_mut=pair_mut,
        nn_params=nn_params, chain_mask=chain_mask, use_adaptive_dielectric=use_adaptive_dielectric,
    )
    dh_c_mut, parts_c_mut = evaluate_dh_batch(
        complex_mut_ens, d_ref_ca, phys_wt, phys_mut, weights, complex_mask, theta,
        single_wt=single_wt, pair_wt=pair_wt, single_mut=single_mut, pair_mut=pair_mut,
        nn_params=nn_params, chain_mask=chain_mask, use_adaptive_dielectric=use_adaptive_dielectric,
    )
    ddg_complex = bar_ddg(dh_c_wt, dh_c_mut, kbt)

    # 2. Unbound Partner A leg (e.g., antibody H+L or target protein)
    ens_a_wt = complex_wt_ens if unbound_a_wt_ens is None else unbound_a_wt_ens
    ens_a_mut = complex_mut_ens if unbound_a_mut_ens is None else unbound_a_mut_ens
    dh_a_wt, _ = evaluate_dh_batch(
        ens_a_wt, d_ref_ca, phys_wt, phys_mut, weights, partner_a_mask, theta,
        single_wt=single_wt, pair_wt=pair_wt, single_mut=single_mut, pair_mut=pair_mut,
        nn_params=nn_params, chain_mask=chain_mask, use_adaptive_dielectric=use_adaptive_dielectric,
    )
    dh_a_mut, _ = evaluate_dh_batch(
        ens_a_mut, d_ref_ca, phys_wt, phys_mut, weights, partner_a_mask, theta,
        single_wt=single_wt, pair_wt=pair_wt, single_mut=single_mut, pair_mut=pair_mut,
        nn_params=nn_params, chain_mask=chain_mask, use_adaptive_dielectric=use_adaptive_dielectric,
    )
    ddg_unbound_a = bar_ddg(dh_a_wt, dh_a_mut, kbt)

    # 3. Unbound Partner B leg (e.g., antigen or small-molecule drug)
    ens_b_wt = complex_wt_ens if unbound_b_wt_ens is None else unbound_b_wt_ens
    ens_b_mut = complex_mut_ens if unbound_b_mut_ens is None else unbound_b_mut_ens
    dh_b_wt, _ = evaluate_dh_batch(
        ens_b_wt, d_ref_ca, phys_wt, phys_mut, weights, partner_b_mask, theta,
        single_wt=single_wt, pair_wt=pair_wt, single_mut=single_mut, pair_mut=pair_mut,
        nn_params=nn_params, chain_mask=chain_mask, use_adaptive_dielectric=use_adaptive_dielectric,
    )
    dh_b_mut, _ = evaluate_dh_batch(
        ens_b_mut, d_ref_ca, phys_wt, phys_mut, weights, partner_b_mask, theta,
        single_wt=single_wt, pair_wt=pair_wt, single_mut=single_mut, pair_mut=pair_mut,
        nn_params=nn_params, chain_mask=chain_mask, use_adaptive_dielectric=use_adaptive_dielectric,
    )
    ddg_unbound_b = bar_ddg(dh_b_wt, dh_b_mut, kbt)

    ddg_unbound = ddg_unbound_a + ddg_unbound_b
    ddg_bind = ddg_complex - ddg_unbound
    diag = bar_overlap_diagnostic(dh_c_wt, dh_c_mut, ddg_complex, kbt)

    return {
        "ddg_bind": ddg_bind,
        "ddg_complex": ddg_complex,
        "ddg_unbound": ddg_unbound,
        "overlap_score": diag["overlap_score"],
        "std_err": diag["std_err"],
        "dh_phys_mean": 0.5 * (jnp.mean(parts_c_wt["dh_phys"]) + jnp.mean(parts_c_mut["dh_phys"])),
        "dh_nn_mean": 0.5 * (jnp.mean(parts_c_wt["dh_nn"]) + jnp.mean(parts_c_mut["dh_nn"])),
    }


def skempi_metrics(y_pred: jnp.ndarray, y_true: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """Computes standard SKEMPI 2.0 evaluation metrics: Pearson r, Spearman rho, RMSE, MAE."""
    yp = jnp.asarray(y_pred, dtype=jnp.float32)
    yt = jnp.asarray(y_true, dtype=jnp.float32)
    diff = yp - yt
    rmse = jnp.sqrt(jnp.mean(diff ** 2))
    mae = jnp.mean(jnp.abs(diff))

    yp_c = yp - jnp.mean(yp)
    yt_c = yt - jnp.mean(yt)
    pearson = jnp.sum(yp_c * yt_c) / jnp.sqrt(
        jnp.maximum(jnp.sum(yp_c ** 2) * jnp.sum(yt_c ** 2), 1e-12)
    )

    # Exact Spearman rank correlation via argsort ranks
    rank_p = jnp.argsort(jnp.argsort(yp)).astype(jnp.float32)
    rank_t = jnp.argsort(jnp.argsort(yt)).astype(jnp.float32)
    rp_c = rank_p - jnp.mean(rank_p)
    rt_c = rank_t - jnp.mean(rank_t)
    spearman = jnp.sum(rp_c * rt_c) / jnp.sqrt(
        jnp.maximum(jnp.sum(rp_c ** 2) * jnp.sum(rt_c ** 2), 1e-12)
    )

    return {
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "rmse": rmse,
        "mae": mae,
    }


# References:
#   - Bennett Acceptance Ratio (BAR) minimum-variance free energy estimator:
#     Bennett (1976), J. Comput. Phys. 22(2):245-268.
#   - Asymptotic variance & phase-space overlap diagnostics for BAR:
#     Shirts, Bair, Hooker & Pande (2003), Phys. Rev. Lett. 91(14):140601.
#   - Thermodynamic cycle for protein-protein & protein-ligand binding ddG (SKEMPI 2.0):
#     Jankauskaite et al. (2019), Bioinformatics 35(3):462-469.
#   - Exact thermodynamic antisymmetry audit for ddG predictors:
#     Pucci et al. (2018), Briefings in Bioinformatics 19(4):599-611;
#     Benevenuta et al. (2021), Bioinformatics 37(20):3425-3433.
