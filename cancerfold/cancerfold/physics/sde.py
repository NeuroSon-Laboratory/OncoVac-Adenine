"""Overdamped Langevin SDE integrator with Equilibrium & Metadynamics modes via jax.lax.scan and jax.vmap."""
from typing import Dict, Optional, Tuple
import jax
import jax.numpy as jnp

from cancerfold.af2_core.stochastic_folding import full_state_energy_and_force
from cancerfold.physics.geometry import center, pairwise_dist
from cancerfold.physics.potentials import virtual_cb

R_GAS = 0.0019872      # Universal gas constant [kcal / (mol * K)]
TEMP_PHYS = 310.15     # Physiological temperature [K] (k_B * T = 0.6163 kcal/mol)
EPS_FORCE = 1e-8       # Regularization for norm-preserving force clipping


def clip_forces(
    forces: jnp.ndarray,
    residue_mask: jnp.ndarray,
    max_force: float = 50.0,
) -> jnp.ndarray:
    """Norm-preserving per-atom force clipping masked to active residues."""
    m = residue_mask.astype(jnp.float32)[:, None, None]
    f_masked = forces * m
    norms = jnp.sqrt(jnp.sum(f_masked ** 2, axis=-1, keepdims=True) + EPS_FORCE)
    scale = jnp.minimum(1.0, max_force / norms)
    return f_masked * scale


def langevin_step(
    backbone: jnp.ndarray,
    prng_key: jax.Array,
    d_ref_ca: jnp.ndarray,
    phys_state: jnp.ndarray,
    weights: jnp.ndarray,
    residue_mask: jnp.ndarray,
    history_cb_D: jnp.ndarray,
    history_mask: jnp.ndarray,
    theta: Dict[str, float],
    single: Optional[jnp.ndarray] = None,
    pair: Optional[jnp.ndarray] = None,
    nn_params: Optional[Dict[str, jnp.ndarray]] = None,
    chain_mask: Optional[jnp.ndarray] = None,
    dt: float = 0.002,
    gamma: float = 1.0,
    temp_k: float = TEMP_PHYS,
    max_force: float = 50.0,
    use_adaptive_dielectric: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Single Euler-Maruyama overdamped Langevin (Brownian dynamics) step:
        X_{t+dt} = X_t + (dt / gamma) * F(X_t) + sqrt(2 * k_B * T * dt / gamma) * xi_t
    Re-centers active backbone coordinates to prevent global float32 COM drift.
    """
    u_tot, f_raw, u_parts = full_state_energy_and_force(
        backbone=backbone,
        d_ref_ca=d_ref_ca,
        phys_state=phys_state,
        weights=weights,
        residue_mask=residue_mask,
        history_cb_D=history_cb_D,
        history_mask=history_mask,
        theta=theta,
        single=single,
        pair=pair,
        nn_params=nn_params,
        chain_mask=chain_mask,
        use_adaptive_dielectric=use_adaptive_dielectric,
    )

    m = residue_mask.astype(jnp.float32)[:, None, None]
    f_clip = clip_forces(f_raw, residue_mask, max_force=max_force)

    kbt = R_GAS * temp_k
    noise_std = jnp.sqrt(2.0 * kbt * dt / gamma)
    xi = jax.random.normal(prng_key, backbone.shape, dtype=jnp.float32) * m

    x_next = backbone + (dt / gamma) * f_clip + noise_std * xi
    x_next = center(x_next, mask=residue_mask)

    f_max = jnp.max(jnp.sqrt(jnp.sum((f_raw * m) ** 2, axis=-1) + EPS_FORCE))
    diag = {
        "u_phys": u_parts["u_phys"],
        "u_nn": u_parts["u_nn"],
        "f_max": f_max,
    }
    return x_next, u_tot, diag


def run_sde_trajectory(
    backbone_init: jnp.ndarray,
    prng_key: jax.Array,
    d_ref_ca: jnp.ndarray,
    phys_state: jnp.ndarray,
    weights: jnp.ndarray,
    residue_mask: jnp.ndarray,
    theta: Dict[str, float],
    single: Optional[jnp.ndarray] = None,
    pair: Optional[jnp.ndarray] = None,
    nn_params: Optional[Dict[str, jnp.ndarray]] = None,
    chain_mask: Optional[jnp.ndarray] = None,
    num_steps: int = 200,
    burn_in: int = 40,
    stride: int = 10,
    dt: float = 0.002,
    gamma: float = 1.0,
    temp_k: float = TEMP_PHYS,
    temp_start: Optional[float] = None,
    max_force: float = 50.0,
    max_hills: int = 16,
    deposit_every: int = 20,
    use_adaptive_dielectric: bool = True,
) -> Dict[str, jnp.ndarray]:
    """
    Runs a static-shape Langevin SDE trajectory via jax.lax.scan.
      - Equilibrium Mode (theta['k_rep'] == 0.0): exact Boltzmann sampling for BAR / SKEMPI 2.0.
      - Exploration Mode (theta['k_rep'] > 0.0): deposits pocket Gaussian hills into a static
        ring buffer every `deposit_every` steps to sample cryptic-pocket opening.
    """
    max_n = backbone_init.shape[0]
    use_meta = float(theta.get("k_rep", 0.0)) > 0.0
    t_start = float(temp_k) if temp_start is None else float(temp_start)

    init_bb = center(backbone_init, mask=residue_mask)
    init_hist_D = jnp.zeros((max_hills, max_n, max_n), dtype=jnp.float32)
    init_hist_m = jnp.zeros((max_hills,), dtype=jnp.float32)
    init_hill_idx = jnp.array(0, dtype=jnp.int32)

    step_keys = jax.random.split(prng_key, num_steps)
    step_indices = jnp.arange(num_steps, dtype=jnp.int32)

    def _scan_body(carry, inputs):
        x_curr, hist_D, hist_m, hill_idx = carry
        step_idx, step_key = inputs

        # Optional linear temperature annealing during burn-in; constant temp_k afterward
        frac = jnp.minimum(1.0, step_idx.astype(jnp.float32) / jnp.maximum(1.0, float(burn_in)))
        t_curr = t_start + frac * (float(temp_k) - t_start)

        x_next, u_tot, diag = langevin_step(
            backbone=x_curr,
            prng_key=step_key,
            d_ref_ca=d_ref_ca,
            phys_state=phys_state,
            weights=weights,
            residue_mask=residue_mask,
            history_cb_D=hist_D,
            history_mask=hist_m,
            theta=theta,
            single=single,
            pair=pair,
            nn_params=nn_params,
            chain_mask=chain_mask,
            dt=dt,
            gamma=gamma,
            temp_k=t_curr,
            max_force=max_force,
            use_adaptive_dielectric=use_adaptive_dielectric,
        )

        def _deposit(state):
            h_D, h_m, h_i = state
            d_cb = pairwise_dist(virtual_cb(x_next))
            slot = h_i % max_hills
            h_D = h_D.at[slot].set(d_cb)
            h_m = h_m.at[slot].set(1.0)
            return h_D, h_m, h_i + 1

        def _no_deposit(state):
            return state

        should_deposit = (
            jnp.asarray(use_meta)
            & (step_idx >= burn_in)
            & (((step_idx - burn_in) % deposit_every) == 0)
        )
        hist_D, hist_m, hill_idx = jax.lax.cond(
            should_deposit, _deposit, _no_deposit, (hist_D, hist_m, hill_idx)
        )

        return (x_next, hist_D, hist_m, hill_idx), (x_next, u_tot, diag["f_max"])

    (final_bb, final_hist_D, final_hist_m, _), (all_bb, all_u, all_fmax) = jax.lax.scan(
        _scan_body,
        (init_bb, init_hist_D, init_hist_m, init_hill_idx),
        (step_indices, step_keys),
    )

    traj = all_bb[burn_in::stride]
    energies = all_u[burn_in::stride]
    f_max_trace = all_fmax[burn_in::stride]

    return {
        "trajectory": traj,
        "energies": energies,
        "f_max": f_max_trace,
        "final_backbone": final_bb,
        "history_cb_D": final_hist_D,
        "history_mask": final_hist_m,
    }


def sample_ensemble(
    backbone_init: jnp.ndarray,
    prng_key: jax.Array,
    d_ref_ca: jnp.ndarray,
    phys_state: jnp.ndarray,
    weights: jnp.ndarray,
    residue_mask: jnp.ndarray,
    theta: Dict[str, float],
    single: Optional[jnp.ndarray] = None,
    pair: Optional[jnp.ndarray] = None,
    nn_params: Optional[Dict[str, jnp.ndarray]] = None,
    chain_mask: Optional[jnp.ndarray] = None,
    num_replicas: int = 4,
    num_steps: int = 200,
    burn_in: int = 40,
    stride: int = 20,
    dt: float = 0.002,
    gamma: float = 1.0,
    temp_k: float = TEMP_PHYS,
    max_force: float = 50.0,
    max_hills: int = 16,
    deposit_every: int = 20,
    use_adaptive_dielectric: bool = True,
) -> Dict[str, jnp.ndarray]:
    """
    Runs `num_replicas` independent Langevin walkers in parallel via jax.vmap and
    flattens post-burn-in snapshots into an (M, MAX_N, 3, 3) ensemble for BAR.
    """
    rep_keys = jax.random.split(prng_key, num_replicas)

    def _run_one(k):
        return run_sde_trajectory(
            backbone_init=backbone_init,
            prng_key=k,
            d_ref_ca=d_ref_ca,
            phys_state=phys_state,
            weights=weights,
            residue_mask=residue_mask,
            theta=theta,
            single=single,
            pair=pair,
            nn_params=nn_params,
            chain_mask=chain_mask,
            num_steps=num_steps,
            burn_in=burn_in,
            stride=stride,
            dt=dt,
            gamma=gamma,
            temp_k=temp_k,
            max_force=max_force,
            max_hills=max_hills,
            deposit_every=deposit_every,
            use_adaptive_dielectric=use_adaptive_dielectric,
        )

    out = jax.vmap(_run_one)(rep_keys)
    traj = out["trajectory"]    # (num_replicas, K, MAX_N, 3, 3)
    energies = out["energies"]  # (num_replicas, K)
    flat_traj = traj.reshape((-1,) + traj.shape[2:])
    flat_energies = energies.reshape((-1,))

    return {
        "ensemble": flat_traj,
        "energies": flat_energies,
        "trajectories_by_replica": traj,
        "max_force": jnp.max(out["f_max"]),
    }


# References:
#   - Overdamped Langevin / Brownian dynamics & Euler-Maruyama discretization:
#     Ermak & McCammon (1978), J. Chem. Phys. 69(4):1352-1360;
#     Kloeden & Platen (1992), Numerical Solution of Stochastic Differential Equations, Springer.
#   - History-dependent Gaussian hill deposition (Metadynamics, Exploration Mode):
#     Laio & Parrinello (2002), PNAS 99(20):12562-12566.
#   - Parallel replica sampling for equilibrium free-energy estimation:
#     Shirts & Chodera (2008), J. Chem. Phys. 129(12):124105.
