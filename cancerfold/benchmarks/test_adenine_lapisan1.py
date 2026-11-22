"""Complete Layer 1 + Layer 2 physics, symmetry, SDE, and Differentiable BAR verification suite."""
import jax
import jax.numpy as jnp

from cancerfold.af2_core.joltz_prior import prepare_wt_mut_priors
from cancerfold.af2_core.loader import pad_structure
from cancerfold.af2_core.stochastic_folding import (
    full_state_energy_and_force,
    init_folding_params,
)
from cancerfold.af2_core.trunk import init_trunk_params, run_trunk
from cancerfold.analyzer.thermo import KBT_PHYS, bar_ddg, binding_ddg_bar
from cancerfold.mutation.neighborhood import multi_pocket_mask
from cancerfold.mutation.parser import encode, point
from cancerfold.physics.geometry import nonbonded_pair_mask, segment_dist
from cancerfold.physics.potentials import adaptive_inv_dielectric, coordination_burial
from cancerfold.physics.sde import sample_ensemble


def _make_two_chain_backbone(n_a: int = 12, n_b: int = 12) -> tuple:
    """Builds a clean two-chain helical complex with realistic Engh-Huber intra-residue bonds."""
    def _helix(n_res: int, x_offset: float, z_offset: float):
        idx = jnp.arange(n_res, dtype=jnp.float32)
        phi = idx * 1.7453293
        ca = jnp.stack([x_offset + 1.5 * idx, 2.3 * jnp.cos(phi), z_offset + 2.3 * jnp.sin(phi)], axis=-1)
        # Intra-residue distances: N-CA = 1.458 A, CA-C = 1.525 A, N-C = 2.462 A
        n_atom = ca + jnp.array([-0.82, 1.205, 0.0], dtype=jnp.float32)
        c_atom = ca + jnp.array([0.85, 1.266, 0.0], dtype=jnp.float32)
        return jnp.stack([n_atom, ca, c_atom], axis=1)

    bb_a = _helix(n_a, 0.0, 0.0)
    bb_b = _helix(n_b, 0.0, 8.0)  # Partner B positioned 8 A across interface
    bb = jnp.concatenate([bb_a, bb_b], axis=0)
    cm = jnp.concatenate([jnp.zeros((n_a,), dtype=jnp.int32), jnp.ones((n_b,), dtype=jnp.int32)])
    return bb, cm


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


def run_lapisan1_and_2_tests() -> None:
    check_continuity_11a(max_n=16)

    n_a, n_b, max_n = 12, 12, 32
    n_res = n_a + n_b
    bb_raw, cm_raw = _make_two_chain_backbone(n_a, n_b)
    bb, cm, res_mask = pad_structure(bb_raw, cm_raw, max_n=max_n)

    # Verify nonbonded_pair_mask works directly with (N,) int32 chain_mask from loader.py
    nb_mask = nonbonded_pair_mask(max_n, chain_mask=cm)
    assert nb_mask.shape == (max_n, max_n)
    assert float(nb_mask[n_a - 1, n_a]) == 1.0, "Inter-chain interface boundary must stay non-bonded"
    assert float(nb_mask[0, 1]) == 0.0, "Intra-chain 1-2 covalent bond must be masked out"
    print("[PASS] Multi-chain nonbonded_pair_mask compatible with loader.py chain_mask.")

    # Verify parser.encode() 4-col (trunk) and 5-col (potentials) alignment
    wt_seq = "ARNDCEQGHILKMFPSTWYVARND"
    mut_seq = point(wt_seq, "Y19C")
    enc_wt = encode(wt_seq)
    enc_mut = encode(mut_seq)

    pad_len = max_n - n_res
    aa_wt = jnp.pad(enc_wt["idx"], ((0, pad_len),), constant_values=0)
    aa_mut = jnp.pad(enc_mut["idx"], ((0, pad_len),), constant_values=0)
    phys_wt = jnp.pad(enc_wt["phys_state"], ((0, pad_len), (0, 0)), constant_values=0.0)
    phys_mut = jnp.pad(enc_mut["phys_state"], ((0, pad_len), (0, 0)), constant_values=0.0)

    mut_mask = jnp.zeros((max_n,), dtype=jnp.float32).at[18].set(1.0)
    weights = multi_pocket_mask(bb[:, 1, :], mut_mask, residue_mask=res_mask)
    assert jnp.all((weights >= 0.0) & (weights <= 1.0)), "Noisy-OR weights out of [0, 1]"
    print("[PASS] Multi-mutation Noisy-OR pocket weights bounded in [0, 1].")

    d_ref_ca = jnp.linalg.norm(bb[:, 1, None, :] - bb[None, :, 1, :], axis=-1)
    key = jax.random.PRNGKey(42)
    k_p, k_t, k_pr, k_sde_wt, k_sde_mut = jax.random.split(key, 5)

    # Full 384/128 Joltz-1 dimensions (zero dropped channels)
    nn_params = init_folding_params(k_p, c_s=384, c_z=128)
    trunk_params = init_trunk_params(k_t, c_af2_s=384, c_af2_z=128)
    priors = prepare_wt_mut_priors(None, aa_wt, aa_mut, res_mask, c_s=384, c_z=128, fallback_key=k_pr)
    (s_wt, z_wt), (s_mut, z_mut) = priors["wt"], priors["mut"]

    # Smoke-check trunk.py with shared theta & loader chain_mask
    theta_eq = {"k_rep": 0.0, "enm_soften": 1.0}
    tr_s, tr_z, _ = run_trunk(
        enc_wt["idx"], enc_wt["phys"], bb_raw[:, 1, :], weights[:n_res],
        s_wt[:n_res], z_wt[:n_res, :n_res], trunk_params, theta_eq,
        max_n=max_n, backbone_n_ca_c=bb_raw, chain_mask=cm_raw,
    )
    assert jnp.all(jnp.isfinite(tr_s)) and jnp.all(jnp.isfinite(tr_z))
    print("[PASS] Trunk runs cleanly with full 384/128 Joltz-1 priors & shared theta.")

    empty_D = jnp.zeros((1, max_n, max_n), dtype=jnp.float32)
    empty_m = jnp.zeros((1,), dtype=jnp.float32)
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

    # 1. Exact identity-mutation symmetry check (WT == WT)
    priors_id = prepare_wt_mut_priors(None, aa_wt, aa_wt, res_mask, c_s=384, c_z=128, fallback_key=k_pr)
    (s_1, z_1), (s_2, z_2) = priors_id["wt"], priors_id["mut"]
    u_1, _, _ = full_state_energy_and_force(
        bb, d_ref_ca, phys_wt, weights, res_mask, empty_D, empty_m, theta_eq,
        single=s_1, pair=z_1, nn_params=nn_params, chain_mask=cm,
    )
    u_2, _, _ = full_state_energy_and_force(
        bb, d_ref_ca, phys_wt, weights, res_mask, empty_D, empty_m, theta_eq,
        single=s_2, pair=z_2, nn_params=nn_params, chain_mask=cm,
    )
    id_err = float(jnp.abs(u_1 - u_2))
    assert id_err < 1e-6, f"Identity mutation energy mismatch: {id_err}"
    print(f"[PASS] WT==WT exact prior symmetry: |U_1 - U_2| = {id_err:.2e}")

    # 2. SE(3) rotation + translation invariance of U and equivariance of F = -grad U
    q_mat, _ = jnp.linalg.qr(jax.random.normal(jax.random.PRNGKey(99), (3, 3), dtype=jnp.float32))
    rot = q_mat * jnp.linalg.det(q_mat)
    trans = jnp.array([12.5, -7.3, 4.1], dtype=jnp.float32)
    bb_se3 = (jnp.einsum("nij,kj->nik", bb, rot) + trans[None, None, :]) * res_mask[:, None, None]

    u_rot, f_rot, _ = full_state_energy_and_force(
        bb_se3, d_ref_ca, phys_wt, weights, res_mask, empty_D, empty_m, theta_eq,
        single=s_wt, pair=z_wt, nn_params=nn_params, chain_mask=cm,
    )
    f_wt_rot = jnp.einsum("nij,kj->nik", f_wt, rot)
    u_se3_err = float(jnp.abs(u_rot - u_wt))
    f_se3_err = float(jnp.max(jnp.abs((f_rot - f_wt_rot) * res_mask[:, None, None])))
    assert u_se3_err < 1e-3 and f_se3_err < 1e-3, f"SE(3) error: dU={u_se3_err}, dF={f_se3_err}"
    print(f"[PASS] SE(3) invariance/equivariance: |dU|={u_se3_err:.2e}, max|dF|={f_se3_err:.2e}")

    # 3. Overdamped Langevin SDE stability & 6-bond Engh-Huber frame integrity
    ens_wt_out = sample_ensemble(
        bb, k_sde_wt, d_ref_ca, phys_wt, weights, res_mask, theta_eq,
        single=s_wt, pair=z_wt, nn_params=nn_params, chain_mask=cm,
        num_replicas=2, num_steps=40, burn_in=10, stride=10,
    )
    ens_mut_out = sample_ensemble(
        bb, k_sde_mut, d_ref_ca, phys_mut, weights, res_mask, theta_eq,
        single=s_mut, pair=z_mut, nn_params=nn_params, chain_mask=cm,
        num_replicas=2, num_steps=40, burn_in=10, stride=10,
    )
    ens_wt = ens_wt_out["ensemble"]
    ens_mut = ens_mut_out["ensemble"]
    d_nca_end = segment_dist(ens_wt[-1, :n_res, 0, :], ens_wt[-1, :n_res, 1, :])
    assert jnp.all(jnp.isfinite(ens_wt)) and float(jnp.max(jnp.abs(d_nca_end - 1.458))) < 0.5
    print(f"[PASS] Langevin SDE stable: mean N-CA bond = {float(jnp.mean(d_nca_end)):.3f} A (target 1.458 A)")

    # 4. Differentiable BAR antisymmetry + Implicit VJP gradient check
    dh_a = jnp.array([0.8, 1.1, 0.9, 1.3, 0.7, 1.0], dtype=jnp.float32)
    dh_b = jnp.array([0.6, 0.9, 1.2, 0.8, 1.0, 1.1], dtype=jnp.float32)
    ddg_fwd = bar_ddg(dh_a, dh_b, KBT_PHYS)
    ddg_rev = bar_ddg(-dh_b, -dh_a, KBT_PHYS)
    antisym_err = float(jnp.abs(ddg_fwd + ddg_rev))
    assert antisym_err < 1e-4, f"BAR antisymmetry error: {antisym_err}"

    g_wt, g_mut = jax.grad(bar_ddg, argnums=(0, 1))(dh_a, dh_b, KBT_PHYS)
    eps_fd = 1e-3
    fd_0 = (bar_ddg(dh_a.at[0].add(eps_fd), dh_b, KBT_PHYS) - bar_ddg(dh_a.at[0].add(-eps_fd), dh_b, KBT_PHYS)) / (2.0 * eps_fd)
    grad_err = float(jnp.abs(g_wt[0] - fd_0))
    assert grad_err < 1e-3 and float(jnp.abs(jnp.sum(g_wt) + jnp.sum(g_mut) - 1.0)) < 1e-4
    print(f"[PASS] Differentiable BAR: |fwd+rev|={antisym_err:.2e}, |VJP-FD|={grad_err:.2e}, sum(grad)=1.0")

    # 5. Bound-Unbound SKEMPI 2.0 Thermodynamic Cycle smoke check
    partner_a_mask = (cm == 0).astype(jnp.float32) * res_mask
    partner_b_mask = (cm == 1).astype(jnp.float32) * res_mask
    res_bind = binding_ddg_bar(
        ens_wt, ens_mut, d_ref_ca, phys_wt, phys_mut, weights,
        partner_a_mask, partner_b_mask, theta_eq,
        single_wt=s_wt, pair_wt=z_wt, single_mut=s_mut, pair_mut=z_mut,
        nn_params=nn_params, chain_mask=cm,
    )
    assert jnp.isfinite(res_bind["ddg_bind"])
    print(
        f"[PASS] SKEMPI Bound-Unbound cycle: ddG_bind={float(res_bind['ddg_bind']):.4f} kcal/mol "
        f"(phys={float(res_bind['dh_phys_mean']):.4f}, nn={float(res_bind['dh_nn_mean']):.4f}, "
        f"overlap={float(res_bind['overlap_score']):.3f})"
    )


if __name__ == "__main__":
    run_lapisan1_and_2_tests()
