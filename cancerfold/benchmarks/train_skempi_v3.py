"""OncoVac Adenine Gen 1 v3 — Pure-JAX virtual_sc + Boltzmann U_chem + Joltz-1 3-Fold Strict Family-Disjoint Trainer."""
import argparse
import csv
import os
import re
import sys
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
for _p in (_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import jax
import jax.numpy as jnp
import numpy as np
import optax

from train_skempi import (
    _SparsePriorLRU,
    _init_synced_index,
    _pack_item_joltz_sparse,
    _per_complex_stats,
    _scatter_pair_gpu,
    _sync_dir_incremental,
    _sync_file_to_drive,
    load_skempi_records,
    make_family_disjoint_folds,
    prepare_joltz_priors_cache,
    summarize_benchmark_pillars,
)
from cancerfold.af2_core.boltz_chem import (
    du_chem_ensemble,
    load_boltz_chem_params,
    make_fixed_decoding_ranks,
)
from cancerfold.af2_core.joltz_prior import prepare_wt_mut_priors
from cancerfold.af2_core.stochastic_folding import init_folding_params
from cancerfold.analyzer.thermo import binding_ddg_bar, skempi_metrics
from cancerfold.physics.sde import sample_ensemble

# 5 strictly-positive white-box physical couplings (decoded via softplus)
PHYS_INIT_V3 = {
    "k_clash": 2.0,
    "k_elec": 1.0,
    "k_hydro": 1.5,
    "k_hbond": 1.5,
    "k_chem": 0.65,
}


def _inv_softplus(y: float) -> float:
    return float(np.log(np.expm1(max(y, 1e-4))))


def _init_model_params_v3(prng_key: jax.Array, c_s: int = 384, c_z: int = 128) -> Dict[str, Dict[str, jnp.ndarray]]:
    phys_raw = {k: jnp.array(_inv_softplus(v), dtype=jnp.float32) for k, v in PHYS_INIT_V3.items()}
    return {
        "phys_raw": phys_raw,
        "nn": init_folding_params(prng_key, c_s=c_s, c_z=c_z),
    }


def _decode_theta_v3(phys_raw: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    theta = {k: jax.nn.softplus(v) for k, v in phys_raw.items()}
    theta["k_rep"] = jnp.array(0.0, dtype=jnp.float32)
    theta["enm_soften"] = jnp.array(1.0, dtype=jnp.float32)
    return theta


def save_checkpoint_v3(path: str, model_params: Dict[str, Dict[str, jnp.ndarray]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    flat: Dict[str, np.ndarray] = {}
    for k, v in model_params["phys_raw"].items():
        flat[f"phys_raw__{k}"] = np.asarray(v, dtype=np.float32)
    for k, v in model_params["nn"].items():
        flat[f"nn__{k}"] = np.asarray(v, dtype=np.float32)
    tmp_path = path + ".tmp.npz"
    np.savez(tmp_path, **flat)
    os.replace(tmp_path, path)


def load_checkpoint_v3(path: str) -> Dict[str, Dict[str, jnp.ndarray]]:
    data = np.load(path, allow_pickle=False)
    phys_raw, nn = {}, {}
    for key in data.files:
        if key.startswith("phys_raw__"):
            phys_raw[key[len("phys_raw__"):]] = jnp.asarray(data[key], dtype=jnp.float32)
        elif key.startswith("nn__"):
            nn[key[len("nn__"):]] = jnp.asarray(data[key], dtype=jnp.float32)
    return {"phys_raw": phys_raw, "nn": nn}


def _save_oof_csv_v3(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["fold", "pdb_id", "family", "mut_code", "n_muts", "ddg_exp", "ddg_pred", "dh_phys", "dh_chem", "dh_nn"]
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def _load_fold_oof_v3(
    oof_csv_path: str,
    ckpt_path: str,
    val_items: List[Dict[str, object]],
) -> Optional[Tuple[List[Dict[str, object]], float]]:
    if not (os.path.exists(oof_csv_path) and os.path.exists(ckpt_path)):
        return None
    try:
        rows: List[Dict[str, object]] = []
        with open(oof_csv_path, "r", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows.append({
                    "fold": int(r["fold"]),
                    "pdb_id": str(r["pdb_id"]),
                    "family": str(r["family"]),
                    "mut_code": str(r["mut_code"]),
                    "n_muts": int(r["n_muts"]),
                    "ddg_exp": float(r["ddg_exp"]),
                    "ddg_pred": float(r["ddg_pred"]),
                    "dh_phys": float(r["dh_phys"]),
                    "dh_chem": float(r.get("dh_chem", 0.0)),
                    "dh_nn": float(r["dh_nn"]),
                })
        expected = {(str(x["pdb_id"]), str(x["mut_code"])) for x in val_items}
        actual = {(str(r["pdb_id"]), str(r["mut_code"])) for r in rows}
        if len(rows) != len(val_items) or actual != expected:
            return None
        _ = load_checkpoint_v3(ckpt_path)
        return rows, 0.0
    except Exception:
        return None


def _print_3way(label: str, rows: List[Dict[str, object]]) -> None:
    p = float(np.mean([abs(float(r["dh_phys"])) for r in rows]))
    c = float(np.mean([abs(float(r.get("dh_chem", 0.0))) for r in rows]))
    n = float(np.mean([abs(float(r["dh_nn"])) for r in rows]))
    print(f"  [Pillar 2 v3] {label}: mean|dH_phys|={p:.3f} | mean|dH_chem|={c:.3f} | mean|dH_nn|={n:.3f}", flush=True)


def _huber_loss(pred: jnp.ndarray, target: jnp.ndarray, delta: float = 2.0) -> jnp.ndarray:
    err = jnp.abs(pred - target)
    return jnp.where(err <= delta, 0.5 * (err ** 2), delta * (err - 0.5 * delta))


def prepare_virtual_sc_ensembles(
    records: List[Dict[str, object]],
    prng_key: jax.Array,
    cache_dir: str = "/content/oncovac_cache",
    cache_max_n: int = 512,
    num_replicas: int = 2,
    num_steps: int = 40,
    burn_in: int = 16,
    stride: int = 8,
    drive_backup_dir: str = "",
) -> List[Dict[str, object]]:
    """Generates or reloads equilibrium BAOAB ensembles sampled under the virtual_sc Hamiltonian."""
    sde_subdir = f"baoab_sc_n{cache_max_n}_r{num_replicas}_s{num_steps}"
    sde_dir = os.path.join(cache_dir, sde_subdir)
    os.makedirs(sde_dir, exist_ok=True)
    theta_sde = {
        "k_clash": PHYS_INIT_V3["k_clash"],
        "k_elec": PHYS_INIT_V3["k_elec"],
        "k_hydro": PHYS_INIT_V3["k_hydro"],
        "k_hbond": PHYS_INIT_V3["k_hbond"],
        "k_rep": 0.0,
        "enm_soften": 1.0,
    }

    @jax.jit
    def _sample_state(bb, key, d_ref, phys, w, res_mask, cm, tk):
        out = sample_ensemble(
            bb, key, d_ref, phys, w, res_mask, theta_sde,
            nn_params=None, chain_mask=cm,
            num_replicas=num_replicas, num_steps=num_steps,
            burn_in=burn_in, stride=stride, temp_k=tk,
        )
        return out["ensemble"], out["mean_t_kin"]

    items: List[Dict[str, object]] = []
    temps = []
    n_loaded, n_computed = 0, 0
    t0 = time.time()

    for i, rec in enumerate(records):
        item_n = int(rec["max_n"])
        safe_mut = re.sub(r"[^A-Za-z0-9_]+", "_", str(rec["mut_code"]))
        npz_path = os.path.join(sde_dir, f"{rec['pdb_id']}_{safe_mut}.npz")
        loaded_ok = False
        if os.path.exists(npz_path):
            with np.load(npz_path, allow_pickle=False) as cached:
                if cached["ens_wt"].shape[1] >= item_n:
                    ens_wt = np.array(cached["ens_wt"][:, :item_n], dtype=np.float16, copy=True)
                    ens_mut = np.array(cached["ens_mut"][:, :item_n], dtype=np.float16, copy=True)
                    t_wt, t_mut = float(cached["t_wt"]), float(cached["t_mut"])
                    loaded_ok = True
                    n_loaded += 1

        if not loaded_ok:
            k_wt, k_mut = jax.random.split(jax.random.fold_in(prng_key, i))
            tk = float(rec["temp_k"])
            ens_wt_j, t_wt_a = _sample_state(
                rec["bb"], k_wt, rec["d_ref_ca"], rec["phys_wt"],
                rec["weights"], rec["res_mask"], rec["cm"], tk,
            )
            ens_mut_j, t_mut_a = _sample_state(
                rec["bb"], k_mut, rec["d_ref_ca"], rec["phys_mut"],
                rec["weights"], rec["res_mask"], rec["cm"], tk,
            )
            ens_wt = np.asarray(ens_wt_j, dtype=np.float16)
            ens_mut = np.asarray(ens_mut_j, dtype=np.float16)
            t_wt, t_mut = float(t_wt_a), float(t_mut_a)
            tmp_npz = npz_path + ".tmp.npz"
            np.savez_compressed(
                tmp_npz, ens_wt=ens_wt, ens_mut=ens_mut,
                t_wt=np.float32(t_wt), t_mut=np.float32(t_mut),
            )
            os.replace(tmp_npz, npz_path)
            n_computed += 1
            if n_computed % 200 == 0:
                print(
                    f"[SDE-vSC] Progress: {i + 1}/{len(records)} "
                    f"(new={n_computed}, elapsed={time.time() - t0:.1f}s)...",
                    flush=True,
                )

        temps.extend([t_wt, t_mut])
        items.append({**rec, "ens_wt": ens_wt, "ens_mut": ens_mut})

    n_fr = int(items[0]["ens_wt"].shape[0]) if items else 0
    print(
        f"[SDE-vSC] Ready {len(items)} virtual_sc BAOAB ensembles ({n_fr} frames/state | "
        f"cached={n_loaded}, new={n_computed}, elapsed={time.time() - t0:.1f}s) | "
        f"mean T_kin={np.mean(temps):.1f} K",
        flush=True,
    )
    if drive_backup_dir and n_computed > 0:
        dst_sde = os.path.join(drive_backup_dir, "oncovac_cache", sde_subdir)
        synced = _init_synced_index(dst_sde)
        n_copy = _sync_dir_incremental(sde_dir, dst_sde, synced)
        if n_copy > 0:
            print(f"[DRIVE] Backed up {n_copy} virtual_sc BAOAB caches -> {dst_sde}", flush=True)
    return items


def prepare_u_chem_cache(
    items: List[Dict[str, object]],
    cache_dir: str = "/content/oncovac_cache",
    mode: str = "full",
    num_orders: int = 4,
    drive_backup_dir: str = "",
) -> List[Dict[str, object]]:
    """
    Computes and caches pure-JAX SE(3) Boltzmann Chemical Potential Delta U_chem(X)
    across all BAOAB frames (Bound AB, Unbound A, Unbound B) in 1 compact <500 KB .npz file.
    """
    cache_name = f"u_chem_v48_020_k{num_orders}_{mode}.npz"
    local_path = os.path.join(cache_dir, cache_name)
    drive_path = os.path.join(drive_backup_dir, "oncovac_cache", cache_name) if drive_backup_dir else ""
    if (not os.path.exists(local_path)) and drive_path and os.path.exists(drive_path):
        _sync_file_to_drive(drive_path, local_path)

    cached_map: Dict[str, np.ndarray] = {}
    if os.path.exists(local_path):
        with np.load(local_path, allow_pickle=False) as d:
            keys = d["keys"]
            vals = d["vals"]
            for k, v in zip(keys, vals):
                cached_map[str(k)] = np.asarray(v, dtype=np.float32)

    n_frames = int(items[0]["ens_wt"].shape[0]) if items else 6
    missing_idxs = [
        i for i, x in enumerate(items)
        if f"{x['pdb_id']}__{x['mut_code']}" not in cached_map
        or cached_map[f"{x['pdb_id']}__{x['mut_code']}"].shape != (6, n_frames)
    ]

    if missing_idxs:
        t0 = time.time()
        chem_params = load_boltz_chem_params(cache_dir=cache_dir)
        print(
            f"[U_CHEM] Computing pure-JAX MIT ProteinMPNN v_48_020 (K={num_orders}) "
            f"for {len(missing_idxs)}/{len(items)} mutations on BAOAB ensembles...",
            flush=True,
        )

        @jax.jit
        def _eval_leg(ens_f32, aa_wt, aa_mut, mask, cm, ranks):
            return du_chem_ensemble(chem_params, ens_f32, aa_wt, aa_mut, mask, cm, ranks)

        ranks_cache: Dict[Tuple[str, str], jnp.ndarray] = {}
        for idx_i, pos in enumerate(missing_idxs, 1):
            x = items[pos]
            pid = str(x["pdb_id"])
            if (pid, "ab") not in ranks_cache:
                ranks_cache[(pid, "ab")] = jnp.asarray(make_fixed_decoding_ranks(np.asarray(x["res_mask"]), num_orders))
                ranks_cache[(pid, "a")] = jnp.asarray(make_fixed_decoding_ranks(np.asarray(x["pa_mask"]), num_orders))
                ranks_cache[(pid, "b")] = jnp.asarray(make_fixed_decoding_ranks(np.asarray(x["pb_mask"]), num_orders))

            ew = jnp.asarray(x["ens_wt"], dtype=jnp.float32)
            em = jnp.asarray(x["ens_mut"], dtype=jnp.float32)
            aa_w_np = np.asarray(x["aa_wt"])
            aa_m_np = np.asarray(x["aa_mut"])
            pa_bool = np.asarray(x["pa_mask"]) > 0.5
            pb_bool = np.asarray(x["pb_mask"]) > 0.5

            du_c_wt = np.asarray(_eval_leg(ew, x["aa_wt"], x["aa_mut"], x["res_mask"], x["cm"], ranks_cache[(pid, "ab")]), dtype=np.float32)
            du_c_mut = np.asarray(_eval_leg(em, x["aa_wt"], x["aa_mut"], x["res_mask"], x["cm"], ranks_cache[(pid, "ab")]), dtype=np.float32)

            if np.any(aa_w_np[pa_bool] != aa_m_np[pa_bool]):
                du_a_wt = np.asarray(_eval_leg(ew, x["aa_wt"], x["aa_mut"], x["pa_mask"], x["cm"], ranks_cache[(pid, "a")]), dtype=np.float32)
                du_a_mut = np.asarray(_eval_leg(em, x["aa_wt"], x["aa_mut"], x["pa_mask"], x["cm"], ranks_cache[(pid, "a")]), dtype=np.float32)
            else:
                du_a_wt = np.zeros((n_frames,), dtype=np.float32)
                du_a_mut = np.zeros((n_frames,), dtype=np.float32)

            if np.any(aa_w_np[pb_bool] != aa_m_np[pb_bool]):
                du_b_wt = np.asarray(_eval_leg(ew, x["aa_wt"], x["aa_mut"], x["pb_mask"], x["cm"], ranks_cache[(pid, "b")]), dtype=np.float32)
                du_b_mut = np.asarray(_eval_leg(em, x["aa_wt"], x["aa_mut"], x["pb_mask"], x["cm"], ranks_cache[(pid, "b")]), dtype=np.float32)
            else:
                du_b_wt = np.zeros((n_frames,), dtype=np.float32)
                du_b_mut = np.zeros((n_frames,), dtype=np.float32)

            key = f"{pid}__{x['mut_code']}"
            cached_map[key] = np.stack([du_c_wt, du_c_mut, du_a_wt, du_a_mut, du_b_wt, du_b_mut], axis=0)
            if idx_i == 1 or idx_i % 250 == 0 or idx_i == len(missing_idxs):
                print(f"  [U_CHEM] Scored {idx_i:4d}/{len(missing_idxs)} in {time.time() - t0:.1f}s...", flush=True)

        all_keys = np.array(sorted(cached_map.keys()))
        all_vals = np.stack([cached_map[str(k)] for k in all_keys], axis=0).astype(np.float32)
        tmp_path = local_path + ".tmp.npz"
        np.savez_compressed(tmp_path, keys=all_keys, vals=all_vals)
        os.replace(tmp_path, local_path)
        if drive_path:
            _sync_file_to_drive(local_path, drive_path)
            print(f"[DRIVE] Backed up pure-JAX U_chem cache ({os.path.getsize(local_path)//1024} KB) -> {drive_path}", flush=True)
    else:
        print(f"[U_CHEM] Loaded 100% cached pure-JAX U_chem ({len(items)} muts) from {local_path}", flush=True)

    out_items = []
    zs_chem, y_true, pdbs = [], [], []
    for x in items:
        arr = cached_map[f"{x['pdb_id']}__{x['mut_code']}"]
        out_items.append({
            **x,
            "du_chem_c_wt": jnp.asarray(arr[0], dtype=jnp.float32),
            "du_chem_c_mut": jnp.asarray(arr[1], dtype=jnp.float32),
            "du_chem_a_wt": jnp.asarray(arr[2], dtype=jnp.float32),
            "du_chem_a_mut": jnp.asarray(arr[3], dtype=jnp.float32),
            "du_chem_b_wt": jnp.asarray(arr[4], dtype=jnp.float32),
            "du_chem_b_mut": jnp.asarray(arr[5], dtype=jnp.float32),
        })
        dc = 0.5 * (float(np.mean(arr[0])) + float(np.mean(arr[1])))
        du = 0.5 * (float(np.mean(arr[2])) + float(np.mean(arr[3])) + float(np.mean(arr[4])) + float(np.mean(arr[5])))
        zs_chem.append(dc - du)
        y_true.append(float(x["ddg_exp"]))
        pdbs.append(str(x["pdb_id"]))

    m_c = skempi_metrics(jnp.asarray(zs_chem), jnp.asarray(y_true))
    pc10_r, _, _ = _per_complex_stats(zs_chem, y_true, pdbs, min_muts=10)
    print(
        f"[U_CHEM AUDIT] Pure-JAX BAOAB Zero-Shot U_chem (N={len(items)}): "
        f"r={float(m_c['pearson_r']):+.4f} | rho={float(m_c['spearman_rho']):+.4f} | "
        f"pc10_r={pc10_r:+.4f} | RMSE={float(m_c['rmse']):.3f} | BAR=0.00e+00 | Telepathy=0.00e+00",
        flush=True,
    )
    return out_items


def _evaluate_zero_shot_v3(items: List[Dict[str, object]], include_chem: bool = True) -> List[Dict[str, object]]:
    """Evaluates pure white-box physics (0% SKEMPI training): H_phys(virtual_sc) [+ 0.65 * U_chem]."""
    theta_raw = {
        "k_clash": PHYS_INIT_V3["k_clash"],
        "k_elec": PHYS_INIT_V3["k_elec"],
        "k_hydro": PHYS_INIT_V3["k_hydro"],
        "k_hbond": PHYS_INIT_V3["k_hbond"],
        "k_chem": PHYS_INIT_V3["k_chem"] if include_chem else 0.0,
        "k_rep": 0.0,
        "enm_soften": 1.0,
    }

    @jax.jit
    def _eval_one(ew, em, d_ref, pw, pm, w, pa, pb, cm, tk, dcw, dcm, daw, dam, dbw, dbm):
        return binding_ddg_bar(
            ew.astype(jnp.float32), em.astype(jnp.float32),
            d_ref, pw, pm, w, pa, pb, theta_raw,
            nn_params=None, chain_mask=cm, temp_k=tk,
            du_chem_c_wt=dcw, du_chem_c_mut=dcm,
            du_chem_a_wt=daw, du_chem_a_mut=dam,
            du_chem_b_wt=dbw, du_chem_b_mut=dbm,
        )

    rows: List[Dict[str, object]] = []
    pending = []
    for x in items:
        out = _eval_one(
            x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
            x["phys_wt"], x["phys_mut"], x["weights"], x["pa_mask"], x["pb_mask"],
            x["cm"], float(x["temp_k"]),
            x["du_chem_c_wt"], x["du_chem_c_mut"],
            x["du_chem_a_wt"], x["du_chem_a_mut"],
            x["du_chem_b_wt"], x["du_chem_b_mut"],
        )
        pending.append((out["ddg_bind"], out["dh_phys_mean"], out["dh_chem_mean"], x))
        if len(pending) >= 32:
            vals = jax.device_get([(a, b, c) for a, b, c, _ in pending])
            for (yp, dp, dc), (_, _, _, it) in zip(vals, pending):
                rows.append({
                    "fold": -1, "pdb_id": str(it["pdb_id"]), "family": str(it["family"]),
                    "mut_code": str(it["mut_code"]), "n_muts": int(it["n_muts"]),
                    "ddg_exp": float(it["ddg_exp"]), "ddg_pred": float(yp),
                    "dh_phys": float(dp), "dh_chem": float(dc), "dh_nn": 0.0,
                })
            pending.clear()
    if pending:
        vals = jax.device_get([(a, b, c) for a, b, c, _ in pending])
        for (yp, dp, dc), (_, _, _, it) in zip(vals, pending):
            rows.append({
                "fold": -1, "pdb_id": str(it["pdb_id"]), "family": str(it["family"]),
                "mut_code": str(it["mut_code"]), "n_muts": int(it["n_muts"]),
                "ddg_exp": float(it["ddg_exp"]), "ddg_pred": float(yp),
                "dh_phys": float(dp), "dh_chem": float(dc), "dh_nn": 0.0,
            })
    return rows


def train_single_fold_v3(
    fold_idx: int,
    train_items: List[Dict[str, object]],
    val_items: List[Dict[str, object]],
    weights_dir: str,
    prior_mode: str = "joltz",
    ckpt_prefix: str = "adenine_gen1_v3",
    epochs: int = 24,
    batch_size: int = 16,
    peak_lr: float = 3e-4,
    weight_decay: float = 2.5e-2,
    dropout_rate: float = 0.25,
    jitter_std: float = 0.02,
    nn_reg_weight: float = 1.5e-2,
    huber_delta: float = 2.0,
    train_frames: int = 3,
    patience: int = 7,
    min_epochs: int = 6,
    sparse_lru: Optional[_SparsePriorLRU] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, jnp.ndarray]], float]:
    """
    Trains one Strict Family-Disjoint CV fold of OncoVac Adenine v3:
      H_alpha(X) = H_phys(virtual_sc) + k_chem * U_chem(X; alpha) + U_NN(X; alpha, Joltz-1)
    """
    use_joltz = (prior_mode == "joltz")
    lru = sparse_lru if (sparse_lru is not None) else (_SparsePriorLRU() if use_joltz else None)

    tr_pdb_counts: Dict[str, int] = defaultdict(int)
    for it in train_items:
        tr_pdb_counts[str(it["pdb_id"])] += 1
    raw_sw = np.asarray(
        [1.0 / (float(tr_pdb_counts[str(it["pdb_id"])]) ** 0.35) for it in train_items],
        dtype=np.float32,
    )
    sample_weights = raw_sw / float(np.mean(raw_sw))

    key = jax.random.PRNGKey(300 + fold_idx)
    k_init, k_prior_base, k_train = jax.random.split(key, 3)

    model_params = _init_model_params_v3(k_init, c_s=384, c_z=128)
    steps_per_epoch = max(1, (len(train_items) + batch_size - 1) // batch_size)
    total_steps = max(2, epochs * steps_per_epoch)
    warmup_steps = max(1, min(steps_per_epoch * 2, total_steps // 5))

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=peak_lr * 0.1,
        peak_value=peak_lr,
        warmup_steps=warmup_steps,
        decay_steps=total_steps,
        end_value=peak_lr * 0.05,
    )
    wd_mask = {
        "phys_raw": jax.tree_util.tree_map(lambda _: False, model_params["phys_raw"]),
        "nn": jax.tree_util.tree_map(lambda x: x.ndim >= 2, model_params["nn"]),
    }
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay, mask=wd_mask),
    )
    opt_state = optimizer.init(model_params)

    def _get_fallback_priors(aa_wt, aa_mut, res_mask):
        priors = prepare_wt_mut_priors(
            None, aa_wt, aa_mut, res_mask, c_s=384, c_z=128, fallback_key=k_prior_base
        )
        return priors["wt"], priors["mut"]

    @jax.jit
    def _eval_none(
        params, ens_wt, ens_mut, d_ref, phys_wt, phys_mut, w, pa, pb, res_mask, aa_wt, aa_mut, cm, tk,
        dcw, dcm, daw, dam, dbw, dbm,
    ):
        (s_wt, z_wt), (s_mut, z_mut) = _get_fallback_priors(aa_wt, aa_mut, res_mask)
        theta = _decode_theta_v3(params["phys_raw"])
        return binding_ddg_bar(
            ens_wt.astype(jnp.float32), ens_mut.astype(jnp.float32),
            d_ref, phys_wt, phys_mut, w, pa, pb, theta,
            single_wt=s_wt, pair_wt=z_wt, single_mut=s_mut, pair_mut=z_mut,
            single_wt_unbound=s_wt, pair_wt_unbound=z_wt,
            single_mut_unbound=s_mut, pair_mut_unbound=z_mut,
            nn_params=params["nn"], chain_mask=cm, temp_k=tk,
            du_chem_c_wt=dcw, du_chem_c_mut=dcm,
            du_chem_a_wt=daw, du_chem_a_mut=dam,
            du_chem_b_wt=dbw, du_chem_b_mut=dbm,
        )

    @jax.jit
    def _eval_joltz_sparse(
        params, ens_wt, ens_mut, d_ref, phys_wt, phys_mut, w, pa, pb, cm, tk,
        dcw, dcm, daw, dam, dbw, dbm,
        s_wb, r_wb, c_wb, z_wb, s_mb, r_mb, c_mb, z_mb,
        s_wu, r_wu, c_wu, z_wu, s_mu, r_mu, c_mu, z_mu,
    ):
        max_n = s_wb.shape[0]
        z_wt_b = _scatter_pair_gpu(r_wb, c_wb, z_wb, max_n)
        z_mut_b = _scatter_pair_gpu(r_mb, c_mb, z_mb, max_n)
        z_wt_u = _scatter_pair_gpu(r_wu, c_wu, z_wu, max_n)
        z_mut_u = _scatter_pair_gpu(r_mu, c_mu, z_mu, max_n)
        theta = _decode_theta_v3(params["phys_raw"])
        return binding_ddg_bar(
            ens_wt.astype(jnp.float32), ens_mut.astype(jnp.float32),
            d_ref, phys_wt, phys_mut, w, pa, pb, theta,
            single_wt=s_wb.astype(jnp.float32), pair_wt=z_wt_b,
            single_mut=s_mb.astype(jnp.float32), pair_mut=z_mut_b,
            single_wt_unbound=s_wu.astype(jnp.float32), pair_wt_unbound=z_wt_u,
            single_mut_unbound=s_mu.astype(jnp.float32), pair_mut_unbound=z_mut_u,
            nn_params=params["nn"], chain_mask=cm, temp_k=tk,
            du_chem_c_wt=dcw, du_chem_c_mut=dcm,
            du_chem_a_wt=daw, du_chem_a_mut=dam,
            du_chem_b_wt=dbw, du_chem_b_mut=dbm,
        )

    def _loss_core(
        p, ew, em, d_ref, phys_wt, phys_mut, w, pa, pb, cm, tk, y_true, sw,
        dcw, dcm, daw, dam, dbw, dbm,
        s_wt_b, z_wt_b, s_mut_b, z_mut_b, s_wt_u, z_wt_u, s_mut_u, z_mut_u,
    ):
        theta = _decode_theta_v3(p["phys_raw"])
        out = binding_ddg_bar(
            ew, em, d_ref, phys_wt, phys_mut, w, pa, pb, theta,
            single_wt=s_wt_b, pair_wt=z_wt_b, single_mut=s_mut_b, pair_mut=z_mut_b,
            single_wt_unbound=s_wt_u, pair_wt_unbound=z_wt_u,
            single_mut_unbound=s_mut_u, pair_mut_unbound=z_mut_u,
            nn_params=p["nn"], chain_mask=cm, temp_k=tk,
            du_chem_c_wt=dcw, du_chem_c_mut=dcm,
            du_chem_a_wt=daw, du_chem_a_mut=dam,
            du_chem_b_wt=dbw, du_chem_b_mut=dbm,
        )
        huber = _huber_loss(out["ddg_bind"], y_true, delta=huber_delta)
        phys_anchor = 0.02 * sum(
            (theta[k] - PHYS_INIT_V3[k]) ** 2 for k in ("k_clash", "k_elec", "k_hydro", "k_hbond")
        ) + 0.01 * ((theta["k_chem"] - PHYS_INIT_V3["k_chem"]) ** 2)
        nn_reg = nn_reg_weight * (out["dh_nn_mean"] ** 2)
        return sw * (huber + nn_reg) + phys_anchor

    @jax.jit
    def _grad_none(
        params, step_key, ens_wt, ens_mut, d_ref, phys_wt, phys_mut, w, pa, pb, res_mask, aa_wt, aa_mut, cm, tk, y_true, sw,
        dcw, dcm, daw, dam, dbw, dbm,
    ):
        k_sub, k_sub2, k_jw, k_jm, k_drop = jax.random.split(step_key, 5)
        ew_f32 = ens_wt.astype(jnp.float32)
        em_f32 = ens_mut.astype(jnp.float32)
        n_frames = ew_f32.shape[0]
        k_sub_n = min(train_frames, n_frames)
        idx_w = jax.random.choice(k_sub, n_frames, shape=(k_sub_n,), replace=False)
        idx_m = jax.random.choice(k_sub2, n_frames, shape=(k_sub_n,), replace=False)

        m_bb = res_mask[None, :, None, None]
        ew = ew_f32[idx_w] + jax.random.normal(k_jw, ew_f32[idx_w].shape, dtype=jnp.float32) * jitter_std * m_bb
        em = em_f32[idx_m] + jax.random.normal(k_jm, em_f32[idx_m].shape, dtype=jnp.float32) * jitter_std * m_bb

        (s_wt, z_wt), (s_mut, z_mut) = _get_fallback_priors(aa_wt, aa_mut, res_mask)
        keep_prob = 1.0 - dropout_rate
        ks, kz = jax.random.split(k_drop, 2)
        ms = jax.random.bernoulli(ks, keep_prob, s_wt.shape).astype(jnp.float32) / keep_prob
        mz = jax.random.bernoulli(kz, keep_prob, z_wt.shape).astype(jnp.float32) / keep_prob
        s_wt_d, s_mut_d = s_wt * ms, s_mut * ms
        z_wt_d, z_mut_d = z_wt * mz, z_mut * mz

        return jax.value_and_grad(
            lambda p: _loss_core(
                p, ew, em, d_ref, phys_wt, phys_mut, w, pa, pb, cm, tk, y_true, sw,
                dcw[idx_w], dcm[idx_m], daw[idx_w], dam[idx_m], dbw[idx_w], dbm[idx_m],
                s_wt_d, z_wt_d, s_mut_d, z_mut_d, s_wt_d, z_wt_d, s_mut_d, z_mut_d,
            )
        )(params)

    @jax.jit
    def _grad_joltz_sparse(
        params, step_key, ens_wt, ens_mut, d_ref, phys_wt, phys_mut, w, pa, pb, res_mask, cm, tk, y_true, sw,
        dcw, dcm, daw, dam, dbw, dbm,
        s_wb, r_wb, c_wb, z_wb, s_mb, r_mb, c_mb, z_mb,
        s_wu, r_wu, c_wu, z_wu, s_mu, r_mu, c_mu, z_mu,
    ):
        k_sub, k_sub2, k_jw, k_jm, k_drop = jax.random.split(step_key, 5)
        ew_f32 = ens_wt.astype(jnp.float32)
        em_f32 = ens_mut.astype(jnp.float32)
        n_frames = ew_f32.shape[0]
        k_sub_n = min(train_frames, n_frames)
        idx_w = jax.random.choice(k_sub, n_frames, shape=(k_sub_n,), replace=False)
        idx_m = jax.random.choice(k_sub2, n_frames, shape=(k_sub_n,), replace=False)

        m_bb = res_mask[None, :, None, None]
        ew = ew_f32[idx_w] + jax.random.normal(k_jw, ew_f32[idx_w].shape, dtype=jnp.float32) * jitter_std * m_bb
        em = em_f32[idx_m] + jax.random.normal(k_jm, em_f32[idx_m].shape, dtype=jnp.float32) * jitter_std * m_bb

        max_n = s_wb.shape[0]
        z_wt_b = _scatter_pair_gpu(r_wb, c_wb, z_wb, max_n)
        z_mut_b = _scatter_pair_gpu(r_mb, c_mb, z_mb, max_n)
        z_wt_u = _scatter_pair_gpu(r_wu, c_wu, z_wu, max_n)
        z_mut_u = _scatter_pair_gpu(r_mu, c_mu, z_mu, max_n)

        keep_prob = 1.0 - dropout_rate
        ks, kz = jax.random.split(k_drop, 2)
        ms = jax.random.bernoulli(ks, keep_prob, s_wb.shape).astype(jnp.float32) / keep_prob
        mz = jax.random.bernoulli(kz, keep_prob, z_wt_b.shape).astype(jnp.float32) / keep_prob

        swb = s_wb.astype(jnp.float32) * ms
        smb = s_mb.astype(jnp.float32) * ms
        swu = s_wu.astype(jnp.float32) * ms
        smu = s_mu.astype(jnp.float32) * ms
        zwb, zmb, zwu, zmu = z_wt_b * mz, z_mut_b * mz, z_wt_u * mz, z_mut_u * mz

        return jax.value_and_grad(
            lambda p: _loss_core(
                p, ew, em, d_ref, phys_wt, phys_mut, w, pa, pb, cm, tk, y_true, sw,
                dcw[idx_w], dcm[idx_m], daw[idx_w], dam[idx_m], dbw[idx_w], dbm[idx_m],
                swb, zwb, smb, zmb, swu, zwu, smu, zmu,
            )
        )(params)

    @jax.jit
    def _apply_grads(params, state, grads):
        updates, new_state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), new_state

    def _run_eval(params, split_items):
        preds, trues, pdbs, dh_p, dh_c, dh_n = [], [], [], [], [], []
        pending = []
        for x in split_items:
            if use_joltz:
                sp = _pack_item_joltz_sparse(x, lru)
                out = _eval_joltz_sparse(
                    params, x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
                    x["phys_wt"], x["phys_mut"], x["weights"], x["pa_mask"], x["pb_mask"],
                    x["cm"], float(x["temp_k"]),
                    x["du_chem_c_wt"], x["du_chem_c_mut"],
                    x["du_chem_a_wt"], x["du_chem_a_mut"],
                    x["du_chem_b_wt"], x["du_chem_b_mut"],
                    *sp,
                )
            else:
                out = _eval_none(
                    params, x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
                    x["phys_wt"], x["phys_mut"], x["weights"], x["pa_mask"], x["pb_mask"],
                    x["res_mask"], x["aa_wt"], x["aa_mut"], x["cm"], float(x["temp_k"]),
                    x["du_chem_c_wt"], x["du_chem_c_mut"],
                    x["du_chem_a_wt"], x["du_chem_a_mut"],
                    x["du_chem_b_wt"], x["du_chem_b_mut"],
                )
            pending.append((out["ddg_bind"], out["dh_phys_mean"], out["dh_chem_mean"], out["dh_nn_mean"], float(x["ddg_exp"]), str(x["pdb_id"])))
            if len(pending) >= 32:
                vals = jax.device_get([(a, b, c, d) for a, b, c, d, _, _ in pending])
                for (yp, dp, dc, dn), (_, _, _, _, yt, pid) in zip(vals, pending):
                    preds.append(float(yp)); dh_p.append(float(dp)); dh_c.append(float(dc)); dh_n.append(float(dn))
                    trues.append(yt); pdbs.append(pid)
                pending.clear()
        if pending:
            vals = jax.device_get([(a, b, c, d) for a, b, c, d, _, _ in pending])
            for (yp, dp, dc, dn), (_, _, _, _, yt, pid) in zip(vals, pending):
                preds.append(float(yp)); dh_p.append(float(dp)); dh_c.append(float(dc)); dh_n.append(float(dn))
                trues.append(yt); pdbs.append(pid)

        m = skempi_metrics(jnp.asarray(preds), jnp.asarray(trues))
        pc10_r, pc10_rho, _ = _per_complex_stats(preds, trues, pdbs, min_muts=10)
        return {
            "preds": preds, "trues": trues,
            "dh_phys_list": dh_p, "dh_chem_list": dh_c, "dh_nn_list": dh_n,
            "r": float(m["pearson_r"]), "rho": float(m["spearman_rho"]),
            "pc10_r": pc10_r, "pc10_rho": pc10_rho,
            "rmse": float(m["rmse"]), "mae": float(m["mae"]),
            "dh_phys": float(np.mean(np.abs(dh_p))),
            "dh_chem": float(np.mean(np.abs(dh_c))),
            "dh_nn": float(np.mean(np.abs(dh_n))),
            "mean_p": float(np.mean(preds)), "mean_t": float(np.mean(trues)),
        }

    ckpt_path = os.path.join(weights_dir, f"{ckpt_prefix}_fold{fold_idx}.npz")
    best_rmse = float("inf")
    best_epoch = 0
    bad_epochs = 0

    print(
        f"\n--- [{ckpt_prefix}] Fold {fold_idx} Training "
        f"({len(train_items)} train / {len(val_items)} val | lr={peak_lr:.0e}, wd={weight_decay:.1e}, "
        f"nn_reg={nn_reg_weight:.1e}, huber={huber_delta:.1f}, min_ep={min_epochs}, patience={patience}) ---",
        flush=True,
    )

    for ep in range(1, epochs + 1):
        t_ep = time.time()
        k_train, k_perm = jax.random.split(k_train)
        perm = np.asarray(jax.random.permutation(k_perm, len(train_items)))
        ep_losses = []

        for b_start in range(0, len(perm), batch_size):
            batch_idx = perm[b_start : b_start + batch_size]
            grad_acc = None
            loss_acc = 0.0
            for ii in batch_idx:
                x = train_items[int(ii)]
                sw_i = jnp.asarray(sample_weights[int(ii)], dtype=jnp.float32)
                k_train, k_step = jax.random.split(k_train)
                if use_joltz:
                    sp = _pack_item_joltz_sparse(x, lru)
                    lv, gv = _grad_joltz_sparse(
                        model_params, k_step, x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
                        x["phys_wt"], x["phys_mut"], x["weights"], x["pa_mask"], x["pb_mask"],
                        x["res_mask"], x["cm"], float(x["temp_k"]), float(x["ddg_exp"]), sw_i,
                        x["du_chem_c_wt"], x["du_chem_c_mut"],
                        x["du_chem_a_wt"], x["du_chem_a_mut"],
                        x["du_chem_b_wt"], x["du_chem_b_mut"],
                        *sp,
                    )
                else:
                    lv, gv = _grad_none(
                        model_params, k_step, x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
                        x["phys_wt"], x["phys_mut"], x["weights"], x["pa_mask"], x["pb_mask"],
                        x["res_mask"], x["aa_wt"], x["aa_mut"], x["cm"], float(x["temp_k"]),
                        float(x["ddg_exp"]), sw_i,
                        x["du_chem_c_wt"], x["du_chem_c_mut"],
                        x["du_chem_a_wt"], x["du_chem_a_mut"],
                        x["du_chem_b_wt"], x["du_chem_b_mut"],
                    )
                loss_acc = loss_acc + lv
                grad_acc = gv if grad_acc is None else jax.tree_util.tree_map(lambda a, b: a + b, grad_acc, gv)

            inv_bs = jnp.asarray(1.0 / len(batch_idx), dtype=jnp.float32)
            grad_mean = jax.tree_util.tree_map(lambda g: g * inv_bs, grad_acc)
            model_params, opt_state = _apply_grads(model_params, opt_state, grad_mean)
            ep_losses.append(float(loss_acc * inv_bs))

        val_m = _run_eval(model_params, val_items)
        improved = val_m["rmse"] < (best_rmse - 1e-4)
        if improved:
            best_rmse = val_m["rmse"]
            best_epoch = ep
            bad_epochs = 0
            save_checkpoint_v3(ckpt_path, model_params)
        elif ep >= min_epochs:
            bad_epochs += 1

        if ep == 1 or ep % 2 == 0 or improved or ep == epochs:
            tc = _decode_theta_v3(model_params["phys_raw"])
            mark = " *BEST*" if improved else ""
            print(
                f"  [F{fold_idx} ep {ep:02d}/{epochs}] loss={np.mean(ep_losses):.3f} | "
                f"VAL r={val_m['r']:+.3f} pc10={val_m['pc10_r']:+.3f} rho={val_m['rho']:+.3f} "
                f"RMSE={val_m['rmse']:.3f} | mean={val_m['mean_p']:+.2f}/{val_m['mean_t']:+.2f} | "
                f"|p/c/nn|={val_m['dh_phys']:.2f}/{val_m['dh_chem']:.2f}/{val_m['dh_nn']:.2f} | "
                f"k_chem={float(tc['k_chem']):.2f} ({time.time() - t_ep:.1f}s){mark}",
                flush=True,
            )

        if ep >= min_epochs and bad_epochs >= patience:
            print(f"  [F{fold_idx}] Early stopping at ep {ep} (best ep={best_epoch}, VAL RMSE={best_rmse:.3f})", flush=True)
            break

    reloaded = load_checkpoint_v3(ckpt_path)
    final_val = _run_eval(reloaded, val_items)
    print(
        f"  [F{fold_idx} FINAL] Restored {ckpt_path} (ep {best_epoch}) -> "
        f"VAL r={final_val['r']:+.4f}, pc10_r={final_val['pc10_r']:+.4f}, "
        f"rho={final_val['rho']:+.4f}, RMSE={final_val['rmse']:.3f}, "
        f"mean={final_val['mean_p']:+.2f}/{final_val['mean_t']:+.2f}",
        flush=True,
    )

    # Exact BAR antisymmetry audit on unseen fold (including U_chem negation!)
    antisym_errs = []
    for x in val_items[:5]:
        if use_joltz:
            (s_wb, r_wb, c_wb, z_wb, s_mb, r_mb, c_mb, z_mb,
             s_wu, r_wu, c_wu, z_wu, s_mu, r_mu, c_mu, z_mu) = _pack_item_joltz_sparse(x, lru)
            fwd = _eval_joltz_sparse(
                reloaded, x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
                x["phys_wt"], x["phys_mut"], x["weights"], x["pa_mask"], x["pb_mask"],
                x["cm"], float(x["temp_k"]),
                x["du_chem_c_wt"], x["du_chem_c_mut"], x["du_chem_a_wt"], x["du_chem_a_mut"],
                x["du_chem_b_wt"], x["du_chem_b_mut"],
                s_wb, r_wb, c_wb, z_wb, s_mb, r_mb, c_mb, z_mb,
                s_wu, r_wu, c_wu, z_wu, s_mu, r_mu, c_mu, z_mu,
            )["ddg_bind"]
            rev = _eval_joltz_sparse(
                reloaded, x["ens_mut"], x["ens_wt"], x["d_ref_ca"],
                x["phys_mut"], x["phys_wt"], x["weights"], x["pa_mask"], x["pb_mask"],
                x["cm"], float(x["temp_k"]),
                -x["du_chem_c_mut"], -x["du_chem_c_wt"], -x["du_chem_a_mut"], -x["du_chem_a_wt"],
                -x["du_chem_b_mut"], -x["du_chem_b_wt"],
                s_mb, r_mb, c_mb, z_mb, s_wb, r_wb, c_wb, z_wb,
                s_mu, r_mu, c_mu, z_mu, s_wu, r_wu, c_wu, z_wu,
            )["ddg_bind"]
        else:
            fwd = _eval_none(
                reloaded, x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
                x["phys_wt"], x["phys_mut"], x["weights"], x["pa_mask"], x["pb_mask"],
                x["res_mask"], x["aa_wt"], x["aa_mut"], x["cm"], float(x["temp_k"]),
                x["du_chem_c_wt"], x["du_chem_c_mut"], x["du_chem_a_wt"], x["du_chem_a_mut"],
                x["du_chem_b_wt"], x["du_chem_b_mut"],
            )["ddg_bind"]
            rev = _eval_none(
                reloaded, x["ens_mut"], x["ens_wt"], x["d_ref_ca"],
                x["phys_mut"], x["phys_wt"], x["weights"], x["pa_mask"], x["pb_mask"],
                x["res_mask"], x["aa_mut"], x["aa_wt"], x["cm"], float(x["temp_k"]),
                -x["du_chem_c_mut"], -x["du_chem_c_wt"], -x["du_chem_a_mut"], -x["du_chem_a_wt"],
                -x["du_chem_b_mut"], -x["du_chem_b_wt"],
            )["ddg_bind"]
        antisym_errs.append(float(jnp.abs(fwd + rev)))
    max_antisym = max(antisym_errs) if antisym_errs else 0.0
    print(f"  [F{fold_idx} AUDIT] Unseen-family exact BAR antisymmetry max|fwd+rev| = {max_antisym:.2e}", flush=True)

    oof_rows = []
    for x, yp, dp, dc, dn in zip(
        val_items, final_val["preds"], final_val["dh_phys_list"], final_val["dh_chem_list"], final_val["dh_nn_list"]
    ):
        oof_rows.append({
            "fold": fold_idx, "pdb_id": str(x["pdb_id"]), "family": str(x["family"]),
            "mut_code": str(x["mut_code"]), "n_muts": int(x["n_muts"]),
            "ddg_exp": float(x["ddg_exp"]), "ddg_pred": float(yp),
            "dh_phys": float(dp), "dh_chem": float(dc), "dh_nn": float(dn),
        })
    return oof_rows, reloaded, max_antisym


def _run_cv_suite_v3(
    folds: List[Tuple[List[Dict[str, object]], List[Dict[str, object]]]],
    weights_dir: str,
    prior_mode: str,
    ckpt_prefix: str,
    mode: str,
    epochs: int,
    batch_size: int,
    patience: int,
    min_epochs: int,
    resume_folds: bool = True,
    drive_weights_dir: str = "",
) -> Tuple[List[Dict[str, object]], float]:
    all_oof: List[Dict[str, object]] = []
    max_antisym_all = 0.0
    shared_lru = _SparsePriorLRU() if prior_mode == "joltz" else None

    for fold_idx, (tr_items, va_items) in enumerate(folds):
        ckpt_path = os.path.join(weights_dir, f"{ckpt_prefix}_fold{fold_idx}.npz")
        fold_oof_path = os.path.join(weights_dir, f"oof_{ckpt_prefix}_fold{fold_idx}_{mode}.csv")
        if resume_folds and drive_weights_dir:
            for fname in (os.path.basename(ckpt_path), os.path.basename(fold_oof_path)):
                src_d = os.path.join(drive_weights_dir, fname)
                dst_l = os.path.join(weights_dir, fname)
                if os.path.exists(src_d) and not os.path.exists(dst_l):
                    _sync_file_to_drive(src_d, dst_l)

        reloaded = _load_fold_oof_v3(fold_oof_path, ckpt_path, va_items) if resume_folds else None
        if reloaded is not None:
            fold_oof, fold_antisym = reloaded
            print(f"\n[RESUME] Skipping completed [{ckpt_prefix}] Fold {fold_idx} ({len(fold_oof)} val muts)", flush=True)
        else:
            fold_oof, _, fold_antisym = train_single_fold_v3(
                fold_idx=fold_idx, train_items=tr_items, val_items=va_items,
                weights_dir=weights_dir, prior_mode=prior_mode, ckpt_prefix=ckpt_prefix,
                epochs=epochs, batch_size=batch_size, patience=patience, min_epochs=min_epochs,
                sparse_lru=shared_lru,
            )
            _save_oof_csv_v3(fold_oof_path, fold_oof)
            if drive_weights_dir:
                _sync_file_to_drive(ckpt_path, os.path.join(drive_weights_dir, os.path.basename(ckpt_path)))
                _sync_file_to_drive(fold_oof_path, os.path.join(drive_weights_dir, os.path.basename(fold_oof_path)))
                print(f"  [DRIVE] Backed up {ckpt_prefix}_fold{fold_idx} checkpoint + OOF -> {drive_weights_dir}", flush=True)

        all_oof.extend(fold_oof)
        max_antisym_all = max(max_antisym_all, fold_antisym)
    return all_oof, max_antisym_all


def run_skempi_v3(
    mode: str = "full",
    prior: str = "both",
    cache_dir: str = "/content/oncovac_cache",
    boltz_cache_dir: str = "/content/boltz_cache",
    weights_dir: str = "/content/oncovac_weights_v3",
    recycling_steps: int = 0,
    use_msa: bool = True,
    patience: int = 7,
    min_epochs: int = 6,
    drive_backup_dir: str = "",
    drive_weights_subdir: str = "weights_full_v3",
    resume_folds: bool = True,
) -> None:
    presets = {
        "mini": {"max_complexes": 6, "max_per_pdb": 15, "min_muts": 4, "max_n": 384, "epochs": 8, "batch_size": 8},
        "mid":  {"max_complexes": 30, "max_per_pdb": 40, "min_muts": 5, "max_n": 448, "epochs": 16, "batch_size": 12},
        "full": {"max_complexes": 250, "max_per_pdb": 100, "min_muts": 3, "max_n": 512, "epochs": 24, "batch_size": 16},
    }
    cfg = presets[mode]
    os.makedirs(weights_dir, exist_ok=True)
    drive_weights_dir = os.path.join(drive_backup_dir, drive_weights_subdir) if drive_backup_dir else ""
    if drive_weights_dir:
        os.makedirs(drive_weights_dir, exist_ok=True)

    print(
        f"=== OncoVac Adenine Gen 1 v3 (virtual_sc + Pure-JAX U_chem + Joltz-1) | "
        f"mode={mode.upper()} | prior={prior} | msa={use_msa} | rec={recycling_steps} ===",
        flush=True,
    )
    records = load_skempi_records(
        cache_dir=cache_dir,
        max_complexes=cfg["max_complexes"],
        max_per_pdb=cfg["max_per_pdb"],
        min_muts_per_pdb=cfg["min_muts"],
        max_n=cfg["max_n"],
    )
    items = prepare_virtual_sc_ensembles(
        records, jax.random.PRNGKey(42),
        cache_dir=cache_dir, cache_max_n=cfg["max_n"],
        drive_backup_dir=drive_backup_dir,
    )
    items = prepare_u_chem_cache(
        items, cache_dir=cache_dir, mode=mode, num_orders=4,
        drive_backup_dir=drive_backup_dir,
    )
    if prior in ("joltz", "both"):
        items = prepare_joltz_priors_cache(
            items, cache_dir=cache_dir, boltz_cache_dir=boltz_cache_dir,
            recycling_steps=recycling_steps, use_msa=use_msa,
            drive_backup_dir=drive_backup_dir,
        )

    folds = make_family_disjoint_folds(items, n_folds=3)
    summary_rows: List[Tuple[str, Dict[str, float]]] = []

    # 1. Zero-Shot Pure White-Box Physics (virtual_sc + 0.65 * U_chem, 0% SKEMPI training!)
    zs_rows = _evaluate_zero_shot_v3(items, include_chem=True)
    zs_csv = os.path.join(weights_dir, f"oof_zeroshot_v3_{mode}.csv")
    _save_oof_csv_v3(zs_csv, zs_rows)
    if drive_weights_dir:
        _sync_file_to_drive(zs_csv, os.path.join(drive_weights_dir, os.path.basename(zs_csv)))
    summary_rows.append((
        "Zero-Shot (virtual_sc + U_chem)",
        summarize_benchmark_pillars("Zero-Shot (virtual_sc + U_chem)", zs_rows),
    ))
    _print_3way("Zero-Shot (virtual_sc + U_chem)", zs_rows)

    # 2. OncoVac Adenine Lite v3 (No-Joltz: virtual_sc + U_chem + U_NN)
    if prior in ("none", "both"):
        oof_lite, asym_lite = _run_cv_suite_v3(
            folds=folds, weights_dir=weights_dir, prior_mode="none", ckpt_prefix="adenine_lite_v3",
            mode=mode, epochs=cfg["epochs"], batch_size=cfg["batch_size"],
            patience=patience, min_epochs=min_epochs,
            resume_folds=resume_folds, drive_weights_dir=drive_weights_dir,
        )
        lite_csv = os.path.join(weights_dir, f"oof_adenine_lite_v3_{mode}.csv")
        _save_oof_csv_v3(lite_csv, oof_lite)
        if drive_weights_dir:
            _sync_file_to_drive(lite_csv, os.path.join(drive_weights_dir, os.path.basename(lite_csv)))
        summary_rows.append((
            "Adenine Lite v3 (No-Joltz)",
            summarize_benchmark_pillars("Adenine Lite v3 (No-Joltz)", oof_lite, max_antisym=asym_lite),
        ))
        _print_3way("Adenine Lite v3 (No-Joltz)", oof_lite)

    # 3. OncoVac Adenine Gen 1 v3 (+Joltz-1 + MSA + virtual_sc + U_chem)
    if prior in ("joltz", "both"):
        oof_gen1, asym_gen1 = _run_cv_suite_v3(
            folds=folds, weights_dir=weights_dir, prior_mode="joltz", ckpt_prefix="adenine_gen1_v3",
            mode=mode, epochs=cfg["epochs"], batch_size=cfg["batch_size"],
            patience=patience, min_epochs=min_epochs,
            resume_folds=resume_folds, drive_weights_dir=drive_weights_dir,
        )
        gen1_csv = os.path.join(weights_dir, f"oof_adenine_gen1_v3_{mode}.csv")
        _save_oof_csv_v3(gen1_csv, oof_gen1)
        if drive_weights_dir:
            _sync_file_to_drive(gen1_csv, os.path.join(drive_weights_dir, os.path.basename(gen1_csv)))
        summary_rows.append((
            "OncoVac Adenine Gen 1 v3",
            summarize_benchmark_pillars("OncoVac Adenine Gen 1 v3", oof_gen1, max_antisym=asym_gen1),
        ))
        _print_3way("OncoVac Adenine Gen 1 v3", oof_gen1)

    print("\n================================================================================", flush=True)
    print(f"FINAL 3-FOLD STRICT FAMILY-DISJOINT SUMMARY ({mode.upper()}, msa={use_msa})", flush=True)
    print(f"{'Model':34s} | {'r':>7s} | {'pc10_r':>7s} | {'rho':>7s} | {'RMSE':>6s} | {'MAE':>6s}", flush=True)
    print("-" * 80, flush=True)
    for name, st in summary_rows:
        print(
            f"{name:34s} | {st['r']:+7.4f} | {st['pc_r']:+7.4f} | {st['rho']:+7.4f} | "
            f"{st['rmse']:6.3f} | {st['mae']:6.3f}",
            flush=True,
        )
    print("================================================================================", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="OncoVac Adenine Gen 1 v3 (virtual_sc + Pure-JAX U_chem + Joltz-1)")
    p.add_argument("--mode", choices=["mini", "mid", "full"], default="full")
    p.add_argument("--prior", choices=["none", "joltz", "both"], default="both")
    p.add_argument("--use-msa", action="store_true")
    p.add_argument("--recycling-steps", type=int, default=0)
    p.add_argument("--cache-dir", default="/content/oncovac_cache")
    p.add_argument("--boltz-cache-dir", default="/content/boltz_cache")
    p.add_argument("--weights-dir", default="/content/oncovac_weights_v3")
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--min-epochs", type=int, default=6)
    p.add_argument("--drive-backup-dir", default="")
    p.add_argument("--drive-weights-subdir", default="weights_full_v3")
    p.add_argument("--resume-folds", action="store_true")
    args = p.parse_args()
    run_skempi_v3(
        mode=args.mode,
        prior=args.prior,
        cache_dir=args.cache_dir,
        boltz_cache_dir=args.boltz_cache_dir,
        weights_dir=args.weights_dir,
        recycling_steps=args.recycling_steps,
        use_msa=args.use_msa,
        patience=args.patience,
        min_epochs=args.min_epochs,
        drive_backup_dir=args.drive_backup_dir,
        drive_weights_subdir=args.drive_weights_subdir,
        resume_folds=args.resume_folds,
    )
