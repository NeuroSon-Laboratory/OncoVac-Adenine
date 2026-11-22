"""Zero-leakage Complex-Disjoint Mini-SKEMPI 2.0 benchmark & Differentiable BAR training runner."""
import csv
import os
import re
import urllib.request
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from cancerfold.af2_core.joltz_prior import prepare_wt_mut_priors
from cancerfold.af2_core.loader import load_pdb_backbone, pad_structure
from cancerfold.af2_core.stochastic_folding import init_folding_params
from cancerfold.analyzer.thermo import binding_ddg_bar, skempi_metrics
from cancerfold.mutation.neighborhood import multi_pocket_mask
from cancerfold.mutation.parser import apply_mutations, encode, parse
from cancerfold.physics.sde import R_GAS, sample_ensemble

SKEMPI_CSV_URL = "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"
RCSB_PDB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"

# Compact, clean canonical SKEMPI 2.0 complexes (N <= 256) for fast complex-disjoint mini-benchmark
MINI_PDB_WHITELIST = [
    "1BRS",  # Barnase - Barstar
    "1PPF",  # Leukocyte elastase - OMTKY3
    "1CHO",  # Alpha-chymotrypsin - OMTKY3
    "1CSE",  # Subtilisin Carlsberg - Eglin c
    "1CBW",  # Chymotrypsin - BPTI
    "2FTL",  # Trypsin - BPTI
    "1TAW",  # Trypsin - APPI
    "1ACB",  # Chymotrypsin - Eglin c
    "1TEC",  # Thermitase - Eglin c
    "1DAN",  # Compact PPI control (filtered automatically if > max_n)
]


def _download_if_missing(url: str, dst_path: str) -> bool:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 200:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=25) as r, open(dst_path, "wb") as f:
            f.write(r.read())
        return True
    except Exception as exc:
        print(f"[WARN] Download failed for {url}: {exc}")
        return False


def _parse_temp_k(raw: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)", str(raw))
    return float(m.group(1)) if m else 298.15


def load_skempi_records(
    cache_dir: str = "/tmp/skempi2_cache",
    pdb_whitelist: Optional[List[str]] = None,
    max_n: int = 256,
    max_per_pdb: int = 20,
) -> List[Dict[str, object]]:
    """
    Downloads official skempi_v2.csv + RCSB PDBs, deduplicates identical mutations by mean ddG,
    and returns validated complex records (zero leakage: grouped by PDB complex ID).
    """
    csv_path = os.path.join(cache_dir, "skempi_v2.csv")
    if not _download_if_missing(SKEMPI_CSV_URL, csv_path):
        raise RuntimeError("Could not download official skempi_v2.csv.")

    whitelist = set(p.upper() for p in pdb_whitelist) if pdb_whitelist else None
    grouped_ddg: Dict[Tuple[str, str, str, str], List[Tuple[float, float]]] = defaultdict(list)

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            pdb_field = row.get("#Pdb", "").strip()
            parts = pdb_field.split("_")
            if len(parts) != 3:
                continue
            pdb_id, ch_a_str, ch_b_str = parts[0].upper(), parts[1], parts[2]
            if whitelist is not None and pdb_id not in whitelist:
                continue

            mut_code = row.get("Mutation(s)_PDB", "").strip()
            if not mut_code:
                continue
            try:
                kd_mut = float(row["Affinity_mut_parsed"])
                kd_wt = float(row["Affinity_wt_parsed"])
                if kd_mut <= 0.0 or kd_wt <= 0.0:
                    continue
            except (ValueError, KeyError, TypeError):
                continue

            temp_k = _parse_temp_k(row.get("Temperature", "298.15"))
            # Official SKEMPI 2.0 convention: ddG = R * T * ln(Kd_mut / Kd_wt) (>0 destabilizing)
            ddg_exp = float(R_GAS * temp_k * np.log(kd_mut / kd_wt))
            if not np.isfinite(ddg_exp) or abs(ddg_exp) > 12.0:
                continue
            grouped_ddg[(pdb_id, ch_a_str, ch_b_str, mut_code)].append((ddg_exp, temp_k))

    # Group by complex to load each PDB once
    by_complex: Dict[Tuple[str, str, str], List[Tuple[str, float, float]]] = defaultdict(list)
    for (pdb_id, ch_a, ch_b, mut_code), vals in grouped_ddg.items():
        ddg_mean = float(np.mean([v[0] for v in vals]))
        temp_mean = float(np.mean([v[1] for v in vals]))
        by_complex[(pdb_id, ch_a, ch_b)].append((mut_code, ddg_mean, temp_mean))

    records: List[Dict[str, object]] = []
    for (pdb_id, ch_a_str, ch_b_str), mut_list in sorted(by_complex.items()):
        pdb_path = os.path.join(cache_dir, "pdbs", f"{pdb_id}.pdb")
        if not _download_if_missing(RCSB_PDB_URL.format(pdb_id=pdb_id), pdb_path):
            continue

        chains_a = list(ch_a_str)
        chains_b = list(ch_b_str)
        try:
            pdb_data = load_pdb_backbone(pdb_path, chains=chains_a + chains_b)
        except Exception as exc:
            print(f"[SKIP] {pdb_id}: {exc}")
            continue

        wt_seq = str(pdb_data["sequence"])
        n_res = len(wt_seq)
        if n_res > max_n:
            print(f"[SKIP] {pdb_id}: N={n_res} exceeds max_n={max_n}")
            continue

        bb, cm, res_mask = pad_structure(pdb_data["backbone"], pdb_data["chain_mask"], max_n=max_n)
        chain_ids: List[str] = pdb_data["chain_ids"]
        pad_len = max_n - n_res

        pa_mask = jnp.pad(
            jnp.array([1.0 if c in chains_a else 0.0 for c in chain_ids], dtype=jnp.float32),
            ((0, pad_len),),
            constant_values=0.0,
        )
        pb_mask = jnp.pad(
            jnp.array([1.0 if c in chains_b else 0.0 for c in chain_ids], dtype=jnp.float32),
            ((0, pad_len),),
            constant_values=0.0,
        )
        if float(jnp.sum(pa_mask)) == 0.0 or float(jnp.sum(pb_mask)) == 0.0:
            continue

        enc_wt = encode(wt_seq)
        aa_wt = jnp.pad(enc_wt["idx"], ((0, pad_len),), constant_values=0)
        phys_wt = jnp.pad(enc_wt["phys_state"], ((0, pad_len), (0, 0)), constant_values=0.0)
        d_ref_ca = jnp.linalg.norm(bb[:, 1, None, :] - bb[None, :, 1, :], axis=-1)

        added = 0
        for mut_code, ddg_exp, temp_k in sorted(mut_list)[:max_per_pdb * 2]:
            if added >= max_per_pdb:
                break
            try:
                mut_seq = apply_mutations(wt_seq, mut_code, res_map=pdb_data["res_map"])
                mut_mask = jnp.zeros((max_n,), dtype=jnp.float32)
                for tok in mut_code.split(","):
                    tok = tok.strip()
                    if tok:
                        info = parse(tok, res_map=pdb_data["res_map"])
                        mut_mask = mut_mask.at[info["idx_0based"]].set(1.0)
                enc_mut = encode(mut_seq)
            except Exception:
                continue

            aa_mut = jnp.pad(enc_mut["idx"], ((0, pad_len),), constant_values=0)
            phys_mut = jnp.pad(enc_mut["phys_state"], ((0, pad_len), (0, 0)), constant_values=0.0)
            weights = multi_pocket_mask(bb[:, 1, :], mut_mask, residue_mask=res_mask)

            records.append({
                "pdb_id": pdb_id,
                "mut_code": mut_code,
                "n_res": n_res,
                "ddg_exp": ddg_exp,
                "temp_k": temp_k,
                "bb": bb,
                "cm": cm,
                "res_mask": res_mask,
                "pa_mask": pa_mask,
                "pb_mask": pb_mask,
                "d_ref_ca": d_ref_ca,
                "weights": weights,
                "aa_wt": aa_wt,
                "aa_mut": aa_mut,
                "phys_wt": phys_wt,
                "phys_mut": phys_mut,
            })
            added += 1

        print(f"[LOAD] {pdb_id} ({ch_a_str}_{ch_b_str}, N={n_res}): {added} validated mutations")

    return records


def prepare_ensembles_and_priors(
    records: List[Dict[str, object]],
    prng_key: jax.Array,
    joltz_forward_fn: Optional[Callable] = None,
    c_s: int = 384,
    c_z: int = 128,
    num_replicas: int = 2,
    num_steps: int = 60,
    burn_in: int = 20,
    stride: int = 20,
) -> List[Dict[str, object]]:
    """
    Stage 1 (Offline): Runs Equilibrium BAOAB SDE (k_rep=0.0) on real crystal PDBs under H_phys
    and extracts frozen WT/Mut priors once at t=0.
    """
    theta_eq = {"k_rep": 0.0, "enm_soften": 1.0}
    sampler_jit = jax.jit(
        lambda bb, k, d_ref, phys, w, m, cm, tk: sample_ensemble(
            bb, k, d_ref, phys, w, m, theta_eq,
            nn_params=None, chain_mask=cm,
            num_replicas=num_replicas, num_steps=num_steps,
            burn_in=burn_in, stride=stride, temp_k=tk,
        )
    )

    t_kin_vals = []
    k_prior_base, k_sde_base = jax.random.split(prng_key)
    prepared = []

    for i, rec in enumerate(records):
        k_wt, k_mut = jax.random.split(jax.random.fold_in(k_sde_base, i))
        out_wt = sampler_jit(
            rec["bb"], k_wt, rec["d_ref_ca"], rec["phys_wt"],
            rec["weights"], rec["res_mask"], rec["cm"], float(rec["temp_k"]),
        )
        out_mut = sampler_jit(
            rec["bb"], k_mut, rec["d_ref_ca"], rec["phys_mut"],
            rec["weights"], rec["res_mask"], rec["cm"], float(rec["temp_k"]),
        )
        t_kin_vals.append(float(out_wt["mean_t_kin"]))

        priors = prepare_wt_mut_priors(
            joltz_forward_fn, rec["aa_wt"], rec["aa_mut"], rec["res_mask"],
            c_s=c_s, c_z=c_z, fallback_key=k_prior_base,
        )
        (s_wt, z_wt), (s_mut, z_mut) = priors["wt"], priors["mut"]

        item = dict(rec)
        item.update({
            "ens_wt": out_wt["ensemble"],
            "ens_mut": out_mut["ensemble"],
            "s_wt": s_wt,
            "z_wt": z_wt,
            "s_mut": s_mut,
            "z_mut": z_mut,
        })
        prepared.append(item)

    print(
        f"[SDE] Cached {len(prepared)} mutation ensembles ({prepared[0]['ens_wt'].shape[0]} frames/state) | "
        f"Real-PDB mean T_kin = {float(np.mean(t_kin_vals)):.1f} K (min={min(t_kin_vals):.1f}, max={max(t_kin_vals):.1f})"
    )
    return prepared


def split_complex_disjoint(
    items: List[Dict[str, object]],
    val_ratio: float = 0.30,
    seed: int = 42,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    """Strict complex-disjoint split: zero PDB overlap between train and validation."""
    pdbs = sorted({str(x["pdb_id"]) for x in items})
    rng = np.random.default_rng(seed)
    rng.shuffle(pdbs)
    n_val = max(1, int(round(len(pdbs) * val_ratio)))
    val_pdbs = set(pdbs[:n_val])
    train_pdbs = set(pdbs[n_val:])
    assert train_pdbs.isdisjoint(val_pdbs), "Data leakage detected!"

    train_items = [x for x in items if x["pdb_id"] in train_pdbs]
    val_items = [x for x in items if x["pdb_id"] in val_pdbs]
    print(f"[SPLIT] Complex-disjoint (ZERO leakage): Train PDBs={sorted(train_pdbs)} ({len(train_items)} muts) | Val PDBs={sorted(val_pdbs)} ({len(val_items)} muts)")
    return train_items, val_items


def _huber_loss(pred: jnp.ndarray, target: jnp.ndarray, delta: float = 1.5) -> jnp.ndarray:
    err = jnp.abs(pred - target)
    return jnp.where(err <= delta, 0.5 * err ** 2, delta * (err - 0.5 * delta))


def run_mini_skempi_benchmark(
    max_n: int = 256,
    max_per_pdb: int = 12,
    epochs: int = 25,
    lr: float = 2e-3,
    weight_decay: float = 1e-2,
    joltz_forward_fn: Optional[Callable] = None,
) -> None:
    print("=== OncoVac-Adenine Gen 1 | Zero-Leakage Mini-SKEMPI 2.0 Benchmark ===")
    records = load_skempi_records(
        pdb_whitelist=MINI_PDB_WHITELIST,
        max_n=max_n,
        max_per_pdb=max_per_pdb,
    )
    if len(records) < 10:
        raise RuntimeError(f"Too few SKEMPI records loaded ({len(records)}).")

    key = jax.random.PRNGKey(42)
    k_sde, k_init = jax.random.split(key)
    items = prepare_ensembles_and_priors(records, k_sde, joltz_forward_fn=joltz_forward_fn)
    train_items, val_items = split_complex_disjoint(items)

    theta_eq = {"k_rep": 0.0, "enm_soften": 1.0}
    nn_params = init_folding_params(k_init, c_s=384, c_z=128)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr, weight_decay=weight_decay),
    )
    opt_state = optimizer.init(nn_params)

    @jax.jit
    def _predict_one(params, ens_wt, ens_mut, d_ref, phys_wt, phys_mut, w, pa, pb, s_wt, z_wt, s_mut, z_mut, cm, tk):
        return binding_ddg_bar(
            ens_wt, ens_mut, d_ref, phys_wt, phys_mut, w, pa, pb, theta_eq,
            single_wt=s_wt, pair_wt=z_wt, single_mut=s_mut, pair_mut=z_mut,
            single_wt_unbound=s_wt, pair_wt_unbound=z_wt,
            single_mut_unbound=s_mut, pair_mut_unbound=z_mut,
            nn_params=params, chain_mask=cm, temp_k=tk,
        )

    def _eval_split(params, split_items, label: str) -> Dict[str, float]:
        preds, trues, phys_abs, nn_abs = [], [], [], []
        for x in split_items:
            out = _predict_one(
                params, x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
                x["phys_wt"], x["phys_mut"], x["weights"], x["pa_mask"], x["pb_mask"],
                x["s_wt"], x["z_wt"], x["s_mut"], x["z_mut"], x["cm"], float(x["temp_k"]),
            )
            preds.append(float(out["ddg_bind"]))
            trues.append(float(x["ddg_exp"]))
            phys_abs.append(abs(float(out["dh_phys_mean"])))
            nn_abs.append(abs(float(out["dh_nn_mean"])))
        m = skempi_metrics(jnp.asarray(preds), jnp.asarray(trues))
        print(
            f"[{label}] r={float(m['pearson_r']):+.3f} | rho={float(m['spearman_rho']):+.3f} | "
            f"RMSE={float(m['rmse']):.3f} | MAE={float(m['mae']):.3f} | "
            f"mean|dH_phys|={np.mean(phys_abs):.3f} vs mean|dH_nn|={np.mean(nn_abs):.3f}"
        )
        return {k: float(v) for k, v in m.items()}

    # Zero-shot pure H_phys baseline (before any ML training)
    print("\n--- Zero-Shot Pure Analytical Physics (H_phys only) ---")
    _eval_split(None, val_items, "VAL Zero-Shot H_phys")

    @jax.jit
    def _loss_and_grad(params, ens_wt, ens_mut, d_ref, phys_wt, phys_mut, w, pa, pb, s_wt, z_wt, s_mut, z_mut, cm, tk, y_true):
        def _loss_fn(p):
            out = binding_ddg_bar(
                ens_wt, ens_mut, d_ref, phys_wt, phys_mut, w, pa, pb, theta_eq,
                single_wt=s_wt, pair_wt=z_wt, single_mut=s_mut, pair_mut=z_mut,
                single_wt_unbound=s_wt, pair_wt_unbound=z_wt,
                single_mut_unbound=s_mut, pair_mut_unbound=z_mut,
                nn_params=p, chain_mask=cm, temp_k=tk,
            )
            # Mild white-box penalty keeping U_NN as a physical residual
            nn_reg = 1e-3 * (out["dh_nn_mean"] ** 2)
            return _huber_loss(out["ddg_bind"], y_true) + nn_reg
        return jax.value_and_grad(_loss_fn)(params)

    print("\n--- Differentiable BAR Training (H_phys + U_NN) ---")
    rng = np.random.default_rng(7)
    for ep in range(1, epochs + 1):
        order = rng.permutation(len(train_items))
        ep_losses = []
        for idx in order:
            x = train_items[int(idx)]
            loss_val, grads = _loss_and_grad(
                nn_params, x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
                x["phys_wt"], x["phys_mut"], x["weights"], x["pa_mask"], x["pb_mask"],
                x["s_wt"], x["z_wt"], x["s_mut"], x["z_mut"], x["cm"],
                float(x["temp_k"]), jnp.asarray(x["ddg_exp"], dtype=jnp.float32),
            )
            updates, opt_state = optimizer.update(grads, opt_state, nn_params)
            nn_params = optax.apply_updates(nn_params, updates)
            ep_losses.append(float(loss_val))

        if ep == 1 or ep % 5 == 0 or ep == epochs:
            print(f"Epoch {ep:02d}/{epochs} | train_loss={np.mean(ep_losses):.4f}")
            _eval_split(nn_params, train_items, f"TRAIN ep{ep:02d}")
            _eval_split(nn_params, val_items, f"VAL   ep{ep:02d}")

    # Exact antisymmetry audit on unseen validation complexes
    antisym_errs = []
    for x in val_items[:5]:
        fwd = _predict_one(
            nn_params, x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
            x["phys_wt"], x["phys_mut"], x["weights"], x["pa_mask"], x["pb_mask"],
            x["s_wt"], x["z_wt"], x["s_mut"], x["z_mut"], x["cm"], float(x["temp_k"]),
        )["ddg_bind"]
        rev = _predict_one(
            nn_params, x["ens_mut"], x["ens_wt"], x["d_ref_ca"],
            x["phys_mut"], x["phys_wt"], x["weights"], x["pa_mask"], x["pb_mask"],
            x["s_mut"], x["z_mut"], x["s_wt"], x["z_wt"], x["cm"], float(x["temp_k"]),
        )["ddg_bind"]
        antisym_errs.append(float(jnp.abs(fwd + rev)))
    print(f"\n[AUDIT] Unseen-complex exact BAR antisymmetry max|fwd+rev| = {max(antisym_errs):.2e}")


if __name__ == "__main__":
    run_mini_skempi_benchmark()
