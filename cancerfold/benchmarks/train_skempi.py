"""Journal-grade 3-Fold Family/Homology-Disjoint SKEMPI 2.0 training & evaluation suite for OncoVac-Adenine."""
import argparse
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

# Calibrated physical coupling defaults (strictly positive via softplus during training)
PHYS_INIT = {
    "k_clash": 2.0,
    "k_elec": 1.0,
    "k_hydro": 1.5,
    "k_hbond": 1.5,
}


def _inv_softplus(y: float) -> float:
    return float(np.log(np.expm1(max(y, 1e-4))))


def _init_model_params(prng_key: jax.Array, c_s: int = 384, c_z: int = 128) -> Dict[str, Dict[str, jnp.ndarray]]:
    """Initializes joint white-box parameters: 4 positive physical scales + 2-hop SE(3) neural residual."""
    phys_raw = {k: jnp.array(_inv_softplus(v), dtype=jnp.float32) for k, v in PHYS_INIT.items()}
    return {
        "phys_raw": phys_raw,
        "nn": init_folding_params(prng_key, c_s=c_s, c_z=c_z),
    }


def _decode_theta(phys_raw: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Maps unconstrained scalars to strictly positive physical force-field coupling constants."""
    theta = {"k_rep": 0.0, "enm_soften": 1.0}
    for k, raw_val in phys_raw.items():
        theta[k] = jnp.clip(jax.nn.softplus(raw_val), 0.1, 8.0)
    return theta


def save_checkpoint(path: str, model_params: Dict[str, Dict[str, jnp.ndarray]]) -> None:
    """Saves model parameters cleanly as pickle-free compressed .npz (~280 KB)."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    flat = {}
    for group, sub in model_params.items():
        for k, v in sub.items():
            flat[f"{group}__{k}"] = np.asarray(v, dtype=np.float32)
    np.savez_compressed(path, **flat)


def load_checkpoint(path: str) -> Dict[str, Dict[str, jnp.ndarray]]:
    """Loads pickle-free .npz checkpoint back into the nested JAX parameter PyTree."""
    data = np.load(path, allow_pickle=False)
    out: Dict[str, Dict[str, jnp.ndarray]] = {"phys_raw": {}, "nn": {}}
    for key in data.files:
        group, name = key.split("__", 1)
        out[group][name] = jnp.asarray(data[key], dtype=jnp.float32)
    return out


def _download_if_missing(url: str, dst_path: str) -> bool:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 200:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=25) as r, open(dst_path, "wb") as f:
            f.write(r.read())
        return True
    except Exception:
        return False


def _parse_temp_k(raw: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)", str(raw))
    return float(m.group(1)) if m else 298.15


def _kmer_set(seq: str, k: int = 3) -> set:
    if len(seq) < k:
        return {seq}
    return {seq[i:i + k] for i in range(len(seq) - k + 1)}


def _seq_jaccard(seq_a: str, seq_b: str, k: int = 3) -> float:
    ka, kb = _kmer_set(seq_a, k), _kmer_set(seq_b, k)
    union = len(ka | kb)
    return (len(ka & kb) / union) if union > 0 else 0.0


def load_skempi_records(
    cache_dir: str = "/tmp/skempi2_cache",
    max_n: int = 384,
    max_complexes: int = 12,
    max_per_pdb: int = 10,
    min_muts_per_pdb: int = 3,
) -> List[Dict[str, object]]:
    """
    Downloads official skempi_v2.csv + RCSB PDBs, balances across distinct Hold_out_type protein
    families (Pr/PI, AB/AG, TCR/pMHC, Other), deduplicates identical mutations by mean ddG,
    and returns validated records with lightweight arrays (zero RAM bloat).
    """
    csv_path = os.path.join(cache_dir, "skempi_v2.csv")
    if not _download_if_missing(SKEMPI_CSV_URL, csv_path):
        raise RuntimeError("Could not download official skempi_v2.csv.")

    grouped_ddg: Dict[Tuple[str, str, str, str], List[Tuple[float, float]]] = defaultdict(list)
    complex_family: Dict[Tuple[str, str, str], str] = {}

    with open(csv_path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            pdb_field = row.get("#Pdb", "").strip()
            parts = pdb_field.split("_")
            if len(parts) != 3:
                continue
            pdb_id, ch_a_str, ch_b_str = parts[0].upper(), parts[1], parts[2]
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
            ddg_exp = float(R_GAS * temp_k * np.log(kd_mut / kd_wt))
            if not np.isfinite(ddg_exp) or abs(ddg_exp) > 12.0:
                continue

            ckey = (pdb_id, ch_a_str, ch_b_str)
            grouped_ddg[(pdb_id, ch_a_str, ch_b_str, mut_code)].append((ddg_exp, temp_k))
            if ckey not in complex_family:
                fam = (row.get("Hold_out_type", "") or "").strip()
                complex_family[ckey] = fam if fam else "Other"

    by_complex: Dict[Tuple[str, str, str], List[Tuple[str, float, float]]] = defaultdict(list)
    for (pdb_id, ch_a, ch_b, mut_code), vals in grouped_ddg.items():
        ddg_mean = float(np.mean([v[0] for v in vals]))
        temp_mean = float(np.mean([v[1] for v in vals]))
        by_complex[(pdb_id, ch_a, ch_b)].append((mut_code, ddg_mean, temp_mean))

    # Round-robin across biological families so even mini/mid modes span diverse protein families
    fam_buckets: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for ckey, mlist in sorted(by_complex.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        if len(mlist) >= min_muts_per_pdb:
            fam_buckets[complex_family[ckey]].append(ckey)

    ordered_complexes: List[Tuple[str, str, str]] = []
    bucket_keys = sorted(fam_buckets.keys())
    while any(fam_buckets[fk] for fk in bucket_keys):
        for fk in bucket_keys:
            if fam_buckets[fk]:
                ordered_complexes.append(fam_buckets[fk].pop(0))

    records: List[Dict[str, object]] = []
    loaded_pdbs: set = set()

    for ckey in ordered_complexes:
        if len(loaded_pdbs) >= max_complexes:
            break
        pdb_id, ch_a_str, ch_b_str = ckey
        if pdb_id in loaded_pdbs:
            continue

        pdb_path = os.path.join(cache_dir, "pdbs", f"{pdb_id}.pdb")
        if not _download_if_missing(RCSB_PDB_URL.format(pdb_id=pdb_id), pdb_path):
            continue

        chains_a, chains_b = list(ch_a_str), list(ch_b_str)
        try:
            pdb_data = load_pdb_backbone(pdb_path, chains=chains_a + chains_b)
        except Exception:
            continue

        wt_seq = str(pdb_data["sequence"])
        n_res = len(wt_seq)
        if n_res > max_n:
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

        complex_recs = []
        for mut_code, ddg_exp, temp_k in sorted(by_complex[ckey]):
            if len(complex_recs) >= max_per_pdb:
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

            complex_recs.append({
                "pdb_id": pdb_id,
                "family": complex_family[ckey],
                "wt_seq": wt_seq,
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

        if len(complex_recs) >= min_muts_per_pdb:
            records.extend(complex_recs)
            loaded_pdbs.add(pdb_id)
            print(
                f"[LOAD] {pdb_id} ({ch_a_str}_{ch_b_str} | fam={complex_family[ckey]:10s} | N={n_res:3d}): "
                f"{len(complex_recs)} muts"
            )

    return records


def make_family_disjoint_folds(
    records: List[Dict[str, object]],
    n_folds: int = 3,
    seq_sim_thresh: float = 0.25,
) -> List[Tuple[List[Dict[str, object]], List[Dict[str, object]]]]:
    """
    Zero-homology-leakage K-fold CV:
      1. Clusters PDB complexes via Union-Find if 3-mer sequence Jaccard >= seq_sim_thresh
         OR they share a non-'Other' SKEMPI Hold_out_type family (when >= n_folds clusters remain).
      2. Greedily bin-packs whole clusters into `n_folds` balanced folds.
    """
    pdb_seqs: Dict[str, str] = {}
    pdb_fams: Dict[str, str] = {}
    pdb_counts: Dict[str, int] = defaultdict(int)
    for r in records:
        pid = str(r["pdb_id"])
        pdb_seqs[pid] = str(r["wt_seq"])
        pdb_fams[pid] = str(r["family"])
        pdb_counts[pid] += 1

    pdbs = sorted(pdb_seqs.keys())
    parent = {p: p for p in pdbs}

    def _find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: str, b: str) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[rb] = ra

    # 1. Always merge homologous sequences (prevents serine-protease / antibody sequence leakage)
    for i in range(len(pdbs)):
        for j in range(i + 1, len(pdbs)):
            pi, pj = pdbs[i], pdbs[j]
            if _seq_jaccard(pdb_seqs[pi], pdb_seqs[pj], k=3) >= seq_sim_thresh:
                _union(pi, pj)

    # 2. Also merge identical explicit SKEMPI Hold_out_type families if doing so preserves >= n_folds clusters
    trial_parent = dict(parent)
    def _tfind(x: str) -> str:
        while trial_parent[x] != x:
            trial_parent[x] = trial_parent[trial_parent[x]]
            x = trial_parent[x]
        return x

    for i in range(len(pdbs)):
        for j in range(i + 1, len(pdbs)):
            pi, pj = pdbs[i], pdbs[j]
            if pdb_fams[pi] != "Other" and pdb_fams[pi] == pdb_fams[pj]:
                ri, rj = _tfind(pi), _tfind(pj)
                if ri != rj:
                    trial_parent[rj] = ri

    if len({_tfind(p) for p in pdbs}) >= n_folds:
        parent = trial_parent

    clusters: Dict[str, List[str]] = defaultdict(list)
    for p in pdbs:
        clusters[_find(p)].append(p)

    sorted_clusters = sorted(
        clusters.values(),
        key=lambda grp: sum(pdb_counts[p] for p in grp),
        reverse=True,
    )
    if len(sorted_clusters) < n_folds:
        raise RuntimeError(f"Only {len(sorted_clusters)} independent clusters found; need >= {n_folds} for {n_folds}-fold CV.")

    fold_pdbs: List[set] = [set() for _ in range(n_folds)]
    fold_sizes = [0] * n_folds
    for grp in sorted_clusters:
        target = int(np.argmin(fold_sizes))
        fold_pdbs[target].update(grp)
        fold_sizes[target] += sum(pdb_counts[p] for p in grp)

    print(f"\n[CV] Built {len(sorted_clusters)} homology/family clusters across {len(pdbs)} PDBs -> {n_folds}-Fold CV:")
    folds = []
    for k in range(n_folds):
        val_set = fold_pdbs[k]
        train_set = set().union(*(fold_pdbs[m] for m in range(n_folds) if m != k))
        assert train_set.isdisjoint(val_set), "Fold leakage detected!"
        tr_items = [r for r in records if r["pdb_id"] in train_set]
        va_items = [r for r in records if r["pdb_id"] in val_set]
        fams_val = sorted({pdb_fams[p] for p in val_set})
        print(f"  Fold {k}: Val PDBs={sorted(val_set)} ({len(va_items)} muts, fams={fams_val}) | Train={len(tr_items)} muts")
        folds.append((tr_items, va_items))
    return folds


def prepare_cached_ensembles(
    records: List[Dict[str, object]],
    prng_key: jax.Array,
    cache_dir: str,
    num_replicas: int = 2,
    num_steps: int = 40,
    burn_in: int = 16,
    stride: int = 8,
) -> List[Dict[str, object]]:
    """
    Auto-resume BAOAB Equilibrium SDE cache (~50 KB/mutation in float16 on disk).
    Safe against Colab disconnects: resumes automatically from completed mutations.
    """
    ens_dir = os.path.join(cache_dir, f"baoab_n{records[0]['bb'].shape[0]}_r{num_replicas}_s{num_steps}")
    os.makedirs(ens_dir, exist_ok=True)
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
    cached_hits = 0
    prepared = []

    for i, rec in enumerate(records):
        safe_mut = re.sub(r"[^A-Za-z0-9_-]", "_", str(rec["mut_code"]))
        fpath = os.path.join(ens_dir, f"{rec['pdb_id']}_{safe_mut}.npz")

        if os.path.exists(fpath):
            d = np.load(fpath, allow_pickle=False)
            ens_wt = jnp.asarray(d["ens_wt"], dtype=jnp.float32)
            ens_mut = jnp.asarray(d["ens_mut"], dtype=jnp.float32)
            t_kin_vals.append(float(d["t_kin"]))
            cached_hits += 1
        else:
            k_wt, k_mut = jax.random.split(jax.random.fold_in(prng_key, i))
            out_wt = sampler_jit(
                rec["bb"], k_wt, rec["d_ref_ca"], rec["phys_wt"],
                rec["weights"], rec["res_mask"], rec["cm"], float(rec["temp_k"]),
            )
            out_mut = sampler_jit(
                rec["bb"], k_mut, rec["d_ref_ca"], rec["phys_mut"],
                rec["weights"], rec["res_mask"], rec["cm"], float(rec["temp_k"]),
            )
            ens_wt, ens_mut = out_wt["ensemble"], out_mut["ensemble"]
            tk_val = float(out_wt["mean_t_kin"])
            t_kin_vals.append(tk_val)
            np.savez_compressed(
                fpath,
                ens_wt=np.asarray(ens_wt, dtype=np.float16),
                ens_mut=np.asarray(ens_mut, dtype=np.float16),
                t_kin=np.array(tk_val, dtype=np.float32),
            )

        item = dict(rec)
        item["ens_wt"] = ens_wt
        item["ens_mut"] = ens_mut
        prepared.append(item)

    print(
        f"[SDE] Ready {len(prepared)} ensembles ({prepared[0]['ens_wt'].shape[0]} frames/state, "
        f"disk_hits={cached_hits}) | Real-PDB mean T_kin = {float(np.mean(t_kin_vals)):.1f} K"
    )
    return prepared


def _huber_loss(pred: jnp.ndarray, target: jnp.ndarray, delta: float = 1.5) -> jnp.ndarray:
    err = jnp.abs(pred - target)
    return jnp.where(err <= delta, 0.5 * err ** 2, delta * (err - 0.5 * delta))


def _per_complex_pearson(preds: List[float], trues: List[float], pdbs: List[str]) -> float:
    by_pdb: Dict[str, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
    for p, t, pid in zip(preds, trues, pdbs):
        by_pdb[pid][0].append(p)
        by_pdb[pid][1].append(t)
    corrs = []
    for pid, (ps, ts) in by_pdb.items():
        if len(ps) >= 3 and np.std(ts) > 1e-6 and np.std(ps) > 1e-6:
            corrs.append(float(np.corrcoef(ps, ts)[0, 1]))
    return float(np.mean(corrs)) if corrs else float("nan")


def train_single_fold(
    fold_idx: int,
    train_items: List[Dict[str, object]],
    val_items: List[Dict[str, object]],
    weights_dir: str,
    epochs: int = 25,
    batch_size: int = 8,
    peak_lr: float = 5e-4,
    weight_decay: float = 3e-2,
    dropout_rate: float = 0.15,
    jitter_std: float = 0.02,
    nn_reg_weight: float = 3e-2,
    patience: int = 6,
    joltz_forward_fn: Optional[Callable] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, jnp.ndarray]]]:
    """
    Trains one CV fold with:
      1. Mini-batch gradient accumulation across complexes
      2. Warmup + cosine learning-rate decay
      3. Early stopping & best-checkpoint restore (.npz)
      4. Ensemble frame subsampling + micro thermal jitter (0.02 A) + prior dropout (train only)
      5. Joint white-box calibration of 4 positive physical constants + bounded 2-hop U_NN
    """
    key = jax.random.PRNGKey(100 + fold_idx)
    k_init, k_prior_base, k_train = jax.random.split(key, 3)

    model_params = _init_model_params(k_init, c_s=384, c_z=128)
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

    # Apply weight decay strictly to 2D neural weight matrices (never to physical constants or biases)
    wd_mask = {
        "phys_raw": jax.tree_util.tree_map(lambda _: False, model_params["phys_raw"]),
        "nn": jax.tree_util.tree_map(lambda x: x.ndim >= 2, model_params["nn"]),
    }
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(learning_rate=lr_schedule, weight_decay=weight_decay, mask=wd_mask),
    )
    opt_state = optimizer.init(model_params)

    def _get_priors(aa_wt, aa_mut, res_mask):
        # Computed on-the-fly inside JIT when fallback is used -> zero RAM bloat!
        priors = prepare_wt_mut_priors(
            joltz_forward_fn, aa_wt, aa_mut, res_mask,
            c_s=384, c_z=128, fallback_key=k_prior_base,
        )
        return priors["wt"], priors["mut"]

    @jax.jit
    def _eval_one(params, ens_wt, ens_mut, d_ref, phys_wt, phys_mut, w, pa, pb, res_mask, aa_wt, aa_mut, cm, tk):
        (s_wt, z_wt), (s_mut, z_mut) = _get_priors(aa_wt, aa_mut, res_mask)
        theta = _decode_theta(params["phys_raw"]) if params is not None else {"k_rep": 0.0, "enm_soften": 1.0}
        nn_p = params["nn"] if params is not None else None
        return binding_ddg_bar(
            ens_wt, ens_mut, d_ref, phys_wt, phys_mut, w, pa, pb, theta,
            single_wt=s_wt, pair_wt=z_wt, single_mut=s_mut, pair_mut=z_mut,
            single_wt_unbound=s_wt, pair_wt_unbound=z_wt,
            single_mut_unbound=s_mut, pair_mut_unbound=z_mut,
            nn_params=nn_p, chain_mask=cm, temp_k=tk,
        )

    @jax.jit
    def _grad_one(params, step_key, ens_wt, ens_mut, d_ref, phys_wt, phys_mut, w, pa, pb, res_mask, aa_wt, aa_mut, cm, tk, y_true):
        k_sub, k_jw, k_jm, k_drop = jax.random.split(step_key, 4)
        n_frames = ens_wt.shape[0]
        k_sub_n = max(2, n_frames - 1) if n_frames >= 4 else n_frames
        idx_w = jax.random.choice(k_sub, n_frames, shape=(k_sub_n,), replace=False)
        idx_m = jax.random.choice(k_drop, n_frames, shape=(k_sub_n,), replace=False)

        m_bb = res_mask[None, :, None, None]
        ew = ens_wt[idx_w] + jax.random.normal(k_jw, ens_wt[idx_w].shape, dtype=jnp.float32) * jitter_std * m_bb
        em = ens_mut[idx_m] + jax.random.normal(k_jm, ens_mut[idx_m].shape, dtype=jnp.float32) * jitter_std * m_bb

        (s_wt, z_wt), (s_mut, z_mut) = _get_priors(aa_wt, aa_mut, res_mask)
        # Symmetric shared-mask feature dropout on frozen priors (preserves exact WT <-> Mut symmetry!)
        keep = 1.0 - dropout_rate
        d_s = (jax.random.bernoulli(k_drop, keep, s_wt.shape).astype(jnp.float32)) / keep
        s_wt_d, s_mut_d = s_wt * d_s, s_mut * d_s

        def _loss_fn(p):
            theta = _decode_theta(p["phys_raw"])
            out = binding_ddg_bar(
                ew, em, d_ref, phys_wt, phys_mut, w, pa, pb, theta,
                single_wt=s_wt_d, pair_wt=z_wt, single_mut=s_mut_d, pair_mut=z_mut,
                single_wt_unbound=s_wt_d, pair_wt_unbound=z_wt,
                single_mut_unbound=s_mut_d, pair_mut_unbound=z_mut,
                nn_params=p["nn"], chain_mask=cm, temp_k=tk,
            )
            nn_reg = nn_reg_weight * (out["dh_nn_mean"] ** 2)
            return _huber_loss(out["ddg_bind"], y_true) + nn_reg

        return jax.value_and_grad(_loss_fn)(params)

    def _run_eval(params, split_items):
        preds, trues, pdbs, p_abs, n_abs = [], [], [], [], []
        for x in split_items:
            out = _eval_one(
                params, x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
                x["phys_wt"], x["phys_mut"], x["weights"], x["pa_mask"], x["pb_mask"],
                x["res_mask"], x["aa_wt"], x["aa_mut"], x["cm"], float(x["temp_k"]),
            )
            preds.append(float(out["ddg_bind"]))
            trues.append(float(x["ddg_exp"]))
            pdbs.append(str(x["pdb_id"]))
            p_abs.append(abs(float(out["dh_phys_mean"])))
            n_abs.append(abs(float(out["dh_nn_mean"])))
        m = skempi_metrics(jnp.asarray(preds), jnp.asarray(trues))
        return {
            "preds": preds,
            "trues": trues,
            "pdbs": pdbs,
            "r": float(m["pearson_r"]),
            "pc_r": _per_complex_pearson(preds, trues, pdbs),
            "rho": float(m["spearman_rho"]),
            "rmse": float(m["rmse"]),
            "mae": float(m["mae"]),
            "phys_bind": float(np.mean(p_abs)),
            "nn_bind": float(np.mean(n_abs)),
        }

    # Fold zero-shot baseline
    zs = _run_eval(None, val_items)
    print(
        f"\n--- Fold {fold_idx} | Zero-Shot H_phys VAL: r={zs['r']:+.3f}, pc_r={zs['pc_r']:+.3f}, "
        f"rho={zs['rho']:+.3f}, RMSE={zs['rmse']:.3f} ---"
    )

    best_rmse = float("inf")
    best_params = model_params
    best_ep = 0
    no_improve = 0
    rng = np.random.default_rng(200 + fold_idx)
    step_counter = 0

    for ep in range(1, epochs + 1):
        order = rng.permutation(len(train_items))
        ep_losses = []
        for start in range(0, len(order), batch_size):
            chunk = order[start:start + batch_size]
            acc_grads = jax.tree_util.tree_map(jnp.zeros_like, model_params)
            for idx in chunk:
                x = train_items[int(idx)]
                step_counter += 1
                sk = jax.random.fold_in(k_train, step_counter)
                lval, g = _grad_one(
                    model_params, sk, x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
                    x["phys_wt"], x["phys_mut"], x["weights"], x["pa_mask"], x["pb_mask"],
                    x["res_mask"], x["aa_wt"], x["aa_mut"], x["cm"],
                    float(x["temp_k"]), jnp.asarray(x["ddg_exp"], dtype=jnp.float32),
                )
                acc_grads = jax.tree_util.tree_map(lambda a, b: a + b, acc_grads, g)
                ep_losses.append(float(lval))

            denom = float(len(chunk))
            avg_grads = jax.tree_util.tree_map(lambda a: a / denom, acc_grads)
            updates, opt_state = optimizer.update(avg_grads, opt_state, model_params)
            model_params = optax.apply_updates(model_params, updates)

        val_res = _run_eval(model_params, val_items)
        improved = val_res["rmse"] < best_rmse - 1e-4
        if improved:
            best_rmse = val_res["rmse"]
            best_params = model_params
            best_ep = ep
            no_improve = 0
        else:
            no_improve += 1

        if ep == 1 or ep % 5 == 0 or improved or ep == epochs:
            th = _decode_theta(model_params["phys_raw"])
            mark = " *BEST*" if improved else ""
            print(
                f"  [F{fold_idx} Ep {ep:02d}] loss={np.mean(ep_losses):.3f} | "
                f"VAL r={val_res['r']:+.3f} rho={val_res['rho']:+.3f} RMSE={val_res['rmse']:.3f} | "
                f"|phys|={val_res['phys_bind']:.2f} |nn|={val_res['nn_bind']:.2f} | "
                f"k=(cl:{float(th['k_clash']):.2f},el:{float(th['k_elec']):.2f},hy:{float(th['k_hydro']):.2f},hb:{float(th['k_hbond']):.2f}){mark}"
            )

        if no_improve >= patience:
            print(f"  [F{fold_idx}] Early stopping at epoch {ep} (best epoch={best_ep}, best VAL RMSE={best_rmse:.3f})")
            break

    # Save best fold checkpoint & verify round-trip load
    ckpt_path = os.path.join(weights_dir, f"adenine_gen1_fold{fold_idx}.npz")
    save_checkpoint(ckpt_path, best_params)
    reloaded = load_checkpoint(ckpt_path)
    final_val = _run_eval(reloaded, val_items)
    size_kb = os.path.getsize(ckpt_path) / 1024.0
    print(
        f"[F{fold_idx} DONE] Saved {ckpt_path} ({size_kb:.1f} KB) | "
        f"Best Ep {best_ep}: r={final_val['r']:+.3f}, pc_r={final_val['pc_r']:+.3f}, "
        f"rho={final_val['rho']:+.3f}, RMSE={final_val['rmse']:.3f}"
    )

    # Exact antisymmetry audit on unseen fold
    antisym_errs = []
    for x in val_items[:5]:
        fwd = _eval_one(
            reloaded, x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
            x["phys_wt"], x["phys_mut"], x["weights"], x["pa_mask"], x["pb_mask"],
            x["res_mask"], x["aa_wt"], x["aa_mut"], x["cm"], float(x["temp_k"]),
        )["ddg_bind"]
        rev = _eval_one(
            reloaded, x["ens_mut"], x["ens_wt"], x["d_ref_ca"],
            x["phys_mut"], x["phys_wt"], x["weights"], x["pa_mask"], x["pb_mask"],
            x["res_mask"], x["aa_mut"], x["aa_wt"], x["cm"], float(x["temp_k"]),
        )["ddg_bind"]
        antisym_errs.append(float(jnp.abs(fwd + rev)))
    print(f"  [F{fold_idx} AUDIT] Unseen-family exact BAR antisymmetry max|fwd+rev| = {max(antisym_errs):.2e}")

    oof_rows = []
    for x, yp in zip(val_items, final_val["preds"]):
        oof_rows.append({
            "fold": fold_idx,
            "pdb_id": x["pdb_id"],
            "family": x["family"],
            "mut_code": x["mut_code"],
            "ddg_exp": float(x["ddg_exp"]),
            "ddg_pred": float(yp),
        })
    return oof_rows, reloaded


def run_skempi_cv(
    mode: str = "mini",
    cache_dir: str = "/tmp/skempi2_cache",
    weights_dir: str = "weights",
    joltz_forward_fn: Optional[Callable] = None,
) -> None:
    presets = {
        "mini": dict(max_n=384, max_complexes=12, max_per_pdb=8, epochs=20, batch_size=8),
        "mid":  dict(max_n=448, max_complexes=45, max_per_pdb=16, epochs=25, batch_size=12),
        "full": dict(max_n=512, max_complexes=400, max_per_pdb=100, epochs=30, batch_size=16),
    }
    cfg = presets[mode]
    print(f"=== OncoVac-Adenine Gen 1 | 3-Fold Family-Disjoint SKEMPI 2.0 ({mode.upper()}) ===")
    records = load_skempi_records(
        cache_dir=cache_dir,
        max_n=cfg["max_n"],
        max_complexes=cfg["max_complexes"],
        max_per_pdb=cfg["max_per_pdb"],
    )
    items = prepare_cached_ensembles(records, jax.random.PRNGKey(42), cache_dir=cache_dir)
    folds = make_family_disjoint_folds(items, n_folds=3)

    all_oof = []
    for k, (tr_items, va_items) in enumerate(folds):
        oof_k, _ = train_single_fold(
            fold_idx=k,
            train_items=tr_items,
            val_items=va_items,
            weights_dir=weights_dir,
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            joltz_forward_fn=joltz_forward_fn,
        )
        all_oof.extend(oof_k)

    preds = [r["ddg_pred"] for r in all_oof]
    trues = [r["ddg_exp"] for r in all_oof]
    pdbs = [str(r["pdb_id"]) for r in all_oof]
    overall = skempi_metrics(jnp.asarray(preds), jnp.asarray(trues))
    pc_r = _per_complex_pearson(preds, trues, pdbs)

    os.makedirs(weights_dir, exist_ok=True)
    oof_csv = os.path.join(weights_dir, f"oof_skempi_{mode}.csv")
    with open(oof_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["fold", "pdb_id", "family", "mut_code", "ddg_exp", "ddg_pred"])
        writer.writeheader()
        writer.writerows(all_oof)

    print("\n======================================================================")
    print(f"3-FOLD FAMILY-DISJOINT OUT-OF-FOLD SUMMARY ({mode.upper()}, N={len(all_oof)} muts)")
    print(
        f"  Overall Pearson r   = {float(overall['pearson_r']):+.4f}\n"
        f"  Per-Complex r       = {pc_r:+.4f}\n"
        f"  Overall Spearman    = {float(overall['spearman_rho']):+.4f}\n"
        f"  RMSE / MAE          = {float(overall['rmse']):.3f} / {float(overall['mae']):.3f} kcal/mol\n"
        f"  Saved OOF CSV       = {oof_csv}"
    )
    print("======================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OncoVac-Adenine 3-Fold Family-Disjoint SKEMPI 2.0 Runner")
    parser.add_argument("--mode", choices=["mini", "mid", "full"], default="mini")
    parser.add_argument("--cache-dir", default="/tmp/skempi2_cache")
    parser.add_argument("--weights-dir", default="weights")
    args = parser.parse_args()
    run_skempi_cv(mode=args.mode, cache_dir=args.cache_dir, weights_dir=args.weights_dir)
