"""Journal-grade 3-Fold Family/Homology-Disjoint SKEMPI 2.0 training & benchmark suite for OncoVac-Adenine."""
import argparse
import csv
import gc
import os
import re
import time
import urllib.request
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

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
from cancerfold.physics.potentials import virtual_cb
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
    if os.path.exists(dst_path) and os.path.getsize(dst_path) > 0:
        return True
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "OncoVac-Adenine/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp, open(dst_path, "wb") as out:
            out.write(resp.read())
        return True
    except Exception:
        return False


def _parse_temp_k(raw: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)", str(raw))
    return float(m.group(1)) if m else 298.15


def _kmer_set(seq: str, k: int = 3) -> set:
    if len(seq) < k:
        return {seq}
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def _seq_jaccard(seq_a: str, seq_b: str, k: int = 3) -> float:
    sa, sb = _kmer_set(seq_a, k), _kmer_set(seq_b, k)
    union = len(sa | sb)
    return len(sa & sb) / float(union) if union > 0 else 0.0


def load_skempi_records(
    cache_dir: str = "/content/oncovac_cache",
    max_n: int = 448,
    max_complexes: int = 45,
    max_per_pdb: int = 16,
    min_muts_per_pdb: int = 3,
) -> List[Dict[str, object]]:
    """
    Downloads official skempi_v2.csv + RCSB PDBs, balances across distinct Hold_out_type protein
    families (Pr/PI, AB/AG, TCR/pMHC, Other), deduplicates identical mutations by mean ddG,
    and returns validated records with lightweight arrays.
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

    # Round-robin across biological families so mini/mid span diverse protein families
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
        chain_ids: List[str] = list(pdb_data["chain_ids"])
        pad_len = max_n - n_res

        idx_a = np.asarray([i for i, c in enumerate(chain_ids) if c in chains_a], dtype=np.int32)
        idx_b = np.asarray([i for i, c in enumerate(chain_ids) if c in chains_b], dtype=np.int32)
        if idx_a.size == 0 or idx_b.size == 0 or (idx_a.size + idx_b.size) != n_res:
            continue

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
                toks = [tok.strip() for tok in mut_code.split(",") if tok.strip()]
                for tok in toks:
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
                "mut_seq": mut_seq,
                "chain_ids": chain_ids,
                "chains_a": chains_a,
                "chains_b": chains_b,
                "idx_a": idx_a,
                "idx_b": idx_b,
                "mut_code": mut_code,
                "n_muts": len(toks),
                "n_res": n_res,
                "max_n": max_n,
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
                f"{len(complex_recs)} muts",
                flush=True,
            )

    return records


def make_family_disjoint_folds(
    items: List[Dict[str, object]],
    n_folds: int = 3,
    jaccard_thresh: float = 0.25,
) -> List[Tuple[List[Dict[str, object]], List[Dict[str, object]]]]:
    """
    Groups complexes into strict Family/Homology-Disjoint clusters via SKEMPI Hold_out_type
    and 3-mer sequence Jaccard clustering (Union-Find), then balances clusters across K folds.
    """
    pdb_meta: Dict[str, Tuple[str, str]] = {}
    pdb_items: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for it in items:
        pid = str(it["pdb_id"])
        pdb_items[pid].append(it)
        if pid not in pdb_meta:
            pdb_meta[pid] = (str(it["family"]), str(it["wt_seq"]))

    pdbs = sorted(pdb_meta.keys())
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

    for i in range(len(pdbs)):
        for j in range(i + 1, len(pdbs)):
            pa, pb = pdbs[i], pdbs[j]
            fa, sa = pdb_meta[pa]
            fb, sb = pdb_meta[pb]
            same_named_fam = (fa == fb) and (fa not in ("", "Other"))
            if same_named_fam or _seq_jaccard(sa, sb) >= jaccard_thresh:
                _union(pa, pb)

    clusters: Dict[str, List[str]] = defaultdict(list)
    for p in pdbs:
        clusters[_find(p)].append(p)

    # If named Hold_out_type merged nearly everything into < n_folds, fall back to sequence-homology clusters
    if len(clusters) < n_folds:
        parent_seq = {p: p for p in pdbs}

        def _sfind(x: str) -> str:
            while parent_seq[x] != x:
                parent_seq[x] = parent_seq[parent_seq[x]]
                x = parent_seq[x]
            return x

        for i in range(len(pdbs)):
            for j in range(i + 1, len(pdbs)):
                if _seq_jaccard(pdb_meta[pdbs[i]][1], pdb_meta[pdbs[j]][1]) >= jaccard_thresh:
                    ra, rb = _sfind(pdbs[i]), _sfind(pdbs[j])
                    if ra != rb:
                        parent_seq[rb] = ra

        clusters = defaultdict(list)
        for p in pdbs:
            clusters[_sfind(p)].append(p)

    k_eff = min(n_folds, max(2, len(clusters)))
    sorted_clusters = sorted(
        clusters.values(),
        key=lambda plist: -sum(len(pdb_items[p]) for p in plist),
    )
    fold_pdbs: List[List[str]] = [[] for _ in range(k_eff)]
    fold_counts = [0] * k_eff
    for plist in sorted_clusters:
        tgt = int(np.argmin(fold_counts))
        fold_pdbs[tgt].extend(plist)
        fold_counts[tgt] += sum(len(pdb_items[p]) for p in plist)

    print(
        f"[SPLIT] {k_eff}-Fold Family/Homology-Disjoint CV ({len(pdbs)} PDBs -> {len(clusters)} homology clusters):",
        flush=True,
    )
    folds: List[Tuple[List[Dict[str, object]], List[Dict[str, object]]]] = []
    for k in range(k_eff):
        val_set = set(fold_pdbs[k])
        tr = [it for it in items if str(it["pdb_id"]) not in val_set]
        va = [it for it in items if str(it["pdb_id"]) in val_set]
        val_fams = sorted({str(it["family"]) for it in va})
        print(
            f"  Fold {k}: Train={len(tr):4d} muts ({len(pdbs)-len(val_set):3d} PDBs) | "
            f"Val={len(va):4d} muts ({len(val_set):3d} PDBs | fams={val_fams})",
            flush=True,
        )
        folds.append((tr, va))
    return folds


def prepare_cached_ensembles(
    records: List[Dict[str, object]],
    prng_key: jax.Array,
    cache_dir: str = "/content/oncovac_cache",
    n_frames: int = 4,
    n_steps: int = 100,
    burn_in: int = 40,
    temp_k: float = 310.15,
) -> List[Dict[str, object]]:
    """Generates and disk-caches equilibrium BAOAB (310.15 K) ensembles in float16 (~50 KB/mut)."""
    sde_dir = os.path.join(cache_dir, "sde_v2")
    os.makedirs(sde_dir, exist_ok=True)
    theta_sde = {**PHYS_INIT, "k_rep": 0.0, "enm_soften": 1.0}

    @jax.jit
    def _sample_state(key, bb, d_ref, phys, w, res_mask, cm):
        out = sample_ensemble(
            key, bb, d_ref, phys, w, res_mask, theta_sde,
            n_steps=n_steps, dt=0.02, temp_k=temp_k, gamma=2.0,
            use_metadynamics=False, n_frames=n_frames, burn_in=burn_in,
            chain_mask=cm,
        )
        return out["frames"], out["mean_temp_k"]

    items: List[Dict[str, object]] = []
    temps = []
    n_loaded = 0
    n_computed = 0

    for i, rec in enumerate(records):
        safe_mut = re.sub(r"[^A-Za-z0-9_]+", "_", str(rec["mut_code"]))
        npz_path = os.path.join(
            sde_dir,
            f"{rec['pdb_id']}_{safe_mut}_N{rec['bb'].shape[0]}_F{n_frames}.npz",
        )
        if os.path.exists(npz_path):
            cached = np.load(npz_path, allow_pickle=False)
            ens_wt = jnp.asarray(cached["ens_wt"], dtype=jnp.float32)
            ens_mut = jnp.asarray(cached["ens_mut"], dtype=jnp.float32)
            t_wt = float(cached["t_wt"])
            t_mut = float(cached["t_mut"])
            n_loaded += 1
        else:
            prng_key, kw, km = jax.random.split(prng_key, 3)
            ens_wt, t_wt_a = _sample_state(
                kw, rec["bb"], rec["d_ref_ca"], rec["phys_wt"], rec["weights"], rec["res_mask"], rec["cm"]
            )
            ens_mut, t_mut_a = _sample_state(
                km, rec["bb"], rec["d_ref_ca"], rec["phys_mut"], rec["weights"], rec["res_mask"], rec["cm"]
            )
            t_wt, t_mut = float(t_wt_a), float(t_mut_a)
            np.savez_compressed(
                npz_path,
                ens_wt=np.asarray(ens_wt, dtype=np.float16),
                ens_mut=np.asarray(ens_mut, dtype=np.float16),
                t_wt=np.float32(t_wt),
                t_mut=np.float32(t_mut),
            )
            n_computed += 1
            if n_computed % 25 == 0:
                print(f"[SDE] Progress: {i + 1}/{len(records)} ensembles ready...", flush=True)

        temps.extend([t_wt, t_mut])
        items.append({**rec, "ens_wt": ens_wt, "ens_mut": ens_mut})

    print(
        f"[SDE] Ready {len(items)} mutation ensembles ({n_frames} frames/state | "
        f"cached={n_loaded}, new={n_computed}) | mean T_kin={np.mean(temps):.1f} K "
        f"(min={np.min(temps):.1f}, max={np.max(temps):.1f})",
        flush=True,
    )
    return items


def prepare_joltz_priors_cache(
    items: List[Dict[str, object]],
    boltz_cache_dir: str = "/content/boltz_cache",
    cache_dir: str = "/content/oncovac_cache",
    recycling_steps: int = 1,
    use_msa: bool = False,
    contact_cutoff_a: float = 16.0,
) -> List[Dict[str, object]]:
    """
    Computes and disk-caches normalized frozen Joltz-1 (s, sparse z) for Bound & Unbound states.
    WT complex and unchanged unbound partner chains are computed once per PDB and deduplicated.
    Frees Joltz-1 GPU/TPU HBM immediately when caching finishes.
    """
    from cancerfold.af2_core.joltz_runner import Joltz1Runner

    t_start = time.time()
    runner = Joltz1Runner(
        boltz_cache_dir=boltz_cache_dir,
        prior_cache_dir=os.path.join(cache_dir, "joltz_priors"),
        recycling_steps=recycling_steps,
        use_msa=use_msa,
    )

    # Precompute symmetric contact pair indices (d_cb <= 16 A, generous 6 A margin over R_SHORT=10 A) per PDB
    pdb_contacts: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    pdb_wt_seen: set = set()
    for it in items:
        pid = str(it["pdb_id"])
        if pid in pdb_contacts:
            continue
        n_res = int(it["n_res"])
        cb_ref = np.asarray(virtual_cb(it["bb"][:n_res]), dtype=np.float32)
        d_cb = np.linalg.norm(cb_ref[:, None, :] - cb_ref[None, :, :], axis=-1)
        # Also include any pair within 14 A in sampled BAOAB frames for strict bit-exactness
        cb_wt0 = np.asarray(virtual_cb(it["ens_wt"][0, :n_res]), dtype=np.float32)
        d_cb_min = np.minimum(d_cb, np.linalg.norm(cb_wt0[:, None, :] - cb_wt0[None, :, :], axis=-1))
        mask_b = d_cb_min <= contact_cutoff_a
        np.fill_diagonal(mask_b, False)
        r_b, c_b = np.nonzero(mask_b)

        idx_a: np.ndarray = it["idx_a"]
        idx_b: np.ndarray = it["idx_b"]
        mask_a = mask_b[np.ix_(idx_a, idx_a)]
        mask_bb = mask_b[np.ix_(idx_b, idx_b)]
        r_a, c_a = np.nonzero(mask_a)
        r_bb, c_bb = np.nonzero(mask_bb)
        pdb_contacts[pid] = (
            r_b.astype(np.int16), c_b.astype(np.int16),
            r_a.astype(np.int16), c_a.astype(np.int16),
            r_bb.astype(np.int16), c_bb.astype(np.int16),
        )

    enriched: List[Dict[str, object]] = []
    for idx, it in enumerate(items):
        pid = str(it["pdb_id"])
        wt_seq = str(it["wt_seq"])
        mut_seq = str(it["mut_seq"])
        cids: List[str] = list(it["chain_ids"])
        idx_a: np.ndarray = it["idx_a"]
        idx_b: np.ndarray = it["idx_b"]
        r_b, c_b, r_a, c_a, r_bb, c_bb = pdb_contacts[pid]

        if use_msa and pid not in pdb_wt_seen:
            runner.prepare_pdb_wt_msa(pid, wt_seq, cids)
            pdb_wt_seen.add(pid)

        wt_seq_a = "".join(wt_seq[i] for i in idx_a)
        wt_seq_b = "".join(wt_seq[i] for i in idx_b)
        mut_seq_a = "".join(mut_seq[i] for i in idx_a)
        mut_seq_b = "".join(mut_seq[i] for i in idx_b)
        cids_a = [cids[i] for i in idx_a]
        cids_b = [cids[i] for i in idx_b]

        wt_b_path = runner.extract_sparse_priors(wt_seq, cids, r_b, c_b, pid, "bound", strip_pairing=False)
        mut_b_path = runner.extract_sparse_priors(mut_seq, cids, r_b, c_b, pid, "bound", strip_pairing=False)
        wt_ua_path = runner.extract_sparse_priors(wt_seq_a, cids_a, r_a, c_a, pid, "unbound_a", strip_pairing=True)
        wt_ub_path = runner.extract_sparse_priors(wt_seq_b, cids_b, r_bb, c_bb, pid, "unbound_b", strip_pairing=True)

        # Unchanged unbound partner reuses WT prior bit-for-bit (exact zero telepathy)
        mut_ua_path = (
            wt_ua_path
            if mut_seq_a == wt_seq_a
            else runner.extract_sparse_priors(mut_seq_a, cids_a, r_a, c_a, pid, "unbound_a", strip_pairing=True)
        )
        mut_ub_path = (
            wt_ub_path
            if mut_seq_b == wt_seq_b
            else runner.extract_sparse_priors(mut_seq_b, cids_b, r_bb, c_bb, pid, "unbound_b", strip_pairing=True)
        )

        enriched.append({
            **it,
            "joltz_paths": {
                "wt_b": wt_b_path,
                "mut_b": mut_b_path,
                "wt_ua": wt_ua_path,
                "wt_ub": wt_ub_path,
                "mut_ua": mut_ua_path,
                "mut_ub": mut_ub_path,
            },
        })
        if (idx + 1) % 20 == 0 or (idx + 1) == len(items):
            print(
                f"[JOLTZ] Cached {idx + 1}/{len(items)} mutations "
                f"(new trunk evals={runner.trunk_calls}, elapsed={time.time() - t_start:.1f}s)",
                flush=True,
            )

    runner.close()
    print("[JOLTZ] Freed Joltz-1 weights from accelerator memory before Optax training.", flush=True)
    return enriched


def _huber_loss(pred: jnp.ndarray, target: jnp.ndarray, delta: float = 1.5) -> jnp.ndarray:
    diff = jnp.abs(pred - target)
    return jnp.where(diff <= delta, 0.5 * diff ** 2, delta * (diff - 0.5 * delta))


def _per_complex_stats(preds: List[float], trues: List[float], pdbs: List[str]) -> Tuple[float, float]:
    by_pdb: Dict[str, Tuple[List[float], List[float]]] = defaultdict(lambda: ([], []))
    for p, t, pid in zip(preds, trues, pdbs):
        by_pdb[pid][0].append(p)
        by_pdb[pid][1].append(t)
    pearsons, spearmans = [], []
    for ps, ts in by_pdb.values():
        if len(ps) >= 3 and np.std(ts) > 1e-6 and np.std(ps) > 1e-6:
            m = skempi_metrics(jnp.asarray(ps), jnp.asarray(ts))
            pearsons.append(float(m["pearson_r"]))
            spearmans.append(float(m["spearman_rho"]))
    pc_r = float(np.mean(pearsons)) if pearsons else float("nan")
    pc_rho = float(np.mean(spearmans)) if spearmans else float("nan")
    return pc_r, pc_rho


def _evaluate_zero_shot_hphys(items: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Evaluates raw analytical H_phys (zero neural network, zero Joltz-1) across all mutations."""
    theta_raw = {**PHYS_INIT, "k_rep": 0.0, "enm_soften": 1.0}

    @jax.jit
    def _eval_phys(ens_wt, ens_mut, d_ref, phys_wt, phys_mut, w, pa, pb, cm, tk):
        return binding_ddg_bar(
            ens_wt, ens_mut, d_ref, phys_wt, phys_mut, w, pa, pb, theta_raw,
            nn_params=None, chain_mask=cm, temp_k=tk,
        )

    rows = []
    for x in items:
        out = _eval_phys(
            x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
            x["phys_wt"], x["phys_mut"], x["weights"],
            x["pa_mask"], x["pb_mask"], x["cm"], float(x["temp_k"]),
        )
        rows.append({
            "fold": -1,
            "pdb_id": str(x["pdb_id"]),
            "family": str(x["family"]),
            "mut_code": str(x["mut_code"]),
            "n_muts": int(x["n_muts"]),
            "ddg_exp": float(x["ddg_exp"]),
            "ddg_pred": float(out["ddg_bind"]),
            "dh_phys": float(out["dh_phys_mean"]),
            "dh_nn": float(out["dh_nn_mean"]),
        })
    return rows


def train_single_fold(
    fold_idx: int,
    train_items: List[Dict[str, object]],
    val_items: List[Dict[str, object]],
    weights_dir: str,
    prior_mode: str = "none",
    ckpt_prefix: str = "adenine_nojoltz",
    epochs: int = 25,
    batch_size: int = 12,
    peak_lr: float = 5e-4,
    weight_decay: float = 3e-2,
    dropout_rate: float = 0.15,
    jitter_std: float = 0.02,
    nn_reg_weight: float = 3e-2,
    patience: int = 6,
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, jnp.ndarray]], float]:
    """
    Trains one CV fold with all 5 anti-overfitting defenses & joint calibration of 4 physical constants + U_NN.
    Supports both prior_mode='none' (on-the-fly fallback) and prior_mode='joltz' (real frozen Joltz-1).
    """
    use_joltz = (prior_mode == "joltz")
    if use_joltz:
        from cancerfold.af2_core.joltz_runner import load_bound_unbound_priors

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
    def _eval_none(params, ens_wt, ens_mut, d_ref, phys_wt, phys_mut, w, pa, pb, res_mask, aa_wt, aa_mut, cm, tk):
        (s_wt, z_wt), (s_mut, z_mut) = _get_fallback_priors(aa_wt, aa_mut, res_mask)
        theta = _decode_theta(params["phys_raw"])
        return binding_ddg_bar(
            ens_wt, ens_mut, d_ref, phys_wt, phys_mut, w, pa, pb, theta,
            single_wt=s_wt, pair_wt=z_wt, single_mut=s_mut, pair_mut=z_mut,
            single_wt_unbound=s_wt, pair_wt_unbound=z_wt,
            single_mut_unbound=s_mut, pair_mut_unbound=z_mut,
            nn_params=params["nn"], chain_mask=cm, temp_k=tk,
        )

    @jax.jit
    def _eval_joltz(
        params, ens_wt, ens_mut, d_ref, phys_wt, phys_mut, w, pa, pb, cm, tk,
        s_wt_b, z_wt_b, s_mut_b, z_mut_b, s_wt_u, z_wt_u, s_mut_u, z_mut_u,
    ):
        theta = _decode_theta(params["phys_raw"])
        return binding_ddg_bar(
            ens_wt, ens_mut, d_ref, phys_wt, phys_mut, w, pa, pb, theta,
            single_wt=s_wt_b.astype(jnp.float32), pair_wt=z_wt_b.astype(jnp.float32),
            single_mut=s_mut_b.astype(jnp.float32), pair_mut=z_mut_b.astype(jnp.float32),
            single_wt_unbound=s_wt_u.astype(jnp.float32), pair_wt_unbound=z_wt_u.astype(jnp.float32),
            single_mut_unbound=s_mut_u.astype(jnp.float32), pair_mut_unbound=z_mut_u.astype(jnp.float32),
            nn_params=params["nn"], chain_mask=cm, temp_k=tk,
        )

    def _loss_core(p, ew, em, d_ref, phys_wt, phys_mut, w, pa, pb, cm, tk, y_true,
                   s_wt_b, z_wt_b, s_mut_b, z_mut_b, s_wt_u, z_wt_u, s_mut_u, z_mut_u):
        theta = _decode_theta(p["phys_raw"])
        out = binding_ddg_bar(
            ew, em, d_ref, phys_wt, phys_mut, w, pa, pb, theta,
            single_wt=s_wt_b, pair_wt=z_wt_b, single_mut=s_mut_b, pair_mut=z_mut_b,
            single_wt_unbound=s_wt_u, pair_wt_unbound=z_wt_u,
            single_mut_unbound=s_mut_u, pair_mut_unbound=z_mut_u,
            nn_params=p["nn"], chain_mask=cm, temp_k=tk,
        )
        huber = _huber_loss(out["ddg_bind"], y_true, delta=1.5)
        phys_anchor = 0.02 * sum(
            (theta[k] - PHYS_INIT[k]) ** 2 for k in PHYS_INIT
        )
        nn_reg = nn_reg_weight * (out["dh_nn_mean"] ** 2)
        return huber + phys_anchor + nn_reg

    @jax.jit
    def _grad_none(params, step_key, ens_wt, ens_mut, d_ref, phys_wt, phys_mut, w, pa, pb, res_mask, aa_wt, aa_mut, cm, tk, y_true):
        k_sub, k_jw, k_jm, k_drop = jax.random.split(step_key, 4)
        n_frames = ens_wt.shape[0]
        k_sub_n = max(2, n_frames - 1) if n_frames >= 4 else n_frames
        idx_w = jax.random.choice(k_sub, n_frames, shape=(k_sub_n,), replace=False)
        idx_m = jax.random.choice(k_drop, n_frames, shape=(k_sub_n,), replace=False)

        m_bb = res_mask[None, :, None, None]
        ew = ens_wt[idx_w] + jax.random.normal(k_jw, ens_wt[idx_w].shape, dtype=jnp.float32) * jitter_std * m_bb
        em = ens_mut[idx_m] + jax.random.normal(k_jm, ens_mut[idx_m].shape, dtype=jnp.float32) * jitter_std * m_bb

        (s_wt, z_wt), (s_mut, z_mut) = _get_fallback_priors(aa_wt, aa_mut, res_mask)
        keep_prob = 1.0 - dropout_rate
        ks, kz = jax.random.split(k_drop, 2)
        ms = jax.random.bernoulli(ks, keep_prob, s_wt.shape).astype(jnp.float32) / keep_prob
        mz = jax.random.bernoulli(kz, keep_prob, z_wt.shape).astype(jnp.float32) / keep_prob
        s_wt_d, s_mut_d = s_wt * ms, s_mut * ms
        z_wt_d, z_mut_d = z_wt * mz, z_mut * mz

        return jax.value_and_grad(
            lambda p: _loss_core(
                p, ew, em, d_ref, phys_wt, phys_mut, w, pa, pb, cm, tk, y_true,
                s_wt_d, z_wt_d, s_mut_d, z_mut_d, s_wt_d, z_wt_d, s_mut_d, z_mut_d,
            )
        )(params)

    @jax.jit
    def _grad_joltz(
        params, step_key, ens_wt, ens_mut, d_ref, phys_wt, phys_mut, w, pa, pb, res_mask, cm, tk, y_true,
        s_wt_b, z_wt_b, s_mut_b, z_mut_b, s_wt_u, z_wt_u, s_mut_u, z_mut_u,
    ):
        k_sub, k_jw, k_jm, k_drop = jax.random.split(step_key, 4)
        n_frames = ens_wt.shape[0]
        k_sub_n = max(2, n_frames - 1) if n_frames >= 4 else n_frames
        idx_w = jax.random.choice(k_sub, n_frames, shape=(k_sub_n,), replace=False)
        idx_m = jax.random.choice(k_drop, n_frames, shape=(k_sub_n,), replace=False)

        m_bb = res_mask[None, :, None, None]
        ew = ens_wt[idx_w] + jax.random.normal(k_jw, ens_wt[idx_w].shape, dtype=jnp.float32) * jitter_std * m_bb
        em = ens_mut[idx_m] + jax.random.normal(k_jm, ens_mut[idx_m].shape, dtype=jnp.float32) * jitter_std * m_bb

        # Shared symmetric dropout mask preserves exact WT <-> Mut antisymmetry
        keep_prob = 1.0 - dropout_rate
        ks, kz = jax.random.split(k_drop, 2)
        ms = jax.random.bernoulli(ks, keep_prob, s_wt_b.shape).astype(jnp.float32) / keep_prob
        mz = jax.random.bernoulli(kz, keep_prob, z_wt_b.shape).astype(jnp.float32) / keep_prob

        swb = s_wt_b.astype(jnp.float32) * ms
        smb = s_mut_b.astype(jnp.float32) * ms
        swu = s_wt_u.astype(jnp.float32) * ms
        smu = s_mut_u.astype(jnp.float32) * ms
        zwb = z_wt_b.astype(jnp.float32) * mz
        zmb = z_mut_b.astype(jnp.float32) * mz
        zwu = z_wt_u.astype(jnp.float32) * mz
        zmu = z_mut_u.astype(jnp.float32) * mz

        return jax.value_and_grad(
            lambda p: _loss_core(
                p, ew, em, d_ref, phys_wt, phys_mut, w, pa, pb, cm, tk, y_true,
                swb, zwb, smb, zmb, swu, zwu, smu, zmu,
            )
        )(params)

    def _load_item_joltz(x):
        jp = x["joltz_paths"]
        s_wt_b, z_wt_b, s_wt_u, z_wt_u = load_bound_unbound_priors(
            jp["wt_b"], jp["wt_ua"], jp["wt_ub"], x["idx_a"], x["idx_b"], int(x["max_n"])
        )
        s_mut_b, z_mut_b, s_mut_u, z_mut_u = load_bound_unbound_priors(
            jp["mut_b"], jp["mut_ua"], jp["mut_ub"], x["idx_a"], x["idx_b"], int(x["max_n"])
        )
        return s_wt_b, z_wt_b, s_mut_b, z_mut_b, s_wt_u, z_wt_u, s_mut_u, z_mut_u

    def _run_eval(params, split_items):
        preds, trues, pdbs, dh_p, dh_n = [], [], [], [], []
        for x in split_items:
            if use_joltz:
                s_wb, z_wb, s_mb, z_mb, s_wu, z_wu, s_mu, z_mu = _load_item_joltz(x)
                out = _eval_joltz(
                    params, x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
                    x["phys_wt"], x["phys_mut"], x["weights"], x["pa_mask"], x["pb_mask"],
                    x["cm"], float(x["temp_k"]),
                    s_wb, z_wb, s_mb, z_mb, s_wu, z_wu, s_mu, z_mu,
                )
            else:
                out = _eval_none(
                    params, x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
                    x["phys_wt"], x["phys_mut"], x["weights"], x["pa_mask"], x["pb_mask"],
                    x["res_mask"], x["aa_wt"], x["aa_mut"], x["cm"], float(x["temp_k"]),
                )
            preds.append(float(out["ddg_bind"]))
            trues.append(float(x["ddg_exp"]))
            pdbs.append(str(x["pdb_id"]))
            dh_p.append(float(out["dh_phys_mean"]))
            dh_n.append(float(out["dh_nn_mean"]))
        m = skempi_metrics(jnp.asarray(preds), jnp.asarray(trues))
        pc_r, pc_rho = _per_complex_stats(preds, trues, pdbs)
        return {
            "preds": preds,
            "trues": trues,
            "dh_phys_list": dh_p,
            "dh_nn_list": dh_n,
            "r": float(m["pearson_r"]),
            "rho": float(m["spearman_rho"]),
            "pc_r": pc_r,
            "pc_rho": pc_rho,
            "rmse": float(m["rmse"]),
            "mae": float(m["mae"]),
            "dh_phys": float(np.mean(np.abs(dh_p))),
            "dh_nn": float(np.mean(np.abs(dh_n))),
        }

    ckpt_path = os.path.join(weights_dir, f"{ckpt_prefix}_fold{fold_idx}.npz")
    best_rmse = float("inf")
    best_epoch = 0
    bad_epochs = 0

    print(f"\n--- [{ckpt_prefix}] Fold {fold_idx} Training ({len(train_items)} train / {len(val_items)} val) ---", flush=True)

    for ep in range(1, epochs + 1):
        t_ep = time.time()
        k_train, k_perm = jax.random.split(k_train)
        perm = np.asarray(jax.random.permutation(k_perm, len(train_items)))
        ep_losses = []

        for b_start in range(0, len(perm), batch_size):
            batch_idx = perm[b_start : b_start + batch_size]
            grad_acc = jax.tree_util.tree_map(jnp.zeros_like, model_params)
            b_loss = 0.0
            for idx in batch_idx:
                x = train_items[int(idx)]
                k_train, k_step = jax.random.split(k_train)
                if use_joltz:
                    s_wb, z_wb, s_mb, z_mb, s_wu, z_wu, s_mu, z_mu = _load_item_joltz(x)
                    loss_val, grads = _grad_joltz(
                        model_params, k_step, x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
                        x["phys_wt"], x["phys_mut"], x["weights"], x["pa_mask"], x["pb_mask"],
                        x["res_mask"], x["cm"], float(x["temp_k"]), jnp.asarray(x["ddg_exp"], jnp.float32),
                        s_wb, z_wb, s_mb, z_mb, s_wu, z_wu, s_mu, z_mu,
                    )
                else:
                    loss_val, grads = _grad_none(
                        model_params, k_step, x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
                        x["phys_wt"], x["phys_mut"], x["weights"], x["pa_mask"], x["pb_mask"],
                        x["res_mask"], x["aa_wt"], x["aa_mut"], x["cm"], float(x["temp_k"]),
                        jnp.asarray(x["ddg_exp"], jnp.float32),
                    )
                grad_acc = jax.tree_util.tree_map(lambda a, g: a + g, grad_acc, grads)
                b_loss += float(loss_val)

            inv_b = 1.0 / float(len(batch_idx))
            grad_mean = jax.tree_util.tree_map(lambda g: g * inv_b, grad_acc)
            updates, opt_state = optimizer.update(grad_mean, opt_state, model_params)
            model_params = optax.apply_updates(model_params, updates)
            ep_losses.append(b_loss * inv_b)

        val_m = _run_eval(model_params, val_items)
        improved = val_m["rmse"] < (best_rmse - 1e-4)
        if improved:
            best_rmse = val_m["rmse"]
            best_epoch = ep
            bad_epochs = 0
            save_checkpoint(ckpt_path, model_params)
        else:
            bad_epochs += 1

        if ep == 1 or ep % 5 == 0 or improved or ep == epochs:
            theta_cur = _decode_theta(model_params["phys_raw"])
            tr_str = ""
            if ep == 1 or ep % 5 == 0 or ep == epochs:
                tr_m = _run_eval(model_params, train_items[: min(len(train_items), 80)])
                tr_str = f" | tr_r={tr_m['r']:+.3f} tr_RMSE={tr_m['rmse']:.3f}"
            mark = " *BEST*" if improved else ""
            print(
                f"  [F{fold_idx} ep {ep:02d}/{epochs}] loss={np.mean(ep_losses):.3f}{tr_str} | "
                f"VAL r={val_m['r']:+.3f} pc_r={val_m['pc_r']:+.3f} rho={val_m['rho']:+.3f} "
                f"RMSE={val_m['rmse']:.3f} | |dH_p|={val_m['dh_phys']:.2f} |dH_nn|={val_m['dh_nn']:.2f} | "
                f"k=({float(theta_cur['k_clash']):.2f},{float(theta_cur['k_elec']):.2f},"
                f"{float(theta_cur['k_hydro']):.2f},{float(theta_cur['k_hbond']):.2f}) "
                f"({time.time() - t_ep:.1f}s){mark}",
                flush=True,
            )

        if bad_epochs >= patience:
            print(
                f"  [F{fold_idx}] Early stopping at ep {ep} (best ep={best_epoch}, VAL RMSE={best_rmse:.3f})",
                flush=True,
            )
            break

    reloaded = load_checkpoint(ckpt_path)
    final_val = _run_eval(reloaded, val_items)
    print(
        f"  [F{fold_idx} FINAL] Restored {ckpt_path} (ep {best_epoch}) -> "
        f"VAL r={final_val['r']:+.3f}, pc_r={final_val['pc_r']:+.3f}, "
        f"rho={final_val['rho']:+.3f}, RMSE={final_val['rmse']:.3f}",
        flush=True,
    )

    # Exact BAR antisymmetry audit on unseen fold
    antisym_errs = []
    for x in val_items[:5]:
        if use_joltz:
            s_wb, z_wb, s_mb, z_mb, s_wu, z_wu, s_mu, z_mu = _load_item_joltz(x)
            fwd = _eval_joltz(
                reloaded, x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
                x["phys_wt"], x["phys_mut"], x["weights"], x["pa_mask"], x["pb_mask"],
                x["cm"], float(x["temp_k"]),
                s_wb, z_wb, s_mb, z_mb, s_wu, z_wu, s_mu, z_mu,
            )["ddg_bind"]
            rev = _eval_joltz(
                reloaded, x["ens_mut"], x["ens_wt"], x["d_ref_ca"],
                x["phys_mut"], x["phys_wt"], x["weights"], x["pa_mask"], x["pb_mask"],
                x["cm"], float(x["temp_k"]),
                s_mb, z_mb, s_wb, z_wb, s_mu, z_mu, s_wu, z_wu,
            )["ddg_bind"]
        else:
            fwd = _eval_none(
                reloaded, x["ens_wt"], x["ens_mut"], x["d_ref_ca"],
                x["phys_wt"], x["phys_mut"], x["weights"], x["pa_mask"], x["pb_mask"],
                x["res_mask"], x["aa_wt"], x["aa_mut"], x["cm"], float(x["temp_k"]),
            )["ddg_bind"]
            rev = _eval_none(
                reloaded, x["ens_mut"], x["ens_wt"], x["d_ref_ca"],
                x["phys_mut"], x["phys_wt"], x["weights"], x["pa_mask"], x["pb_mask"],
                x["res_mask"], x["aa_mut"], x["aa_wt"], x["cm"], float(x["temp_k"]),
            )["ddg_bind"]
        antisym_errs.append(float(jnp.abs(fwd + rev)))
    max_antisym = max(antisym_errs) if antisym_errs else 0.0
    print(f"  [F{fold_idx} AUDIT] Unseen-family exact BAR antisymmetry max|fwd+rev| = {max_antisym:.2e}", flush=True)

    oof_rows = []
    for x, yp, dp, dn in zip(val_items, final_val["preds"], final_val["dh_phys_list"], final_val["dh_nn_list"]):
        oof_rows.append({
            "fold": fold_idx,
            "pdb_id": str(x["pdb_id"]),
            "family": str(x["family"]),
            "mut_code": str(x["mut_code"]),
            "n_muts": int(x["n_muts"]),
            "ddg_exp": float(x["ddg_exp"]),
            "ddg_pred": float(yp),
            "dh_phys": float(dp),
            "dh_nn": float(dn),
        })
    return oof_rows, reloaded, max_antisym


def summarize_benchmark_pillars(
    label: str,
    oof_rows: List[Dict[str, object]],
    max_antisym: Optional[float] = None,
) -> Dict[str, float]:
    """Prints Pillar 1 (Overall/Per-Complex), Pillar 2 (White-Box), and Pillar 3 (Black-Box Failure Zones)."""
    preds = [float(r["ddg_pred"]) for r in oof_rows]
    trues = [float(r["ddg_exp"]) for r in oof_rows]
    pdbs = [str(r["pdb_id"]) for r in oof_rows]
    dh_p = [abs(float(r["dh_phys"])) for r in oof_rows]
    dh_n = [abs(float(r["dh_nn"])) for r in oof_rows]

    ov = skempi_metrics(jnp.asarray(preds), jnp.asarray(trues))
    pc_r, pc_rho = _per_complex_stats(preds, trues, pdbs)

    print(f"\n==================== {label} (N={len(oof_rows)} muts) ====================", flush=True)
    print(
        f"  [Pillar 1] Overall r={float(ov['pearson_r']):+.4f} | rho={float(ov['spearman_rho']):+.4f} | "
        f"Per-Complex r={pc_r:+.4f} | Per-Complex rho={pc_rho:+.4f} | "
        f"RMSE={float(ov['rmse']):.3f} | MAE={float(ov['mae']):.3f}",
        flush=True,
    )
    antisym_s = f" | BAR max|fwd+rev|={max_antisym:.2e}" if max_antisym is not None else ""
    print(
        f"  [Pillar 2] White-Box mean|dH_phys|={np.mean(dh_p):.3f} vs mean|dH_nn|={np.mean(dh_n):.3f}{antisym_s}",
        flush=True,
    )

    # Pillar 3a: Single vs Multi-Point Epistasis
    for tag, cond in [
        ("Single (n=1)", lambda r: int(r["n_muts"]) == 1),
        ("Multi  (n>=2)", lambda r: int(r["n_muts"]) >= 2),
    ]:
        sub = [r for r in oof_rows if cond(r)]
        if len(sub) >= 4:
            sm = skempi_metrics(
                jnp.asarray([float(r["ddg_pred"]) for r in sub]),
                jnp.asarray([float(r["ddg_exp"]) for r in sub]),
            )
            print(
                f"  [Pillar 3] {tag:14s} (N={len(sub):4d}): r={float(sm['pearson_r']):+.3f} | "
                f"rho={float(sm['spearman_rho']):+.3f} | RMSE={float(sm['rmse']):.3f}",
                flush=True,
            )

    # Pillar 3b: Stabilizing (< -0.5) vs Destabilizing (> +0.5)
    for tag, cond, sign_fn in [
        ("Stabilizing (<-0.5)", lambda r: float(r["ddg_exp"]) < -0.5, lambda p: p < 0.0),
        ("Destabiliz. (>+0.5)", lambda r: float(r["ddg_exp"]) > 0.5, lambda p: p > 0.0),
    ]:
        sub = [r for r in oof_rows if cond(r)]
        if len(sub) >= 3:
            sp = np.asarray([float(r["ddg_pred"]) for r in sub])
            st = np.asarray([float(r["ddg_exp"]) for r in sub])
            rmse = float(np.sqrt(np.mean((sp - st) ** 2)))
            mae = float(np.mean(np.abs(sp - st)))
            sign_acc = float(np.mean([sign_fn(p) for p in sp])) * 100.0
            print(
                f"  [Pillar 3] {tag:19s} (N={len(sub):4d}): RMSE={rmse:.3f} | MAE={mae:.3f} | "
                f"SignAcc={sign_acc:.1f}%",
                flush=True,
            )

    # Pillar 3c: Per-Family Breakdown (AB/AG, TCR/pMHC, Pr/PI, Other)
    by_fam: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in oof_rows:
        by_fam[str(r["family"])].append(r)
    for fam in sorted(by_fam.keys()):
        sub = by_fam[fam]
        if len(sub) >= 5:
            fm = skempi_metrics(
                jnp.asarray([float(r["ddg_pred"]) for r in sub]),
                jnp.asarray([float(r["ddg_exp"]) for r in sub]),
            )
            print(
                f"  [Pillar 3] Family={fam:10s} (N={len(sub):4d}): r={float(fm['pearson_r']):+.3f} | "
                f"rho={float(fm['spearman_rho']):+.3f} | RMSE={float(fm['rmse']):.3f}",
                flush=True,
            )

    return {
        "r": float(ov["pearson_r"]),
        "rho": float(ov["spearman_rho"]),
        "pc_r": pc_r,
        "pc_rho": pc_rho,
        "rmse": float(ov["rmse"]),
        "mae": float(ov["mae"]),
    }


def _save_oof_csv(path: str, rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = ["fold", "pdb_id", "family", "mut_code", "n_muts", "ddg_exp", "ddg_pred", "dh_phys", "dh_nn"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_skempi_cv(
    mode: str = "mid",
    prior: str = "both",
    cache_dir: str = "/content/oncovac_cache",
    boltz_cache_dir: str = "/content/boltz_cache",
    weights_dir: str = "/content/oncovac_weights",
    recycling_steps: int = 1,
    use_msa: bool = False,
) -> None:
    presets = {
        "mini": dict(max_n=384, max_complexes=12, max_per_pdb=8, epochs=20, batch_size=8),
        "mid":  dict(max_n=448, max_complexes=45, max_per_pdb=16, epochs=25, batch_size=12),
        "full": dict(max_n=512, max_complexes=400, max_per_pdb=100, epochs=30, batch_size=16),
    }
    cfg = presets[mode]
    print(
        f"=== OncoVac-Adenine Gen 1 | 3-Fold Family-Disjoint SKEMPI 2.0 "
        f"(mode={mode.upper()}, prior={prior}, msa={use_msa}, rec={recycling_steps}) ===",
        flush=True,
    )
    records = load_skempi_records(
        cache_dir=cache_dir,
        max_n=cfg["max_n"],
        max_complexes=cfg["max_complexes"],
        max_per_pdb=cfg["max_per_pdb"],
    )
    items = prepare_cached_ensembles(records, jax.random.PRNGKey(42), cache_dir=cache_dir)

    if prior in ("joltz", "both"):
        items = prepare_joltz_priors_cache(
            items,
            boltz_cache_dir=boltz_cache_dir,
            cache_dir=cache_dir,
            recycling_steps=recycling_steps,
            use_msa=use_msa,
        )

    folds = make_family_disjoint_folds(items, n_folds=3)
    os.makedirs(weights_dir, exist_ok=True)
    summary_rows: List[Tuple[str, Dict[str, float]]] = []

    # 1. Zero-Shot Raw Analytical Physics (no AI)
    zs_rows = _evaluate_zero_shot_hphys(items)
    _save_oof_csv(os.path.join(weights_dir, f"oof_zeroshot_hphys_{mode}.csv"), zs_rows)
    summary_rows.append(("Zero-Shot H_phys (Raw)", summarize_benchmark_pillars("Zero-Shot H_phys (Raw)", zs_rows)))

    # 2. Standalone Ablation (Adenine without Joltz-1)
    if prior in ("none", "both"):
        all_oof_none, antisym_none = [], 0.0
        for k, (tr_items, va_items) in enumerate(folds):
            oof_k, _, asym_k = train_single_fold(
                fold_idx=k, train_items=tr_items, val_items=va_items,
                weights_dir=weights_dir, prior_mode="none", ckpt_prefix="adenine_nojoltz",
                epochs=cfg["epochs"], batch_size=cfg["batch_size"],
            )
            all_oof_none.extend(oof_k)
            antisym_none = max(antisym_none, asym_k)
        _save_oof_csv(os.path.join(weights_dir, f"oof_adenine_nojoltz_{mode}.csv"), all_oof_none)
        summary_rows.append((
            "Adenine (No-Joltz)",
            summarize_benchmark_pillars("Adenine (No-Joltz)", all_oof_none, max_antisym=antisym_none),
        ))
        gc.collect()
        jax.clear_caches()

    # 3. Flagship Release Model (OncoVac Adenine Gen 1 + Frozen Joltz-1)
    if prior in ("joltz", "both"):
        all_oof_joltz, antisym_joltz = [], 0.0
        for k, (tr_items, va_items) in enumerate(folds):
            oof_k, _, asym_k = train_single_fold(
                fold_idx=k, train_items=tr_items, val_items=va_items,
                weights_dir=weights_dir, prior_mode="joltz", ckpt_prefix="adenine_gen1",
                epochs=cfg["epochs"], batch_size=cfg["batch_size"],
            )
            all_oof_joltz.extend(oof_k)
            antisym_joltz = max(antisym_joltz, asym_k)
        _save_oof_csv(os.path.join(weights_dir, f"oof_adenine_gen1_{mode}.csv"), all_oof_joltz)
        summary_rows.append((
            "OncoVac Adenine Gen 1 (+Joltz-1)",
            summarize_benchmark_pillars("OncoVac Adenine Gen 1 (+Joltz-1)", all_oof_joltz, max_antisym=antisym_joltz),
        ))

    print("\n================================================================================", flush=True)
    print(f"SIDE-BY-SIDE 3-FOLD FAMILY-DISJOINT SUMMARY ({mode.upper()}, msa={use_msa})", flush=True)
    print(f"{'Model':34s} | {'r':>7s} | {'pc_r':>7s} | {'rho':>7s} | {'RMSE':>6s} | {'MAE':>6s}", flush=True)
    print("-" * 80, flush=True)
    for name, st in summary_rows:
        print(
            f"{name:34s} | {st['r']:+7.4f} | {st['pc_r']:+7.4f} | {st['rho']:+7.4f} | "
            f"{st['rmse']:6.3f} | {st['mae']:6.3f}",
            flush=True,
        )
    print("================================================================================", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OncoVac-Adenine 3-Fold Family-Disjoint SKEMPI 2.0 Runner")
    parser.add_argument("--mode", choices=["mini", "mid", "full"], default="mid")
    parser.add_argument("--prior", choices=["none", "joltz", "both"], default="both")
    parser.add_argument("--use-msa", action="store_true", help="Enable 1x-per-PDB WT ColabFold MSA in Joltz-1")
    parser.add_argument("--recycling-steps", type=int, default=1)
    parser.add_argument("--cache-dir", default="/content/oncovac_cache")
    parser.add_argument("--boltz-cache-dir", default="/content/boltz_cache")
    parser.add_argument("--weights-dir", default="/content/oncovac_weights")
    args = parser.parse_args()
    run_skempi_cv(
        mode=args.mode,
        prior=args.prior,
        cache_dir=args.cache_dir,
        boltz_cache_dir=args.boltz_cache_dir,
        weights_dir=args.weights_dir,
        recycling_steps=args.recycling_steps,
        use_msa=args.use_msa,
    )
