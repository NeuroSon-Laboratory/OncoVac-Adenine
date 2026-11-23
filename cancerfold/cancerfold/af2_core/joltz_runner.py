"""Frozen Joltz-1 (Boltz-1 JAX) Pairformer runner with LayerNorm scaling, XLA bucketing, and sparse contact cache."""
import contextlib
from dataclasses import asdict
import gc
import hashlib
import io
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tempfile
import time
import types
from typing import Dict, List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np

STD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def install_fairscale_shim() -> None:
    """Provides a lightweight inference-only shim for fairscale on Python 3.13 runtimes without wheels."""
    if "fairscale" in sys.modules:
        return
    try:
        import fairscale  # noqa: F401
    except ImportError:
        for mod_name in (
            "fairscale",
            "fairscale.nn",
            "fairscale.nn.checkpoint",
            "fairscale.nn.checkpoint.checkpoint_activations",
        ):
            sys.modules[mod_name] = types.ModuleType(mod_name)
        sys.modules["fairscale.nn.checkpoint.checkpoint_activations"].checkpoint_wrapper = (
            lambda module, *args, **kwargs: module
        )


def ensure_protein_ccd(boltz_cache_dir: str) -> Path:
    """
    Downloads Boltz-1 weights/CCD if missing and builds a slim canonical-residue ccd_protein_v2.pkl
    preserving ALL RDKit molecule/atom/conformer properties (atom 'name', conformer 'Ideal', etc.)
    inside an isolated subprocess so the ~6 GB full-CCD RAM spike is returned to the OS immediately.
    """
    cache = Path(boltz_cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ckpt_path = cache / "boltz1_conf.ckpt"
    slim_path = cache / "ccd_protein_v2.pkl"
    if ckpt_path.exists() and slim_path.exists():
        return slim_path

    helper = """
import pickle, sys, types
from pathlib import Path
from rdkit import Chem
Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

try:
    import fairscale
except ImportError:
    for m in ("fairscale", "fairscale.nn", "fairscale.nn.checkpoint", "fairscale.nn.checkpoint.checkpoint_activations"):
        sys.modules[m] = types.ModuleType(m)
    sys.modules["fairscale.nn.checkpoint.checkpoint_activations"].checkpoint_wrapper = lambda mod, *a, **kw: mod

from boltz.main import download_boltz1
from boltz.data import const

cache = Path(sys.argv[1])
cache.mkdir(parents=True, exist_ok=True)
download_boltz1(cache)
dst = cache / "ccd_protein_v2.pkl"
if not dst.exists():
    print("[JOLTZ] Filtering ccd.pkl -> ccd_protein_v2.pkl with RDKit AllProps (saving ~6 GB RAM)...", flush=True)
    with (cache / "ccd.pkl").open("rb") as f:
        full_ccd = pickle.load(f)
    keep = set(const.tokens) | set(const.canonical_tokens)
    slim = {k: v for k, v in full_ccd.items() if k in keep}
    with dst.open("wb") as f:
        pickle.dump(slim, f)
    with dst.open("rb") as f:
        check = pickle.load(f)
    assert all(a.HasProp("name") for a in check["ALA"].GetAtoms()), "RDKit atom 'name' prop missing!"
    print(f"[JOLTZ] Saved {len(slim)} canonical tokens to {dst} ({dst.stat().st_size / 1024:.1f} KB, props OK)", flush=True)
"""
    subprocess.run([sys.executable, "-c", helper, str(cache)], check=True)
    return slim_path


def _clean_seq(seq: str) -> str:
    return "".join(ch if ch in STD_AA else "X" for ch in seq.upper())


def _group_chains(seq: str, chain_ids: Sequence[str]) -> Tuple[List[Tuple[str, List[int], str]], np.ndarray]:
    """Groups residues by chain ID in first-appearance order and returns permutation back to original indices."""
    order: List[str] = []
    idx_by_chain: Dict[str, List[int]] = {}
    for idx, cid in enumerate(chain_ids):
        if cid not in idx_by_chain:
            order.append(cid)
            idx_by_chain[cid] = []
        idx_by_chain[cid].append(idx)

    groups: List[Tuple[str, List[int], str]] = []
    perm_list: List[int] = []
    for cid in order:
        idxs = idx_by_chain[cid]
        subseq = _clean_seq("".join(seq[i] for i in idxs))
        groups.append((cid, idxs, subseq))
        perm_list.extend(idxs)
    return groups, np.asarray(perm_list, dtype=np.int32)


def _ascii_chain_name(idx: int) -> str:
    if idx < 26:
        return chr(65 + idx)
    return f"C{idx}"


def _to_numpy_dict(features_dict: Dict[str, object]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for k, v in features_dict.items():
        if k == "record":
            continue
        if hasattr(v, "detach"):
            out[k] = v.detach().cpu().numpy()
        else:
            out[k] = np.asarray(v)
    return out


class Joltz1Runner:
    """
    Runs frozen Joltz-1 Pairformer trunk with Boltz-1's canonical LayerNorms (s_norm, z_norm),
    static XLA length buckets, optional 1x-per-PDB WT MSA reuse, and bit-exact sparse contact caching.
    """

    def __init__(
        self,
        boltz_cache_dir: str = "/content/boltz_cache",
        prior_cache_dir: str = "/content/oncovac_cache/joltz_priors",
        recycling_steps: int = 0,
        use_msa: bool = False,
        max_msa_seqs: int = 64,
        buckets: Tuple[int, ...] = (128, 256, 320, 384, 448, 512),
    ) -> None:
        install_fairscale_shim()
        jax.config.update("jax_default_matmul_precision", "float32")

        from rdkit import Chem
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

        self.boltz_cache_dir = Path(boltz_cache_dir)
        self.prior_cache_dir = Path(prior_cache_dir)
        self.prior_cache_dir.mkdir(parents=True, exist_ok=True)
        self.msa_cache_dir = self.prior_cache_dir / "wt_msas"
        self.msa_cache_dir.mkdir(parents=True, exist_ok=True)
        self.recycling_steps = int(recycling_steps)
        self.use_msa = bool(use_msa)
        self.max_msa_seqs = int(max_msa_seqs) if use_msa else 1
        self.buckets = tuple(sorted(buckets))
        self.trunk_calls = 0

        slim_ccd_path = ensure_protein_ccd(str(self.boltz_cache_dir))
        with slim_ccd_path.open("rb") as f:
            self.ccd = pickle.load(f)
        assert all(a.HasProp("name") for a in self.ccd["ALA"].GetAtoms()), "ccd_protein_v2.pkl missing atom props!"

        import equinox as eqx
        import torch
        import joltz
        from boltz.data.feature.featurizer import BoltzFeaturizer
        from boltz.data.module.inference import collate
        from boltz.data.parse.csv import parse_csv
        from boltz.data.parse.schema import parse_boltz_schema
        from boltz.data.tokenize.boltz import BoltzTokenizer
        from boltz.data.types import Input
        from boltz.main import BoltzDiffusionParams, compute_msa
        from boltz.model.models.boltz1 import Boltz1

        self._collate = collate
        self._parse_csv = parse_csv
        self._parse_boltz_schema = parse_boltz_schema
        self._Input = Input
        self._compute_msa = compute_msa
        self._tokenizer = BoltzTokenizer()
        self._featurizer = BoltzFeaturizer()

        t0 = time.time()
        print("[JOLTZ] Loading Boltz-1 checkpoint (CPU) and converting to JAX (Joltz-1)...", flush=True)
        orig_load = torch.load
        torch.load = lambda *a, **kw: orig_load(*a, **{**kw, "weights_only": False})
        try:
            torch_model = Boltz1.load_from_checkpoint(
                self.boltz_cache_dir / "boltz1_conf.ckpt",
                strict=True,
                map_location="cpu",
                predict_args={"recycling_steps": self.recycling_steps, "sampling_steps": 25, "diffusion_samples": 1},
                diffusion_process_args=asdict(BoltzDiffusionParams()),
                ema=False,
            )
        finally:
            torch.load = orig_load

        # Move NumPy weight leaves to JAX device arrays BEFORE JIT so XLA never constant-folds them!
        raw_joltz = joltz.from_torch(torch_model)
        del torch_model
        gc.collect()
        self.model = jax.tree_util.tree_map(
            lambda x: jnp.asarray(x) if isinstance(x, np.ndarray) else x,
            raw_joltz,
        )
        del raw_joltz
        gc.collect()
        print(
            f"[JOLTZ] Converted & placed on {jax.devices()[0]} in {time.time() - t0:.1f}s "
            f"(rec={self.recycling_steps}, max_msa={self.max_msa_seqs})",
            flush=True,
        )

        rec = self.recycling_steps

        # Pass `model` as an explicit dynamic argument to avoid XLA constant-folding 48 layers!
        @eqx.filter_jit
        def _run_trunk_norm(model, feats: Dict[str, jax.Array]) -> Tuple[jax.Array, jax.Array]:
            out = model.trunk(
                feats,
                recycling_steps=rec,
                key=jax.random.key(0),
                deterministic=True,
            )
            # Canonical Boltz-1 LayerNorms normalize raw residual streams (std ~649 / ~43 -> ~1.0)
            s_norm = model.s_norm(out.s)[0]
            z_norm = model.z_norm(out.z)[0]
            return s_norm, z_norm

        self._run_trunk_norm = _run_trunk_norm

    def prepare_pdb_wt_msa(self, pdb_id: str, wt_seq: str, chain_ids: Sequence[str]) -> bool:
        """Fetches ColabFold MSA once per WT PDB complex (cached on disk, reused for all mutants)."""
        if not self.use_msa:
            return False
        groups, _ = _group_chains(wt_seq, chain_ids)
        data_map: Dict[str, str] = {}
        missing = False
        for cid, _, subseq in groups:
            key = f"{pdb_id}_{cid}"
            data_map[key] = subseq
            if not (self.msa_cache_dir / f"{key}.csv").exists():
                missing = True
        if not missing:
            return True
        try:
            print(f"[MSA] Fetching WT ColabFold MSA once for {pdb_id} ({len(data_map)} chains)...", flush=True)
            self._compute_msa(
                data=data_map,
                target_id=pdb_id,
                msa_dir=self.msa_cache_dir,
                msa_server_url="https://api.colabfold.com",
                msa_pairing_strategy="greedy",
            )
            return True
        except Exception as e:
            print(f"[MSA][WARN] ColabFold MSA failed for {pdb_id} ({e}); falling back to single-sequence.", flush=True)
            return False

    def _build_chain_msas(
        self,
        pdb_id: str,
        groups: List[Tuple[str, List[int], str]],
        target_chains: Sequence[object],
        strip_pairing: bool = False,
    ) -> Dict[int, object]:
        """Builds per-chain MSA objects by updating only row-0 query to the mutant sequence."""
        if not self.use_msa:
            return {}
        msas: Dict[int, object] = {}
        for (orig_cid, _, cur_seq), rec_chain in zip(groups, target_chains):
            wt_csv = self.msa_cache_dir / f"{pdb_id}_{orig_cid}.csv"
            if not wt_csv.exists():
                continue
            lines = wt_csv.read_text(encoding="utf-8").splitlines()
            if len(lines) < 2:
                continue
            new_lines = [lines[0]]
            for row_idx, line in enumerate(lines[1:]):
                parts = line.split(",", 1)
                if len(parts) != 2:
                    continue
                row_key, row_seq = parts[0].strip(), parts[1].strip()
                if strip_pairing:
                    row_key = "-1"
                if row_idx == 0 and len(cur_seq) == len(row_seq):
                    row_seq = cur_seq
                new_lines.append(f"{row_key},{row_seq}")
            with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False, encoding="utf-8") as tf:
                tf.write("\n".join(new_lines) + "\n")
                tmp_path = Path(tf.name)
            try:
                msas[rec_chain.chain_id] = self._parse_csv(tmp_path, max_seqs=self.max_msa_seqs)
            finally:
                tmp_path.unlink(missing_ok=True)
        return msas

    def _pick_bucket(self, n_tokens: int) -> int:
        for b in self.buckets:
            if n_tokens <= b:
                return b
        raise ValueError(f"Sequence length {n_tokens} exceeds maximum bucket {self.buckets[-1]}.")

    def extract_sparse_priors(
        self,
        seq: str,
        chain_ids: Sequence[str],
        keep_rows: np.ndarray,
        keep_cols: np.ndarray,
        pdb_id: str,
        state_tag: str,
        strip_pairing: bool = False,
    ) -> str:
        """
        Extracts and caches normalized frozen Joltz-1 (s, sparse z) for a protein state to an uncompressed .npz.
        Identical states (e.g., WT complex or unchanged unbound partner) hit cache with zero recomputation.
        """
        groups, perm = _group_chains(seq, chain_ids)
        n_res = len(seq)
        sig_payload = (
            f"v1|{pdb_id}|{state_tag}|rec={self.recycling_steps}|msa={int(self.use_msa)}|"
            f"nseq={self.max_msa_seqs}|strip={int(strip_pairing)}|"
            + "|".join(f"{cid}:{subseq}" for cid, _, subseq in groups)
        )
        h = hashlib.sha256(sig_payload.encode("utf-8")).hexdigest()[:24]
        cache_path = self.prior_cache_dir / f"{pdb_id}_{state_tag}_{h}.npz"
        if cache_path.exists():
            return str(cache_path)

        schema = {
            "version": 1,
            "sequences": [
                {
                    "protein": {
                        "id": _ascii_chain_name(i),
                        "sequence": subseq,
                        "modifications": [],
                        "msa": "empty",
                    }
                }
                for i, (_, _, subseq) in enumerate(groups)
            ],
            "bonds": [],
        }

        with contextlib.redirect_stdout(io.StringIO()):
            target = self._parse_boltz_schema(f"{pdb_id}_{state_tag}", schema, self.ccd, None, False)
        msas = self._build_chain_msas(pdb_id, groups, target.record.chains, strip_pairing=strip_pairing)
        tokenized = self._tokenizer.tokenize(self._Input(target.structure, msas, target.record, None))
        n_tok = len(tokenized.tokens)
        if n_tok != n_res:
            raise RuntimeError(f"Boltz token mismatch for {pdb_id}_{state_tag}: {n_tok} != {n_res}")

        bucket_tok = self._pick_bucket(n_tok)
        n_atoms = len(tokenized.structure.atoms)
        bucket_atoms = max(bucket_tok * 16, ((n_atoms + 127) // 128) * 128)

        feats = self._featurizer.process(
            tokenized,
            training=False,
            max_seqs=self.max_msa_seqs,
            max_tokens=bucket_tok,
            max_atoms=bucket_atoms,
            pad_to_max_seqs=True,
            symmetries={},
            compute_symmetries=False,
            compute_constraint_features=False,
        )
        batched = self._collate([feats])
        jax_feats = _to_numpy_dict(batched)

        t_call = time.time()
        s_tok, z_tok = self._run_trunk_norm(self.model, jax_feats)
        s_np = np.asarray(jax.device_get(s_tok[:n_res]), dtype=np.float32)
        z_np = np.asarray(jax.device_get(z_tok[:n_res, :n_res]), dtype=np.float32)
        dt = time.time() - t_call
        self.trunk_calls += 1
        if self.trunk_calls <= 6 or self.trunk_calls % 25 == 0:
            print(
                f"[JOLTZ] trunk #{self.trunk_calls:4d} ({pdb_id}_{state_tag}, N={n_res}->bucket={bucket_tok}) "
                f"in {dt:.2f}s",
                flush=True,
            )

        # Map back from Boltz grouped-chain order to loader.py residue index order
        s_orig = np.zeros_like(s_np)
        z_orig = np.zeros_like(z_np)
        s_orig[perm] = s_np
        z_orig[np.ix_(perm, perm)] = z_np

        r_idx = np.asarray(keep_rows, dtype=np.int16)
        c_idx = np.asarray(keep_cols, dtype=np.int16)
        s_f16 = np.clip(s_orig, -3.0e4, 3.0e4).astype(np.float16)
        z_vals_f16 = np.clip(z_orig[r_idx, c_idx], -3.0e4, 3.0e4).astype(np.float16)

        tmp_out = str(cache_path) + ".tmp.npz"
        np.savez(tmp_out, s=s_f16, rows=r_idx, cols=c_idx, z_vals=z_vals_f16, n_res=np.int32(n_res))
        os.replace(tmp_out, cache_path)
        return str(cache_path)

    def close(self) -> None:
        """Releases Joltz-1 weights from GPU/TPU HBM before downstream Optax training begins."""
        self.model = None
        self._run_trunk_norm = None
        gc.collect()
        jax.clear_caches()


def load_bound_unbound_priors(
    bound_path: str,
    unbound_a_path: str,
    unbound_b_path: str,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    max_n: int,
    c_s: int = 384,
    c_z: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstructs padded float16 (s_bound, z_bound, s_unbound, z_unbound) from sparse contact caches.
    Unbound arrays contain strictly intra-partner A and intra-partner B blocks (zero cross-partner telepathy).
    """
    s_b = np.zeros((max_n, c_s), dtype=np.float16)
    z_b = np.zeros((max_n, max_n, c_z), dtype=np.float16)
    with np.load(bound_path, allow_pickle=False) as d:
        n_res = int(d["n_res"])
        s_b[:n_res] = d["s"]
        if d["rows"].size > 0:
            z_b[d["rows"], d["cols"]] = d["z_vals"]

    s_u = np.zeros((max_n, c_s), dtype=np.float16)
    z_u = np.zeros((max_n, max_n, c_z), dtype=np.float16)
    with np.load(unbound_a_path, allow_pickle=False) as da:
        s_u[idx_a] = da["s"]
        if da["rows"].size > 0:
            z_u[idx_a[da["rows"]], idx_a[da["cols"]]] = da["z_vals"]
    with np.load(unbound_b_path, allow_pickle=False) as db:
        s_u[idx_b] = db["s"]
        if db["rows"].size > 0:
            z_u[idx_b[db["rows"]], idx_b[db["cols"]]] = db["z_vals"]

    return s_b, z_b, s_u, z_u
