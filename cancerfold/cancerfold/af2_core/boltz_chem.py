"""Pure-JAX SE(3)-invariant Boltzmann Chemical Potential U_chem(X; alpha) via MIT ProteinMPNN v_48_020."""
import io
import os
import pickle
import urllib.request
import zipfile
from collections import OrderedDict
from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from cancerfold.mutation.parser import AA_ORDER

MPNN_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/dauparas/ProteinMPNN/main/vanilla_model_weights/v_48_020.pt"
)
MPNN_ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
MPNN_TO_IDX = {aa: i for i, aa in enumerate(MPNN_ALPHABET)}
# Maps OncoVac AA_ORDER ("ARNDCQEGHILKMFPSTWYV") -> ProteinMPNN alphabetical index (pad/unknown -> 20)
ONCOVAC_TO_MPNN_NP = np.array([MPNN_TO_IDX[aa] for aa in AA_ORDER] + [20], dtype=np.int32)
ONCOVAC_TO_MPNN = jnp.asarray(ONCOVAC_TO_MPNN_NP, dtype=jnp.int32)

KBT_CHEM_STD = 0.5924  # RT at 298.15 K [kcal/mol]


def _load_pt_pure_numpy(pt_path: str) -> Dict[str, object]:
    """Reads PyTorch ZIP checkpoint (.pt) in pure Python + NumPy with ZERO PyTorch dependency."""
    with open(pt_path, "rb") as f:
        raw_bytes = f.read()
    zf = zipfile.ZipFile(io.BytesIO(raw_bytes), "r")
    pkl_name = [n for n in zf.namelist() if n.endswith("data.pkl")][0]
    prefix = pkl_name[:-len("data.pkl")]
    storages: Dict[str, np.ndarray] = {}

    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        if len(size) == 0:
            return storage[storage_offset : storage_offset + 1].reshape(()).copy()
        return np.lib.stride_tricks.as_strided(
            storage[storage_offset:],
            shape=tuple(size),
            strides=tuple(s * storage.itemsize for s in stride),
        ).copy()

    dtype_map = {
        "FloatStorage": np.float32,
        "DoubleStorage": np.float64,
        "HalfStorage": np.float16,
        "LongStorage": np.int64,
        "IntStorage": np.int32,
        "ShortStorage": np.int16,
        "CharStorage": np.int8,
        "ByteStorage": np.uint8,
        "BoolStorage": np.bool_,
    }

    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module == "torch._utils" and name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if module == "collections" and name == "OrderedDict":
                return OrderedDict
            if module == "torch" and name in dtype_map:
                return dtype_map[name]
            raise pickle.UnpicklingError(f"Unsupported checkpoint global: {module}.{name}")

        def persistent_load(self, pid):
            _, dtype, key, _, numel = pid
            if key not in storages:
                buf = zf.read(f"{prefix}data/{key}")
                storages[key] = np.frombuffer(buf, dtype=dtype, count=numel).copy()
            return storages[key]

    return _Unpickler(io.BytesIO(zf.read(pkl_name))).load()


def load_boltz_chem_params(cache_dir: str = "/content/oncovac_cache") -> Dict[str, jnp.ndarray]:
    """Downloads (if missing) and loads official MIT ProteinMPNN v_48_020 weights into pure JAX."""
    os.makedirs(cache_dir, exist_ok=True)
    pt_path = os.path.join(cache_dir, "proteinmpnn_v_48_020.pt")
    if not (os.path.exists(pt_path) and os.path.getsize(pt_path) > 1_000_000):
        tmp_path = pt_path + ".tmp"
        urllib.request.urlretrieve(MPNN_WEIGHTS_URL, tmp_path)
        os.replace(tmp_path, pt_path)
    ckpt = _load_pt_pure_numpy(pt_path)
    sd = ckpt["model_state_dict"]
    return {k: jnp.asarray(v, dtype=jnp.float32) for k, v in sd.items()}


def oncovac_idx_to_mpnn(aa_idx: jnp.ndarray, res_mask: jnp.ndarray) -> jnp.ndarray:
    """Converts OncoVac AA indices (ARNDCQEGHILKMFPSTWYV) to ProteinMPNN alphabet indices (pad -> 20)."""
    safe_idx = jnp.where(res_mask > 0.5, jnp.clip(aa_idx, 0, 19), 20)
    return ONCOVAC_TO_MPNN[safe_idx]


def make_fixed_decoding_ranks(res_mask_np: np.ndarray, num_orders: int = 4) -> np.ndarray:
    """
    Deterministic, sequence-independent decoding ranks (K, MAX_N) for exact BAR antisymmetry.
    Active residues receive ranks 0..N_real-1; masked/padded residues receive MAX_N + 10.
    """
    mask_bool = np.asarray(res_mask_np) > 0.5
    max_n = int(mask_bool.shape[0])
    active = np.where(mask_bool)[0]
    n_real = int(active.size)
    ranks = np.full((num_orders, max_n), max_n + 10, dtype=np.int32)
    if n_real == 0:
        return ranks
    rng = np.random.default_rng(42 + n_real)
    orders = [
        np.arange(n_real, dtype=np.int32),
        np.arange(n_real - 1, -1, -1, dtype=np.int32),
    ]
    while len(orders) < num_orders:
        orders.append(rng.permutation(n_real).astype(np.int32))
    for k in range(num_orders):
        ranks[k, active] = np.argsort(orders[k]).astype(np.int32)
    return ranks


def _layer_norm(x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * w + b


def _linear(x: jnp.ndarray, w: jnp.ndarray, b: jnp.ndarray = None) -> jnp.ndarray:
    y = jnp.matmul(x, w.T)
    return y if b is None else y + b


def _gelu(x: jnp.ndarray) -> jnp.ndarray:
    return jax.nn.gelu(x, approximate=False)


def build_peptide_o_and_cb(
    backbone: jnp.ndarray,
    chain_mask: jnp.ndarray,
    res_mask: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    SE(3)-equivariant analytical Engh-Huber peptide-plane carbonyl O & chiral pseudo-CB
    reconstructed purely from live (N, CA, C) coordinates (0% static X_0 leakage).
    """
    n_pos, ca_pos, c_pos = backbone[:, 0, :], backbone[:, 1, :], backbone[:, 2, :]
    b = ca_pos - n_pos
    c = c_pos - ca_pos
    a = jnp.cross(b, c)
    cb_pos = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + ca_pos

    next_n = jnp.concatenate([n_pos[1:], n_pos[-1:]], axis=0)
    next_cm = jnp.concatenate([chain_mask[1:], jnp.array([-999], dtype=chain_mask.dtype)], axis=0)
    next_m = jnp.concatenate([res_mask[1:], jnp.array([0.0], dtype=jnp.float32)], axis=0)
    cn_dist = jnp.sqrt(jnp.sum((c_pos - next_n) ** 2, axis=-1) + 1e-12)
    has_next = (chain_mask == next_cm) & (chain_mask >= 0) & (next_m > 0.5) & (res_mask > 0.5) & (cn_dist < 2.2)

    v1 = c_pos - ca_pos
    v1 = v1 / jnp.sqrt(jnp.sum(v1 ** 2, axis=-1, keepdims=True) + 1e-12)
    v2 = c_pos - next_n
    v2 = v2 / jnp.sqrt(jnp.sum(v2 ** 2, axis=-1, keepdims=True) + 1e-12)
    d_mid = v1 + v2
    d_mid = d_mid / jnp.sqrt(jnp.sum(d_mid ** 2, axis=-1, keepdims=True) + 1e-12)

    v0 = n_pos - ca_pos
    vp = v0 - jnp.sum(v0 * v1, axis=-1, keepdims=True) * v1
    vp = vp / jnp.sqrt(jnp.sum(vp ** 2, axis=-1, keepdims=True) + 1e-12)
    d_term = jnp.cos(jnp.deg2rad(59.5)) * v1 - jnp.sin(jnp.deg2rad(59.5)) * vp
    d_term = d_term / jnp.sqrt(jnp.sum(d_term ** 2, axis=-1, keepdims=True) + 1e-12)

    o_pos = c_pos + 1.231 * jnp.where(has_next[:, None], d_mid, d_term)
    return n_pos, ca_pos, c_pos, o_pos, cb_pos


def _rbf(d: jnp.ndarray) -> jnp.ndarray:
    d_mu = jnp.linspace(2.0, 22.0, 16, dtype=jnp.float32)
    d_sigma = (22.0 - 2.0) / 16.0
    return jnp.exp(-((d[..., None] - d_mu) / d_sigma) ** 2)


def _pair_dist_neighbors(a: jnp.ndarray, b: jnp.ndarray, e_idx: jnp.ndarray) -> jnp.ndarray:
    diff = a[:, None, :] - b[e_idx]
    return jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-6)


def encode_backbone_chem(
    params: Dict[str, jnp.ndarray],
    backbone: jnp.ndarray,
    res_mask: jnp.ndarray,
    chain_mask: jnp.ndarray,
    top_k: int = 48,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Sequence-independent SE(3)-invariant ProteinMPNN backbone encoder (exact padding & partner isolation)."""
    n_len = backbone.shape[0]
    k_eff = min(top_k, n_len)
    m = res_mask.astype(jnp.float32)
    n_pos, ca_pos, c_pos, o_pos, cb_pos = build_peptide_o_and_cb(backbone, chain_mask, m)

    mask_2d = m[:, None] * m[None, :]
    d_ca_full = mask_2d * jnp.sqrt(jnp.sum((ca_pos[:, None, :] - ca_pos[None, :, :]) ** 2, axis=-1) + 1e-6)
    d_max = jnp.max(d_ca_full, axis=-1, keepdims=True)
    d_adjust = d_ca_full + (1.0 - mask_2d) * (d_max + 1e5)
    neg_d, e_idx = jax.lax.top_k(-d_adjust, k_eff)
    row_idx = jnp.arange(n_len)[:, None]
    d_neighbors = jnp.where(mask_2d[row_idx, e_idx] > 0.5, -neg_d, d_max)

    pairs = (
        (n_pos, n_pos), (c_pos, c_pos), (o_pos, o_pos), (cb_pos, cb_pos),
        (ca_pos, n_pos), (ca_pos, c_pos), (ca_pos, o_pos), (ca_pos, cb_pos),
        (n_pos, c_pos), (n_pos, o_pos), (n_pos, cb_pos), (cb_pos, c_pos), (cb_pos, o_pos), (o_pos, c_pos),
        (n_pos, ca_pos), (c_pos, ca_pos), (o_pos, ca_pos), (cb_pos, ca_pos),
        (c_pos, n_pos), (o_pos, n_pos), (cb_pos, n_pos), (c_pos, cb_pos), (o_pos, cb_pos), (c_pos, o_pos),
    )
    rbfs = [_rbf(d_neighbors)] + [_rbf(_pair_dist_neighbors(a, b, e_idx)) for (a, b) in pairs]
    rbf_all = jnp.concatenate(rbfs, axis=-1)

    res_idx = jnp.arange(n_len, dtype=jnp.int32)
    offset = res_idx[:, None] - res_idx[e_idx]
    e_chains = ((chain_mask[:, None] == chain_mask[e_idx]) & (chain_mask[:, None] >= 0)).astype(jnp.int32)
    d_pos = jnp.clip(offset + 32, 0, 64) * e_chains + (1 - e_chains) * 65
    d_oh = jax.nn.one_hot(d_pos, 66, dtype=jnp.float32)
    e_pos = _linear(d_oh, params["features.embeddings.linear.weight"], params["features.embeddings.linear.bias"])

    e_feat = jnp.concatenate([e_pos, rbf_all], axis=-1)
    e_feat = _linear(e_feat, params["features.edge_embedding.weight"])
    e_feat = _layer_norm(e_feat, params["features.norm_edges.weight"], params["features.norm_edges.bias"])

    h_v = jnp.zeros((n_len, e_feat.shape[-1]), dtype=jnp.float32)
    h_e = _linear(e_feat, params["W_e.weight"], params["W_e.bias"])
    m_ij = m[:, None] * m[e_idx]

    for i in range(3):
        p = f"encoder_layers.{i}."
        h_v_nb = h_v[e_idx]
        h_ev = jnp.concatenate([jnp.broadcast_to(h_v[:, None, :], h_v_nb.shape), h_e, h_v_nb], axis=-1)
        msg = _linear(
            _gelu(_linear(_gelu(_linear(h_ev, params[p + "W1.weight"], params[p + "W1.bias"])),
                          params[p + "W2.weight"], params[p + "W2.bias"])),
            params[p + "W3.weight"], params[p + "W3.bias"],
        )
        dh = jnp.sum(m_ij[..., None] * msg, axis=-2) / 30.0
        h_v = _layer_norm(h_v + dh, params[p + "norm1.weight"], params[p + "norm1.bias"])
        dh = _linear(
            _gelu(_linear(h_v, params[p + "dense.W_in.weight"], params[p + "dense.W_in.bias"])),
            params[p + "dense.W_out.weight"], params[p + "dense.W_out.bias"],
        )
        h_v = _layer_norm(h_v + dh, params[p + "norm2.weight"], params[p + "norm2.bias"])
        h_v = m[:, None] * h_v

        h_v_nb = h_v[e_idx]
        h_ev = jnp.concatenate([jnp.broadcast_to(h_v[:, None, :], h_v_nb.shape), h_e, h_v_nb], axis=-1)
        msg_e = _linear(
            _gelu(_linear(_gelu(_linear(h_ev, params[p + "W11.weight"], params[p + "W11.bias"])),
                          params[p + "W12.weight"], params[p + "W12.bias"])),
            params[p + "W13.weight"], params[p + "W13.bias"],
        )
        h_e = _layer_norm(h_e + msg_e, params[p + "norm3.weight"], params[p + "norm3.bias"])

    zeros_s = jnp.zeros_like(h_v)
    h_ex_enc = jnp.concatenate([h_e, zeros_s[e_idx]], axis=-1)
    h_exv_enc = jnp.concatenate([h_ex_enc, h_v[e_idx]], axis=-1)
    return h_v, h_e, e_idx, m, m_ij, h_exv_enc


def decode_u_chem(
    params: Dict[str, jnp.ndarray],
    enc: Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray],
    s_mpnn: jnp.ndarray,
    ranks_k: jnp.ndarray,
    kbt: float = KBT_CHEM_STD,
) -> jnp.ndarray:
    """
    Evaluates exact Boltzmann Chemical Potential U_chem(X; alpha) = -RT * mean_k ln P_{pi_k}(alpha | X).
    Pure state function of (X, alpha) -> exact 0.00e+00 BAR antisymmetry & Zero-Telepathy.
    """
    h_v0, h_e, e_idx, m, m_ij, h_exv_enc = enc
    h_s = params["W_s.weight"][s_mpnn]
    h_es = jnp.concatenate([h_e, h_s[e_idx]], axis=-1)
    n_len, k_eff = e_idx.shape[0], e_idx.shape[1]
    d_h = h_v0.shape[1]

    def _one_order(rank: jnp.ndarray) -> jnp.ndarray:
        m_bw_raw = (rank[e_idx] < rank[:, None]).astype(jnp.float32)
        m_bw = (m_ij * m_bw_raw)[..., None]
        m_fw = (m_ij * (1.0 - m_bw_raw))[..., None]
        h_fw = m_fw * h_exv_enc
        h_v = h_v0
        for i in range(3):
            p = f"decoder_layers.{i}."
            h_esv = jnp.concatenate([h_es, h_v[e_idx]], axis=-1)
            h_esv = m_bw * h_esv + h_fw
            h_in = jnp.concatenate([jnp.broadcast_to(h_v[:, None, :], (n_len, k_eff, d_h)), h_esv], axis=-1)
            msg = _linear(
                _gelu(_linear(_gelu(_linear(h_in, params[p + "W1.weight"], params[p + "W1.bias"])),
                              params[p + "W2.weight"], params[p + "W2.bias"])),
                params[p + "W3.weight"], params[p + "W3.bias"],
            )
            dh = jnp.sum(m_ij[..., None] * msg, axis=-2) / 30.0
            h_v = _layer_norm(h_v + dh, params[p + "norm1.weight"], params[p + "norm1.bias"])
            dh = _linear(
                _gelu(_linear(h_v, params[p + "dense.W_in.weight"], params[p + "dense.W_in.bias"])),
                params[p + "dense.W_out.weight"], params[p + "dense.W_out.bias"],
            )
            h_v = _layer_norm(h_v + dh, params[p + "norm2.weight"], params[p + "norm2.bias"])
            h_v = m[:, None] * h_v
        logits = _linear(h_v, params["W_out.weight"], params["W_out.bias"])
        logp = jax.nn.log_softmax(logits, axis=-1)
        lp_true = logp[jnp.arange(n_len), s_mpnn]
        return -jnp.sum(lp_true * m)

    nlls = jax.vmap(_one_order)(ranks_k)
    return jnp.asarray(kbt, dtype=jnp.float32) * jnp.mean(nlls)


def u_chem_energy(
    params: Dict[str, jnp.ndarray],
    backbone: jnp.ndarray,
    aa_idx: jnp.ndarray,
    res_mask: jnp.ndarray,
    chain_mask: jnp.ndarray,
    ranks_k: jnp.ndarray,
    kbt: float = KBT_CHEM_STD,
    top_k: int = 48,
) -> jnp.ndarray:
    """Single-frame end-to-end differentiable U_chem(X; alpha) for live MD or inference."""
    enc = encode_backbone_chem(params, backbone, res_mask, chain_mask, top_k=top_k)
    s_mpnn = oncovac_idx_to_mpnn(aa_idx, res_mask)
    return decode_u_chem(params, enc, s_mpnn, ranks_k, kbt=kbt)


def du_chem_ensemble(
    params: Dict[str, jnp.ndarray],
    backbone_ens: jnp.ndarray,
    aa_wt: jnp.ndarray,
    aa_mut: jnp.ndarray,
    res_mask: jnp.ndarray,
    chain_mask: jnp.ndarray,
    ranks_k: jnp.ndarray,
    kbt: float = KBT_CHEM_STD,
    top_k: int = 48,
) -> jnp.ndarray:
    """
    Computes per-frame Delta U_chem(X) = U_chem(X; Mut) - U_chem(X; WT) over an (M, MAX_N, 3, 3) ensemble.
    Encodes each frame ONCE and decodes both WT and Mut with identical fixed ranks_k.
    """
    s_wt = oncovac_idx_to_mpnn(aa_wt, res_mask)
    s_mut = oncovac_idx_to_mpnn(aa_mut, res_mask)

    def _one_frame(x_bb: jnp.ndarray) -> jnp.ndarray:
        enc = encode_backbone_chem(params, x_bb, res_mask, chain_mask, top_k=top_k)
        u_wt = decode_u_chem(params, enc, s_wt, ranks_k, kbt=kbt)
        u_mut = decode_u_chem(params, enc, s_mut, ranks_k, kbt=kbt)
        return u_mut - u_wt

    return jax.vmap(_one_frame)(backbone_ens)


# References:
#   - Inverse-folding statistical Boltzmann potential (ProteinMPNN, MIT License):
#     Dauparas et al. (2022), Science 378(6615):49-56.
#   - Peptide-plane carbonyl geometry: Engh & Huber (1991), Acta Cryst. A 47(4):392-400.
