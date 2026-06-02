#!/usr/bin/env python3
# qwen25vl_oracle.py — DEV-ONLY numpy ground truth for the Qwen2.5-VL text
# encoder (text-only path), the Qwen-Image text encoder.
#
# Reads the REAL Qwen-Image-2512 text_encoder BF16 weights straight from the
# shard files (raw safetensors header parse + bf16->f32 upcast; NO torch, NO
# transformers required — this mirrors the verified qwen3 oracle.py approach and
# matches HF's Qwen2_5_VLForConditionalGeneration text decoder math exactly).
# Emits per-layer hidden states (post each decoder layer, PRE-final-norm) plus
# the final last_hidden_state (after model.norm) as JSONL, for a FIXED token-id
# list.
#
# Qwen2.5-VL decoder differs from Qwen3 in exactly two ways:
#   * Q/K/V projections HAVE BIASES (attention_bias=True); o_proj is bias-free.
#   * NO per-head q_norm / k_norm (those weights do not exist).
# Everything else is identical: RMSNorm(f32), GQA, RoPE half-split (theta=1e6),
# causal attention (scale 1/sqrt(head_dim)), SwiGLU MLP.
#
# For text-only input the Qwen2.5-VL mRoPE collapses to standard 1D positions
# (the 3 mrope sections all index the same per-token position id when there are
# no vision tokens), so 1D half-split RoPE is exact.
#
# Run: pixi run python serenitymojo/models/text_encoder/parity/qwen25vl_oracle.py
import glob
import json
import os
import struct
import sys

import numpy as np

# Qwen-Image-2512 text_encoder snapshot. Resolve the snapshot dir at runtime so
# this works regardless of the exact revision hash.
_HUB = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen-Image-2512/snapshots"
)
_cands = sorted(glob.glob(os.path.join(_HUB, "*", "text_encoder")))
if not _cands:
    sys.exit("FATAL: no Qwen-Image-2512 text_encoder snapshot found under " + _HUB)
TE_DIR = _cands[0]

# Fixed token ids (a short hardcoded "prompt" — independent of any tokenizer).
# 8 tokens, plausible Qwen vocab ids (no pad). MUST match the parity driver.
TOKEN_IDS = [9707, 11, 1879, 0, 358, 1079, 264, 1467]

CFG = dict(
    hidden=3584, layers=28, heads=28, kv_heads=4, head_dim=128,
    eps=1e-6, theta=1e6,
)
N_EMIT_LAYERS = 4  # dump first N per-layer states (plus final)


def _shard_index(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen))
    data_off = 8 + hlen
    out = {}
    for name, info in hdr.items():
        if name == "__metadata__":
            continue
        s, e = info["data_offsets"]
        out[name] = (data_off + s, e - s, info["dtype"], info["shape"])
    return out


class Weights:
    """Lazy loader: name -> float32 ndarray (bf16 upcast)."""

    def __init__(self, d):
        idxp = os.path.join(d, "model.safetensors.index.json")
        self.wm = json.load(open(idxp))["weight_map"]
        self.d = d
        self._idx_cache = {}
        self._cache = {}

    def _idx(self, fn):
        if fn not in self._idx_cache:
            self._idx_cache[fn] = _shard_index(os.path.join(self.d, fn))
        return self._idx_cache[fn]

    def get(self, name):
        if name in self._cache:
            return self._cache[name]
        fn = self.wm[name]
        off, size, dtype, shape = self._idx(fn)[name]
        with open(os.path.join(self.d, fn), "rb") as f:
            f.seek(off)
            blob = f.read(size)
        if dtype == "BF16":
            u16 = np.frombuffer(blob, dtype="<u2")
            u32 = u16.astype(np.uint32) << 16
            arr = u32.view(np.float32).reshape(shape)
        elif dtype in ("F32", "F32"):
            arr = np.frombuffer(blob, dtype="<f4").reshape(shape)
        elif dtype == "F16":
            arr = np.frombuffer(blob, dtype="<f2").astype(np.float32).reshape(shape)
        else:
            raise AssertionError(f"{name} unsupported dtype {dtype}")
        self._cache[name] = arr
        return arr


def rms_norm(x, w, eps):
    var = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    xn = x / np.sqrt(var + eps)
    return (xn * w).astype(np.float32)


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope_tables(seq, head_dim, theta):
    half = head_dim // 2
    i = np.arange(half, dtype=np.float64)
    inv_freq = np.exp(-np.log(theta) * (2.0 * i) / head_dim)
    pos = np.arange(seq, dtype=np.float64)
    ang = np.outer(pos, inv_freq)  # [seq, half]
    return np.cos(ang).astype(np.float32), np.sin(ang).astype(np.float32)


def apply_rope_halfsplit(x, cos, sin):
    # x: [seq, heads, head_dim]; cos/sin: [seq, half]
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    c = cos[:, None, :]
    s = sin[:, None, :]
    out1 = x1 * c - x2 * s
    out2 = x2 * c + x1 * s
    return np.concatenate([out1, out2], axis=-1)


def main():
    W = Weights(TE_DIR)
    ids = np.array(TOKEN_IDS, dtype=np.int64)
    seq = len(ids)
    H, HKV, DH = CFG["heads"], CFG["kv_heads"], CFG["head_dim"]
    n_rep = H // HKV
    eps = CFG["eps"]
    scale = 1.0 / np.sqrt(DH)

    emb = W.get("model.embed_tokens.weight")  # [vocab, hidden]
    hidden = emb[ids].astype(np.float32)  # [seq, hidden]

    cos, sin = rope_tables(seq, DH, CFG["theta"])

    # causal mask additive (float32)
    causal = np.triu(np.full((seq, seq), -1e30, dtype=np.float32), k=1)

    out = open(os.path.join(os.path.dirname(__file__), "qwen25vl_ref.jsonl"), "w")

    def emit(tag, arr):
        out.write(json.dumps({
            "tag": tag, "shape": list(arr.shape),
            "data": arr.reshape(-1).astype(np.float32).tolist(),
        }) + "\n")

    emit("token_ids", ids.astype(np.float32))
    emit("embed", hidden)

    for L in range(CFG["layers"]):
        p = f"model.layers.{L}"
        residual = hidden
        x = rms_norm(hidden, W.get(f"{p}.input_layernorm.weight"), eps)
        if L == 0:
            emit("l0_input_norm", x)
        # Q/K/V projections WITH biases (Qwen2.5-VL attention_bias=True).
        q = x @ W.get(f"{p}.self_attn.q_proj.weight").T + W.get(f"{p}.self_attn.q_proj.bias")
        k = x @ W.get(f"{p}.self_attn.k_proj.weight").T + W.get(f"{p}.self_attn.k_proj.bias")
        v = x @ W.get(f"{p}.self_attn.v_proj.weight").T + W.get(f"{p}.self_attn.v_proj.bias")
        q = q.reshape(seq, H, DH)
        k = k.reshape(seq, HKV, DH)
        v = v.reshape(seq, HKV, DH)
        # NO per-head qk-norm in Qwen2.5-VL.
        # rope (half-split)
        q = apply_rope_halfsplit(q, cos, sin)
        k = apply_rope_halfsplit(k, cos, sin)
        if L == 0:
            emit("l0_q_rope", q)  # [seq, H, DH]
            emit("l0_k_rope", k)  # [seq, HKV, DH]
        # gqa repeat
        k = np.repeat(k, n_rep, axis=1)  # [seq, H, DH]
        v = np.repeat(v, n_rep, axis=1)
        qh = q.transpose(1, 0, 2)  # [H, seq, DH]
        kh = k.transpose(1, 0, 2)
        vh = v.transpose(1, 0, 2)
        scores = np.einsum("hqd,hkd->hqk", qh, kh) * scale
        scores = scores + causal[None, :, :]
        scores = scores - scores.max(axis=-1, keepdims=True)
        w_ = np.exp(scores)
        w_ = w_ / w_.sum(axis=-1, keepdims=True)
        attn = np.einsum("hqk,hkd->hqd", w_, vh)  # [H, seq, DH]
        attn = attn.transpose(1, 0, 2).reshape(seq, H * DH)
        attn = attn @ W.get(f"{p}.self_attn.o_proj.weight").T  # o_proj bias-free
        hidden = (residual + attn).astype(np.float32)

        residual = hidden
        x = rms_norm(hidden, W.get(f"{p}.post_attention_layernorm.weight"), eps)
        gate = x @ W.get(f"{p}.mlp.gate_proj.weight").T
        up = x @ W.get(f"{p}.mlp.up_proj.weight").T
        act = silu(gate) * up
        down = act @ W.get(f"{p}.mlp.down_proj.weight").T
        hidden = (residual + down).astype(np.float32)

        if L < N_EMIT_LAYERS or L == CFG["layers"] - 1:
            emit(f"layer{L}", hidden)

    final = rms_norm(hidden, W.get("model.norm.weight"), eps)
    emit("last_hidden_state", final)
    out.close()
    print("qwen25vl_oracle: wrote qwen25vl_ref.jsonl  seq=%d  te_dir=%s"
          % (seq, TE_DIR), file=sys.stderr)


if __name__ == "__main__":
    main()
