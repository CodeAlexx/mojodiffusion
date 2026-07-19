#!/usr/bin/env python3
# oracle.py — DEV-ONLY numpy ground truth for the Qwen3 text encoder.
#
# Reads the REAL Z-Image text_encoder BF16 weights straight from the shard
# files (raw safetensors header parse + bf16->f32 upcast; NO torch, NO
# transformers — neither is installable on this py3.14 env). Implements the
# Qwen3ForCausalLM decoder forward in float32 numpy and emits per-layer hidden
# states (post each decoder layer, PRE-final-norm) plus the final
# last_hidden_state (after model.norm) as JSONL, for a FIXED token-id list.
#
# This mirrors transformers' Qwen3 math exactly:
#   * RMSNorm in float32: x * rsqrt(mean(x^2)+eps) * w
#   * GQA: 32 q heads, 8 kv heads (n_rep=4), head_dim=128
#   * per-head q_norm / k_norm (RMSNorm over head_dim) applied BEFORE rope
#   * RoPE half-split (rotate_half), theta=1e6, inv_freq = theta^(-2i/dim)
#   * causal attention (lower-triangular), scale = 1/sqrt(head_dim)
#   * SwiGLU MLP: down(silu(gate(x)) * up(x))
#
# Run: pixi run python serenitymojo/models/text_encoder/parity/oracle.py
import json, os, struct, sys
import numpy as np

TE_DIR = (
    "/home/alex/.cache/huggingface/hub/models--Tongyi-MAI--Z-Image/"
    "snapshots/04cc4abb7c5069926f75c9bfde9ef43d49423021/text_encoder"
)

# Fixed token ids (a short hardcoded "prompt" — independent of any tokenizer).
# 8 tokens. Values are plausible Qwen vocab ids (no pad).
TOKEN_IDS = [9707, 11, 1879, 0, 358, 1079, 264, 1467]

CFG = dict(
    hidden=2560, layers=36, heads=32, kv_heads=8, head_dim=128,
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
        assert dtype == "BF16", f"{name} dtype {dtype}"
        u16 = np.frombuffer(blob, dtype="<u2")
        # bf16 -> f32: place the 16 bits in the high half of a uint32.
        u32 = u16.astype(np.uint32) << 16
        arr = u32.view(np.float32).reshape(shape)
        self._cache[name] = arr
        return arr


def rms_norm(x, w, eps):
    # x: [..., D], w: [D]; float32 throughout.
    var = np.mean(x.astype(np.float64) ** 2, axis=-1, keepdims=True)
    xn = x / np.sqrt(var + eps)
    return (xn * w).astype(np.float32)


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope_tables(seq, head_dim, theta):
    # half-split: inv_freq[i] = theta^(-2i/dim), i in [0, dim/2)
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

    out = open(os.path.join(os.path.dirname(__file__), "ref.jsonl"), "w")

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
        q = x @ W.get(f"{p}.self_attn.q_proj.weight").T  # [seq, H*DH]
        k = x @ W.get(f"{p}.self_attn.k_proj.weight").T  # [seq, HKV*DH]
        v = x @ W.get(f"{p}.self_attn.v_proj.weight").T
        q = q.reshape(seq, H, DH)
        k = k.reshape(seq, HKV, DH)
        v = v.reshape(seq, HKV, DH)
        # per-head qk-norm
        q = rms_norm(q, W.get(f"{p}.self_attn.q_norm.weight"), eps)
        k = rms_norm(k, W.get(f"{p}.self_attn.k_norm.weight"), eps)
        # rope
        q = apply_rope_halfsplit(q, cos, sin)
        k = apply_rope_halfsplit(k, cos, sin)
        if L == 0:
            # store in BSHD flat order [seq, H, DH] / [seq, HKV, DH]
            emit("l0_q_rope", q)
            emit("l0_k_rope", k)
        # gqa repeat
        k = np.repeat(k, n_rep, axis=1)  # [seq, H, DH]
        v = np.repeat(v, n_rep, axis=1)
        # attention per head: q,k,v [seq, H, DH] -> scores [H, seq, seq]
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
        attn = attn @ W.get(f"{p}.self_attn.o_proj.weight").T
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
    print("oracle: wrote ref.jsonl  seq=%d" % seq, file=sys.stderr)


if __name__ == "__main__":
    main()
