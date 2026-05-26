#!/usr/bin/env python3
# gen_moe_reference.py — DEV-ONLY numpy oracle for the MoE op parity smoke.
#
# NOT in the runtime path. Run with `pixi run python` to emit the SAME inputs
# the Mojo smoke driver (serenitymojo/ops/moe_smoke.mojo) inlines, plus the
# expected per-stage and final reference outputs. The Mojo driver inlines both
# the inputs and the reference numbers printed here; this script is how those
# numbers were produced and how they can be regenerated/verified.
#
# Mirrors the flame-core MoE primitives this Mojo port targets:
#   * top_k_router        — token-choice top-k (descending logit, ties -> lower
#                           expert index), softmax over the k SELECTED logits.
#                           (flame-core moe_routing.rs does its top-k host-side
#                           with `b.partial_cmp(a).then(a.idx.cmp(b.idx))`; this
#                           router is token-choice rather than expert-choice but
#                           reuses the same selection + tie rule.)
#   * grouped_expert_ffn  — per-expert SwiGLU FFN:
#                           down( silu(x @ gate^T) * (x @ up^T) )
#                           (grouped_mm.rs: per-expert matmul, F32 accumulate.)
#   * gated_scatter_add   — accum[indices[t]] += expert_out[t] * gating[t]
#                           (fused_gated_scatter_add.rs, in-place F32 accum;
#                           top-k>1 means k slots collide on each token row.)
#
# Layout the Mojo side uses for the routed/permuted slots: TOKEN-MAJOR then the
# k picks for that token, i.e. slot s = t*k + j is token t's j-th expert pick.
# (This is a token-choice layout; flame-core's expert-choice uses expert-major,
# but the scatter-add formula + softmax-over-topk gating semantics are identical
# and that is what the parity gate checks.)

import numpy as np

SEED = 7777
np.random.seed(SEED)

T = 16   # tokens
E = 4    # experts
K = 2    # top-k
H = 32   # hidden dim
F = 64   # ffn (intermediate) dim

# ── inputs ───────────────────────────────────────────────────────────────────
tokens = (np.random.randn(T, H).astype(np.float32) * 0.5)
logits = (np.random.randn(T, E).astype(np.float32) * 1.5)
# Per-expert SwiGLU FFN weights, PyTorch row-major [out, in].
gate_w = (np.random.randn(E, F, H).astype(np.float32) * 0.1)   # [E, F, H]
up_w   = (np.random.randn(E, F, H).astype(np.float32) * 0.1)   # [E, F, H]
down_w = (np.random.randn(E, H, F).astype(np.float32) * 0.1)   # [E, H, F]


def silu(x):
    return x / (1.0 + np.exp(-x))


# ── stage 1: top-k router (token-choice) ──────────────────────────────────────
# For each token: select top-K experts by logit. Descending by logit; ties
# broken by LOWER expert index (matches flame-core's tie rule). Then softmax
# over ONLY the K selected logits -> gating weights that sum to 1 per token.
expert_ids = np.zeros((T, K), dtype=np.int64)
gating = np.zeros((T, K), dtype=np.float32)
for t in range(T):
    row = logits[t]
    # sort key: (-logit, index) ascending == (logit desc, index asc).
    order = sorted(range(E), key=lambda e: (-row[e], e))
    sel = order[:K]
    sel_logits = np.array([row[e] for e in sel], dtype=np.float32)
    m = sel_logits.max()
    ex = np.exp(sel_logits - m)
    sm = ex / ex.sum()
    for j in range(K):
        expert_ids[t, j] = sel[j]
        gating[t, j] = sm[j]

# ── stage 2: grouped expert FFN ───────────────────────────────────────────────
# Each routed slot (t, j) runs token t's hidden vec through expert
# expert_ids[t,j]'s SwiGLU FFN. expert_out is [T*K, H], token-major.
expert_out = np.zeros((T * K, H), dtype=np.float32)
for t in range(T):
    for j in range(K):
        e = int(expert_ids[t, j])
        x = tokens[t]                              # [H]
        g = x @ gate_w[e].T                        # [F]
        u = x @ up_w[e].T                          # [F]
        h = silu(g) * u                            # [F]
        y = h @ down_w[e].T                        # [H]
        expert_out[t * K + j] = y

# ── stage 3: gated scatter-add ────────────────────────────────────────────────
# accum[indices[s]] += expert_out[s] * gating_flat[s], in-place F32.
# indices[s] = token id for slot s (token-major: slot t*K+j -> token t).
# Both of a token's K slots collide on the same accum row -> the gated combine.
indices = np.zeros(T * K, dtype=np.int64)
gating_flat = np.zeros(T * K, dtype=np.float32)
for t in range(T):
    for j in range(K):
        indices[t * K + j] = t
        gating_flat[t * K + j] = gating[t, j]

accum = np.zeros((T, H), dtype=np.float32)
for s in range(T * K):
    accum[indices[s]] += expert_out[s] * gating_flat[s]


def _fn_f32(name, arr):
    """Emit a Mojo `def name() -> List[Float32]` returning the flat array."""
    flat = np.asarray(arr, dtype=np.float32).reshape(-1).tolist()
    lines = [f"def {name}() -> List[Float32]:", "    var v = List[Float32]()"]
    for x in flat:
        lines.append(f"    v.append(Float32({x:.8f}))")
    lines.append("    return v^")
    return "\n".join(lines)


def _fn_int(name, arr):
    """Emit a Mojo `def name() -> List[Int]` returning the flat array."""
    flat = np.asarray(arr, dtype=np.int64).reshape(-1).tolist()
    lines = [f"def {name}() -> List[Int]:", "    var v = List[Int]()"]
    for x in flat:
        lines.append(f"    v.append({int(x)})")
    lines.append("    return v^")
    return "\n".join(lines)


HEADER = f"""# moe_ref_data.mojo — GENERATED by gen_moe_reference.py. DO NOT EDIT BY HAND.
#
# DEV-ONLY parity fixture for serenitymojo/ops/moe_smoke.mojo. Holds the fixed
# inputs and the numpy reference outputs (per stage + final) for the MoE op
# parity gate. Regenerate with:
#   pixi run python serenitymojo/ops/parity/gen_moe_reference.py > \\
#       serenitymojo/ops/parity/moe_ref_data.mojo
# No Python at runtime: this is plain Mojo data, the numpy is only the oracle.
#
# Config: T={T} E={E} K={K} H={H} F={F}  seed={SEED}

comptime MOE_T = {T}
comptime MOE_E = {E}
comptime MOE_K = {K}
comptime MOE_H = {H}
comptime MOE_F = {F}
"""


if __name__ == "__main__":
    parts = [HEADER, ""]
    parts.append(_fn_f32("tokens", tokens))
    parts.append(_fn_f32("logits", logits))
    parts.append(_fn_f32("gate_w", gate_w))
    parts.append(_fn_f32("up_w", up_w))
    parts.append(_fn_f32("down_w", down_w))
    parts.append(_fn_int("ref_expert_ids", expert_ids))
    parts.append(_fn_f32("ref_gating", gating))
    parts.append(_fn_f32("ref_expert_out", expert_out))
    parts.append(_fn_f32("ref_accum", accum))
    print("\n\n".join(parts))
