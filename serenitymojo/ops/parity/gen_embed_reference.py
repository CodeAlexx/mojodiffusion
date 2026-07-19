#!/usr/bin/env python3
# gen_embed_reference.py — DEV-ONLY numpy oracle for the embeddings op parity smoke.
#
# NOT in the runtime path. Run with `pixi run python` to emit the SAME inputs
# the Mojo smoke driver (serenitymojo/ops/embed_smoke.mojo) inlines, plus the
# expected reference outputs. The Mojo driver inlines both the inputs and the
# reference numbers printed here; this script is how those numbers were produced
# and how they can be regenerated/verified. No Python at runtime.
#
# Mirrors the EXACT formulas in the Z-Image NextDiT reference:
#   /home/alex/EriDiffusion/inference-flame/src/models/zimage_nextdit.rs
#
#   timestep_embed (lines 411-440):
#     half = freq_dim/2; max_period = 10000
#     freq_i = exp(-ln(max_period) * i / half)          for i in [0, half)
#     angle  = t * freq_i
#     emb[ 0 : half ]        = cos(angle)               (COS FIRST)
#     emb[ half : freq_dim ] = sin(angle)               (SIN SECOND)
#     -> Linear(mlp.0) -> SiLU -> Linear(mlp.2)
#
#   build_3d_rope freqs (lines 693-709), single-axis form per the builder task:
#     half = head_dim/2
#     inv_freq_i = 1 / theta^(i / half)                 for i in [0, half)
#     angle      = position * inv_freq_i
#     cos[pos, i] = cos(angle); sin[pos, i] = sin(angle)
#     cos/sin tables shape [num_positions, head_dim/2]  (what rope_halfsplit eats)
#
#   rope_halfsplit (serenitymojo/ops/rope.mojo lines 100-118):
#     pair = (x[i], x[i + D/2]); angle index i in [0, D/2)
#     out[i]       = x[i]*cos[i] - x[i+D/2]*sin[i]
#     out[i+D/2]   = x[i+D/2]*cos[i] + x[i]*sin[i]
#
# Regenerate with:
#   pixi run python serenitymojo/ops/parity/gen_embed_reference.py > \
#       serenitymojo/ops/parity/embed_ref_data.mojo

import numpy as np

SEED = 4242
np.random.seed(SEED)

# ── timestep_embedding config ────────────────────────────────────────────────
N = 3                 # number of timesteps in the batch
DIM = 16              # embedding dim (freq_dim); must be even
MAX_PERIOD = 10000.0
T_VALS = np.array([0.5, 17.0, 999.0], dtype=np.float32)   # arbitrary timesteps

# ── t_embedder MLP config ────────────────────────────────────────────────────
HIDDEN = 24           # MLP hidden width
# weights: mlp.0 [HIDDEN, DIM], mlp.2 [DIM, HIDDEN]  (PyTorch row-major [out,in])
mlp0_w = (np.random.randn(HIDDEN, DIM).astype(np.float32) * 0.2)
mlp0_b = (np.random.randn(HIDDEN).astype(np.float32) * 0.1)
mlp2_w = (np.random.randn(DIM, HIDDEN).astype(np.float32) * 0.2)
mlp2_b = (np.random.randn(DIM).astype(np.float32) * 0.1)

# ── rope-tables config ───────────────────────────────────────────────────────
HEAD_DIM = 8          # rope head_dim; must be even -> half = 4
THETA = 256.0         # Z-Image rope_theta
POSITIONS = np.array([1.0, 2.0, 5.0, 11.0], dtype=np.float32)   # 4 positions
ROPE_ROWS = len(POSITIONS)


def timestep_embedding(t, dim, max_period=10000.0):
    half = dim // 2
    i = np.arange(half, dtype=np.float32)
    freqs = np.exp(-np.log(max_period) * i / half)        # [half]
    angles = t[:, None] * freqs[None, :]                  # [N, half]
    # COS first, then SIN — matches zimage_nextdit.rs:425-426
    emb = np.concatenate([np.cos(angles), np.sin(angles)], axis=1)  # [N, dim]
    return emb.astype(np.float32)


def silu(x):
    return x / (1.0 + np.exp(-x))


def t_embedder(t, dim, max_period=10000.0):
    emb = timestep_embedding(t, dim, max_period)          # [N, dim]
    h = emb @ mlp0_w.T + mlp0_b                            # [N, HIDDEN]
    h = silu(h)
    out = h @ mlp2_w.T + mlp2_b                            # [N, dim]
    return out.astype(np.float32)


def build_rope_tables(positions, head_dim, theta):
    half = head_dim // 2
    i = np.arange(half, dtype=np.float32)
    inv_freq = 1.0 / (theta ** (i / half))                # [half]  == 1/theta^(i/half)
    angles = positions[:, None] * inv_freq[None, :]       # [rows, half]
    cos = np.cos(angles).astype(np.float32)
    sin = np.sin(angles).astype(np.float32)
    return cos, sin


def rope_halfsplit(x, cos, sin):
    # x: [rows, D]; cos/sin: [rows, D/2]
    rows, d = x.shape
    half = d // 2
    out = np.empty_like(x)
    x0 = x[:, :half]
    x1 = x[:, half:]
    out[:, :half] = x0 * cos - x1 * sin
    out[:, half:] = x1 * cos + x0 * sin
    return out.astype(np.float32)


def fmt(arr):
    flat = np.asarray(arr, dtype=np.float32).reshape(-1)
    lines = []
    for v in flat:
        lines.append("    v.append(Float32({:.8f}))".format(float(v)))
    return "\n".join(lines)


def emit_fn(name, arr):
    print("\ndef {}() -> List[Float32]:".format(name))
    print("    var v = List[Float32]()")
    print(fmt(arr))
    print("    return v^")


# ── compute references ────────────────────────────────────────────────────────
ts_emb = timestep_embedding(T_VALS, DIM, MAX_PERIOD)       # [N, DIM]
te_out = t_embedder(T_VALS, DIM, MAX_PERIOD)               # [N, DIM]

cos_tab, sin_tab = build_rope_tables(POSITIONS, HEAD_DIM, THETA)  # [rows, HEAD_DIM/2]

# rope-through input x: [ROPE_ROWS, HEAD_DIM]
rope_x = (np.random.randn(ROPE_ROWS, HEAD_DIM).astype(np.float32))
rope_out = rope_halfsplit(rope_x, cos_tab, sin_tab)        # [rows, HEAD_DIM]


# ── emit Mojo data module ──────────────────────────────────────────────────────
print("# embed_ref_data.mojo — GENERATED by gen_embed_reference.py. DO NOT EDIT BY HAND.")
print("#")
print("# DEV-ONLY parity fixture for serenitymojo/ops/embed_smoke.mojo. Holds the")
print("# fixed inputs and the numpy reference outputs for the embeddings op parity")
print("# gate. Regenerate with:")
print("#   pixi run python serenitymojo/ops/parity/gen_embed_reference.py > \\")
print("#       serenitymojo/ops/parity/embed_ref_data.mojo")
print("# No Python at runtime: this is plain Mojo data, the numpy is only the oracle.")
print("#")
print("# Config: N={} DIM={} HIDDEN={} MAX_PERIOD={} HEAD_DIM={} THETA={} ROPE_ROWS={} seed={}".format(
    N, DIM, HIDDEN, MAX_PERIOD, HEAD_DIM, THETA, ROPE_ROWS, SEED))
print()
print("comptime EMB_N = {}".format(N))
print("comptime EMB_DIM = {}".format(DIM))
print("comptime EMB_HIDDEN = {}".format(HIDDEN))
print("comptime EMB_MAX_PERIOD = Float32({})".format(MAX_PERIOD))
print("comptime ROPE_HEAD_DIM = {}".format(HEAD_DIM))
print("comptime ROPE_THETA = Float32({})".format(THETA))
print("comptime ROPE_ROWS = {}".format(ROPE_ROWS))

emit_fn("t_vals", T_VALS)
emit_fn("ts_emb_ref", ts_emb)

emit_fn("mlp0_w", mlp0_w)
emit_fn("mlp0_b", mlp0_b)
emit_fn("mlp2_w", mlp2_w)
emit_fn("mlp2_b", mlp2_b)
emit_fn("t_embedder_ref", te_out)

emit_fn("positions", POSITIONS)
emit_fn("cos_tab_ref", cos_tab)
emit_fn("sin_tab_ref", sin_tab)

emit_fn("rope_x", rope_x)
emit_fn("rope_out_ref", rope_out)
