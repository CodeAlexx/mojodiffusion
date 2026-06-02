# LTX-2 DiT block-0 real-weight smoke (video self-attn + text cross-attn + FFN,
# video-only path).
#
# Loads block-0 attn1 + attn2 + ff + scale_shift_table + prompt_scale_shift_table
# from the LTX-2 22B checkpoint, builds 3D split-RoPE tables, feeds a synthetic
# BF16 hidden + 9-param timestep embedding + text context through
# ltx2_block_forward_video_only, and asserts the output is finite with sane
# stats. The full block-0 forward now runs: gated self-attn -> gated text
# cross-attn (attn2) -> FFN.
#
# Bounded comptime sizes for fast compile: F=4, H=8, W=8 -> S=256 tokens.
# Production hidden/heads/head_dim come from LTX2Config.ltx2()
# (inner_dim 4096, heads 32, head_dim 128, FFN 16384).
#
# *** CODE-ONLY: compile-verified; NOT executed (no GPU run). ***

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import mul_scalar
from serenitymojo.models.dit.ltx2_dit import (
    LTX2Config,
    LTX2BlockWeights,
    ltx2_block_forward_video_only,
)
from serenitymojo.models.dit.ltx2_rope import build_ltx2_rope


comptime CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev.safetensors"
comptime SEED = UInt64(20260528)
comptime INPUT_SCALE = Float32(0.1)

comptime F = 4
comptime H_LAT = 8
comptime W_LAT = 8
comptime S = F * H_LAT * W_LAT   # 256
comptime N_TXT = 32              # synthetic text-context length for attn2


def _stats(name: String, t: Tensor, ctx: DeviceContext, max_abs: Float64) raises:
    var h = t.to_host(ctx)
    if len(h) == 0:
        raise Error(String("empty tensor: ") + name)
    var s = 0.0
    var s2 = 0.0
    var amax = 0.0
    for i in range(len(h)):
        var v = Float64(h[i])
        if v != v:
            raise Error(String("NaN in ") + name)
        if v > max_abs or v < -max_abs:
            raise Error(String("unstable value in ") + name)
        s += v
        s2 += v * v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    var mean = s / Float64(len(h))
    var var_ = s2 / Float64(len(h)) - mean * mean
    if var_ < 0.0:
        var_ = 0.0
    print("  ", name, "mean/std/absmax:", Float32(mean), Float32(sqrt(var_)), Float32(amax))


def _require_shape3(label: String, got: List[Int], a: Int, b: Int, c: Int) raises:
    if len(got) != 3 or got[0] != a or got[1] != b or got[2] != c:
        raise Error(String("shape mismatch for ") + label)


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    var dim = cfg.inner_dim

    print("=== LTX-2 DiT block-0 real-weight smoke (video-only) ===")
    print("  F/H/W/S:", F, H_LAT, W_LAT, S)
    print("  heads/head_dim/inner_dim/ffn:", cfg.num_heads, cfg.head_dim, dim, cfg.ffn_hidden)

    print("  [load] block-0 attn1 + attn2 + ff + scale_shift_table")
    var weights = LTX2BlockWeights.load(String(CKPT), 0, cfg, ctx)
    print("  [load] done; has_gate:", weights.has_gate, " has_gate2:", weights.has_gate2)

    # synthetic BF16 hidden [1, S, dim]
    var hidden_sh = List[Int]()
    hidden_sh.append(1)
    hidden_sh.append(S)
    hidden_sh.append(dim)
    var hidden = mul_scalar(randn(hidden_sh^, SEED, STDtype.BF16, ctx), INPUT_SCALE, ctx)
    _require_shape3(String("hidden"), hidden.shape(), 1, S, dim)
    _stats(String("hidden_in"), hidden, ctx, 64.0)

    # synthetic timestep embedding temb [1, 9*dim] (rows 0-5 self-attn+FFN,
    # rows 6-8 cross-attn query modulation — the ComfyUI 9-param path).
    var temb_sh = List[Int]()
    temb_sh.append(1)
    temb_sh.append(9 * dim)
    var temb = mul_scalar(randn(temb_sh^, SEED + 1, STDtype.BF16, ctx), INPUT_SCALE, ctx)
    _stats(String("temb_in"), temb, ctx, 64.0)

    # synthetic text encoder context [1, N_TXT, dim] for attn2 cross-attn.
    var ctx_sh = List[Int]()
    ctx_sh.append(1)
    ctx_sh.append(N_TXT)
    ctx_sh.append(dim)
    var context = mul_scalar(randn(ctx_sh^, SEED + 2, STDtype.BF16, ctx), INPUT_SCALE, ctx)
    _stats(String("context_in"), context, ctx, 64.0)

    print("  [rope] build 3D split-RoPE tables")
    # max_positions per axis = axis extent (keeps grid fractional in [0,1)).
    var rope = build_ltx2_rope[F, H_LAT, W_LAT](
        cfg.num_heads,
        cfg.head_dim,
        cfg.rope_theta,
        Float64(F),
        Float64(H_LAT),
        Float64(W_LAT),
        STDtype.BF16,
        ctx,
    )
    if rope[0].shape()[0] != S * cfg.num_heads or rope[0].shape()[1] != cfg.head_dim // 2:
        raise Error("rope_cos shape mismatch")

    print("  [block0] forward (self-attn gated + attn2 cross-attn + FFN)")
    var out = ltx2_block_forward_video_only[1, S, N_TXT](
        weights,
        hidden,
        temb,
        context,
        rope[0],
        rope[1],
        cfg.num_heads,
        cfg.head_dim,
        cfg.eps,
        ctx,
    )
    _require_shape3(String("block0_out"), out.shape(), 1, S, dim)
    _stats(String("block0_out"), out, ctx, 65536.0)
    print("LTX-2 DiT block-0 smoke PASS")
