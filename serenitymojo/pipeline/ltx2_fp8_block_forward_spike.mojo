# pipeline/ltx2_fp8_block_forward_spike.mojo — P2 FP8 block-forward feasibility spike.
#
# LTX2_PORT_PLAN_2026-05-28 §P2 (de-risk #2). End-to-end proof, on a REAL GPU,
# that one DiT block of ltx-2.3-22b-distilled-fp8.safetensors can be:
#   (1) STREAMED from disk via the existing offload loader (LTX2BlockStream),
#   (2) DEQUANTED FP8 E4M3 -> BF16 on-use (ops/fp8.mojo, bit-exact vs torch —
#       gated separately by pipeline/fp8_dequant_smoke.mojo),
#   (3) run through the REAL block forward (ltx2_block_forward_video_only:
#       gated self-attn + text cross-attn + FFN) producing a FINITE,
#       correctly-shaped [1, S, 4096] output,
#   (4) all while PEAK VRAM stays under the 24 GB ceiling.
#
# Unlike ltx2_stream_ceiling_smoke (a proxy forward over all 48 blocks), this
# spike runs the ACTUAL block forward on ONE real FP8 block (block 4 — the first
# block whose attn/FFN weights are float8_e4m3fn). That is the feasibility gate:
# a real dequant feeding the real attention/FFN math, finite out, bounded VRAM.
#
# Build:
#   pixi run mojo build -I . -Xlinker -lm \
#     serenitymojo/pipeline/ltx2_fp8_block_forward_spike.mojo \
#     -o /tmp/ltx2_fp8_block_forward_spike
#
# GATE (P2 spike, HARD RULE — real GPU, never compile-only):
#   - block forward completes on GPU, exit 0
#   - output shape == [1, S, 4096], ALL elements finite
#   - the streamed block actually carried FP8 weights (n_fp8 > 0) → dequant path
#     was exercised
#   - peak resident VRAM < 24000 MiB (target ~16 GB per the proven path)
#
# HEAVY GPU. Peak VRAM is also cross-checked via nvidia-smi by the wrapper that
# launches this binary.

from std.gpu.host import DeviceContext
from std.math import sqrt, isfinite

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import mul_scalar
from serenitymojo.offload.ltx2_block_stream import LTX2BlockStream, drop_block
from serenitymojo.models.dit.ltx2_dit import (
    LTX2Config,
    LTX2BlockWeights,
    ltx2_block_forward_video_only,
)
from serenitymojo.models.dit.ltx2_rope import build_ltx2_rope


comptime CKPT = String(
    "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-distilled-fp8.safetensors"
)
comptime MIB = 1024.0 * 1024.0
comptime SEED = UInt64(20260528)
comptime INPUT_SCALE = Float32(0.1)

# MVP-shape video token sequence (matches the block-0 smoke: F=4,H=8,W=8 → 256).
comptime F = 4
comptime H_LAT = 8
comptime W_LAT = 8
comptime S = F * H_LAT * W_LAT   # 256
comptime N_TXT = 32              # synthetic text-context length for attn2

# Block index to spike: first block whose attn/FFN weights are F8_E4M3.
# (Blocks 0,1,46,47 are BF16; 4 is the first all-FP8 block — verified from the
#  checkpoint header.)
comptime SPIKE_BLOCK = 4


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    var dim = cfg.inner_dim

    print("=== LTX-2 FP8 block-forward feasibility spike (P2) ===")
    print("  ckpt:", CKPT)
    print("  block:", SPIKE_BLOCK, " F/H/W/S:", F, H_LAT, W_LAT, S)
    print(
        "  heads/head_dim/inner_dim/ffn:",
        cfg.num_heads, cfg.head_dim, dim, cfg.ffn_hidden,
    )

    var mi0 = ctx.get_memory_info()
    var total_vram = Int(mi0[1])
    var free_start = Int(mi0[0])
    var min_free = free_start
    print(
        "  [vram] total:", Int(Float64(total_vram) / MIB),
        "MiB  free at start:", Int(Float64(free_start) / MIB), "MiB",
    )

    # ── 1. open the FP8 stream + 2. stream ONE block (dequant FP8→BF16 on-use) ──
    var stream = LTX2BlockStream.open(CKPT)
    print("  [stream] checkpoint blocks:", stream.block_count())
    var n_fp8 = stream.fp8_tensor_count(SPIKE_BLOCK)
    print("  [stream] block", SPIKE_BLOCK, "FP8 weight tensors:", n_fp8)
    if n_fp8 <= 0:
        raise Error(
            "spike block has no FP8 weights — dequant path would not be exercised"
        )

    var block = stream.load_block_bf16(SPIKE_BLOCK, ctx)
    var mi_loaded = Int(ctx.get_memory_info()[0])
    if mi_loaded < min_free:
        min_free = mi_loaded
    print(
        "  [stream] block loaded+dequanted; resident:",
        Int(Float64(total_vram - mi_loaded) / MIB), "MiB",
    )

    # ── build LTX2BlockWeights from the streamed (already-BF16) block ──
    var weights = LTX2BlockWeights.from_fp8_block(block^, cfg, ctx)
    print("  [weights] has_gate:", weights.has_gate, " has_gate2:", weights.has_gate2)

    # ── synthetic inputs (same construction as the block-0 smoke) ──
    var hidden_sh = List[Int]()
    hidden_sh.append(1)
    hidden_sh.append(S)
    hidden_sh.append(dim)
    var hidden = mul_scalar(randn(hidden_sh^, SEED, STDtype.BF16, ctx), INPUT_SCALE, ctx)

    var temb_sh = List[Int]()
    temb_sh.append(1)
    temb_sh.append(9 * dim)
    var temb = mul_scalar(randn(temb_sh^, SEED + 1, STDtype.BF16, ctx), INPUT_SCALE, ctx)

    var ctx_sh = List[Int]()
    ctx_sh.append(1)
    ctx_sh.append(N_TXT)
    ctx_sh.append(dim)
    var context = mul_scalar(randn(ctx_sh^, SEED + 2, STDtype.BF16, ctx), INPUT_SCALE, ctx)

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

    # ── 3. REAL block forward (gated self-attn + cross-attn + FFN) ──
    print("  [forward] running real ltx2_block_forward_video_only ...")
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
    ctx.synchronize()

    var mi_fwd = Int(ctx.get_memory_info()[0])
    if mi_fwd < min_free:
        min_free = mi_fwd

    # ── 4. finiteness + shape + peak VRAM ──
    var osh = out.shape()
    var shape_ok = len(osh) == 3 and osh[0] == 1 and osh[1] == S and osh[2] == dim
    var h = out.to_host(ctx)
    var n_finite = 0
    var amax = 0.0
    var s_sum = 0.0
    for i in range(len(h)):
        var v = Float64(h[i])
        if isfinite(h[i]):
            n_finite += 1
        s_sum += v
        var av = v if v >= 0.0 else -v
        if av > amax:
            amax = av
    var all_finite = (n_finite == len(h)) and (len(h) > 0)
    var mean = s_sum / Float64(len(h)) if len(h) > 0 else 0.0

    var peak_mib = Int(Float64(total_vram - min_free) / MIB)

    print("──────────────────────────────")
    print("  [out] shape ok:", shape_ok, " numel:", out.numel())
    print(
        "  [out] finite:", all_finite, " (", n_finite, "/", len(h), ")",
        " mean:", Float32(mean), " absmax:", Float32(amax),
    )
    print("  [vram] PEAK resident:", peak_mib, "MiB  (ceiling 24000)")

    _ = out^

    # ── GATE ──
    var gate = shape_ok and all_finite and (n_fp8 > 0) and (peak_mib < 24000)
    if gate:
        print(
            "P2 SPIKE GATE PASS: 1 FP8 block streamed+dequanted, real forward",
            "finite [1,", S, ",", dim, "], peak", peak_mib, "MiB < 24000.",
        )
    else:
        print(
            "P2 SPIKE GATE FAIL: shape_ok", shape_ok, " finite", all_finite,
            " n_fp8", n_fp8, " peak", peak_mib, "MiB",
        )
        raise Error("P2 FP8 block-forward spike gate FAILED")
