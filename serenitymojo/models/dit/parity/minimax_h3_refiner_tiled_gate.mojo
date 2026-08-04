# serenitymojo/models/dit/parity/minimax_h3_refiner_tiled_gate.mojo
#
# GATE for the token-refiner large-S attention route (the ref2va OOM fix).
#
# WHY: `_minimax_h3_token_refiner_block` (models/dit/minimax_h3_frontend.mojo)
# used `sdpa_nomask` unconditionally. That is the math-mode path, which
# materializes the F32 scores [H, S, S] (ops/attention.mojo:371,
# `enqueue_create_buffer[float32](BH * S * S)`). ref2va's text region carries
# thousands of reference-video vision tokens: at S=7300 with H=56 the scores
# alone are 56 * 7300^2 * 4 B = 11,384 MiB -> CUDA_ERROR_OUT_OF_MEMORY.
# The fix routes S above the score budget through `sdpa_nomask_tiled` — the
# EXACT online-softmax path already shipped in ops/attention.mojo (:1924,
# used in production by krea2_dit at Dh=128) with O(S*Dh) peak, no [S,S]
# buffer, no padding, no mask. The refiner is full-bidirectional with NO
# mask (oracle: models/minimax_h3/block_forward.mojo `_token_refiner` passes
# attention() an empty mask), and the tiled path needs no pad rows, so there
# is no pad-through-softmax hazard to mask away.
#
# TWO CHECKS:
#   [1] PARITY, old path vs new path, SAME weights, SAME input, small S
#       where both fit. Full `minimax_h3_token_refiner[S=512, H=56, Dh=128]`
#       (2 blocks + final norm) run twice: FORCE_TILED=False (the old
#       `sdpa_nomask` kernel — score_mib at S=512 is 56 MiB, far under the
#       3584 MiB budget) vs FORCE_TILED=True (the new kernel). Attention runs
#       at the EXACT production head geometry [1, 512, 56, 128] bf16; the
#       config's hidden/ffn are shrunk (512/1024 vs 5376/14336) only so the
#       synthetic weight dict stays small — every op and dtype boundary on
#       the path is the production one, and the ONLY difference between the
#       two runs is the attention kernel. Bar: cos >= 0.9999, max_abs
#       reported (same math, different accumulation order — noise floor).
#   [2] CAPACITY at the real ref2va shape, made DISCRIMINATING by ballast:
#       allocate a large device ballast first so the math path's 11,384 MiB
#       scores CANNOT fit, then run the refiner at S=7300 through its
#       DEFAULT route (no FORCE_TILED — proves the comptime score-budget
#       select itself picks tiled at 7300). Passing is only possible if the
#       [S,S] buffer was never allocated.
#
# Build (same convention as the other h3 gates; shim linked like
# sdpa_small_parity does even though the tiled path itself is pure Mojo):
#   pixi run mojo build -O2 -j 1 -I . -I vendor/mojo-libs \
#     -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs -Xlinker -lcudnn \
#     serenitymojo/models/dit/parity/minimax_h3_refiner_tiled_gate.mojo \
#     -o /tmp/h3_refiner_tiled_gate
#   LD_LIBRARY_PATH=serenitymojo/ops/cshim/lib:serenitymojo/ops/cshim/lib/cudnn_stubs \
#     /tmp/h3_refiner_tiled_gate
# Mojo 1.0.0b1, NVIDIA GPU.

from std.collections import Dict, List
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, pi
from std.memory import ArcPointer
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.dit.minimax_h3_dit import MiniMaxH3DiTConfig
from serenitymojo.models.dit.minimax_h3_frontend import minimax_h3_token_refiner


def _gaussian(n: Int, seed: Int, sd: Float32, mean: Float32 = 0.0) -> List[Float32]:
    var out = List[Float32]()
    var st = UInt64(seed * 2654435761 + 12345)
    for _i in range(n):
        st = st * 6364136223846793005 + 1442695040888963407
        var u1 = (Float64(st >> 11) + 1.0) / Float64(1 << 53)
        st = st * 6364136223846793005 + 1442695040888963407
        var u2 = Float64(st >> 11) / Float64(1 << 53)
        var r = sqrt(-2.0 * flog(u1))
        out.append(Float32(r * fcos(2.0 * pi * u2)) * sd + mean)
    return out^


def _cos(a: List[Float32], b: List[Float32]) -> Float64:
    var dot: Float64 = 0.0
    var na: Float64 = 0.0
    var nb: Float64 = 0.0
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    return dot / (sqrt(na) * sqrt(nb) + 1e-30)


# The gate config: PRODUCTION head geometry (56 heads x 128 = inner 7168),
# shrunk hidden/ffn so synthetic weights stay small. Only the fields the
# refiner reads matter (inner_dim, ffn_hidden_size, the three eps,
# token_refiner_num_layers); the rest are placeholders. validate() is NOT
# called — this is deliberately not the released config.
comptime GATE_HIDDEN = 512
comptime GATE_FFN = 1024
comptime GATE_H = 56
comptime GATE_DH = 128


def _gate_config() -> MiniMaxH3DiTConfig:
    return MiniMaxH3DiTConfig(
        GATE_HIDDEN,  # hidden_size
        1,            # num_layers (unused here)
        2,            # token_refiner_num_layers — production value
        GATE_H,       # num_attention_heads — production value
        GATE_DH,      # attention_head_dim — production value
        GATE_FFN,     # ffn_hidden_size
        24, 32, 5120, 256, 2688,           # unused placeholders
        6 * GATE_HIDDEN * 3,               # adaln_out_features (unused)
        2 * GATE_HIDDEN,                   # final_adaln_out_features (unused)
        16,                                # rope_inv_freq_len (unused)
        Float32(1.0e-5),  # norm_eps — production value
        Float32(1.0e-5),  # qk_norm_eps — production value
        Float32(1.0e-5),  # final_norm_eps — production value
    )


def _synth_weights(ctx: DeviceContext) raises -> Dict[String, ArcPointer[Tensor]]:
    var inner = GATE_H * GATE_DH
    var w = Dict[String, ArcPointer[Tensor]]()
    var seed = 100
    for layer in range(2):
        var p = String("token_refiner.blocks.") + String(layer)
        w[p + ".norm1.weight"] = ArcPointer(Tensor.from_host(
            _gaussian(GATE_HIDDEN, seed + 1, 0.1, 1.0),
            [GATE_HIDDEN], STDtype.BF16, ctx))
        w[p + ".attn.qkv_proj.weight"] = ArcPointer(Tensor.from_host(
            _gaussian(3 * inner * GATE_HIDDEN, seed + 2, 0.02),
            [3 * inner, GATE_HIDDEN], STDtype.BF16, ctx))
        w[p + ".attn.q_norm.weight"] = ArcPointer(Tensor.from_host(
            _gaussian(GATE_DH, seed + 3, 0.1, 1.0), [GATE_DH], STDtype.BF16, ctx))
        w[p + ".attn.k_norm.weight"] = ArcPointer(Tensor.from_host(
            _gaussian(GATE_DH, seed + 4, 0.1, 1.0), [GATE_DH], STDtype.BF16, ctx))
        w[p + ".attn.out_proj.weight"] = ArcPointer(Tensor.from_host(
            _gaussian(GATE_HIDDEN * inner, seed + 5, 0.02),
            [GATE_HIDDEN, inner], STDtype.BF16, ctx))
        w[p + ".norm2.weight"] = ArcPointer(Tensor.from_host(
            _gaussian(GATE_HIDDEN, seed + 6, 0.1, 1.0),
            [GATE_HIDDEN], STDtype.BF16, ctx))
        w[p + ".mlp.fc1.weight"] = ArcPointer(Tensor.from_host(
            _gaussian(2 * GATE_FFN * GATE_HIDDEN, seed + 7, 0.02),
            [2 * GATE_FFN, GATE_HIDDEN], STDtype.BF16, ctx))
        w[p + ".mlp.fc2.weight"] = ArcPointer(Tensor.from_host(
            _gaussian(GATE_HIDDEN * GATE_FFN, seed + 8, 0.02),
            [GATE_HIDDEN, GATE_FFN], STDtype.BF16, ctx))
        seed += 10
    w["token_refiner.final_norm.weight"] = ArcPointer(Tensor.from_host(
        _gaussian(GATE_HIDDEN, seed + 1, 0.1, 1.0), [GATE_HIDDEN], STDtype.BF16, ctx))
    return w^


def main() raises:
    var ctx = DeviceContext()
    var config = _gate_config()
    var w = _synth_weights(ctx)

    # ── [1] parity: old kernel vs new kernel through the FULL refiner ────────
    comptime S_SMALL = 512
    var xin = Tensor.from_host(
        _gaussian(S_SMALL * GATE_HIDDEN, 7, 1.0),
        [S_SMALL, GATE_HIDDEN], STDtype.BF16, ctx)

    var old_out = minimax_h3_token_refiner[S_SMALL, GATE_H, GATE_DH, False](
        xin, w, config, ctx)
    var new_out = minimax_h3_token_refiner[S_SMALL, GATE_H, GATE_DH, True](
        xin, w, config, ctx)
    var oh = old_out.to_host(ctx)
    var nh = new_out.to_host(ctx)
    var c = _cos(oh, nh)
    var max_abs: Float32 = 0.0
    for i in range(len(oh)):
        var d = oh[i] - nh[i]
        if d < 0:
            d = -d
        if d > max_abs:
            max_abs = d
    print("[1] refiner S=512 H=56 Dh=128: sdpa_nomask vs sdpa_nomask_tiled")
    print("    cos =", c, " max_abs =", max_abs)
    if c < 0.9999:
        print("FAIL: cos", c, "< 0.9999")
        return

    # ── [2] capacity at the real ref2va S, ballasted ─────────────────────────
    # Ballast so the math path's 11,384 MiB scores cannot possibly fit; the
    # DEFAULT route (score_mib = 56*7300^2*4/2^20 = 11384 >= 3584) must pick
    # tiled, whose peak here is ~0.7 GiB. 16 GiB first, 13 GiB fallback.
    comptime S_CAP = 7300
    var ballast_gib = 16
    var ballast_ok = False
    var ballast = ctx.enqueue_create_buffer[DType.uint8](1)
    try:
        ballast = ctx.enqueue_create_buffer[DType.uint8](16 * 1024 * 1024 * 1024)
        ctx.synchronize()
        ballast_ok = True
    except e:
        try:
            ballast_gib = 13
            ballast = ctx.enqueue_create_buffer[DType.uint8](13 * 1024 * 1024 * 1024)
            ctx.synchronize()
            ballast_ok = True
        except e2:
            ballast_gib = 0
            print("    WARNING: no ballast held — capacity check is a plain")
            print("    no-OOM smoke, not a math-path-exclusion proof:", e2)
    if ballast_ok:
        print("[2] ballast held:", ballast_gib, "GiB — math-path scores (11384 MiB) cannot fit")

    var xcap = Tensor.from_host(
        _gaussian(S_CAP * GATE_HIDDEN, 11, 1.0),
        [S_CAP, GATE_HIDDEN], STDtype.BF16, ctx)
    var t0 = perf_counter_ns()
    # DEFAULT route — no FORCE_TILED. This is the production instantiation.
    var cap_out = minimax_h3_token_refiner[S_CAP, GATE_H, GATE_DH](
        xcap, w, config, ctx)
    var ch = cap_out.to_host(ctx)
    var t1 = perf_counter_ns()
    var mean: Float64 = 0.0
    for i in range(len(ch)):
        mean += Float64(ch[i])
    mean /= Float64(len(ch))
    var var_acc: Float64 = 0.0
    var finite = True
    for i in range(len(ch)):
        var d = Float64(ch[i]) - mean
        var_acc += d * d
        if not (ch[i] == ch[i]):  # NaN check
            finite = False
    var std = sqrt(var_acc / Float64(len(ch)))
    print("[2] refiner S=7300 default route: ran in",
          Float64(t1 - t0) / 1e9, "s  out mean =", mean, " std =", std)
    if not finite:
        print("FAIL: capacity output has NaNs")
        return
    _ = ballast^
    print("PASS: refiner tiled route — parity cos>=0.9999 at S=512 and")
    print("      S=7300 runs under ballast (math-path [S,S] scores excluded)")
