# serenitymojo/models/ltx2/parity/ltx2_video_stack_real_smoke.mojo
#
# LTX-2.3 VIDEO-MODE STACK real-depth FIT + FINITENESS smoke (the 24 GB proof).
#
# Runs the FULL 48-block video-mode training stack at milestone-1 geometry
# (latents [128,4,9,16] -> S_V=576, N_TXT=1024) with the REAL head, 48 blocks
# STREAMED from the dev-fp8 checkpoint (per-tensor fp8->bf16 dequant, then upcast
# to F32), and the frozen torchref tail — one forward + one backward with 384
# attached LoRA adapters (8/block, both A and B seeded nonzero so d_A and d_B are
# exercised).
#
# DTYPE: F32 end-to-end. The frozen per-block backward arm
# (ltx2_video_backward.mojo) routes d_x through linear_backward_dx, which always
# emits F32, so the stack backward is an F32 carrier. Running the whole 48-block
# fwd+bwd in F32 is a STRICTLY STRONGER 24 GB fit proof than BF16: F32 doubles
# every activation/weight, so F32 fitting implies the BF16 production path fits.
# (The BF16 forward path was separately observed finite — the AV MVP spine runs
# the same BF16-capable video forward and generates video.)
#
# Loads a REAL torchref cache sample (latent + POST-connector video_prompt_embeds).
# The loss proxy is 0.5*sum(pred^2) (d_pred = pred), which drives a full backward.
#
# Asserts: finite pred, finite d_input, finite + NONZERO LoRA grads, correct
# unpatchified velocity shape, and completion without OOM (the 24 GB fit proof).
# Reports resident saved-input bytes + wall time per fwd / bwd.
#
# Run (capture peak VRAM externally, e.g. nvidia-smi polling):
#   rm -f serenitymojo.mojopkg
#   pixi run mojo build -O2 -I . \
#       serenitymojo/models/ltx2/parity/ltx2_video_stack_real_smoke.mojo \
#       -o /tmp/ltx2_video_stack_real_smoke && /tmp/ltx2_video_stack_real_smoke

from max.gpu.host import DeviceContext
from std.collections import List
from std.memory import ArcPointer
from std.math import sqrt
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.random import randn
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape, mul_scalar
from serenitymojo.models.dit.ltx2_dit import LTX2Config
from serenitymojo.models.ltx2.ltx2_video_stack import (
    LTX2VideoBlockSource, LTX2VideoTail, LTX2VideoStackHead,
    ltx2_video_stack_lora_forward, ltx2_video_stack_lora_backward,
    unpatchify_video, video_lora_names, assert_prompt_mask_all_ones,
)

comptime DEV_CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"
comptime CACHE_LAT = "/home/alex/datasets/ltx2_ref_v3/cache/0288f3d69c08e816d81b014da620db49_00000-025_0512x0288_ltx2.safetensors"
comptime CACHE_TE = "/home/alex/datasets/ltx2_ref_v3/cache/0288f3d69c08e816d81b014da620db49_ltx2_te.safetensors"

comptime NF = 4
comptime NH = 9
comptime NW = 16
comptime S_V = NF * NH * NW      # 576
comptime N_TXT = 1024
comptime NUM_LAYERS = 48
comptime C = 128
comptime VD = 4096
comptime RANK = 16
comptime LORA_SCALE = Float32(0.5)
comptime SIGMA = Float32(0.7)
comptime FRAME_RATE = Float64(25.0)
comptime EPS = Float32(1e-6)


def _sh5(a: Int, b: Int, c: Int, d: Int, e: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a); s.append(b); s.append(c); s.append(d); s.append(e)
    return s^


def _load_bf16(path: String, name: String, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(path)
    return Tensor.from_view_as_bf16(st.tensor_view(name), ctx)


def _finite_stats(name: String, h: List[Float32]) raises -> Bool:
    var s = 0.0; var s2 = 0.0; var amax = 0.0; var finite = True
    for i in range(len(h)):
        var v = Float64(h[i])
        if not (v == v) or v > 1.0e38 or v < -1.0e38:
            finite = False
        s += v; s2 += v * v
        var av = v if v >= 0.0 else -v
        if av > amax: amax = av
    var mean = s / Float64(len(h))
    var var_ = s2 / Float64(len(h)) - mean * mean
    if var_ < 0.0: var_ = 0.0
    print("   [stat]", name, " mean=", Float32(mean), " std=", Float32(sqrt(var_)),
          " absmax=", Float32(amax), " finite=", finite)
    return finite


def _sumabs(v: List[Float32]) -> Float64:
    var s = 0.0
    for i in range(len(v)):
        var x = Float64(v[i])
        s += x if x >= 0.0 else -x
    return s


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()

    print("=== LTX-2.3 VIDEO-MODE STACK real-depth smoke (24 GB fit proof) ===")
    print("  geom S_V=", S_V, " N_TXT=", N_TXT, " blocks=", NUM_LAYERS,
          " rank=", RANK, " sigma=", SIGMA)

    # ── real cache sample ────────────────────────────────────────────────────
    var latent2d = _load_bf16(String(CACHE_LAT), "latents_4x9x16_bfloat16", ctx)  # [128,4,9,16]
    var latent = reshape(latent2d, _sh5(1, C, NF, NH, NW), ctx)
    # POST-connector context used DIRECTLY (no connector here). Fail loud unless
    # the cache mask is all-ones (maskless attn2 faithful only then).
    assert_prompt_mask_all_ones(String(CACHE_TE))
    print("  [mask] prompt_attention_mask all-ones asserted")
    var enc2d = _load_bf16(String(CACHE_TE), "video_prompt_embeds_bfloat16", ctx)  # [1024,4096]
    var enc = reshape(enc2d, [1, N_TXT, VD], ctx)
    _ = _finite_stats(String("latent"), latent.to_host(ctx))
    _ = _finite_stats(String("enc(post-connector)"), enc.to_host(ctx))

    # ── head (frozen, BF16) then cast outputs to F32 ─────────────────────────
    # The stack runs F32 end-to-end: the frozen per-block backward arm
    # (ltx2_video_backward.mojo) routes d_x through linear_backward_dx, which
    # always emits F32, so the backward is an F32 carrier (it CANNOT run pure
    # BF16 without touching the frozen arm). Running the whole 48-block fwd+bwd
    # in F32 is a STRICTLY STRONGER 24 GB fit proof: F32 doubles every
    # activation/weight, so if F32 fits, the BF16 production path fits trivially.
    print("  [head] load + forward (patchify_proj / adaln / rope), cast->F32")
    var head = LTX2VideoStackHead.load(String(DEV_CKPT), ctx)
    var ho = head.forward[S_V, N_TXT](latent, enc, SIGMA, NF, NH, NW, FRAME_RATE, ctx)
    var hidden = cast_tensor(ho.hidden, STDtype.F32, ctx)
    var v_temb = cast_tensor(ho.v_temb, STDtype.F32, ctx)
    var v_embedded = cast_tensor(ho.v_embedded, STDtype.F32, ctx)
    var v_prompt_ts = cast_tensor(ho.v_prompt_ts, STDtype.F32, ctx)
    var v_cos = cast_tensor(ho.v_cos, STDtype.F32, ctx)
    var v_sin = cast_tensor(ho.v_sin, STDtype.F32, ctx)
    var encf = cast_tensor(enc, STDtype.F32, ctx)
    _ = _finite_stats(String("hidden0"), hidden.to_host(ctx))
    _ = _finite_stats(String("v_temb"), v_temb.to_host(ctx))

    # ── tail + streamed block source (dev-fp8, streamed then upcast F32) ─────
    var tail = LTX2VideoTail.load(String(DEV_CKPT), True, ctx)
    var src = LTX2VideoBlockSource.open(String(DEV_CKPT), cfg, True)
    if src.block_count() != NUM_LAYERS:
        raise Error(String("dev-fp8 block_count != 48: ") + String(src.block_count()))

    # ── LoRA carrier: 384 adapters (8/block), A and B both small randn (F32) ─
    var names = video_lora_names()
    var lora_a = List[ArcPointer[Tensor]]()
    var lora_b = List[ArcPointer[Tensor]]()
    var seed = UInt64(20260709)
    for _bi in range(NUM_LAYERS):
        for _s in range(len(names)):
            var a = mul_scalar(randn([RANK, VD], seed, STDtype.F32, ctx), Float32(0.02), ctx)
            seed += 1
            var b = mul_scalar(randn([VD, RANK], seed, STDtype.F32, ctx), Float32(0.02), ctx)
            seed += 1
            lora_a.append(ArcPointer[Tensor](a^))
            lora_b.append(ArcPointer[Tensor](b^))
    print("  [lora] built", len(lora_a), "adapters (F32, A&B nonzero)")

    # ── forward ──────────────────────────────────────────────────────────────
    print("  [forward] 48 streamed blocks (F32) + F32 tail ...")
    var t0 = perf_counter_ns()
    var fwd = ltx2_video_stack_lora_forward[S_V, N_TXT](
        hidden, encf, v_temb, v_embedded, v_prompt_ts, v_cos, v_sin,
        tail, src, lora_a, lora_b, LORA_SCALE, NUM_LAYERS, EPS, ctx,
    )
    var t_fwd = Float64(perf_counter_ns() - t0) / 1.0e9

    var pred_finite = _finite_stats(String("pred(velocity)"), fwd.pred.to_host(ctx))
    var vel = unpatchify_video[S_V](fwd.pred, NF, NH, NW, ctx)
    var vsh = vel.shape()
    print("   velocity NCFHW: [", vsh[0], ",", vsh[1], ",", vsh[2], ",", vsh[3], ",", vsh[4], "]")

    var saved_bytes = 0
    for ref ap in fwd.saved_inputs:
        saved_bytes += ap[].nbytes()
    print("   [mem] resident saved block-inputs:", saved_bytes, "bytes (",
          Float64(saved_bytes) / 1.0e6, "MB )")

    # ── loss proxy 0.5*sum(pred^2) -> d_pred = pred ──────────────────────────
    var predh = fwd.pred.to_host(ctx)
    var loss = 0.0
    for i in range(len(predh)):
        loss += 0.5 * Float64(predh[i]) * Float64(predh[i])
    print("   loss-proxy 0.5*sum(pred^2) =", Float32(loss))
    var d_pred = fwd.pred.clone(ctx)

    # ── backward ─────────────────────────────────────────────────────────────
    print("  [backward] tail bwd -> 48 reverse blocks (recompute+bwd) ...")
    var t1 = perf_counter_ns()
    var grads = ltx2_video_stack_lora_backward[S_V, N_TXT](
        d_pred, fwd.saved_inputs, fwd.x_last,
        encf, v_temb, v_embedded, v_prompt_ts, v_cos, v_sin,
        tail, src, lora_a, lora_b, LORA_SCALE, NUM_LAYERS, EPS, ctx,
    )
    var t_bwd = Float64(perf_counter_ns() - t1) / 1.0e9

    var di_finite = _finite_stats(String("d_input"), grads.d_input)

    # nonzero + finite LoRA grads
    var n_ad = NUM_LAYERS * len(names)
    var total_absum = 0.0
    var zero_adapters = 0
    for i in range(n_ad):
        var sa = _sumabs(grads.d_a[i])
        var sb = _sumabs(grads.d_b[i])
        total_absum += sa + sb
        if sa == 0.0 or sb == 0.0:
            zero_adapters += 1
    print("   [lora grads] adapters:", n_ad, " sum|dA|+|dB| =", Float32(total_absum),
          " zero-adapters:", zero_adapters, " nonfinite:", grads.nonfinite)

    print("  [time] forward:", t_fwd, "s  backward:", t_bwd, "s  total:",
          t_fwd + t_bwd, "s")

    var ok = True
    if not pred_finite:
        print("  FAIL: pred non-finite"); ok = False
    if not di_finite:
        print("  FAIL: d_input non-finite"); ok = False
    if grads.nonfinite != 0:
        print("  FAIL: nonfinite LoRA grads:", grads.nonfinite); ok = False
    if total_absum <= 0.0:
        print("  FAIL: all LoRA grads zero"); ok = False
    if zero_adapters != 0:
        print("  FAIL: some adapters have zero d_A or d_B:", zero_adapters); ok = False
    if vsh[0] != 1 or vsh[1] != C or vsh[2] != NF or vsh[3] != NH or vsh[4] != NW:
        print("  FAIL: unpatchified velocity shape wrong"); ok = False

    if not ok:
        raise Error("LTX-2 VIDEO STACK REAL SMOKE FAIL")
    print("LTX-2.3 VIDEO-MODE STACK REAL-DEPTH SMOKE PASS (48 blocks fwd+bwd, "
          "finite nonzero grads, no OOM)")
