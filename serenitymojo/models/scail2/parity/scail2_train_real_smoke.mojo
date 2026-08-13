# serenitymojo/models/scail2/parity/scail2_train_real_smoke.mojo
#
# G2 REAL-WEIGHT FINITE SMOKE for the SCAIL-2 i2v LoRA trainer (CHUNK 2b).
#
# Opens the REAL FP8 stream (transformer_fp8) + shared weights, loads a REAL
# staged replacement sample, and runs the WHOLE training step end to end at TRUE
# depth-40 / dim-5120 on real streamed weights:
#   embed -> time/text/image/rope -> 40-block STREAMING LoRA stack fwd ->
#   _head_video -> flow-match MSE loss (noise - video_latent) -> head backward ->
#   40-block STREAMING recompute backward -> scail2_lora_adamw_step.
#
# STREAMING (16 GB): exactly one FP8 base block is materialized at a time in both
# the forward and the backward-recompute loops (holding all 40 resident is ~28 GB
# = guaranteed OOM). Peak VRAM is dominated by the resident shared weights + a
# single ~0.8 GB block, NOT by depth.
#
# SEQUENCE NOTE (honest): the CHUNK-1 F32-residual training block uses DENSE
# `sdpa_nomask` (its certified backward), so the production S=39,872 (full 65f /
# 896x512) is out of reach on 16 GB without a flash training block (separate
# work). This smoke therefore runs at real depth-40 / dim-5120 / real streamed
# weights / real conditioning but on a REDUCED sequence derived by cropping the
# real staged latents to a small grid (FT=1, GH=8, GW=8 => S=144). Depth, dim,
# streaming and the full fwd/head/bwd/AdamW path are exercised end to end.
#
# ASSERTS: fwd output + loss finite; every LoRA grad finite and some nonzero; one
# AdamW step MOVES the adapters; no OOM at depth-40/dim-5120. Prints loss, grad
# norm, s/step (peak VRAM is captured externally via nvidia-smi polling).
#
# Run:
#   rm -f serenitymojo.mojopkg
#   ( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits; sleep 0.2; done > /tmp/vram.log & POLL=$!; \
#     pixi run mojo run -I . serenitymojo/models/scail2/parity/scail2_train_real_smoke.mojo; \
#     kill $POLL; echo "peak VRAM MiB:"; sort -n /tmp/vram.log | tail -1 )

from std.collections import List, Optional
from std.ffi import external_call
from max.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import UnsafePointer, alloc
from std.time import perf_counter

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.tensor import Tensor
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.random_torch import randn_torch
from serenitymojo.ops.tensor_algebra import (
    concat, full_device, reshape, slice, mul_scalar, add, sub,
)
from serenitymojo.models.scail2.scail2_streamed_dit import Scail2StreamedDiT
from serenitymojo.models.scail2.scail2_fp8_stream import Scail2A14BFP8Stream
from serenitymojo.models.scail2.scail2_rope import (
    Scail2SequencePlan, build_scail2_rope_tables,
)
from serenitymojo.training.train_step import LoraAdapter
from serenitymojo.models.scail2.scail2_stack_lora import (
    Scail2LoraSet, scail2_total_adapters, scail2_lora_adamw_step,
)
from serenitymojo.models.scail2.scail2_stack_train_full import (
    scail2_stack_lora_forward_streamed, scail2_stack_lora_backward_streamed,
    scail2_head_video_backward,
)


comptime _CStr = UnsafePointer[UInt8, MutExternalOrigin]

# ── reduced-sequence geometry (real dim/depth; cropped real latents) ──
comptime DIM = 5120
comptime FFN = 13824
comptime HEADS = 40
comptime HEAD_DIM = 128
comptime NUM_LAYERS = 40
comptime OUT_DIM = 16
comptime TXT = 512
comptime CTXL = 512
comptime IMG = 257
comptime EPS = Float32(1.0e-6)

comptime FT = 1
comptime GH = 8
comptime GW = 8
comptime H_LAT = GH * 2        # 16
comptime W_LAT = GW * 2        # 16
comptime AR = 0
comptime S = (1 + FT) * GH * GW + FT * (GH // 2) * (GW // 2)   # 144
comptime CHANNELS = 16

comptime CACHE_DIR = "/home/alex/.serenity/models/checkpoints/SCAIL-2-Mojo/transformer_fp8"
comptime RUN_DIR = "/home/alex/.serenity/runs/scail2-gates/replacement"


def _cstr(s: String) -> _CStr:
    var n = s.byte_length()
    var buf = alloc[UInt8](n + 1)
    var src = s.as_bytes()
    for i in range(n):
        buf[i] = src[i]
    buf[n] = 0
    return _CStr(unsafe_from_address=Int(buf))


def _sync_alloc():
    _ = external_call["setenv", Int32](
        _cstr(String("MODULAR_DEVICE_CONTEXT_SYNC_MODE")),
        _cstr(String("true")), Int32(1),
    )


def _load_view(path: String, key: String, ctx: DeviceContext) raises -> Tensor:
    var st = ShardedSafeTensors.open(path)
    if key not in st.names():
        raise Error(String("missing key ") + key + String(" in ") + path)
    return Tensor.from_view(st.tensor_view(key), ctx)


# crop a [C,T,H,W] latent to [C, t, h, w] (top-left corner, first frames).
def _crop4(x: Tensor, t: Int, h: Int, w: Int, ctx: DeviceContext) raises -> Tensor:
    var a = slice(x, 1, 0, t, ctx)
    a = slice(a, 2, 0, h, ctx)
    a = slice(a, 3, 0, w, ctx)
    return a^


def _channels20(latent16: Tensor, marker: Float32, t: Int, h: Int, w: Int,
                ctx: DeviceContext) raises -> Tensor:
    var marker4 = full_device([4, t, h, w], marker, STDtype.F32, ctx)
    return concat(0, ctx, latent16, marker4)


def _nonfinite(v: List[Float32]) -> Int:
    var bad = 0
    for i in range(len(v)):
        var x = v[i]
        if (x != x) or (x - x != Float32(0.0)):
            bad += 1
    return bad


def _make_adapter(rank: Int, in_f: Int, out_f: Int, scale: Float32,
                  seed: UInt64) -> LoraAdapter:
    # A != 0 AND B != 0 so every LoRA grad is nonzero (not PEFT-identity).
    var a = List[Float32]()
    var b = List[Float32]()
    var st = seed
    for _ in range(rank * in_f):
        st = st * 6364136223846793005 + 1442695040888963407
        a.append((Float32(Int(st >> 40)) * Float32(1.0 / 16777216.0) - 0.5) * 0.04)
    for _ in range(out_f * rank):
        st = st * 6364136223846793005 + 1442695040888963407
        b.append((Float32(Int(st >> 40)) * Float32(1.0 / 16777216.0) - 0.5) * 0.04)
    var za = List[Float32]()
    za.resize(rank * in_f, Float32(0.0))
    var zb = List[Float32]()
    zb.resize(out_f * rank, Float32(0.0))
    return LoraAdapter(a^, b^, rank, in_f, out_f, scale,
                       za.copy(), za^, zb.copy(), zb^)


def _build_lora_set(rank: Int, alpha: Float32) -> Scail2LoraSet:
    var scale = alpha / Float32(rank)
    var ad = List[LoraAdapter]()
    var seed = UInt64(90210)
    for _ in range(NUM_LAYERS):
        for _ in range(8):                                   # sa/ca q,k,v,o
            ad.append(_make_adapter(rank, DIM, DIM, scale, seed)); seed += 7
        ad.append(_make_adapter(rank, DIM, FFN, scale, seed)); seed += 7   # ffn0
        ad.append(_make_adapter(rank, FFN, DIM, scale, seed)); seed += 7   # ffn2
        ad.append(_make_adapter(rank, DIM, DIM, scale, seed)); seed += 7   # img_k
        ad.append(_make_adapter(rank, DIM, DIM, scale, seed)); seed += 7   # img_v
    return Scail2LoraSet(ad^, NUM_LAYERS, rank)


def _adapter_checksum(lora: Scail2LoraSet) -> Float64:
    var s = Float64(0.0)
    for i in range(len(lora.ad)):
        for j in range(len(lora.ad[i].a)):
            s += Float64(Float32(lora.ad[i].a[j]))
        for j in range(len(lora.ad[i].b)):
            s += Float64(Float32(lora.ad[i].b[j]))
    return s


def main() raises:
    _sync_alloc()
    var ctx = DeviceContext()
    print("=== SCAIL-2 i2v LoRA REAL-WEIGHT finite smoke (CHUNK 2b) ===")
    print("depth=", NUM_LAYERS, " dim=", DIM, " heads=", HEADS, " S=", S,
          " (FT=", FT, " GH=", GH, " GW=", GW, ") TXT=", TXT, " IMG=", IMG)

    # ── frozen shared weights + head/embed/conditioning host ──
    var dit = Scail2StreamedDiT.open(CACHE_DIR, ctx)
    # separate lightweight stream handle for the block-by-block base loader.
    var stream = Scail2A14BFP8Stream.open(CACHE_DIR)

    # ── real staged latents (cropped to the reduced grid) ──
    var ref5 = _load_view(RUN_DIR + String("/reference.safetensors"),
                          String("reference_latent"), ctx)         # [1,16,1,64,112]
    var ref4 = reshape(ref5, [CHANNELS, 1, 64, 112], ctx)
    var ref_crop = _crop4(ref4, 1, H_LAT, W_LAT, ctx)              # [16,1,16,16]

    var pose5 = _load_view(RUN_DIR + String("/pose.safetensors"),
                           String("pose_latent"), ctx)             # [1,16,17,32,56]
    var pose4 = reshape(pose5, [CHANNELS, 17, 32, 56], ctx)
    var pose_crop = _crop4(pose4, FT, H_LAT // 2, W_LAT // 2, ctx) # [16,1,8,8]

    var vid4 = _load_view(RUN_DIR + String("/latent_replacement_1step.safetensors"),
                          String("latent"), ctx)                   # [16,17,64,112]
    var x0 = _crop4(vid4, FT, H_LAT, W_LAT, ctx)                   # [16,1,16,16]

    var refm = _load_view(RUN_DIR + String("/stage.safetensors"),
                          String("ref_masks"), ctx)                # [28,18,64,112]
    var ref_masks = _crop4(refm, FT + 1, H_LAT, W_LAT, ctx)        # [28,2,16,16]
    var drvm = _load_view(RUN_DIR + String("/stage.safetensors"),
                          String("driving_masks"), ctx)            # [28,17,32,56]
    var driving_masks = _crop4(drvm, FT, H_LAT // 2, W_LAT // 2, ctx)  # [28,1,8,8]

    # text (pos) + clip conditioning (real)
    var pos = _load_view(RUN_DIR + String("/text.safetensors"),
                         String("pos_embed"), ctx)                 # [1,512,4096]
    var pos2 = reshape(pos, [TXT, 4096], ctx)
    var clip = _load_view(RUN_DIR + String("/clip.safetensors"),
                          String("clip_context"), ctx)             # [1,257,1280]

    # ── flow-match target: x_t = (1-sigma)*x0 + sigma*noise ; v* = noise - x0 ──
    var sigma = Float32(0.5)
    var timestep = Float32(500.0)
    var noise = randn_torch([CHANNELS, FT, H_LAT, W_LAT], UInt64(1234), ctx)
    var x0_h = x0.to_host(ctx)
    var noise_h = noise.to_host(ctx)
    var xt_h = List[Float32]()
    var target_h = List[Float32]()
    for i in range(len(x0_h)):
        xt_h.append((1.0 - sigma) * x0_h[i] + sigma * noise_h[i])
        target_h.append(noise_h[i] - x0_h[i])
    var xt = Tensor.from_host(xt_h^, [CHANNELS, FT, H_LAT, W_LAT], STDtype.F32, ctx)

    # ── channel-20 i2v marker packing (0.0 = noised video ; 1.0 = ref / pose) ──
    var video20 = _channels20(xt^, 0.0, FT, H_LAT, W_LAT, ctx)
    var reference20 = _channels20(ref_crop^, 1.0, 1, H_LAT, W_LAT, ctx)
    var pose20 = _channels20(pose_crop^, 1.0, FT, H_LAT // 2, W_LAT // 2, ctx)

    var t_start = perf_counter()

    # ── frozen embed ──
    var seq_t = dit._embed_base_sequence[FT, GH, GW, S](
        video20, reference20, pose20, ref_masks, driving_masks, ctx,
    )
    var sequence = cast_tensor(seq_t, STDtype.F32, ctx).to_host(ctx)   # [S*dim]

    # ── frozen conditioning ──
    var time = dit._time(timestep, STDtype.BF16, ctx)
    var e0_host = time[0].to_host(ctx)                                 # [6*dim]
    var text = dit._text[TXT, CTXL](pos2, STDtype.BF16, ctx)
    var context_txt = cast_tensor(text, STDtype.F32, ctx).to_host(ctx) # [TXT*dim]
    var image = dit._image[IMG](clip, ctx)
    var context_img = cast_tensor(image, STDtype.F32, ctx).to_host(ctx)# [IMG*dim]

    var plan = Scail2SequencePlan(
        video_t=FT, grid_h=GH, grid_w=GW,
        additional_ref_count=AR, replace_flag=True,
    )
    plan.validate()
    if plan.sequence_length() != S:
        raise Error("SCAIL-2 reduced sequence length mismatch")
    var rope = build_scail2_rope_tables(plan, ctx, STDtype.F32)

    # ── LoRA (A!=0, B!=0) ──
    var rank = 8
    var lora = _build_lora_set(rank, Float32(8.0))
    var n_ad = scail2_total_adapters(lora)
    var chk_before = _adapter_checksum(lora)

    # ── 40-block STREAMING forward ──
    var fwd = scail2_stack_lora_forward_streamed[HEADS, HEAD_DIM, S, TXT, IMG](
        sequence^, e0_host, context_txt.copy(), context_img.copy(),
        rope[0], rope[1], stream, NUM_LAYERS, lora, DIM, FFN, HEAD_DIM, EPS, ctx,
    )
    var nf_fwd = _nonfinite(fwd.x_out)
    print("  stack fwd: x_out nonfinite =", nf_fwd, " (must be 0)")

    # ── head forward (reuse _head_video) ──
    var img_t = Tensor.from_host(fwd.x_out.copy(), [1, S, DIM], STDtype.F32, ctx)
    var head_out = dit._head_video[FT, GH, GW, AR, S](
        img_t, time[1], STDtype.BF16, ctx,
    )
    var head_h = cast_tensor(head_out, STDtype.F32, ctx).to_host(ctx)  # [16*1*16*16]
    var nf_head = _nonfinite(head_h)

    # ── flow-match MSE loss + d(head) ──
    var N = len(head_h)
    var loss = Float64(0.0)
    var d_head_h = List[Float32]()
    for i in range(N):
        var e = head_h[i] - target_h[i]
        loss += Float64(e) * Float64(e)
        d_head_h.append(Float32(2.0 / Float32(N)) * e)
    loss /= Float64(N)
    print("  head fwd: nonfinite =", nf_head, "  MSE loss =", loss)

    var d_head = Tensor.from_host(d_head_h^, [OUT_DIM, FT, GH * 2, GW * 2], STDtype.F32, ctx)

    # ── NEW head backward -> d_out into video region of block-40 output ──
    var video_offset = (AR + 1) * GH * GW
    var video_rows = FT * GH * GW
    var video_region = List[Float32]()
    for r in range(video_rows):
        var srcb = (video_offset + r) * DIM
        for d in range(DIM):
            video_region.append(fwd.x_out[srcb + d])
    var head_w = dit._w(String("head.head.weight")).clone(ctx)
    var head_mod = dit._w(String("head.modulation")).clone(ctx)
    var d_seq = scail2_head_video_backward[FT, GH, GW, AR, S](
        d_head, video_region, time[1], head_mod, head_w, OUT_DIM, DIM, EPS, ctx,
    )

    # ── 40-block STREAMING recompute backward ──
    var bwd = scail2_stack_lora_backward_streamed[HEADS, HEAD_DIM, S, TXT, IMG](
        d_seq^, e0_host, context_txt.copy(), context_img.copy(),
        rope[0], rope[1], stream, NUM_LAYERS, lora, fwd, DIM, FFN, HEAD_DIM, EPS, ctx,
    )

    # ── grad finiteness / nonzero / norm ──
    var gnorm = Float64(0.0)
    var nonzero = 0
    var nonfinite_grads = bwd.grads.nonfinite_lora_grads
    for i in range(n_ad):
        for j in range(len(bwd.grads.d_a[i])):
            var g = bwd.grads.d_a[i][j]
            gnorm += Float64(g) * Float64(g)
            if g != Float32(0.0):
                nonzero += 1
        for j in range(len(bwd.grads.d_b[i])):
            var g = bwd.grads.d_b[i][j]
            gnorm += Float64(g) * Float64(g)
            if g != Float32(0.0):
                nonzero += 1
    gnorm = sqrt(gnorm)
    print("  bwd: LoRA grad nonfinite =", nonfinite_grads,
          "  nonzero entries =", nonzero, "  grad_norm =", gnorm)
    print("  bwd: d_x(sequence) nonfinite =", _nonfinite(bwd.d_x))

    # ── one AdamW step; assert the adapters MOVE ──
    scail2_lora_adamw_step(lora, bwd.grads, 1, Float32(1.0e-3), ctx)
    var chk_after = _adapter_checksum(lora)
    var moved = chk_after != chk_before

    ctx.synchronize()
    var elapsed = perf_counter() - t_start

    print("")
    print("  adapter checksum before =", chk_before, "  after =", chk_after,
          "  MOVED =", moved)
    print("  s/step (fwd+head+bwd+adamw) =", elapsed)
    print("")

    var ok = (nf_fwd == 0) and (nf_head == 0) and (nonfinite_grads == 0) \
        and (nonzero > 0) and moved and (_nonfinite(bwd.d_x) == 0)
    if ok:
        print("VERDICT: PASS -- fwd/head/loss finite; LoRA grads finite+nonzero;",
              "AdamW moved adapters; 40 blocks @ dim 5120 streamed with no OOM")
    else:
        print("VERDICT: FAIL -- see the metrics above")
