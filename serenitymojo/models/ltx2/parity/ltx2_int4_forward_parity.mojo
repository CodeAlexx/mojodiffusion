# ltx2_int4_forward_parity.mojo — INT4 vs BF16 LTX-2.3 full-stack forward parity.
#
# Runs the SAME 48-block video-mode stack forward TWICE on the SAME inputs (head
# outputs / LoRA / tail all shared) with two block sources:
#   BF16 arm : LTX2VideoBlockSource.open(dev-fp8)          (fp8 → bf16 on load)
#   INT4 arm : LTX2VideoBlockSource.open_int4(dev-fp8, svdint4-slab)
#             (int4 → bf16 reconstruct via ops/svdquant per block)
# and reports cos(pred_int4, pred_bf16) over the flattened velocity — the
# end-to-end weight-quant fidelity through the real model. Weights STREAM per
# block in both arms (no co-resident weight store), so running both arms in one
# process is VRAM-safe; the memory WIN is the on-disk slab size (15.44GB int4 vs
# ~42GB bf16 / ~21GB fp8), reported separately by the quantizer.
#
# Build (link cuDNN flash shim — the AV block uses ops/attention flash):
#   rm -f serenitymojo.mojopkg
#   export LD_LIBRARY_PATH=/home/alex/.local/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
#   pixi run mojo build -O2 -I . -Xlinker -lm -Xlinker -lcuda \
#     -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -Lserenitymojo/ops/cshim/lib/cudnn_stubs -Xlinker -lcudnn \
#     -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     -Xlinker -rpath -Xlinker /home/alex/.local/lib/python3.12/site-packages/nvidia/cudnn/lib \
#     serenitymojo/models/ltx2/parity/ltx2_int4_forward_parity.mojo -o /tmp/ltx2_int4_parity

from std.gpu.host import DeviceContext
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
    ltx2_video_stack_lora_forward, video_lora_names, assert_prompt_mask_all_ones,
)

comptime DEV_CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"
comptime SLAB = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-svdint4-r32.safetensors"
comptime CACHE_LAT = "/home/alex/datasets/ltx2_musubi_v3/cache/0288f3d69c08e816d81b014da620db49_00000-025_0512x0288_ltx2.safetensors"
comptime CACHE_TE = "/home/alex/datasets/ltx2_musubi_v3/cache/0288f3d69c08e816d81b014da620db49_ltx2_te.safetensors"

comptime NF = 4
comptime NH = 9
comptime NW = 16
comptime S_V = NF * NH * NW
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


def _stats(name: String, h: List[Float32]) raises:
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


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error("cos len mismatch")
    var dot = 0.0; var na = 0.0; var nb = 0.0
    for i in range(len(a)):
        var av = Float64(a[i]); var bv = Float64(b[i])
        dot += av * bv; na += av * av; nb += bv * bv
    var denom = sqrt(na) * sqrt(nb)
    if denom == 0.0:
        raise Error("cos zero-norm")
    return dot / denom


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    print("=== LTX-2.3 INT4 vs BF16 full-stack forward parity (48 blocks, S_V=", S_V, ") ===")

    # ── shared inputs: real cache sample → head outputs ──────────────────────
    var latent = reshape(_load_bf16(String(CACHE_LAT), "latents_4x9x16_bfloat16", ctx),
                         _sh5(1, C, NF, NH, NW), ctx)
    assert_prompt_mask_all_ones(String(CACHE_TE))
    var enc = reshape(_load_bf16(String(CACHE_TE), "video_prompt_embeds_bfloat16", ctx),
                     [1, N_TXT, VD], ctx)
    var head = LTX2VideoStackHead.load(String(DEV_CKPT), ctx)
    var ho = head.forward[S_V, N_TXT](latent, enc, SIGMA, NF, NH, NW, FRAME_RATE, ctx)
    var hidden = cast_tensor(ho.hidden, STDtype.F32, ctx)
    var v_temb = cast_tensor(ho.v_temb, STDtype.F32, ctx)
    var v_embedded = cast_tensor(ho.v_embedded, STDtype.F32, ctx)
    var v_prompt_ts = cast_tensor(ho.v_prompt_ts, STDtype.F32, ctx)
    var v_cos = cast_tensor(ho.v_cos, STDtype.F32, ctx)
    var v_sin = cast_tensor(ho.v_sin, STDtype.F32, ctx)
    var encf = cast_tensor(enc, STDtype.F32, ctx)
    var tail = LTX2VideoTail.load(String(DEV_CKPT), True, ctx)

    # ── shared LoRA (identical seed → cancels in the cos; isolates weight quant) ─
    var names = video_lora_names()
    var lora_a = List[ArcPointer[Tensor]]()
    var lora_b = List[ArcPointer[Tensor]]()
    var seed = UInt64(20260710)
    for _bi in range(NUM_LAYERS):
        for _s in range(len(names)):
            lora_a.append(ArcPointer[Tensor](mul_scalar(randn([RANK, VD], seed, STDtype.F32, ctx), Float32(0.02), ctx)))
            seed += 1
            lora_b.append(ArcPointer[Tensor](mul_scalar(randn([VD, RANK], seed, STDtype.F32, ctx), Float32(0.02), ctx)))
            seed += 1

    # Shared-setup VRAM baseline (head/tail/lora resident in BOTH arms). Per-arm
    # working set = free_base - free_after_forward (only the streaming forward's
    # transient blocks + saved block-inputs grow past this point).
    ctx.synchronize()
    var free_base = ctx.get_memory_info()[0]
    print("  [mem] free VRAM after shared setup:", Float64(free_base) / 1.0e9, "GB")

    # ── ARM 1: BF16 (fp8 → bf16 per block) ───────────────────────────────────
    print("  [bf16] fp8→bf16 streamed forward ...")
    var src_bf16 = LTX2VideoBlockSource.open(String(DEV_CKPT), cfg, True)
    var t0 = perf_counter_ns()
    var fwd_bf16 = ltx2_video_stack_lora_forward[S_V, N_TXT](
        hidden, encf, v_temb, v_embedded, v_prompt_ts, v_cos, v_sin,
        tail, src_bf16, lora_a, lora_b, LORA_SCALE, NUM_LAYERS, EPS, ctx)
    var t_bf16 = Float64(perf_counter_ns() - t0) / 1.0e9
    var pred_bf16 = fwd_bf16.pred.to_host(ctx)
    _stats(String("pred_bf16"), pred_bf16)
    var saved_bf16 = 0
    for ref ap in fwd_bf16.saved_inputs:
        saved_bf16 += ap[].nbytes()
    ctx.synchronize()
    # SIGNED diff (get_memory_info is unsigned; free-after can exceed free_base
    # once arm A is dropped → naive subtraction underflows). Clamp at 0.
    var d_bf16 = Int(free_base) - Int(ctx.get_memory_info()[0])
    var used_bf16 = d_bf16 if d_bf16 > 0 else 0
    print("   [mem bf16] fwd working set:", Float64(used_bf16) / 1.0e9,
          "GB  (saved block-inputs", Float64(saved_bf16) / 1.0e6, "MB)")
    # Free arm-A working set (saved block-inputs + pred device) before arm B so
    # arm B's VRAM read is not contaminated by arm A residency. pred_bf16 is the
    # host copy and survives.
    _ = fwd_bf16^
    ctx.synchronize()

    # ── ARM 2: INT4 (svdint4 slab → bf16 reconstruct per block) ──────────────
    print("  [int4] svdint4→bf16 streamed forward ...")
    var src_int4 = LTX2VideoBlockSource.open_int4(String(DEV_CKPT), String(SLAB), cfg, True)
    var t1 = perf_counter_ns()
    var fwd_int4 = ltx2_video_stack_lora_forward[S_V, N_TXT](
        hidden, encf, v_temb, v_embedded, v_prompt_ts, v_cos, v_sin,
        tail, src_int4, lora_a, lora_b, LORA_SCALE, NUM_LAYERS, EPS, ctx)
    var t_int4 = Float64(perf_counter_ns() - t1) / 1.0e9
    var pred_int4 = fwd_int4.pred.to_host(ctx)
    _stats(String("pred_int4"), pred_int4)
    var saved_int4 = 0
    for ref ap in fwd_int4.saved_inputs:
        saved_int4 += ap[].nbytes()
    ctx.synchronize()
    var d_int4 = Int(free_base) - Int(ctx.get_memory_info()[0])
    var used_int4 = d_int4 if d_int4 > 0 else 0
    print("   [mem int4] fwd working set:", Float64(used_int4) / 1.0e9,
          "GB  (saved block-inputs", Float64(saved_int4) / 1.0e6, "MB)")

    # ── verdict ──────────────────────────────────────────────────────────────
    var cos = _cos(pred_int4, pred_bf16)
    print("  [time] bf16", t_bf16, "s   int4", t_int4, "s")
    print("  [mem] fwd working set  bf16", Float64(used_bf16) / 1.0e9, "GB   int4",
          Float64(used_int4) / 1.0e9, "GB  (both STREAM per block — no resident weight store)")
    print("  [PARITY] cos(pred_int4, pred_bf16) =", cos)
    print("  [memory] slab 15.44GB int4 vs ~42GB bf16 / ~21GB fp8 on-disk (0.37x bf16)")
    if cos >= 0.99:
        print("LTX-2.3 INT4 FULL-STACK FORWARD PARITY PASS (cos", cos, ">= 0.99)")
    else:
        print("LTX-2.3 INT4 FULL-STACK FORWARD PARITY: cos", cos, "< 0.99 — investigate")
