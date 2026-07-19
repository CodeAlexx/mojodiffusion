# ltx2_stream_cost_probe.mojo — WHERE do the 58s/step go? (speed campaign probe)
#
# Times, separately, on the real dev-fp8 checkpoint at milestone-1 geometry:
#   (A) streamed get_block (mmap scan + H2D + per-tensor SYNC dequant) x 48
#   (B) fp8-RESIDENT get_block (raw fp8 in VRAM + no-sync dequant) x 48
#   (C) block compute only: ltx2_video_block_train_forward on ONE materialized
#       block x 48 iterations (weights already on device)
#   (D) block backward only x 48 on the same block
# The 58s step ~= 2*(A or B) + 48*(C_fwd) + 48*(C_fwd recompute + D).
#
# Build: rm -f serenitymojo.mojopkg && pixi run mojo build -O2 -I . \
#   -Xlinker -lm -Xlinker -lcuda \
#   serenitymojo/models/ltx2/parity/ltx2_stream_cost_probe.mojo -o /tmp/ltx2_probe

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.offload.ltx2_int4_block_stream import LTX2Int4BlockStream
from std.memory import ArcPointer
from std.time import perf_counter_ns

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import reshape
from serenitymojo.ops.random import randn
from serenitymojo.ops.tensor_algebra import mul_scalar
from serenitymojo.models.dit.ltx2_dit import LTX2Config
from serenitymojo.models.ltx2.ltx2_video_stack import (
    LTX2VideoBlockSource, LTX2VideoStackHead, video_lora_names, _attach_block_lora,
)
from serenitymojo.models.ltx2.ltx2_video_backward import (
    ltx2_video_block_train_forward, ltx2_video_block_backward,
)
from serenitymojo.offload.ltx2_block_stream import LTX2BlockStream
from serenitymojo.models.dit.ltx2_dit import LTX2AVBlockWeights

comptime DEV_CKPT = "/home/alex/.serenity/models/checkpoints/ltx-2.3-22b-dev-fp8.safetensors"
comptime CACHE_LAT = "/home/alex/datasets/ltx2_musubi_v3/cache/0288f3d69c08e816d81b014da620db49_00000-025_0512x0288_ltx2.safetensors"
comptime CACHE_TE = "/home/alex/datasets/ltx2_musubi_v3/cache/0288f3d69c08e816d81b014da620db49_ltx2_te.safetensors"
comptime NF = 4
comptime NH = 9
comptime NW = 16
comptime S_V = 576
comptime N_TXT = 1024
comptime VD = 4096
comptime EPS = Float32(1e-6)


def _sec(t0: UInt, t1: UInt) -> Float64:
    return Float64(Int(t1 - t0)) / 1.0e9


def main() raises:
    var ctx = DeviceContext()
    var cfg = LTX2Config.ltx2()
    print("=== LTX2 58s/step cost probe (dev-fp8, S_V=576) ===")

    # head outputs (one fixed sigma) for the compute legs
    var stl = ShardedSafeTensors.open(String(CACHE_LAT))
    var lat2 = Tensor.from_view_as_bf16(stl.tensor_view(String("latents_4x9x16_bfloat16")), ctx)
    var latent = reshape(lat2, [1, 128, NF, NH, NW], ctx)
    var stt = ShardedSafeTensors.open(String(CACHE_TE))
    var enc2 = Tensor.from_view_as_bf16(stt.tensor_view(String("video_prompt_embeds_bfloat16")), ctx)
    var enc = reshape(enc2, [1, N_TXT, VD], ctx)
    var head = LTX2VideoStackHead.load(String(DEV_CKPT), ctx)
    var ho = head.forward[S_V, N_TXT](latent, enc, Float32(0.7), NF, NH, NW, Float64(25.0), ctx)
    var hidden = cast_tensor(ho.hidden, STDtype.F32, ctx)
    var v_temb = cast_tensor(ho.v_temb, STDtype.F32, ctx)
    var v_prompt_ts = cast_tensor(ho.v_prompt_ts, STDtype.F32, ctx)
    var v_cos = cast_tensor(ho.v_cos, STDtype.F32, ctx)
    var v_sin = cast_tensor(ho.v_sin, STDtype.F32, ctx)
    var encf = cast_tensor(enc, STDtype.F32, ctx)

    # (A) streamed get_block x48 (the trainer's current path, F32)
    var src = LTX2VideoBlockSource.open(String(DEV_CKPT), cfg, True)
    var tA0 = perf_counter_ns()
    for i in range(48):
        var w = src.get_block(i, ctx)
        _ = w^
    ctx.synchronize()
    var tA1 = perf_counter_ns()
    print("  (A) streamed get_block x48 (F32):", _sec(tA0, tA1), "s")

    # (B) fp8-RESIDENT: preload raw fp8 to VRAM, then materialize x48
    var stream2 = LTX2BlockStream.open(String(DEV_CKPT))
    var tP0 = perf_counter_ns()
    stream2.enable_fp8_resident_range(0, 47, ctx)
    ctx.synchronize()
    var tP1 = perf_counter_ns()
    print("  (B) resident preload (48 blocks raw fp8):", _sec(tP0, tP1), "s;",
          "resident bytes:", stream2.resident_bytes())
    var srcR = LTX2VideoBlockSource(stream2^, Optional[LTX2Int4BlockStream](None), cfg, True)
    var tB0 = perf_counter_ns()
    for i in range(48):
        var w = srcR.get_block(i, ctx)
        _ = w^
    ctx.synchronize()
    var tB1 = perf_counter_ns()
    print("  (B) resident get_block x48 (F32):", _sec(tB0, tB1), "s")

    # (C)+(D) compute-only: one materialized block reused 48x, fwd + bwd, LoRA attached
    var wblk = srcR.get_block(0, ctx)
    var names = video_lora_names()
    var la = List[ArcPointer[Tensor]]()
    var lb = List[ArcPointer[Tensor]]()
    var seed = UInt64(7)
    for _s in range(8):
        la.append(ArcPointer[Tensor](mul_scalar(randn([16, VD], seed, STDtype.F32, ctx), Float32(0.02), ctx)))
        seed += 1
        lb.append(ArcPointer[Tensor](mul_scalar(randn([VD, 16], seed, STDtype.F32, ctx), Float32(0.02), ctx)))
        seed += 1
    _attach_block_lora(wblk, 0, names, la, lb, Float32(0.5))

    var tC0 = perf_counter_ns()
    for _ in range(48):
        var f = ltx2_video_block_train_forward[S_V, N_TXT](
            wblk, hidden, encf, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx)
        _ = f^
    ctx.synchronize()
    var tC1 = perf_counter_ns()
    print("  (C) block train-FORWARD x48 (weights resident):", _sec(tC0, tC1), "s")

    var fwd1 = ltx2_video_block_train_forward[S_V, N_TXT](
        wblk, hidden, encf, v_temb, v_prompt_ts, v_cos, v_sin, EPS, ctx)
    var d_out = hidden.clone(ctx)
    var tD0 = perf_counter_ns()
    for _ in range(48):
        var g = ltx2_video_block_backward[S_V, N_TXT](
            wblk, fwd1.acts, d_out, v_temb, v_cos, v_sin, EPS, ctx)
        _ = g^
    ctx.synchronize()
    var tD1 = perf_counter_ns()
    print("  (D) block BACKWARD x48 (weights resident):", _sec(tD0, tD1), "s")

    print("  model: step ~= A_fwd + (A_recompute+... ) ; compare (A) vs (B) vs (C)/(D)")
    print("DONE probe")
