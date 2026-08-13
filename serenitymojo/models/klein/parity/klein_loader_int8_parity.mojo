# serenitymojo/models/klein/parity/klein_loader_int8_parity.mojo
#
# LOADER-LEVEL PARITY GATE for the int8-W8A8 RESIDENT weight-loading path (Klein
# int8 slice 4). Proves the TurboPlannedLoader's quantize-at-load
# (pin_residents_int8) produces the SAME int8 forward result as the slice-1/2
# block-direct quantize — i.e. the loader is a faithful int8 weight source.
#
# Flow (ONE loader, real Klein-9B checkpoint):
#   1. bf16 REFERENCE: stream double block 0 + single block 8 through the normal
#      bf16 await_block path, build DoubleBlockWeights/SingleBlockWeights (int8
#      sidecar None), run the slice-2 scratch forward with int8=None → REFERENCE.
#   2. pin_residents_int8(budget): quantize each block's base matmul weights to
#      int8 [N,K] + F32 scalar scale ONCE at load, held resident.
#   3. int8: await_block the SAME (now int8-resident) blocks — the int8 fast path
#      returns the int8 weights + scales (NO dequant); the block-weight builder
#      detects the I8 dtype and assembles the StreamInt8 / SingleBlockInt8 payload;
#      run the slice-2 scratch forward with the int8 payload.
#   4. Compare int8-loader forward vs bf16 forward. Bar: cos >= 0.996 (int8 class).
#
# This file only REPORTS the cos; the human gates.
#
# Build+run with the cshim recipe (int8 GEMM + cuDNN flash SDPA need the link):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 2 --target-accelerator sm_120 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -lcuda -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     -Xlinker -rpath -Xlinker /home/alex/.serenity/cudnn/lib \
#     serenitymojo/models/klein/parity/klein_loader_int8_parity.mojo -o /tmp/klein_loader_i8 && \
#   LD_LIBRARY_PATH=.pixi/envs/default/lib:serenitymojo/ops/cshim/lib:/home/alex/.serenity/cudnn/lib:/usr/lib/x86_64-linux-gnu /tmp/klein_loader_i8

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt, log as flog, cos as fcos, pi
from std.memory import ArcPointer
from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.scratch_ring import ScratchRingAllocator

from serenitymojo.offload.plan import build_klein9b_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader

from serenitymojo.models.klein.single_block import (
    SingleBlockWeights, SingleModVecs, single_modvecs_to_device,
    SingleBlockLoraDevice, SingleBlockInt8,
    single_block_lora_forward_device_resident_scratch,
)
from serenitymojo.models.klein.double_block import (
    DoubleBlockWeights, ModVecs, modvecs_to_device,
    StreamLoraDevice, DoubleBlockLoraDevice, DoubleBlockInt8,
    double_block_lora_forward_device_resident_scratch,
)
from serenitymojo.models.klein.klein_stack_lora import (
    _double_weights_from_block, _single_weights_from_block,
)

comptime TArc = ArcPointer[Tensor]

comptime KLEIN9B_PATH = "/home/alex/.serenity/models/checkpoints/flux-2-klein-base-9b.safetensors"

# REAL Klein-9B block dims (configs/klein9b.json).
comptime H = 32
comptime Dh = 128
comptime D = H * Dh          # 4096  == inner_dim
comptime F = 12288           # mlp_hidden
comptime EPS = Float32(1e-06)

# double block joint sequence (128-aligned for the cuDNN flash SDPA path).
comptime N_IMG = 96
comptime N_TXT = 32
comptime S2 = N_IMG + N_TXT   # 128
# single block sequence length.
comptime S1 = 128

# int8-resident budget: covers the 8 double blocks (~3.5GB) + the first singles,
# so double block 0 AND single block 8 are int8-resident. The rest stream.
comptime I8_BUDGET = 4 * 1024 * 1024 * 1024


def _gaussian(n: Int, seed: Int, sd: Float32) -> List[Float32]:
    var out = List[Float32]()
    var st = UInt64(seed * 2654435761 + 12345)
    for _i in range(n):
        st = st * 6364136223846793005 + 1442695040888963407
        var u1 = (Float64(st >> 11) + 1.0) / Float64(1 << 53)
        st = st * 6364136223846793005 + 1442695040888963407
        var u2 = Float64(st >> 11) / Float64(1 << 53)
        var r = sqrt(-2.0 * flog(u1))
        out.append(Float32(r * fcos(2.0 * pi * u2)) * sd)
    return out^


def _ones(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(1.0))
    return o^


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


def _mod(seed0: Int) raises -> ModVecs:
    return ModVecs(
        _gaussian(D, seed0 + 0, 0.5), _gaussian(D, seed0 + 1, 0.5),
        _gaussian(D, seed0 + 2, 0.5), _gaussian(D, seed0 + 3, 0.5),
        _gaussian(D, seed0 + 4, 0.5), _gaussian(D, seed0 + 5, 0.5),
    )


def _report(mut harness: ParityHarness, name: String, t: Tensor,
            ref_h: List[Float32], ctx: DeviceContext) raises -> Float64:
    var r = harness.compare(t, ref_h, ctx)
    print("---- ", name, " int8-LOADER vs bf16-LOADER ----")
    print("  cos =", r.cos, "  max_abs =", r.max_abs, "  n =", r.n)
    if r.cos >= 0.996:
        print("  RESULT: cos >= 0.996 (task bar) -> PASS")
    elif r.cos >= 0.99:
        print("  RESULT: 0.99 <= cos < 0.996 (int8 class met, task bar missed)")
    else:
        print("  RESULT: cos < 0.99 -> FAIL")
    return r.cos


def main() raises:
    var ctx = DeviceContext()
    print("==== klein_loader_int8_parity (slice 4: int8-resident weight loading) ====")
    print("  checkpoint:", KLEIN9B_PATH)
    var harness = ParityHarness(0.996)

    var plan = build_klein9b_block_plan()   # 8 double (0..7) + 24 single (8..31)
    var loader = TurboPlannedLoader.open(
        KLEIN9B_PATH, plan^, OffloadConfig.synchronous_single(), ctx,
        fill_block_store=False,
    )

    # ── shared non-degenerate inputs (identical for bf16 ref + int8) ──
    # DOUBLE block 0.
    var img_x = TArc(Tensor.from_host(_gaussian(N_IMG * D, 500, 1.0), [N_IMG, D], STDtype.F32, ctx))
    var txt_x = TArc(Tensor.from_host(_gaussian(N_TXT * D, 600, 1.0), [N_TXT, D], STDtype.F32, ctx))
    var im = modvecs_to_device(_mod(300), D, ctx)
    var tm = modvecs_to_device(_mod(400), D, ctx)
    var cos2 = Tensor.from_host(_gaussian(S2 * H * (Dh // 2), 700, 1.0), [S2 * H, Dh // 2], STDtype.F32, ctx)
    var sin2 = Tensor.from_host(_gaussian(S2 * H * (Dh // 2), 800, 1.0), [S2 * H, Dh // 2], STDtype.F32, ctx)
    var ones2 = Tensor.from_host(_ones(D), [D], STDtype.F32, ctx)
    var zeros2 = Tensor.from_host(_zeros(D), [D], STDtype.F32, ctx)
    var empty_img = StreamLoraDevice(None, None, None, None, None, None)
    var empty_txt = StreamLoraDevice(None, None, None, None, None, None)
    var dlora = DoubleBlockLoraDevice(empty_img^, empty_txt^)

    # SINGLE block 8.
    var x1 = TArc(Tensor.from_host(_gaussian(S1 * D, 1, 1.0), [S1, D], STDtype.F32, ctx))
    var smv = single_modvecs_to_device(
        SingleModVecs(_gaussian(D, 6, 0.5), _gaussian(D, 7, 0.5), _gaussian(D, 8, 0.5)), D, ctx
    )
    var slora = SingleBlockLoraDevice(None, None)
    var cos1 = Tensor.from_host(_gaussian(S1 * H * (Dh // 2), 9, 1.0), [S1 * H, Dh // 2], STDtype.F32, ctx)
    var sin1 = Tensor.from_host(_gaussian(S1 * H * (Dh // 2), 10, 1.0), [S1 * H, Dh // 2], STDtype.F32, ctx)
    var ones1 = Tensor.from_host(_ones(D), [D], STDtype.F32, ctx)
    var zeros1 = Tensor.from_host(_zeros(D), [D], STDtype.F32, ctx)

    # ══════════════════════════════════════════════════════════════════════════
    # bf16 REFERENCE (streamed) — build BEFORE pinning int8 (so await streams bf16)
    # ══════════════════════════════════════════════════════════════════════════
    print("")
    print("-- bf16 reference forwards (streamed) --")
    loader.prefetch_with_ctx(0, ctx)
    var h0 = loader.await_block(0, ctx)
    print("  double block 0 prefix:", h0.prefix)
    var w0_ref = _double_weights_from_block(h0.block, h0.prefix, ctx)
    var sc_r0 = ScratchRingAllocator(ctx, 512 * 1024 * 1024, 2)
    var ref0 = double_block_lora_forward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S2](
        img_x, txt_x, w0_ref, im, tm, dlora, cos2, sin2, D, F, EPS,
        ones2, zeros2, ctx, sc_r0, int8=Optional[DoubleBlockInt8](None),
    )
    var ref0_img = ref0.img_out[].to_host(ctx)
    var ref0_txt = ref0.txt_out[].to_host(ctx)
    loader.mark_active_block_done(ctx)

    loader.prefetch_with_ctx(8, ctx)
    var h8 = loader.await_block(8, ctx)
    print("  single block 8 prefix:", h8.prefix)
    var w8_ref = _single_weights_from_block(h8.block, h8.prefix, D, F, ctx)
    var sc_r1 = ScratchRingAllocator(ctx, 512 * 1024 * 1024, 2)
    var ref8 = single_block_lora_forward_device_resident_scratch[H, Dh, S1](
        x1, w8_ref, smv, slora, cos1, sin1, D, F, EPS, ones1, zeros1, ctx,
        sc_r1, int8=Optional[SingleBlockInt8](None),
    )
    var ref8_h = ref8.out[].to_host(ctx)
    loader.mark_active_block_done(ctx)

    # ══════════════════════════════════════════════════════════════════════════
    # pin int8-resident (quantize-at-load), then int8 forwards on the SAME blocks
    # ══════════════════════════════════════════════════════════════════════════
    print("")
    print("-- pin_residents_int8 (budget", I8_BUDGET, "bytes) --")
    var pinned = loader.pin_residents_int8(I8_BUDGET, ctx)
    print("  pinned int8 blocks:", pinned, "of", loader.block_count())
    if pinned <= 8:
        raise Error(
            String("gate needs double block 0 AND single block 8 int8-resident; ")
            + "pinned=" + String(pinned) + " (<=8). Raise I8_BUDGET."
        )

    print("")
    print("-- int8-LOADER forwards (int8-resident, quantize-at-load) --")
    var h0b = loader.await_block(0, ctx)
    var w0_i8 = _double_weights_from_block(h0b.block, h0b.prefix, ctx)
    if not w0_i8.img.int8:
        raise Error("double block 0 did not load int8-resident (img.int8 is None)")
    var sc_i0 = ScratchRingAllocator(ctx, 512 * 1024 * 1024, 2)
    var i80 = double_block_lora_forward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S2](
        img_x, txt_x, w0_i8, im, tm, dlora, cos2, sin2, D, F, EPS,
        ones2, zeros2, ctx, sc_i0, int8=w0_i8.int8_payload(),
    )
    var c_img = _report(harness, "DOUBLE block 0 img_out", i80.img_out[], ref0_img, ctx)
    var c_txt = _report(harness, "DOUBLE block 0 txt_out", i80.txt_out[], ref0_txt, ctx)

    var h8b = loader.await_block(8, ctx)
    var w8_i8 = _single_weights_from_block(h8b.block, h8b.prefix, D, F, ctx)
    if not w8_i8.int8:
        raise Error("single block 8 did not load int8-resident (int8 is None)")
    var sc_i1 = ScratchRingAllocator(ctx, 512 * 1024 * 1024, 2)
    var i88 = single_block_lora_forward_device_resident_scratch[H, Dh, S1](
        x1, w8_i8, smv, slora, cos1, sin1, D, F, EPS, ones1, zeros1, ctx,
        sc_i1, int8=w8_i8.int8,
    )
    var c_sgl = _report(harness, "SINGLE block 8 out", i88.out[], ref8_h, ctx)

    print("")
    print("==== SUMMARY ====")
    print("  DOUBLE block 0 img cos =", c_img)
    print("  DOUBLE block 0 txt cos =", c_txt)
    print("  SINGLE block 8     cos =", c_sgl)
    var worst = c_img
    if c_txt < worst:
        worst = c_txt
    if c_sgl < worst:
        worst = c_sgl
    print("  WORST cos =", worst, " (task bar: cos >= 0.996)")
    print("NOTE: this file only REPORTS the numbers; the human gates on cos >= 0.996.")
