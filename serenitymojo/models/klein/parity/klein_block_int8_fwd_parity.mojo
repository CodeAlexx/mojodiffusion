# serenitymojo/models/klein/parity/klein_block_int8_fwd_parity.mojo
#
# PARITY GATE for the int8-W8A8 quantized-resident BASE path wired into the ACTUAL
# trainer-called block forwards (slice 2):
#   * single_block.single_block_lora_forward_device_resident_scratch  (w1 + w2)
#   * double_block.double_block_lora_forward_device_resident_scratch  (8 base
#       matmuls: img+txt × {wqkv,wproj,wgu,wd})
# For EACH block kind we build ONE real-dim Klein-9B block (H=32, Dh=128, D=4096,
# F=12288) with random non-degenerate inputs, then run the SAME scratch forward
# twice:
#   * int8=None    → the bf16 base matmuls (unchanged) — the REFERENCE.
#   * int8=payload → the frozen base matmuls run int8 W8A8 (int8_linear_fwd).
# LoRA is absent in both (base-only). We compare the two block outputs with
# ParityHarness and print cos + max_abs. This file only REPORTS; the human gates.
# Bar: cos >= 0.996 (int8 W8A8 whole-block class).
#
# Build+run with the cshim recipe (int8 GEMM + cuDNN flash SDPA need the link):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 2 --target-accelerator sm_120 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     -Xlinker -rpath -Xlinker /home/alex/.serenity/cudnn/lib \
#     serenitymojo/models/klein/parity/klein_block_int8_fwd_parity.mojo -o /tmp/klein_blk_i8_fwd && \
#   LD_LIBRARY_PATH=.pixi/envs/default/lib:serenitymojo/ops/cshim/lib:/home/alex/.serenity/cudnn/lib:/usr/lib/x86_64-linux-gnu /tmp/klein_blk_i8_fwd

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt, log as flog, cos as fcos, pi
from std.memory import ArcPointer
from serenitymojo.parity import ParityHarness, ParityResult
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.scratch_ring import ScratchRingAllocator

from serenitymojo.models.klein.single_block import (
    SingleBlockWeights, SingleModVecs, SingleModVecsDevice,
    single_modvecs_to_device, SingleBlockLoraDevice,
    SingleBlockInt8, quantize_single_block_int8,
    single_block_lora_forward_device_resident_scratch,
)
from serenitymojo.models.klein.double_block import (
    StreamWeights, DoubleBlockWeights, ModVecs, ModVecsDevice, modvecs_to_device,
    StreamLoraDevice, DoubleBlockLoraDevice,
    DoubleBlockInt8, quantize_double_block_int8,
    double_block_lora_forward_device_resident_scratch,
)

comptime TArc = ArcPointer[Tensor]

# REAL Klein-9B block dims (configs/klein9b.json).
comptime H = 32
comptime Dh = 128
comptime D = H * Dh          # 4096  == inner_dim
comptime F = 12288           # mlp_hidden
comptime EPS = Float32(1e-06)

# single block sequence length (modest, for gate speed).
comptime S1 = 128
# double block joint sequence (128-aligned for the cuDNN flash SDPA path).
comptime N_IMG = 96
comptime N_TXT = 32
comptime S2 = N_IMG + N_TXT   # 128


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


def _report(mut harness: ParityHarness, name: String, t: Tensor,
            ref_h: List[Float32], ctx: DeviceContext) raises -> Float64:
    var r = harness.compare(t, ref_h, ctx)
    print("---- ", name, " int8 base vs bf16 base ----")
    print("  cos =", r.cos, "  max_abs =", r.max_abs, "  n =", r.n)
    if r.cos >= 0.996:
        print("  RESULT: cos >= 0.996 (task bar) -> PASS")
    elif r.cos >= 0.99:
        print("  RESULT: 0.99 <= cos < 0.996 (krea2 int8 bar met, task bar missed)")
    else:
        print("  RESULT: cos < 0.99 -> FAIL")
    return r.cos


def _single_block(ctx: DeviceContext, mut harness: ParityHarness) raises -> Float64:
    print("")
    print("==== SINGLE block scratch forward (H=", H, " Dh=", Dh, " D=", D,
          " F=", F, " S=", S1, ") ====")
    var xh = _gaussian(S1 * D, 1, 1.0)
    var w1h = _gaussian((3 * D + 2 * F) * D, 2, 0.02)
    var w2h = _gaussian(D * (D + F), 3, 0.02)
    var qnh = _gaussian(Dh, 4, 1.0)
    var knh = _gaussian(Dh, 5, 1.0)
    var shifth = _gaussian(D, 6, 0.5)
    var scaleh = _gaussian(D, 7, 0.5)
    var gateh = _gaussian(D, 8, 0.5)
    var cosh = _gaussian(S1 * H * (Dh // 2), 9, 1.0)
    var sinh = _gaussian(S1 * H * (Dh // 2), 10, 1.0)

    # F32 activations, BF16 base weights (== the trainer regime).
    var x_arc = TArc(Tensor.from_host(xh.copy(), [S1, D], STDtype.F32, ctx))
    var w1_arc = TArc(Tensor.from_host(w1h.copy(), [3 * D + 2 * F, D], STDtype.BF16, ctx))
    var w2_arc = TArc(Tensor.from_host(w2h.copy(), [D, D + F], STDtype.BF16, ctx))
    var qn_arc = TArc(Tensor.from_host(qnh.copy(), [Dh], STDtype.F32, ctx))
    var kn_arc = TArc(Tensor.from_host(knh.copy(), [Dh], STDtype.F32, ctx))
    var w = SingleBlockWeights(w1_arc^, w2_arc^, qn_arc^, kn_arc^, D, F, ctx, True)

    var mv = single_modvecs_to_device(
        SingleModVecs(shifth.copy(), scaleh.copy(), gateh.copy()), D, ctx
    )
    var lora = SingleBlockLoraDevice(None, None)
    var cos_t = Tensor.from_host(cosh.copy(), [S1 * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sinh.copy(), [S1 * H, Dh // 2], STDtype.F32, ctx)
    var ones_t = Tensor.from_host(_ones(D), [D], STDtype.F32, ctx)
    var zeros_t = Tensor.from_host(_zeros(D), [D], STDtype.F32, ctx)

    var payload = quantize_single_block_int8(w.w1[], w.w2[], ctx)

    var scratch_ref = ScratchRingAllocator(ctx, 512 * 1024 * 1024, 2)
    var ref_fwd = single_block_lora_forward_device_resident_scratch[H, Dh, S1](
        x_arc, w, mv, lora, cos_t, sin_t, D, F, EPS, ones_t, zeros_t, ctx,
        scratch_ref, int8=Optional[SingleBlockInt8](None),
    )
    var ref_h = ref_fwd.out[].to_host(ctx)

    var scratch_i8 = ScratchRingAllocator(ctx, 512 * 1024 * 1024, 2)
    var i8_fwd = single_block_lora_forward_device_resident_scratch[H, Dh, S1](
        x_arc, w, mv, lora, cos_t, sin_t, D, F, EPS, ones_t, zeros_t, ctx,
        scratch_i8, int8=Optional[SingleBlockInt8](payload^),
    )
    return _report(harness, "SINGLE block", i8_fwd.out[], ref_h, ctx)


def _stream(seed0: Int, ctx: DeviceContext) raises -> StreamWeights:
    var wqkv = _gaussian((3 * D) * D, seed0 + 0, 0.02)
    var wproj = _gaussian(D * D, seed0 + 1, 0.02)
    var wgu = _gaussian((2 * F) * D, seed0 + 2, 0.02)
    var wd = _gaussian(D * F, seed0 + 3, 0.02)
    var qn = _gaussian(Dh, seed0 + 4, 1.0)
    var kn = _gaussian(Dh, seed0 + 5, 1.0)
    return StreamWeights(
        TArc(Tensor.from_host(wqkv, [3 * D, D], STDtype.BF16, ctx)),
        TArc(Tensor.from_host(wproj, [D, D], STDtype.BF16, ctx)),
        TArc(Tensor.from_host(wgu, [2 * F, D], STDtype.BF16, ctx)),
        TArc(Tensor.from_host(wd, [D, F], STDtype.BF16, ctx)),
        TArc(Tensor.from_host(qn, [Dh], STDtype.F32, ctx)),
        TArc(Tensor.from_host(kn, [Dh], STDtype.F32, ctx)),
    )


def _mod(seed0: Int) raises -> ModVecs:
    return ModVecs(
        _gaussian(D, seed0 + 0, 0.5), _gaussian(D, seed0 + 1, 0.5),
        _gaussian(D, seed0 + 2, 0.5), _gaussian(D, seed0 + 3, 0.5),
        _gaussian(D, seed0 + 4, 0.5), _gaussian(D, seed0 + 5, 0.5),
    )


def _double_block(ctx: DeviceContext, mut harness: ParityHarness) raises -> Float64:
    print("")
    print("==== DOUBLE block scratch forward (H=", H, " Dh=", Dh, " D=", D,
          " F=", F, " N_IMG=", N_IMG, " N_TXT=", N_TXT, ") ====")

    var iw = _stream(100, ctx)
    var tw = _stream(200, ctx)
    var w = DoubleBlockWeights(iw^, tw^)
    var im = modvecs_to_device(_mod(300), D, ctx)
    var tm = modvecs_to_device(_mod(400), D, ctx)

    var img_x = TArc(Tensor.from_host(_gaussian(N_IMG * D, 500, 1.0), [N_IMG, D], STDtype.F32, ctx))
    var txt_x = TArc(Tensor.from_host(_gaussian(N_TXT * D, 600, 1.0), [N_TXT, D], STDtype.F32, ctx))

    var cosh = _gaussian(S2 * H * (Dh // 2), 700, 1.0)
    var sinh = _gaussian(S2 * H * (Dh // 2), 800, 1.0)
    var cos_t = Tensor.from_host(cosh, [S2 * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sinh, [S2 * H, Dh // 2], STDtype.F32, ctx)
    var ones_t = Tensor.from_host(_ones(D), [D], STDtype.F32, ctx)
    var zeros_t = Tensor.from_host(_zeros(D), [D], STDtype.F32, ctx)

    var empty_img = StreamLoraDevice(None, None, None, None, None, None)
    var empty_txt = StreamLoraDevice(None, None, None, None, None, None)
    var lora = DoubleBlockLoraDevice(empty_img^, empty_txt^)

    var payload = quantize_double_block_int8(w.img, w.txt, ctx)

    var scratch_ref = ScratchRingAllocator(ctx, 512 * 1024 * 1024, 2)
    var ref_fwd = double_block_lora_forward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S2](
        img_x, txt_x, w, im, tm, lora, cos_t, sin_t, D, F, EPS,
        ones_t, zeros_t, ctx, scratch_ref, int8=Optional[DoubleBlockInt8](None),
    )
    var ref_img_h = ref_fwd.img_out[].to_host(ctx)

    var scratch_i8 = ScratchRingAllocator(ctx, 512 * 1024 * 1024, 2)
    var i8_fwd = double_block_lora_forward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S2](
        img_x, txt_x, w, im, tm, lora, cos_t, sin_t, D, F, EPS,
        ones_t, zeros_t, ctx, scratch_i8, int8=Optional[DoubleBlockInt8](payload^),
    )
    # Report on the img stream output (txt reported alongside for completeness).
    var c_img = _report(harness, "DOUBLE block img_out", i8_fwd.img_out[], ref_img_h, ctx)
    var ref_txt_h = ref_fwd.txt_out[].to_host(ctx)
    var c_txt = _report(harness, "DOUBLE block txt_out", i8_fwd.txt_out[], ref_txt_h, ctx)
    return c_img if c_img < c_txt else c_txt


def main() raises:
    var ctx = DeviceContext()
    print("==== klein_block_int8_fwd_parity (slice 2: trainer scratch forwards) ====")
    var harness = ParityHarness(0.996)

    var single_cos = _single_block(ctx, harness)
    var double_cos = _double_block(ctx, harness)

    print("")
    print("==== SUMMARY ====")
    print("  SINGLE block cos =", single_cos)
    print("  DOUBLE block cos (min of img/txt) =", double_cos)
    print("NOTE: this file only REPORTS the numbers; the human gates on cos >= 0.996.")
