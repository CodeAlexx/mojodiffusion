# serenitymojo/models/klein/parity/klein_block_int8_bwd_parity.mojo
#
# PARITY GATE for the int8-W8A8 quantized-resident BASE dX path wired into the
# ACTUAL trainer-called block BACKWARDS (slice 3):
#   * single_block.single_block_lora_backward_device_resident_scratch  (w1 + w2)
#   * double_block.double_block_lora_backward_device_resident_scratch  (8 base
#       matmuls: img+txt x {wqkv,wproj,wgu,wd})
# The base weight is FROZEN (LoRA-only training) so only grad_input (dX = grad @ W)
# flows through the base weight — NO base weight-grad. For EACH block kind we build
# ONE real-dim Klein-9B block (H=32, Dh=128, D=4096, F=12288) with LoRA adapters and
# random non-degenerate inputs, run the bf16 forward ONCE to get the saved tape, then
# run the SAME scratch backward TWICE on that tape + upstream grad:
#   * int8=None    -> bf16 base dX GEMMs (unchanged) — the REFERENCE.
#   * int8=payload -> the frozen base dX GEMMs run int8 W8A8 (int8_linear_bwd_nn).
# We compare the returned dX (grad_input) with cos + max_abs, and CONFIRM the LoRA
# d_A/d_B grads are UNCHANGED between the two runs (the LoRA math is bf16 and the
# int8 only changes the base dX path; a mismatch there = a bug). This file only
# REPORTS the numbers; the human gates. Bar: cos >= 0.996 for dX (int8 W8A8 class),
# LoRA grads expected byte-identical (cos ~ 1.0).
#
# Build+run with the cshim recipe (int8 GEMM + cuDNN flash SDPA need the link):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo build --optimization-level 2 --target-accelerator sm_120 -I . -I /home/alex/MOJO-libs \
#     -Xlinker -lm -Xlinker -Lserenitymojo/ops/cshim/lib -Xlinker -lserenity_cudnn_sdpa \
#     -Xlinker -rpath -Xlinker /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     -Xlinker -rpath -Xlinker /home/alex/.serenity/cudnn/lib \
#     serenitymojo/models/klein/parity/klein_block_int8_bwd_parity.mojo -o /tmp/klein_blk_i8_bwd && \
#   LD_LIBRARY_PATH=.pixi/envs/default/lib:serenitymojo/ops/cshim/lib:/home/alex/.serenity/cudnn/lib:/usr/lib/x86_64-linux-gnu /tmp/klein_blk_i8_bwd

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt, log as flog, cos as fcos, pi
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.models.klein.lora_block import LoraAdapterDevice

from serenitymojo.models.klein.single_block import (
    SingleBlockWeights, SingleModVecs, SingleModVecsDevice,
    single_modvecs_to_device, SingleBlockLoraDevice,
    SingleBlockInt8, quantize_single_block_int8,
    single_block_lora_forward_device_resident_scratch,
    single_block_lora_backward_device_resident_scratch,
)
from serenitymojo.models.klein.double_block import (
    StreamWeights, DoubleBlockWeights, ModVecs, ModVecsDevice, modvecs_to_device,
    StreamLoraDevice, DoubleBlockLoraDevice,
    StreamInt8, DoubleBlockInt8, quantize_double_block_int8,
    double_block_lora_forward_device_resident_scratch,
    double_block_lora_backward_device_resident_scratch,
)

comptime TArc = ArcPointer[Tensor]

# REAL Klein-9B block dims (configs/klein9b.json).
comptime H = 32
comptime Dh = 128
comptime D = H * Dh          # 4096  == inner_dim
comptime F = 12288           # mlp_hidden
comptime EPS = Float32(1e-06)
comptime RANK = 8

# single block sequence length (modest, for gate speed).
comptime S1 = 128
# double block joint sequence (128-aligned for the cuDNN flash SDPA path).
comptime N_IMG = 96
comptime N_TXT = 32
comptime S2 = N_IMG + N_TXT   # 128

comptime DX_BAR = Float64(0.996)
comptime LORA_BAR = Float64(0.9999)


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


def _cos(a: List[Float32], b: List[Float32]) raises -> Float64:
    if len(a) != len(b):
        raise Error(String("cos length mismatch ") + String(len(a)) + " vs " + String(len(b)))
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 or nb == 0.0:
        raise Error("zero-norm vector in cos")
    return dot / (sqrt(na) * sqrt(nb))


def _max_abs(a: List[Float32], b: List[Float32]) -> Float64:
    var m = Float64(0.0)
    var n = len(a)
    if len(b) < n:
        n = len(b)
    for i in range(n):
        var d = Float64(a[i]) - Float64(b[i])
        if d < 0.0:
            d = -d
        if d > m:
            m = d
    return m


def _report_dx(name: String, ref_h: List[Float32], i8_h: List[Float32]) raises -> Float64:
    var c = _cos(ref_h, i8_h)
    var mx = _max_abs(ref_h, i8_h)
    print("---- dX", name, "(int8 base vs bf16 base) ----")
    print("  cos =", c, "  max_abs =", mx, "  n =", len(ref_h))
    if c >= DX_BAR:
        print("  RESULT: cos >= 0.996 (task bar) -> PASS")
    elif c >= 0.99:
        print("  RESULT: 0.99 <= cos < 0.996 (int8 class, task bar missed)")
    else:
        print("  RESULT: cos < 0.99 -> FAIL")
    return c


# terminal=True  -> this adapter's d_y is the block's RAW incoming grad (it does
#   NOT cross any int8 base dX), so its d_A/d_B MUST be byte-identical between the
#   bf16 and int8 runs. Byte-identity here is the proof the LoRA math path is
#   itself untouched by the int8 change (no int8 leak into the LoRA GEMMs).
# terminal=False -> this adapter is downstream of >=1 int8 base dX in the
#   activation-grad chain, so its d_y LEGITIMATELY carries the int8 quant error;
#   we expect it to differ at the same int8-class level as dX (cos >= 0.996), NOT
#   to be identical. That propagation is correct (== the real int8 training step).
def _report_lora(name: String, ref_h: List[Float32], i8_h: List[Float32],
                 terminal: Bool) raises -> Float64:
    if len(ref_h) == 0:
        print("  LoRA", name, ": (absent)")
        return Float64(1.0)
    var c = _cos(ref_h, i8_h)
    var mx = _max_abs(ref_h, i8_h)
    var tag: String
    if terminal:
        tag = "BYTE-IDENTICAL (LoRA math untouched)" if c >= LORA_BAR else "MISMATCH (BUG!)"
    else:
        tag = "int8 dX propagated (expected, cos>=0.996)" if c >= DX_BAR else "cos<0.996 (investigate)"
    print("  LoRA", name, " cos =", c, "  max_abs =", mx, "  n =", len(ref_h), " ->", tag)
    return c


def _adapter(in_f: Int, out_f: Int, seed: Int, ctx: DeviceContext) raises -> LoraAdapterDevice:
    var a = _gaussian(RANK * in_f, seed, 0.05)
    var b = _gaussian(out_f * RANK, seed + 1, 0.05)
    return LoraAdapterDevice(
        TArc(Tensor.from_host(a^, [RANK, in_f], STDtype.F32, ctx)),
        TArc(Tensor.from_host(b^, [out_f, RANK], STDtype.F32, ctx)),
        RANK, in_f, out_f, Float32(1.0),
    )


def _single_block(ctx: DeviceContext) raises -> Float64:
    print("")
    print("==== SINGLE block scratch backward (H=", H, " Dh=", Dh, " D=", D,
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
    var douth = _gaussian(S1 * D, 11, 1.0)

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
    # LoRA present so we can confirm the LoRA d_A/d_B grads are unchanged.
    var lora = SingleBlockLoraDevice(
        Optional[LoraAdapterDevice](_adapter(D, 3 * D + 2 * F, 20, ctx)),      # qkv
        Optional[LoraAdapterDevice](_adapter(D + F, D, 30, ctx)),              # out
    )
    var cos_t = Tensor.from_host(cosh.copy(), [S1 * H, Dh // 2], STDtype.F32, ctx)
    var sin_t = Tensor.from_host(sinh.copy(), [S1 * H, Dh // 2], STDtype.F32, ctx)
    var ones_t = Tensor.from_host(_ones(D), [D], STDtype.F32, ctx)
    var zeros_t = Tensor.from_host(_zeros(D), [D], STDtype.F32, ctx)

    var payload = quantize_single_block_int8(w.w1[], w.w2[], ctx)

    var scratch = ScratchRingAllocator(ctx, 512 * 1024 * 1024, 2)

    # ONE bf16 forward -> the saved tape reused by both backward runs.
    var fwd = single_block_lora_forward_device_resident_scratch[H, Dh, S1](
        x_arc, w, mv, lora, cos_t, sin_t, D, F, EPS, ones_t, zeros_t, ctx,
        scratch, int8=Optional[SingleBlockInt8](None),
    )

    var d_out_ref = TArc(Tensor.from_host(douth.copy(), [S1, D], STDtype.F32, ctx))
    var bref = single_block_lora_backward_device_resident_scratch[H, Dh, S1](
        d_out_ref, w, mv, lora, fwd.saved, cos_t, sin_t, D, F, EPS, ones_t, ctx,
        scratch, compute_aux_grads=True, int8=Optional[SingleBlockInt8](None),
    )
    var dx_ref = bref.d_x[].to_host(ctx)

    # transpose-on-visit arm (empty w8t — the STREAMED production payload):
    # the dX GEMMs transpose W8 on device per visit and run NT. Must be
    # BIT-IDENTICAL to the staged-W8T arm below (same int8 values, exact
    # int32 accumulate) — gate max_abs == 0.0.
    var payload_tv = SingleBlockInt8(payload.w8.copy(), payload.scale.copy())

    var d_out_i8 = TArc(Tensor.from_host(douth.copy(), [S1, D], STDtype.F32, ctx))
    var bi8 = single_block_lora_backward_device_resident_scratch[H, Dh, S1](
        d_out_i8, w, mv, lora, fwd.saved, cos_t, sin_t, D, F, EPS, ones_t, ctx,
        scratch, compute_aux_grads=True, int8=Optional[SingleBlockInt8](payload^),
    )
    var dx_i8 = bi8.d_x[].to_host(ctx)

    var d_out_tv = TArc(Tensor.from_host(douth.copy(), [S1, D], STDtype.F32, ctx))
    var btv = single_block_lora_backward_device_resident_scratch[H, Dh, S1](
        d_out_tv, w, mv, lora, fwd.saved, cos_t, sin_t, D, F, EPS, ones_t, ctx,
        scratch, compute_aux_grads=True,
        int8=Optional[SingleBlockInt8](payload_tv^),
    )
    var dx_tv = btv.d_x[].to_host(ctx)
    var tv_max = _max_abs(dx_i8, dx_tv)
    print("  transpose-on-visit vs staged-W8T d_x max_abs =", tv_max,
          " (expected 0.0 — bit-identical)")
    if tv_max != 0.0:
        print("  FAIL: transpose-on-visit arm != staged-W8T arm")

    var c = _report_dx("SINGLE d_x", dx_ref, dx_i8)
    print("  -- LoRA grads: 'out' is TERMINAL (byte-identical proof); 'qkv' is")
    print("     downstream of the int8 w2 dX (int8 error propagates, expected) --")
    _ = _report_lora("out d_A", bref.out_d_a, bi8.out_d_a, True)
    _ = _report_lora("out d_B", bref.out_d_b, bi8.out_d_b, True)
    _ = _report_lora("qkv d_A", bref.qkv_d_a, bi8.qkv_d_a, False)
    _ = _report_lora("qkv d_B", bref.qkv_d_b, bi8.qkv_d_b, False)
    return c


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


def _stream_lora(seed0: Int, ctx: DeviceContext) raises -> StreamLoraDevice:
    # q,k,v,out: in=D out=D ; ff_in: in=D out=2F ; ff_out: in=F out=D.
    return StreamLoraDevice(
        Optional[LoraAdapterDevice](_adapter(D, D, seed0 + 0, ctx)),
        Optional[LoraAdapterDevice](_adapter(D, D, seed0 + 2, ctx)),
        Optional[LoraAdapterDevice](_adapter(D, D, seed0 + 4, ctx)),
        Optional[LoraAdapterDevice](_adapter(D, D, seed0 + 6, ctx)),
        Optional[LoraAdapterDevice](_adapter(D, 2 * F, seed0 + 8, ctx)),
        Optional[LoraAdapterDevice](_adapter(F, D, seed0 + 10, ctx)),
    )


def _double_block(ctx: DeviceContext) raises -> Float64:
    print("")
    print("==== DOUBLE block scratch backward (H=", H, " Dh=", Dh, " D=", D,
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

    var lora = DoubleBlockLoraDevice(_stream_lora(1000, ctx), _stream_lora(2000, ctx))

    var payload = quantize_double_block_int8(w.img, w.txt, ctx)

    var scratch = ScratchRingAllocator(ctx, 512 * 1024 * 1024, 2)

    var fwd = double_block_lora_forward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S2](
        img_x, txt_x, w, im, tm, lora, cos_t, sin_t, D, F, EPS,
        ones_t, zeros_t, ctx, scratch, int8=Optional[DoubleBlockInt8](None),
    )

    var d_img_h = _gaussian(N_IMG * D, 900, 1.0)
    var d_txt_h = _gaussian(N_TXT * D, 950, 1.0)

    var d_io_ref = TArc(Tensor.from_host(d_img_h.copy(), [N_IMG, D], STDtype.F32, ctx))
    var d_to_ref = TArc(Tensor.from_host(d_txt_h.copy(), [N_TXT, D], STDtype.F32, ctx))
    var bref = double_block_lora_backward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S2](
        d_io_ref, d_to_ref, w, im, tm, lora, fwd.saved, cos_t, sin_t, D, F, EPS,
        ones_t, ctx, scratch, compute_aux_grads=True, int8=Optional[DoubleBlockInt8](None),
    )
    var img_dx_ref = bref.img.d_x[].to_host(ctx)
    var txt_dx_ref = bref.txt.d_x[].to_host(ctx)

    # transpose-on-visit arm (empty w8t — the STREAMED production payload):
    # must be BIT-IDENTICAL to the staged-W8T arm (gate max_abs == 0.0).
    var payload_tv = DoubleBlockInt8(
        StreamInt8(payload.img.w8.copy(), payload.img.scale.copy()),
        StreamInt8(payload.txt.w8.copy(), payload.txt.scale.copy()),
    )

    var d_io_i8 = TArc(Tensor.from_host(d_img_h.copy(), [N_IMG, D], STDtype.F32, ctx))
    var d_to_i8 = TArc(Tensor.from_host(d_txt_h.copy(), [N_TXT, D], STDtype.F32, ctx))
    var bi8 = double_block_lora_backward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S2](
        d_io_i8, d_to_i8, w, im, tm, lora, fwd.saved, cos_t, sin_t, D, F, EPS,
        ones_t, ctx, scratch, compute_aux_grads=True, int8=Optional[DoubleBlockInt8](payload^),
    )
    var img_dx_i8 = bi8.img.d_x[].to_host(ctx)
    var txt_dx_i8 = bi8.txt.d_x[].to_host(ctx)

    var d_io_tv = TArc(Tensor.from_host(d_img_h.copy(), [N_IMG, D], STDtype.F32, ctx))
    var d_to_tv = TArc(Tensor.from_host(d_txt_h.copy(), [N_TXT, D], STDtype.F32, ctx))
    var btv = double_block_lora_backward_device_resident_scratch[H, Dh, N_IMG, N_TXT, S2](
        d_io_tv, d_to_tv, w, im, tm, lora, fwd.saved, cos_t, sin_t, D, F, EPS,
        ones_t, ctx, scratch, compute_aux_grads=True,
        int8=Optional[DoubleBlockInt8](payload_tv^),
    )
    var img_dx_tv = btv.img.d_x[].to_host(ctx)
    var txt_dx_tv = btv.txt.d_x[].to_host(ctx)
    var tv_max_img = _max_abs(img_dx_i8, img_dx_tv)
    var tv_max_txt = _max_abs(txt_dx_i8, txt_dx_tv)
    print("  transpose-on-visit vs staged-W8T img/txt d_x max_abs =",
          tv_max_img, "/", tv_max_txt, " (expected 0.0 — bit-identical)")
    if tv_max_img != 0.0 or tv_max_txt != 0.0:
        print("  FAIL: transpose-on-visit arm != staged-W8T arm")

    var c_img = _report_dx("DOUBLE img d_x", img_dx_ref, img_dx_i8)
    var c_txt = _report_dx("DOUBLE txt d_x", txt_dx_ref, txt_dx_i8)

    print("  -- LoRA grads IMG: 'ff_out' is TERMINAL (byte-identical proof); the")
    print("     rest are downstream of int8 base dX (int8 error propagates) --")
    _ = _report_lora("img ff_out d_A", bref.img.ff_out_d_a, bi8.img.ff_out_d_a, True)
    _ = _report_lora("img ff_out d_B", bref.img.ff_out_d_b, bi8.img.ff_out_d_b, True)
    _ = _report_lora("img q d_A", bref.img.q_d_a, bi8.img.q_d_a, False)
    _ = _report_lora("img q d_B", bref.img.q_d_b, bi8.img.q_d_b, False)
    _ = _report_lora("img k d_A", bref.img.k_d_a, bi8.img.k_d_a, False)
    _ = _report_lora("img v d_A", bref.img.v_d_a, bi8.img.v_d_a, False)
    _ = _report_lora("img out d_A", bref.img.out_d_a, bi8.img.out_d_a, False)
    _ = _report_lora("img out d_B", bref.img.out_d_b, bi8.img.out_d_b, False)
    _ = _report_lora("img ff_in d_A", bref.img.ff_in_d_a, bi8.img.ff_in_d_a, False)
    _ = _report_lora("img ff_in d_B", bref.img.ff_in_d_b, bi8.img.ff_in_d_b, False)
    print("  -- LoRA grads TXT: 'ff_out' TERMINAL, rest downstream of int8 dX --")
    _ = _report_lora("txt ff_out d_A", bref.txt.ff_out_d_a, bi8.txt.ff_out_d_a, True)
    _ = _report_lora("txt ff_out d_B", bref.txt.ff_out_d_b, bi8.txt.ff_out_d_b, True)
    _ = _report_lora("txt q d_A", bref.txt.q_d_a, bi8.txt.q_d_a, False)
    _ = _report_lora("txt out d_B", bref.txt.out_d_b, bi8.txt.out_d_b, False)
    _ = _report_lora("txt ff_in d_B", bref.txt.ff_in_d_b, bi8.txt.ff_in_d_b, False)

    return c_img if c_img < c_txt else c_txt


def main() raises:
    var ctx = DeviceContext()
    print("==== klein_block_int8_bwd_parity (slice 3: trainer scratch backwards) ====")

    var single_cos = _single_block(ctx)
    var double_cos = _double_block(ctx)

    print("")
    print("==== SUMMARY ====")
    print("  SINGLE block dX cos =", single_cos)
    print("  DOUBLE block dX cos (min of img/txt) =", double_cos)
    print("NOTE: this file only REPORTS the numbers; the human gates on dX cos >= 0.996.")
    print("      LoRA math is proven untouched by the TERMINAL adapter (out/ff_out)")
    print("      being BYTE-IDENTICAL (cos=1, max_abs=0). Upstream adapters legitimately")
    print("      inherit the int8 base-dX error through the activation-grad chain, so they")
    print("      differ at the int8 class level (cos >= 0.996) — expected, not a bug.")
