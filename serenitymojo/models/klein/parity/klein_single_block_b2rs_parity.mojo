# models/klein/parity/klein_single_block_b2rs_parity.mojo
#
# TRUE-batch row-stacked single-block gate (klein b2rs rung A).
# Reference = the established b1 pair run per sample (math sdpa path):
#   single_block_lora_forward_device_resident_scratch  (x0), (x1)
#   single_block_lora_backward_device_resident_scratch (compute_aux_grads=False)
# Test = the new batched pair ONCE on the row-stacked [2S, D] input with
#   [2, D] mod packs, B=2 cuDNN flash, and the duplicated rope table.
# Bars: value-tolerance (flash-vs-math class, sdpa_flash_parity precedent):
#   fwd out / d_x per-sample cos >= 0.999999; shared LoRA d_A/d_B vs the SUM of
#   the two per-sample host grads cos >= 0.999999. FAIL LOUD below bar.
#
# Build+run (cuDNN shim link REQUIRED for the flash arm):
#   cd /home/alex/mojodiffusion && pixi run mojo build --optimization-level 2 \
#     -I . -Xlinker -lm -Xlinker -lcuda -Xlinker -Lserenitymojo/ops/cshim/lib \
#     -Xlinker -lserenity_cudnn_sdpa -Xlinker -rpath -Xlinker \
#     /home/alex/mojodiffusion/serenitymojo/ops/cshim/lib \
#     serenitymojo/models/klein/parity/klein_single_block_b2rs_parity.mojo \
#     -o output/bin/klein_single_block_b2rs_parity && \
#   output/bin/klein_single_block_b2rs_parity

from std.collections import List, Optional
from std.math import sqrt
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.models.klein.lora_block import LoraAdapterDevice
from serenitymojo.models.klein.single_block import (
    SingleBlockWeights, SingleModVecsDevice, SingleBlockLoraDevice,
    single_block_lora_forward_device_resident_scratch,
    single_block_lora_backward_device_resident_scratch,
    single_block_lora_forward_device_resident_scratch_batch,
    single_block_lora_backward_device_resident_scratch_tensors_batch,
)

comptime TArc = ArcPointer[Tensor]
comptime B = 2
comptime H = 2
comptime Dh = 128
comptime D = H * Dh
comptime F = 512
comptime S = 128
comptime ROWS = B * S
comptime RANK = 8
comptime EPS = Float32(1.0e-6)
comptime COS_BAR = Float32(0.999999)


def _fill(n: Int, seed: Int) -> List[Float32]:
    # deterministic LCG in [-0.5, 0.5]
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = (state * 1103515245 + 12345) % 2147483648
        out.append(Float32(state) / Float32(2147483648.0) - Float32(0.5))
    return out^


def _dup(a: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(a)):
        out.append(a[i])
    for i in range(len(a)):
        out.append(a[i])
    return out^


def _cat(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(a)):
        out.append(a[i])
    for i in range(len(b)):
        out.append(b[i])
    return out^


def _sum2(a: List[Float32], b: List[Float32]) raises -> List[Float32]:
    if len(a) != len(b):
        raise Error("b2rs smoke: grad length mismatch")
    var out = List[Float32]()
    for i in range(len(a)):
        out.append(a[i] + b[i])
    return out^


def _cos(a: List[Float32], b: List[Float32]) raises -> Float32:
    if len(a) != len(b):
        raise Error("b2rs smoke: cos length mismatch")
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    for i in range(len(a)):
        dot += Float64(a[i]) * Float64(b[i])
        na += Float64(a[i]) * Float64(a[i])
        nb += Float64(b[i]) * Float64(b[i])
    if na == 0.0 or nb == 0.0:
        raise Error("b2rs smoke: zero-norm vector in cos")
    return Float32(dot / (sqrt(na) * sqrt(nb)))


def _max_abs_diff(a: List[Float32], b: List[Float32]) -> Float32:
    var m = Float32(0.0)
    var n = len(a)
    if len(b) < n:
        n = len(b)
    for i in range(n):
        var d = a[i] - b[i]
        if d < 0:
            d = -d
        if d > m:
            m = d
    return m


def _check(name: String, got: Float32, bar: Float32) raises:
    print("[b2rs]", name, "cos=", got, "bar=", bar)
    if got < bar:
        raise Error(String("klein b2rs parity FAIL: ") + name)


def _slice_rows(a: List[Float32], start: Int, count: Int) -> List[Float32]:
    var out = List[Float32]()
    for i in range(count):
        out.append(a[start + i])
    return out^


def _t(var vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals^, shape^, STDtype.F32, ctx)


def _arc(var vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> TArc:
    return TArc(_t(vals^, shape^, ctx))


def main() raises:
    var ctx = DeviceContext()
    var scratch = ScratchRingAllocator(ctx, 256 * 1024 * 1024, 2)

    # weights (shared)
    var w1 = _fill((3 * D + 2 * F) * D, 11)
    var w2 = _fill(D * (D + F), 22)
    var qn = _fill(Dh, 33)
    var kn = _fill(Dh, 44)
    var w = SingleBlockWeights(w1^, w2^, qn^, kn^, D, F, Dh, ctx)

    # per-sample mod vecs
    var shift0 = _fill(D, 55)
    var scale0 = _fill(D, 66)
    var gate0 = _fill(D, 77)
    var shift1 = _fill(D, 88)
    var scale1 = _fill(D, 99)
    var gate1 = _fill(D, 111)

    var mv0 = SingleModVecsDevice(
        _arc(shift0.copy(), [D], ctx),
        _arc(scale0.copy(), [D], ctx),
        _arc(gate0.copy(), [D], ctx),
    )
    var mv1 = SingleModVecsDevice(
        _arc(shift1.copy(), [D], ctx),
        _arc(scale1.copy(), [D], ctx),
        _arc(gate1.copy(), [D], ctx),
    )
    var mvb = SingleModVecsDevice(
        _arc(_cat(shift0, shift1), [2, D], ctx),
        _arc(_cat(scale0, scale1), [2, D], ctx),
        _arc(_cat(gate0, gate1), [2, D], ctx),
    )

    # shared LoRA adapters (qkv fused + out)
    var qkv_a = _fill(RANK * D, 121)
    var qkv_b = _fill((3 * D + 2 * F) * RANK, 131)
    var out_a = _fill(RANK * (D + F), 141)
    var out_b = _fill(D * RANK, 151)
    var lora = SingleBlockLoraDevice(
        Optional[LoraAdapterDevice](LoraAdapterDevice(
            _arc(qkv_a.copy(), [RANK, D], ctx),
            _arc(qkv_b.copy(), [3 * D + 2 * F, RANK], ctx),
            RANK, D, 3 * D + 2 * F, Float32(1.0),
        )),
        Optional[LoraAdapterDevice](LoraAdapterDevice(
            _arc(out_a.copy(), [RANK, D + F], ctx),
            _arc(out_b.copy(), [D, RANK], ctx),
            RANK, D + F, D, Float32(1.0),
        )),
    )

    # rope tables: b1 [S*H, Dh/2]; batched = duplicated rows
    var cos_vals = _fill(S * H * (Dh // 2), 161)
    var sin_vals = _fill(S * H * (Dh // 2), 171)
    var cos1 = _t(cos_vals.copy(), [S * H, Dh // 2], ctx)
    var sin1 = _t(sin_vals.copy(), [S * H, Dh // 2], ctx)
    var cos2 = _t(_dup(cos_vals), [ROWS * H, Dh // 2], ctx)
    var sin2 = _t(_dup(sin_vals), [ROWS * H, Dh // 2], ctx)

    var ones = _t(_fill_ones(), [D], ctx)
    var zeros = _t(_fill_zeros(), [D], ctx)

    # inputs + upstream grads
    var x0 = _fill(S * D, 181)
    var x1 = _fill(S * D, 191)
    var g0 = _fill(S * D, 201)
    var g1 = _fill(S * D, 211)

    # ── reference: b1 pair per sample (math sdpa) ────────────────────────────
    var f0 = single_block_lora_forward_device_resident_scratch[H, Dh, S](
        _arc(x0.copy(), [S, D], ctx), w, mv0, lora, cos1, sin1,
        D, F, EPS, ones, zeros, ctx, scratch,
    )
    var b0 = single_block_lora_backward_device_resident_scratch[H, Dh, S](
        _arc(g0.copy(), [S, D], ctx), w, mv0, lora, f0.saved, cos1, sin1,
        D, F, EPS, ones, ctx, scratch, False,
    )
    var f1 = single_block_lora_forward_device_resident_scratch[H, Dh, S](
        _arc(x1.copy(), [S, D], ctx), w, mv1, lora, cos1, sin1,
        D, F, EPS, ones, zeros, ctx, scratch,
    )
    var b1 = single_block_lora_backward_device_resident_scratch[H, Dh, S](
        _arc(g1.copy(), [S, D], ctx), w, mv1, lora, f1.saved, cos1, sin1,
        D, F, EPS, ones, ctx, scratch, False,
    )

    # ── test: batched pair once on row-stacked input ─────────────────────────
    var fb = single_block_lora_forward_device_resident_scratch_batch[B, H, Dh, S](
        _arc(_cat(x0, x1), [ROWS, D], ctx), w, mvb, lora, cos2, sin2,
        D, F, EPS, ones, zeros, ctx, scratch,
    )
    var bb = single_block_lora_backward_device_resident_scratch_tensors_batch[B, H, Dh, S](
        _arc(_cat(g0, g1), [ROWS, D], ctx), w, mvb, lora, fb.saved, cos2, sin2,
        D, F, EPS, ones, ctx, scratch,
    )

    # ── compare ───────────────────────────────────────────────────────────────
    var out_ref0 = f0.out[].to_host(ctx)
    var out_ref1 = f1.out[].to_host(ctx)
    var out_b2 = fb.out[].to_host(ctx)
    _check("fwd_out_s0", _cos(out_ref0, _slice_rows(out_b2, 0, S * D)), COS_BAR)
    _check("fwd_out_s1", _cos(out_ref1, _slice_rows(out_b2, S * D, S * D)), COS_BAR)

    var dx_ref0 = b0.d_x[].to_host(ctx)
    var dx_ref1 = b1.d_x[].to_host(ctx)
    var dx_b2 = bb.d_x[].to_host(ctx)
    _check("d_x_s0", _cos(dx_ref0, _slice_rows(dx_b2, 0, S * D)), COS_BAR)
    _check("d_x_s1", _cos(dx_ref1, _slice_rows(dx_b2, S * D, S * D)), COS_BAR)

    if not bb.qkv_d_a or not bb.qkv_d_b or not bb.out_d_a or not bb.out_d_b:
        raise Error("klein b2rs parity FAIL: batched LoRA grads missing")
    var qkv_da_sum = _sum2(b0.qkv_d_a, b1.qkv_d_a)
    var qkv_db_sum = _sum2(b0.qkv_d_b, b1.qkv_d_b)
    var out_da_sum = _sum2(b0.out_d_a, b1.out_d_a)
    var out_db_sum = _sum2(b0.out_d_b, b1.out_d_b)
    var qkv_da_b2 = bb.qkv_d_a.value()[].to_host(ctx)
    var qkv_db_b2 = bb.qkv_d_b.value()[].to_host(ctx)
    var out_da_b2 = bb.out_d_a.value()[].to_host(ctx)
    var out_db_b2 = bb.out_d_b.value()[].to_host(ctx)
    _check("lora_qkv_d_a", _cos(qkv_da_sum, qkv_da_b2), COS_BAR)
    _check("lora_qkv_d_b", _cos(qkv_db_sum, qkv_db_b2), COS_BAR)
    _check("lora_out_d_a", _cos(out_da_sum, out_da_b2), COS_BAR)
    _check("lora_out_d_b", _cos(out_db_sum, out_db_b2), COS_BAR)
    print("[b2rs] max_abs fwd_s0=", _max_abs_diff(out_ref0, _slice_rows(out_b2, 0, S * D)))
    print("[b2rs] max_abs qkv_d_b=", _max_abs_diff(qkv_db_sum, qkv_db_b2))

    print("KLEIN SINGLE-BLOCK B2RS PARITY: ALL PASS")


def _fill_ones() -> List[Float32]:
    var out = List[Float32]()
    for _ in range(D):
        out.append(Float32(1.0))
    return out^


def _fill_zeros() -> List[Float32]:
    var out = List[Float32]()
    for _ in range(D):
        out.append(Float32(0.0))
    return out^
