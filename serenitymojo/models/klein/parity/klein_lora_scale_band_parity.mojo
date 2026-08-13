# serenitymojo/models/klein/parity/klein_lora_scale_band_parity.mojo
#
# PARITY GATE for two Klein LoRA-chain fusions (2026-07-11):
#
# 1. ALPHA-FOLD (single_block._klein_lora_fwd_dropout, F32 arm):
#      legacy: dy = linear(t, B) [F32 C = exact accumulator]; mul_scalar(dy, s)
#      fused:  linear_rows(t, B, 0, out_f, alpha=s)
#    Both apply ONE F32 multiply fl(s·acc) to the exact F32 accumulator (the
#    cuBLAS F32 epilogue alpha == mul_scalar on the exactly-stored C), so the
#    gate demands BIT-IDENTITY: max_abs == 0.0.
#
# 2. BANDED IN-PLACE ADD (ops/tensor_algebra.add_band_in_place_f32):
#      legacy: add_in_place_f32(dst, slice(src, 1, off, W))
#      fused:  add_band_in_place_f32(dst, src, off)
#    Same F32 add, same slice element mapping → BIT-IDENTITY: max_abs == 0.0.
#
# Build+run:
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#     serenitymojo/models/klein/parity/klein_lora_scale_band_parity.mojo

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear, linear_rows
from serenitymojo.ops.tensor_algebra import (
    slice, mul_scalar, mul_scalar_bf16out, add_in_place_f32, add_band_in_place_f32,
)
from serenitymojo.ops.cast import cast_tensor


def _lcg(mut state: UInt64) -> Float32:
    state = state * 6364136223846793005 + 1442695040888963407
    var u = (state >> 40) & 0xFFFFFF
    return (Float32(Int(u)) / Float32(1 << 23)) - 1.0


def _rand(n: Int, mut state: UInt64, scale: Float32) -> List[Float32]:
    var v = List[Float32]()
    for _ in range(n):
        v.append(_lcg(state) * scale)
    return v^


def _assert_bit_identical(name: String, got: Tensor, expected: Tensor, ctx: DeviceContext) raises:
    var ref_h = expected.to_host(ctx)
    var h = ParityHarness(0.999999)
    var r = h.compare(got, ref_h, ctx)
    print(name, " cos=", r.cos, " max_abs=", r.max_abs, " pass=", r.passed)
    if r.max_abs != 0.0:
        raise Error("lora scale/band fusion parity FAILED (not bit-identical): " + name)


def _run_alpha_case(name: String, M: Int, in_f: Int, rank: Int, out_f: Int, s: Float32, ctx: DeviceContext) raises:
    var seed: UInt64 = 0xA1FA ^ UInt64(M * 131 + out_f)
    var xh = _rand(M * in_f, seed, 2.0)
    var ah = _rand(rank * in_f, seed, 0.4)
    var bh = _rand(out_f * rank, seed, 0.4)
    var x = Tensor.from_host(xh, [M, in_f], STDtype.F32, ctx)
    var a = Tensor.from_host(ah, [rank, in_f], STDtype.BF16, ctx)
    var b = Tensor.from_host(bh, [out_f, rank], STDtype.BF16, ctx)

    var nb1 = Optional[Tensor](None)
    var t = linear(x, a, nb1^, ctx)                    # [M,rank] F32
    var nb2 = Optional[Tensor](None)
    var dy = linear(t, b, nb2^, ctx)                   # [M,out] F32 exact C
    var legacy = mul_scalar(dy, s, ctx)                # legacy chain

    var fused = linear_rows(t, b, 0, out_f, ctx, alpha=s)
    _assert_bit_identical(name, fused, legacy, ctx)


def _run_band_case(name: String, R: Int, C: Int, off: Int, W: Int, ctx: DeviceContext) raises:
    var seed: UInt64 = 0xBAD5EED ^ UInt64(R * 17 + C + off)
    var dh = _rand(R * W, seed, 3.0)
    var sh = _rand(R * C, seed, 3.0)
    var src = Tensor.from_host(sh, [R, C], STDtype.F32, ctx)

    var dst_ref = Tensor.from_host(dh.copy(), [R, W], STDtype.F32, ctx)
    add_in_place_f32(dst_ref, slice(src, 1, off, W, ctx), ctx)   # legacy

    var dst_fused = Tensor.from_host(dh, [R, W], STDtype.F32, ctx)
    add_band_in_place_f32(dst_fused, src, off, ctx)              # fused
    _assert_bit_identical(name, dst_fused, dst_ref, ctx)


def _run_mulcast_case(name: String, R: Int, C: Int, s: Float32, ctx: DeviceContext) raises:
    # mul_scalar(F32) -> cast(BF16) vs the fused single-pass mul_scalar_bf16out
    # (Klein LoRA-bwd d_dy fusion, pass 2): same fl32 multiply, same RNE bf16
    # rounding -> BIT-IDENTICAL (compare after exact widening to F32).
    var seed: UInt64 = 0x5CA1E ^ UInt64(R * 29 + C)
    var dh = _rand(R * C, seed, 3.0)
    var d = Tensor.from_host(dh, [R, C], STDtype.F32, ctx)
    var legacy = cast_tensor(mul_scalar(d, s, ctx), STDtype.BF16, ctx)
    var fused = mul_scalar_bf16out(d, s, ctx)
    _assert_bit_identical(
        name,
        cast_tensor(fused, STDtype.F32, ctx),
        cast_tensor(legacy, STDtype.F32, ctx),
        ctx,
    )


def main() raises:
    var ctx = DeviceContext()
    # LoRA alpha-fold: Klein r16 shapes (scaled seq), incl. the w1 fused out
    # (3D+2F-like) and the small out-proj.
    _run_alpha_case("alpha r16 wide ", 192, 512, 16, 1536, Float32(1.0), ctx)
    _run_alpha_case("alpha r16 out  ", 192, 640, 16, 512, Float32(2.0), ctx)
    _run_alpha_case("alpha r16 frac ", 96, 384, 16, 768, Float32(0.8125), ctx)
    # Banded add: the 4 w1 bands (q/k/v at D, gate_up at 2F) + odd tail.
    _run_band_case("band q   ", 192, 1536, 0, 384, ctx)
    _run_band_case("band k   ", 192, 1536, 384, 384, ctx)
    _run_band_case("band gu  ", 192, 1536, 768, 768, ctx)
    _run_band_case("band odd ", 97, 1000, 123, 456, ctx)
    # Fused scalar-mul + bf16 store (LoRA-bwd d_dy chain).
    _run_mulcast_case("mulcast wide", 192, 1536, Float32(1.0), ctx)
    _run_mulcast_case("mulcast frac", 97, 1000, Float32(0.8125), ctx)
    print("ALL lora alpha-fold + banded-add parity cases PASSED (bit-identical)")
