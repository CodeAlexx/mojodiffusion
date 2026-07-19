# serenitymojo/models/klein/parity/klein_int8_f32_fusion_parity.mojo
#
# PARITY GATE for the F32-boundary int8 GEMM fusion (ops/int8_quant.mojo +
# ops/int8_linear.mojo, 2026-07-11) used at every Klein int8 site:
#
#   legacy: cast_tensor(x,BF16) → rowscale/encode(bf16) → GEMM → dequant(bf16)
#           → cast_tensor(o,F32) [→ slice per column band]
#   fused:  rowscale/encode read F32 with the bf16 rounding IN-REGISTER; the
#           dequant writes F32 (bf16-rounded then exactly widened) [into the
#           column bands directly].
#
# Every collapsed step is a pure representation change (bf16→f32 widening is
# exact; the band write uses slice's element mapping), so the gate demands
# BIT-IDENTITY: max_abs == 0.0 on fwd, fwd-bands, bwd_nn, and bwd_nn-bands.
#
# Build+run:
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#     serenitymojo/models/klein/parity/klein_int8_f32_fusion_parity.mojo

from std.gpu.host import DeviceContext
from std.collections import List
from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.tensor_algebra import slice, add_band_in_place_f32
from serenitymojo.ops.int8_quant import int8_tensorwise_scale, int8_encode_tensorwise
from serenitymojo.ops.int8_linear import (
    int8_linear_fwd, int8_linear_bwd_nn,
    int8_linear_fwd_f32, int8_linear_bwd_nn_f32,
    int8_linear_fwd_f32_bands, int8_linear_bwd_nn_f32_bands,
    int8_linear_fwd_f32_bands_addfull,
    int8_band_tensor,
)


fn _lcg(mut state: UInt64) -> Float32:
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
        raise Error("int8 f32 fusion parity FAILED (not bit-identical): " + name)


def _run_case(name: String, M: Int, K: Int, w0: Int, w1: Int, w2: Int, w3: Int, ctx: DeviceContext) raises:
    var N = w0 + w1 + w2 + w3
    var seed: UInt64 = 0xC0FFEE ^ UInt64(M * 31 + K * 7 + N)
    var xh = _rand(M * K, seed, 2.5)
    var wh = _rand(N * K, seed, 0.7)

    var x = Tensor.from_host(xh, [M, K], STDtype.F32, ctx)       # F32 activation
    var w_bf = Tensor.from_host(wh, [N, K], STDtype.BF16, ctx)   # frozen bf16 weight
    var ws = int8_tensorwise_scale(w_bf, ctx)                    # [1] F32
    var w8 = int8_encode_tensorwise(w_bf, ws, ctx)               # [N,K] I8

    # ── fwd: legacy cast chain vs fused F32 boundary ─────────────────────────
    var xb = cast_tensor(x, STDtype.BF16, ctx)
    var o_bf = int8_linear_fwd(xb, w8, ws, ctx)                  # [M,N] bf16
    var o_ref = cast_tensor(o_bf, STDtype.F32, ctx)              # [M,N] f32
    var o_fused = int8_linear_fwd_f32(x, w8, ws, ctx)
    _assert_bit_identical(name + " fwd", o_fused, o_ref, ctx)

    # ── fwd bands vs slices of the legacy output ─────────────────────────────
    var bands = int8_linear_fwd_f32_bands(x, w8, ws, w0, w1, w2, w3, ctx)
    var offs: List[Int] = [0, w0, w0 + w1, w0 + w1 + w2]
    var widths: List[Int] = [w0, w1, w2, w3]
    for bi in range(len(bands)):
        if widths[bi] == 0:
            continue
        var ref_band = slice(o_ref, 1, offs[bi], widths[bi], ctx)
        _assert_bit_identical(
            name + " fwd band " + String(bi), int8_band_tensor(bands[bi]), ref_band, ctx,
        )

    # ── fwd bands + FULL-WIDTH addend (LoRA delta-add fusion, pass 2) ────────
    # reference: plain bands, then add_band_in_place_f32 per band (the exact
    # chain the fused kernel replaces).
    var adh = _rand(M * N, seed, 0.9)
    var ad = Tensor.from_host(adh, [M, N], STDtype.F32, ctx)
    var bands_ref = int8_linear_fwd_f32_bands(x, w8, ws, w0, w1, w2, w3, ctx)
    for bi in range(len(bands_ref)):
        if widths[bi] == 0:
            continue
        add_band_in_place_f32(int8_band_tensor(bands_ref[bi]), ad, offs[bi], ctx)
    var bands_add = int8_linear_fwd_f32_bands_addfull(x, w8, ws, ad, w0, w1, w2, w3, ctx)
    for bi in range(len(bands_add)):
        if widths[bi] == 0:
            continue
        _assert_bit_identical(
            name + " fwd band+add " + String(bi),
            int8_band_tensor(bands_add[bi]), int8_band_tensor(bands_ref[bi]), ctx,
        )

    # ── bwd_nn: grad [M,N] F32 → dX [M,K]; legacy cast chain vs fused ────────
    var gh = _rand(M * N, seed, 1.5)
    var gr = Tensor.from_host(gh, [M, N], STDtype.F32, ctx)
    var gb = cast_tensor(gr, STDtype.BF16, ctx)
    var dx_bf = int8_linear_bwd_nn(gb, w8, ws, ctx)              # [M,K] bf16
    var dx_ref = cast_tensor(dx_bf, STDtype.F32, ctx)
    var dx_fused = int8_linear_bwd_nn_f32(gr, w8, ws, ctx)
    _assert_bit_identical(name + " bwd", dx_fused, dx_ref, ctx)

    # ── bwd bands (2-band split, the single-block w2 d_att|d_mlp case) ───────
    var kb0 = K // 2
    var kb1 = K - kb0
    var dbands = int8_linear_bwd_nn_f32_bands(gr, w8, ws, kb0, kb1, 0, 0, ctx)
    var d0_ref = slice(dx_ref, 1, 0, kb0, ctx)
    var d1_ref = slice(dx_ref, 1, kb0, kb1, ctx)
    _assert_bit_identical(name + " bwd band 0", int8_band_tensor(dbands[0]), d0_ref, ctx)
    _assert_bit_identical(name + " bwd band 1", int8_band_tensor(dbands[1]), d1_ref, ctx)


def main() raises:
    var ctx = DeviceContext()
    # Klein-shaped (scaled-down) splits: w1 [q|k|v|gate_up] and wqkv [q|k|v].
    _run_case("w1-like  4band", 192, 256, 128, 128, 128, 512, ctx)
    _run_case("wqkv-like 3band", 160, 384, 128, 128, 128, 0, ctx)
    # Non-%256 sizes to exercise the grid-stride tails (cshim needs M,N,K %4).
    _run_case("odd      3band", 100, 204, 104, 60, 40, 0, ctx)
    print("ALL int8 F32-boundary fusion parity cases PASSED (bit-identical)")
