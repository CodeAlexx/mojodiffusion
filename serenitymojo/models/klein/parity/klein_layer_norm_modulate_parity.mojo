# serenitymojo/models/klein/parity/klein_layer_norm_modulate_parity.mojo
#
# PARITY GATE for the fused `layer_norm_modulate` op (ops/norm.mojo) that replaces
# the `layer_norm(x, ones, zeros, eps)` + `modulate(ln, scale, shift)` PAIR at the
# Klein forward/recompute sites. Builds a random non-degenerate [S, D] x plus
# scale/shift and asserts the fused op's ln and norm BOTH match the two-step
# reference at cos >= 0.9999 with a tiny max_abs (should be bit-identical -> 1.0).
# Covers bf16 (Klein activation dtype) and f32, and the [B, D] per-sample-vec path.
#
# Build+run:
#   cd /home/alex/mojodiffusion && pixi run mojo run -I . \
#     serenitymojo/models/klein/parity/klein_layer_norm_modulate_parity.mojo

from max.gpu.host import DeviceContext
from std.collections import List
from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.norm import layer_norm, layer_norm_modulate, lnmod_placeholder
from serenitymojo.ops.elementwise import modulate

comptime EPS = Float32(1e-06)


def _lcg(mut state: UInt64) -> Float32:
    # simple deterministic LCG in [-1, 1)
    state = state * 6364136223846793005 + 1442695040888963407
    var u = (state >> 40) & 0xFFFFFF  # 24 bits
    return (Float32(Int(u)) / Float32(1 << 23)) - 1.0


def _rand(n: Int, mut state: UInt64, scale: Float32) -> List[Float32]:
    var v = List[Float32]()
    for _ in range(n):
        v.append(_lcg(state) * scale)
    return v^


def _ones(d: Int) -> List[Float32]:
    var v = List[Float32]()
    for _ in range(d):
        v.append(1.0)
    return v^


def _zeros(d: Int) -> List[Float32]:
    var v = List[Float32]()
    for _ in range(d):
        v.append(0.0)
    return v^


def _run_case(
    name: String, S: Int, D: Int, nvec: Int, dt: STDtype, ctx: DeviceContext
) raises:
    var seed: UInt64 = 0x1234ABCD ^ UInt64(S * 131 + D * 7 + nvec)
    var xh = _rand(S * D, seed, 3.0)
    var sch = _rand(nvec * D, seed, 0.5)
    var shh = _rand(nvec * D, seed, 0.5)

    var xshape: List[Int] = [S, D]
    var vshape: List[Int]
    if nvec == 1:
        vshape = [D]
    else:
        vshape = [nvec, D]

    var x = Tensor.from_host(xh, xshape.copy(), dt, ctx)
    var scale = Tensor.from_host(sch, vshape.copy(), dt, ctx)
    var shift = Tensor.from_host(shh, vshape.copy(), dt, ctx)

    # two-step reference
    var ones = Tensor.from_host(_ones(D), [D], dt, ctx)
    var zeros = Tensor.from_host(_zeros(D), [D], dt, ctx)
    var ln_ref = layer_norm(x, ones, zeros, EPS, ctx)
    var norm_ref = modulate(ln_ref, scale, shift, ctx)

    # fused (exercise the mut-out-param pattern used at the call sites)
    var f_ln = lnmod_placeholder(ctx)
    var f_norm = layer_norm_modulate(x, scale, shift, EPS, ctx, f_ln)

    var ln_ref_h = ln_ref.to_host(ctx)
    var norm_ref_h = norm_ref.to_host(ctx)

    var h = ParityHarness(0.9999)
    var r_ln = h.compare(f_ln, ln_ref_h, ctx)
    var r_no = h.compare(f_norm, norm_ref_h, ctx)
    print(
        name,
        "ln  cos=", r_ln.cos, " max_abs=", r_ln.max_abs, " pass=", r_ln.passed,
    )
    print(
        name,
        "nrm cos=", r_no.cos, " max_abs=", r_no.max_abs, " pass=", r_no.passed,
    )
    if not r_ln.passed or not r_no.passed:
        raise Error("layer_norm_modulate parity FAILED for " + name)


def main() raises:
    var ctx = DeviceContext()
    # Klein real inner dim D=4096; a few seq lengths + the [B,D] stacked path.
    _run_case("bf16 S=128 D=4096 v1", 128, 4096, 1, STDtype.BF16, ctx)
    _run_case("bf16 S=256 D=4096 v2", 256, 4096, 2, STDtype.BF16, ctx)
    _run_case("f32  S=128 D=4096 v1", 128, 4096, 1, STDtype.F32, ctx)
    _run_case("f32  S=256 D=4096 v2", 256, 4096, 2, STDtype.F32, ctx)
    print("ALL layer_norm_modulate parity cases PASSED (cos >= 0.9999)")
