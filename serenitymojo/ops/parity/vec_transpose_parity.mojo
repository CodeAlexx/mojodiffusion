# vec_transpose_parity.mojo — gate vec_transpose AGAINST the general scalar
# transpose (ops/tensor_algebra.mojo) on [8,3840] and [3840,8]. Reference =
# scalar GPU output. Microbench + bitrot (pass BITROT).
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . serenitymojo/ops/parity/vec_transpose_parity.mojo

from sys import argv
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.tensor_algebra import transpose
from serenitymojo.ops.vec_transpose import vec_transpose


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _run_case(R: Int, C: Int, bitrot: Bool, ctx: DeviceContext) raises -> Bool:
    print("--- transpose [", R, ",", C, "] -> [", C, ",", R, "] ---")
    var x = Tensor.from_host(_fill(R * C, 11, 2.0), [R, C], STDtype.F32, ctx)
    var refh = transpose(x, 0, 1, ctx).to_host(ctx)
    var got = vec_transpose(x, ctx)
    if bitrot:
        for i in range(len(refh)):
            refh[i] = refh[i] + 2.0
    var h = ParityHarness(0.999)
    var r = h.compare(got, refh, ctx)
    print("    cos:", r)

    var reps = 300
    _ = transpose(x, 0, 1, ctx).to_host(ctx)
    var t0 = perf_counter_ns()
    for _ in range(reps):
        var yy = transpose(x, 0, 1, ctx)
        _ = yy.numel()
    ctx.synchronize()
    var t1 = perf_counter_ns()
    _ = vec_transpose(x, ctx).to_host(ctx)
    var t2 = perf_counter_ns()
    for _ in range(reps):
        var yy = vec_transpose(x, ctx)
        _ = yy.numel()
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var scal_ms = Float64(t1 - t0) / 1.0e6 / Float64(reps)
    var vec_ms = Float64(t3 - t2) / 1.0e6 / Float64(reps)
    print("    [microbench] scalar=", scal_ms, "ms  vec=", vec_ms,
          "ms  speedup=", scal_ms / vec_ms, "x")
    return r.passed


def main() raises:
    var bitrot = False
    var args = argv()
    for i in range(len(args)):
        if args[i] == String("BITROT"):
            bitrot = True
    var ctx = DeviceContext()
    print("=== vec_transpose parity vs general transpose ===")
    var p1 = _run_case(8, 3840, bitrot, ctx)
    var p2 = _run_case(3840, 8, bitrot, ctx)
    # a square-ish big case too (shows the tiled win on a balanced shape)
    var p3 = _run_case(2048, 2048, bitrot, ctx)
    if p1 and p2 and p3:
        print("PASS: vec_transpose matches general transpose cos>=0.999 (all cases)")
    else:
        raise Error("vec_transpose_parity gate FAILED")
