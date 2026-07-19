# vec_swiglu_parity.mojo — gate vec_swiglu AGAINST the scalar swiglu
# (ops/activations.mojo). Reference = scalar GPU output. Microbench + bitrot.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . serenitymojo/ops/parity/vec_swiglu_parity.mojo

from sys import argv
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.vec_swiglu import vec_swiglu


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def main() raises:
    var bitrot = False
    var args = argv()
    for i in range(len(args)):
        if args[i] == String("BITROT"):
            bitrot = True
    var ctx = DeviceContext()
    var rows = 4096        # tokens
    var d = 4096           # MLP hidden (multiple of 4)
    var n = rows * d
    print("=== vec_swiglu parity vs scalar (n=", n, ") ===")

    var g = Tensor.from_host(_fill(n, 11, 4.0), [rows, d], STDtype.F32, ctx)
    var u = Tensor.from_host(_fill(n, 22, 4.0), [rows, d], STDtype.F32, ctx)

    var refh = swiglu(g, u, ctx).to_host(ctx)
    var got = vec_swiglu(g, u, ctx)
    if bitrot:
        for i in range(len(refh)):
            refh[i] = refh[i] + 1.0

    var h = ParityHarness(0.999)
    var r = h.compare(got, refh, ctx)
    print("    swiglu:", r)

    var reps = 200
    _ = swiglu(g, u, ctx).to_host(ctx)
    var t0 = perf_counter_ns()
    for _ in range(reps):
        var yy = swiglu(g, u, ctx)
        _ = yy.numel()
    ctx.synchronize()
    var t1 = perf_counter_ns()
    _ = vec_swiglu(g, u, ctx).to_host(ctx)
    var t2 = perf_counter_ns()
    for _ in range(reps):
        var yy = vec_swiglu(g, u, ctx)
        _ = yy.numel()
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var scal_ms = Float64(t1 - t0) / 1.0e6 / Float64(reps)
    var vec_ms = Float64(t3 - t2) / 1.0e6 / Float64(reps)
    print("    [microbench] scalar=", scal_ms, "ms  vec=", vec_ms,
          "ms  speedup=", scal_ms / vec_ms, "x")

    if r.passed:
        print("PASS: vec_swiglu matches scalar swiglu cos>=0.999")
    else:
        raise Error("vec_swiglu_parity gate FAILED")
