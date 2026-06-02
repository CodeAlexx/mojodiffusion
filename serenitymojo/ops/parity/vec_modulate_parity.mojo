# vec_modulate_parity.mojo — gate vec_modulate AGAINST the scalar modulate
# (ops/elementwise.mojo). Reference = scalar GPU output. Microbench + bitrot.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . serenitymojo/ops/parity/vec_modulate_parity.mojo

from sys import argv
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.elementwise import modulate
from serenitymojo.ops.vec_modulate import vec_modulate


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
    var rows = 4608        # tokens
    var d = 3072           # channels (multiple of 4)
    var n = rows * d
    print("=== vec_modulate parity vs scalar (rows=", rows, " D=", d, ") ===")

    var x = Tensor.from_host(_fill(n, 11, 2.0), [rows, d], STDtype.F32, ctx)
    var s = Tensor.from_host(_fill(d, 22, 1.0), [d], STDtype.F32, ctx)
    var sh = Tensor.from_host(_fill(d, 33, 1.0), [d], STDtype.F32, ctx)

    var refh = modulate(x, s, sh, ctx).to_host(ctx)
    var got = vec_modulate(x, s, sh, ctx)
    if bitrot:
        for i in range(len(refh)):
            refh[i] = refh[i] + 1.0

    var h = ParityHarness(0.999)
    var r = h.compare(got, refh, ctx)
    print("    modulate:", r)

    var reps = 200
    _ = modulate(x, s, sh, ctx).to_host(ctx)
    var t0 = perf_counter_ns()
    for _ in range(reps):
        var yy = modulate(x, s, sh, ctx)
        _ = yy.numel()
    ctx.synchronize()
    var t1 = perf_counter_ns()
    _ = vec_modulate(x, s, sh, ctx).to_host(ctx)
    var t2 = perf_counter_ns()
    for _ in range(reps):
        var yy = vec_modulate(x, s, sh, ctx)
        _ = yy.numel()
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var scal_ms = Float64(t1 - t0) / 1.0e6 / Float64(reps)
    var vec_ms = Float64(t3 - t2) / 1.0e6 / Float64(reps)
    print("    [microbench] scalar=", scal_ms, "ms  vec=", vec_ms,
          "ms  speedup=", scal_ms / vec_ms, "x")

    if r.passed:
        print("PASS: vec_modulate matches scalar modulate cos>=0.999")
    else:
        raise Error("vec_modulate_parity gate FAILED")
