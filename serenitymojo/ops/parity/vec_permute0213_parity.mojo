# vec_permute0213_parity.mojo — gate vec_permute0213 AGAINST the general scalar
# permute(x,[0,2,1,3]) (ops/tensor_algebra.mojo). Reference = scalar GPU output.
# Microbench: scalar general permute ms vs vec ms. Bitrot: pass BITROT.
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#      pixi run mojo run -I . serenitymojo/ops/parity/vec_permute0213_parity.mojo

from sys import argv
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.ops.tensor_algebra import permute
from serenitymojo.ops.vec_permute0213 import vec_permute0213


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _perm0213() -> List[Int]:
    var p = List[Int]()
    p.append(0); p.append(2); p.append(1); p.append(3)
    return p^


def main() raises:
    var bitrot = False
    var args = argv()
    for i in range(len(args)):
        if args[i] == String("BITROT"):
            bitrot = True

    var ctx = DeviceContext()
    var B = 2
    var S = 1024     # tokens
    var H = 24       # heads
    var Dh = 128     # head dim (multiple of 4)
    print("=== vec_permute0213 parity vs general permute (B=", B, " S=", S,
          " H=", H, " Dh=", Dh, ") ===")

    var x_h = _fill(B * S * H * Dh, 11, 2.0)
    var x = Tensor.from_host(x_h.copy(), [B, S, H, Dh], STDtype.F32, ctx)

    var refh = permute(x, _perm0213(), ctx).to_host(ctx)
    var got = vec_permute0213(x, ctx)

    if bitrot:
        for i in range(len(refh)):
            refh[i] = refh[i] + 3.0

    var h = ParityHarness(0.999)
    var r = h.compare(got, refh, ctx)
    print("    permute:", r)

    var reps = 200
    _ = permute(x, _perm0213(), ctx).to_host(ctx)
    var t0 = perf_counter_ns()
    for _ in range(reps):
        var yy = permute(x, _perm0213(), ctx)
        _ = yy.numel()
    ctx.synchronize()
    var t1 = perf_counter_ns()
    _ = vec_permute0213(x, ctx).to_host(ctx)
    var t2 = perf_counter_ns()
    for _ in range(reps):
        var yy = vec_permute0213(x, ctx)
        _ = yy.numel()
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var scal_ms = Float64(t1 - t0) / 1.0e6 / Float64(reps)
    var vec_ms = Float64(t3 - t2) / 1.0e6 / Float64(reps)
    print("    [microbench] scalar=", scal_ms, "ms  vec=", vec_ms,
          "ms  speedup=", scal_ms / vec_ms, "x")

    if r.passed:
        print("PASS: vec_permute0213 matches general permute cos>=0.999")
    else:
        raise Error("vec_permute0213_parity gate FAILED")
