# on_device_global_norm_parity.mojo — gate on_device_global_norm AGAINST a host
# F64 sum-of-squares over the SAME grads (the reference the scalar
# clip_grads_by_global_norm computes). Microbench: device norm vs host-readback
# sum-of-squares. Bitrot: pass BITROT (corrupts ref → gate exits NONZERO).
#
# Run: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo run -I . serenitymojo/training/parity/on_device_global_norm_parity.mojo

from std.sys import argv
from std.math import sqrt
from std.memory import ArcPointer
from std.time import perf_counter_ns
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.training.on_device_global_norm import on_device_global_norm, TArc


def _fill(n: Int, seed: UInt64, scale: Float32) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    for _ in range(n):
        state = state * 6364136223846793005 + 1442695040888963407
        var u = Float32(Int(state >> 40)) * Float32(1.0 / 16777216.0)
        out.append((u - Float32(0.5)) * scale)
    return out^


def _storage_dtype_gate(dtype: STDtype, ctx: DeviceContext) raises:
    var sizes = List[Int]()
    sizes.append(1024)
    sizes.append(3072)
    sizes.append(777)

    var grads = List[TArc]()
    for i in range(len(sizes)):
        grads.append(
            TArc(Tensor.from_host(
                _fill(sizes[i], 2000 + UInt64(i), 0.3), [sizes[i]], dtype, ctx
            ))
        )

    var total_sq = Float64(0.0)
    for i in range(len(sizes)):
        if grads[i][].dtype() != dtype:
            raise Error("on_device_global_norm storage gate: grad dtype changed")
        var h = grads[i][].to_host(ctx)
        for j in range(len(h)):
            var x = Float64(h[j])
            total_sq += x * x
    var ref_norm = Float32(sqrt(total_sq))
    var dev_norm = on_device_global_norm(grads, ctx)
    var rel = Float64(dev_norm - ref_norm)
    if rel < 0.0:
        rel = -rel
    rel = rel / Float64(ref_norm)
    if rel < 1.0e-3:
        print("PASS: on_device_global_norm preserves ", dtype.name(), " grad storage")
    else:
        raise Error(
            String("on_device_global_norm ")
            + dtype.name()
            + " storage gate FAILED"
        )


def main() raises:
    var bitrot = False
    var args = argv()
    for i in range(len(args)):
        if args[i] == String("BITROT"):
            bitrot = True
    var ctx = DeviceContext()

    var sizes = List[Int]()
    sizes.append(4096); sizes.append(8192); sizes.append(1024)
    sizes.append(16384); sizes.append(2048); sizes.append(262144)
    sizes.append(512); sizes.append(131072)
    var nt = len(sizes)
    print("=== on_device_global_norm parity vs host sum-of-squares (N=", nt, ") ===")

    var srcs = List[List[Float32]]()
    for i in range(nt):
        srcs.append(_fill(sizes[i], 100 + UInt64(i), 0.3))

    # host reference: F64 sum-of-squares over all grads, then sqrt
    var total_sq = Float64(0.0)
    for i in range(nt):
        for j in range(sizes[i]):
            var x = Float64(srcs[i][j])
            total_sq += x * x
    var ref_norm = Float32(sqrt(total_sq))
    if bitrot:
        ref_norm = ref_norm + 10.0

    var grads = List[TArc]()
    for i in range(nt):
        grads.append(TArc(Tensor.from_host(srcs[i].copy(), [sizes[i]], STDtype.F32, ctx)))

    var dev_norm = on_device_global_norm(grads, ctx)
    var rel = Float64(dev_norm - ref_norm)
    if rel < 0.0:
        rel = -rel
    rel = rel / Float64(ref_norm)
    print("    device_norm=", dev_norm, "  ref_norm=", ref_norm,
          "  rel_err=", rel)

    # ── microbench: device norm vs host-readback sum-of-squares ──
    var reps = 50
    var t0 = perf_counter_ns()
    for _ in range(reps):
        var dn = on_device_global_norm(grads, ctx)
        _ = dn
    var t1 = perf_counter_ns()
    var t2 = perf_counter_ns()
    for _ in range(reps):
        var ts = Float64(0.0)
        for i in range(nt):
            var h = grads[i][].to_host(ctx)
            for j in range(len(h)):
                var x = Float64(h[j])
                ts += x * x
        _ = Float32(sqrt(ts))
    var t3 = perf_counter_ns()
    var dev_ms = Float64(t1 - t0) / 1.0e6 / Float64(reps)
    var host_ms = Float64(t3 - t2) / 1.0e6 / Float64(reps)
    print("    [microbench] host-readback=", host_ms, "ms  device=", dev_ms,
          "ms  speedup=", host_ms / dev_ms, "x")

    # gate: relative error tiny (F32 accumulation) — treat <1e-3 as pass.
    if rel < 1.0e-3:
        print("PASS: on_device_global_norm matches host sum-of-squares (rel<1e-3)")
    else:
        raise Error("on_device_global_norm_parity gate FAILED")

    _storage_dtype_gate(STDtype.BF16, ctx)
    _storage_dtype_gate(STDtype.F16, ctx)
