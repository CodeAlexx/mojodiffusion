# Smoke gate for serenitymojo/scratch_ring.mojo.
#
# Run:
#   pixi run mojo run -I . serenitymojo/scratch_ring_smoke.mojo

from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra_scratch import (
    concat2_scratch,
    concat3_scratch,
    slice_scratch,
)


def _lf(*values: Float64) -> List[Float32]:
    var out = List[Float32]()
    for i in range(len(values)):
        out.append(Float32(values[i]))
    return out^


def main() raises:
    var ctx = DeviceContext()
    var h = ParityHarness()
    var all_pass = True
    comptime F32 = STDtype.F32

    var vals = _lf(1.0, -2.0, 3.5, 4.25, -5.0, 6.75)
    var src = Tensor.from_host(vals, [2, 3], F32, ctx)

    var ring = ScratchRingAllocator(ctx, 64, 2)
    var first = ring.clone_tensor(src, ctx)
    var r = h.compare(first, vals, ctx)
    print("scratch clone       ", r)
    all_pass = all_pass and r.passed

    var used_after_first = ring.used_bytes()
    if used_after_first != 32:
        raise Error("scratch_ring_smoke: F32 [2,3] should align to 32 bytes")

    var mark = ring.mark()
    var second = ring.clone_tensor(src, ctx)
    r = h.compare(second, vals, ctx)
    print("scratch second      ", r)
    all_pass = all_pass and r.passed

    ctx.synchronize()
    ring.rewind(mark)
    if ring.used_bytes() != used_after_first:
        raise Error("scratch_ring_smoke: rewind did not restore offset")

    var third = ring.clone_tensor(src, ctx)
    r = h.compare(third, vals, ctx)
    print("scratch rewind      ", r)
    all_pass = all_pass and r.passed

    ring.reset()
    if ring.used_bytes() != 0:
        raise Error("scratch_ring_smoke: reset did not clear offset")

    var fourth = ring.clone_tensor(src, ctx)
    r = h.compare(fourth, vals, ctx)
    print("scratch reset       ", r)
    all_pass = all_pass and r.passed

    var a = Tensor.from_host(vals, [2, 3], F32, ctx)
    var b_vals = _lf(10.0, 11.0, 12.0, 13.0)
    var b = Tensor.from_host(b_vals, [2, 2], F32, ctx)
    var c_vals = _lf(20.0, 21.0, 22.0, 23.0)
    var c = Tensor.from_host(c_vals, [2, 2], F32, ctx)

    ring.reset()
    var cat2_ref = _lf(1.0, -2.0, 3.5, 10.0, 11.0, 4.25, -5.0, 6.75, 12.0, 13.0)
    var cat2 = concat2_scratch(1, ctx, ring, a, b)
    r = h.compare(cat2, cat2_ref, ctx)
    print("scratch concat2     ", r)
    all_pass = all_pass and r.passed

    var sliced_ref = _lf(-2.0, 3.5, 10.0, -5.0, 6.75, 12.0)
    var sliced = slice_scratch(cat2, 1, 1, 3, ctx, ring)
    r = h.compare(sliced, sliced_ref, ctx)
    print("scratch slice       ", r)
    all_pass = all_pass and r.passed

    ring.reset()
    var cat3_ref = _lf(
        1.0, -2.0, 3.5, 10.0, 11.0, 20.0, 21.0,
        4.25, -5.0, 6.75, 12.0, 13.0, 22.0, 23.0,
    )
    var cat3 = concat3_scratch(1, ctx, ring, a, b, c)
    r = h.compare(cat3, cat3_ref, ctx)
    print("scratch concat3     ", r)
    all_pass = all_pass and r.passed

    if all_pass:
        print("ALL SCRATCH RING GATES PASSED")
    else:
        raise Error("scratch_ring_smoke parity gate failed")
