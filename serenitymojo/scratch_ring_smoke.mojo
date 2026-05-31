# Smoke gate for serenitymojo/scratch_ring.mojo.
#
# Run:
#   pixi run mojo run -I . serenitymojo/scratch_ring_smoke.mojo

from std.gpu.host import DeviceContext
from std.collections import Optional

from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.scratch_ring import ScratchRingAllocator
from serenitymojo.tensor import Tensor
from serenitymojo.ops.tensor_algebra_scratch import (
    concat2_scratch,
    concat3_scratch,
    slice_scratch,
)
from serenitymojo.ops.linalg_backward import (
    linear_backward_dx,
    linear_backward_dx_scratch,
)
from serenitymojo.ops.linear import linear, linear_scratch
from serenitymojo.ops.attention_backward import sdpa_backward, sdpa_backward_scratch


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

    ring.reset()
    var front_mark = ring.mark()
    var front = ring.clone_tensor(src, ctx)
    r = h.compare(front, vals, ctx)
    print("scratch front       ", r)
    all_pass = all_pass and r.passed
    var reverse = ring.clone_tensor_reverse(src, ctx)
    r = h.compare(reverse, vals, ctx)
    print("scratch reverse     ", r)
    all_pass = all_pass and r.passed
    if ring.used_bytes() != 64:
        raise Error("scratch_ring_smoke: forward+reverse should use one slab")

    ctx.synchronize()
    ring.rewind(front_mark)
    if ring.used_bytes() != 0:
        raise Error("scratch_ring_smoke: rewind did not restore reverse cursor")

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

    ring.reset()
    var ra = Tensor.from_host(_lf(1.0, 2.0, 3.0, 4.0), [1, 2, 1, 2], F32, ctx)
    var rb = Tensor.from_host(_lf(10.0, 11.0), [1, 1, 1, 2], F32, ctx)
    var rcat = concat2_scratch(1, ctx, ring, ra, rb, True)
    r = h.compare(rcat, _lf(1.0, 2.0, 3.0, 4.0, 10.0, 11.0), ctx)
    print("scratch rank4 cat   ", r)
    all_pass = all_pass and r.passed

    var rslice = slice_scratch(rcat, 1, 1, 2, ctx, ring, True)
    r = h.compare(rslice, _lf(3.0, 4.0, 10.0, 11.0), ctx)
    print("scratch rank4 slice ", r)
    all_pass = all_pass and r.passed

    ring.reset()
    var gy = Tensor.from_host(_lf(1.0, -2.0, 3.0, 0.5, 4.0, -1.0), [2, 3], F32, ctx)
    var wt = Tensor.from_host(
        _lf(0.25, 1.0, -0.5, 2.0, 1.5, -1.0, 0.75, 0.5, -2.0, 0.125, 1.25, -0.25),
        [3, 4], F32, ctx,
    )
    var dx_ref_t = linear_backward_dx(gy, wt, 2, 4, 3, ctx)
    var dx_ref = dx_ref_t.to_host(ctx)
    var dx_scratch = linear_backward_dx_scratch(gy, wt, 2, 4, 3, ctx, ring)
    r = h.compare(dx_scratch, dx_ref, ctx)
    print("scratch linear dx   ", r)
    all_pass = all_pass and r.passed

    ring.reset()
    var lin_x = Tensor.from_host(_lf(1.0, -2.0, 0.5, 3.0, 2.0, 1.5, -1.0, 0.25), [2, 4], F32, ctx)
    var no_bias = Optional[Tensor](None)
    var lin_ref_t = linear(lin_x, wt, no_bias^, ctx)
    var lin_ref = lin_ref_t.to_host(ctx)
    var no_bias_scratch = Optional[Tensor](None)
    var lin_scratch = linear_scratch(lin_x, wt, no_bias_scratch^, ctx, ring)
    r = h.compare(lin_scratch, lin_ref, ctx)
    print("scratch linear fwd  ", r)
    all_pass = all_pass and r.passed

    var att_ring = ScratchRingAllocator(ctx, 4096, 2)
    var qv = List[Float32]()
    var kv = List[Float32]()
    var vv = List[Float32]()
    var gov = List[Float32]()
    for i in range(24):
        qv.append((Float32((i * 7) % 17) - 8.0) * 0.03)
        kv.append((Float32((i * 5) % 19) - 9.0) * 0.025)
        vv.append((Float32((i * 3) % 13) - 6.0) * 0.04)
        gov.append((Float32((i * 11) % 23) - 11.0) * 0.02)
    var q = Tensor.from_host(qv, [1, 3, 2, 4], F32, ctx)
    var k = Tensor.from_host(kv, [1, 3, 2, 4], F32, ctx)
    var v = Tensor.from_host(vv, [1, 3, 2, 4], F32, ctx)
    var go = Tensor.from_host(gov, [1, 3, 2, 4], F32, ctx)
    var sdpa_ref = sdpa_backward[1, 3, 2, 4](q, k, v, go, Float32(0.5), ctx)
    var sdpa_scratch = sdpa_backward_scratch[1, 3, 2, 4](
        q, k, v, go, Float32(0.5), ctx, att_ring,
    )
    var dq_ref = sdpa_ref.d_q.to_host(ctx)
    r = h.compare(sdpa_scratch.d_q, dq_ref, ctx)
    print("scratch sdpa d_q    ", r)
    all_pass = all_pass and r.passed
    var dk_ref = sdpa_ref.d_k.to_host(ctx)
    r = h.compare(sdpa_scratch.d_k, dk_ref, ctx)
    print("scratch sdpa d_k    ", r)
    all_pass = all_pass and r.passed
    var dv_ref = sdpa_ref.d_v.to_host(ctx)
    r = h.compare(sdpa_scratch.d_v, dv_ref, ctx)
    print("scratch sdpa d_v    ", r)
    all_pass = all_pass and r.passed

    if all_pass:
        print("ALL SCRATCH RING GATES PASSED")
    else:
        raise Error("scratch_ring_smoke parity gate failed")
