# autograd_v2/tests/capture_smoke.mojo — P5 FEASIBILITY SMOKE (AUTOGRAD_V2 C9).
#
# HYPOTHESIS: CUDA stream capture + graph replay can wrap kernels launched
# through Mojo MAX's DeviceContext (vendor cuBLAS matmul + enqueue_function
# elementwise add) on this box (RTX 3090 Ti, CUDA 12.4).
#
# Stream-handle mechanism: the default compute stream's raw CUstream is
# obtained via CUDA(ctx.stream()) — the exact converter turbo_loader.mojo:102
# uses to hand a DeviceStream to cuMemcpyHtoDAsync_v2. ctx.stream() is the
# DEFAULT stream MAX enqueues compute on (turbo_loader.mojo:8-19 design notes).
#
# Driver FFI style copied from offload/vmm_cuda.mojo (out-params via 1-elem
# alloc + BytePtr) and offload/turbo_loader.mojo (CUstream passed straight to
# external_call).
#
# Every CUresult is checked; nonzero is a VALID finding (900=CAPTURE_UNSUPPORTED,
# 901=CAPTURE_INVALIDATED, ...) and is printed, not hidden.
#
# Mojo 1.0.0b1, Linux x86-64, NVIDIA RTX 3090 Ti, MAX 26.3.

from std.ffi import external_call
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.gpu.host._nvidia_cuda import CUDA, CUstream
from std.gpu import global_idx
from std.memory import alloc
from std.time import perf_counter_ns
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from linalg.matmul.vendor.blas import matmul
from serenitymojo.io.ffi import BytePtr

comptime N = 512
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _ITERS = 200
comptime CU_STREAM_CAPTURE_MODE_GLOBAL: Int32 = 0


def _ptr[pointee: AnyType](p: UnsafePointer[pointee, MutAnyOrigin]) -> BytePtr:
    return BytePtr(unsafe_from_address=Int(p))


def _null() -> BytePtr:
    return BytePtr(unsafe_from_address=Int(0))


def _errname(rc: Int) -> String:
    if rc == 0:
        return String("CUDA_SUCCESS")
    if rc == 900:
        return String("CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED(900)")
    if rc == 901:
        return String("CUDA_ERROR_STREAM_CAPTURE_INVALIDATED(901)")
    if rc == 902:
        return String("CUDA_ERROR_STREAM_CAPTURE_MERGE(902)")
    if rc == 903:
        return String("CUDA_ERROR_STREAM_CAPTURE_UNMATCHED(903)")
    if rc == 904:
        return String("CUDA_ERROR_STREAM_CAPTURE_UNJOINED(904)")
    if rc == 905:
        return String("CUDA_ERROR_STREAM_CAPTURE_ISOLATION(905)")
    if rc == 906:
        return String("CUDA_ERROR_STREAM_CAPTURE_IMPLICIT(906)")
    return String("CUresult(") + String(rc) + ")"


# ─── driver FFI (same external_call idiom as turbo_loader/vmm_cuda) ──────────

def _cu_stream_is_capturing(stream: CUstream) -> Int:
    """Returns capture status (0=none, 1=active, 2=invalidated), or -1000-rc
    on driver error."""
    var status = alloc[Int32](1)
    status[0] = 0
    var rc = Int(external_call["cuStreamIsCapturing", Int32](
        stream, _ptr(status)
    ))
    var s = Int(status[0])
    status.free()
    if rc != 0:
        return -1000 - rc
    return s


def _cu_begin_capture(stream: CUstream) -> Int:
    return Int(external_call["cuStreamBeginCapture_v2", Int32](
        stream, CU_STREAM_CAPTURE_MODE_GLOBAL
    ))


def _cu_end_capture(stream: CUstream, mut graph: UInt64) -> Int:
    var out = alloc[UInt64](1)
    out[0] = 0
    var rc = Int(external_call["cuStreamEndCapture", Int32](stream, _ptr(out)))
    graph = out[0]
    out.free()
    return rc


def _cu_graph_node_count(graph: UInt64) -> Int:
    """cuGraphGetNodes(graph, NULL, &count) → count, or -1000-rc on error."""
    var count = alloc[Int](1)
    count[0] = 0
    var rc = Int(external_call["cuGraphGetNodes", Int32](
        graph, _null(), _ptr(count)
    ))
    var c = count[0]
    count.free()
    if rc != 0:
        return -1000 - rc
    return c


def _cu_graph_instantiate(graph: UInt64, mut exec_h: UInt64) -> Int:
    """cuGraphInstantiateWithFlags (CUDA 11.4+/12.x), fallback to
    cuGraphInstantiate_v2 if it errors."""
    var out = alloc[UInt64](1)
    out[0] = 0
    var rc = Int(external_call["cuGraphInstantiateWithFlags", Int32](
        _ptr(out), graph, UInt64(0)
    ))
    if rc != 0:
        print(
            "  cuGraphInstantiateWithFlags rc=", _errname(rc),
            "— falling back to cuGraphInstantiate_v2"
        )
        out[0] = 0
        rc = Int(external_call["cuGraphInstantiate_v2", Int32](
            _ptr(out), graph, _null(), _null(), Int(0)
        ))
    exec_h = out[0]
    out.free()
    return rc


def _cu_graph_launch(exec_h: UInt64, stream: CUstream) -> Int:
    return Int(external_call["cuGraphLaunch", Int32](exec_h, stream))


def _cu_graph_destroy(graph: UInt64) -> Int:
    return Int(external_call["cuGraphDestroy", Int32](graph))


def _cu_graph_exec_destroy(exec_h: UInt64) -> Int:
    return Int(external_call["cuGraphExecDestroy", Int32](exec_h))


# ─── the ops under capture ────────────────────────────────────────────────────
# matmul: the SAME vendor-cuBLAS entrypoint ops/linear.mojo uses
# (linalg.matmul.vendor.blas, transpose_b=True, c_row_major=True).
# add: elementwise D = C + A through ctx.enqueue_function — the same launch
# mechanism ops/tensor_algebra.add uses (its kernel allocates a fresh output
# Tensor per call, which cannot be replayed against fixed pointers, so the
# fixed-buffer equivalent kernel lives here; allocation-in-capture is probed
# separately in probe 2).

def _add_kernel(
    c: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    a: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    d: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    var total = n * n
    if i < total:
        var r = i // n
        var col = i % n
        var v = rebind[Scalar[DType.float32]](c[r, col]) + rebind[
            Scalar[DType.float32]
        ](a[r, col])
        d[r, col] = rebind[d.element_type](v)


def _run_ops(
    ctx: DeviceContext,
    a_lt: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    b_lt: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    c_lt: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    d_lt: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
) raises:
    """C = A @ Bᵀ (vendor cuBLAS), then D = C + A (enqueue_function kernel)."""
    matmul(ctx, c_lt, a_lt, b_lt, transpose_b=True, c_row_major=True)
    var total = N * N
    var grid = (total + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_add_kernel, _add_kernel](
        c_lt, a_lt, d_lt, N, grid_dim=grid, block_dim=_BLOCK
    )


# ─── host-side fill / verify ─────────────────────────────────────────────────

def _fill_host(host: HostBuffer[DType.uint8], seed: Int):
    var fp = host.unsafe_ptr().bitcast[Float32]()
    for i in range(N * N):
        var h = (i * 1103515245 + seed * 987654321 + 12345) % 2000
        fp[i] = Float32(h - 1000) / 2000.0


def _verify(
    host_a: HostBuffer[DType.uint8],
    host_b: HostBuffer[DType.uint8],
    host_d: HostBuffer[DType.uint8],
    label: String,
) raises -> Bool:
    """Check sampled entries of D against host-F64 A@Bᵀ + A."""
    var ap = host_a.unsafe_ptr().bitcast[Float32]()
    var bp = host_b.unsafe_ptr().bitcast[Float32]()
    var dp = host_d.unsafe_ptr().bitcast[Float32]()
    var rows = [0, 1, 7, 255, 511, 511]
    var cols = [0, 2, 511, 255, 0, 511]
    var ok = True
    for s in range(len(rows)):
        var r = rows[s]
        var c = cols[s]
        var acc = Float64(0.0)
        for k in range(N):
            acc += Float64(ap[r * N + k]) * Float64(bp[c * N + k])
        acc += Float64(ap[r * N + c])
        var got = Float64(dp[r * N + c])
        var diff = got - acc
        if diff < 0:
            diff = -diff
        var bound = Float64(1e-3)
        var mag = acc if acc >= 0 else -acc
        if mag > 1.0:
            bound = mag * 1e-3
        if diff > bound:
            print(
                "  ", label, " MISMATCH D[", r, ",", c, "] got=", got,
                " expected=", acc, " diff=", diff,
            )
            ok = False
    if ok:
        print("  ", label, ": D matches host oracle on 6 sampled entries")
    return ok


def _readback(
    ctx: DeviceContext,
    dev: DeviceBuffer[DType.uint8],
    host: HostBuffer[DType.uint8],
) raises:
    ctx.enqueue_copy(dst_buf=host, src_buf=dev)
    ctx.synchronize()


def main() raises:
    print("=== P5 capture smoke: CUDA graph capture over MAX DeviceContext ===")
    var ctx = DeviceContext()

    # The crux: raw CUstream of the DEFAULT stream MAX compute runs on.
    # Mechanism: CUDA(ctx.stream()) — turbo_loader.mojo:102 idiom.
    var default_stream = ctx.stream()
    var raw = CUDA(default_stream)
    var pre_status = _cu_stream_is_capturing(raw)
    print(
        "stream handle via CUDA(ctx.stream()); cuStreamIsCapturing status=",
        pre_status, " (0=none expected; negative = -1000-rc driver error)",
    )

    # Fixed buffers (allocated BEFORE capture; graph must re-read these ptrs).
    var nbytes = N * N * 4
    var host_a = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var host_b = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var host_d = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var dev_a = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    var dev_b = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    var dev_c = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    var dev_d = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    var rl = RuntimeLayout[_DYN2].row_major(IndexList[2](N, N))
    var a_lt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dev_a.unsafe_ptr().bitcast[Float32](), rl
    )
    var b_lt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dev_b.unsafe_ptr().bitcast[Float32](), rl
    )
    var c_lt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dev_c.unsafe_ptr().bitcast[Float32](), rl
    )
    var d_lt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        dev_d.unsafe_ptr().bitcast[Float32](), rl
    )

    # ── warmup: lets cuBLAS create handles/workspace + MAX compile kernels
    # OUTSIDE capture.
    _fill_host(host_a, 1)
    _fill_host(host_b, 1001)
    ctx.enqueue_copy(dst_buf=dev_a, src_buf=host_a)
    ctx.enqueue_copy(dst_buf=dev_b, src_buf=host_b)
    _run_ops(ctx, a_lt, b_lt, c_lt, d_lt)
    ctx.synchronize()
    _readback(ctx, dev_d, host_d)
    var warm_ok = _verify(host_a, host_b, host_d, String("warmup"))
    if not warm_ok:
        print("VERDICT: CAPTURE FEASIBLE: NO (warmup math itself wrong — abort)")
        return

    # ── capture ───────────────────────────────────────────────────────────────
    var rc_begin = _cu_begin_capture(raw)
    print("cuStreamBeginCapture_v2 rc=", _errname(rc_begin))
    if rc_begin != 0:
        print("VERDICT: CAPTURE FEASIBLE: NO — begin-capture refused on the")
        print("  default-stream handle from CUDA(ctx.stream()).")
        return
    var mid_status = _cu_stream_is_capturing(raw)
    print("  capture-active check: cuStreamIsCapturing status=", mid_status,
          " (1=ACTIVE expected)")

    var capture_enqueue_failed = False
    try:
        _run_ops(ctx, a_lt, b_lt, c_lt, d_lt)
    except e:
        capture_enqueue_failed = True
        print("  enqueue DURING capture raised:", e)

    var graph: UInt64 = 0
    var rc_end = _cu_end_capture(raw, graph)
    print("cuStreamEndCapture rc=", _errname(rc_end), " graph=", graph)
    if rc_end != 0 or capture_enqueue_failed or graph == 0:
        print("VERDICT: CAPTURE FEASIBLE: NO — capture began but did not yield")
        print("  a graph (MAX enqueue path is incompatible; see rc above —")
        print("  901 means something in the MAX runtime invalidated capture,")
        print("  e.g. an internal sync or non-capturable allocation).")
        return

    var node_count = _cu_graph_node_count(graph)
    print("cuGraphGetNodes node count=", node_count,
          " (negative = -1000-rc error)")
    if node_count == 0:
        print("WARNING: 0 nodes — MAX likely routed kernels to a different")
        print("  stream than CUDA(ctx.stream()); replay check below will")
        print("  show stale C/D if so.")

    var exec_h: UInt64 = 0
    var rc_inst = _cu_graph_instantiate(graph, exec_h)
    print("graph instantiate rc=", _errname(rc_inst), " exec=", exec_h)
    if rc_inst != 0 or exec_h == 0:
        print("VERDICT: CAPTURE FEASIBLE: NO — captured a graph with",
              node_count, "nodes but instantiation failed.")
        _ = _cu_graph_destroy(graph)
        return

    # ── replay twice with NEW inputs (graph must re-read same pointers) ──────
    var replays_ok = True
    for trial in range(2):
        var seed = 2 + trial
        _fill_host(host_a, seed)
        _fill_host(host_b, 1000 + seed)
        ctx.enqueue_copy(dst_buf=dev_a, src_buf=host_a)   # outside the graph
        ctx.enqueue_copy(dst_buf=dev_b, src_buf=host_b)
        var rc_launch = _cu_graph_launch(exec_h, raw)
        if rc_launch != 0:
            print("cuGraphLaunch rc=", _errname(rc_launch))
            replays_ok = False
            break
        ctx.synchronize()
        _readback(ctx, dev_d, host_d)
        var ok = _verify(
            host_a, host_b, host_d, String("replay seed=") + String(seed)
        )
        if not ok:
            replays_ok = False
    if not replays_ok:
        print("VERDICT: CAPTURE FEASIBLE: PARTIAL/NO — graph captured (",
              node_count, "nodes ) but replay did not produce fresh-input")
        print("  results (stale C/D ⇒ kernels were NOT captured on the handle")
        print("  from CUDA(ctx.stream()), or launch failed).")
        _ = _cu_graph_exec_destroy(exec_h)
        _ = _cu_graph_destroy(graph)
        return

    # ── timing: 200 × normal enqueue vs 200 × cuGraphLaunch ─────────────────
    ctx.synchronize()
    var t0 = perf_counter_ns()
    for _ in range(_ITERS):
        _run_ops(ctx, a_lt, b_lt, c_lt, d_lt)
    var t0e = perf_counter_ns()
    ctx.synchronize()
    var t1 = perf_counter_ns()
    var normal_us = Float64(t1 - t0) / Float64(_ITERS) / 1000.0
    var normal_enq_us = Float64(t0e - t0) / Float64(_ITERS) / 1000.0

    var launch_rc_bad = 0
    var t2 = perf_counter_ns()
    for _ in range(_ITERS):
        var rc = _cu_graph_launch(exec_h, raw)
        if rc != 0:
            launch_rc_bad = rc
            break
    var t2e = perf_counter_ns()
    ctx.synchronize()
    var t3 = perf_counter_ns()
    var graph_us = Float64(t3 - t2) / Float64(_ITERS) / 1000.0
    var graph_enq_us = Float64(t2e - t2) / Float64(_ITERS) / 1000.0
    if launch_rc_bad != 0:
        print("timing loop: cuGraphLaunch failed rc=", _errname(launch_rc_bad))
    print("timing (", _ITERS, "iters, matmul512 + add; wall = enqueue+GPU,")
    print("        enqueue-only = CPU submit cost before the final sync):")
    print("  normal MAX enqueue:  wall ", normal_us, " us/iter, enqueue-only ",
          normal_enq_us, " us/iter")
    print("  cuGraphLaunch:       wall ", graph_us, " us/iter, enqueue-only ",
          graph_enq_us, " us/iter")

    # ── probe 2: does a MAX allocation DURING capture invalidate it? ─────────
    # ops/tensor_algebra.add allocates its output via enqueue_create_buffer
    # every call (caching allocator → cuMemAllocFromPoolAsync per
    # vmm_cuda.mojo:138 notes). NB: a cache hit may serve the buffer without
    # any driver call, so an odd size is used to push toward a real pool alloc.
    var rc_b2 = _cu_begin_capture(raw)
    print("probe2 (alloc during capture): begin rc=", _errname(rc_b2))
    if rc_b2 == 0:
        var alloc_raised = False
        try:
            var tmp = ctx.enqueue_create_buffer[DType.uint8](3_000_064)
            _run_ops(ctx, a_lt, b_lt, c_lt, d_lt)
            _ = tmp^
        except e:
            alloc_raised = True
            print("  alloc/enqueue during capture raised:", e)
        var graph2: UInt64 = 0
        var rc_e2 = _cu_end_capture(raw, graph2)
        var nodes2 = _cu_graph_node_count(graph2) if graph2 != 0 else -1
        print("  probe2 end rc=", _errname(rc_e2), " nodes=", nodes2,
              " alloc_raised=", alloc_raised)
        if graph2 != 0:
            _ = _cu_graph_destroy(graph2)

    _ = _cu_graph_exec_destroy(exec_h)
    _ = _cu_graph_destroy(graph)

    print("VERDICT: CAPTURE FEASIBLE: YES —", node_count, "nodes captured on")
    print("  CUDA(ctx.stream()); replay re-read fixed pointers with fresh")
    print("  inputs twice; timings above.")
