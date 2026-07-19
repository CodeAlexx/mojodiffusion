# autograd_v2/capture.mojo - CUDA-graph capture/replay FFI + lifecycle state
# (contract C9, AUTOGRAD_V2_MOJO_DESIGN.md; Phase P5).
#
# The driver declarations are copied VERBATIM from the proven feasibility
# smoke tests/capture_smoke.mojo (independently re-run 2026-06-11 on this box:
# 5-node graph captured through MAX DeviceContext via cuStreamBeginCapture_v2
# on CUDA(ctx.stream()) - the turbo_loader.mojo:102 stream-handle idiom -
# correct replay twice through fixed pointers). Lifecycle mirrors flame
# BackwardGraphCache (flame-core/src/cuda_graph.rs:192-276): step 0 warmup,
# step 1 capture, step >=2 replay.
#
# FAIL-LOUD (Tenet 4 / P5 instruction): any nonzero CUresult RAISES with the
# decoded name - no silent fallback to the uncaptured path.
#
# Mojo 1.0.0b1, Linux x86-64, NVIDIA, MAX 26.3.

from std.ffi import external_call
from std.gpu.host import DeviceContext
from std.gpu.host._nvidia_cuda import CUDA, CUstream
from std.memory import alloc
from serenitymojo.io.ffi import BytePtr

comptime CU_STREAM_CAPTURE_MODE_GLOBAL: Int32 = 0


def _ptr[pointee: AnyType](p: UnsafePointer[pointee, MutAnyOrigin]) -> BytePtr:
    return BytePtr(unsafe_from_address=Int(p))


def _null() -> BytePtr:
    return BytePtr(unsafe_from_address=Int(0))


def cu_errname(rc: Int) -> String:
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


# ─── driver FFI (capture_smoke.mojo declarations, verbatim) ──────────────────

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
    """cuGraphGetNodes(graph, NULL, &count) -> count, or -1000-rc on error."""
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
            "  cuGraphInstantiateWithFlags rc=", cu_errname(rc),
            "- falling back to cuGraphInstantiate_v2"
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


# ─── raised wrappers over the DEFAULT MAX compute stream ─────────────────────
# Stream handle: CUDA(ctx.stream()) - the stream MAX enqueues compute on
# (capture_smoke.mojo:247-258 / turbo_loader.mojo:102).

@fieldwise_init
struct CudaGraphHandle(Copyable, Movable):
    """One instantiated CUDA graph: the recorded graph object, its executable
    instance, and the node count (sanity probe, printed once at capture)."""
    var graph: UInt64
    var exec_h: UInt64
    var nodes: Int


def cuda_capture_begin(ctx: DeviceContext) raises:
    """Begin GLOBAL-mode stream capture on the default MAX compute stream.
    Nonzero CUresult RAISES (fail loud - no silent fallback)."""
    var raw = CUDA(ctx.stream())
    var rc = _cu_begin_capture(raw)
    if rc != 0:
        raise Error(
            String("[CAPTURE] cuStreamBeginCapture_v2 failed rc=")
            + cu_errname(rc)
        )


def cuda_capture_end_instantiate(ctx: DeviceContext) raises -> CudaGraphHandle:
    """End capture, count nodes, instantiate. Any failure RAISES with the
    call name + decoded rc."""
    var raw = CUDA(ctx.stream())
    var graph: UInt64 = 0
    var rc_end = _cu_end_capture(raw, graph)
    if rc_end != 0 or graph == 0:
        raise Error(
            String("[CAPTURE] cuStreamEndCapture failed rc=")
            + cu_errname(rc_end) + " graph=" + String(graph)
        )
    var nodes = _cu_graph_node_count(graph)
    if nodes < 0:
        raise Error(
            String("[CAPTURE] cuGraphGetNodes failed rc=")
            + cu_errname(-1000 - nodes)
        )
    var exec_h: UInt64 = 0
    var rc_inst = _cu_graph_instantiate(graph, exec_h)
    if rc_inst != 0 or exec_h == 0:
        _ = _cu_graph_destroy(graph)
        raise Error(
            String("[CAPTURE] graph instantiate failed rc=")
            + cu_errname(rc_inst) + " (nodes=" + String(nodes) + ")"
        )
    return CudaGraphHandle(graph, exec_h, nodes)


def cuda_graph_launch(handle: CudaGraphHandle, ctx: DeviceContext) raises:
    """Launch the instantiated graph on the default MAX compute stream
    (kernels re-read the FIXED pointers recorded at capture)."""
    var raw = CUDA(ctx.stream())
    var rc = _cu_graph_launch(handle.exec_h, raw)
    if rc != 0:
        raise Error(
            String("[CAPTURE] cuGraphLaunch failed rc=") + cu_errname(rc)
        )
