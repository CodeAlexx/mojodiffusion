# ops/vec_swiglu.mojo — VECTORIZED fused SwiGLU (silu(gate)*up), F32 fast path.
#
# NEW STANDALONE kernel. Does NOT replace ops/activations.mojo `swiglu`; it is a
# faster sibling. Parity gated against the scalar swiglu (vec_swiglu_parity.mojo).
#
# Math (matches ops/activations.mojo swiglu EXACTLY): o = silu(gate) * up, with
# silu(v) = v / (1 + exp(-v)) computed in F32. The scalar kernel does one
# element per thread; this one loads `gate` and `up` as width-4 SIMD vectors and
# computes 4 SwiGLU lanes per thread (one fewer thread launch + 16-byte loads),
# mirroring flame-core's swiglu_fused_bf16_vec2_kernel (FLAME_KERNELS.md
# bf16_ops.rs:1819 — pair loads, F32 sigmoid). flame-core uses vec2 for BF16
# (__nv_bfloat162); for F32 storage we use vec4 (16-byte coalesced).
#
# Requirement: numel % 4 == 0 (the gate/up MLP hidden width is always a multiple
# of 4 in these models). Else RAISE — caller falls back to scalar swiglu
# (AGENT-DEFAULT: raise, no scalar tail in the fast path).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import exp
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.activations import swiglu as _scalar_swiglu


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime _VW = 4


# silu over a SIMD[F32,4]: v / (1 + exp(-v)), lanewise. `exp` is overloaded for
# SIMD so this stays one vector op (no per-lane scalar loop).
@always_inline
def _silu_vec(v: SIMD[DType.float32, _VW]) -> SIMD[DType.float32, _VW]:
    return v / (SIMD[DType.float32, _VW](1.0) + exp(-v))


# One thread per vec4 chunk: 4 SwiGLU lanes/thread, coalesced 16-byte loads.
def _vec_swiglu_kernel(
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    u: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    nchunks: Int,
):
    var chunk = Int(global_idx.x)
    if chunk >= nchunks:
        return
    var base = chunk * _VW
    var gv = g.ptr.load[width=_VW](base)
    var uv = u.ptr.load[width=_VW](base)
    o.ptr.store[width=_VW](base, _silu_vec(gv) * uv)


def vec_swiglu(x_gate: Tensor, x_up: Tensor, ctx: DeviceContext) raises -> Tensor:
    """Vectorized swiglu(gate, up) = silu(gate)*up for F32.

    BF16/F16 use the dtype-preserving scalar implementation instead of
    materializing F32 fast-path storage."""
    if x_gate.dtype() != STDtype.F32 or x_up.dtype() != STDtype.F32:
        return _scalar_swiglu(x_gate, x_up, ctx)
    if x_gate.numel() != x_up.numel():
        raise Error("vec_swiglu: gate/up numel mismatch")
    var n = x_gate.numel()
    if n % _VW != 0:
        raise Error(
            String("vec_swiglu: numel must be a multiple of 4 (got ")
            + String(n) + ") — use the scalar swiglu"
        )
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](x_gate.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x_gate.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var U = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        x_up.buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), rl
    )
    var nchunks = n // _VW
    var grid = (nchunks + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_vec_swiglu_kernel, _vec_swiglu_kernel](
        G, U, O, nchunks, grid_dim=grid, block_dim=_BLOCK
    )
    return Tensor(out_buf^, x_gate.shape(), STDtype.F32)
