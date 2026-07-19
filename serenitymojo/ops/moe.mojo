# ops/moe.mojo — Mixture-of-Experts / Mixture-of-Transformers primitives.
#
# Three primitives, mirroring the flame-core CUDA kernels for EXACT semantics:
#
#   top_k_router(logits[T,E], k) -> (expert_ids[T*k], gating[T*k])
#     Token-choice top-k: per token, select the k experts with the highest
#     router logits (descending logit; ties broken by LOWER expert index — the
#     same rule as flame-core moe_routing.rs's host-side select:
#       `b.partial_cmp(a).then(a.idx.cmp(b.idx))`). Gating = softmax over ONLY
#     the k selected logits (sums to 1 per token). Computed host-side, exactly
#     as flame-core computes its top-k host-side (download -> partial sort).
#     Returns host Lists (token-major: slot s = t*k + j).
#
#   grouped_expert_ffn(tokens[T,H], gate_w/up_w[E,F,H], down_w[E,H,F],
#                      expert_ids[T*k]) -> expert_out[T*k, H]
#     Each routed slot s = (t,j) runs token t's hidden vec through expert
#     expert_ids[s]'s SwiGLU FFN:  down( silu(x @ gate^T) * (x @ up^T) ).
#     Implemented as a LOOP over experts (flame-core grouped_mm.rs notes per-
#     expert cuBLASLt is competitive): for each expert gather its slots into a
#     contiguous block, call the callable `linalg`-backed `linear` per expert,
#     apply swiglu, scatter the rows back to their slots. F32-accumulated GEMM.
#
#   gated_scatter_add(expert_out[T*k,H], gating[T*k], indices[T*k], accum[N,H])
#     accum[indices[s]] += expert_out[s] * gating[s]   (in-place, all s).
#     Mirrors fused_gated_scatter_add.rs: atomic-add scatter through an F32
#     accumulator workspace (so the k slots that collide on the same token row
#     combine correctly for top-k>1), then stores back in the caller's accum
#     dtype. Out-of-range / negative indices are skipped.
#     `indices` is a host List[Int] copied HtoD as a REAL i32 buffer (flame-core
#     passes host &[i32] the same way — its DType::I32 Tensors are f32-bytes-
#     relabeled, so it never routes indices through a Tensor either).
#
# (MoT note: SenseNova-U1's per-modality dispatch is just top_k_router applied
# per modality-stream — no extra kernel; the router here covers it.)
#
# Mojo 1.0.0b1, NVIDIA GPU. F32 accumulation, storage-preserving boundaries.
# Reuses ops.linear + ops.swiglu.

from std.math import exp
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.atomic import Atomic
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.cast import cast_tensor


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256


# ── stage 1: top-k router (host-side, mirrors flame-core's host top-k) ────────
@fieldwise_init
struct RouterPlan(Movable):
    """Output of `top_k_router`. Token-major: slot s = t*k + j is token t's
    j-th expert pick. `expert_ids` and `gating` both have length T*k."""

    var expert_ids: List[Int]
    var gating: List[Float32]
    var num_tokens: Int
    var num_experts: Int
    var top_k: Int


def top_k_router(
    logits: Tensor, k: Int, ctx: DeviceContext
) raises -> RouterPlan:
    """Per-token top-k expert selection + softmax-over-topk gating.

    logits: [T, E] (any compute dtype). k <= E.
    Returns a RouterPlan with host Lists (length T*k, token-major).
    Selection: descending logit, ties broken by lower expert index.
    """
    var sh = logits.shape()
    if len(sh) != 2:
        raise Error("top_k_router: logits must be 2-D [T, E]")
    var t = sh[0]
    var e = sh[1]
    if k < 1 or k > e:
        raise Error(
            String("top_k_router: k=") + String(k) + " out of range [1, E=" + String(e) + "]"
        )

    # Download logits to host as F32 (flame-core does top-k host-side too).
    var host = logits.to_host(ctx)  # length T*E, row-major

    var expert_ids = List[Int]()
    var gating = List[Float32]()
    for ti in range(t):
        var base = ti * e
        # Selection sort for the top-k of this row. E is tiny (experts count),
        # so an O(E*k) pass is fine and lets us apply the exact tie rule.
        # `taken[ei]` marks experts already pulled into the top-k.
        var taken = List[Bool]()
        for _ in range(e):
            taken.append(False)
        var sel = List[Int]()
        for _ in range(k):
            var best = -1
            var best_v: Float32 = 0.0
            for ei in range(e):
                if taken[ei]:
                    continue
                var v = host[base + ei]
                # Strictly-greater keeps the FIRST (lowest-index) expert on a
                # tie, since we scan ei ascending — matching flame-core's
                # `.then(a.idx.cmp(b.idx))` lower-index-wins tie break.
                if best == -1 or v > best_v:
                    best = ei
                    best_v = v
            taken[best] = True
            sel.append(best)
        # Softmax over ONLY the k selected logits (numerically stable).
        var maxv = host[base + sel[0]]
        for j in range(1, k):
            var v = host[base + sel[j]]
            if v > maxv:
                maxv = v
        var denom: Float32 = 0.0
        var exps = List[Float32]()
        for j in range(k):
            var ev = exp(host[base + sel[j]] - maxv)
            exps.append(ev)
            denom += ev
        for j in range(k):
            expert_ids.append(sel[j])
            gating.append(exps[j] / denom)

    return RouterPlan(
        expert_ids=expert_ids^,
        gating=gating^,
        num_tokens=t,
        num_experts=e,
        top_k=k,
    )


# ── row gather / scatter kernels (build per-expert blocks; write back) ────────
# gather_rows: dst[i, :] = src[row_idx[i], :]   (row_idx is a device i32 buffer)
def _gather_rows_kernel[dtype: DType](
    src: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    row_idx: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    n: Int,
    h: Int,
):
    var idx = Int(global_idx.x)
    var total = n * h
    if idx < total:
        var i = idx // h
        var col = idx % h
        var srow = Int(rebind[Scalar[DType.int32]](row_idx[i]))
        dst[i, col] = rebind[dst.element_type](
            rebind[Scalar[dtype]](src[srow, col])
        )


# scatter_rows: dst[row_idx[i], :] = src[i, :]
def _scatter_rows_kernel[dtype: DType](
    src: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    row_idx: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    n: Int,
    h: Int,
):
    var idx = Int(global_idx.x)
    var total = n * h
    if idx < total:
        var i = idx // h
        var col = idx % h
        var drow = Int(rebind[Scalar[DType.int32]](row_idx[i]))
        dst[drow, col] = rebind[dst.element_type](
            rebind[Scalar[dtype]](src[i, col])
        )


def _copy_bytes_kernel(
    src: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin],
    src_offset: Int,
    nbytes: Int,
):
    var idx = Int(global_idx.x)
    if idx < nbytes:
        dst[idx] = rebind[dst.element_type](
            rebind[Scalar[DType.uint8]](src[src_offset + idx])
        )


def _idx_to_dev_i32(
    idx: List[Int], ctx: DeviceContext
) raises -> DeviceBuffer[DType.uint8]:
    """HtoD copy of host Int indices into a REAL int32 device buffer.
    Mirrors flame-core: indices ride a true-i32 CudaSlice, not a Tensor."""
    var n = len(idx)
    var host = ctx.enqueue_create_host_buffer[DType.uint8](n * 4)
    var hp = host.unsafe_ptr().bitcast[Int32]()
    for i in range(n):
        hp[i] = Int32(idx[i])
    var dev = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return dev^


def _copy_tensor_slab_bytes(
    src: Tensor,
    elem_offset: Int,
    elem_count: Int,
    ctx: DeviceContext,
) raises -> DeviceBuffer[DType.uint8]:
    var bsz = src.dtype().byte_size()
    var nbytes = elem_count * bsz
    var dst = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](src.nbytes()))
    var dst_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](nbytes))
    var src_lt = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        src.buf.unsafe_ptr(), src_rl
    )
    var dst_lt = LayoutTensor[DType.uint8, _DYN1, MutAnyOrigin](
        dst.unsafe_ptr(), dst_rl
    )
    var grid = (nbytes + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_copy_bytes_kernel, _copy_bytes_kernel](
        src_lt, dst_lt, elem_offset * bsz, nbytes,
        grid_dim=grid, block_dim=_BLOCK,
    )
    return dst^


def _enqueue_gather_rows(
    tokens: Tensor,
    row_idx_dev: DeviceBuffer[DType.uint8],
    block_buf: DeviceBuffer[DType.uint8],
    n: Int,
    h: Int,
    ctx: DeviceContext,
) raises:
    var tsh = tokens.shape()
    var t = tsh[0]
    var tok_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](t, h))
    var blk_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](n, h))
    var idx_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var idx_lt = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        row_idx_dev.unsafe_ptr().bitcast[Int32](), idx_rl
    )
    var grid = (n * h + _BLOCK - 1) // _BLOCK
    var dt = tokens.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var src = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            tokens.buf.unsafe_ptr().bitcast[Float32](), tok_rl
        )
        var dst = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            block_buf.unsafe_ptr().bitcast[Float32](), blk_rl
        )
        ctx.enqueue_function[
            _gather_rows_kernel[DType.float32],
            _gather_rows_kernel[DType.float32],
        ](src, idx_lt, dst, n, h, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var src = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            tokens.buf.unsafe_ptr().bitcast[BFloat16](), tok_rl
        )
        var dst = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            block_buf.unsafe_ptr().bitcast[BFloat16](), blk_rl
        )
        ctx.enqueue_function[
            _gather_rows_kernel[DType.bfloat16],
            _gather_rows_kernel[DType.bfloat16],
        ](src, idx_lt, dst, n, h, grid_dim=grid, block_dim=_BLOCK)
    else:
        var src = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            tokens.buf.unsafe_ptr().bitcast[Float16](), tok_rl
        )
        var dst = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            block_buf.unsafe_ptr().bitcast[Float16](), blk_rl
        )
        ctx.enqueue_function[
            _gather_rows_kernel[DType.float16],
            _gather_rows_kernel[DType.float16],
        ](src, idx_lt, dst, n, h, grid_dim=grid, block_dim=_BLOCK)


def _enqueue_scatter_rows(
    src_tensor: Tensor,
    row_idx_dev: DeviceBuffer[DType.uint8],
    dst_buf: DeviceBuffer[DType.uint8],
    n_slots: Int,
    n: Int,
    h: Int,
    ctx: DeviceContext,
) raises:
    var src_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](n, h))
    var dst_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](n_slots, h))
    var idx_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var idx_lt = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        row_idx_dev.unsafe_ptr().bitcast[Int32](), idx_rl
    )
    var grid = (n * h + _BLOCK - 1) // _BLOCK
    var dt = src_tensor.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var src = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            src_tensor.buf.unsafe_ptr().bitcast[Float32](), src_rl
        )
        var dst = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            dst_buf.unsafe_ptr().bitcast[Float32](), dst_rl
        )
        ctx.enqueue_function[
            _scatter_rows_kernel[DType.float32],
            _scatter_rows_kernel[DType.float32],
        ](src, idx_lt, dst, n, h, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var src = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            src_tensor.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
        )
        var dst = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            dst_buf.unsafe_ptr().bitcast[BFloat16](), dst_rl
        )
        ctx.enqueue_function[
            _scatter_rows_kernel[DType.bfloat16],
            _scatter_rows_kernel[DType.bfloat16],
        ](src, idx_lt, dst, n, h, grid_dim=grid, block_dim=_BLOCK)
    else:
        var src = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            src_tensor.buf.unsafe_ptr().bitcast[Float16](), src_rl
        )
        var dst = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            dst_buf.unsafe_ptr().bitcast[Float16](), dst_rl
        )
        ctx.enqueue_function[
            _scatter_rows_kernel[DType.float16],
            _scatter_rows_kernel[DType.float16],
        ](src, idx_lt, dst, n, h, grid_dim=grid, block_dim=_BLOCK)


# ── stage 2: grouped expert FFN (loop over experts, callable matmul each) ─────
def grouped_expert_ffn(
    tokens: Tensor,
    gate_w: Tensor,
    up_w: Tensor,
    down_w: Tensor,
    plan: RouterPlan,
    ctx: DeviceContext,
) raises -> Tensor:
    """Per-expert SwiGLU FFN over the routed slots.

    tokens: [T, H]            (F32/BF16/F16 storage)
    gate_w: [E, F, H]         per-expert gate proj (PyTorch row-major [out,in])
    up_w:   [E, F, H]         per-expert up   proj
    down_w: [E, H, F]         per-expert down proj
    plan:   from top_k_router (expert_ids length T*k, token-major)
    returns expert_out [T*k, H], token-major slots (slot s = t*k + j).

    Implementation: loop over experts. For each expert gather the token rows of
    its routed slots into a contiguous block, run the SwiGLU FFN via the
    callable `linear` (linalg matmul) three times, then scatter the result rows
    back to their slot positions in expert_out. F32-accumulated GEMM throughout;
    tensor storage at the boundary stays in the input/weight dtype.
    """
    var tsh = tokens.shape()
    if len(tsh) != 2:
        raise Error("grouped_expert_ffn: tokens must be 2-D [T, H]")
    var h = tsh[1]
    var gsh = gate_w.shape()
    var ush = up_w.shape()
    var dsh = down_w.shape()
    if len(gsh) != 3 or len(ush) != 3 or len(dsh) != 3:
        raise Error("grouped_expert_ffn: expert weights must be 3-D [E, *, *]")
    var e = gsh[0]
    var f = gsh[1]
    if gsh[2] != h or ush[2] != h or ush[1] != f:
        raise Error("grouped_expert_ffn: gate/up weight shape mismatch vs [E,F,H]")
    if dsh[0] != e or dsh[1] != h or dsh[2] != f:
        raise Error("grouped_expert_ffn: down weight must be [E, H, F]")
    if e != plan.num_experts:
        raise Error("grouped_expert_ffn: expert count mismatch vs plan")

    var storage = tokens.dtype()
    _ = storage.to_mojo_dtype()
    if (
        gate_w.dtype() != storage
        or up_w.dtype() != storage
        or down_w.dtype() != storage
    ):
        raise Error("grouped_expert_ffn: tokens and weights must share dtype")

    var n_slots = len(plan.expert_ids)  # T*k

    # Output buffer [T*k, H]; every routed slot is filled by its expert pass.
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        n_slots * h * storage.byte_size()
    )

    for ei in range(e):
        # Collect the slots routed to expert ei. `src_tokens[i]` = token id of
        # the i-th slot for this expert; `dst_slots[i]` = its slot index in
        # expert_out (= t*k + j), preserving token-major slot order.
        var src_tokens = List[Int]()
        var dst_slots = List[Int]()
        for s in range(n_slots):
            if plan.expert_ids[s] == ei:
                src_tokens.append(s // plan.top_k)  # token id of this slot
                dst_slots.append(s)
        var n_e = len(src_tokens)
        if n_e == 0:
            continue

        # Gather this expert's token rows into a [n_e, H] block on-GPU.
        var src_idx_dev = _idx_to_dev_i32(src_tokens, ctx)
        var blk_buf = ctx.enqueue_create_buffer[DType.uint8](
            n_e * h * storage.byte_size()
        )
        _enqueue_gather_rows(tokens, src_idx_dev, blk_buf, n_e, h, ctx)
        var x_e = Tensor(blk_buf^, [n_e, h], storage)

        # Copy this expert's contiguous weight slabs on device, preserving the
        # checkpoint storage dtype. `linear` owns the F32 GEMM accumulator.
        var gbase = ei * f * h
        var dbase = ei * h * f
        var gate_buf = _copy_tensor_slab_bytes(gate_w, gbase, f * h, ctx)
        var up_buf = _copy_tensor_slab_bytes(up_w, gbase, f * h, ctx)
        var down_buf = _copy_tensor_slab_bytes(down_w, dbase, h * f, ctx)
        var gate_we = Tensor(gate_buf^, [f, h], storage)
        var up_we = Tensor(up_buf^, [f, h], storage)
        var down_we = Tensor(down_buf^, [h, f], storage)

        # SwiGLU FFN:  down( silu(x@gate^T) * (x@up^T) ).
        var g_proj = linear(x_e, gate_we, None, ctx)   # [n_e, F]
        var u_proj = linear(x_e, up_we, None, ctx)      # [n_e, F]
        var hmid = swiglu(g_proj, u_proj, ctx)          # [n_e, F]
        var y_e = linear(hmid, down_we, None, ctx)      # [n_e, H]

        # Scatter the result rows back to their slots in expert_out.
        var dst_idx_dev = _idx_to_dev_i32(dst_slots, ctx)
        _enqueue_scatter_rows(y_e, dst_idx_dev, out_buf, n_slots, n_e, h, ctx)
        ctx.synchronize()

    return Tensor(out_buf^, [n_slots, h], storage)


# ── stage 3: gated scatter-add (atomic, mirrors fused_gated_scatter_add) ──────
def _gated_scatter_add_kernel[dtype: DType](
    expert_out: LayoutTensor[dtype, _DYN2, MutAnyOrigin],
    gating: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    indices: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    accum: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    n_slots: Int,
    d: Int,
    n_rows: Int,
):
    # One thread per (slot, dim) element. accum[idx, col] += eo[s,col]*gate[s].
    var idx = Int(global_idx.x)
    var total = n_slots * d
    if idx < total:
        var s = idx // d
        var col = idx % d
        var row = Int(rebind[Scalar[DType.int32]](indices[s]))
        if row < 0 or row >= n_rows:
            return  # out-of-range row skipped (flame-core's "no expert" case)
        var g = rebind[Scalar[DType.float32]](gating[s])
        var v = rebind[Scalar[dtype]](expert_out[s, col]).cast[DType.float32]() * g
        # Atomic add: k>1 slots collide on the same accum row.
        var dst_ptr = accum.ptr + (row * d + col)
        _ = Atomic[DType.float32].fetch_add(dst_ptr, v)


def _enqueue_gated_scatter_add_f32_accum(
    expert_out: Tensor,
    gating_buf: DeviceBuffer[DType.uint8],
    idx_dev: DeviceBuffer[DType.uint8],
    accum_buf: DeviceBuffer[DType.uint8],
    n_slots: Int,
    d: Int,
    n_rows: Int,
    ctx: DeviceContext,
) raises:
    var eo_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](n_slots, d))
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_slots))
    var i_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_slots))
    var acc_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](n_rows, d))

    var g_lt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        gating_buf.unsafe_ptr().bitcast[Float32](), g_rl
    )
    var i_lt = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        idx_dev.unsafe_ptr().bitcast[Int32](), i_rl
    )
    var acc_lt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        accum_buf.unsafe_ptr().bitcast[Float32](), acc_rl
    )

    var grid = (n_slots * d + _BLOCK - 1) // _BLOCK
    var dt = expert_out.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var eo_lt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            expert_out.buf.unsafe_ptr().bitcast[Float32](), eo_rl
        )
        ctx.enqueue_function[
            _gated_scatter_add_kernel[DType.float32],
            _gated_scatter_add_kernel[DType.float32],
        ](
            eo_lt, g_lt, i_lt, acc_lt, n_slots, d, n_rows,
            grid_dim=grid, block_dim=_BLOCK,
        )
    elif dt == DType.bfloat16:
        var eo_lt = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            expert_out.buf.unsafe_ptr().bitcast[BFloat16](), eo_rl
        )
        ctx.enqueue_function[
            _gated_scatter_add_kernel[DType.bfloat16],
            _gated_scatter_add_kernel[DType.bfloat16],
        ](
            eo_lt, g_lt, i_lt, acc_lt, n_slots, d, n_rows,
            grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        var eo_lt = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            expert_out.buf.unsafe_ptr().bitcast[Float16](), eo_rl
        )
        ctx.enqueue_function[
            _gated_scatter_add_kernel[DType.float16],
            _gated_scatter_add_kernel[DType.float16],
        ](
            eo_lt, g_lt, i_lt, acc_lt, n_slots, d, n_rows,
            grid_dim=grid, block_dim=_BLOCK,
        )


def gated_scatter_add(
    expert_out: Tensor,
    gating: List[Float32],
    indices: List[Int],
    mut accum: Tensor,
    ctx: DeviceContext,
) raises:
    """accum[indices[s]] += expert_out[s] * gating[s]   (in-place, all s).

    expert_out: [T*k, D]  (F32/BF16/F16 storage)
    gating:     host List[Float32] length T*k
    indices:    host List[Int]     length T*k (negative/out-of-range skipped)
    accum:      [N, D]    F32/BF16/F16, updated IN PLACE. BF16/F16 accum uses
                a private F32 atomic accumulator scratch and casts back before
                returning.
    """
    var eosh = expert_out.shape()
    var accsh = accum.shape()
    if len(eosh) != 2:
        raise Error("gated_scatter_add: expert_out must be 2-D [T*k, D]")
    if len(accsh) != 2:
        raise Error("gated_scatter_add: accum must be 2-D [N, D]")
    var n_slots = eosh[0]
    var d = eosh[1]
    var n_rows = accsh[0]
    if accsh[1] != d:
        raise Error("gated_scatter_add: accum D != expert_out D")
    if len(gating) != n_slots:
        raise Error("gated_scatter_add: gating length != T*k")
    if len(indices) != n_slots:
        raise Error("gated_scatter_add: indices length != T*k")
    _ = expert_out.dtype().to_mojo_dtype()
    _ = accum.dtype().to_mojo_dtype()

    # Stage gating (F32) + indices (i32) to device buffers.
    var g_buf = ctx.enqueue_create_buffer[DType.uint8](n_slots * 4)
    var g_host = ctx.enqueue_create_host_buffer[DType.uint8](n_slots * 4)
    var gp = g_host.unsafe_ptr().bitcast[Float32]()
    for i in range(n_slots):
        gp[i] = gating[i]
    ctx.enqueue_copy(dst_buf=g_buf, src_buf=g_host)
    ctx.synchronize()
    var idx_dev = _idx_to_dev_i32(indices, ctx)

    if accum.dtype() == STDtype.F32:
        _enqueue_gated_scatter_add_f32_accum(
            expert_out, g_buf, idx_dev, accum.buf, n_slots, d, n_rows, ctx
        )
        ctx.synchronize()
    else:
        var accum_f32 = cast_tensor(accum, STDtype.F32, ctx, False)
        _enqueue_gated_scatter_add_f32_accum(
            expert_out, g_buf, idx_dev, accum_f32.buf, n_slots, d, n_rows, ctx
        )
        var accum_storage = cast_tensor(accum_f32, accum.dtype(), ctx, False)
        ctx.enqueue_copy(dst_buf=accum.buf, src_buf=accum_storage.buf)
        ctx.synchronize()
