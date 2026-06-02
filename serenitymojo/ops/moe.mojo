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
#     Mirrors fused_gated_scatter_add.rs: atomic-add scatter into an F32
#     accumulator (so the k slots that collide on the same token row combine
#     correctly for top-k>1). Out-of-range / negative indices are skipped.
#     `indices` is a host List[Int] copied HtoD as a REAL i32 buffer (flame-core
#     passes host &[i32] the same way — its DType::I32 Tensors are f32-bytes-
#     relabeled, so it never routes indices through a Tensor either).
#
# (MoT note: SenseNova-U1's per-modality dispatch is just top_k_router applied
# per modality-stream — no extra kernel; the router here covers it.)
#
# Mojo 1.0.0b1, NVIDIA GPU. F32 accumulation. Reuses ops.linear + ops.swiglu.

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
def _gather_rows_kernel_f32(
    src: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    row_idx: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    n: Int,
    h: Int,
):
    var idx = Int(global_idx.x)
    var total = n * h
    if idx < total:
        var i = idx // h
        var col = idx % h
        var srow = Int(rebind[Scalar[DType.int32]](row_idx[i]))
        dst[i, col] = src[srow, col]


# scatter_rows: dst[row_idx[i], :] = src[i, :]
def _scatter_rows_kernel_f32(
    src: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    row_idx: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
    n: Int,
    h: Int,
):
    var idx = Int(global_idx.x)
    var total = n * h
    if idx < total:
        var i = idx // h
        var col = idx % h
        var drow = Int(rebind[Scalar[DType.int32]](row_idx[i]))
        dst[drow, col] = src[i, col]


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

    tokens: [T, H]            (compute dtype)
    gate_w: [E, F, H]         per-expert gate proj (PyTorch row-major [out,in])
    up_w:   [E, F, H]         per-expert up   proj
    down_w: [E, H, F]         per-expert down proj
    plan:   from top_k_router (expert_ids length T*k, token-major)
    returns expert_out [T*k, H], token-major slots (slot s = t*k + j).

    Implementation: loop over experts. For each expert gather the token rows of
    its routed slots into a contiguous block, run the SwiGLU FFN via the
    callable `linear` (linalg matmul) three times, then scatter the result rows
    back to their slot positions in expert_out. F32-accumulated GEMM throughout.
    """
    var tsh = tokens.shape()
    if len(tsh) != 2:
        raise Error("grouped_expert_ffn: tokens must be 2-D [T, H]")
    var t = tsh[0]
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

    var dt = tokens.dtype().to_mojo_dtype()
    if dt != DType.float32:
        # Gather/scatter kernels here are F32-typed for clarity; the parity gate
        # runs F32 storage. (BF16 path would add bf16-typed gather/scatter twins
        # — omitted to keep this minimal and correct.)
        raise Error("grouped_expert_ffn: only F32 storage supported in this build")

    var n_slots = len(plan.expert_ids)  # T*k

    # Per-expert weight slices live contiguously in gate_w/up_w/down_w; we build
    # a [out, in] weight Tensor for expert e by H2D-uploading that slab. We read
    # the full weights once to host (dev oracle path is F32 anyway), then upload
    # per-expert blocks. Tokens are gathered on-GPU via the row kernel.
    var gate_host = gate_w.to_host(ctx)  # E*F*H
    var up_host = up_w.to_host(ctx)      # E*F*H
    var down_host = down_w.to_host(ctx)  # E*H*F

    # Output buffer [T*k, H], zero-init (slots get filled per expert).
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](n_slots * h * 4)
    var out_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](n_slots, h))

    # tokens as a device LayoutTensor for the gather kernel.
    var tok_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](t, h))
    var tok_lt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        tokens.buf.unsafe_ptr().bitcast[Float32](), tok_rl
    )
    var out_lt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[Float32](), out_rl
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
        var blk_buf = ctx.enqueue_create_buffer[DType.uint8](n_e * h * 4)
        var blk_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](n_e, h))
        var src_idx_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_e))
        var blk_lt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            blk_buf.unsafe_ptr().bitcast[Float32](), blk_rl
        )
        var src_idx_lt = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
            src_idx_dev.unsafe_ptr().bitcast[Int32](), src_idx_rl
        )
        var ggrid = (n_e * h + _BLOCK - 1) // _BLOCK
        ctx.enqueue_function[_gather_rows_kernel_f32, _gather_rows_kernel_f32](
            tok_lt, src_idx_lt, blk_lt, n_e, h,
            grid_dim=ggrid, block_dim=_BLOCK,
        )
        ctx.synchronize()
        var x_e = Tensor(blk_buf^, [n_e, h], STDtype.F32)

        # Upload this expert's weight slabs.
        var gate_slab = List[Float32]()
        var up_slab = List[Float32]()
        var down_slab = List[Float32]()
        var gbase = ei * f * h
        for i in range(f * h):
            gate_slab.append(gate_host[gbase + i])
            up_slab.append(up_host[gbase + i])
        var dbase = ei * h * f
        for i in range(h * f):
            down_slab.append(down_host[dbase + i])
        var gate_we = Tensor.from_host(gate_slab, [f, h], STDtype.F32, ctx)
        var up_we = Tensor.from_host(up_slab, [f, h], STDtype.F32, ctx)
        var down_we = Tensor.from_host(down_slab, [h, f], STDtype.F32, ctx)

        # SwiGLU FFN:  down( silu(x@gate^T) * (x@up^T) ).
        var g_proj = linear(x_e, gate_we, None, ctx)   # [n_e, F]
        var u_proj = linear(x_e, up_we, None, ctx)      # [n_e, F]
        var hmid = swiglu(g_proj, u_proj, ctx)          # [n_e, F]
        var y_e = linear(hmid, down_we, None, ctx)      # [n_e, H]

        # Scatter the result rows back to their slots in expert_out.
        var dst_idx_dev = _idx_to_dev_i32(dst_slots, ctx)
        var dst_idx_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_e))
        var dst_idx_lt = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
            dst_idx_dev.unsafe_ptr().bitcast[Int32](), dst_idx_rl
        )
        var y_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](n_e, h))
        var y_lt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            y_e.buf.unsafe_ptr().bitcast[Float32](), y_rl
        )
        ctx.enqueue_function[_scatter_rows_kernel_f32, _scatter_rows_kernel_f32](
            y_lt, dst_idx_lt, out_lt, n_e, h,
            grid_dim=ggrid, block_dim=_BLOCK,
        )
        ctx.synchronize()

    return Tensor(out_buf^, [n_slots, h], STDtype.F32)


# ── stage 3: gated scatter-add (atomic, mirrors fused_gated_scatter_add) ──────
def _gated_scatter_add_kernel_f32(
    expert_out: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],
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
        var v = rebind[Scalar[DType.float32]](expert_out[s, col]) * g
        # Atomic add: k>1 slots collide on the same accum row.
        var dst_ptr = accum.ptr + (row * d + col)
        _ = Atomic[DType.float32].fetch_add(dst_ptr, v)


def gated_scatter_add(
    expert_out: Tensor,
    gating: List[Float32],
    indices: List[Int],
    mut accum: Tensor,
    ctx: DeviceContext,
) raises:
    """accum[indices[s]] += expert_out[s] * gating[s]   (in-place, all s).

    expert_out: [T*k, D]  (F32 storage)
    gating:     host List[Float32] length T*k
    indices:    host List[Int]     length T*k (negative/out-of-range skipped)
    accum:      [N, D]    F32, updated IN PLACE via atomic add.
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
    if expert_out.dtype() != STDtype.F32 or accum.dtype() != STDtype.F32:
        raise Error("gated_scatter_add: expert_out and accum must be F32")

    # Stage gating (F32) + indices (i32) to device buffers.
    var g_buf = ctx.enqueue_create_buffer[DType.uint8](n_slots * 4)
    var g_host = ctx.enqueue_create_host_buffer[DType.uint8](n_slots * 4)
    var gp = g_host.unsafe_ptr().bitcast[Float32]()
    for i in range(n_slots):
        gp[i] = gating[i]
    ctx.enqueue_copy(dst_buf=g_buf, src_buf=g_host)
    ctx.synchronize()
    var idx_dev = _idx_to_dev_i32(indices, ctx)

    var eo_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](n_slots, d))
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_slots))
    var i_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_slots))
    var acc_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](n_rows, d))

    var eo_lt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        expert_out.buf.unsafe_ptr().bitcast[Float32](), eo_rl
    )
    var g_lt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        g_buf.unsafe_ptr().bitcast[Float32](), g_rl
    )
    var i_lt = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        idx_dev.unsafe_ptr().bitcast[Int32](), i_rl
    )
    var acc_lt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        accum.buf.unsafe_ptr().bitcast[Float32](), acc_rl
    )

    var grid = (n_slots * d + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[
        _gated_scatter_add_kernel_f32, _gated_scatter_add_kernel_f32
    ](
        eo_lt, g_lt, i_lt, acc_lt, n_slots, d, n_rows,
        grid_dim=grid, block_dim=_BLOCK,
    )
    ctx.synchronize()
