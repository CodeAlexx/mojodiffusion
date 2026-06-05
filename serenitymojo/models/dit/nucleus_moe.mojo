# nucleus_moe.mojo — Nucleus-Image expert-choice MoE FFN (NEW, Nucleus-only).
#
# WHY THIS FILE EXISTS (ops/moe divergence — see report):
#   serenitymojo's shared `ops/moe.mojo` implements TOKEN-CHOICE top-k routing:
#   every token picks its k best experts (softmax over the k selected logits).
#   Nucleus-Image uses EXPERT-CHOICE routing with a per-expert capacity: every
#   (batch, expert) picks its top-C tokens by affinity, gating = the affinity
#   value renormalised PER-TOKEN across that token's picks and scaled by
#   `route_scale=2.5`. These are fundamentally different routing topologies, so
#   `ops/moe.top_k_router` + `grouped_expert_ffn` cannot be reused for Nucleus.
#   We therefore build the expert-choice route + grouped FFN here and REUSE
#   `ops/moe.gated_scatter_add` (which IS a bit-for-bit fit — `accum[idx[s]] +=
#   eo[s]*gate[s]` atomic add is exactly the Nucleus weighted unpermute).
#
# Mirrors flame-core `ops/nucleus_moe.rs::nucleus_moe_expert_forward` +
# `ops/moe_routing.rs::expert_choice_route`, which mirror diffusers
# `transformer_nucleusmoe_image.py::NucleusMoELayer.forward` /
# `SwiGLUExperts._run_experts_grouped_mm`.
#
# Out of scope here (caller handles): the router matmul + softmax + transpose
# producing `affinity [B,E,S]`, the modulation split (modulated -> experts,
# unmodulated -> router), and the shared-expert FFN add. Tensor boundaries
# preserve x/weight storage dtype; F32 remains inside GEMM/scatter math.

from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import global_idx
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.tensor_algebra import gather_rows
from serenitymojo.ops.moe import gated_scatter_add


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


# ── expert-choice routing plan (host-side; mirrors expert_choice_route) ───────
@fieldwise_init
struct ExpertChoicePlan(Movable):
    """Output of `expert_choice_route`. Expert-major picks: for expert ei and
    pick slot c, the global token index is `global_token_indices[ei*B*C + ...]`
    and its renormalised+scaled gate is `gating_flat[...]`. Length = E*B*C."""

    var global_token_indices: List[Int]   # length E*B*C, expert-major
    var gating_flat: List[Float32]         # length E*B*C, renorm * route_scale
    var batch_size: Int
    var seq_len: Int
    var num_experts: Int
    var capacity: Int


def expert_choice_route(
    affinity: Tensor, capacity: Int, route_scale: Float32, ctx: DeviceContext
) raises -> ExpertChoicePlan:
    """Expert-choice top-C routing. Mirrors flame-core expert_choice_route +
    the renorm/scale in nucleus_moe.rs's scalar_ref.

    affinity: [B, E, S] F32 — post softmax(router_logits).transpose(1,2).
    capacity: per-(batch,expert) pick count C.
    route_scale: gate multiplier (Nucleus 2.5).

    Returns an ExpertChoicePlan with expert-major global token indices + gates.
    Selection: descending affinity, ties broken by LOWER token index (matches
    flame-core `b.partial_cmp(a).then(a.idx.cmp(b.idx))`).
    """
    var sh = affinity.shape()
    if len(sh) != 3:
        raise Error("expert_choice_route: affinity must be 3-D [B, E, S]")
    var b = sh[0]
    var e = sh[1]
    var s = sh[2]
    if capacity < 1:
        raise Error("expert_choice_route: capacity must be >= 1")
    if capacity > s:
        raise Error("expert_choice_route: capacity exceeds seq_len")

    var host = affinity.to_host(ctx)  # B*E*S row-major

    # top-C per (b, e): top_idx[(b*e+ei)*C + c], top_w[...] — token-major within.
    var top_idx = List[Int]()
    var top_w = List[Float32]()
    top_idx.resize(b * e * capacity, 0)
    top_w.resize(b * e * capacity, Float32(0.0))

    for bi in range(b):
        for ei in range(e):
            var rbase = (bi * e + ei) * s
            # selection sort top-C; S can be large but C is small (2-4).
            var taken = List[Bool]()
            taken.resize(s, False)
            var dst = (bi * e + ei) * capacity
            for c in range(capacity):
                var best = -1
                var best_v: Float32 = 0.0
                for si in range(s):
                    if taken[si]:
                        continue
                    var v = host[rbase + si]
                    # strictly-greater keeps the FIRST (lowest) token on a tie.
                    if best == -1 or v > best_v:
                        best = si
                        best_v = v
                taken[best] = True
                top_idx[dst + c] = best
                top_w[dst + c] = best_v

    # Flatten to expert-major (E, B, C), converting token idx to global
    # (batch-offset) token idx. global = bi*s + top_idx.
    var n_picks = e * b * capacity
    var global_idx = List[Int]()
    var gating = List[Float32]()
    global_idx.resize(n_picks, 0)
    gating.resize(n_picks, Float32(0.0))
    for ei in range(e):
        for bi in range(b):
            var src = (bi * e + ei) * capacity
            var dstp = (ei * b + bi) * capacity
            var batch_off = bi * s
            for c in range(capacity):
                global_idx[dstp + c] = batch_off + top_idx[src + c]
                gating[dstp + c] = top_w[src + c]

    # Renormalise per token across its picks, then scale.
    var total_tokens = b * s
    var tsum = List[Float32]()
    tsum.resize(total_tokens, Float32(0.0))
    for k in range(n_picks):
        var i = global_idx[k]
        if i >= 0 and i < total_tokens:
            tsum[i] += gating[k]
    for k in range(n_picks):
        var i = global_idx[k]
        var denom = tsum[i] + Float32(1e-12)
        gating[k] = (gating[k] / denom) * route_scale

    return ExpertChoicePlan(
        global_token_indices=global_idx^,
        gating_flat=gating^,
        batch_size=b,
        seq_len=s,
        num_experts=e,
        capacity=capacity,
    )


def _gate_up_slab_kernel[dtype: DType](
    src: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    gate: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    up: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    expert: Int,
    d: Int,
    inter: Int,
):
    var idx = Int(global_idx.x)
    var total = inter * d
    if idx < total:
        var ni = idx // d
        var di = idx % d
        var two_inter = inter * 2
        var src_base = expert * d * two_inter + di * two_inter
        gate[idx] = rebind[gate.element_type](
            rebind[Scalar[dtype]](src[src_base + ni])
        )
        up[idx] = rebind[up.element_type](
            rebind[Scalar[dtype]](src[src_base + inter + ni])
        )


def _down_slab_kernel[dtype: DType](
    src: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    dst: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    expert: Int,
    d: Int,
    inter: Int,
):
    var idx = Int(global_idx.x)
    var total = d * inter
    if idx < total:
        var di = idx // inter
        var fi = idx % inter
        var src_idx = expert * inter * d + fi * d + di
        dst[idx] = rebind[dst.element_type](
            rebind[Scalar[dtype]](src[src_idx])
        )


def _enqueue_gate_up_slabs(
    gate_up_w: Tensor,
    gate_buf: DeviceBuffer[DType.uint8],
    up_buf: DeviceBuffer[DType.uint8],
    expert: Int,
    d: Int,
    inter: Int,
    ctx: DeviceContext,
) raises:
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](gate_up_w.numel()))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](inter * d))
    var grid = (inter * d + _BLOCK - 1) // _BLOCK
    var dt = gate_up_w.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var src = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            gate_up_w.buf.unsafe_ptr().bitcast[Float32](), src_rl
        )
        var gate = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            gate_buf.unsafe_ptr().bitcast[Float32](), out_rl
        )
        var up = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            up_buf.unsafe_ptr().bitcast[Float32](), out_rl
        )
        ctx.enqueue_function[
            _gate_up_slab_kernel[DType.float32],
            _gate_up_slab_kernel[DType.float32],
        ](src, gate, up, expert, d, inter, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var src = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            gate_up_w.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
        )
        var gate = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            gate_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
        )
        var up = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            up_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
        )
        ctx.enqueue_function[
            _gate_up_slab_kernel[DType.bfloat16],
            _gate_up_slab_kernel[DType.bfloat16],
        ](src, gate, up, expert, d, inter, grid_dim=grid, block_dim=_BLOCK)
    else:
        var src = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            gate_up_w.buf.unsafe_ptr().bitcast[Float16](), src_rl
        )
        var gate = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            gate_buf.unsafe_ptr().bitcast[Float16](), out_rl
        )
        var up = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            up_buf.unsafe_ptr().bitcast[Float16](), out_rl
        )
        ctx.enqueue_function[
            _gate_up_slab_kernel[DType.float16],
            _gate_up_slab_kernel[DType.float16],
        ](src, gate, up, expert, d, inter, grid_dim=grid, block_dim=_BLOCK)


def _enqueue_down_slab(
    down_w: Tensor,
    down_buf: DeviceBuffer[DType.uint8],
    expert: Int,
    d: Int,
    inter: Int,
    ctx: DeviceContext,
) raises:
    var src_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](down_w.numel()))
    var out_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](d * inter))
    var grid = (d * inter + _BLOCK - 1) // _BLOCK
    var dt = down_w.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var src = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            down_w.buf.unsafe_ptr().bitcast[Float32](), src_rl
        )
        var dst = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            down_buf.unsafe_ptr().bitcast[Float32](), out_rl
        )
        ctx.enqueue_function[
            _down_slab_kernel[DType.float32],
            _down_slab_kernel[DType.float32],
        ](src, dst, expert, d, inter, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var src = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            down_w.buf.unsafe_ptr().bitcast[BFloat16](), src_rl
        )
        var dst = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            down_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
        )
        ctx.enqueue_function[
            _down_slab_kernel[DType.bfloat16],
            _down_slab_kernel[DType.bfloat16],
        ](src, dst, expert, d, inter, grid_dim=grid, block_dim=_BLOCK)
    else:
        var src = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            down_w.buf.unsafe_ptr().bitcast[Float16](), src_rl
        )
        var dst = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            down_buf.unsafe_ptr().bitcast[Float16](), out_rl
        )
        ctx.enqueue_function[
            _down_slab_kernel[DType.float16],
            _down_slab_kernel[DType.float16],
        ](src, dst, expert, d, inter, grid_dim=grid, block_dim=_BLOCK)


# ── expert-choice grouped SwiGLU FFN ──────────────────────────────────────────
def nucleus_moe_expert_forward(
    x_flat: Tensor,        # [B*S, D]  modulated hidden states (storage dtype)
    affinity: Tensor,      # [B, E, S] F32
    gate_up_w: Tensor,     # [E, D, 2*inter]  per-expert stacked gate||up
    down_w: Tensor,        # [E, inter, D]
    capacity: Int,
    route_scale: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    """SwiGLU MoE expert forward (expert-choice). Returns [B*S, D] in x dtype.

    Mirrors flame-core nucleus_moe_expert_forward step-for-step but with a
    per-expert loop (callable `linear`) instead of a grouped GEMM, and reusing
    ops/moe.gated_scatter_add for the weighted unpermute. Tokens not picked by
    any expert get 0 here (the caller's shared expert supplies their value).

    gate_up_w layout: cols [0:inter] = gate, [inter:2*inter] = up
    (SwiGLUExperts `[gate, up]` order — opposite of the dense FFN `[up, gate]`).
    """
    var xd = x_flat.shape()
    if len(xd) != 2:
        raise Error("nucleus_moe_expert_forward: x_flat must be 2-D [B*S, D]")
    var d = xd[1]
    var gud = gate_up_w.shape()
    if len(gud) != 3 or gud[1] != d:
        raise Error("nucleus_moe_expert_forward: gate_up_w must be [E, D, 2*inter]")
    var e = gud[0]
    var two_inter = gud[2]
    if two_inter % 2 != 0:
        raise Error("nucleus_moe_expert_forward: gate_up_w last dim must be even")
    var inter = two_inter // 2
    var dwd = down_w.shape()
    if len(dwd) != 3 or dwd[0] != e or dwd[1] != inter or dwd[2] != d:
        raise Error("nucleus_moe_expert_forward: down_w must be [E, inter, D]")
    var storage = x_flat.dtype()
    _ = storage.to_mojo_dtype()
    if gate_up_w.dtype() != storage or down_w.dtype() != storage:
        raise Error("nucleus_moe_expert_forward: x/weight dtype mismatch")

    var plan = expert_choice_route(affinity, capacity, route_scale, ctx)
    var total_tokens = plan.batch_size * plan.seq_len

    # Accumulator [N, D] in model storage dtype. gated_scatter_add uses private
    # F32 atomic scratch for BF16/F16 and writes back before returning.
    var acc_bytes = total_tokens * d * storage.byte_size()
    var acc_buf = ctx.enqueue_create_buffer[DType.uint8](acc_bytes)
    ctx.enqueue_memset[DType.uint8](acc_buf, 0)
    ctx.synchronize()
    var accum = Tensor(acc_buf^, [total_tokens, d], storage)

    var bc = plan.batch_size * plan.capacity  # picks per expert

    for ei in range(e):
        # This expert's contiguous pick slots are [ei*bc, (ei+1)*bc).
        var slot0 = ei * bc
        # Build the [bc] gather index (global token id per pick) on host.
        var gidx = List[Int]()
        gidx.resize(bc, 0)
        for c in range(bc):
            gidx[c] = plan.global_token_indices[slot0 + c]

        # Gather this expert's token rows -> [bc, D] via foundation gather_rows.
        var x_e = gather_rows(x_flat, gidx, ctx)  # [bc, D]

        # Slice this expert's gate_up [D, 2*inter] and down [inter, D] slabs on
        # device, preserving storage dtype. linear() wants weight [out, in].
        # gate_up_w is [E, D, 2*inter]; linear() wants weight [out, in].
        # gate proj: out=inter, in=D  -> rows are cols [0:inter] of gate_up.
        # down_w [E, inter, D]; down weight[o=di, in=fi] = down[fi, di].
        var gate_buf = ctx.enqueue_create_buffer[DType.uint8](
            inter * d * storage.byte_size()
        )
        var up_buf = ctx.enqueue_create_buffer[DType.uint8](
            inter * d * storage.byte_size()
        )
        var down_buf = ctx.enqueue_create_buffer[DType.uint8](
            d * inter * storage.byte_size()
        )
        _enqueue_gate_up_slabs(gate_up_w, gate_buf, up_buf, ei, d, inter, ctx)
        _enqueue_down_slab(down_w, down_buf, ei, d, inter, ctx)
        var gate_we = Tensor(gate_buf^, [inter, d], storage)
        var up_we = Tensor(up_buf^, [inter, d], storage)
        var down_we = Tensor(down_buf^, [d, inter], storage)

        var g_proj = linear(x_e, gate_we, None, ctx)   # [bc, inter]
        var u_proj = linear(x_e, up_we, None, ctx)      # [bc, inter]
        var hmid = swiglu(g_proj, u_proj, ctx)          # [bc, inter]
        var y_e = linear(hmid, down_we, None, ctx)      # [bc, D]

        # Weighted scatter-add into accum (REUSES ops/moe.gated_scatter_add).
        var gates = List[Float32]()
        gates.resize(bc, Float32(0.0))
        for c in range(bc):
            gates[c] = plan.gating_flat[slot0 + c]
        gated_scatter_add(y_e, gates, gidx, accum, ctx)

    return accum^
