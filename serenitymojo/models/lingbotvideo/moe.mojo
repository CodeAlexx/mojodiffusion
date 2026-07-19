# models/lingbotvideo/moe.mojo — LingBotVideoSparseMoeBlock (Chunk A2).
#
# Grouped-sigmoid token-choice MoE router + 128 SwiGLU experts (top-8) + 1
# shared expert. Mirrors transformer_lingbot_video.py exactly:
#   LingBotVideoRouter.forward (369-385)
#   LingBotVideoSparseMoeBlock.forward (854-872)
#   _run_experts_for_loop (555-568)  [eager per-expert path]
#
# ROUTER (per token, all FP32, host-side — like ops.moe.top_k_router):
#   logits = tokens.float() @ router_weightᵀ            (S,128), no bias
#   scores = sigmoid(logits)                             (S,128)
#   scores_for_choice = scores + e_score_correction_bias (S,128)
#   group-limited top-k: view (S,4,32), sum top-2 per group -> (S,4); keep
#     top-2 groups; mask experts NOT in those groups to -inf; top-8 of the
#     masked scores_for_choice -> selected expert ids.
#   CRITICAL ASYMMETRY: SELECTION uses scores_for_choice (bias-added), but the
#     gating WEIGHT gathers from the BIAS-FREE `scores`.
#   top_scores = scores.gather(top_indices); /= (sum + 1e-20); *= 2.5
#
#   Ordering: torch.topk(..., sorted=False) on CUDA returns the strictly-greater
#   experts in ASCENDING index order, followed by the threshold (== k-th value)
#   experts in ascending index order. We reproduce that order exactly so the
#   selected-ids array matches the oracle position-by-position (verified 576/576).
#
# EXPERTS (bf16 SwiGLU): down(silu(t@w1ᵀ)·(t@w3ᵀ)) with w2 as down.
#   Reuses ops.moe.grouped_expert_ffn (gate_w=w1, up_w=w3, down_w=w2) + the
#   weighted top-8 combine via ops.moe.gated_scatter_add (F32 accumulate).
# SHARED EXPERT: down(silu(t@gateᵀ)·(t@upᵀ)); out = routed + shared.
#
# Mojo 1.0.0b1, NVIDIA GPU. Inference-only, GPU-only.

from std.math import exp
from std.gpu.host import DeviceContext, DeviceBuffer
from std.gpu import thread_idx, block_idx, barrier, global_idx
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.tensor_algebra import add
from serenitymojo.ops.moe import RouterPlan, grouped_expert_ffn


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)


comptime H = 2048
comptime E = 128
comptime TOP_K = 8
comptime MOE_I = 768
comptime N_GROUP = 4
comptime TOPK_GROUP = 2
comptime EXPERTS_PER_GROUP = 32  # E // N_GROUP
comptime ROUTED_SCALING = Float32(2.5)
comptime _NEG_INF = Float32(-1.0e30)


def _sigmoid_stable(x: Float32) -> Float32:
    # numerically-stable logistic sigmoid.
    if x >= 0.0:
        return 1.0 / (1.0 + exp(-x))
    var ex = exp(x)
    return ex / (1.0 + ex)


def lingbot_grouped_sigmoid_router(
    logits: Tensor, bias: Tensor, ctx: DeviceContext
) raises -> RouterPlan:
    """Grouped-sigmoid token-choice router (host-side, all F32).

    logits: [S, E]  F32 (tokens.float() @ router_weightᵀ, no bias).
    bias:   [E]     F32 (e_score_correction_bias).
    Returns a RouterPlan whose `expert_ids` (length S*TOP_K, token-major) is the
    selected top-8 per token in torch sorted=False order, and whose `gating`
    holds the normalised, ×2.5-scaled weights gathered from the BIAS-FREE
    sigmoid scores.
    """
    var sh = logits.shape()
    if len(sh) != 2:
        raise Error("lingbot_router: logits must be 2-D [S, E]")
    var s = sh[0]
    var e = sh[1]
    if e != E:
        raise Error("lingbot_router: expected E=128 experts")
    var bsh = bias.shape()
    if len(bsh) != 1 or bsh[0] != e:
        raise Error("lingbot_router: bias must be [E]")

    var lg = logits.to_host(ctx)   # length S*E, row-major F32
    var bh = bias.to_host(ctx)     # length E, F32

    var expert_ids = List[Int]()
    var gating = List[Float32]()

    for t in range(s):
        var base = t * e
        # scores (bias-free) and scores_for_choice (bias-added).
        var scores = List[Float32]()
        var sfc = List[Float32]()
        for ei in range(e):
            var sc = _sigmoid_stable(lg[base + ei])
            scores.append(sc)
            sfc.append(sc + bh[ei])

        # group_scores: sum of top-2 sfc per group of 32.
        var group_score = List[Float32]()
        for g in range(N_GROUP):
            var gb = g * EXPERTS_PER_GROUP
            var m1 = _NEG_INF
            var m2 = _NEG_INF
            for k in range(EXPERTS_PER_GROUP):
                var v = sfc[gb + k]
                if v > m1:
                    m2 = m1
                    m1 = v
                elif v > m2:
                    m2 = v
            group_score.append(m1 + m2)

        # keep top TOPK_GROUP groups (descending value, lower-index tie break).
        var group_taken = List[Bool]()
        for _ in range(N_GROUP):
            group_taken.append(False)
        for _ in range(TOPK_GROUP):
            var best = -1
            var bv = _NEG_INF
            for g in range(N_GROUP):
                if group_taken[g]:
                    continue
                if best == -1 or group_score[g] > bv:
                    best = g
                    bv = group_score[g]
            group_taken[best] = True

        # masked scores_for_choice: -inf for experts in dropped groups.
        var msfc = List[Float32]()
        for ei in range(e):
            if group_taken[ei // EXPERTS_PER_GROUP]:
                msfc.append(sfc[ei])
            else:
                msfc.append(_NEG_INF)

        # select top-8 (descending value, lower-index tie break).
        var sel_taken = List[Bool]()
        for _ in range(e):
            sel_taken.append(False)
        for _ in range(TOP_K):
            var best = -1
            var bv = _NEG_INF
            for ei in range(e):
                if sel_taken[ei]:
                    continue
                if best == -1 or msfc[ei] > bv:
                    best = ei
                    bv = msfc[ei]
            sel_taken[best] = True

        # threshold = min value among the 8 selected (= the k-th largest).
        var thr = _NEG_INF
        var first = True
        for ei in range(e):
            if sel_taken[ei]:
                if first or msfc[ei] < thr:
                    thr = msfc[ei]
                    first = False

        # torch sorted=False order: strictly-greater experts (ascending idx),
        # then threshold-value experts (ascending idx).
        var order = List[Int]()
        for ei in range(e):
            if sel_taken[ei] and msfc[ei] > thr:
                order.append(ei)
        for ei in range(e):
            if sel_taken[ei] and msfc[ei] == thr:
                order.append(ei)

        # gating from the BIAS-FREE scores at the selected order.
        var gsum = Float32(0.0)
        var gvals = List[Float32]()
        for j in range(TOP_K):
            var gv = scores[order[j]]
            gvals.append(gv)
            gsum += gv
        var denom = gsum + Float32(1.0e-20)
        for j in range(TOP_K):
            expert_ids.append(order[j])
            gating.append((gvals[j] / denom) * ROUTED_SCALING)

    return RouterPlan(
        expert_ids=expert_ids^,
        gating=gating^,
        num_tokens=s,
        num_experts=e,
        top_k=TOP_K,
    )


# ── GPU router kernel: ONE thread-block per token ─────────────────────────────
# 128 threads (one per expert) cooperatively load the token's 128 logits and
# compute the bias-free sigmoid `scores` + bias-added `sfc` into shared memory;
# then thread 0 replicates the EXACT host selection (group-limited top-2-of-4
# groups, top-8 of the masked sfc, torch sorted=False order, bias-free gather,
# normalize ×2.5). Output: out_ids[S*TOP_K] i32, out_gate[S*TOP_K] f32. Because
# the selection runs per token in an independent block, the S-wide host loop
# that dominated at full res is now fully parallel across tokens.
def _lingbot_router_kernel(
    logits: LayoutTensor[DType.float32, _DYN2, MutAnyOrigin],   # [S, E]
    bias: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],     # [E]
    out_ids: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],    # [S*TOP_K]
    out_gate: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin], # [S*TOP_K]
    s: Int,
):
    var tok = Int(block_idx.x)
    if tok >= s:
        return
    var e = Int(thread_idx.x)

    var scores = stack_allocation[
        E, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var sfc = stack_allocation[
        E, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var msfc = stack_allocation[
        E, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var seltaken = stack_allocation[
        E, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var gscore = stack_allocation[
        N_GROUP, Scalar[DType.float32], address_space = AddressSpace.SHARED
    ]()
    var gtaken = stack_allocation[
        N_GROUP, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()
    var order = stack_allocation[
        TOP_K, Scalar[DType.int32], address_space = AddressSpace.SHARED
    ]()

    # cooperative sigmoid: one thread per expert (numerically-stable, matches host).
    if e < E:
        var lg = rebind[Scalar[DType.float32]](logits[tok, e])
        var sc: Float32
        if lg >= 0.0:
            sc = 1.0 / (1.0 + exp(-lg))
        else:
            var ex = exp(lg)
            sc = ex / (1.0 + ex)
        scores[e] = sc
        sfc[e] = sc + rebind[Scalar[DType.float32]](bias[e])
        seltaken[e] = 0
    barrier()

    if e != 0:
        return

    # ── group_scores: sum of top-2 sfc per group of 32 ───────────────────────
    for g in range(N_GROUP):
        var gb = g * EXPERTS_PER_GROUP
        var m1 = _NEG_INF
        var m2 = _NEG_INF
        for k in range(EXPERTS_PER_GROUP):
            var v = sfc[gb + k]
            if v > m1:
                m2 = m1
                m1 = v
            elif v > m2:
                m2 = v
        gscore[g] = m1 + m2
        gtaken[g] = 0

    # ── keep top TOPK_GROUP groups (descending, lower-index tie break) ────────
    for _ in range(TOPK_GROUP):
        var best = -1
        var bv = _NEG_INF
        for g in range(N_GROUP):
            if gtaken[g] != 0:
                continue
            if best == -1 or gscore[g] > bv:
                best = g
                bv = gscore[g]
        gtaken[best] = 1

    # ── masked scores_for_choice: -inf outside kept groups ───────────────────
    for ei in range(E):
        if gtaken[ei // EXPERTS_PER_GROUP] != 0:
            msfc[ei] = sfc[ei]
        else:
            msfc[ei] = _NEG_INF

    # ── select top-8 (descending, lower-index tie break) ─────────────────────
    for _ in range(TOP_K):
        var best = -1
        var bv = _NEG_INF
        for ei in range(E):
            if seltaken[ei] != 0:
                continue
            if best == -1 or msfc[ei] > bv:
                best = ei
                bv = msfc[ei]
        seltaken[best] = 1

    # ── threshold = min value among the 8 selected (= the k-th largest) ──────
    var thr = _NEG_INF
    var first = True
    for ei in range(E):
        if seltaken[ei] != 0:
            if first or msfc[ei] < thr:
                thr = msfc[ei]
                first = False

    # ── torch sorted=False order: strictly-greater (asc idx) then == thr ─────
    var oc = 0
    for ei in range(E):
        if seltaken[ei] != 0 and msfc[ei] > thr:
            order[oc] = Int32(ei)
            oc += 1
    for ei in range(E):
        if seltaken[ei] != 0 and msfc[ei] == thr:
            order[oc] = Int32(ei)
            oc += 1

    # ── gating from the BIAS-FREE scores, normalize, ×2.5 ────────────────────
    var gsum = Float32(0.0)
    for j in range(TOP_K):
        gsum += scores[Int(order[j])]
    var denom = gsum + Float32(1.0e-20)
    var obase = tok * TOP_K
    for j in range(TOP_K):
        var oid = order[j]
        out_ids[obase + j] = oid
        out_gate[obase + j] = (scores[Int(oid)] / denom) * ROUTED_SCALING


def lingbot_grouped_sigmoid_router_gpu(
    logits: Tensor, bias: Tensor, ctx: DeviceContext
) raises -> RouterPlan:
    """GPU grouped-sigmoid token-choice router (one block per token).

    Numerically mirrors `lingbot_grouped_sigmoid_router` but performs the
    per-token sigmoid + group-limited top-8 selection on the GPU, so the
    S-wide selection loop is parallel across tokens instead of a host serial
    loop. Returns a RouterPlan with host Lists (expert_ids [S*TOP_K] i32,
    gating [S*TOP_K] f32), token-major, exactly like the host path.
    """
    var sh = logits.shape()
    if len(sh) != 2:
        raise Error("lingbot_router_gpu: logits must be 2-D [S, E]")
    var s = sh[0]
    var e = sh[1]
    if e != E:
        raise Error("lingbot_router_gpu: expected E=128 experts")
    var bsh = bias.shape()
    if len(bsh) != 1 or bsh[0] != e:
        raise Error("lingbot_router_gpu: bias must be [E]")
    if logits.dtype() != STDtype.F32 or bias.dtype() != STDtype.F32:
        raise Error("lingbot_router_gpu: logits and bias must be F32")

    var n_slots = s * TOP_K
    var ids_buf = ctx.enqueue_create_buffer[DType.uint8](n_slots * 4)
    var gate_buf = ctx.enqueue_create_buffer[DType.uint8](n_slots * 4)

    var lg_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](s, e))
    var b_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](e))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_slots))
    var lg_lt = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        logits.buf.unsafe_ptr().bitcast[Float32](), lg_rl
    )
    var b_lt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        bias.buf.unsafe_ptr().bitcast[Float32](), b_rl
    )
    var ids_lt = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        ids_buf.unsafe_ptr().bitcast[Int32](), o_rl
    )
    var gate_lt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        gate_buf.unsafe_ptr().bitcast[Float32](), o_rl
    )

    ctx.enqueue_function[_lingbot_router_kernel, _lingbot_router_kernel](
        lg_lt, b_lt, ids_lt, gate_lt, s,
        grid_dim=s, block_dim=E,
    )
    ctx.synchronize()

    # copy ids + gating back to host Lists (grouped_expert_ffn + combine consume
    # host Lists; the selection itself is now off the host critical path).
    var ids_host = ctx.enqueue_create_host_buffer[DType.uint8](n_slots * 4)
    var gate_host = ctx.enqueue_create_host_buffer[DType.uint8](n_slots * 4)
    ctx.enqueue_copy(dst_buf=ids_host, src_buf=ids_buf)
    ctx.enqueue_copy(dst_buf=gate_host, src_buf=gate_buf)
    ctx.synchronize()

    var idp = ids_host.unsafe_ptr().bitcast[Int32]()
    var gp = gate_host.unsafe_ptr().bitcast[Float32]()
    var expert_ids = List[Int]()
    var gating = List[Float32]()
    expert_ids.resize(n_slots, 0)
    gating.resize(n_slots, Float32(0.0))
    for i in range(n_slots):
        expert_ids[i] = Int(idp[i])
        gating[i] = gp[i]

    return RouterPlan(
        expert_ids=expert_ids^,
        gating=gating^,
        num_tokens=s,
        num_experts=e,
        top_k=TOP_K,
    )


# ── deterministic GPU top-8 combine (replaces host download + mul-add) ────────
# One thread per (token, channel). For channel c of token t, F32-accumulate the
# 8 routed slots in ASCENDING j order, then round to bf16. Bit-identical to the
# host `_restore_tokens` combine (F32 scratch -> bf16, fixed order) — NO atomics,
# so it stays run-to-run deterministic — but keeps the [S*TOP_K, H] expert output
# ON DEVICE (no 132MB DtoH download, no 33M host mul-adds).
def _det_combine_kernel(
    eo: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],   # [S*TOP_K, H]
    gate: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],  # [S*TOP_K]
    outp: LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin],  # [S, H]
    s: Int,
    h: Int,
    topk: Int,
):
    var idx = Int(global_idx.x)
    var total = s * h
    if idx < total:
        var t = idx // h
        var c = idx % h
        var acc = Float32(0.0)
        for j in range(topk):
            var slot = t * topk + j
            var g = rebind[Scalar[DType.float32]](gate[slot])
            var v = rebind[Scalar[DType.bfloat16]](eo[slot, c]).cast[
                DType.float32
            ]()
            acc += v * g
        outp[t, c] = rebind[outp.element_type](acc.cast[DType.bfloat16]())


def _deterministic_gated_combine(
    expert_out: Tensor,        # [S*TOP_K, H] bf16
    gating: List[Float32],     # host, length S*TOP_K
    s: Int,
    h: Int,
    topk: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """[S,H] bf16 = deterministic F32-accumulated weighted top-k sum, on GPU."""
    var n_slots = s * topk
    # upload gating (tiny: S*TOP_K f32).
    var g_dev = ctx.enqueue_create_buffer[DType.uint8](n_slots * 4)
    var g_host = ctx.enqueue_create_host_buffer[DType.uint8](n_slots * 4)
    var gp = g_host.unsafe_ptr().bitcast[Float32]()
    for i in range(n_slots):
        gp[i] = gating[i]
    ctx.enqueue_copy(dst_buf=g_dev, src_buf=g_host)

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](s * h * 2)  # bf16
    var eo_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](n_slots, h))
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n_slots))
    var out_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](s, h))
    var eo_lt = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        expert_out.buf.unsafe_ptr().bitcast[BFloat16](), eo_rl
    )
    var g_lt = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        g_dev.unsafe_ptr().bitcast[Float32](), g_rl
    )
    var out_lt = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
        out_buf.unsafe_ptr().bitcast[BFloat16](), out_rl
    )
    var total = s * h
    var grid = (total + 255) // 256
    ctx.enqueue_function[_det_combine_kernel, _det_combine_kernel](
        eo_lt, g_lt, out_lt, s, h, topk,
        grid_dim=grid, block_dim=256,
    )
    ctx.synchronize()
    return Tensor(out_buf^, [s, h], STDtype.BF16)


def lingbot_video_moe(
    ffn_in_bf16: Tensor,     # [S, H] bf16 tokens (view(-1,H) of hidden_states)
    ffn_in_f32: Tensor,      # [S, H] f32 (tokens.float() for the router GEMM)
    router_weight: Tensor,   # [E, H] f32
    bias: Tensor,            # [E]    f32
    w1: Tensor,              # [E, I, H] bf16   (gate)
    w3: Tensor,              # [E, I, H] bf16   (up)
    w2: Tensor,              # [E, H, I] bf16   (down)
    shared_gate: Tensor,     # [I, H] bf16
    shared_up: Tensor,       # [I, H] bf16
    shared_down: Tensor,     # [H, I] bf16
    ctx: DeviceContext,
) raises -> Tensor:
    """Full LingBotVideoSparseMoeBlock forward. Returns [S, H] bf16."""
    var sh = ffn_in_bf16.shape()
    var s = sh[0]
    var hdim = sh[1]

    # ── router (F32, GPU: one block per token) ───────────────────────────────
    var logits = linear(ffn_in_f32, router_weight, None, ctx)  # [S, E] f32
    var plan = lingbot_grouped_sigmoid_router_gpu(logits, bias, ctx)

    # ── routed experts: per-expert SwiGLU, weighted top-8 combine ────────────
    var expert_out = grouped_expert_ffn(
        ffn_in_bf16, w1, w3, w2, plan, ctx
    )  # [S*TOP_K, H] bf16
    # DETERMINISTIC GPU combine (replaces the host DtoH download + S*TOP_K*H
    # host mul-add loop). Mirrors the reference `_restore_tokens`: sum the top_k
    # slots per token in ASCENDING j order, F32-accumulating, round to bf16 — the
    # exact numeric boundary (F32 scratch -> bf16) in a FIXED order (no atomics),
    # so the forward is bit-reproducible. Keeps expert_out on device.
    var topk = plan.top_k
    var accum = _deterministic_gated_combine(
        expert_out, plan.gating, s, hdim, topk, ctx
    )  # [S,H] bf16

    # ── shared expert: down(silu(t@gateᵀ)·(t@upᵀ)) ───────────────────────────
    var sg = linear(ffn_in_bf16, shared_gate, None, ctx)  # [S, I]
    var su = linear(ffn_in_bf16, shared_up, None, ctx)    # [S, I]
    var sh_mid = swiglu(sg, su, ctx)                       # [S, I]
    var shared_out = linear(sh_mid, shared_down, None, ctx)  # [S, H]

    # out = routed + shared
    return add(accum, shared_out, ctx)
