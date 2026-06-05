# ops/celoss_embed_backward.mojo — BACKWARD for the loss/embedding arms:
#   CrossEntropy, NLLLoss, BCELoss, Embedding.
#
# Phase T1 of FULL_PORT_TRAINING_PLAN.md (loss + token-conditioning backward).
# CrossEntropy backward feeds SenseNova's +0.1*CE aux loss; Embedding backward
# feeds token conditioning.
#
# Each fn returns a fresh Tensor in the input storage dtype. All interior math is
# F32. Conventions mirror ops/reduce_backward.mojo + ops/shape_backward.mojo:
#   * device buffers are DType.uint8 (Tensor's storage),
#   * kernels cast scalar elements to F32 for math and cast back on store,
#   * loud-fail on shape/dtype mismatch.
#
# Integer indices are passed host-side as List[Int] (the caller's natural form),
# staged into a device int32 buffer for the kernel — exactly the pattern
# shape_backward.index_select_backward uses.
#
# ── Math (verified against flame-core autograd.rs @ HEAD) ─────────────────────
#   CrossEntropy (reduction="mean", autograd.rs cross_entropy):
#     forward  loss = mean_r( -log_softmax(logits)[r, target[r]] )
#     backward d_logits[r,c] = (softmax(logits)[r,c] - onehot(target[r])[c]) / N
#   NLLLoss (reduction="mean", autograd.rs:6327 "-1/batch_size at target idx"):
#     forward  loss = mean_r( -log_probs[r, target[r]] )   (input is log-probs)
#     backward d_log_probs[r,c] = -onehot(target[r])[c] / N
#   BCELoss (PLAIN probability form, autograd.rs:6302):
#     forward  loss = mean_i( -(t*log(p) + (1-t)*log(1-p)) )   p in (0,1)
#     backward d_p[i] = (p - t) / (p*(1-p)) / N
#     >>> GATED VARIANT = PLAIN prob form (torch F.binary_cross_entropy), NOT
#         the with-logits form. <<<
#   Embedding (autograd.rs:5842 scatter-add into weight matrix):
#     forward  out[i,:] = table[indices[i], :]
#     backward d_table[indices[i], :] += grad_out[i, :]  (repeats accumulate)
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import exp
from std.gpu.host import DeviceContext
from std.gpu import global_idx, thread_idx, block_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.utils.index import IndexList
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime _TPB = 256


def _require_same_dtype(a: Tensor, b: Tensor, who: String) raises:
    if a.dtype() != b.dtype():
        raise Error(who + ": input dtype mismatch")


# ══════════════════════════════════════════════════════════════════════════════
# CROSS ENTROPY — d_logits[r,c] = (softmax(logits)[r,c] - onehot)/N.
# One block per row; F32 shared-mem reductions for row-max and row-sum (stable
# softmax recompute), then write (p - onehot)/N. Matches the softmax recompute
# style of attention_backward._softmax_rows_f32.
# ══════════════════════════════════════════════════════════════════════════════
def _ce_bwd_rows_k[dtype: DType](
    logits: LayoutTensor[dtype, _DYN2, MutAnyOrigin],           # [N, C]
    targets: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],    # [N]
    o: LayoutTensor[dtype, _DYN2, MutAnyOrigin],                # [N, C] d_logits
    C: Int, inv_n: Float32,
):
    var row = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var shared = stack_allocation[
        _TPB, Scalar[DType.float32], address_space=AddressSpace.SHARED
    ]()
    # row max
    var lmax: Float32 = -3.0e38
    var c = tid
    while c < C:
        var v = rebind[Scalar[dtype]](logits[row, c]).cast[DType.float32]()
        if v > lmax:
            lmax = v
        c += _TPB
    shared[tid] = lmax
    barrier()
    var active = _TPB // 2
    while active > 0:
        if tid < active:
            var a = shared[tid]
            var bb = shared[tid + active]
            shared[tid] = a if a > bb else bb
        barrier()
        active //= 2
    var rmax = shared[0]
    barrier()
    # row sum of exp(x - max)
    var lsum: Float32 = 0.0
    c = tid
    while c < C:
        lsum += exp(rebind[Scalar[dtype]](logits[row, c]).cast[DType.float32]() - rmax)
        c += _TPB
    shared[tid] = lsum
    barrier()
    active = _TPB // 2
    while active > 0:
        if tid < active:
            shared[tid] = shared[tid] + shared[tid + active]
        barrier()
        active //= 2
    var rsum = shared[0]
    barrier()
    var inv = 1.0 / rsum
    var tgt = Int(rebind[Scalar[DType.int32]](targets[row]))
    c = tid
    while c < C:
        var p = exp(rebind[Scalar[dtype]](logits[row, c]).cast[DType.float32]() - rmax) * inv
        var onehot = Float32(1.0) if c == tgt else Float32(0.0)
        o[row, c] = rebind[o.element_type](((p - onehot) * inv_n).cast[dtype]())
        c += _TPB


def cross_entropy_backward(
    logits: Tensor, target_idx: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """d_logits = (softmax(logits) - onehot(target)) / N  (mean reduction).

    logits: [N, C]; target_idx: N class indices. Returns [N, C] logits dtype.
    Matches torch.nn.functional.cross_entropy(reduction="mean")."""
    var sh = logits.shape()
    if len(sh) != 2:
        raise Error("cross_entropy_backward: expected [N, C]")
    var N = sh[0]
    var C = sh[1]
    if len(target_idx) != N:
        raise Error("cross_entropy_backward: len(target_idx) != N")

    # stage targets host -> device int32
    var t_host = ctx.enqueue_create_host_buffer[DType.int32](N)
    var tp = t_host.unsafe_ptr()
    for i in range(N):
        var t = target_idx[i]
        if t < 0 or t >= C:
            raise Error(String("cross_entropy_backward: target ") + String(t)
                        + " out of range [0," + String(C) + ")")
        tp[i] = Int32(t)
    var t_dev = ctx.enqueue_create_buffer[DType.int32](N)
    ctx.enqueue_copy(dst_buf=t_dev, src_buf=t_host)

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](N * C * logits.dtype().byte_size())
    var lg_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](N, C))
    var t_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](N))
    var TG = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        t_dev.unsafe_ptr(), t_rl)
    var dt = logits.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var LG = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            logits.buf.unsafe_ptr().bitcast[Float32](), lg_rl)
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), lg_rl)
        ctx.enqueue_function[
            _ce_bwd_rows_k[DType.float32], _ce_bwd_rows_k[DType.float32]
        ](LG, TG, O, C, Float32(1.0) / Float32(N), grid_dim=N, block_dim=_TPB)
    elif dt == DType.bfloat16:
        var LG = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            logits.buf.unsafe_ptr().bitcast[BFloat16](), lg_rl)
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), lg_rl)
        ctx.enqueue_function[
            _ce_bwd_rows_k[DType.bfloat16], _ce_bwd_rows_k[DType.bfloat16]
        ](LG, TG, O, C, Float32(1.0) / Float32(N), grid_dim=N, block_dim=_TPB)
    else:
        var LG = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            logits.buf.unsafe_ptr().bitcast[Float16](), lg_rl)
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), lg_rl)
        ctx.enqueue_function[
            _ce_bwd_rows_k[DType.float16], _ce_bwd_rows_k[DType.float16]
        ](LG, TG, O, C, Float32(1.0) / Float32(N), grid_dim=N, block_dim=_TPB)
    ctx.synchronize()
    var os = [N, C]
    return Tensor(out_buf^, os^, logits.dtype())


# ══════════════════════════════════════════════════════════════════════════════
# NLL — d_log_probs = -onehot(target)/N. Zero everywhere except the target column
# of each row. One thread per OUTPUT element; the target column gets -1/N.
# ══════════════════════════════════════════════════════════════════════════════
def _nll_bwd_k[dtype: DType](
    targets: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],   # [N]
    o: LayoutTensor[dtype, _DYN2, MutAnyOrigin],               # [N, C]
    C: Int, neg_inv_n: Float32,
):
    var t = Int(global_idx.x)
    var row = t // C
    var col = t % C
    var N = o.dim[0]()
    if row < N:
        var tgt = Int(rebind[Scalar[DType.int32]](targets[row]))
        var v = neg_inv_n if col == tgt else Float32(0.0)
        o[row, col] = rebind[o.element_type](v.cast[dtype]())


def nll_backward(
    log_probs: Tensor, target_idx: List[Int], ctx: DeviceContext
) raises -> Tensor:
    """d_log_probs = -onehot(target) / N  (mean reduction).

    log_probs: [N, C] (already log-probabilities); target_idx: N indices.
    Returns [N, C] log_probs dtype.
    Matches torch.nn.functional.nll_loss(reduction="mean")."""
    var sh = log_probs.shape()
    if len(sh) != 2:
        raise Error("nll_backward: expected [N, C]")
    var N = sh[0]
    var C = sh[1]
    if len(target_idx) != N:
        raise Error("nll_backward: len(target_idx) != N")

    var t_host = ctx.enqueue_create_host_buffer[DType.int32](N)
    var tp = t_host.unsafe_ptr()
    for i in range(N):
        var t = target_idx[i]
        if t < 0 or t >= C:
            raise Error(String("nll_backward: target ") + String(t)
                        + " out of range [0," + String(C) + ")")
        tp[i] = Int32(t)
    var t_dev = ctx.enqueue_create_buffer[DType.int32](N)
    ctx.enqueue_copy(dst_buf=t_dev, src_buf=t_host)

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        N * C * log_probs.dtype().byte_size()
    )
    var o_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](N, C))
    var t_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](N))
    var TG = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        t_dev.unsafe_ptr(), t_rl)
    var n = N * C
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = log_probs.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var O = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl)
        ctx.enqueue_function[
            _nll_bwd_k[DType.float32], _nll_bwd_k[DType.float32]
        ](TG, O, C, Float32(-1.0) / Float32(N), grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var O = LayoutTensor[DType.bfloat16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl)
        ctx.enqueue_function[
            _nll_bwd_k[DType.bfloat16], _nll_bwd_k[DType.bfloat16]
        ](TG, O, C, Float32(-1.0) / Float32(N), grid_dim=grid, block_dim=_BLOCK)
    else:
        var O = LayoutTensor[DType.float16, _DYN2, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl)
        ctx.enqueue_function[
            _nll_bwd_k[DType.float16], _nll_bwd_k[DType.float16]
        ](TG, O, C, Float32(-1.0) / Float32(N), grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    var os = [N, C]
    return Tensor(out_buf^, os^, log_probs.dtype())


# ══════════════════════════════════════════════════════════════════════════════
# BCE (PLAIN probability form) — d_p[i] = (p - t) / (p*(1-p)) / N.
# One thread per element. Matches torch F.binary_cross_entropy (NOT with-logits).
# ══════════════════════════════════════════════════════════════════════════════
def _bce_bwd_k[dtype: DType](
    pred: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    target: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],
    n: Int, inv_n: Float32,
):
    var i = Int(global_idx.x)
    if i < n:
        var p = rebind[Scalar[dtype]](pred[i]).cast[DType.float32]()
        var t = rebind[Scalar[dtype]](target[i]).cast[DType.float32]()
        o[i] = rebind[o.element_type](
            (((p - t) / (p * (Float32(1.0) - p))) * inv_n).cast[dtype]())


def bce_backward(
    pred: Tensor, target: Tensor, ctx: DeviceContext
) raises -> Tensor:
    """d_pred = (pred - target) / (pred*(1-pred)) / N  (mean reduction).

    PLAIN probability form: pred in (0,1). Matches
    torch.nn.functional.binary_cross_entropy(reduction="mean"). pred/target are
    equal-shape tensors with the same dtype. Returns d_pred in pred dtype."""
    _require_same_dtype(pred, target, "bce_backward")
    var n = pred.numel()
    if target.numel() != n:
        raise Error("bce_backward: pred/target numel mismatch")
    var out_buf = ctx.enqueue_create_buffer[DType.uint8](pred.nbytes())
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    var grid = (n + _BLOCK - 1) // _BLOCK
    var dt = pred.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var P = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            pred.buf.unsafe_ptr().bitcast[Float32](), rl)
        var T = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            target.buf.unsafe_ptr().bitcast[Float32](), rl)
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), rl)
        ctx.enqueue_function[
            _bce_bwd_k[DType.float32], _bce_bwd_k[DType.float32]
        ](P, T, O, n, Float32(1.0) / Float32(n), grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var P = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            pred.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var T = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            target.buf.unsafe_ptr().bitcast[BFloat16](), rl)
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), rl)
        ctx.enqueue_function[
            _bce_bwd_k[DType.bfloat16], _bce_bwd_k[DType.bfloat16]
        ](P, T, O, n, Float32(1.0) / Float32(n), grid_dim=grid, block_dim=_BLOCK)
    else:
        var P = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            pred.buf.unsafe_ptr().bitcast[Float16](), rl)
        var T = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            target.buf.unsafe_ptr().bitcast[Float16](), rl)
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), rl)
        ctx.enqueue_function[
            _bce_bwd_k[DType.float16], _bce_bwd_k[DType.float16]
        ](P, T, O, n, Float32(1.0) / Float32(n), grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    return Tensor(out_buf^, pred.shape(), pred.dtype())


# ══════════════════════════════════════════════════════════════════════════════
# EMBEDDING — scatter-ADD grad_out rows into a zero [num_embeddings, dim] table
# at the given indices (repeated indices accumulate). Same scatter-add as
# shape_backward.index_select_backward: one thread per d_table element (row v,
# col d), summing grad_out[i, d] over all i with idx[i] == v.
# ══════════════════════════════════════════════════════════════════════════════
def _embedding_bwd_k[dtype: DType](
    g: LayoutTensor[dtype, _DYN1, MutAnyOrigin],             # grad_out [Ni, D]
    idx: LayoutTensor[DType.int32, _DYN1, MutAnyOrigin],     # indices [Ni]
    o: LayoutTensor[dtype, _DYN1, MutAnyOrigin],             # d_table [V, D]
    Ni: Int, D: Int, V: Int,
):
    var t = Int(global_idx.x)
    if t < V * D:
        var v = t // D
        var d = t % D
        var acc: Float32 = 0.0
        for i in range(Ni):
            if Int(rebind[Scalar[DType.int32]](idx[i])) == v:
                acc += rebind[Scalar[dtype]](g[i * D + d]).cast[DType.float32]()
        o[t] = rebind[o.element_type](acc.cast[dtype]())


def embedding_backward(
    grad_out: Tensor, indices: List[Int], num_embeddings: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Scatter-ADD grad_out rows into zeros([num_embeddings, dim]) at `indices`.

    grad_out: [Ni, D] (Ni == len(indices)); repeated indices accumulate.
    Returns d_table [num_embeddings, D] in grad_out dtype.
    Matches nn.Embedding weight grad."""
    var gsh = grad_out.shape()
    if len(gsh) != 2:
        raise Error("embedding_backward: grad_out must be [Ni, D]")
    var Ni = gsh[0]
    var D = gsh[1]
    if len(indices) != Ni:
        raise Error("embedding_backward: len(indices) != grad_out.shape[0]")

    var id_host = ctx.enqueue_create_host_buffer[DType.int32](Ni)
    var ip = id_host.unsafe_ptr()
    for i in range(Ni):
        var r = indices[i]
        if r < 0 or r >= num_embeddings:
            raise Error(String("embedding_backward: index ") + String(r)
                        + " out of range [0," + String(num_embeddings) + ")")
        ip[i] = Int32(r)
    var id_dev = ctx.enqueue_create_buffer[DType.int32](Ni)
    ctx.enqueue_copy(dst_buf=id_dev, src_buf=id_host)

    var out_buf = ctx.enqueue_create_buffer[DType.uint8](
        num_embeddings * D * grad_out.dtype().byte_size()
    )
    var g_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](Ni * D))
    var id_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](Ni))
    var o_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](num_embeddings * D))
    var IDS = LayoutTensor[DType.int32, _DYN1, MutAnyOrigin](
        id_dev.unsafe_ptr(), id_rl)
    var grid = (num_embeddings * D + _BLOCK - 1) // _BLOCK
    var dt = grad_out.dtype().to_mojo_dtype()
    if dt == DType.float32:
        var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float32](), g_rl)
        var O = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float32](), o_rl)
        ctx.enqueue_function[
            _embedding_bwd_k[DType.float32], _embedding_bwd_k[DType.float32]
        ](G, IDS, O, Ni, D, num_embeddings, grid_dim=grid, block_dim=_BLOCK)
    elif dt == DType.bfloat16:
        var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[BFloat16](), g_rl)
        var O = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[BFloat16](), o_rl)
        ctx.enqueue_function[
            _embedding_bwd_k[DType.bfloat16], _embedding_bwd_k[DType.bfloat16]
        ](G, IDS, O, Ni, D, num_embeddings, grid_dim=grid, block_dim=_BLOCK)
    else:
        var G = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            grad_out.buf.unsafe_ptr().bitcast[Float16](), g_rl)
        var O = LayoutTensor[DType.float16, _DYN1, MutAnyOrigin](
            out_buf.unsafe_ptr().bitcast[Float16](), o_rl)
        ctx.enqueue_function[
            _embedding_bwd_k[DType.float16], _embedding_bwd_k[DType.float16]
        ](G, IDS, O, Ni, D, num_embeddings, grid_dim=grid, block_dim=_BLOCK)
    ctx.synchronize()
    var os = [num_embeddings, D]
    return Tensor(out_buf^, os^, grad_out.dtype())
