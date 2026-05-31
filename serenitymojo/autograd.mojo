# autograd.mojo - reverse-mode autograd TAPE engine (Phase T1 of
# FULL_PORT_TRAINING_PLAN.md). Port of flame-core's tape (src/autograd.rs):
# TensorId-keyed entries + reverse compute_gradients -> id->grad map.
#
# Architecture decisions (USER, 2026-05-30):
#  * EXPLICIT threaded Tape struct (no global thread-local; serenitymojo has no
#    global mutable state, Mojo 1.0.0b1 globals are shaky). Tape passed `mut`.
#  * Tensor carries an `id` (tensor.mojo). 0 = untracked; op outputs/track stamp.
#
# Move-only Tensor can't be a List/Dict element (Mojo collections need Copyable).
# We box with ArcPointer[Tensor] (Copyable = refcount bump) - the SAME idiom
# block_loader/wan22_decoder/sensenova already use for Dict[..., Tensor]. No
# manual alloc/free; the Arc refcount frees the saved/grad tensors.
#
# T1 proves the ENGINE on Add/Sub/Mul. The 66-arm set (plan s3) is added
# incrementally, each parity-gated. F32 only for the spike. Mojo 1.0.0b1, NVIDIA.
#
# NOTE (micro-opt for later): every op currently saves BOTH operands (uniform).
# Add/Sub don't need saved tensors for backward; a per-op "needs_saved" gate
# will drop those clones. Inert for correctness; trivial for the T1 shapes.

from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.utils.index import IndexList
from std.memory import ArcPointer
from std.collections import Dict
from std.collections.optional import Optional
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from linalg.matmul.vendor.blas import matmul
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import mm_backward, linear_backward
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import silu, swiglu
from serenitymojo.ops.reduce import reduce_sum
from serenitymojo.ops.norm_backward import rms_norm_backward, RmsNormBackward
from serenitymojo.ops.activation_backward import silu_backward
from serenitymojo.ops.loss_swiglu_backward import (
    swiglu_backward,
    SwigluGrads,
    mse_backward,
)


comptime _DYN1 = Layout.row_major(-1)
comptime _DYN2 = Layout.row_major(-1, -1)
comptime _BLOCK = 256
comptime TArc = ArcPointer[Tensor]

comptime OP_ADD = 0
comptime OP_SUB = 1
comptime OP_MUL = 2
comptime OP_MATMUL = 3
comptime OP_LINEAR = 4
comptime OP_RMSNORM = 5
comptime OP_SILU = 6
comptime OP_SWIGLU = 7
comptime OP_MSE = 8

# RMSNorm eps. The TapeEntry struct fields are LEAD-OWNED and frozen for this
# task (no new Float field to carry eps), so the record method and the backward
# arm share ONE comptime eps. Callers/gates use the same eps. AGENT-DEFAULT —
# matches the diffusers/DiT RMSNorm default (1e-6).
comptime _RMS_EPS = Float32(1e-6)


# tiny F32 elementwise kernels (self-contained for the T1 spike)
def _k_add(
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        o[i] = rebind[o.element_type](
            rebind[Scalar[DType.float32]](a[i]) + rebind[Scalar[DType.float32]](b[i])
        )


def _k_sub(
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        o[i] = rebind[o.element_type](
            rebind[Scalar[DType.float32]](a[i]) - rebind[Scalar[DType.float32]](b[i])
        )


def _k_mul(
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    b: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        o[i] = rebind[o.element_type](
            rebind[Scalar[DType.float32]](a[i]) * rebind[Scalar[DType.float32]](b[i])
        )


def _k_neg(
    a: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        o[i] = rebind[o.element_type](-rebind[Scalar[DType.float32]](a[i]))


def _k_fill(
    o: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    val: Float32,
    n: Int,
):
    var i = Int(global_idx.x)
    if i < n:
        o[i] = rebind[o.element_type](val)


# F32 tensor helpers (untracked outputs)
def _empty_f32(var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    var n = 1
    for i in range(len(shape)):
        n *= shape[i]
    var buf = ctx.enqueue_create_buffer[DType.uint8](n * 4)
    return Tensor(buf^, shape^, STDtype.F32, 0)


def _lt(t: Tensor, n: Int) -> LayoutTensor[DType.float32, _DYN1, MutAnyOrigin]:
    var rl = RuntimeLayout[_DYN1].row_major(IndexList[1](n))
    return LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        t.buf.unsafe_ptr().bitcast[Float32](), rl
    )


def _raw_add(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    var n = a.numel()
    var o = _empty_f32(a.shape(), ctx)
    var g = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_k_add, _k_add](
        _lt(a, n), _lt(b, n), _lt(o, n), n, grid_dim=g, block_dim=_BLOCK)
    return o^


def _raw_sub(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    var n = a.numel()
    var o = _empty_f32(a.shape(), ctx)
    var g = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_k_sub, _k_sub](
        _lt(a, n), _lt(b, n), _lt(o, n), n, grid_dim=g, block_dim=_BLOCK)
    return o^


def _raw_mul(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    var n = a.numel()
    var o = _empty_f32(a.shape(), ctx)
    var g = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_k_mul, _k_mul](
        _lt(a, n), _lt(b, n), _lt(o, n), n, grid_dim=g, block_dim=_BLOCK)
    return o^


def _raw_neg(a: Tensor, ctx: DeviceContext) raises -> Tensor:
    var n = a.numel()
    var o = _empty_f32(a.shape(), ctx)
    var g = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_k_neg, _k_neg](
        _lt(a, n), _lt(o, n), n, grid_dim=g, block_dim=_BLOCK)
    return o^


# F32 GEMM via the vendor blas (transpose_a/transpose_b both supported — proven
# in ops/attention_backward.mojo). C[rc,cc] in F32. Inputs must be F32.
def _raw_gemm(
    a: Tensor, b: Tensor, rc: Int, cc: Int, kk: Int,
    ta: Bool, tb: Bool, ctx: DeviceContext,
) raises -> Tensor:
    var out_shape = List[Int]()
    out_shape.append(rc)
    out_shape.append(cc)
    var o = _empty_f32(out_shape^, ctx)
    # a is [rc,kk] (or [kk,rc] if ta); b is [kk,cc] (or [cc,kk] if tb).
    var a_rows = kk if ta else rc
    var a_cols = rc if ta else kk
    var b_rows = cc if tb else kk
    var b_cols = kk if tb else cc
    var a_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](a_rows, a_cols))
    var b_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](b_rows, b_cols))
    var c_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](rc, cc))
    var A = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        a.buf.unsafe_ptr().bitcast[Float32](), a_rl)
    var B = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        b.buf.unsafe_ptr().bitcast[Float32](), b_rl)
    var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        o.buf.unsafe_ptr().bitcast[Float32](), c_rl)
    matmul(ctx, C, A, B, transpose_a=ta, transpose_b=tb, c_row_major=True)
    ctx.synchronize()
    return o^


def ones_like(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    var n = t.numel()
    var o = _empty_f32(t.shape(), ctx)
    var g = (n + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_k_fill, _k_fill](
        _lt(o, n), Float32(1.0), n, grid_dim=g, block_dim=_BLOCK)
    return o^


# F32 2D matmul forward C[M,N] = A[M,K] @ B[K,N] via vendor blas (no transpose).
def _raw_matmul(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
    var ash = a.shape()
    var bsh = b.shape()
    var M = ash[0]
    var K = ash[1]
    var N = bsh[1]
    var out_shape = List[Int]()
    out_shape.append(M)
    out_shape.append(N)
    var o = _empty_f32(out_shape^, ctx)
    var a_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, K))
    var b_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](K, N))
    var c_rl = RuntimeLayout[_DYN2].row_major(IndexList[2](M, N))
    var A = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        a.buf.unsafe_ptr().bitcast[Float32](), a_rl)
    var B = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        b.buf.unsafe_ptr().bitcast[Float32](), b_rl)
    var C = LayoutTensor[DType.float32, _DYN2, MutAnyOrigin](
        o.buf.unsafe_ptr().bitcast[Float32](), c_rl)
    matmul(ctx, C, A, B, transpose_a=False, transpose_b=False, c_row_major=True)
    ctx.synchronize()
    return o^


struct TapeEntry(Copyable, Movable):
    var out_id: Int
    var op_kind: Int
    var lhs_id: Int
    var rhs_id: Int
    var saved0: TArc   # lhs clone (used by Mul/MatMul/Linear backward)
    var saved1: TArc   # rhs clone
    var dim_m: Int     # MatMul/Linear: C[M,N]=A[M,K]@B[K,N] ; Linear M/in/out
    var dim_n: Int
    var dim_k: Int
    # Optional 3rd input slot — for ops with >2 trainable inputs (Linear: x,W,b).
    # Defaults keep every existing 2-input arm byte-identical.
    var third_id: Int
    var saved2: Optional[TArc]

    def __init__(
        out self, out_id: Int, op_kind: Int, lhs_id: Int, rhs_id: Int,
        var saved0: TArc, var saved1: TArc,
        dim_m: Int = 0, dim_n: Int = 0, dim_k: Int = 0,
        third_id: Int = 0, var saved2: Optional[TArc] = None,
    ):
        self.out_id = out_id
        self.op_kind = op_kind
        self.lhs_id = lhs_id
        self.rhs_id = rhs_id
        self.saved0 = saved0^
        self.saved1 = saved1^
        self.dim_m = dim_m
        self.dim_n = dim_n
        self.dim_k = dim_k
        self.third_id = third_id
        self.saved2 = saved2^


struct Tape(Movable):
    var next_id: Int
    var entries: List[TapeEntry]

    def __init__(out self):
        self.next_id = 1
        self.entries = List[TapeEntry]()

    def _fresh(mut self) -> Int:
        var id = self.next_id
        self.next_id += 1
        return id

    def track(mut self, mut t: Tensor):
        if t.id == 0:
            t.set_id(self._fresh())

    # training-mode ops as methods (mut self) — the proven mutation idiom in
    # this codebase (cf. ResidencyManager.transition). Each records an entry.
    def record_add(mut self, a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
        var out = _raw_add(a, b, ctx)
        var oid = self._fresh()
        out.set_id(oid)
        self.entries.append(
            TapeEntry(oid, OP_ADD, a.id, b.id,
                      TArc(a.clone(ctx)), TArc(b.clone(ctx))))
        return out^

    def record_sub(mut self, a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
        var out = _raw_sub(a, b, ctx)
        var oid = self._fresh()
        out.set_id(oid)
        self.entries.append(
            TapeEntry(oid, OP_SUB, a.id, b.id,
                      TArc(a.clone(ctx)), TArc(b.clone(ctx))))
        return out^

    def record_mul(mut self, a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
        var out = _raw_mul(a, b, ctx)
        var oid = self._fresh()
        out.set_id(oid)
        # saved0 = lhs (grad_rhs = grad*lhs); saved1 = rhs (grad_lhs = grad*rhs)
        self.entries.append(
            TapeEntry(oid, OP_MUL, a.id, b.id,
                      TArc(a.clone(ctx)), TArc(b.clone(ctx))))
        return out^

    def record_matmul(mut self, a: Tensor, b: Tensor, ctx: DeviceContext) raises -> Tensor:
        # C[M,N] = A[M,K] @ B[K,N]. saved0=a, saved1=b; dims carried for backward.
        var out = _raw_matmul(a, b, ctx)
        var oid = self._fresh()
        out.set_id(oid)
        var ash = a.shape()
        var bsh = b.shape()
        self.entries.append(
            TapeEntry(oid, OP_MATMUL, a.id, b.id,
                      TArc(a.clone(ctx)), TArc(b.clone(ctx)),
                      ash[0], bsh[1], ash[1]))   # dim_m=M, dim_n=N, dim_k=K
        return out^

    def record_linear(
        mut self, x: Tensor, w: Tensor, b: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        # y = x @ Wᵀ + b. x:[M,in], W:[out,in], b:[out]. 3 trainable inputs →
        # lhs=x, rhs=W, third=b; saved0=x, saved1=W, saved2=b (b unused by bwd
        # but carried for symmetry). dims: dim_m=M, dim_n=out, dim_k=in.
        var out = linear(x, w, Optional[Tensor](b.clone(ctx)), ctx)
        var oid = self._fresh()
        out.set_id(oid)
        var xsh = x.shape()
        var wsh = w.shape()
        var M = xsh[0]
        var in_f = xsh[1]
        var out_f = wsh[0]
        self.entries.append(
            TapeEntry(oid, OP_LINEAR, x.id, w.id,
                      TArc(x.clone(ctx)), TArc(w.clone(ctx)),
                      dim_m=M, dim_n=out_f, dim_k=in_f,
                      third_id=b.id, saved2=Optional[TArc](TArc(b.clone(ctx)))))
        return out^

    def record_rms_norm(
        mut self, x: Tensor, weight: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        # y = rms_norm(x, weight) over the last dim. 2 trainable inputs →
        # lhs=x, rhs=weight; saved0=x, saved1=weight. eps is the shared comptime
        # _RMS_EPS (TapeEntry has no Float slot to carry it). Backward via
        # rms_norm_backward → d_x to lhs_id, d_gamma to rhs_id.
        var out = rms_norm(x, weight, _RMS_EPS, ctx)
        var oid = self._fresh()
        out.set_id(oid)
        self.entries.append(
            TapeEntry(oid, OP_RMSNORM, x.id, weight.id,
                      TArc(x.clone(ctx)), TArc(weight.clone(ctx))))
        return out^

    def record_silu(mut self, x: Tensor, ctx: DeviceContext) raises -> Tensor:
        # y = silu(x). 1 trainable input → lhs=x; saved0=x. rhs unused (id 0;
        # saved1 carries a clone of x for struct symmetry, never read). Backward
        # via silu_backward(grad_out, x) → d_x to lhs_id.
        var out = silu(x, ctx)
        var oid = self._fresh()
        out.set_id(oid)
        self.entries.append(
            TapeEntry(oid, OP_SILU, x.id, 0,
                      TArc(x.clone(ctx)), TArc(x.clone(ctx))))
        return out^

    def record_swiglu(
        mut self, gate: Tensor, up: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        # y = silu(gate) * up. 2 trainable inputs → lhs=gate, rhs=up;
        # saved0=gate, saved1=up. Backward via swiglu_backward(grad_out,gate,up)
        # → d_gate to lhs_id, d_up to rhs_id.
        var out = swiglu(gate, up, ctx)
        var oid = self._fresh()
        out.set_id(oid)
        self.entries.append(
            TapeEntry(oid, OP_SWIGLU, gate.id, up.id,
                      TArc(gate.clone(ctx)), TArc(up.clone(ctx))))
        return out^

    def mse_loss(
        mut self, pred: Tensor, target: Tensor, ctx: DeviceContext
    ) raises -> Tensor:
        # SPECIAL leaf that SEEDS the chain. loss = mean((pred-target)^2). The
        # scalar VALUE is cosmetic — backward's OP_MSE arm calls mse_backward
        # which produces the full d_pred (including the 2/N factor) and IGNORES
        # the incoming out-grad. We still emit a scalar so the chain has a head.
        # lhs=pred (trainable), rhs=target (constant — left untracked by the
        # caller so no grad flows: _accum to id 0 is a no-op). saved0=pred,
        # saved1=target. The forward scalar is reduce_sum over all dims of the
        # squared diff (value not used by backward; correctness of d_pred is
        # gated independently against the closed form).
        var diff = _raw_sub(pred, target, ctx)
        var sq = _raw_mul(diff, diff, ctx)
        var all_dims = List[Int]()
        for i in range(len(sq.shape())):
            all_dims.append(i)
        var out = reduce_sum(sq, all_dims^, False, ctx)   # scalar
        var oid = self._fresh()
        out.set_id(oid)
        self.entries.append(
            TapeEntry(oid, OP_MSE, pred.id, target.id,
                      TArc(pred.clone(ctx)), TArc(target.clone(ctx))))
        return out^


# gradient accumulation into the id->grad map (Arc-boxed)
def _accum(
    mut grads: Dict[Int, TArc], id: Int, var g: Tensor, ctx: DeviceContext,
) raises:
    if id == 0:
        return
    if grads.__contains__(id):
        var oldarc = grads[id]              # refcount-bump copy
        var summed = _raw_add(oldarc[], g, ctx)
        grads[id] = TArc(summed^)
    else:
        grads[id] = TArc(g^)


# backward driver (reverse walk; port of compute_gradients)
def backward(
    tape: Tape, loss: Tensor, ctx: DeviceContext
) raises -> Dict[Int, TArc]:
    var grads = Dict[Int, TArc]()
    grads[loss.id] = TArc(ones_like(loss, ctx))

    var i = len(tape.entries) - 1
    while i >= 0:
        var ek = tape.entries[i].op_kind
        var eout = tape.entries[i].out_id
        var elhs = tape.entries[i].lhs_id
        var erhs = tape.entries[i].rhs_id
        if grads.__contains__(eout):
            var gop = grads[eout]                 # refcount-bump copy of out-grad
            if ek == OP_ADD:
                _accum(grads, elhs, gop[].clone(ctx), ctx)
                _accum(grads, erhs, gop[].clone(ctx), ctx)
            elif ek == OP_SUB:
                _accum(grads, elhs, gop[].clone(ctx), ctx)
                _accum(grads, erhs, _raw_neg(gop[], ctx), ctx)
            elif ek == OP_MUL:
                var s0 = tape.entries[i].saved0
                var s1 = tape.entries[i].saved1
                _accum(grads, elhs, _raw_mul(gop[], s1[], ctx), ctx)
                _accum(grads, erhs, _raw_mul(gop[], s0[], ctx), ctx)
            elif ek == OP_MATMUL:
                # d_a = grad_c @ Bᵀ ; d_b = Aᵀ @ grad_c  (via verified mm_backward)
                var sa = tape.entries[i].saved0
                var sb = tape.entries[i].saved1
                var mg = mm_backward(
                    gop[], sa[], sb[],
                    tape.entries[i].dim_m, tape.entries[i].dim_n,
                    tape.entries[i].dim_k, ctx)
                # CLONE the struct fields (borrow → fresh Tensor) rather than
                # move them out: Mojo forbids partially moving a field out of a
                # still-live value that has a destructor. mg destructs at scope end.
                _accum(grads, elhs, mg.d_a.clone(ctx), ctx)
                _accum(grads, erhs, mg.d_b.clone(ctx), ctx)
            elif ek == OP_LINEAR:
                # y = x@Wᵀ+b (lhs=x, rhs=W, third=b). d_x,d_W,d_b via verified
                # linear_backward(grad_y, x, W, M, in_features, out_features).
                var lx = tape.entries[i].saved0
                var lw = tape.entries[i].saved1
                var lg = linear_backward(
                    gop[], lx[], lw[],
                    tape.entries[i].dim_m, tape.entries[i].dim_k,
                    tape.entries[i].dim_n, ctx)
                _accum(grads, elhs, lg.d_x.clone(ctx), ctx)
                _accum(grads, erhs, lg.d_w.clone(ctx), ctx)
                _accum(grads, tape.entries[i].third_id, lg.d_b.clone(ctx), ctx)
            elif ek == OP_RMSNORM:
                # y = rms_norm(x, gamma) (lhs=x, rhs=gamma). d_x, d_gamma via
                # rms_norm_backward(grad_out, x, weight, eps). eps = _RMS_EPS,
                # shared with record_rms_norm.
                var rx = tape.entries[i].saved0
                var rw = tape.entries[i].saved1
                var rg = rms_norm_backward(gop[], rx[], rw[], _RMS_EPS, ctx)
                _accum(grads, elhs, rg.d_x.clone(ctx), ctx)
                _accum(grads, erhs, rg.d_g.clone(ctx), ctx)
            elif ek == OP_SILU:
                # y = silu(x) (lhs=x; rhs unused). d_x via silu_backward(go, x).
                var sx = tape.entries[i].saved0
                _accum(grads, elhs, silu_backward(gop[], sx[], ctx), ctx)
            elif ek == OP_SWIGLU:
                # y = silu(gate)*up (lhs=gate, rhs=up). d_gate, d_up via
                # swiglu_backward(grad_out, gate, up).
                var wgate = tape.entries[i].saved0
                var wup = tape.entries[i].saved1
                var wgrads = swiglu_backward(gop[], wgate[], wup[], ctx)
                _accum(grads, elhs, wgrads.d_gate.clone(ctx), ctx)
                _accum(grads, erhs, wgrads.d_up.clone(ctx), ctx)
            elif ek == OP_MSE:
                # SPECIAL loss leaf: d_pred = mse_backward(pred, target) — the
                # full gradient INCLUDING the 2/N factor; the incoming gop is
                # intentionally ignored. d to lhs_id (pred); target (rhs) gets no
                # grad (left untracked → _accum no-op).
                var mp = tape.entries[i].saved0
                var mt = tape.entries[i].saved1
                _accum(grads, elhs, mse_backward(mp[], mt[], ctx), ctx)
        i -= 1
    return grads^
