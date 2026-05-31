# loop.mojo — reusable TRAINING-LOOP HARNESS for a generic parameter set.
# F32 master weights + BF16 compute, AdamW state, grad-accumulation,
# resumable checkpointing via the byte-exact safetensors writer/reader.
#
# Generalizes training/parity/mixed_precision_parity.mojo (the PROVEN single
# BF16-compute/F32-master step) into a multi-step loop, plus persistence so a
# real run (e.g. T5 Z-Image) can checkpoint and resume.
#
# ============================================================================
# DESIGN: grads-as-input, NOT callbacks. (Mojo 1.0.0b1 has no storable closures
# — see training/checkpoint.mojo's header for the same finding.) The harness
# therefore CANNOT hold a `forward_backward_fn`. The caller owns the compute:
#
#   loop over micro-batches:
#     1. caller reads each param's BF16 compute view: w_bf = state.compute_weight(i, ctx)
#        (cast_tensor F32 master -> BF16; the master stays F32)
#     2. caller runs its OWN BF16 forward + backward and obtains BF16 grads,
#     3. caller hands those BF16 grads back: state.accumulate_grads(grads, ctx)
#        (cast BF16 -> F32, sum into the F32 accumulators)
#   after `micro_steps` micro-batches:
#     4. state.apply_step(lr, ctx)   # mean-grad AdamW on the F32 masters
#
# The harness OWNS: F32 master weights, AdamW (m, v) device tensors + the 1-based
# step counter `t`, and one F32 grad accumulator per param. It is the single
# source of truth for trained weights + optimizer state, so checkpoint save/load
# is a pure harness concern.
#
# Move-only Tensor cannot be a bare List element (Mojo collections need
# Copyable), so every per-param list is List[ArcPointer[Tensor]] — the SAME idiom
# autograd.mojo / block_loader / lora.mojo use (ArcPointer copy == refcount bump).
#
# ============================================================================
# CHECKPOINT FORMAT (byte-exact via io/safetensors_writer.save_safetensors):
#   "param.<i>"   F32 master weights        (read back bit-for-bit -> max_abs 0)
#   "adam_m.<i>"  F32 first moment
#   "adam_v.<i>"  F32 second moment
#   "__meta__"    F32 [t_step, accum_count]  (optimizer scalars)
# load_checkpoint reads them back through io/safetensors.SafeTensors and rebuilds
# device tensors by H2D byte copy (the from_view path). F32 round-trips exactly
# (no BF16 truncation on the master path) → the resume-correctness gate is
# max_abs == 0 on the masters AND t restored.
#
# Toolchain: cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#   pixi run mojo run -I . serenitymojo/training/parity/loop_parity.mojo

from std.collections import List
from std.memory import ArcPointer
from std.gpu.host import DeviceContext
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.training.optim import adamw_step
from serenitymojo.io.safetensors_writer import save_safetensors
from serenitymojo.io.safetensors import SafeTensors


comptime TArc = ArcPointer[Tensor]


# ── F32 zero tensor with the same shape as `t` ───────────────────────────────
def _zeros_like_f32(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    var z = List[Float32]()
    for _ in range(t.numel()):
        z.append(Float32(0.0))
    return Tensor.from_host(z^, t.shape(), STDtype.F32, ctx)


# ── fresh 1-D F32 tensor from a host list ────────────────────────────────────
def _f32_1d(var values: List[Float32], n: Int, ctx: DeviceContext) raises -> Tensor:
    var sh = List[Int]()
    sh.append(n)
    return Tensor.from_host(values^, sh^, STDtype.F32, ctx)


# ----------------------------------------------------------------------------
# TrainState — the harness. F32 masters + AdamW (m,v) + F32 grad accumulators,
# all as List[ArcPointer[Tensor]] (move-only Tensor can't live in a List bare).
# ----------------------------------------------------------------------------
struct TrainState(Movable):
    var masters: List[ArcPointer[Tensor]]   # F32 master weights (source of truth)
    var m: List[ArcPointer[Tensor]]         # AdamW first moment per param (F32)
    var v: List[ArcPointer[Tensor]]         # AdamW second moment per param (F32)
    var accum: List[ArcPointer[Tensor]]     # F32 grad accumulator per param
    var t: Int                              # 1-based AdamW step counter
    var accum_count: Int                    # micro-batches since the last apply

    def __init__(
        out self, var init_masters: List[ArcPointer[Tensor]], ctx: DeviceContext
    ) raises:
        # Take ownership of the initial F32 masters; build zeroed m/v/accum to
        # match each param's shape. Masters MUST be F32 (the optim.mojo dtype).
        self.masters = List[ArcPointer[Tensor]]()
        self.m = List[ArcPointer[Tensor]]()
        self.v = List[ArcPointer[Tensor]]()
        self.accum = List[ArcPointer[Tensor]]()
        self.t = 0
        self.accum_count = 0
        for i in range(len(init_masters)):
            if init_masters[i][].dtype() != STDtype.F32:
                raise Error("TrainState: master weights must be F32")
            self.masters.append(init_masters[i])  # Arc refcount bump
            self.m.append(ArcPointer(_zeros_like_f32(init_masters[i][], ctx)))
            self.v.append(ArcPointer(_zeros_like_f32(init_masters[i][], ctx)))
            self.accum.append(ArcPointer(_zeros_like_f32(init_masters[i][], ctx)))

    def num_params(self) -> Int:
        return len(self.masters)

    # Cast the i-th F32 master DOWN to BF16 for the caller's compute. The master
    # stays F32 (cast_tensor returns a fresh tensor). This is the BF16-compute /
    # F32-master split proven in mixed_precision_parity.mojo.
    def compute_weight(self, i: Int, ctx: DeviceContext) raises -> Tensor:
        return cast_tensor(self.masters[i][], STDtype.BF16, ctx)

    # Read the i-th F32 master to host (inspection / checkpoint compare).
    def master_host(self, i: Int, ctx: DeviceContext) raises -> List[Float32]:
        return self.masters[i][].to_host(ctx)

    # Accumulate one micro-batch's grads. `grads` may be BF16 (the compute dtype)
    # — each is cast to F32 and SUMMED into the F32 accumulators. Order must match
    # the param order. Host-side add (parity-grade; not a hot path).
    def accumulate_grads(
        mut self, grads: List[ArcPointer[Tensor]], ctx: DeviceContext
    ) raises:
        if len(grads) != len(self.accum):
            raise Error("accumulate_grads: grad count != param count")
        for i in range(len(grads)):
            var g_f32 = cast_tensor(grads[i][], STDtype.F32, ctx)
            if g_f32.numel() != self.accum[i][].numel():
                raise Error("accumulate_grads: grad numel mismatch at param")
            var acc_h = self.accum[i][].to_host(ctx)
            var g_h = g_f32.to_host(ctx)
            var summed = List[Float32]()
            for j in range(len(acc_h)):
                summed.append(acc_h[j] + g_h[j])
            self.accum[i] = ArcPointer(
                Tensor.from_host(summed^, self.accum[i][].shape(), STDtype.F32, ctx))
        self.accum_count += 1

    # Apply ONE AdamW step from the accumulated grads (mean over the micro-batches
    # so loss scale is invariant to the micro-batch count), then zero the
    # accumulators. Raises if nothing was accumulated (surfaces a caller bug).
    def apply_step(
        mut self,
        lr: Float32,
        ctx: DeviceContext,
        beta1: Float32 = 0.9,
        beta2: Float32 = 0.999,
        eps: Float32 = 1e-8,
        weight_decay: Float32 = 0.0,
    ) raises:
        if self.accum_count == 0:
            raise Error("apply_step called with no accumulated grads")
        self.t += 1
        var inv = Float32(1.0) / Float32(self.accum_count)
        for i in range(len(self.masters)):
            # mean grad = accum / accum_count (fresh F32 grad tensor)
            var acc_h = self.accum[i][].to_host(ctx)
            var mean_h = List[Float32]()
            for j in range(len(acc_h)):
                mean_h.append(acc_h[j] * inv)
            var grad = Tensor.from_host(
                mean_h^, self.accum[i][].shape(), STDtype.F32, ctx)
            # AdamW updates master, m, v IN PLACE on the device buffers. The
            # ArcPointer payload is the SAME device buffer, so the in-place write
            # persists (refcount keeps it alive).
            var p = self.masters[i]
            var mm = self.m[i]
            var vv = self.v[i]
            adamw_step(p[], grad, mm[], vv[], self.t,
                       lr, beta1, beta2, eps, weight_decay, ctx)
            # zero the accumulator for the next cycle
            self.accum[i] = ArcPointer(_zeros_like_f32(self.masters[i][], ctx))
        self.accum_count = 0

    def opt_step(self) -> Int:
        return self.t


# ----------------------------------------------------------------------------
# Checkpoint save via the byte-exact writer (save_safetensors).
# ----------------------------------------------------------------------------
def save_checkpoint(state: TrainState, path: String, ctx: DeviceContext) raises:
    var names = List[String]()
    var tensors = List[ArcPointer[Tensor]]()
    var n = state.num_params()

    for i in range(n):
        names.append(String("param.") + String(i))
        tensors.append(state.masters[i])            # Arc bump, no copy
    for i in range(n):
        names.append(String("adam_m.") + String(i))
        tensors.append(state.m[i])
        names.append(String("adam_v.") + String(i))
        tensors.append(state.v[i])

    # __meta__ = [t_step, accum_count] as a 2-elem F32 tensor.
    var meta = List[Float32]()
    meta.append(Float32(state.t))
    meta.append(Float32(state.accum_count))
    names.append(String("__meta__"))
    tensors.append(ArcPointer(_f32_1d(meta^, 2, ctx)))

    save_safetensors(names, tensors, path, ctx)


# ----------------------------------------------------------------------------
# Checkpoint load via the reader (SafeTensors.open) → fresh TrainState.
# Masters are rebuilt by H2D byte copy of the mmap'd bytes (the from_view path),
# so F32 masters come back BIT-FOR-BIT. m/v/t/accum_count are restored too, so a
# resumed apply_step continues exactly where the saved run left off.
# ----------------------------------------------------------------------------
def _tensor_from_st(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)         # origin-bound mmap view
    var nbytes = info.size
    var staging = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var sp = staging.unsafe_ptr()
    for i in range(nbytes):
        sp[i] = bytes[i]
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=staging)
    ctx.synchronize()
    return Tensor(dev^, info.shape.copy(), info.dtype, 0)


def load_checkpoint(path: String, ctx: DeviceContext) raises -> TrainState:
    var st = SafeTensors.open(path)

    # Count params by probing param.<i> until missing (names() is unordered).
    var n = 0
    while st.tensors.__contains__(String("param.") + String(n)):
        n += 1
    if n == 0:
        raise Error("load_checkpoint: no param.* tensors in " + path)

    var masters = List[ArcPointer[Tensor]]()
    for i in range(n):
        masters.append(
            ArcPointer(_tensor_from_st(st, String("param.") + String(i), ctx)))
    var state = TrainState(masters^, ctx)

    for i in range(n):
        state.m[i] = ArcPointer(_tensor_from_st(st, String("adam_m.") + String(i), ctx))
        state.v[i] = ArcPointer(_tensor_from_st(st, String("adam_v.") + String(i), ctx))

    var meta_h = _tensor_from_st(st, String("__meta__"), ctx).to_host(ctx)
    state.t = Int(meta_h[0])
    state.accum_count = Int(meta_h[1])

    return state^
