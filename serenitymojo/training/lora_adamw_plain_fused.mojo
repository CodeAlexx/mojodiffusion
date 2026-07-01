# training/lora_adamw_plain_fused.mojo — GPU fused PLAIN-AdamW for LoRA
# adapter lists (the train_step.mojo `_adamw_host_list` semantics: F32 moments,
# plain RNE bf16 param writeback, NO stochastic rounding, NO moment
# quantization). Sibling of lora_adamw_ot_fused.mojo, which implements the
# OneTrainer bf16-moment+SR semantics Klein uses — do NOT mix them up: a
# trainer must fuse the SAME math its host loop ran or its loss anchors die.
#
# WHY: the Z-Image product trainer's optim stage is the host scalar loop
# (`_lora_adamw` → `_adamw_host_list`) over ~28M adapter elements — MEASURED
# 2026-06-11 at 0.183-0.190 s of a 2.0-2.1 s step (TIMING lines, 5-step run,
# zimage 64x64 bucket). Same disease Klein had at 6.0 s (fixed 06-10).
#
# Per-element math mirrored EXACTLY from `_adamw_host_list`
# (training/train_step.mojo:242):
#   mi = beta1*m[i] + (1-beta1)*g            # classic form (OT file uses the
#   vi = beta2*v[i] + (1-beta2)*g*g          #  m+(1-b1)(g-m) rearrangement —
#   m[i] = mi ; v[i] = vi                    #  keep THIS file's form verbatim)
#   m_hat = mi/bc1 ; v_hat = vi/bc2          # bc{1,2} = 1-beta^t, host loop
#   pv = f32(p[i]) ; if wd>0: pv *= (1-lr*wd)
#   pv -= lr*m_hat/(sqrt(v_hat)+eps)
#   p[i] = bf16_rne(pv)                      # hardware .cast RNE, like host
#
# CONTRACT (see lora_adamw_plain_fused_parity.mojo for the measured numbers):
# host F32 chain vs device F32 chain — every op correctly rounded on both, but
# codegen FMA contraction/reassociation can flip RNE ties by 1 ulp (the
# documented klein fused-AdamW class). Gate: p/m/v each within ±1 ulp at a
# tiny rate, plus the trainer's own loss anchors at 4dp.
#
# Data movement mirrors lora_adamw_ot_fused (pack via sys_memcpy into pinned
# staging, ONE H2D per role, ONE launch, ONE D2H per role, memcpy back).
#
# Mojo 1.0.0b1, NVIDIA GPU.

from std.math import sqrt
from std.gpu import global_idx
from std.gpu.host import DeviceContext, DeviceBuffer, HostBuffer
from std.utils.index import IndexList
from std.memory import ArcPointer
from std.time import perf_counter_ns
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.ffi import BytePtr, sys_memcpy
from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.training.device_train_step import (
    DeviceAdamWState,
    DeviceGradSet,
    DeviceTrainableSet,
    TrainStepDeviceResult,
    device_adamw_train_step_update_with_arena,
)
from serenitymojo.training.on_device_global_norm import on_device_grad_stats
from serenitymojo.training.on_device_global_norm import on_device_grad_stats_with_arena
from serenitymojo.training.perf_record import (
    PERF_FAST_PATH_DEVICE,
    TrainingPhaseTimings,
)
from serenitymojo.training.training_arena import (
    TRAINING_ARENA_PHASE_OPTIMIZER,
    TrainingArena,
)
from serenitymojo.training.train_step import LoraAdapter


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256
comptime TArc = ArcPointer[Tensor]


def _lora_adamw_plain_kernel(
    p: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    g: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    m: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    v: LayoutTensor[DType.float32, _DYN1, MutAnyOrigin],
    total: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    bc1: Float32,
    bc2: Float32,
    eps: Float32,
    weight_decay: Float32,
    clip_scale: Float32,   # global-norm clip factor (1.0 = no clip); folded here so
    # the clip is a free per-element GPU mul (no separate 54M-element host pass).
):
    var gid = Int(global_idx.x)
    if gid >= total:
        return
    var gv = rebind[Scalar[DType.float32]](g[gid]) * clip_scale
    var mi = beta1 * rebind[Scalar[DType.float32]](m[gid]) + (
        Float32(1.0) - beta1
    ) * gv
    var vi = beta2 * rebind[Scalar[DType.float32]](v[gid]) + (
        Float32(1.0) - beta2
    ) * gv * gv
    m[gid] = rebind[m.element_type](mi)
    v[gid] = rebind[v.element_type](vi)
    var m_hat = mi / bc1
    var v_hat = vi / bc2
    var pv = rebind[Scalar[DType.bfloat16]](p[gid]).cast[DType.float32]()
    if weight_decay > Float32(0.0):
        pv = pv * (Float32(1.0) - lr * weight_decay)
    pv = pv - lr * m_hat / (sqrt(v_hat) + eps)
    p[gid] = rebind[p.element_type](pv.cast[DType.bfloat16]())


def _lora_adamw_plain_kernel_bf16_state(
    p: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    g: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    m: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    v: LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin],
    total: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    bc1: Float32,
    bc2: Float32,
    eps: Float32,
    weight_decay: Float32,
    clip_scale: Float32,
):
    var gid = Int(global_idx.x)
    if gid >= total:
        return
    var gv = rebind[Scalar[DType.bfloat16]](g[gid]).cast[DType.float32]() * clip_scale
    var mi = beta1 * rebind[Scalar[DType.bfloat16]](m[gid]).cast[DType.float32]() + (
        Float32(1.0) - beta1
    ) * gv
    var vi = beta2 * rebind[Scalar[DType.bfloat16]](v[gid]).cast[DType.float32]() + (
        Float32(1.0) - beta2
    ) * gv * gv
    var m_q = mi.cast[DType.bfloat16]()
    var v_q = vi.cast[DType.bfloat16]()
    m[gid] = rebind[m.element_type](m_q)
    v[gid] = rebind[v.element_type](v_q)
    var m_hat = m_q.cast[DType.float32]() / bc1
    var v_hat = v_q.cast[DType.float32]() / bc2
    var pv = rebind[Scalar[DType.bfloat16]](p[gid]).cast[DType.float32]()
    if weight_decay > Float32(0.0):
        pv = pv * (Float32(1.0) - lr * weight_decay)
    pv = pv - lr * m_hat / (sqrt(v_hat) + eps)
    p[gid] = rebind[p.element_type](pv.cast[DType.bfloat16]())


def fused_lora_adamw_plain_step(
    mut adapters: List[LoraAdapter],
    d_a: List[List[Float32]],
    d_b: List[List[Float32]],
    start: Int,
    end: Int,
    t: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    ctx: DeviceContext,
    clip_scale: Float32 = Float32(1.0),   # global-norm clip factor folded into the
    # kernel (default 1.0 = no clip = byte-identical to the pre-clip-param callers).
) raises:
    """One fused PLAIN-AdamW step over adapters[start:end] (A and B params).
    Mutates a/b (bf16 RNE writeback) and ma/va/mb/vb (plain F32 moments) in
    place — same math as looping `_lora_adamw`. d_a/d_b are indexed by the
    SAME absolute adapter index as `adapters` (grads lists cover the full
    set; only [start:end) is stepped, like zimage's main-only loop)."""
    if start < 0 or end > len(adapters) or start >= end:
        if start == end:
            return
        raise Error("fused_lora_adamw_plain_step: bad adapter range")
    if len(d_a) < end or len(d_b) < end:
        raise Error("fused_lora_adamw_plain_step: grads shorter than range")
    if t < 1:
        raise Error("fused_lora_adamw_plain_step: t must be >= 1")

    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p

    var total = 0
    var seg_len = List[Int]()
    for i in range(start, end):
        var n_a = len(adapters[i].a)
        var n_b = len(adapters[i].b)
        if len(d_a[i]) != n_a or len(adapters[i].ma) != n_a or len(adapters[i].va) != n_a:
            raise Error(
                "fused_lora_adamw_plain_step: A-side len mismatch at adapter "
                + String(i)
            )
        if len(d_b[i]) != n_b or len(adapters[i].mb) != n_b or len(adapters[i].vb) != n_b:
            raise Error(
                "fused_lora_adamw_plain_step: B-side len mismatch at adapter "
                + String(i)
            )
        seg_len.append(n_a)
        seg_len.append(n_b)
        total += n_a + n_b
    if total == 0:
        return

    var host_p = ctx.enqueue_create_host_buffer[DType.uint8](total * 2)
    var host_g = ctx.enqueue_create_host_buffer[DType.uint8](total * 4)
    var host_m = ctx.enqueue_create_host_buffer[DType.uint8](total * 4)
    var host_v = ctx.enqueue_create_host_buffer[DType.uint8](total * 4)

    var hp = Int(host_p.unsafe_ptr())
    var hg = Int(host_g.unsafe_ptr())
    var hm = Int(host_m.unsafe_ptr())
    var hv = Int(host_v.unsafe_ptr())

    var off = 0
    for i in range(start, end):
        var n_a = seg_len[2 * (i - start)]
        var n_b = seg_len[2 * (i - start) + 1]
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hp + off * 2),
            BytePtr(unsafe_from_address=Int(adapters[i].a.unsafe_ptr())),
            n_a * 2,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hg + off * 4),
            BytePtr(unsafe_from_address=Int(d_a[i].unsafe_ptr())),
            n_a * 4,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hm + off * 4),
            BytePtr(unsafe_from_address=Int(adapters[i].ma.unsafe_ptr())),
            n_a * 4,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hv + off * 4),
            BytePtr(unsafe_from_address=Int(adapters[i].va.unsafe_ptr())),
            n_a * 4,
        )
        off += n_a
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hp + off * 2),
            BytePtr(unsafe_from_address=Int(adapters[i].b.unsafe_ptr())),
            n_b * 2,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hg + off * 4),
            BytePtr(unsafe_from_address=Int(d_b[i].unsafe_ptr())),
            n_b * 4,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hm + off * 4),
            BytePtr(unsafe_from_address=Int(adapters[i].mb.unsafe_ptr())),
            n_b * 4,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hv + off * 4),
            BytePtr(unsafe_from_address=Int(adapters[i].vb.unsafe_ptr())),
            n_b * 4,
        )
        off += n_b

    var dev_p = ctx.enqueue_create_buffer[DType.uint8](total * 2)
    var dev_g = ctx.enqueue_create_buffer[DType.uint8](total * 4)
    var dev_m = ctx.enqueue_create_buffer[DType.uint8](total * 4)
    var dev_v = ctx.enqueue_create_buffer[DType.uint8](total * 4)
    ctx.enqueue_copy(dst_buf=dev_p, src_buf=host_p)
    ctx.enqueue_copy(dst_buf=dev_g, src_buf=host_g)
    ctx.enqueue_copy(dst_buf=dev_m, src_buf=host_m)
    ctx.enqueue_copy(dst_buf=dev_v, src_buf=host_v)

    var t_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](total))
    var P = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        dev_p.unsafe_ptr().bitcast[BFloat16](), t_rl
    )
    var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        dev_g.unsafe_ptr().bitcast[Float32](), t_rl
    )
    var M = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        dev_m.unsafe_ptr().bitcast[Float32](), t_rl
    )
    var V = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        dev_v.unsafe_ptr().bitcast[Float32](), t_rl
    )

    var grid = (total + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_lora_adamw_plain_kernel, _lora_adamw_plain_kernel](
        P, G, M, V, total, lr, beta1, beta2, bc1, bc2, eps, weight_decay, clip_scale,
        grid_dim=grid, block_dim=_BLOCK,
    )

    ctx.enqueue_copy(dst_buf=host_p, src_buf=dev_p)
    ctx.enqueue_copy(dst_buf=host_m, src_buf=dev_m)
    ctx.enqueue_copy(dst_buf=host_v, src_buf=dev_v)
    ctx.synchronize()

    off = 0
    for i in range(start, end):
        var n_a = seg_len[2 * (i - start)]
        var n_b = seg_len[2 * (i - start) + 1]
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(adapters[i].a.unsafe_ptr())),
            BytePtr(unsafe_from_address=hp + off * 2),
            n_a * 2,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(adapters[i].ma.unsafe_ptr())),
            BytePtr(unsafe_from_address=hm + off * 4),
            n_a * 4,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(adapters[i].va.unsafe_ptr())),
            BytePtr(unsafe_from_address=hv + off * 4),
            n_a * 4,
        )
        off += n_a
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(adapters[i].b.unsafe_ptr())),
            BytePtr(unsafe_from_address=hp + off * 2),
            n_b * 2,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(adapters[i].mb.unsafe_ptr())),
            BytePtr(unsafe_from_address=hm + off * 4),
            n_b * 4,
        )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(adapters[i].vb.unsafe_ptr())),
            BytePtr(unsafe_from_address=hv + off * 4),
            n_b * 4,
        )
        off += n_b

# ══════════════════════════════════════════════════════════════════════════════
# v2 ENGINE (resident-set workstream, mandate 2026-06-11): persistent device
# P/M/V. The per-step host<->device round trip above moves ~490 MB/step
# (P 70 MB bf16 + G/M/V 140 MB F32 each, both directions for P/M/V) — MEASURED
# as the 0.139 s `opt` stage of the 1.9 s Z-Image B1 step. Resident state
# keeps P/M/V on device across steps; per step only G goes up and P comes
# back (host a/b mirror stays exact for b_absum/save/resume). M/V sync to
# host only at save-state cadence (`sync_moments_to_host`).
#
# NUMERICS: bit-identical to fused_lora_adamw_plain_step by construction —
# same kernel, same values (the old path's transit memcpys were
# value-preserving; removing them cannot change a bit). Gates: trainer loss
# anchors + b2dup/b1match trajectory identity.
# ══════════════════════════════════════════════════════════════════════════════


struct LoraAdamWPlainDeviceState(Movable):
    """Persistent device AdamW state for adapters[start:end) (A then B per
    adapter, flat). `dev_p` is THE live bf16 parameter buffer — build the
    model's device LoRA views as sub-buffers of it so the in-place optimizer
    update IS the next step's weights (no per-step param upload)."""
    var dev_p: DeviceBuffer[DType.uint8]
    var dev_g: DeviceBuffer[DType.uint8]
    var dev_m: DeviceBuffer[DType.uint8]
    var dev_v: DeviceBuffer[DType.uint8]
    var host_p: HostBuffer[DType.uint8]    # pinned: P readback mirror
    var host_g: HostBuffer[DType.uint8]    # pinned: G pack staging
    var host_mv: HostBuffer[DType.uint8]   # pinned: M/V save-cadence staging
    var seg_len: List[Int]                 # per adapter in range: n_a, n_b
    var start: Int
    var end: Int
    var total: Int
    var grad_dtype: STDtype
    var moment_dtype: STDtype

    def __init__(
        out self,
        var dev_p: DeviceBuffer[DType.uint8],
        var dev_g: DeviceBuffer[DType.uint8],
        var dev_m: DeviceBuffer[DType.uint8],
        var dev_v: DeviceBuffer[DType.uint8],
        var host_p: HostBuffer[DType.uint8],
        var host_g: HostBuffer[DType.uint8],
        var host_mv: HostBuffer[DType.uint8],
        var seg_len: List[Int],
        start: Int, end: Int, total: Int,
        grad_dtype: STDtype = STDtype.F32,
        moment_dtype: STDtype = STDtype.F32,
    ):
        self.dev_p = dev_p^
        self.dev_g = dev_g^
        self.dev_m = dev_m^
        self.dev_v = dev_v^
        self.host_p = host_p^
        self.host_g = host_g^
        self.host_mv = host_mv^
        self.seg_len = seg_len^
        self.start = start
        self.end = end
        self.total = total
        self.grad_dtype = grad_dtype
        self.moment_dtype = moment_dtype

    def elem_offset(self, adapter_idx: Int, b_side: Bool) -> Int:
        """Flat element offset of adapter `adapter_idx`'s A (or B) segment."""
        var off = 0
        for i in range(self.start, adapter_idx):
            off += self.seg_len[2 * (i - self.start)]
            off += self.seg_len[2 * (i - self.start) + 1]
        if b_side:
            off += self.seg_len[2 * (adapter_idx - self.start)]
        return off


def _state_dtype_bytes(dtype: STDtype) raises -> Int:
    if dtype == STDtype.F32 or dtype == STDtype.BF16:
        return dtype.byte_size()
    raise Error("LoraAdamWPlainDeviceState: only F32 and BF16 grad/moment buffers are supported")


def _copy_f32_list_to_host_dtype(
    values: List[Float32],
    host: HostBuffer[DType.uint8],
    elem_offset: Int,
    n: Int,
    dtype: STDtype,
) raises:
    if len(values) < n:
        raise Error("_copy_f32_list_to_host_dtype: source shorter than requested length")
    if dtype == STDtype.F32:
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(host.unsafe_ptr()) + elem_offset * 4),
            BytePtr(unsafe_from_address=Int(values.unsafe_ptr())),
            n * 4,
        )
    elif dtype == STDtype.BF16:
        var dst = host.unsafe_ptr().bitcast[BFloat16]()
        for i in range(n):
            dst[elem_offset + i] = values[i].cast[DType.bfloat16]()
    else:
        raise Error("_copy_f32_list_to_host_dtype: unsupported dtype")


def _copy_host_dtype_to_f32_list(
    host: HostBuffer[DType.uint8],
    elem_offset: Int,
    mut values: List[Float32],
    n: Int,
    dtype: STDtype,
) raises:
    if len(values) < n:
        raise Error("_copy_host_dtype_to_f32_list: destination shorter than requested length")
    if dtype == STDtype.F32:
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(values.unsafe_ptr())),
            BytePtr(unsafe_from_address=Int(host.unsafe_ptr()) + elem_offset * 4),
            n * 4,
        )
    elif dtype == STDtype.BF16:
        var src = host.unsafe_ptr().bitcast[BFloat16]()
        for i in range(n):
            values[i] = src[elem_offset + i].cast[DType.float32]()
    else:
        raise Error("_copy_host_dtype_to_f32_list: unsupported dtype")


def lora_adamw_plain_device_state_init(
    adapters: List[LoraAdapter],
    start: Int,
    end: Int,
    ctx: DeviceContext,
    grad_dtype: STDtype = STDtype.F32,
    moment_dtype: STDtype = STDtype.F32,
) raises -> LoraAdamWPlainDeviceState:
    """Pack host P/M/V for adapters[start:end) and upload ONCE."""
    if start < 0 or end > len(adapters) or start >= end:
        raise Error("lora_adamw_plain_device_state_init: bad adapter range")
    var grad_bsz = _state_dtype_bytes(grad_dtype)
    var moment_bsz = _state_dtype_bytes(moment_dtype)
    var total = 0
    var seg_len = List[Int]()
    for i in range(start, end):
        seg_len.append(len(adapters[i].a))
        seg_len.append(len(adapters[i].b))
        total += len(adapters[i].a) + len(adapters[i].b)

    var host_p = ctx.enqueue_create_host_buffer[DType.uint8](total * 2)
    var host_g = ctx.enqueue_create_host_buffer[DType.uint8](total * grad_bsz)
    var host_mv = ctx.enqueue_create_host_buffer[DType.uint8](total * moment_bsz)
    var host_m0 = ctx.enqueue_create_host_buffer[DType.uint8](total * moment_bsz)
    var host_v0 = ctx.enqueue_create_host_buffer[DType.uint8](total * moment_bsz)
    var hp = Int(host_p.unsafe_ptr())
    var off = 0
    for i in range(start, end):
        var n_a = len(adapters[i].a)
        var n_b = len(adapters[i].b)
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hp + off * 2),
            BytePtr(unsafe_from_address=Int(adapters[i].a.unsafe_ptr())),
            n_a * 2,
        )
        _copy_f32_list_to_host_dtype(adapters[i].ma, host_m0, off, n_a, moment_dtype)
        _copy_f32_list_to_host_dtype(adapters[i].va, host_v0, off, n_a, moment_dtype)
        off += n_a
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hp + off * 2),
            BytePtr(unsafe_from_address=Int(adapters[i].b.unsafe_ptr())),
            n_b * 2,
        )
        _copy_f32_list_to_host_dtype(adapters[i].mb, host_m0, off, n_b, moment_dtype)
        _copy_f32_list_to_host_dtype(adapters[i].vb, host_v0, off, n_b, moment_dtype)
        off += n_b

    var dev_p = ctx.enqueue_create_buffer[DType.uint8](total * 2)
    var dev_g = ctx.enqueue_create_buffer[DType.uint8](total * grad_bsz)
    var dev_m = ctx.enqueue_create_buffer[DType.uint8](total * moment_bsz)
    var dev_v = ctx.enqueue_create_buffer[DType.uint8](total * moment_bsz)
    ctx.enqueue_copy(dst_buf=dev_p, src_buf=host_p)
    ctx.enqueue_copy(dst_buf=dev_m, src_buf=host_m0)
    ctx.enqueue_copy(dst_buf=dev_v, src_buf=host_v0)
    ctx.synchronize()

    return LoraAdamWPlainDeviceState(
        dev_p^, dev_g^, dev_m^, dev_v^,
        host_p^, host_g^, host_mv^, seg_len^,
        start, end, total, grad_dtype, moment_dtype,
    )


def fused_lora_adamw_plain_step_resident(
    mut state: LoraAdamWPlainDeviceState,
    mut adapters: List[LoraAdapter],
    d_a: List[List[Float32]],
    d_b: List[List[Float32]],
    t: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    ctx: DeviceContext,
) raises:
    """Resident AdamW step: G up, kernel in-place on persistent P/M/V, P back
    to the host a/b mirror. Identical math to fused_lora_adamw_plain_step."""
    if t < 1:
        raise Error("fused_lora_adamw_plain_step_resident: t must be >= 1")
    var off = 0
    for i in range(state.start, state.end):
        var n_a = state.seg_len[2 * (i - state.start)]
        var n_b = state.seg_len[2 * (i - state.start) + 1]
        if len(d_a[i]) != n_a or len(d_b[i]) != n_b:
            raise Error(
                "fused_lora_adamw_plain_step_resident: grad len mismatch at "
                + String(i)
            )
        _copy_f32_list_to_host_dtype(d_a[i], state.host_g, off, n_a, state.grad_dtype)
        off += n_a
        _copy_f32_list_to_host_dtype(d_b[i], state.host_g, off, n_b, state.grad_dtype)
        off += n_b

    ctx.enqueue_copy(dst_buf=state.dev_g, src_buf=state.host_g)

    _lora_adamw_plain_resident_launch(
        state, t, lr, beta1, beta2, eps, weight_decay, ctx
    )
    lora_adamw_plain_device_state_sync_params(state, adapters, ctx)


def _lora_adamw_plain_resident_launch(
    state: LoraAdamWPlainDeviceState,
    t: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    ctx: DeviceContext,
    clip_scale: Float32 = Float32(1.0),
) raises:
    if t < 1:
        raise Error("_lora_adamw_plain_resident_launch: t must be >= 1")
    if state.total <= 0:
        raise Error("_lora_adamw_plain_resident_launch: empty resident state")

    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p

    var t_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](state.total))
    var P = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        state.dev_p.unsafe_ptr().bitcast[BFloat16](), t_rl
    )
    var grid = (state.total + _BLOCK - 1) // _BLOCK
    if state.grad_dtype == STDtype.F32 and state.moment_dtype == STDtype.F32:
        var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            state.dev_g.unsafe_ptr().bitcast[Float32](), t_rl
        )
        var M = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            state.dev_m.unsafe_ptr().bitcast[Float32](), t_rl
        )
        var V = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
            state.dev_v.unsafe_ptr().bitcast[Float32](), t_rl
        )
        ctx.enqueue_function[_lora_adamw_plain_kernel, _lora_adamw_plain_kernel](
            P, G, M, V, state.total, lr, beta1, beta2, bc1, bc2, eps, weight_decay,
            clip_scale,
            grid_dim=grid, block_dim=_BLOCK,
        )
    elif state.grad_dtype == STDtype.BF16 and state.moment_dtype == STDtype.BF16:
        var G = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            state.dev_g.unsafe_ptr().bitcast[BFloat16](), t_rl
        )
        var M = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            state.dev_m.unsafe_ptr().bitcast[BFloat16](), t_rl
        )
        var V = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
            state.dev_v.unsafe_ptr().bitcast[BFloat16](), t_rl
        )
        ctx.enqueue_function[
            _lora_adamw_plain_kernel_bf16_state,
            _lora_adamw_plain_kernel_bf16_state,
        ](
            P, G, M, V, state.total, lr, beta1, beta2, bc1, bc2, eps,
            weight_decay, clip_scale,
            grid_dim=grid, block_dim=_BLOCK,
        )
    else:
        raise Error("_lora_adamw_plain_resident_launch: unsupported grad/moment dtype combination")


def lora_adamw_plain_device_state_sync_params(
    mut state: LoraAdamWPlainDeviceState,
    mut adapters: List[LoraAdapter],
    ctx: DeviceContext,
) raises:
    """Pull persistent BF16 params back to the host adapter mirror.

    This is for save/inspection cadence. Fast product loops can keep the live
    model using state.dev_p views and skip this per step.
    """
    ctx.enqueue_copy(dst_buf=state.host_p, src_buf=state.dev_p)
    ctx.synchronize()

    var hp = Int(state.host_p.unsafe_ptr())
    var off = 0
    for i in range(state.start, state.end):
        var n_a = state.seg_len[2 * (i - state.start)]
        var n_b = state.seg_len[2 * (i - state.start) + 1]
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(adapters[i].a.unsafe_ptr())),
            BytePtr(unsafe_from_address=hp + off * 2),
            n_a * 2,
        )
        off += n_a
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(adapters[i].b.unsafe_ptr())),
            BytePtr(unsafe_from_address=hp + off * 2),
            n_b * 2,
        )
        off += n_b


def fused_lora_adamw_plain_step_resident_device_grads(
    mut state: LoraAdamWPlainDeviceState,
    mut adapters: List[LoraAdapter],
    grad_indices: List[Int],
    d_a: List[TArc],
    d_b: List[TArc],
    t: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    ctx: DeviceContext,
    clip_scale: Float32 = Float32(1.0),
    sync_params_to_host: Bool = False,
    max_grad_norm: Float32 = Float32(0.0),
) raises -> Float32:
    """Resident AdamW step with device-resident F32 grads.

    `grad_indices[i]` is the absolute adapter index for `d_a[i]`/`d_b[i]`.
    The function copies only device grad buffers into state.dev_g, launches the
    same resident AdamW kernel as the host-list compatibility path, and updates
    state.dev_p/dev_m/dev_v in place. It does not read gradients back to host.
    """
    if len(grad_indices) != len(d_a) or len(grad_indices) != len(d_b):
        raise Error("fused_lora_adamw_plain_step_resident_device_grads: grad list length mismatch")
    if state.start < 0 or state.end <= state.start:
        raise Error("fused_lora_adamw_plain_step_resident_device_grads: bad state range")
    if state.end > len(adapters):
        raise Error("fused_lora_adamw_plain_step_resident_device_grads: state range exceeds adapters")

    var seen = List[Int]()
    for _ in range(state.end - state.start):
        seen.append(0)

    var stat_grads = List[TArc]()
    for i in range(len(grad_indices)):
        var flat = grad_indices[i]
        if flat < state.start or flat >= state.end:
            raise Error(
                "fused_lora_adamw_plain_step_resident_device_grads: grad index outside optimizer range "
                + String(flat)
            )
        var local = flat - state.start
        if seen[local] != 0:
            raise Error(
                "fused_lora_adamw_plain_step_resident_device_grads: duplicate grad index "
                + String(flat)
            )
        seen[local] = 1

        var n_a = state.seg_len[2 * local]
        var n_b = state.seg_len[2 * local + 1]
        if d_a[i][].dtype() != state.grad_dtype or d_b[i][].dtype() != state.grad_dtype:
            raise Error(
                "fused_lora_adamw_plain_step_resident_device_grads: device grad dtype mismatch"
            )
        if d_a[i][].numel() != n_a or d_b[i][].numel() != n_b:
            raise Error(
                "fused_lora_adamw_plain_step_resident_device_grads: grad numel mismatch at "
                + String(flat)
            )

        var a_off = state.elem_offset(flat, False)
        var b_off = state.elem_offset(flat, True)
        var g_bsz = state.grad_dtype.byte_size()
        var dst_a = state.dev_g.create_sub_buffer[DType.uint8](a_off * g_bsz, n_a * g_bsz)
        var dst_b = state.dev_g.create_sub_buffer[DType.uint8](b_off * g_bsz, n_b * g_bsz)
        ctx.enqueue_copy(dst_buf=dst_a, src_buf=d_a[i][].buf)
        ctx.enqueue_copy(dst_buf=dst_b, src_buf=d_b[i][].buf)
        stat_grads.append(d_a[i].copy())
        stat_grads.append(d_b[i].copy())

    for i in range(len(seen)):
        if seen[i] == 0:
            raise Error(
                "fused_lora_adamw_plain_step_resident_device_grads: missing device grad for adapter "
                + String(state.start + i)
            )

    var stats = on_device_grad_stats(stat_grads, ctx)
    if stats.nonfinite_count != 0:
        raise Error(
            "fused_lora_adamw_plain_step_resident_device_grads: nonfinite device grads"
        )
    var effective_clip = clip_scale
    if max_grad_norm > Float32(0.0) and stats.grad_norm > max_grad_norm:
        effective_clip = effective_clip * (max_grad_norm / stats.grad_norm)

    _lora_adamw_plain_resident_launch(
        state, t, lr, beta1, beta2, eps, weight_decay, ctx, effective_clip
    )
    if sync_params_to_host:
        lora_adamw_plain_device_state_sync_params(state, adapters, ctx)
    return stats.grad_norm


def lora_adamw_plain_device_state_copy_device_grad_pair(
    mut state: LoraAdamWPlainDeviceState,
    adapter_idx: Int,
    d_a: TArc,
    d_b: TArc,
    ctx: DeviceContext,
) raises:
    """Copy one adapter's device-resident dA/dB into state.dev_g.

    This is the block-streaming companion to
    fused_lora_adamw_plain_step_resident_device_grads: callers can copy each
    block's transient device grads into the persistent optimizer grad buffer
    before those per-block tensors are freed, then run the preloaded-grad AdamW
    helper below once after backward completes.
    """
    if adapter_idx < state.start or adapter_idx >= state.end:
        raise Error(
            "lora_adamw_plain_device_state_copy_device_grad_pair: grad index outside optimizer range "
            + String(adapter_idx)
        )
    var local = adapter_idx - state.start
    var n_a = state.seg_len[2 * local]
    var n_b = state.seg_len[2 * local + 1]
    if d_a[].dtype() != state.grad_dtype or d_b[].dtype() != state.grad_dtype:
        raise Error(
            "lora_adamw_plain_device_state_copy_device_grad_pair: device grad dtype mismatch"
        )
    if d_a[].numel() != n_a or d_b[].numel() != n_b:
        raise Error(
            "lora_adamw_plain_device_state_copy_device_grad_pair: grad numel mismatch at "
            + String(adapter_idx)
        )
    var a_off = state.elem_offset(adapter_idx, False)
    var b_off = state.elem_offset(adapter_idx, True)
    var g_bsz = state.grad_dtype.byte_size()
    var dst_a = state.dev_g.create_sub_buffer[DType.uint8](a_off * g_bsz, n_a * g_bsz)
    var dst_b = state.dev_g.create_sub_buffer[DType.uint8](b_off * g_bsz, n_b * g_bsz)
    ctx.enqueue_copy(dst_buf=dst_a, src_buf=d_a[].buf)
    ctx.enqueue_copy(dst_buf=dst_b, src_buf=d_b[].buf)


def fused_lora_adamw_plain_step_resident_preloaded_grads(
    mut state: LoraAdamWPlainDeviceState,
    mut adapters: List[LoraAdapter],
    t: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    ctx: DeviceContext,
    clip_scale: Float32 = Float32(1.0),
    sync_params_to_host: Bool = False,
    max_grad_norm: Float32 = Float32(0.0),
) raises -> Float32:
    """Resident AdamW step using state.dev_g already filled on device.

    No gradient tensors are read back to host. This path is for streaming
    backward implementations that copy each transient per-block grad into
    state.dev_g as it is produced, avoiding a full device-grad list lifetime.
    """
    if state.start < 0 or state.end <= state.start:
        raise Error("fused_lora_adamw_plain_step_resident_preloaded_grads: bad state range")
    if state.end > len(adapters):
        raise Error(
            "fused_lora_adamw_plain_step_resident_preloaded_grads: state range exceeds adapters"
        )
    if state.total <= 0:
        raise Error("fused_lora_adamw_plain_step_resident_preloaded_grads: empty grad buffer")

    var stat_grads = List[TArc]()
    stat_grads.append(TArc(Tensor(state.dev_g.copy(), [state.total], state.grad_dtype)))
    var stats = on_device_grad_stats(stat_grads, ctx)
    if stats.nonfinite_count != 0:
        raise Error(
            "fused_lora_adamw_plain_step_resident_preloaded_grads: nonfinite device grads"
        )
    var effective_clip = clip_scale
    if max_grad_norm > Float32(0.0) and stats.grad_norm > max_grad_norm:
        effective_clip = effective_clip * (max_grad_norm / stats.grad_norm)

    _lora_adamw_plain_resident_launch(
        state, t, lr, beta1, beta2, eps, weight_decay, ctx, effective_clip
    )
    if sync_params_to_host:
        lora_adamw_plain_device_state_sync_params(state, adapters, ctx)
    return stats.grad_norm


def lora_adamw_plain_preloaded_shared_abi_train_step(
    mut state: LoraAdamWPlainDeviceState,
    loss: Float32,
    t: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    mut arena: TrainingArena,
    ctx: DeviceContext,
    max_grad_norm: Float32 = Float32(0.0),
) raises -> TrainStepDeviceResult:
    """Run resident flat LoRA P/G/M/V through the shared device train-step ABI.

    `state.dev_g` must already contain the flat F32 gradient buffer. This keeps
    Krea2/ZImage-style resident LoRA optimizers on the shared
    DeviceTrainableSet/DeviceGradSet/TrainStepDeviceResult contract while
    preserving BF16 param storage and F32 moment storage.
    """
    if state.start < 0 or state.end <= state.start:
        raise Error("lora_adamw_plain_preloaded_shared_abi_train_step: bad state range")
    if state.total <= 0:
        raise Error("lora_adamw_plain_preloaded_shared_abi_train_step: empty grad buffer")

    if state.grad_dtype == STDtype.BF16 and state.moment_dtype == STDtype.BF16:
        var arena_before = arena.stats()
        var norm_t0 = perf_counter_ns()
        var sh_bf16: List[Int] = [state.total]
        var stat_grads = List[TArc]()
        stat_grads.append(TArc(Tensor(state.dev_g.copy(), sh_bf16.copy(), STDtype.BF16)))
        var stats = on_device_grad_stats_with_arena(stat_grads, arena, ctx)
        if stats.nonfinite_count != 0:
            raise Error(
                "lora_adamw_plain_preloaded_shared_abi_train_step: nonfinite BF16 device grads"
            )
        var norm_t1 = perf_counter_ns()
        var clip = Float32(1.0)
        if max_grad_norm > Float32(0.0) and stats.grad_norm > max_grad_norm:
            clip = max_grad_norm / stats.grad_norm
        _lora_adamw_plain_resident_launch(
            state, t, lr, beta1, beta2, eps, weight_decay, ctx, clip
        )
        var opt_t1 = perf_counter_ns()
        var arena_after = arena.stats()
        var phases = TrainingPhaseTimings(
            0.0,
            0.0,
            0.0,
            Float64(norm_t1 - norm_t0) / 1.0e9,
            0.0,
            Float64(opt_t1 - norm_t1) / 1.0e9,
            0.0,
            0.0,
        )
        var result = TrainStepDeviceResult(
            loss,
            stats.grad_norm,
            clip,
            phases^,
            stats.scalar_readback_count,
            0,
            arena_after.sync_count - arena_before.sync_count,
            stats.nonfinite_count,
            PERF_FAST_PATH_DEVICE,
            String("krea2-ai-toolkit-bf16-adamw-preloaded"),
            String(""),
        )
        result.validate()
        return result^

    var sh: List[Int] = [state.total]
    var trainables = DeviceTrainableSet()
    var p = Tensor(state.dev_p.copy(), sh.copy(), STDtype.BF16)
    trainables.append(
        String("lora.flat.params"),
        TArc(p^),
        String("plain-lora-flat-bf16"),
    )
    var grads = DeviceGradSet()
    var g = Tensor(state.dev_g.copy(), sh.copy(), STDtype.F32)
    grads.append(
        String("lora.flat.params"),
        TArc(g^),
        String("plain-lora-flat-f32-grad"),
    )
    var adamw_state = DeviceAdamWState()
    var m = Tensor(state.dev_m.copy(), sh.copy(), STDtype.F32)
    var v = Tensor(state.dev_v.copy(), sh.copy(), STDtype.F32)
    adamw_state.append(TArc(m^), TArc(v^))

    var mark = arena.mark(TRAINING_ARENA_PHASE_OPTIMIZER)
    var result = device_adamw_train_step_update_with_arena(
        trainables,
        grads,
        adamw_state,
        loss,
        t,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        max_grad_norm,
        arena,
        ctx,
    )
    arena.rewind(mark)
    return result^


def lora_adamw_plain_device_state_sync_moments(
    mut state: LoraAdamWPlainDeviceState,
    mut adapters: List[LoraAdapter],
    ctx: DeviceContext,
) raises:
    """Pull M then V back to the host adapter lists (save-state cadence only —
    NOT per step). Host m/v match the old path's values exactly."""
    # M
    ctx.enqueue_copy(dst_buf=state.host_mv, src_buf=state.dev_m)
    ctx.synchronize()
    var off = 0
    for i in range(state.start, state.end):
        var n_a = state.seg_len[2 * (i - state.start)]
        var n_b = state.seg_len[2 * (i - state.start) + 1]
        _copy_host_dtype_to_f32_list(state.host_mv, off, adapters[i].ma, n_a, state.moment_dtype)
        off += n_a
        _copy_host_dtype_to_f32_list(state.host_mv, off, adapters[i].mb, n_b, state.moment_dtype)
        off += n_b
    # V
    ctx.enqueue_copy(dst_buf=state.host_mv, src_buf=state.dev_v)
    ctx.synchronize()
    off = 0
    for i in range(state.start, state.end):
        var n_a = state.seg_len[2 * (i - state.start)]
        var n_b = state.seg_len[2 * (i - state.start) + 1]
        _copy_host_dtype_to_f32_list(state.host_mv, off, adapters[i].va, n_a, state.moment_dtype)
        off += n_a
        _copy_host_dtype_to_f32_list(state.host_mv, off, adapters[i].vb, n_b, state.moment_dtype)
        off += n_b
