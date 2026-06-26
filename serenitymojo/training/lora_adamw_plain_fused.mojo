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
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout

from serenitymojo.io.ffi import BytePtr, sys_memcpy
from serenitymojo.training.train_step import LoraAdapter


comptime _DYN1 = Layout.row_major(-1)
comptime _BLOCK = 256


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

    def elem_offset(self, adapter_idx: Int, b_side: Bool) -> Int:
        """Flat element offset of adapter `adapter_idx`'s A (or B) segment."""
        var off = 0
        for i in range(self.start, adapter_idx):
            off += self.seg_len[2 * (i - self.start)]
            off += self.seg_len[2 * (i - self.start) + 1]
        if b_side:
            off += self.seg_len[2 * (adapter_idx - self.start)]
        return off


def lora_adamw_plain_device_state_init(
    adapters: List[LoraAdapter],
    start: Int,
    end: Int,
    ctx: DeviceContext,
) raises -> LoraAdamWPlainDeviceState:
    """Pack host P/M/V for adapters[start:end) and upload ONCE."""
    if start < 0 or end > len(adapters) or start >= end:
        raise Error("lora_adamw_plain_device_state_init: bad adapter range")
    var total = 0
    var seg_len = List[Int]()
    for i in range(start, end):
        seg_len.append(len(adapters[i].a))
        seg_len.append(len(adapters[i].b))
        total += len(adapters[i].a) + len(adapters[i].b)

    var host_p = ctx.enqueue_create_host_buffer[DType.uint8](total * 2)
    var host_g = ctx.enqueue_create_host_buffer[DType.uint8](total * 4)
    var host_mv = ctx.enqueue_create_host_buffer[DType.uint8](total * 4)
    var host_m0 = ctx.enqueue_create_host_buffer[DType.uint8](total * 4)
    var host_v0 = ctx.enqueue_create_host_buffer[DType.uint8](total * 4)
    var hp = Int(host_p.unsafe_ptr())
    var hm = Int(host_m0.unsafe_ptr())
    var hv = Int(host_v0.unsafe_ptr())
    var off = 0
    for i in range(start, end):
        var n_a = len(adapters[i].a)
        var n_b = len(adapters[i].b)
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hp + off * 2),
            BytePtr(unsafe_from_address=Int(adapters[i].a.unsafe_ptr())),
            n_a * 2,
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
    ctx.enqueue_copy(dst_buf=dev_m, src_buf=host_m0)
    ctx.enqueue_copy(dst_buf=dev_v, src_buf=host_v0)
    ctx.synchronize()

    return LoraAdamWPlainDeviceState(
        dev_p^, dev_g^, dev_m^, dev_v^,
        host_p^, host_g^, host_mv^, seg_len^,
        start, end, total,
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
    var b1p = Float32(1.0)
    var b2p = Float32(1.0)
    for _ in range(t):
        b1p *= beta1
        b2p *= beta2
    var bc1 = Float32(1.0) - b1p
    var bc2 = Float32(1.0) - b2p

    var hg = Int(state.host_g.unsafe_ptr())
    var off = 0
    for i in range(state.start, state.end):
        var n_a = state.seg_len[2 * (i - state.start)]
        var n_b = state.seg_len[2 * (i - state.start) + 1]
        if len(d_a[i]) != n_a or len(d_b[i]) != n_b:
            raise Error(
                "fused_lora_adamw_plain_step_resident: grad len mismatch at "
                + String(i)
            )
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hg + off * 4),
            BytePtr(unsafe_from_address=Int(d_a[i].unsafe_ptr())),
            n_a * 4,
        )
        off += n_a
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=hg + off * 4),
            BytePtr(unsafe_from_address=Int(d_b[i].unsafe_ptr())),
            n_b * 4,
        )
        off += n_b

    ctx.enqueue_copy(dst_buf=state.dev_g, src_buf=state.host_g)

    var t_rl = RuntimeLayout[_DYN1].row_major(IndexList[1](state.total))
    var P = LayoutTensor[DType.bfloat16, _DYN1, MutAnyOrigin](
        state.dev_p.unsafe_ptr().bitcast[BFloat16](), t_rl
    )
    var G = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        state.dev_g.unsafe_ptr().bitcast[Float32](), t_rl
    )
    var M = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        state.dev_m.unsafe_ptr().bitcast[Float32](), t_rl
    )
    var V = LayoutTensor[DType.float32, _DYN1, MutAnyOrigin](
        state.dev_v.unsafe_ptr().bitcast[Float32](), t_rl
    )
    var grid = (state.total + _BLOCK - 1) // _BLOCK
    ctx.enqueue_function[_lora_adamw_plain_kernel, _lora_adamw_plain_kernel](
        P, G, M, V, state.total, lr, beta1, beta2, bc1, bc2, eps, weight_decay,
        grid_dim=grid, block_dim=_BLOCK,
    )

    ctx.enqueue_copy(dst_buf=state.host_p, src_buf=state.dev_p)
    ctx.synchronize()

    var hp = Int(state.host_p.unsafe_ptr())
    off = 0
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


def lora_adamw_plain_device_state_sync_moments(
    mut state: LoraAdamWPlainDeviceState,
    mut adapters: List[LoraAdapter],
    ctx: DeviceContext,
) raises:
    """Pull M then V back to the host adapter lists (save-state cadence only —
    NOT per step). Host m/v match the old path's values exactly."""
    var hmv = Int(state.host_mv.unsafe_ptr())
    # M
    ctx.enqueue_copy(dst_buf=state.host_mv, src_buf=state.dev_m)
    ctx.synchronize()
    var off = 0
    for i in range(state.start, state.end):
        var n_a = state.seg_len[2 * (i - state.start)]
        var n_b = state.seg_len[2 * (i - state.start) + 1]
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(adapters[i].ma.unsafe_ptr())),
            BytePtr(unsafe_from_address=hmv + off * 4),
            n_a * 4,
        )
        off += n_a
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(adapters[i].mb.unsafe_ptr())),
            BytePtr(unsafe_from_address=hmv + off * 4),
            n_b * 4,
        )
        off += n_b
    # V
    ctx.enqueue_copy(dst_buf=state.host_mv, src_buf=state.dev_v)
    ctx.synchronize()
    off = 0
    for i in range(state.start, state.end):
        var n_a = state.seg_len[2 * (i - state.start)]
        var n_b = state.seg_len[2 * (i - state.start) + 1]
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(adapters[i].va.unsafe_ptr())),
            BytePtr(unsafe_from_address=hmv + off * 4),
            n_a * 4,
        )
        off += n_a
        _ = sys_memcpy(
            BytePtr(unsafe_from_address=Int(adapters[i].vb.unsafe_ptr())),
            BytePtr(unsafe_from_address=hmv + off * 4),
            n_b * 4,
        )
        off += n_b
