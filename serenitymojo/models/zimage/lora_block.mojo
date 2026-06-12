# serenitymojo/models/zimage/lora_block.mojo
#
# LoRA-ON-PROJECTION for the Z-Image (NextDiT) blocks. Mirrors the PROVEN Ernie
# LoRA template (models/ernie/lora_block.mojo), specialized to Z-Image's SEVEN
# un-fused target projections per block:
#   attention.{to_q, to_k, to_v, to_out.0}  and  feed_forward.{w1, w3, w2}
# (Z-Image has separate q/k/v — like Ernie — so 7 separate adapters.) The MLP is
# SwiGLU so the three MLP linears are w1 (gate), w3 (up), w2 (down), NOT
# gate_proj/up_proj/linear_fc2 — see the OT/diffusers key map in zimage_stack_lora.
#
# Z-Image has TWO block flavors, BOTH with these 7 Linear projections (so BOTH get
# LoRA per OneTrainer's default — LoRAModuleWrapper adapts EVERY nn.Linear child of
# the transformer; ZImageLoRASetup.py:57 passes no restrictive default filter):
#   * MODULATED block (noise refiners + main layers) — adaLN-tanh + SwiGLU.
#   * UNMODULATED block (context refiners) — plain sandwich-norm residual + SwiGLU.
# So this file provides a LoRA-aware forward/backward for EACH flavor, each reducing
# bit-for-bit to its base when adapters are absent.
#
# WHY THE TRAINER MATH IS THE AUTHORITY (identical to Ernie/Klein lora_block.mojo)
#   For a projection y = linear(x, W) (W [out,in]), the LoRA-adapted output is
#       y' = linear(x, W) + scale·((x @ Aᵀ) @ Bᵀ)
#   A=[rank,in], B=[out,rank], scale = alpha/rank. This MATCHES the inference merge
#   in inference-flame zimage_nextdit.rs `lora.apply(...)` (W' = W + scale·B@A) and
#   the OneTrainer forward (LoRAModule.py:328-329: orig_forward(x) + up(down(x)) *
#   alpha/rank).
#
#   BACKWARD (given d_y' at the projection output, x the saved projection input):
#       d_dy = scale·d_y'                    [M,out]
#       d_B  = d_dyᵀ @ t   (t = x @ Aᵀ)      [out,rank]
#       d_t  = d_dy  @ B                      [M,rank]
#       d_A  = d_tᵀ  @ x                      [rank,in]
#       d_x  = d_t   @ A                      [M,in]   (LoRA branch's contribution
#                                                       to the projection INPUT grad)
#   The base path (frozen W) ALSO yields d_x_base = d_y' @ W; the caller SUMS d_x
#   into that. d_A/d_B go to the optimizer; the base W grad is discarded for LoRA.
#
# Base weights are frozen during LoRA training, so the base projection backward
# computes d_x only. LoRA A/B grads still use full low-rank linear_backward.
#
# Mojo 1.0.0b1: def not fn; Tensor move-only crosses API as host List[Float32].

from std.gpu.host import DeviceContext, HostBuffer
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from std.time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear, linear_slab
from serenitymojo.ops.linalg_backward import (
    linear_backward, linear_backward_dx, linear_backward_dw,
    linear_backward_dx_slab, linear_backward_dw_slab,
)
from serenitymojo.autograd_v2.step_slab import StepSlab

# REUSE the trainer's LoRA structs (the target authority).
from serenitymojo.training.train_step import LoraAdapter, LoraGrads

# Forward + backward ops shared with the base block (Tenet 1: nothing new here).
from serenitymojo.ops.norm import rms_norm, rms_norm_slab
from serenitymojo.ops.activations import swiglu, swiglu_slab
from serenitymojo.ops.vec_swiglu import vec_swiglu
from serenitymojo.ops.elementwise import (
    modulate, residual_gate, modulate_slab, residual_gate_slab,
)
from serenitymojo.ops.vec_modulate import vec_modulate
from serenitymojo.ops.rope import rope_interleaved, rope_interleaved_slab
from serenitymojo.ops.attention import sdpa_nomask, sdpa_nomask_slab
# cuDNN flash SDPA for the zimage GRAPH backward path (approved 2026-06-11).
# bf16-native, S=1248 padded to 1280 inside the flash wrapper. v1 scope:
# the graph recompute+backward SDPA only (record_sdpa_slab + OPK_SDPA arm);
# the v3 forward keeps math sdpa. CAPTURE MUST BE OFF with this on (the
# flash wrapper allocates per call -> breaks replay; ZIMAGE_V2_CAPTURE).
comptime ZIMAGE_SDPA_FLASH = True
from serenitymojo.ops.unary import tanh_op, tanh_op_slab
from serenitymojo.ops.tensor_algebra import (
    reshape_owned, reshape_in_place, add, mul_scalar, add_slab, mul_scalar_slab,
)
from serenitymojo.ops.norm_backward import rms_norm_backward, rms_norm_backward_dx
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.rope_struct_backward import (
    gate_residual_backward, gate_residual_backward_dxdy, rope_backward,
)
from serenitymojo.ops.activation_backward import tanh_backward
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward
from serenitymojo.models.zimage.weights import ZImageBlockWeights
from serenitymojo.models.zimage.block import (
    ZImageModVecs,
    ZImageBlockSaved, ZImageBlockGrads,
    ZImageRefinerSaved, ZImageRefinerGrads,
)


comptime TArc = ArcPointer[Tensor]


def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


struct ZImageModVecsDevice(Copyable, Movable):
    var scale_msa: TArc
    var gate_msa: TArc
    var scale_mlp: TArc
    var gate_mlp: TArc
    var zeros: TArc

    def __init__(
        out self,
        var scale_msa: TArc, var gate_msa: TArc,
        var scale_mlp: TArc, var gate_mlp: TArc, var zeros: TArc,
    ):
        self.scale_msa = scale_msa^
        self.gate_msa = gate_msa^
        self.scale_mlp = scale_mlp^
        self.gate_mlp = gate_mlp^
        self.zeros = zeros^


def zimage_modvecs_to_device(
    mv: ZImageModVecs, D: Int, ctx: DeviceContext
) raises -> ZImageModVecsDevice:
    return ZImageModVecsDevice(
        TArc(_t(mv.scale_msa.copy(), [D], ctx)),
        TArc(_t(mv.gate_msa.copy(), [D], ctx)),
        TArc(_t(mv.scale_mlp.copy(), [D], ctx)),
        TArc(_t(mv.gate_mlp.copy(), [D], ctx)),
        TArc(_t(_zeros(D), [D], ctx)),
    )


struct ZImageModVecsAllDevice(Movable):
    """ALL blocks' adaLN mod-vecs as ONE device slab (v2 engine: sync-site
    elimination). One packed H2D upload per step replaces the per-vector
    syncing `_t()` uploads × blocks × (fwd + recompute + bwd) — the measured
    cuStreamSynchronize storm in the B1 trainer (~1.9k syncs/step, 0.56-0.60 s
    GPU idle; nsys /tmp/zt.sqlite). Per-block [D] tensors are zero-copy
    sub-buffer views; `slab` owns the device allocation and MUST outlive
    `per_block` (repo discipline: sub-buffer views do not extend lifetime —
    see scratch_ring.mojo).

    Slab layout (F32): block-major, 4 vecs per block in slot order
    scale_msa, gate_msa, scale_mlp, gate_mlp; one shared zero [D] vec at the
    tail serving every block's `zeros` (read-only shift operand)."""
    var slab: Tensor
    var per_block: List[ZImageModVecsDevice]

    def __init__(
        out self, var slab: Tensor, var per_block: List[ZImageModVecsDevice]
    ):
        self.slab = slab^
        self.per_block = per_block^


def zimage_modvecs_all_to_device(
    mods: List[ZImageModVecs], D: Int, ctx: DeviceContext
) raises -> ZImageModVecsAllDevice:
    """Pack every block's 4 mod-vecs + one shared zeros vec into a single
    H2D upload (ONE sync), then carve per-block ZImageModVecsDevice views.
    Values are byte-identical to per-vec `_t()` uploads (same F32 carrier)."""
    var nb = len(mods)
    var vb = D * 4                      # bytes per [D] F32 vec
    var nbytes = (nb * 4 + 1) * vb      # +1: shared zeros vec at the tail
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var fp = host.unsafe_ptr().bitcast[Float32]()
    for i in range(nb):
        var base = i * 4 * D
        for c in range(D):
            fp[base + c] = mods[i].scale_msa[c]
        for c in range(D):
            fp[base + D + c] = mods[i].gate_msa[c]
        for c in range(D):
            fp[base + 2 * D + c] = mods[i].scale_mlp[c]
        for c in range(D):
            fp[base + 3 * D + c] = mods[i].gate_mlp[c]
    var zbase = nb * 4 * D
    for c in range(D):
        fp[zbase + c] = Float32(0.0)
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()                   # the ONE modvec upload sync per step

    var zeros = TArc(Tensor(
        dev.create_sub_buffer[DType.uint8](zbase * 4, vb), [D], STDtype.F32
    ))
    var per_block = List[ZImageModVecsDevice]()
    for i in range(nb):
        var base = i * 4 * vb
        per_block.append(ZImageModVecsDevice(
            TArc(Tensor(
                dev.create_sub_buffer[DType.uint8](base, vb), [D], STDtype.F32
            )),
            TArc(Tensor(
                dev.create_sub_buffer[DType.uint8](base + vb, vb), [D], STDtype.F32
            )),
            TArc(Tensor(
                dev.create_sub_buffer[DType.uint8](base + 2 * vb, vb), [D], STDtype.F32
            )),
            TArc(Tensor(
                dev.create_sub_buffer[DType.uint8](base + 3 * vb, vb), [D], STDtype.F32
            )),
            zeros.copy(),
        ))
    return ZImageModVecsAllDevice(
        Tensor(dev^, [nb * 4 + 1, D], STDtype.F32), per_block^
    )


def _add_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i] + b[i])
    return o^


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(Float32(0.0))
    return o^


# Adapter forward contribution on x [M,in] -> [M,out] (host list in/out).
# Byte-identical to train_step._lora_fwd / ernie_lora_fwd.
def zimage_lora_fwd(
    x_h: List[Float32], lo: LoraAdapter, M: Int, ctx: DeviceContext
) raises -> List[Float32]:
    var nb1 = Optional[Tensor](None)
    var t = linear(
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx),
        nb1^, ctx,
    ).to_host(ctx)                                   # [M,rank]
    var nb2 = Optional[Tensor](None)
    var dy = linear(
        Tensor.from_host(t.copy(), [M, lo.rank], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx),
        nb2^, ctx,
    ).to_host(ctx)                                   # [M,out]
    var out = List[Float32]()
    for i in range(len(dy)):
        out.append(lo.scale * dy[i])
    return out^


# Optionally-applied adapter forward: if `lo` is present, return base_y + LoRA;
# else return base_y unchanged (base-path no-regression when an adapter is absent).
def zimage_lora_apply(
    base_y: List[Float32], x_h: List[Float32], lo: Optional[LoraAdapter],
    M: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    if not lo:
        return base_y.copy()
    var contrib = zimage_lora_fwd(x_h, lo.value(), M, ctx)
    var out = List[Float32]()
    for i in range(len(base_y)):
        out.append(base_y[i] + contrib[i])
    return out^


# LoRA backward that ALSO returns the LoRA branch's contribution to d_x.
struct ZImageLoraGrads(Copyable, Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]
    var d_x: List[Float32]   # LoRA contribution to the projection INPUT grad [M,in]

    def __init__(
        out self, var d_a: List[Float32], var d_b: List[Float32], var d_x: List[Float32]
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x = d_x^


def zimage_lora_bwd(
    d_contrib_h: List[Float32], x_h: List[Float32], lo: LoraAdapter,
    M: Int, ctx: DeviceContext,
) raises -> ZImageLoraGrads:
    # t = x @ Aᵀ  (recompute; cheap)
    var nb_t = Optional[Tensor](None)
    var t = linear(
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx),
        nb_t^, ctx,
    ).to_host(ctx)                                   # [M,rank]
    var d_dy = List[Float32]()
    for i in range(len(d_contrib_h)):
        d_dy.append(lo.scale * d_contrib_h[i])       # [M,out]
    # dy = t @ Bᵀ  -> d_B (d_w) and d_t (d_x)
    var lbB = linear_backward(
        Tensor.from_host(d_dy^, [M, lo.out_f], STDtype.BF16, ctx),
        Tensor.from_host(t.copy(), [M, lo.rank], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx),
        M, lo.rank, lo.out_f, ctx,
    )
    var d_t = lbB.d_x.to_host(ctx)                   # [M,rank]
    var d_b = lbB.d_w.to_host(ctx)                   # [out_f,rank]
    # t = x @ Aᵀ  -> d_A (d_w) and d_x_lo (d_x)
    var lbA = linear_backward(
        Tensor.from_host(d_t^, [M, lo.rank], STDtype.BF16, ctx),
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.BF16, ctx),
        Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx),
        M, lo.in_f, lo.rank, ctx,
    )
    var d_x_lo = lbA.d_x.to_host(ctx)                # [M,in_f]
    var d_a = lbA.d_w.to_host(ctx)                   # [rank,in_f]
    return ZImageLoraGrads(d_a^, d_b^, d_x_lo^)


# ── per-block LoRA carrier: the 7 optional adapters (slot order is canonical) ──
# slot 0 to_q, 1 to_k, 2 to_v, 3 to_out.0, 4 feed_forward.w1, 5 .w3, 6 .w2.
comptime ZIMAGE_SLOTS = 7
comptime SLOT_Q = 0
comptime SLOT_K = 1
comptime SLOT_V = 2
comptime SLOT_O = 3
comptime SLOT_W1 = 4    # feed_forward.w1 (SwiGLU gate)
comptime SLOT_W3 = 5    # feed_forward.w3 (SwiGLU up)
comptime SLOT_W2 = 6    # feed_forward.w2 (SwiGLU down)


struct ZImageBlockLora(Copyable, Movable):
    var to_q: Optional[LoraAdapter]
    var to_k: Optional[LoraAdapter]
    var to_v: Optional[LoraAdapter]
    var to_out: Optional[LoraAdapter]
    var w1: Optional[LoraAdapter]
    var w3: Optional[LoraAdapter]
    var w2: Optional[LoraAdapter]

    def __init__(
        out self,
        var to_q: Optional[LoraAdapter], var to_k: Optional[LoraAdapter],
        var to_v: Optional[LoraAdapter], var to_out: Optional[LoraAdapter],
        var w1: Optional[LoraAdapter], var w3: Optional[LoraAdapter],
        var w2: Optional[LoraAdapter],
    ):
        self.to_q = to_q^
        self.to_k = to_k^
        self.to_v = to_v^
        self.to_out = to_out^
        self.w1 = w1^
        self.w3 = w3^
        self.w2 = w2^


struct ZImageLoraAdapterDevice(Copyable, Movable):
    var a: TArc
    var b: TArc
    var rank: Int
    var in_f: Int
    var out_f: Int
    var scale: Float32

    def __init__(
        out self, var a: TArc, var b: TArc,
        rank: Int, in_f: Int, out_f: Int, scale: Float32,
    ):
        self.a = a^
        self.b = b^
        self.rank = rank
        self.in_f = in_f
        self.out_f = out_f
        self.scale = scale


struct ZImageBlockLoraDevice(Copyable, Movable):
    var to_q: ZImageLoraAdapterDevice
    var to_k: ZImageLoraAdapterDevice
    var to_v: ZImageLoraAdapterDevice
    var to_out: ZImageLoraAdapterDevice
    var w1: ZImageLoraAdapterDevice
    var w3: ZImageLoraAdapterDevice
    var w2: ZImageLoraAdapterDevice

    def __init__(
        out self,
        var to_q: ZImageLoraAdapterDevice, var to_k: ZImageLoraAdapterDevice,
        var to_v: ZImageLoraAdapterDevice, var to_out: ZImageLoraAdapterDevice,
        var w1: ZImageLoraAdapterDevice, var w3: ZImageLoraAdapterDevice,
        var w2: ZImageLoraAdapterDevice,
    ):
        self.to_q = to_q^
        self.to_k = to_k^
        self.to_v = to_v^
        self.to_out = to_out^
        self.w1 = w1^
        self.w3 = w3^
        self.w2 = w2^


def zimage_lora_adapter_to_device(
    lo: LoraAdapter, ctx: DeviceContext
) raises -> ZImageLoraAdapterDevice:
    return ZImageLoraAdapterDevice(
        TArc(Tensor.from_host_bf16(lo.a.copy(), [lo.rank, lo.in_f], ctx)),
        TArc(Tensor.from_host_bf16(lo.b.copy(), [lo.out_f, lo.rank], ctx)),
        lo.rank, lo.in_f, lo.out_f, lo.scale,
    )


def zimage_lora_apply_device(
    var base_y: Tensor, x: Tensor, lo: ZImageLoraAdapterDevice,
    M: Int, ctx: DeviceContext,
) raises -> Tensor:
    if lo.scale == Float32(0.0):
        return base_y^
    var nb1 = Optional[Tensor](None)
    var t = linear(x, lo.a[], nb1^, ctx)
    var nb2 = Optional[Tensor](None)
    var dy = linear(t, lo.b[], nb2^, ctx)
    var contrib = mul_scalar(dy, lo.scale, ctx)
    return add(base_y^, contrib^, ctx)


def zimage_lora_apply_device_slab(
    var base_y: Tensor, x: Tensor, lo: ZImageLoraAdapterDevice,
    M: Int, ctx: DeviceContext, mut slab: StepSlab,
) raises -> Tensor:
    """StepSlab variant of `zimage_lora_apply_device` (this file :390) —
    byte-identical math (same linear/mul_scalar/add calls in the same order);
    internal ops route to their _slab siblings (autograd_v2 contract C8, P4)."""
    if lo.scale == Float32(0.0):
        return base_y^
    var nb1 = Optional[Tensor](None)
    var t = linear_slab(x, lo.a[], nb1^, ctx, slab)
    var nb2 = Optional[Tensor](None)
    var dy = linear_slab(t, lo.b[], nb2^, ctx, slab)
    var contrib = mul_scalar_slab(dy, lo.scale, ctx, slab)
    return add_slab(base_y^, contrib^, ctx, slab)


struct _HostGradPair(Copyable, Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]

    def __init__(out self, var d_a: List[Float32], var d_b: List[Float32]):
        self.d_a = d_a^
        self.d_b = d_b^


def _host_from_f32_buffer(
    host: HostBuffer[DType.uint8], n: Int
) -> List[Float32]:
    var out = List[Float32]()
    var fp = host.unsafe_ptr().bitcast[Float32]()
    for i in range(n):
        out.append(fp[i])
    return out^


def _tensor_to_host_f32(t: Tensor, ctx: DeviceContext) raises -> List[Float32]:
    if t.dtype() == STDtype.F32:
        var host = ctx.enqueue_create_host_buffer[DType.uint8](t.nbytes())
        ctx.enqueue_copy(dst_buf=host, src_buf=t.buf)
        ctx.synchronize()
        return _host_from_f32_buffer(host, t.numel())
    # Host AdamW stores master params and moments as F32; device grads may be BF16.
    var t32 = cast_tensor(t, STDtype.F32, ctx)
    var host = ctx.enqueue_create_host_buffer[DType.uint8](t32.nbytes())
    ctx.enqueue_copy(dst_buf=host, src_buf=t32.buf)
    ctx.synchronize()
    return _host_from_f32_buffer(host, t32.numel())


def _to_host_pair_f32(a: Tensor, b: Tensor, ctx: DeviceContext) raises -> _HostGradPair:
    var ah = _tensor_to_host_f32(a, ctx)
    var bh = _tensor_to_host_f32(b, ctx)
    return _HostGradPair(ah^, bh^)


struct ZImageLoraDeviceGrads(Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]
    var d_x: Tensor

    def __init__(
        out self, var d_a: List[Float32], var d_b: List[Float32], var d_x: Tensor
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x = d_x^


def zimage_lora_bwd_device_resident(
    d_contrib: Tensor, x: Tensor, lo: ZImageLoraAdapterDevice,
    M: Int, ctx: DeviceContext,
) raises -> ZImageLoraDeviceGrads:
    var nb_t = Optional[Tensor](None)
    var t = linear(x, lo.a[], nb_t^, ctx)
    var d_dy = mul_scalar(d_contrib, lo.scale, ctx)

    var d_t = linear_backward_dx(d_dy, lo.b[], M, lo.rank, lo.out_f, ctx)
    var d_b_t = linear_backward_dw(d_dy, t, M, lo.rank, lo.out_f, ctx)

    var d_x_lo = linear_backward_dx(d_t, lo.a[], M, lo.in_f, lo.rank, ctx)
    var d_a_t = linear_backward_dw(d_t, x, M, lo.in_f, lo.rank, ctx)

    var pair = _to_host_pair_f32(d_a_t, d_b_t, ctx)
    return ZImageLoraDeviceGrads(pair.d_a.copy(), pair.d_b.copy(), d_x_lo^)


struct ZImageLoraDeviceGradTensors(Copyable, Movable):
    var d_a: TArc
    var d_b: TArc
    var d_x: TArc

    def __init__(out self, var d_a: TArc, var d_b: TArc, var d_x: TArc):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x = d_x^


def zimage_lora_bwd_device_resident_tensors(
    d_contrib: Tensor, x: Tensor, lo: ZImageLoraAdapterDevice,
    M: Int, ctx: DeviceContext,
) raises -> ZImageLoraDeviceGradTensors:
    var nb_t = Optional[Tensor](None)
    var t = linear(x, lo.a[], nb_t^, ctx)
    var d_dy = mul_scalar(d_contrib, lo.scale, ctx)

    var d_t = linear_backward_dx(d_dy, lo.b[], M, lo.rank, lo.out_f, ctx)
    var d_b_t = linear_backward_dw(d_dy, t, M, lo.rank, lo.out_f, ctx)

    var d_x_lo = linear_backward_dx(d_t, lo.a[], M, lo.in_f, lo.rank, ctx)
    var d_a_t = linear_backward_dw(d_t, x, M, lo.in_f, lo.rank, ctx)

    return ZImageLoraDeviceGradTensors(TArc(d_a_t^), TArc(d_b_t^), TArc(d_x_lo^))


def zimage_lora_bwd_device_resident_tensors_slab(
    d_contrib: Tensor, x: Tensor, lo: ZImageLoraAdapterDevice,
    M: Int, ctx: DeviceContext, mut slab: StepSlab,
) raises -> ZImageLoraDeviceGradTensors:
    """StepSlab variant of `zimage_lora_bwd_device_resident_tensors` (this
    file :485ff) — byte-identical math (same op calls in the same order);
    internal ops route to their _slab siblings (autograd_v2 contract C8, P4)."""
    var nb_t = Optional[Tensor](None)
    var t = linear_slab(x, lo.a[], nb_t^, ctx, slab)
    var d_dy = mul_scalar_slab(d_contrib, lo.scale, ctx, slab)

    var d_t = linear_backward_dx_slab(d_dy, lo.b[], M, lo.rank, lo.out_f, ctx, slab)
    var d_b_t = linear_backward_dw_slab(d_dy, t, M, lo.rank, lo.out_f, ctx, slab)

    var d_x_lo = linear_backward_dx_slab(d_t, lo.a[], M, lo.in_f, lo.rank, ctx, slab)
    var d_a_t = linear_backward_dw_slab(d_t, x, M, lo.in_f, lo.rank, ctx, slab)

    return ZImageLoraDeviceGradTensors(TArc(d_a_t^), TArc(d_b_t^), TArc(d_x_lo^))


# ── per-block LoRA grads (parallel to the 7 slots) ───────────────────────────
struct ZImageBlockLoraGrads(Copyable, Movable):
    var d_a: List[List[Float32]]   # ZIMAGE_SLOTS entries (empty if slot absent)
    var d_b: List[List[Float32]]

    def __init__(out self, var d_a: List[List[Float32]], var d_b: List[List[Float32]]):
        self.d_a = d_a^
        self.d_b = d_b^


# proj-backward result: d_x [M,in] (base + LoRA summed). Base d_w is discarded
# for LoRA training and must not be materialized for Z-Image full depth.
struct _ProjGrads(Movable):
    var d_x: Tensor

    def __init__(out self, var d_x: Tensor):
        self.d_x = d_x^


# helper: run frozen-base d_x then add the LoRA branch's d_x (if present),
# collecting the LoRA d_a/d_b into the slot lists. Returns the SUMMED d_x [M,in].
def _proj_bwd_with_lora(
    d_y: Tensor, x_in: Tensor, w: Tensor, x_in_h: List[Float32],
    lo: Optional[LoraAdapter], slot: Int,
    M: Int, in_f: Int, out_f: Int,
    mut d_a_slots: List[List[Float32]], mut d_b_slots: List[List[Float32]],
    ctx: DeviceContext,
) raises -> _ProjGrads:
    var base_dx = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)
    if lo:
        var d_y_h = d_y.to_host(ctx)
        var lg = zimage_lora_bwd(d_y_h, x_in_h, lo.value(), M, ctx)
        d_a_slots[slot] = lg.d_a.copy()
        d_b_slots[slot] = lg.d_b.copy()
        var base_dx_h = base_dx.to_host(ctx)
        var summed = _add_lists(base_dx_h, lg.d_x)
        return _ProjGrads(_t(summed, [M, in_f], ctx))
    return _ProjGrads(base_dx^)


def _proj_bwd_with_lora_device(
    d_y: Tensor, x_in: Tensor, w: Tensor,
    lo: ZImageLoraAdapterDevice, slot: Int,
    M: Int, in_f: Int, out_f: Int,
    mut d_a_slots: List[List[Float32]], mut d_b_slots: List[List[Float32]],
    ctx: DeviceContext,
) raises -> _ProjGrads:
    var base_dx = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)
    var lg = zimage_lora_bwd_device_resident(d_y, x_in, lo, M, ctx)
    d_a_slots[slot] = lg.d_a.copy()
    d_b_slots[slot] = lg.d_b.copy()
    var summed = add(base_dx^, lg.d_x^, ctx)
    return _ProjGrads(summed^)


struct _ProjTensorGrads(Movable):
    var d_x: Tensor
    var d_a: TArc
    var d_b: TArc

    def __init__(out self, var d_x: Tensor, var d_a: TArc, var d_b: TArc):
        self.d_x = d_x^
        self.d_a = d_a^
        self.d_b = d_b^


def _proj_bwd_with_lora_device_tensors(
    d_y: Tensor, x_in: Tensor, w: Tensor,
    lo: ZImageLoraAdapterDevice,
    M: Int, in_f: Int, out_f: Int,
    ctx: DeviceContext,
) raises -> _ProjTensorGrads:
    var base_dx = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)
    var lg = zimage_lora_bwd_device_resident_tensors(d_y, x_in, lo, M, ctx)
    var summed = add(base_dx^, lg.d_x[], ctx)
    return _ProjTensorGrads(summed^, lg.d_a.copy(), lg.d_b.copy())


# ══════════════════════════════════════════════════════════════════════════════
# MODULATED block (noise refiners + main layers) — LoRA-aware fwd + bwd.
# Mirrors zimage_block_forward/_backward EXACTLY (models/zimage/block.mojo), adding
# the LoRA contribution to each of the 7 trained projection outputs BEFORE the
# downstream op consumes it. When all 7 adapters are absent this reduces bit-for-bit
# to the base forward. The `saved` activations are the LoRA-MODIFIED ones, so
# backward recompute regenerates them identically (same checkpoint contract).
# ══════════════════════════════════════════════════════════════════════════════
struct ZImageBlockForwardLora(Movable):
    var out: List[Float32]
    var saved: ZImageBlockSaved

    def __init__(out self, var out: List[Float32], var saved: ZImageBlockSaved):
        self.out = out^
        self.saved = saved^


struct ZImageBlockForwardLoraTensor(Movable):
    var out: TArc
    var saved: ZImageBlockSaved

    def __init__(out self, var out: TArc, var saved: ZImageBlockSaved):
        self.out = out^
        self.saved = saved^


def zimage_block_lora_forward[
    H: Int, Dh: Int, S: Int
](
    x: List[Float32],
    w: ZImageBlockWeights, mv: ZImageModVecs, lora: ZImageBlockLora,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageBlockForwardLora:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var zeros = _zeros(D)
    var x_t = _t(x, [S, D], ctx)

    # --- attention sub-block (sandwich norm) ---
    var xn1 = rms_norm(x_t, w.n1[], eps, ctx)
    var xn1s = modulate(
        xn1, _t(mv.scale_msa.copy(), [D], ctx), _t(zeros.copy(), [D], ctx), ctx
    )
    var xn1s_h = xn1s.to_host(ctx)                              # [S,D] (LoRA input for q/k/v)

    var no_bias = Optional[Tensor](None)
    var q_base = linear(xn1s, w.wq[], no_bias^, ctx).to_host(ctx)
    var no_bias_k = Optional[Tensor](None)
    var k_base = linear(xn1s, w.wk[], no_bias_k^, ctx).to_host(ctx)
    var no_bias_v = Optional[Tensor](None)
    var v_base = linear(xn1s, w.wv[], no_bias_v^, ctx).to_host(ctx)

    var q_h = zimage_lora_apply(q_base, xn1s_h, lora.to_q, S, ctx)
    var k_h = zimage_lora_apply(k_base, xn1s_h, lora.to_k, S, ctx)
    var v_h = zimage_lora_apply(v_base, xn1s_h, lora.to_v, S, ctx)

    var q_pre = reshape_owned(_t(q_h, [S, D], ctx)^, [1, S, H, Dh])
    var k_pre = reshape_owned(_t(k_h, [S, D], ctx)^, [1, S, H, Dh])
    var v = reshape_owned(_t(v_h, [S, D], ctx)^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)

    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])
    var att_flat_h = att_flat.to_host(ctx)                      # [S,D] (LoRA input for wo)

    var no_bias_o = Optional[Tensor](None)
    var att_o_base = linear(att_flat, w.wo[], no_bias_o^, ctx).to_host(ctx)
    var att_o_h = zimage_lora_apply(att_o_base, att_flat_h, lora.to_out, S, ctx)
    var att_o = _t(att_o_h, [S, D], ctx)

    var attn_n2 = rms_norm(att_o, w.n2[], eps, ctx)
    var gate_msa_raw = _t(mv.gate_msa.copy(), [D], ctx)
    var gate_msa_t = tanh_op(gate_msa_raw, ctx)
    var h = residual_gate(x_t, gate_msa_t, attn_n2, ctx)

    # --- MLP sub-block (SwiGLU, sandwich norm) ---
    var xfn1 = rms_norm(h, w.fn1[], eps, ctx)
    var xfn1s = modulate(
        xfn1, _t(mv.scale_mlp.copy(), [D], ctx), _t(zeros.copy(), [D], ctx), ctx
    )
    var xfn1s_h = xfn1s.to_host(ctx)                            # [S,D] (LoRA input for w1/w3)

    var no_bias_g = Optional[Tensor](None)
    var g_base = linear(xfn1s, w.w1[], no_bias_g^, ctx).to_host(ctx)
    var no_bias_u = Optional[Tensor](None)
    var u_base = linear(xfn1s, w.w3[], no_bias_u^, ctx).to_host(ctx)
    var g_pre_h = zimage_lora_apply(g_base, xfn1s_h, lora.w1, S, ctx)
    var u_h = zimage_lora_apply(u_base, xfn1s_h, lora.w3, S, ctx)
    var g_pre = _t(g_pre_h, [S, F], ctx)
    var u = _t(u_h, [S, F], ctx)

    var act = swiglu(g_pre, u, ctx)
    var act_h = act.to_host(ctx)                                # [S,F] (LoRA input for w2)

    var no_bias_d = Optional[Tensor](None)
    var ff_base = linear(act, w.w2[], no_bias_d^, ctx).to_host(ctx)
    var ff_h = zimage_lora_apply(ff_base, act_h, lora.w2, S, ctx)
    var ff = _t(ff_h, [S, D], ctx)

    var ff_n2 = rms_norm(ff, w.fn2[], eps, ctx)
    var gate_mlp_raw = _t(mv.gate_mlp.copy(), [D], ctx)
    var gate_mlp_t = tanh_op(gate_mlp_raw, ctx)
    var result = residual_gate(h, gate_mlp_t, ff_n2, ctx).to_host(ctx)

    var saved = ZImageBlockSaved(
        TArc(x_t^), TArc(xn1^), TArc(xn1s^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rms^), TArc(k_rms^), TArc(q_rope^), TArc(k_rope^),
        TArc(att_flat^), TArc(att_o^),
        TArc(gate_msa_t^), TArc(gate_msa_raw^), TArc(h^),
        TArc(xfn1^), TArc(xfn1s^),
        TArc(g_pre^), TArc(u^), TArc(act^), TArc(ff^),
        TArc(gate_mlp_t^), TArc(gate_mlp_raw^),
    )
    return ZImageBlockForwardLora(result^, saved^)


struct ZImageBlockLoraBackward(Movable):
    var base: ZImageBlockGrads
    var lora: ZImageBlockLoraGrads

    def __init__(out self, var base: ZImageBlockGrads, var lora: ZImageBlockLoraGrads):
        self.base = base^
        self.lora = lora^


def zimage_block_lora_backward[
    H: Int, Dh: Int, S: Int
](
    d_out: List[Float32],
    w: ZImageBlockWeights, mv: ZImageModVecs, lora: ZImageBlockLora,
    saved: ZImageBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageBlockLoraBackward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var d_out_t = _t(d_out, [S, D], ctx)

    var d_a_slots = List[List[Float32]]()
    var d_b_slots = List[List[Float32]]()
    for _ in range(ZIMAGE_SLOTS):
        d_a_slots.append(List[Float32]())
        d_b_slots.append(List[Float32]())

    # out = residual_gate(h, gate_mlp_t, ff_n2); recompute ff_n2 = rms_norm(ff, fn2)
    var ff_n2_y = rms_norm(saved.ff[], w.fn2[], eps, ctx)
    var grg2 = gate_residual_backward(
        d_out_t, saved.h[], saved.gate_mlp_t[], ff_n2_y, ctx
    )
    var d_gate_mlp = tanh_backward(grg2.d_g, saved.gate_mlp_raw[], ctx).to_host(ctx)

    # ff_n2 = rms_norm(ff, fn2)
    var rb_fn2 = rms_norm_backward(grg2.d_y, saved.ff[], w.fn2[], eps, ctx)
    var d_fn2 = rb_fn2.d_g.to_host(ctx)

    # ff = linear(act, w2)[+LoRA(w2)]  W [D, F]
    var act_h = saved.act[].to_host(ctx)
    var lb_w2 = _proj_bwd_with_lora(
        rb_fn2.d_x, saved.act[], w.w2[], act_h,
        lora.w2, SLOT_W2, S, F, D, d_a_slots, d_b_slots, ctx,
    )
    var d_w2 = List[Float32]()

    # act = swiglu(g_pre, u) -> d_g_pre, d_u
    var sg = swiglu_backward(lb_w2.d_x, saved.g_pre[], saved.u[], ctx)

    # g_pre = linear(xfn1s, w1)[+LoRA] ; u = linear(xfn1s, w3)[+LoRA]  W [F, D]
    var xfn1s_h = saved.xfn1s[].to_host(ctx)
    var lb_w1 = _proj_bwd_with_lora(
        sg.d_gate, saved.xfn1s[], w.w1[], xfn1s_h,
        lora.w1, SLOT_W1, S, D, F, d_a_slots, d_b_slots, ctx,
    )
    var d_w1 = List[Float32]()
    var lb_w3 = _proj_bwd_with_lora(
        sg.d_up, saved.xfn1s[], w.w3[], xfn1s_h,
        lora.w3, SLOT_W3, S, D, F, d_a_slots, d_b_slots, ctx,
    )
    var d_w3 = List[Float32]()
    var d_xfn1s = add(lb_w1.d_x, lb_w3.d_x, ctx)

    # xfn1s = modulate(xfn1, scale_mlp, 0)
    var mb_mlp = modulate_backward(d_xfn1s, saved.xfn1[], _t(mv.scale_mlp.copy(), [D], ctx), ctx)
    var d_scale_mlp = mb_mlp.d_scale.to_host(ctx)
    var rb_fn1 = rms_norm_backward(mb_mlp.d_x, saved.h[], w.fn1[], eps, ctx)
    var d_fn1 = rb_fn1.d_g.to_host(ctx)
    var d_h = add(grg2.d_x, rb_fn1.d_x, ctx)

    # --- attention sub-block backward ---
    # h = residual_gate(x, gate_msa_t, attn_n2); recompute attn_n2 = rms_norm(att_o, n2)
    var attn_n2_y = rms_norm(saved.att_o[], w.n2[], eps, ctx)
    var grg1 = gate_residual_backward(
        d_h, saved.x[], saved.gate_msa_t[], attn_n2_y, ctx
    )
    var d_gate_msa = tanh_backward(grg1.d_g, saved.gate_msa_raw[], ctx).to_host(ctx)

    var rb_n2 = rms_norm_backward(grg1.d_y, saved.att_o[], w.n2[], eps, ctx)
    var d_n2 = rb_n2.d_g.to_host(ctx)

    # att_o = linear(att_flat, wo)[+LoRA(to_out)]  W [D, D]
    var att_flat_h = saved.att_flat[].to_host(ctx)
    var lb_o = _proj_bwd_with_lora(
        rb_n2.d_x, saved.att_flat[], w.wo[], att_flat_h,
        lora.to_out, SLOT_O, S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wo = List[Float32]()

    reshape_in_place(lb_o.d_x, [1, S, H, Dh])
    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v[], lb_o.d_x, scale, ctx
    )
    var d_q_rms = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_rms = rope_backward(sb.d_k, cos, sin, True, ctx)

    var rb_q = rms_norm_backward(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)
    var d_k_norm = rb_k.d_g.to_host(ctx)

    reshape_in_place(rb_q.d_x, [S, D])
    reshape_in_place(rb_k.d_x, [S, D])
    reshape_in_place(sb.d_v, [S, D])

    # q/k/v = linear(xn1s, w{q,k,v})[+LoRA]  W [D, D]; xn1s feeds all three.
    var xn1s_h = saved.xn1s[].to_host(ctx)
    var lb_q = _proj_bwd_with_lora(
        rb_q.d_x, saved.xn1s[], w.wq[], xn1s_h,
        lora.to_q, SLOT_Q, S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wq = List[Float32]()
    var lb_k = _proj_bwd_with_lora(
        rb_k.d_x, saved.xn1s[], w.wk[], xn1s_h,
        lora.to_k, SLOT_K, S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wk = List[Float32]()
    var lb_v = _proj_bwd_with_lora(
        sb.d_v, saved.xn1s[], w.wv[], xn1s_h,
        lora.to_v, SLOT_V, S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wv = List[Float32]()
    var d_xn1s = add(add(lb_q.d_x, lb_k.d_x, ctx), lb_v.d_x, ctx)

    # xn1s = modulate(xn1, scale_msa, 0)
    var mb_sa = modulate_backward(d_xn1s, saved.xn1[], _t(mv.scale_msa.copy(), [D], ctx), ctx)
    var d_scale_msa = mb_sa.d_scale.to_host(ctx)
    var rb_n1 = rms_norm_backward(mb_sa.d_x, saved.x[], w.n1[], eps, ctx)
    var d_n1 = rb_n1.d_g.to_host(ctx)

    var d_x_res = grg1.d_x.to_host(ctx)
    var d_x_norm = rb_n1.d_x.to_host(ctx)
    var d_x = _add_lists(d_x_res, d_x_norm)

    var base = ZImageBlockGrads(
        d_x^, d_n1^,
        d_wq^, d_wk^, d_wv^, d_wo^,
        d_q_norm^, d_k_norm^,
        d_n2^, d_fn1^,
        d_w1^, d_w3^, d_w2^, d_fn2^,
        d_scale_msa^, d_gate_msa^,
        d_scale_mlp^, d_gate_mlp^,
    )
    return ZImageBlockLoraBackward(base^, ZImageBlockLoraGrads(d_a_slots^, d_b_slots^))


def zimage_block_lora_forward_device[
    H: Int, Dh: Int, S: Int
](
    x: List[Float32],
    w: ZImageBlockWeights, mv: ZImageModVecs, lora: ZImageBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageBlockForwardLora:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var zeros = _zeros(D)
    var x_t = _t(x, [S, D], ctx)

    var xn1 = rms_norm(x_t, w.n1[], eps, ctx)
    var xn1s = modulate(
        xn1, _t(mv.scale_msa.copy(), [D], ctx), _t(zeros.copy(), [D], ctx), ctx
    )

    var no_bias = Optional[Tensor](None)
    var q_base = linear(xn1s, w.wq[], no_bias^, ctx)
    var no_bias_k = Optional[Tensor](None)
    var k_base = linear(xn1s, w.wk[], no_bias_k^, ctx)
    var no_bias_v = Optional[Tensor](None)
    var v_base = linear(xn1s, w.wv[], no_bias_v^, ctx)

    var q = zimage_lora_apply_device(q_base^, xn1s, lora.to_q, S, ctx)
    var k = zimage_lora_apply_device(k_base^, xn1s, lora.to_k, S, ctx)
    var v_flat = zimage_lora_apply_device(v_base^, xn1s, lora.to_v, S, ctx)

    var q_pre = reshape_owned(q^, [1, S, H, Dh])
    var k_pre = reshape_owned(k^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)

    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])

    var no_bias_o = Optional[Tensor](None)
    var att_o_base = linear(att_flat, w.wo[], no_bias_o^, ctx)
    var att_o = zimage_lora_apply_device(att_o_base^, att_flat, lora.to_out, S, ctx)

    var attn_n2 = rms_norm(att_o, w.n2[], eps, ctx)
    var gate_msa_raw = _t(mv.gate_msa.copy(), [D], ctx)
    var gate_msa_t = tanh_op(gate_msa_raw, ctx)
    var h = residual_gate(x_t, gate_msa_t, attn_n2, ctx)

    var xfn1 = rms_norm(h, w.fn1[], eps, ctx)
    var xfn1s = modulate(
        xfn1, _t(mv.scale_mlp.copy(), [D], ctx), _t(zeros.copy(), [D], ctx), ctx
    )

    var no_bias_g = Optional[Tensor](None)
    var g_base = linear(xfn1s, w.w1[], no_bias_g^, ctx)
    var no_bias_u = Optional[Tensor](None)
    var u_base = linear(xfn1s, w.w3[], no_bias_u^, ctx)
    var g_pre = zimage_lora_apply_device(g_base^, xfn1s, lora.w1, S, ctx)
    var u = zimage_lora_apply_device(u_base^, xfn1s, lora.w3, S, ctx)

    var act = swiglu(g_pre, u, ctx)

    var no_bias_d = Optional[Tensor](None)
    var ff_base = linear(act, w.w2[], no_bias_d^, ctx)
    var ff = zimage_lora_apply_device(ff_base^, act, lora.w2, S, ctx)

    var ff_n2 = rms_norm(ff, w.fn2[], eps, ctx)
    var gate_mlp_raw = _t(mv.gate_mlp.copy(), [D], ctx)
    var gate_mlp_t = tanh_op(gate_mlp_raw, ctx)
    var result = residual_gate(h, gate_mlp_t, ff_n2, ctx).to_host(ctx)

    var saved = ZImageBlockSaved(
        TArc(x_t^), TArc(xn1^), TArc(xn1s^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rms^), TArc(k_rms^), TArc(q_rope^), TArc(k_rope^),
        TArc(att_flat^), TArc(att_o^),
        TArc(gate_msa_t^), TArc(gate_msa_raw^), TArc(h^),
        TArc(xfn1^), TArc(xfn1s^),
        TArc(g_pre^), TArc(u^), TArc(act^), TArc(ff^),
        TArc(gate_mlp_t^), TArc(gate_mlp_raw^),
    )
    return ZImageBlockForwardLora(result^, saved^)


def zimage_block_lora_forward_device_tensor[
    H: Int, Dh: Int, S: Int
](
    x_arc: TArc,
    w: ZImageBlockWeights, mv: ZImageModVecs, lora: ZImageBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageBlockForwardLoraTensor:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var zeros = _zeros(D)

    var xn1 = rms_norm(x_arc[], w.n1[], eps, ctx)
    var xn1s = modulate(
        xn1, _t(mv.scale_msa.copy(), [D], ctx), _t(zeros.copy(), [D], ctx), ctx
    )

    var no_bias = Optional[Tensor](None)
    var q_base = linear(xn1s, w.wq[], no_bias^, ctx)
    var no_bias_k = Optional[Tensor](None)
    var k_base = linear(xn1s, w.wk[], no_bias_k^, ctx)
    var no_bias_v = Optional[Tensor](None)
    var v_base = linear(xn1s, w.wv[], no_bias_v^, ctx)

    var q = zimage_lora_apply_device(q_base^, xn1s, lora.to_q, S, ctx)
    var k = zimage_lora_apply_device(k_base^, xn1s, lora.to_k, S, ctx)
    var v_flat = zimage_lora_apply_device(v_base^, xn1s, lora.to_v, S, ctx)

    var q_pre = reshape_owned(q^, [1, S, H, Dh])
    var k_pre = reshape_owned(k^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)

    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])

    var no_bias_o = Optional[Tensor](None)
    var att_o_base = linear(att_flat, w.wo[], no_bias_o^, ctx)
    var att_o = zimage_lora_apply_device(att_o_base^, att_flat, lora.to_out, S, ctx)

    var attn_n2 = rms_norm(att_o, w.n2[], eps, ctx)
    var gate_msa_raw = _t(mv.gate_msa.copy(), [D], ctx)
    var gate_msa_t = tanh_op(gate_msa_raw, ctx)
    var h = residual_gate(x_arc[], gate_msa_t, attn_n2, ctx)

    var xfn1 = rms_norm(h, w.fn1[], eps, ctx)
    var xfn1s = modulate(
        xfn1, _t(mv.scale_mlp.copy(), [D], ctx), _t(zeros.copy(), [D], ctx), ctx
    )

    var no_bias_g = Optional[Tensor](None)
    var g_base = linear(xfn1s, w.w1[], no_bias_g^, ctx)
    var no_bias_u = Optional[Tensor](None)
    var u_base = linear(xfn1s, w.w3[], no_bias_u^, ctx)
    var g_pre = zimage_lora_apply_device(g_base^, xfn1s, lora.w1, S, ctx)
    var u = zimage_lora_apply_device(u_base^, xfn1s, lora.w3, S, ctx)

    var act = swiglu(g_pre, u, ctx)

    var no_bias_d = Optional[Tensor](None)
    var ff_base = linear(act, w.w2[], no_bias_d^, ctx)
    var ff = zimage_lora_apply_device(ff_base^, act, lora.w2, S, ctx)

    var ff_n2 = rms_norm(ff, w.fn2[], eps, ctx)
    var gate_mlp_raw = _t(mv.gate_mlp.copy(), [D], ctx)
    var gate_mlp_t = tanh_op(gate_mlp_raw, ctx)
    var result = residual_gate(h, gate_mlp_t, ff_n2, ctx)

    var saved = ZImageBlockSaved(
        x_arc.copy(), TArc(xn1^), TArc(xn1s^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rms^), TArc(k_rms^), TArc(q_rope^), TArc(k_rope^),
        TArc(att_flat^), TArc(att_o^),
        TArc(gate_msa_t^), TArc(gate_msa_raw^), TArc(h^),
        TArc(xfn1^), TArc(xfn1s^),
        TArc(g_pre^), TArc(u^), TArc(act^), TArc(ff^),
        TArc(gate_mlp_t^), TArc(gate_mlp_raw^),
    )
    return ZImageBlockForwardLoraTensor(TArc(result^), saved^)


def zimage_block_lora_predict_device_tensor[
    H: Int, Dh: Int, S: Int
](
    x_arc: TArc,
    w: ZImageBlockWeights, mv: ZImageModVecs, lora: ZImageBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> TArc:
    var mv_dev = zimage_modvecs_to_device(mv, D, ctx)
    return zimage_block_lora_predict_device_tensor_moddev[H, Dh, S](
        x_arc, w, mv_dev, lora, cos, sin, D, F, eps, ctx,
    )


def zimage_block_lora_predict_device_tensor_moddev[
    H: Int, Dh: Int, S: Int
](
    x_arc: TArc,
    w: ZImageBlockWeights, mv: ZImageModVecsDevice, lora: ZImageBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> TArc:
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var xn1 = rms_norm(x_arc[], w.n1[], eps, ctx)
    var xn1s = vec_modulate(
        xn1, mv.scale_msa[], mv.zeros[], ctx
    )

    var no_bias = Optional[Tensor](None)
    var q_base = linear(xn1s, w.wq[], no_bias^, ctx)
    var no_bias_k = Optional[Tensor](None)
    var k_base = linear(xn1s, w.wk[], no_bias_k^, ctx)
    var no_bias_v = Optional[Tensor](None)
    var v_base = linear(xn1s, w.wv[], no_bias_v^, ctx)

    var q = zimage_lora_apply_device(q_base^, xn1s, lora.to_q, S, ctx)
    var k = zimage_lora_apply_device(k_base^, xn1s, lora.to_k, S, ctx)
    var v_flat = zimage_lora_apply_device(v_base^, xn1s, lora.to_v, S, ctx)

    var q_pre = reshape_owned(q^, [1, S, H, Dh])
    var k_pre = reshape_owned(k^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)

    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])

    var no_bias_o = Optional[Tensor](None)
    var att_o_base = linear(att_flat, w.wo[], no_bias_o^, ctx)
    var att_o = zimage_lora_apply_device(att_o_base^, att_flat, lora.to_out, S, ctx)

    var attn_n2 = rms_norm(att_o, w.n2[], eps, ctx)
    var gate_msa_t = tanh_op(mv.gate_msa[], ctx)
    var h = residual_gate(x_arc[], gate_msa_t, attn_n2, ctx)

    var xfn1 = rms_norm(h, w.fn1[], eps, ctx)
    var xfn1s = vec_modulate(
        xfn1, mv.scale_mlp[], mv.zeros[], ctx
    )

    var no_bias_g = Optional[Tensor](None)
    var g_base = linear(xfn1s, w.w1[], no_bias_g^, ctx)
    var no_bias_u = Optional[Tensor](None)
    var u_base = linear(xfn1s, w.w3[], no_bias_u^, ctx)
    var g_pre = zimage_lora_apply_device(g_base^, xfn1s, lora.w1, S, ctx)
    var u = zimage_lora_apply_device(u_base^, xfn1s, lora.w3, S, ctx)

    var act = vec_swiglu(g_pre, u, ctx)

    var no_bias_d = Optional[Tensor](None)
    var ff_base = linear(act, w.w2[], no_bias_d^, ctx)
    var ff = zimage_lora_apply_device(ff_base^, act, lora.w2, S, ctx)

    var ff_n2 = rms_norm(ff, w.fn2[], eps, ctx)
    var gate_mlp_t = tanh_op(mv.gate_mlp[], ctx)
    var result = residual_gate(h, gate_mlp_t, ff_n2, ctx)
    return TArc(result^)


def zimage_block_lora_backward_device[
    H: Int, Dh: Int, S: Int
](
    d_out: List[Float32],
    w: ZImageBlockWeights, mv: ZImageModVecs, lora: ZImageBlockLoraDevice,
    saved: ZImageBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageBlockLoraBackward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var d_out_t = _t(d_out, [S, D], ctx)

    var d_a_slots = List[List[Float32]]()
    var d_b_slots = List[List[Float32]]()
    for _ in range(ZIMAGE_SLOTS):
        d_a_slots.append(List[Float32]())
        d_b_slots.append(List[Float32]())

    var ff_n2_y = rms_norm(saved.ff[], w.fn2[], eps, ctx)
    var grg2 = gate_residual_backward(
        d_out_t, saved.h[], saved.gate_mlp_t[], ff_n2_y, ctx
    )
    var d_gate_mlp = tanh_backward(grg2.d_g, saved.gate_mlp_raw[], ctx).to_host(ctx)

    var rb_fn2 = rms_norm_backward(grg2.d_y, saved.ff[], w.fn2[], eps, ctx)
    var d_fn2 = rb_fn2.d_g.to_host(ctx)

    var lb_w2 = _proj_bwd_with_lora_device(
        rb_fn2.d_x, saved.act[], w.w2[], lora.w2, SLOT_W2,
        S, F, D, d_a_slots, d_b_slots, ctx,
    )
    var d_w2 = List[Float32]()

    var sg = swiglu_backward(lb_w2.d_x, saved.g_pre[], saved.u[], ctx)

    var lb_w1 = _proj_bwd_with_lora_device(
        sg.d_gate, saved.xfn1s[], w.w1[], lora.w1, SLOT_W1,
        S, D, F, d_a_slots, d_b_slots, ctx,
    )
    var d_w1 = List[Float32]()
    var lb_w3 = _proj_bwd_with_lora_device(
        sg.d_up, saved.xfn1s[], w.w3[], lora.w3, SLOT_W3,
        S, D, F, d_a_slots, d_b_slots, ctx,
    )
    var d_w3 = List[Float32]()
    var d_xfn1s = add(lb_w1.d_x, lb_w3.d_x, ctx)

    var mb_mlp = modulate_backward(d_xfn1s, saved.xfn1[], _t(mv.scale_mlp.copy(), [D], ctx), ctx)
    var d_scale_mlp = mb_mlp.d_scale.to_host(ctx)
    var rb_fn1 = rms_norm_backward(mb_mlp.d_x, saved.h[], w.fn1[], eps, ctx)
    var d_fn1 = rb_fn1.d_g.to_host(ctx)
    var d_h = add(grg2.d_x, rb_fn1.d_x, ctx)

    var attn_n2_y = rms_norm(saved.att_o[], w.n2[], eps, ctx)
    var grg1 = gate_residual_backward(
        d_h, saved.x[], saved.gate_msa_t[], attn_n2_y, ctx
    )
    var d_gate_msa = tanh_backward(grg1.d_g, saved.gate_msa_raw[], ctx).to_host(ctx)

    var rb_n2 = rms_norm_backward(grg1.d_y, saved.att_o[], w.n2[], eps, ctx)
    var d_n2 = rb_n2.d_g.to_host(ctx)

    var lb_o = _proj_bwd_with_lora_device(
        rb_n2.d_x, saved.att_flat[], w.wo[], lora.to_out, SLOT_O,
        S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wo = List[Float32]()

    reshape_in_place(lb_o.d_x, [1, S, H, Dh])
    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v[], lb_o.d_x, scale, ctx
    )
    var d_q_rms = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_rms = rope_backward(sb.d_k, cos, sin, True, ctx)

    var rb_q = rms_norm_backward(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)
    var d_k_norm = rb_k.d_g.to_host(ctx)

    reshape_in_place(rb_q.d_x, [S, D])
    reshape_in_place(rb_k.d_x, [S, D])
    reshape_in_place(sb.d_v, [S, D])

    var lb_q = _proj_bwd_with_lora_device(
        rb_q.d_x, saved.xn1s[], w.wq[], lora.to_q, SLOT_Q,
        S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wq = List[Float32]()
    var lb_k = _proj_bwd_with_lora_device(
        rb_k.d_x, saved.xn1s[], w.wk[], lora.to_k, SLOT_K,
        S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wk = List[Float32]()
    var lb_v = _proj_bwd_with_lora_device(
        sb.d_v, saved.xn1s[], w.wv[], lora.to_v, SLOT_V,
        S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wv = List[Float32]()
    var d_xn1s = add(add(lb_q.d_x, lb_k.d_x, ctx), lb_v.d_x, ctx)

    var mb_sa = modulate_backward(d_xn1s, saved.xn1[], _t(mv.scale_msa.copy(), [D], ctx), ctx)
    var d_scale_msa = mb_sa.d_scale.to_host(ctx)
    var rb_n1 = rms_norm_backward(mb_sa.d_x, saved.x[], w.n1[], eps, ctx)
    var d_n1 = rb_n1.d_g.to_host(ctx)

    var d_x_res = grg1.d_x.to_host(ctx)
    var d_x_norm = rb_n1.d_x.to_host(ctx)
    var d_x = _add_lists(d_x_res, d_x_norm)

    var base = ZImageBlockGrads(
        d_x^, d_n1^,
        d_wq^, d_wk^, d_wv^, d_wo^,
        d_q_norm^, d_k_norm^,
        d_n2^, d_fn1^,
        d_w1^, d_w3^, d_w2^, d_fn2^,
        d_scale_msa^, d_gate_msa^,
        d_scale_mlp^, d_gate_mlp^,
    )
    return ZImageBlockLoraBackward(base^, ZImageBlockLoraGrads(d_a_slots^, d_b_slots^))


struct ZImageBlockLoraTensorBackward(Copyable, Movable):
    var d_x: TArc
    var d_a: List[TArc]
    var d_b: List[TArc]

    def __init__(out self, var d_x: TArc, var d_a: List[TArc], var d_b: List[TArc]):
        self.d_x = d_x^
        self.d_a = d_a^
        self.d_b = d_b^


def zimage_block_lora_backward_device_tensors[
    H: Int, Dh: Int, S: Int
](
    d_out: Tensor,
    w: ZImageBlockWeights, mv: ZImageModVecs, lora: ZImageBlockLoraDevice,
    saved: ZImageBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
    trace: Bool = False,
) raises -> ZImageBlockLoraTensorBackward:
    var ts0 = perf_counter_ns()
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var ff_n2_y = rms_norm(saved.ff[], w.fn2[], eps, ctx)
    var grg2 = gate_residual_backward(
        d_out, saved.h[], saved.gate_mlp_t[], ff_n2_y, ctx
    )

    var d_ff = rms_norm_backward_dx(grg2.d_y, saved.ff[], w.fn2[], eps, ctx)
    var lb_w2 = _proj_bwd_with_lora_device_tensors(
        d_ff, saved.act[], w.w2[], lora.w2, S, F, D, ctx,
    )
    if trace:
        ctx.synchronize()
    var ts_w2 = perf_counter_ns()

    var sg = swiglu_backward(lb_w2.d_x, saved.g_pre[], saved.u[], ctx)

    var lb_w1 = _proj_bwd_with_lora_device_tensors(
        sg.d_gate, saved.xfn1s[], w.w1[], lora.w1, S, D, F, ctx,
    )
    var lb_w3 = _proj_bwd_with_lora_device_tensors(
        sg.d_up, saved.xfn1s[], w.w3[], lora.w3, S, D, F, ctx,
    )
    var d_xfn1s = add(lb_w1.d_x, lb_w3.d_x, ctx)
    if trace:
        ctx.synchronize()
    var ts_w13 = perf_counter_ns()

    var mb_mlp = modulate_backward(d_xfn1s, saved.xfn1[], _t(mv.scale_mlp.copy(), [D], ctx), ctx)
    var d_h_norm = rms_norm_backward_dx(mb_mlp.d_x, saved.h[], w.fn1[], eps, ctx)
    var d_h = add(grg2.d_x, d_h_norm, ctx)

    var attn_n2_y = rms_norm(saved.att_o[], w.n2[], eps, ctx)
    var grg1 = gate_residual_backward(
        d_h, saved.x[], saved.gate_msa_t[], attn_n2_y, ctx
    )

    var d_att_o = rms_norm_backward_dx(grg1.d_y, saved.att_o[], w.n2[], eps, ctx)
    var lb_o = _proj_bwd_with_lora_device_tensors(
        d_att_o, saved.att_flat[], w.wo[], lora.to_out, S, D, D, ctx,
    )
    if trace:
        ctx.synchronize()
    var ts_o = perf_counter_ns()

    reshape_in_place(lb_o.d_x, [1, S, H, Dh])
    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v[], lb_o.d_x, scale, ctx
    )
    var d_q_rms = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_rms = rope_backward(sb.d_k, cos, sin, True, ctx)

    var d_q_pre = rms_norm_backward_dx(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_k_pre = rms_norm_backward_dx(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)
    if trace:
        ctx.synchronize()
    var ts_attn = perf_counter_ns()

    reshape_in_place(d_q_pre, [S, D])
    reshape_in_place(d_k_pre, [S, D])
    reshape_in_place(sb.d_v, [S, D])

    var lb_q = _proj_bwd_with_lora_device_tensors(
        d_q_pre, saved.xn1s[], w.wq[], lora.to_q, S, D, D, ctx,
    )
    var lb_k = _proj_bwd_with_lora_device_tensors(
        d_k_pre, saved.xn1s[], w.wk[], lora.to_k, S, D, D, ctx,
    )
    var lb_v = _proj_bwd_with_lora_device_tensors(
        sb.d_v, saved.xn1s[], w.wv[], lora.to_v, S, D, D, ctx,
    )
    var d_xn1s = add(add(lb_q.d_x, lb_k.d_x, ctx), lb_v.d_x, ctx)
    if trace:
        ctx.synchronize()
    var ts_qkv = perf_counter_ns()

    var mb_sa = modulate_backward(d_xn1s, saved.xn1[], _t(mv.scale_msa.copy(), [D], ctx), ctx)
    var d_x_norm = rms_norm_backward_dx(mb_sa.d_x, saved.x[], w.n1[], eps, ctx)

    var d_x_t = add(grg1.d_x, d_x_norm, ctx)
    var ts_end = perf_counter_ns()
    if trace:
        print("[BWD_DETAIL] w2=", Float32(Float64(ts_w2 - ts0) / 1.0e9),
              " w1w3=", Float32(Float64(ts_w13 - ts_w2) / 1.0e9),
              " out=", Float32(Float64(ts_o - ts_w13) / 1.0e9),
              " sdpa_qk=", Float32(Float64(ts_attn - ts_o) / 1.0e9),
              " qkv=", Float32(Float64(ts_qkv - ts_attn) / 1.0e9),
              " tail=", Float32(Float64(ts_end - ts_qkv) / 1.0e9))

    var d_a_slots = List[TArc]()
    d_a_slots.append(lb_q.d_a.copy())
    d_a_slots.append(lb_k.d_a.copy())
    d_a_slots.append(lb_v.d_a.copy())
    d_a_slots.append(lb_o.d_a.copy())
    d_a_slots.append(lb_w1.d_a.copy())
    d_a_slots.append(lb_w3.d_a.copy())
    d_a_slots.append(lb_w2.d_a.copy())
    var d_b_slots = List[TArc]()
    d_b_slots.append(lb_q.d_b.copy())
    d_b_slots.append(lb_k.d_b.copy())
    d_b_slots.append(lb_v.d_b.copy())
    d_b_slots.append(lb_o.d_b.copy())
    d_b_slots.append(lb_w1.d_b.copy())
    d_b_slots.append(lb_w3.d_b.copy())
    d_b_slots.append(lb_w2.d_b.copy())

    return ZImageBlockLoraTensorBackward(TArc(d_x_t^), d_a_slots^, d_b_slots^)


# ══════════════════════════════════════════════════════════════════════════════
# UNMODULATED block (context refiners) — LoRA-aware fwd + bwd.
# Mirrors zimage_refiner_forward/_backward EXACTLY (models/zimage/block.mojo). Same
# 7 LoRA target projections (attention + feed_forward Linears), no modulation/gate.
# ══════════════════════════════════════════════════════════════════════════════
struct ZImageRefinerForwardLora(Movable):
    var out: List[Float32]
    var saved: ZImageRefinerSaved

    def __init__(out self, var out: List[Float32], var saved: ZImageRefinerSaved):
        self.out = out^
        self.saved = saved^


def zimage_refiner_lora_forward[
    H: Int, Dh: Int, S: Int
](
    x: List[Float32],
    w: ZImageBlockWeights, lora: ZImageBlockLora,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageRefinerForwardLora:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var x_t = _t(x, [S, D], ctx)

    # --- attention sub-block (sandwich norm, NO modulation) ---
    var xn1 = rms_norm(x_t, w.n1[], eps, ctx)
    var xn1_h = xn1.to_host(ctx)                                # [S,D] (LoRA input for q/k/v)

    var no_bias = Optional[Tensor](None)
    var q_base = linear(xn1, w.wq[], no_bias^, ctx).to_host(ctx)
    var no_bias_k = Optional[Tensor](None)
    var k_base = linear(xn1, w.wk[], no_bias_k^, ctx).to_host(ctx)
    var no_bias_v = Optional[Tensor](None)
    var v_base = linear(xn1, w.wv[], no_bias_v^, ctx).to_host(ctx)

    var q_h = zimage_lora_apply(q_base, xn1_h, lora.to_q, S, ctx)
    var k_h = zimage_lora_apply(k_base, xn1_h, lora.to_k, S, ctx)
    var v_h = zimage_lora_apply(v_base, xn1_h, lora.to_v, S, ctx)

    var q_pre = reshape_owned(_t(q_h, [S, D], ctx)^, [1, S, H, Dh])
    var k_pre = reshape_owned(_t(k_h, [S, D], ctx)^, [1, S, H, Dh])
    var v = reshape_owned(_t(v_h, [S, D], ctx)^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)

    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])
    var att_flat_h = att_flat.to_host(ctx)                      # [S,D] (LoRA input for wo)

    var no_bias_o = Optional[Tensor](None)
    var att_o_base = linear(att_flat, w.wo[], no_bias_o^, ctx).to_host(ctx)
    var att_o_h = zimage_lora_apply(att_o_base, att_flat_h, lora.to_out, S, ctx)
    var att_o = _t(att_o_h, [S, D], ctx)

    var attn_n2 = rms_norm(att_o, w.n2[], eps, ctx)
    var h = add(x_t, attn_n2, ctx)                  # PLAIN residual (no gate)

    # --- MLP sub-block (SwiGLU, sandwich norm, NO modulation) ---
    var xfn1 = rms_norm(h, w.fn1[], eps, ctx)
    var xfn1_h = xfn1.to_host(ctx)                              # [S,D] (LoRA input for w1/w3)

    var no_bias_g = Optional[Tensor](None)
    var g_base = linear(xfn1, w.w1[], no_bias_g^, ctx).to_host(ctx)
    var no_bias_u = Optional[Tensor](None)
    var u_base = linear(xfn1, w.w3[], no_bias_u^, ctx).to_host(ctx)
    var g_pre_h = zimage_lora_apply(g_base, xfn1_h, lora.w1, S, ctx)
    var u_h = zimage_lora_apply(u_base, xfn1_h, lora.w3, S, ctx)
    var g_pre = _t(g_pre_h, [S, F], ctx)
    var u = _t(u_h, [S, F], ctx)

    var act = swiglu(g_pre, u, ctx)
    var act_h = act.to_host(ctx)                                # [S,F] (LoRA input for w2)

    var no_bias_d = Optional[Tensor](None)
    var ff_base = linear(act, w.w2[], no_bias_d^, ctx).to_host(ctx)
    var ff_h = zimage_lora_apply(ff_base, act_h, lora.w2, S, ctx)
    var ff = _t(ff_h, [S, D], ctx)

    var ff_n2 = rms_norm(ff, w.fn2[], eps, ctx)
    var result = add(h, ff_n2, ctx).to_host(ctx)    # PLAIN residual (no gate)

    var saved = ZImageRefinerSaved(
        TArc(x_t^), TArc(xn1^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rms^), TArc(k_rms^), TArc(q_rope^), TArc(k_rope^),
        TArc(att_flat^), TArc(att_o^), TArc(h^),
        TArc(xfn1^), TArc(g_pre^), TArc(u^), TArc(act^), TArc(ff^),
    )
    return ZImageRefinerForwardLora(result^, saved^)


struct ZImageRefinerLoraBackward(Movable):
    var base: ZImageRefinerGrads
    var lora: ZImageBlockLoraGrads

    def __init__(out self, var base: ZImageRefinerGrads, var lora: ZImageBlockLoraGrads):
        self.base = base^
        self.lora = lora^


def zimage_refiner_lora_backward[
    H: Int, Dh: Int, S: Int
](
    d_out: List[Float32],
    w: ZImageBlockWeights, lora: ZImageBlockLora, saved: ZImageRefinerSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageRefinerLoraBackward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var d_out_t = _t(d_out, [S, D], ctx)

    var d_a_slots = List[List[Float32]]()
    var d_b_slots = List[List[Float32]]()
    for _ in range(ZIMAGE_SLOTS):
        d_a_slots.append(List[Float32]())
        d_b_slots.append(List[Float32]())

    # out = h + ff_n2 (plain residual); ff_n2 = rms_norm(ff, fn2)
    var rb_fn2 = rms_norm_backward(d_out_t, saved.ff[], w.fn2[], eps, ctx)
    var d_fn2 = rb_fn2.d_g.to_host(ctx)

    # ff = linear(act, w2)[+LoRA(w2)]  W [D, F]
    var act_h = saved.act[].to_host(ctx)
    var lb_w2 = _proj_bwd_with_lora(
        rb_fn2.d_x, saved.act[], w.w2[], act_h,
        lora.w2, SLOT_W2, S, F, D, d_a_slots, d_b_slots, ctx,
    )
    var d_w2 = List[Float32]()

    var sg = swiglu_backward(lb_w2.d_x, saved.g_pre[], saved.u[], ctx)

    # g_pre = linear(xfn1, w1)[+LoRA] ; u = linear(xfn1, w3)[+LoRA]  W [F, D]
    var xfn1_h = saved.xfn1[].to_host(ctx)
    var lb_w1 = _proj_bwd_with_lora(
        sg.d_gate, saved.xfn1[], w.w1[], xfn1_h,
        lora.w1, SLOT_W1, S, D, F, d_a_slots, d_b_slots, ctx,
    )
    var d_w1 = List[Float32]()
    var lb_w3 = _proj_bwd_with_lora(
        sg.d_up, saved.xfn1[], w.w3[], xfn1_h,
        lora.w3, SLOT_W3, S, D, F, d_a_slots, d_b_slots, ctx,
    )
    var d_w3 = List[Float32]()
    var d_xfn1 = add(lb_w1.d_x, lb_w3.d_x, ctx)

    # xfn1 = rms_norm(h, fn1)  (no modulation between)
    var rb_fn1 = rms_norm_backward(d_xfn1, saved.h[], w.fn1[], eps, ctx)
    var d_fn1 = rb_fn1.d_g.to_host(ctx)
    var d_h = add(d_out_t, rb_fn1.d_x, ctx)

    # --- attention sub-block backward ---
    # h = x + attn_n2 (plain residual); attn_n2 = rms_norm(att_o, n2)
    var rb_n2 = rms_norm_backward(d_h, saved.att_o[], w.n2[], eps, ctx)
    var d_n2 = rb_n2.d_g.to_host(ctx)

    # att_o = linear(att_flat, wo)[+LoRA(to_out)]  W [D, D]
    var att_flat_h = saved.att_flat[].to_host(ctx)
    var lb_o = _proj_bwd_with_lora(
        rb_n2.d_x, saved.att_flat[], w.wo[], att_flat_h,
        lora.to_out, SLOT_O, S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wo = List[Float32]()

    reshape_in_place(lb_o.d_x, [1, S, H, Dh])
    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v[], lb_o.d_x, scale, ctx
    )
    var d_q_rms = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_rms = rope_backward(sb.d_k, cos, sin, True, ctx)

    var rb_q = rms_norm_backward(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var rb_k = rms_norm_backward(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)
    var d_k_norm = rb_k.d_g.to_host(ctx)

    reshape_in_place(rb_q.d_x, [S, D])
    reshape_in_place(rb_k.d_x, [S, D])
    reshape_in_place(sb.d_v, [S, D])

    # q/k/v = linear(xn1, w{q,k,v})[+LoRA]  W [D, D]; xn1 feeds all three.
    var xn1_h = saved.xn1[].to_host(ctx)
    var lb_q = _proj_bwd_with_lora(
        rb_q.d_x, saved.xn1[], w.wq[], xn1_h,
        lora.to_q, SLOT_Q, S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wq = List[Float32]()
    var lb_k = _proj_bwd_with_lora(
        rb_k.d_x, saved.xn1[], w.wk[], xn1_h,
        lora.to_k, SLOT_K, S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wk = List[Float32]()
    var lb_v = _proj_bwd_with_lora(
        sb.d_v, saved.xn1[], w.wv[], xn1_h,
        lora.to_v, SLOT_V, S, D, D, d_a_slots, d_b_slots, ctx,
    )
    var d_wv = List[Float32]()
    var d_xn1 = add(add(lb_q.d_x, lb_k.d_x, ctx), lb_v.d_x, ctx)

    # xn1 = rms_norm(x, n1)  (no modulation between)
    var rb_n1 = rms_norm_backward(d_xn1, saved.x[], w.n1[], eps, ctx)
    var d_n1 = rb_n1.d_g.to_host(ctx)

    var d_x_res = d_h.to_host(ctx)
    var d_x_norm = rb_n1.d_x.to_host(ctx)
    var d_x = _add_lists(d_x_res, d_x_norm)

    var base = ZImageRefinerGrads(
        d_x^, d_n1^,
        d_wq^, d_wk^, d_wv^, d_wo^,
        d_q_norm^, d_k_norm^,
        d_n2^, d_fn1^,
        d_w1^, d_w3^, d_w2^, d_fn2^,
    )
    return ZImageRefinerLoraBackward(base^, ZImageBlockLoraGrads(d_a_slots^, d_b_slots^))


# ══════════════════════════════════════════════════════════════════════════════
# BATCH-B (stacked-rows) modulated block — fwd + bwd for the batch-2 trainer.
# x is [B*S, D] (B samples stacked along rows). Per-sample adaLN: the modvec
# tensors in `mv` are [B, D] (one row per sample) — modulate/residual_gate/
# modulate_backward/gate_residual_backward_dxdy apply row-range-wise (ops
# extended 2026-06-11). cos/sin are the TILED per-sample tables
# [B*S*H?, ...] matching rope_interleaved's [rows, Dh/2] contract for
# x reshaped [B, S, H, Dh]. Attention uses sdpa_nomask[B, S, H, Dh] — no
# cross-sample attention. Same math per sample as the B=1 functions.
# GATE: zimage_batch2_parity smoke (B=2 vs 2× B=1, identical draws).
# ══════════════════════════════════════════════════════════════════════════════
def zimage_modvecs_pack2_to_device(
    mv0: ZImageModVecs, mv1: ZImageModVecs, D: Int, ctx: DeviceContext
) raises -> ZImageModVecsDevice:
    """Pack two per-sample modvec sets into [2, D] device tensors."""
    var sm = mv0.scale_msa.copy()
    for i in range(D):
        sm.append(mv1.scale_msa[i])
    var gm = mv0.gate_msa.copy()
    for i in range(D):
        gm.append(mv1.gate_msa[i])
    var sp = mv0.scale_mlp.copy()
    for i in range(D):
        sp.append(mv1.scale_mlp[i])
    var gp = mv0.gate_mlp.copy()
    for i in range(D):
        gp.append(mv1.gate_mlp[i])
    return ZImageModVecsDevice(
        TArc(_t(sm^, [2, D], ctx)),
        TArc(_t(gm^, [2, D], ctx)),
        TArc(_t(sp^, [2, D], ctx)),
        TArc(_t(gp^, [2, D], ctx)),
        TArc(_t(_zeros(2 * D), [2, D], ctx)),
    )


def zimage_block_lora_forward_device_tensor_batch[
    B: Int, H: Int, Dh: Int, S: Int
](
    x_arc: TArc,
    w: ZImageBlockWeights, mv: ZImageModVecsDevice, lora: ZImageBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageBlockForwardLoraTensor:
    comptime ROWS = B * S
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var xn1 = rms_norm(x_arc[], w.n1[], eps, ctx)
    var xn1s = modulate(xn1, mv.scale_msa[], mv.zeros[], ctx)

    var no_bias = Optional[Tensor](None)
    var q_base = linear(xn1s, w.wq[], no_bias^, ctx)
    var no_bias_k = Optional[Tensor](None)
    var k_base = linear(xn1s, w.wk[], no_bias_k^, ctx)
    var no_bias_v = Optional[Tensor](None)
    var v_base = linear(xn1s, w.wv[], no_bias_v^, ctx)

    var q = zimage_lora_apply_device(q_base^, xn1s, lora.to_q, ROWS, ctx)
    var k = zimage_lora_apply_device(k_base^, xn1s, lora.to_k, ROWS, ctx)
    var v_flat = zimage_lora_apply_device(v_base^, xn1s, lora.to_v, ROWS, ctx)

    var q_pre = reshape_owned(q^, [B, S, H, Dh])
    var k_pre = reshape_owned(k^, [B, S, H, Dh])
    var v = reshape_owned(v_flat^, [B, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)
    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)

    var att = sdpa_nomask[B, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [ROWS, D])

    var no_bias_o = Optional[Tensor](None)
    var att_o_base = linear(att_flat, w.wo[], no_bias_o^, ctx)
    var att_o = zimage_lora_apply_device(att_o_base^, att_flat, lora.to_out, ROWS, ctx)

    var attn_n2 = rms_norm(att_o, w.n2[], eps, ctx)
    # v2 engine: tanh straight off the shared device gate vec; the saved tape
    # holds a TArc refcount copy instead of a clone (clone = d2d + sync; the
    # gate is read-only in backward — predict_moddev precedent at tanh_op
    # call above zimage_block_lora_predict_device_tensor_moddev).
    var gate_msa_t = tanh_op(mv.gate_msa[], ctx)
    var h = residual_gate(x_arc[], gate_msa_t, attn_n2, ctx)

    var xfn1 = rms_norm(h, w.fn1[], eps, ctx)
    var xfn1s = modulate(xfn1, mv.scale_mlp[], mv.zeros[], ctx)

    var no_bias_g = Optional[Tensor](None)
    var g_base = linear(xfn1s, w.w1[], no_bias_g^, ctx)
    var no_bias_u = Optional[Tensor](None)
    var u_base = linear(xfn1s, w.w3[], no_bias_u^, ctx)
    var g_pre = zimage_lora_apply_device(g_base^, xfn1s, lora.w1, ROWS, ctx)
    var u = zimage_lora_apply_device(u_base^, xfn1s, lora.w3, ROWS, ctx)

    var act = swiglu(g_pre, u, ctx)

    var no_bias_d = Optional[Tensor](None)
    var ff_base = linear(act, w.w2[], no_bias_d^, ctx)
    var ff = zimage_lora_apply_device(ff_base^, act, lora.w2, ROWS, ctx)

    var ff_n2 = rms_norm(ff, w.fn2[], eps, ctx)
    var gate_mlp_t = tanh_op(mv.gate_mlp[], ctx)
    var result = residual_gate(h, gate_mlp_t, ff_n2, ctx)

    var saved = ZImageBlockSaved(
        x_arc.copy(), TArc(xn1^), TArc(xn1s^),
        TArc(q_pre^), TArc(k_pre^), TArc(v^),
        TArc(q_rms^), TArc(k_rms^), TArc(q_rope^), TArc(k_rope^),
        TArc(att_flat^), TArc(att_o^),
        TArc(gate_msa_t^), mv.gate_msa.copy(), TArc(h^),
        TArc(xfn1^), TArc(xfn1s^),
        TArc(g_pre^), TArc(u^), TArc(act^), TArc(ff^),
        TArc(gate_mlp_t^), mv.gate_mlp.copy(),
    )
    return ZImageBlockForwardLoraTensor(TArc(result^), saved^)


def zimage_block_lora_backward_device_tensors_batch[
    B: Int, H: Int, Dh: Int, S: Int
](
    d_out: Tensor,
    w: ZImageBlockWeights, mv: ZImageModVecsDevice, lora: ZImageBlockLoraDevice,
    saved: ZImageBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageBlockLoraTensorBackward:
    comptime ROWS = B * S
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var grg2 = gate_residual_backward_dxdy(d_out, saved.gate_mlp_t[], ctx)

    var d_ff = rms_norm_backward_dx(grg2.d_y, saved.ff[], w.fn2[], eps, ctx)
    var lb_w2 = _proj_bwd_with_lora_device_tensors(
        d_ff, saved.act[], w.w2[], lora.w2, ROWS, F, D, ctx,
    )

    var sg = swiglu_backward(lb_w2.d_x, saved.g_pre[], saved.u[], ctx)

    var lb_w1 = _proj_bwd_with_lora_device_tensors(
        sg.d_gate, saved.xfn1s[], w.w1[], lora.w1, ROWS, D, F, ctx,
    )
    var lb_w3 = _proj_bwd_with_lora_device_tensors(
        sg.d_up, saved.xfn1s[], w.w3[], lora.w3, ROWS, D, F, ctx,
    )
    var d_xfn1s = add(lb_w1.d_x, lb_w3.d_x, ctx)

    var mb_mlp = modulate_backward(
        d_xfn1s, saved.xfn1[], mv.scale_mlp[], ctx, compute_param_grads=False,
    )
    var d_h_norm = rms_norm_backward_dx(mb_mlp.d_x, saved.h[], w.fn1[], eps, ctx)
    var d_h = add(grg2.d_x, d_h_norm, ctx)

    var grg1 = gate_residual_backward_dxdy(d_h, saved.gate_msa_t[], ctx)

    var d_att_o = rms_norm_backward_dx(grg1.d_y, saved.att_o[], w.n2[], eps, ctx)
    var lb_o = _proj_bwd_with_lora_device_tensors(
        d_att_o, saved.att_flat[], w.wo[], lora.to_out, ROWS, D, D, ctx,
    )

    reshape_in_place(lb_o.d_x, [B, S, H, Dh])
    var sb = sdpa_backward[B, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v[], lb_o.d_x, scale, ctx
    )
    var d_q_rms = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_rms = rope_backward(sb.d_k, cos, sin, True, ctx)

    var d_q_pre = rms_norm_backward_dx(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_k_pre = rms_norm_backward_dx(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)

    reshape_in_place(d_q_pre, [ROWS, D])
    reshape_in_place(d_k_pre, [ROWS, D])
    reshape_in_place(sb.d_v, [ROWS, D])

    var lb_q = _proj_bwd_with_lora_device_tensors(
        d_q_pre, saved.xn1s[], w.wq[], lora.to_q, ROWS, D, D, ctx,
    )
    var lb_k = _proj_bwd_with_lora_device_tensors(
        d_k_pre, saved.xn1s[], w.wk[], lora.to_k, ROWS, D, D, ctx,
    )
    var lb_v = _proj_bwd_with_lora_device_tensors(
        sb.d_v, saved.xn1s[], w.wv[], lora.to_v, ROWS, D, D, ctx,
    )
    var d_xn1s = add(add(lb_q.d_x, lb_k.d_x, ctx), lb_v.d_x, ctx)

    var mb_sa = modulate_backward(
        d_xn1s, saved.xn1[], mv.scale_msa[], ctx, compute_param_grads=False,
    )
    var d_x_norm = rms_norm_backward_dx(mb_sa.d_x, saved.x[], w.n1[], eps, ctx)

    var d_x_t = add(grg1.d_x, d_x_norm, ctx)

    var d_a_slots = List[TArc]()
    d_a_slots.append(lb_q.d_a.copy())
    d_a_slots.append(lb_k.d_a.copy())
    d_a_slots.append(lb_v.d_a.copy())
    d_a_slots.append(lb_o.d_a.copy())
    d_a_slots.append(lb_w1.d_a.copy())
    d_a_slots.append(lb_w3.d_a.copy())
    d_a_slots.append(lb_w2.d_a.copy())
    var d_b_slots = List[TArc]()
    d_b_slots.append(lb_q.d_b.copy())
    d_b_slots.append(lb_k.d_b.copy())
    d_b_slots.append(lb_v.d_b.copy())
    d_b_slots.append(lb_o.d_b.copy())
    d_b_slots.append(lb_w1.d_b.copy())
    d_b_slots.append(lb_w3.d_b.copy())
    d_b_slots.append(lb_w2.d_b.copy())

    return ZImageBlockLoraTensorBackward(TArc(d_x_t^), d_a_slots^, d_b_slots^)


# ══════════════════════════════════════════════════════════════════════════════
# Phase D.1 (MOJO_V2_ENGINE_PLAN.md): FROZEN refiner blocks → device-tensor
# forwards. These are the EXACT op sequences of block.zimage_block_forward
# (modulated, NR) and block.zimage_refiner_forward (unmodulated, CR) with only
# WHERE the tensors live changed: device in/out (no host-List round trip), the
# per-call `_t()` modvec/zeros uploads replaced by ZImageModVecsDevice views
# (same F32 bytes — packed slab, zimage_modvecs_all_to_device), and NO saved
# tape (frozen blocks: the main-only backward never touches them) and NO
# to_host (the caller chains device tensors). Same ops, same order, same
# values → bit-identical (C14). Placed HERE, not block.mojo, because these
# consume ZImageModVecsDevice and block.mojo cannot import lora_block
# (lora_block imports block — circular). Old paths untouched (C13).
def zimage_block_forward_device_moddev[
    H: Int, Dh: Int, S: Int
](
    x_arc: TArc,
    w: ZImageBlockWeights, mv: ZImageModVecsDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> TArc:
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    # --- attention sub-block (sandwich norm) ---
    var xn1 = rms_norm(x_arc[], w.n1[], eps, ctx)
    var xn1s = modulate(xn1, mv.scale_msa[], mv.zeros[], ctx)

    var no_bias = Optional[Tensor](None)
    var q_flat = linear(xn1s, w.wq[], no_bias^, ctx)            # [S,D]
    var no_bias_k = Optional[Tensor](None)
    var k_flat = linear(xn1s, w.wk[], no_bias_k^, ctx)
    var no_bias_v = Optional[Tensor](None)
    var v_flat = linear(xn1s, w.wv[], no_bias_v^, ctx)

    var q_pre = reshape_owned(q_flat^, [1, S, H, Dh])
    var k_pre = reshape_owned(k_flat^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)

    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)

    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])

    var no_bias_o = Optional[Tensor](None)
    var att_o = linear(att_flat, w.wo[], no_bias_o^, ctx)      # [S,D]

    var attn_n2 = rms_norm(att_o, w.n2[], eps, ctx)
    var gate_msa_t = tanh_op(mv.gate_msa[], ctx)
    var h = residual_gate(x_arc[], gate_msa_t, attn_n2, ctx)   # x + gate*attn_n2

    # --- MLP sub-block (SwiGLU, sandwich norm) ---
    var xfn1 = rms_norm(h, w.fn1[], eps, ctx)
    var xfn1s = modulate(xfn1, mv.scale_mlp[], mv.zeros[], ctx)

    var no_bias_g = Optional[Tensor](None)
    var g_pre = linear(xfn1s, w.w1[], no_bias_g^, ctx)         # [S,F]
    var no_bias_u = Optional[Tensor](None)
    var u = linear(xfn1s, w.w3[], no_bias_u^, ctx)            # [S,F]
    var act = swiglu(g_pre, u, ctx)                            # silu(g_pre)*u
    var no_bias_d = Optional[Tensor](None)
    var ff = linear(act, w.w2[], no_bias_d^, ctx)             # [S,D]

    var ff_n2 = rms_norm(ff, w.fn2[], eps, ctx)
    var gate_mlp_t = tanh_op(mv.gate_mlp[], ctx)
    var result = residual_gate(h, gate_mlp_t, ff_n2, ctx)
    return TArc(result^)


# UNMODULATED (context-refiner) frozen block, device in/out. Exact op sequence
# of block.zimage_refiner_forward (plain residuals, no modulation), no saved,
# no to_host.
def zimage_refiner_forward_device[
    H: Int, Dh: Int, S: Int
](
    x_arc: TArc,
    w: ZImageBlockWeights,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> TArc:
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    # --- attention sub-block (sandwich norm, NO modulation) ---
    var xn1 = rms_norm(x_arc[], w.n1[], eps, ctx)

    var no_bias = Optional[Tensor](None)
    var q_flat = linear(xn1, w.wq[], no_bias^, ctx)
    var no_bias_k = Optional[Tensor](None)
    var k_flat = linear(xn1, w.wk[], no_bias_k^, ctx)
    var no_bias_v = Optional[Tensor](None)
    var v_flat = linear(xn1, w.wv[], no_bias_v^, ctx)

    var q_pre = reshape_owned(q_flat^, [1, S, H, Dh])
    var k_pre = reshape_owned(k_flat^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    var q_rms = rms_norm(q_pre, w.q_norm[], eps, ctx)
    var k_rms = rms_norm(k_pre, w.k_norm[], eps, ctx)

    var q_rope = rope_interleaved(q_rms, cos, sin, ctx)
    var k_rope = rope_interleaved(k_rms, cos, sin, ctx)

    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])

    var no_bias_o = Optional[Tensor](None)
    var att_o = linear(att_flat, w.wo[], no_bias_o^, ctx)

    var attn_n2 = rms_norm(att_o, w.n2[], eps, ctx)
    var h = add(x_arc[], attn_n2, ctx)              # PLAIN residual (no gate)

    # --- MLP sub-block (SwiGLU, sandwich norm, NO modulation) ---
    var xfn1 = rms_norm(h, w.fn1[], eps, ctx)

    var no_bias_g = Optional[Tensor](None)
    var g_pre = linear(xfn1, w.w1[], no_bias_g^, ctx)
    var no_bias_u = Optional[Tensor](None)
    var u = linear(xfn1, w.w3[], no_bias_u^, ctx)
    var act = swiglu(g_pre, u, ctx)
    var no_bias_d = Optional[Tensor](None)
    var ff = linear(act, w.w2[], no_bias_d^, ctx)

    var ff_n2 = rms_norm(ff, w.fn2[], eps, ctx)
    var result = add(h, ff_n2, ctx)                 # PLAIN residual (no gate)
    return TArc(result^)


# ══════════════════════════════════════════════════════════════════════════════
# Phase P5 (AUTOGRAD_V2_MOJO_DESIGN.md C9): StepSlab FORWARD block functions for
# the capture-compatible _v5 step. Each is the EXACT op sequence of its non-slab
# sibling above with every allocating op routed to its _slab variant (same
# kernels, same order, same values — C14; only the allocation source changes,
# the P4 precedent). NO saved tape is returned for the LoRA main block: the
# graph backward (_v4/_v5) recomputes the block from its saved INPUT, so the
# forward only needs the output (the batch fwd's saved tape was refcount
# copies — zero kernels — so omitting it changes no values). NO syncs anywhere
# in these bodies (C9: capture-safe; single-stream ordering, TIER2 precedent
# ops/attention.mojo). Old paths untouched (C13).
# ══════════════════════════════════════════════════════════════════════════════
def zimage_block_lora_forward_device_only_slab[
    B: Int, H: Int, Dh: Int, S: Int
](
    x_arc: TArc,
    w: ZImageBlockWeights, mv: ZImageModVecsDevice, lora: ZImageBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> TArc:
    """Output-only StepSlab sibling of
    zimage_block_lora_forward_device_tensor_batch (this file): identical op
    chain; the result is slab-resident — the caller copies it out before the
    per-block rewind."""
    comptime ROWS = B * S
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var xn1 = rms_norm_slab(x_arc[], w.n1[], eps, ctx, slab)
    var xn1s = modulate_slab(xn1, mv.scale_msa[], mv.zeros[], ctx, slab)

    var no_bias = Optional[Tensor](None)
    var q_base = linear_slab(xn1s, w.wq[], no_bias^, ctx, slab)
    var no_bias_k = Optional[Tensor](None)
    var k_base = linear_slab(xn1s, w.wk[], no_bias_k^, ctx, slab)
    var no_bias_v = Optional[Tensor](None)
    var v_base = linear_slab(xn1s, w.wv[], no_bias_v^, ctx, slab)

    var q = zimage_lora_apply_device_slab(q_base^, xn1s, lora.to_q, ROWS, ctx, slab)
    var k = zimage_lora_apply_device_slab(k_base^, xn1s, lora.to_k, ROWS, ctx, slab)
    var v_flat = zimage_lora_apply_device_slab(v_base^, xn1s, lora.to_v, ROWS, ctx, slab)

    var q_pre = reshape_owned(q^, [B, S, H, Dh])
    var k_pre = reshape_owned(k^, [B, S, H, Dh])
    var v = reshape_owned(v_flat^, [B, S, H, Dh])

    var q_rms = rms_norm_slab(q_pre, w.q_norm[], eps, ctx, slab)
    var k_rms = rms_norm_slab(k_pre, w.k_norm[], eps, ctx, slab)
    var q_rope = rope_interleaved_slab(q_rms, cos, sin, ctx, slab)
    var k_rope = rope_interleaved_slab(k_rms, cos, sin, ctx, slab)

    var att = sdpa_nomask_slab[B, S, H, Dh](q_rope, k_rope, v, scale, ctx, slab)
    var att_flat = reshape_owned(att^, [ROWS, D])

    var no_bias_o = Optional[Tensor](None)
    var att_o_base = linear_slab(att_flat, w.wo[], no_bias_o^, ctx, slab)
    var att_o = zimage_lora_apply_device_slab(
        att_o_base^, att_flat, lora.to_out, ROWS, ctx, slab
    )

    var attn_n2 = rms_norm_slab(att_o, w.n2[], eps, ctx, slab)
    var gate_msa_t = tanh_op_slab(mv.gate_msa[], ctx, slab)
    var h = residual_gate_slab(x_arc[], gate_msa_t, attn_n2, ctx, slab)

    var xfn1 = rms_norm_slab(h, w.fn1[], eps, ctx, slab)
    var xfn1s = modulate_slab(xfn1, mv.scale_mlp[], mv.zeros[], ctx, slab)

    var no_bias_g = Optional[Tensor](None)
    var g_base = linear_slab(xfn1s, w.w1[], no_bias_g^, ctx, slab)
    var no_bias_u = Optional[Tensor](None)
    var u_base = linear_slab(xfn1s, w.w3[], no_bias_u^, ctx, slab)
    var g_pre = zimage_lora_apply_device_slab(g_base^, xfn1s, lora.w1, ROWS, ctx, slab)
    var u = zimage_lora_apply_device_slab(u_base^, xfn1s, lora.w3, ROWS, ctx, slab)

    var act = swiglu_slab(g_pre, u, ctx, slab)

    var no_bias_d = Optional[Tensor](None)
    var ff_base = linear_slab(act, w.w2[], no_bias_d^, ctx, slab)
    var ff = zimage_lora_apply_device_slab(ff_base^, act, lora.w2, ROWS, ctx, slab)

    var ff_n2 = rms_norm_slab(ff, w.fn2[], eps, ctx, slab)
    var gate_mlp_t = tanh_op_slab(mv.gate_mlp[], ctx, slab)
    var result = residual_gate_slab(h, gate_mlp_t, ff_n2, ctx, slab)
    return TArc(result^)


def zimage_block_forward_device_moddev_slab[
    H: Int, Dh: Int, S: Int
](
    x_arc: TArc,
    w: ZImageBlockWeights, mv: ZImageModVecsDevice,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> TArc:
    """StepSlab sibling of zimage_block_forward_device_moddev (frozen NR
    block, this file): identical op chain, slab-resident output."""
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var xn1 = rms_norm_slab(x_arc[], w.n1[], eps, ctx, slab)
    var xn1s = modulate_slab(xn1, mv.scale_msa[], mv.zeros[], ctx, slab)

    var no_bias = Optional[Tensor](None)
    var q_flat = linear_slab(xn1s, w.wq[], no_bias^, ctx, slab)
    var no_bias_k = Optional[Tensor](None)
    var k_flat = linear_slab(xn1s, w.wk[], no_bias_k^, ctx, slab)
    var no_bias_v = Optional[Tensor](None)
    var v_flat = linear_slab(xn1s, w.wv[], no_bias_v^, ctx, slab)

    var q_pre = reshape_owned(q_flat^, [1, S, H, Dh])
    var k_pre = reshape_owned(k_flat^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    var q_rms = rms_norm_slab(q_pre, w.q_norm[], eps, ctx, slab)
    var k_rms = rms_norm_slab(k_pre, w.k_norm[], eps, ctx, slab)

    var q_rope = rope_interleaved_slab(q_rms, cos, sin, ctx, slab)
    var k_rope = rope_interleaved_slab(k_rms, cos, sin, ctx, slab)

    var att = sdpa_nomask_slab[1, S, H, Dh](q_rope, k_rope, v, scale, ctx, slab)
    var att_flat = reshape_owned(att^, [S, D])

    var no_bias_o = Optional[Tensor](None)
    var att_o = linear_slab(att_flat, w.wo[], no_bias_o^, ctx, slab)

    var attn_n2 = rms_norm_slab(att_o, w.n2[], eps, ctx, slab)
    var gate_msa_t = tanh_op_slab(mv.gate_msa[], ctx, slab)
    var h = residual_gate_slab(x_arc[], gate_msa_t, attn_n2, ctx, slab)

    var xfn1 = rms_norm_slab(h, w.fn1[], eps, ctx, slab)
    var xfn1s = modulate_slab(xfn1, mv.scale_mlp[], mv.zeros[], ctx, slab)

    var no_bias_g = Optional[Tensor](None)
    var g_pre = linear_slab(xfn1s, w.w1[], no_bias_g^, ctx, slab)
    var no_bias_u = Optional[Tensor](None)
    var u = linear_slab(xfn1s, w.w3[], no_bias_u^, ctx, slab)
    var act = swiglu_slab(g_pre, u, ctx, slab)
    var no_bias_d = Optional[Tensor](None)
    var ff = linear_slab(act, w.w2[], no_bias_d^, ctx, slab)

    var ff_n2 = rms_norm_slab(ff, w.fn2[], eps, ctx, slab)
    var gate_mlp_t = tanh_op_slab(mv.gate_mlp[], ctx, slab)
    var result = residual_gate_slab(h, gate_mlp_t, ff_n2, ctx, slab)
    return TArc(result^)


def zimage_refiner_forward_device_slab[
    H: Int, Dh: Int, S: Int
](
    x_arc: TArc,
    w: ZImageBlockWeights,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> TArc:
    """StepSlab sibling of zimage_refiner_forward_device (frozen CR block,
    this file): identical op chain, slab-resident output."""
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var xn1 = rms_norm_slab(x_arc[], w.n1[], eps, ctx, slab)

    var no_bias = Optional[Tensor](None)
    var q_flat = linear_slab(xn1, w.wq[], no_bias^, ctx, slab)
    var no_bias_k = Optional[Tensor](None)
    var k_flat = linear_slab(xn1, w.wk[], no_bias_k^, ctx, slab)
    var no_bias_v = Optional[Tensor](None)
    var v_flat = linear_slab(xn1, w.wv[], no_bias_v^, ctx, slab)

    var q_pre = reshape_owned(q_flat^, [1, S, H, Dh])
    var k_pre = reshape_owned(k_flat^, [1, S, H, Dh])
    var v = reshape_owned(v_flat^, [1, S, H, Dh])

    var q_rms = rms_norm_slab(q_pre, w.q_norm[], eps, ctx, slab)
    var k_rms = rms_norm_slab(k_pre, w.k_norm[], eps, ctx, slab)

    var q_rope = rope_interleaved_slab(q_rms, cos, sin, ctx, slab)
    var k_rope = rope_interleaved_slab(k_rms, cos, sin, ctx, slab)

    var att = sdpa_nomask_slab[1, S, H, Dh](q_rope, k_rope, v, scale, ctx, slab)
    var att_flat = reshape_owned(att^, [S, D])

    var no_bias_o = Optional[Tensor](None)
    var att_o = linear_slab(att_flat, w.wo[], no_bias_o^, ctx, slab)

    var attn_n2 = rms_norm_slab(att_o, w.n2[], eps, ctx, slab)
    var h = add_slab(x_arc[], attn_n2, ctx, slab)    # PLAIN residual (no gate)

    var xfn1 = rms_norm_slab(h, w.fn1[], eps, ctx, slab)

    var no_bias_g = Optional[Tensor](None)
    var g_pre = linear_slab(xfn1, w.w1[], no_bias_g^, ctx, slab)
    var no_bias_u = Optional[Tensor](None)
    var u = linear_slab(xfn1, w.w3[], no_bias_u^, ctx, slab)
    var act = swiglu_slab(g_pre, u, ctx, slab)
    var no_bias_d = Optional[Tensor](None)
    var ff = linear_slab(act, w.w2[], no_bias_d^, ctx, slab)

    var ff_n2 = rms_norm_slab(ff, w.fn2[], eps, ctx, slab)
    var result = add_slab(h, ff_n2, ctx, slab)       # PLAIN residual (no gate)
    return TArc(result^)


# ══════════════════════════════════════════════════════════════════════════════
# T2.C FULL-RANK FINETUNE (2026-06-11) — modulated-block backward that ALSO
# materializes the 7 base-projection weight grads d_W = d_Yᵀ @ X (the grads the
# LoRA path deliberately skips — "frozen base d_W ... must not be materialized
# for Z-Image full depth" applies to the FULL-MODEL-RESIDENT case; here each
# block's d_W set (~354 MB bf16) is transient: computed, D2H'd by the stack
# loop, then freed before the next block — the 24 GB budget holds, measured in
# train_zimage_real full-FT gates).
#
# Math: byte-identical op chain to zimage_block_lora_backward_device_tensors_
# batch at B=1 for the d_x spine, with the LoRA branches REMOVED (full-FT has
# no adapters; the recompute forward runs with the zero/scale-0 LoRA set so
# the forward is the BASE forward) and one linear_backward_dw per slot added:
#   d_Wq = d_q_preᵀ @ xn1s     d_Wk = d_k_preᵀ @ xn1s    d_Wv = d_vᵀ @ xn1s
#   d_Wo = d_att_oᵀ @ att_flat
#   d_W1 = d_gateᵀ @ xfn1s     d_W3 = d_upᵀ @ xfn1s      d_W2 = d_ffᵀ @ act
# d_W returned BF16 (C12 bf16-first grads; F32 accumulation inside the GEMM).
# ADDITIVE: no existing path is touched (C13).
# ══════════════════════════════════════════════════════════════════════════════
struct ZImageBlockFullFTBackward(Movable):
    var d_x: TArc
    var d_w: List[TArc]   # ZIMAGE_SLOTS entries, slot order Q,K,V,O,W1,W3,W2

    def __init__(out self, var d_x: TArc, var d_w: List[TArc]):
        self.d_x = d_x^
        self.d_w = d_w^


def zimage_block_backward_device_tensors_fullft[
    H: Int, Dh: Int, S: Int
](
    d_out: Tensor,
    w: ZImageBlockWeights, mv: ZImageModVecsDevice,
    saved: ZImageBlockSaved,
    cos: Tensor, sin: Tensor,
    D: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> ZImageBlockFullFTBackward:
    comptime ROWS = S
    var scale = Float32(1.0) / sqrt(Float32(Dh))

    var grg2 = gate_residual_backward_dxdy(d_out, saved.gate_mlp_t[], ctx)

    var d_ff = rms_norm_backward_dx(grg2.d_y, saved.ff[], w.fn2[], eps, ctx)
    var d_w2 = linear_backward_dw(
        d_ff, saved.act[], ROWS, F, D, ctx, STDtype.BF16
    )
    var d_x_w2 = linear_backward_dx(d_ff, w.w2[], ROWS, F, D, ctx)

    var sg = swiglu_backward(d_x_w2, saved.g_pre[], saved.u[], ctx)

    var d_w1 = linear_backward_dw(
        sg.d_gate, saved.xfn1s[], ROWS, D, F, ctx, STDtype.BF16
    )
    var d_w3 = linear_backward_dw(
        sg.d_up, saved.xfn1s[], ROWS, D, F, ctx, STDtype.BF16
    )
    var d_x_w1 = linear_backward_dx(sg.d_gate, w.w1[], ROWS, D, F, ctx)
    var d_x_w3 = linear_backward_dx(sg.d_up, w.w3[], ROWS, D, F, ctx)
    var d_xfn1s = add(d_x_w1, d_x_w3, ctx)

    var mb_mlp = modulate_backward(
        d_xfn1s, saved.xfn1[], mv.scale_mlp[], ctx, compute_param_grads=False,
    )
    var d_h_norm = rms_norm_backward_dx(mb_mlp.d_x, saved.h[], w.fn1[], eps, ctx)
    var d_h = add(grg2.d_x, d_h_norm, ctx)

    var grg1 = gate_residual_backward_dxdy(d_h, saved.gate_msa_t[], ctx)

    var d_att_o = rms_norm_backward_dx(grg1.d_y, saved.att_o[], w.n2[], eps, ctx)
    var d_wo = linear_backward_dw(
        d_att_o, saved.att_flat[], ROWS, D, D, ctx, STDtype.BF16
    )
    var d_x_o = linear_backward_dx(d_att_o, w.wo[], ROWS, D, D, ctx)

    reshape_in_place(d_x_o, [1, S, H, Dh])
    var sb = sdpa_backward[1, S, H, Dh](
        saved.q_rope[], saved.k_rope[], saved.v[], d_x_o, scale, ctx
    )
    var d_q_rms = rope_backward(sb.d_q, cos, sin, True, ctx)
    var d_k_rms = rope_backward(sb.d_k, cos, sin, True, ctx)

    var d_q_pre = rms_norm_backward_dx(d_q_rms, saved.q_pre[], w.q_norm[], eps, ctx)
    var d_k_pre = rms_norm_backward_dx(d_k_rms, saved.k_pre[], w.k_norm[], eps, ctx)

    reshape_in_place(d_q_pre, [ROWS, D])
    reshape_in_place(d_k_pre, [ROWS, D])
    reshape_in_place(sb.d_v, [ROWS, D])

    var d_wq = linear_backward_dw(
        d_q_pre, saved.xn1s[], ROWS, D, D, ctx, STDtype.BF16
    )
    var d_wk = linear_backward_dw(
        d_k_pre, saved.xn1s[], ROWS, D, D, ctx, STDtype.BF16
    )
    var d_wv = linear_backward_dw(
        sb.d_v, saved.xn1s[], ROWS, D, D, ctx, STDtype.BF16
    )
    var d_x_q = linear_backward_dx(d_q_pre, w.wq[], ROWS, D, D, ctx)
    var d_x_k = linear_backward_dx(d_k_pre, w.wk[], ROWS, D, D, ctx)
    var d_x_v = linear_backward_dx(sb.d_v, w.wv[], ROWS, D, D, ctx)
    var d_xn1s = add(add(d_x_q, d_x_k, ctx), d_x_v, ctx)

    var mb_sa = modulate_backward(
        d_xn1s, saved.xn1[], mv.scale_msa[], ctx, compute_param_grads=False,
    )
    var d_x_norm = rms_norm_backward_dx(mb_sa.d_x, saved.x[], w.n1[], eps, ctx)

    var d_x_t = add(grg1.d_x, d_x_norm, ctx)

    var d_w_slots = List[TArc]()
    d_w_slots.append(TArc(d_wq^))
    d_w_slots.append(TArc(d_wk^))
    d_w_slots.append(TArc(d_wv^))
    d_w_slots.append(TArc(d_wo^))
    d_w_slots.append(TArc(d_w1^))
    d_w_slots.append(TArc(d_w3^))
    d_w_slots.append(TArc(d_w2^))

    return ZImageBlockFullFTBackward(TArc(d_x_t^), d_w_slots^)
