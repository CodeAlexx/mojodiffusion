# serenitymojo/models/anima/lora_block.mojo
#
# LoRA-ON-PROJECTION for the ANIMA (Cosmos-Predict2 MiniTrainDIT) block. Mirrors
# the JUST-VERIFIED-GREEN ERNIE LoRA pattern (models/ernie/lora_block.mojo) and
# the original proven Klein template, specialized to Anima's TEN un-fused target
# projections (the LoRA targets the inference-flame chokepoint linears, see
# inference-flame/src/models/anima.rs):
#   self_attn  : q_proj, k_proj, v_proj, output_proj   (anima.rs:365-390)
#   cross_attn : q_proj, k_proj, v_proj, output_proj   (anima.rs:411-440)
#   mlp        : layer1, layer2                         (anima.rs:449-451)
# The inference LoRA chokepoint is `linear_no_bias(weight_key)` (anima.rs:158-182):
#   out = x @ W.T ; then LoraStack::apply(weight_key, x, out) adds scale·up(down(x)).
# So a target's LoRA INPUT is exactly the SAME `x` the base linear consumes — for
# cross-attn k/v that input is the FROZEN `context` (anima.rs:413-414), NOT x_mod.
#
# WHY THE TRAINER MATH IS THE AUTHORITY (identical to Klein/Ernie lora_block.mojo)
#   For a projection y = linear(x, W) (W [out,in]), the LoRA-adapted output is
#       y' = linear(x, W) + scale·((x @ Aᵀ) @ Bᵀ)
#   A=[rank,in], B=[out,rank], scale = alpha/rank. This MATCHES the inference merge
#   in inference-flame anima.rs (W' = W + scale·B@A applied at the chokepoint).
#
#   BACKWARD (given d_y' at the projection output, x the saved projection input):
#       d_dy = scale·d_y'                    [M,out]
#       d_B  = d_dyᵀ @ t   (t = x @ Aᵀ)      [out,rank]
#       d_t  = d_dy  @ B                      [M,rank]
#       d_A  = d_tᵀ  @ x                      [rank,in]
#       d_x  = d_t   @ A                      [M,in]   (LoRA branch's contribution
#                                                       to the projection INPUT grad)
#   The base path (frozen W) ALSO yields d_x_base = d_y' @ W; the caller SUMS d_x
#   into that WHEN the projection input is a trained-stream tensor. For cross-attn
#   k/v the input is frozen `context`, so the LoRA d_x is DISCARDED there (matching
#   the base block which already discards d_input on the frozen-context k/v linears).
#
# anima_lora_fwd / anima_lora_bwd are byte-identical to ernie_lora_fwd/bwd (which
# are byte-identical to train_step._lora_fwd/_lora_bwd), plus the d_x term needed
# to thread the LoRA grad back into the projection input.
#
# NO NEW ops/ PRIMITIVE: forward = two linear()s; backward = two linear_backward()s.
#
# VERIFIED ARCH FACTS NOT REGRESSED (LoRA wraps the q/k/v/out/mlp LINEARS ONLY):
#   AdaLN-pre = LayerNorm-no-affine eps1e-6 (NOT RMSNorm); modulation silus RAW
#   t_cond ONCE internally (the double-silu fix stays — t_silu is computed once in
#   the block fwd and reused); self_attn RoPE half-split interleaved=False; cross_attn
#   NO RoPE + NO text-pad mask; GELU tanh-approx. None of these touch the LoRA path.
#
# Mojo 1.0.0b1: def not fn; Tensor move-only crosses API as host List[Float32].

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt as fsqrt
from std.memory import ArcPointer
from std.time import perf_counter_ns
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import (
    linear_backward, linear_backward_dx, linear_backward_dw,
)

# REUSE the trainer's LoRA structs (the target authority).
from serenitymojo.training.train_step import LoraAdapter, LoraGrads

# Forward + backward ops shared with the base block (Tenet 1: nothing new here).
from serenitymojo.ops.norm import rms_norm, layer_norm
from serenitymojo.ops.activations import gelu, silu
from serenitymojo.ops.tensor_algebra import (
    reshape, reshape_in_place, slice, add, mul, mul_scalar, concat,
)
from serenitymojo.ops.reduce import reduce_sum
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.models.dit.sdxl_attention import sdxl_sdpa
from serenitymojo.ops.norm_backward import (
    rms_norm_backward, rms_norm_backward_dx, layer_norm_backward_dx,
)
from serenitymojo.ops.activation_backward import silu_backward, gelu_backward
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.rope_struct_backward import rope_backward
from serenitymojo.ops.attention_backward import sdpa_backward, sdpa_backward_rect

from serenitymojo.models.anima.weights import AnimaBlockWeights
from serenitymojo.models.anima.block import (
    AnimaBlockSaved, AnimaBlockGrads, AnimaBlockForward,
    _SubSaved, _AttnSaved, _adaln_pre, _rms_per_head,
    _gate_residual_bwd,
)
from serenitymojo.models.dit.anima_contract import ANIMA_HIDDEN  # 2048


comptime TArc = ArcPointer[Tensor]


# ── slot order (canonical; MUST match anima_stack_lora + oracle SLOTS list) ───
comptime ANIMA_SLOTS = 10
comptime SLOT_SA_Q = 0
comptime SLOT_SA_K = 1
comptime SLOT_SA_V = 2
comptime SLOT_SA_O = 3
comptime SLOT_CA_Q = 4
comptime SLOT_CA_K = 5
comptime SLOT_CA_V = 6
comptime SLOT_CA_O = 7
comptime SLOT_MLP1 = 8
comptime SLOT_MLP2 = 9


def _ones(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(1.0)
    return o^


def _zeros(n: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(n):
        o.append(0.0)
    return o^


def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


def _add_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i] + b[i])
    return o^


def _dim1() -> List[Int]:
    var d = List[Int](); d.append(1); return d^


# Adapter forward contribution on x [M,in] -> [M,out] (host list in/out).
# Byte-identical to train_step._lora_fwd / ernie_lora_fwd.
def anima_lora_fwd(
    x_h: List[Float32], lo: LoraAdapter, M: Int, ctx: DeviceContext
) raises -> List[Float32]:
    var nb1 = Optional[Tensor](None)
    var t = linear(
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.F32, ctx),
        Tensor.from_host(lo.a.copy(), [lo.rank, lo.in_f], STDtype.F32, ctx),
        nb1^, ctx,
    ).to_host(ctx)                                   # [M,rank]
    var nb2 = Optional[Tensor](None)
    var dy = linear(
        Tensor.from_host(t.copy(), [M, lo.rank], STDtype.F32, ctx),
        Tensor.from_host(lo.b.copy(), [lo.out_f, lo.rank], STDtype.F32, ctx),
        nb2^, ctx,
    ).to_host(ctx)                                   # [M,out]
    var out = List[Float32]()
    for i in range(len(dy)):
        out.append(lo.scale * dy[i])
    return out^


# Optionally-applied adapter forward: if present, return base_y + LoRA contribution
# on host-list `x_h`; else return base_y unchanged (base-path no-regression).
def anima_lora_apply(
    base_y: List[Float32], x_h: List[Float32], lo: Optional[LoraAdapter],
    M: Int, ctx: DeviceContext,
) raises -> List[Float32]:
    if not lo:
        return base_y.copy()
    var contrib = anima_lora_fwd(x_h, lo.value(), M, ctx)
    var out = List[Float32]()
    for i in range(len(base_y)):
        out.append(base_y[i] + contrib[i])
    return out^


# LoRA backward that ALSO returns the LoRA branch's contribution to d_x.
# d_a/d_b match train_step._lora_bwd exactly; d_x is the term that file drops.
struct AnimaLoraGrads(Copyable, Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]
    var d_x: List[Float32]   # LoRA contribution to the projection INPUT grad [M,in]

    def __init__(
        out self, var d_a: List[Float32], var d_b: List[Float32], var d_x: List[Float32]
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x = d_x^


def anima_lora_bwd(
    d_contrib_h: List[Float32], x_h: List[Float32], lo: LoraAdapter,
    M: Int, ctx: DeviceContext,
) raises -> AnimaLoraGrads:
    # t = x @ Aᵀ  (recompute; cheap)
    var nb_t = Optional[Tensor](None)
    var t = linear(
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.F32, ctx),
        Tensor.from_host(lo.a.copy(), [lo.rank, lo.in_f], STDtype.F32, ctx),
        nb_t^, ctx,
    ).to_host(ctx)                                   # [M,rank]
    var d_dy = List[Float32]()
    for i in range(len(d_contrib_h)):
        d_dy.append(lo.scale * d_contrib_h[i])       # [M,out]
    # dy = t @ Bᵀ  -> d_B (d_w) and d_t (d_x)
    var lbB = linear_backward(
        Tensor.from_host(d_dy^, [M, lo.out_f], STDtype.F32, ctx),
        Tensor.from_host(t.copy(), [M, lo.rank], STDtype.F32, ctx),
        Tensor.from_host(lo.b.copy(), [lo.out_f, lo.rank], STDtype.F32, ctx),
        M, lo.rank, lo.out_f, ctx,
    )
    var d_t = lbB.d_x.to_host(ctx)                   # [M,rank]
    var d_b = lbB.d_w.to_host(ctx)                   # [out_f,rank]
    # t = x @ Aᵀ  -> d_A (d_w) and d_x_lo (d_x)
    var lbA = linear_backward(
        Tensor.from_host(d_t^, [M, lo.rank], STDtype.F32, ctx),
        Tensor.from_host(x_h.copy(), [M, lo.in_f], STDtype.F32, ctx),
        Tensor.from_host(lo.a.copy(), [lo.rank, lo.in_f], STDtype.F32, ctx),
        M, lo.in_f, lo.rank, ctx,
    )
    var d_x_lo = lbA.d_x.to_host(ctx)                # [M,in_f]
    var d_a = lbA.d_w.to_host(ctx)                   # [rank,in_f]
    return AnimaLoraGrads(d_a^, d_b^, d_x_lo^)


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE-RESIDENT LoRA (mirrors models/zimage/lora_block.mojo ZImageLoraAdapterDevice
# + zimage_lora_apply_device + zimage_lora_bwd_device_resident_tensors). The adapter
# A/B live as device Tensors (BF16 for production, F32 for the parity gate), uploaded
# ONCE per step. forward = two device linear()s (F32-out via mixed_base when BF16);
# backward = linear_backward_dx/_dw on device → d_a/d_b/d_x stay DEVICE tensors.
# ══════════════════════════════════════════════════════════════════════════════
struct AnimaLoraAdapterDevice(Copyable, Movable):
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


def anima_lora_adapter_to_device(
    lo: LoraAdapter, dt: STDtype, ctx: DeviceContext
) raises -> AnimaLoraAdapterDevice:
    return AnimaLoraAdapterDevice(
        TArc(Tensor.from_host(lo.a.copy(), [lo.rank, lo.in_f], dt, ctx)),
        TArc(Tensor.from_host(lo.b.copy(), [lo.out_f, lo.rank], dt, ctx)),
        lo.rank, lo.in_f, lo.out_f, lo.scale,
    )


# base_y[M,out] + scale·((x @ Aᵀ) @ Bᵀ). x is the SAME device tensor the base
# linear consumes; output stays F32 (mixed_base when adapters are BF16).
def anima_lora_apply_device(
    var base_y: Tensor, x: Tensor, lo: AnimaLoraAdapterDevice,
    M: Int, ctx: DeviceContext,
) raises -> Tensor:
    var nb1 = Optional[Tensor](None)
    var t = linear(x, lo.a[], nb1^, ctx)            # [M,rank] F32
    var nb2 = Optional[Tensor](None)
    var dy = linear(t, lo.b[], nb2^, ctx)           # [M,out]  F32
    var contrib = mul_scalar(dy, lo.scale, ctx)
    return add(base_y^, contrib^, ctx)


# LoRA backward as DEVICE tensors. Returns d_a/d_b (F32) and the LoRA branch's
# d_x [M,in] (F32). Mirrors zimage_lora_bwd_device_resident_tensors.
struct AnimaLoraDeviceGradTensors(Copyable, Movable):
    var d_a: TArc
    var d_b: TArc
    var d_x: TArc

    def __init__(out self, var d_a: TArc, var d_b: TArc, var d_x: TArc):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x = d_x^


def anima_lora_bwd_device_tensors(
    d_contrib: Tensor, x: Tensor, lo: AnimaLoraAdapterDevice,
    M: Int, ctx: DeviceContext,
) raises -> AnimaLoraDeviceGradTensors:
    var nb_t = Optional[Tensor](None)
    var t = linear(x, lo.a[], nb_t^, ctx)           # [M,rank] F32
    var d_dy = mul_scalar(d_contrib, lo.scale, ctx) # [M,out]  F32

    var d_t = linear_backward_dx(d_dy, lo.b[], M, lo.rank, lo.out_f, ctx)   # [M,rank]
    var d_b_t = linear_backward_dw(d_dy, t, M, lo.rank, lo.out_f, ctx)      # [out,rank]

    var d_x_lo = linear_backward_dx(d_t, lo.a[], M, lo.in_f, lo.rank, ctx)  # [M,in]
    var d_a_t = linear_backward_dw(d_t, x, M, lo.in_f, lo.rank, ctx)        # [rank,in]
    return AnimaLoraDeviceGradTensors(TArc(d_a_t^), TArc(d_b_t^), TArc(d_x_lo^))


# ── per-block device LoRA carrier: the 10 device adapters (slot order canonical) ─
struct AnimaBlockLoraDevice(Copyable, Movable):
    var sa_q: AnimaLoraAdapterDevice
    var sa_k: AnimaLoraAdapterDevice
    var sa_v: AnimaLoraAdapterDevice
    var sa_out: AnimaLoraAdapterDevice
    var ca_q: AnimaLoraAdapterDevice
    var ca_k: AnimaLoraAdapterDevice
    var ca_v: AnimaLoraAdapterDevice
    var ca_out: AnimaLoraAdapterDevice
    var mlp1: AnimaLoraAdapterDevice
    var mlp2: AnimaLoraAdapterDevice

    def __init__(
        out self,
        var sa_q: AnimaLoraAdapterDevice, var sa_k: AnimaLoraAdapterDevice,
        var sa_v: AnimaLoraAdapterDevice, var sa_out: AnimaLoraAdapterDevice,
        var ca_q: AnimaLoraAdapterDevice, var ca_k: AnimaLoraAdapterDevice,
        var ca_v: AnimaLoraAdapterDevice, var ca_out: AnimaLoraAdapterDevice,
        var mlp1: AnimaLoraAdapterDevice, var mlp2: AnimaLoraAdapterDevice,
    ):
        self.sa_q = sa_q^; self.sa_k = sa_k^; self.sa_v = sa_v^; self.sa_out = sa_out^
        self.ca_q = ca_q^; self.ca_k = ca_k^; self.ca_v = ca_v^; self.ca_out = ca_out^
        self.mlp1 = mlp1^; self.mlp2 = mlp2^


# proj-backward: frozen-base d_x (BF16/F32 weight) + LoRA branch d_x (summed when
# keep_dx). Returns the summed d_x + the LoRA d_a/d_b DEVICE tensors. For frozen
# cross-attn k/v inputs keep_dx=False (their base d_x flows into frozen context).
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
    lo: AnimaLoraAdapterDevice,
    M: Int, in_f: Int, out_f: Int, keep_dx: Bool,
    ctx: DeviceContext,
) raises -> _ProjTensorGrads:
    var lg = anima_lora_bwd_device_tensors(d_y, x_in, lo, M, ctx)
    if keep_dx:
        var base_dx = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)
        var summed = add(base_dx^, lg.d_x[], ctx)
        return _ProjTensorGrads(summed^, lg.d_a.copy(), lg.d_b.copy())
    # frozen-context input: base d_x is discarded; return a placeholder d_x (the
    # caller ignores it for ca k/v). Keep the LoRA d_a/d_b.
    var base_dx2 = linear_backward_dx(d_y, w, M, in_f, out_f, ctx)
    return _ProjTensorGrads(base_dx2^, lg.d_a.copy(), lg.d_b.copy())


# ── per-block LoRA carrier: the 10 optional adapters (slot order is canonical) ──
struct AnimaBlockLora(Copyable, Movable):
    var sa_q: Optional[LoraAdapter]
    var sa_k: Optional[LoraAdapter]
    var sa_v: Optional[LoraAdapter]
    var sa_out: Optional[LoraAdapter]
    var ca_q: Optional[LoraAdapter]
    var ca_k: Optional[LoraAdapter]
    var ca_v: Optional[LoraAdapter]
    var ca_out: Optional[LoraAdapter]
    var mlp1: Optional[LoraAdapter]
    var mlp2: Optional[LoraAdapter]

    def __init__(
        out self,
        var sa_q: Optional[LoraAdapter], var sa_k: Optional[LoraAdapter],
        var sa_v: Optional[LoraAdapter], var sa_out: Optional[LoraAdapter],
        var ca_q: Optional[LoraAdapter], var ca_k: Optional[LoraAdapter],
        var ca_v: Optional[LoraAdapter], var ca_out: Optional[LoraAdapter],
        var mlp1: Optional[LoraAdapter], var mlp2: Optional[LoraAdapter],
    ):
        self.sa_q = sa_q^; self.sa_k = sa_k^; self.sa_v = sa_v^; self.sa_out = sa_out^
        self.ca_q = ca_q^; self.ca_k = ca_k^; self.ca_v = ca_v^; self.ca_out = ca_out^
        self.mlp1 = mlp1^; self.mlp2 = mlp2^


# ── per-block LoRA grads (parallel to the 10 slots) ──────────────────────────
struct AnimaBlockLoraGrads(Copyable, Movable):
    var d_a: List[List[Float32]]   # ANIMA_SLOTS entries (empty if slot absent)
    var d_b: List[List[Float32]]

    def __init__(out self, var d_a: List[List[Float32]], var d_b: List[List[Float32]]):
        self.d_a = d_a^
        self.d_b = d_b^


# ── LoRA-aware block forward ──────────────────────────────────────────────────
# Mirrors anima_block_forward EXACTLY (models/anima/block.mojo), adding the LoRA
# contribution to each of the 10 trained projection outputs BEFORE the downstream
# op consumes it. When all 10 adapters are absent this reduces bit-for-bit to the
# base forward (each anima_lora_apply returns base_y unchanged). The `saved`
# activations are the LoRA-MODIFIED ones, so backward recompute regenerates them
# identically (same checkpoint contract as Klein/Ernie).
def anima_block_lora_forward[
    H: Int, Dh: Int, S_IMG: Int, S_TXT: Int
](
    x: List[Float32],          # [B,S_img,D] F32 residual-stream input
    t_cond: List[Float32],     # [B,2048] F32
    base_adaln: List[Float32], # [B,6144] F32
    context: List[Float32],    # [B,S_txt,1024] F32  (frozen text context)
    w: AnimaBlockWeights, lora: AnimaBlockLora,
    cos: Tensor, sin: Tensor,  # [B*S_img*H, Dh/2] F32
    B: Int, D: Int, JOINT: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> AnimaBlockForward:
    var ln_ones = _t(_ones(D), [D], ctx)
    var ln_zeros = _t(_zeros(D), [D], ctx)
    var sa_scale = Float32(1.0) / fsqrt(Float32(Dh))

    var x_f32 = _t(x, [B, S_IMG, D], ctx)
    var t_cond_t = _t(t_cond, [B, ANIMA_HIDDEN], ctx)
    var base_t = _t(base_adaln, [B, 3 * ANIMA_HIDDEN], ctx)
    var ctx_t = _t(context, [B, S_TXT, JOINT], ctx)
    var context_h = ctx_t.to_host(ctx)                    # [B*S_txt,JOINT] LoRA input (ca k/v)
    var t_silu = silu(t_cond_t, ctx)   # [B,2048], computed ONCE (double-silu fix held)

    # ── sub-block 1: SELF-ATTENTION ──────────────────────────────────────────
    var sa_h = linear(t_silu, w.sa_mod1[], Optional[Tensor](None), ctx)   # [B,256]
    var sa_modout = linear(sa_h, w.sa_mod2[], Optional[Tensor](None), ctx)
    var sa_added = add(sa_modout, base_t, ctx)
    var sa_shift = slice(sa_added, 1, 0, D, ctx)
    var sa_scalev = slice(sa_added, 1, D, D, ctx)
    var sa_gate = slice(sa_added, 1, 2 * D, D, ctx)

    var sa_ln = layer_norm(x_f32, ln_ones, ln_zeros, eps, ctx)
    var sa_xmod = _adaln_pre(x_f32, sa_shift, sa_scalev, ln_ones, ln_zeros, eps, ctx)
    var sa_xmod_h = sa_xmod.to_host(ctx)                  # [B*S_img,D] LoRA input (sa q/k/v)

    var sa_q_base = linear(sa_xmod, w.sa_q[], Optional[Tensor](None), ctx).to_host(ctx)
    var sa_k_base = linear(sa_xmod, w.sa_k[], Optional[Tensor](None), ctx).to_host(ctx)
    var sa_v_base = linear(sa_xmod, w.sa_v[], Optional[Tensor](None), ctx).to_host(ctx)
    var sa_q_h = anima_lora_apply(sa_q_base, sa_xmod_h, lora.sa_q, B * S_IMG, ctx)
    var sa_k_h = anima_lora_apply(sa_k_base, sa_xmod_h, lora.sa_k, B * S_IMG, ctx)
    var sa_v_h = anima_lora_apply(sa_v_base, sa_xmod_h, lora.sa_v, B * S_IMG, ctx)

    var sh4 = List[Int](); sh4.append(B); sh4.append(S_IMG); sh4.append(H); sh4.append(Dh)
    var sa_q4 = reshape(_t(sa_q_h, [B, S_IMG, D], ctx), sh4.copy(), ctx)
    var sa_k4 = reshape(_t(sa_k_h, [B, S_IMG, D], ctx), sh4.copy(), ctx)
    var sa_v4 = reshape(_t(sa_v_h, [B, S_IMG, D], ctx), sh4.copy(), ctx)
    var sa_qrms = _rms_per_head(sa_q4, w.sa_qn[], ctx)
    var sa_krms = _rms_per_head(sa_k4, w.sa_kn[], ctx)
    var sa_qrope = rope_halfsplit(sa_qrms, cos, sin, ctx)
    var sa_krope = rope_halfsplit(sa_krms, cos, sin, ctx)
    var sa_att = sdpa_nomask[1, S_IMG, H, Dh](sa_qrope, sa_krope, sa_v4, sa_scale, ctx)
    var saf = List[Int](); saf.append(B); saf.append(S_IMG); saf.append(D)
    var sa_attflat = reshape(sa_att, saf.copy(), ctx)
    var sa_attflat_h = sa_attflat.to_host(ctx)            # [B*S_img,D] LoRA input (sa out)

    var sa_out_base = linear(sa_attflat, w.sa_out[], Optional[Tensor](None), ctx).to_host(ctx)
    var sa_out_h = anima_lora_apply(sa_out_base, sa_attflat_h, lora.sa_out, B * S_IMG, ctx)
    var sa_out = _t(sa_out_h, [B, S_IMG, D], ctx)

    var g3 = List[Int](); g3.append(B); g3.append(1); g3.append(D)
    var sa_gate3 = reshape(sa_gate, g3.copy(), ctx)
    var sa_gated = mul(sa_out, sa_gate3, ctx)
    var x_after_sa = add(x_f32, sa_gated, ctx)

    # ── sub-block 2: CROSS-ATTENTION (no RoPE, rectangular SDPA, no mask) ─────
    var ca_h = linear(t_silu, w.ca_mod1[], Optional[Tensor](None), ctx)
    var ca_modout = linear(ca_h, w.ca_mod2[], Optional[Tensor](None), ctx)
    var ca_added = add(ca_modout, base_t, ctx)
    var ca_shift = slice(ca_added, 1, 0, D, ctx)
    var ca_scalev = slice(ca_added, 1, D, D, ctx)
    var ca_gate = slice(ca_added, 1, 2 * D, D, ctx)

    var ca_ln = layer_norm(x_after_sa, ln_ones, ln_zeros, eps, ctx)
    var ca_xmod = _adaln_pre(x_after_sa, ca_shift, ca_scalev, ln_ones, ln_zeros, eps, ctx)
    var ca_xmod_h = ca_xmod.to_host(ctx)                  # [B*S_img,D] LoRA input (ca q)

    var ca_q_base = linear(ca_xmod, w.ca_q[], Optional[Tensor](None), ctx).to_host(ctx)
    var ca_k_base = linear(ctx_t, w.ca_k[], Optional[Tensor](None), ctx).to_host(ctx)
    var ca_v_base = linear(ctx_t, w.ca_v[], Optional[Tensor](None), ctx).to_host(ctx)
    var ca_q_h = anima_lora_apply(ca_q_base, ca_xmod_h, lora.ca_q, B * S_IMG, ctx)
    var ca_k_h = anima_lora_apply(ca_k_base, context_h, lora.ca_k, B * S_TXT, ctx)
    var ca_v_h = anima_lora_apply(ca_v_base, context_h, lora.ca_v, B * S_TXT, ctx)

    var caq4s = List[Int](); caq4s.append(B); caq4s.append(S_IMG); caq4s.append(H); caq4s.append(Dh)
    var cak4s = List[Int](); cak4s.append(B); cak4s.append(S_TXT); cak4s.append(H); cak4s.append(Dh)
    var ca_q4 = reshape(_t(ca_q_h, [B, S_IMG, D], ctx), caq4s.copy(), ctx)
    var ca_k4 = reshape(_t(ca_k_h, [B, S_TXT, D], ctx), cak4s.copy(), ctx)
    var ca_v4 = reshape(_t(ca_v_h, [B, S_TXT, D], ctx), cak4s.copy(), ctx)
    var ca_qrms = _rms_per_head(ca_q4, w.ca_qn[], ctx)
    var ca_krms = _rms_per_head(ca_k4, w.ca_kn[], ctx)
    var ca_att = sdxl_sdpa[1, S_IMG, S_TXT, H, Dh](ca_qrms, ca_krms, ca_v4, sa_scale, ctx)
    var caf = List[Int](); caf.append(B); caf.append(S_IMG); caf.append(D)
    var ca_attflat = reshape(ca_att, caf.copy(), ctx)
    var ca_attflat_h = ca_attflat.to_host(ctx)            # [B*S_img,D] LoRA input (ca out)

    var ca_out_base = linear(ca_attflat, w.ca_out[], Optional[Tensor](None), ctx).to_host(ctx)
    var ca_out_h = anima_lora_apply(ca_out_base, ca_attflat_h, lora.ca_out, B * S_IMG, ctx)
    var ca_out = _t(ca_out_h, [B, S_IMG, D], ctx)

    var ca_gate3 = reshape(ca_gate, g3.copy(), ctx)
    var ca_gated = mul(ca_out, ca_gate3, ctx)
    var x_after_ca = add(x_after_sa, ca_gated, ctx)

    # ── sub-block 3: MLP (GELU) ──────────────────────────────────────────────
    var mlp_h_ = linear(t_silu, w.mlp_mod1[], Optional[Tensor](None), ctx)
    var mlp_modout = linear(mlp_h_, w.mlp_mod2[], Optional[Tensor](None), ctx)
    var mlp_added = add(mlp_modout, base_t, ctx)
    var mlp_shift = slice(mlp_added, 1, 0, D, ctx)
    var mlp_scalev = slice(mlp_added, 1, D, D, ctx)
    var mlp_gate = slice(mlp_added, 1, 2 * D, D, ctx)

    var mlp_ln = layer_norm(x_after_ca, ln_ones, ln_zeros, eps, ctx)
    var mlp_xmod = _adaln_pre(x_after_ca, mlp_shift, mlp_scalev, ln_ones, ln_zeros, eps, ctx)
    var mlp_xmod_h = mlp_xmod.to_host(ctx)                # [B*S_img,D] LoRA input (mlp1)

    var mlp_h1_base = linear(mlp_xmod, w.mlp1[], Optional[Tensor](None), ctx).to_host(ctx)
    var mlp_h1_h = anima_lora_apply(mlp_h1_base, mlp_xmod_h, lora.mlp1, B * S_IMG, ctx)
    var mlp_h1 = _t(mlp_h1_h, [B, S_IMG, F], ctx)
    var mlp_ha = gelu(mlp_h1, ctx)
    var mlp_ha_h = mlp_ha.to_host(ctx)                    # [B*S_img,F] LoRA input (mlp2)

    var mlp_out_base = linear(mlp_ha, w.mlp2[], Optional[Tensor](None), ctx).to_host(ctx)
    var mlp_out_h = anima_lora_apply(mlp_out_base, mlp_ha_h, lora.mlp2, B * S_IMG, ctx)
    var mlp_out = _t(mlp_out_h, [B, S_IMG, D], ctx)

    var mlp_gate3 = reshape(mlp_gate, g3.copy(), ctx)
    var mlp_gated = mul(mlp_out, mlp_gate3, ctx)
    var x_final = add(x_after_ca, mlp_gated, ctx)

    var out_host = x_final.to_host(ctx)

    # ── save activations (LoRA-MODIFIED) — same struct contract as base block ──
    var sa_xmod_a = TArc(sa_xmod^)
    var ca_xmod_a = TArc(ca_xmod^)
    var mlp_xmod_a = TArc(mlp_xmod^)
    var t_silu_a = TArc(t_silu^)

    var sa_sub = _SubSaved(
        TArc(x_f32^), TArc(sa_ln^), sa_xmod_a.copy(),
        TArc(sa_shift^), TArc(sa_scalev^), TArc(sa_gate^),
        t_silu_a.copy(), TArc(sa_h^), TArc(sa_out^),
    )
    var ca_sub = _SubSaved(
        TArc(x_after_sa^), TArc(ca_ln^), ca_xmod_a.copy(),
        TArc(ca_shift^), TArc(ca_scalev^), TArc(ca_gate^),
        t_silu_a.copy(), TArc(ca_h^), TArc(ca_out^),
    )
    var mlp_sub = _SubSaved(
        TArc(x_after_ca^), TArc(mlp_ln^), mlp_xmod_a.copy(),
        TArc(mlp_shift^), TArc(mlp_scalev^), TArc(mlp_gate^),
        t_silu_a.copy(), TArc(mlp_h_^), TArc(mlp_out^),
    )
    var ctx_t_a = TArc(ctx_t^)
    var sa_attn = _AttnSaved(
        TArc(sa_qrope^), TArc(sa_krope^), TArc(sa_v4^),
        TArc(sa_q4^), TArc(sa_k4^), TArc(sa_qrms^), TArc(sa_krms^),
        TArc(sa_attflat^), sa_xmod_a.copy(), sa_xmod_a.copy(),
    )
    var ca_qrms_a = TArc(ca_qrms^)
    var ca_krms_a = TArc(ca_krms^)
    var ca_attn = _AttnSaved(
        ca_qrms_a.copy(), ca_krms_a.copy(), TArc(ca_v4^),
        TArc(ca_q4^), TArc(ca_k4^), ca_qrms_a.copy(), ca_krms_a.copy(),
        TArc(ca_attflat^), ca_xmod_a.copy(), ctx_t_a.copy(),
    )
    var saved = AnimaBlockSaved(
        sa_sub^, ca_sub^, mlp_sub^, sa_attn^, ca_attn^, TArc(mlp_h1^), TArc(mlp_ha^),
    )
    return AnimaBlockForward(out_host^, saved^)


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE-RESIDENT LoRA-aware block forward. Mirrors anima_block_lora_forward but:
#   * x_arc enters as a DEVICE Tensor (no host List[Float32] carrier);
#   * the 10 LoRA projections use anima_lora_apply_device (no to_host per proj);
#   * the output stays a DEVICE Tensor (TArc) — no per-block to_host/from_host;
#   * the F32 residual stream is preserved (BF16 base weights give F32 via
#     linear's mixed_base path) so parity vs the host path holds at F32 LoRA.
# Saves the SAME AnimaBlockSaved struct contract, so the existing device backward
# consumes it unchanged.
# ══════════════════════════════════════════════════════════════════════════════
struct AnimaBlockForwardLoraTensor(Movable):
    var out: TArc
    var saved: AnimaBlockSaved

    def __init__(out self, var out: TArc, var saved: AnimaBlockSaved):
        self.out = out^
        self.saved = saved^


def anima_block_lora_forward_device_tensor[
    H: Int, Dh: Int, S_IMG: Int, S_TXT: Int
](
    x_arc: TArc,                 # [B,S_img,D] F32 residual-stream input (DEVICE)
    t_silu_arc: TArc,            # [B,2048] silu(t_cond) — uploaded ONCE per step
    base_t_arc: TArc,            # [B,6144] base_adaln — uploaded ONCE per step
    ctx_t_arc: TArc,             # [B,S_txt,1024] frozen text context — ONCE per step
    w: AnimaBlockWeights, lora: AnimaBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    B: Int, D: Int, JOINT: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> AnimaBlockForwardLoraTensor:
    var ln_ones = _t(_ones(D), [D], ctx)
    var ln_zeros = _t(_zeros(D), [D], ctx)
    var sa_scale = Float32(1.0) / fsqrt(Float32(Dh))

    # ── sub-block 1: SELF-ATTENTION ──────────────────────────────────────────
    var sa_h = linear(t_silu_arc[], w.sa_mod1[], Optional[Tensor](None), ctx)   # [B,256]
    var sa_modout = linear(sa_h, w.sa_mod2[], Optional[Tensor](None), ctx)
    var sa_added = add(sa_modout, base_t_arc[], ctx)
    var sa_shift = slice(sa_added, 1, 0, D, ctx)
    var sa_scalev = slice(sa_added, 1, D, D, ctx)
    var sa_gate = slice(sa_added, 1, 2 * D, D, ctx)

    var sa_ln = layer_norm(x_arc[], ln_ones, ln_zeros, eps, ctx)
    var sa_xmod = _adaln_pre_dev(x_arc[], sa_shift, sa_scalev, ln_ones, ln_zeros, B, D, eps, ctx)

    var sa_q_base = linear(sa_xmod, w.sa_q[], Optional[Tensor](None), ctx)
    var sa_k_base = linear(sa_xmod, w.sa_k[], Optional[Tensor](None), ctx)
    var sa_v_base = linear(sa_xmod, w.sa_v[], Optional[Tensor](None), ctx)
    var sa_q_f = anima_lora_apply_device(sa_q_base^, sa_xmod, lora.sa_q, B * S_IMG, ctx)
    var sa_k_f = anima_lora_apply_device(sa_k_base^, sa_xmod, lora.sa_k, B * S_IMG, ctx)
    var sa_v_f = anima_lora_apply_device(sa_v_base^, sa_xmod, lora.sa_v, B * S_IMG, ctx)

    var sh4 = List[Int](); sh4.append(B); sh4.append(S_IMG); sh4.append(H); sh4.append(Dh)
    var sa_q4 = reshape(sa_q_f, sh4.copy(), ctx)
    var sa_k4 = reshape(sa_k_f, sh4.copy(), ctx)
    var sa_v4 = reshape(sa_v_f, sh4.copy(), ctx)
    var sa_qrms = _rms_per_head(sa_q4, w.sa_qn[], ctx)
    var sa_krms = _rms_per_head(sa_k4, w.sa_kn[], ctx)
    var sa_qrope = rope_halfsplit(sa_qrms, cos, sin, ctx)
    var sa_krope = rope_halfsplit(sa_krms, cos, sin, ctx)
    var sa_att = sdpa_nomask[1, S_IMG, H, Dh](sa_qrope, sa_krope, sa_v4, sa_scale, ctx)
    var saf = List[Int](); saf.append(B); saf.append(S_IMG); saf.append(D)
    var sa_attflat = reshape(sa_att, saf.copy(), ctx)

    var sa_out_base = linear(sa_attflat, w.sa_out[], Optional[Tensor](None), ctx)
    var sa_out = anima_lora_apply_device(sa_out_base^, sa_attflat, lora.sa_out, B * S_IMG, ctx)

    var g3 = List[Int](); g3.append(B); g3.append(1); g3.append(D)
    var sa_gate3 = reshape(sa_gate, g3.copy(), ctx)
    var sa_gated = mul(sa_out, sa_gate3, ctx)
    var x_after_sa = add(x_arc[], sa_gated, ctx)

    # ── sub-block 2: CROSS-ATTENTION (no RoPE, rectangular SDPA, no mask) ─────
    var ca_h = linear(t_silu_arc[], w.ca_mod1[], Optional[Tensor](None), ctx)
    var ca_modout = linear(ca_h, w.ca_mod2[], Optional[Tensor](None), ctx)
    var ca_added = add(ca_modout, base_t_arc[], ctx)
    var ca_shift = slice(ca_added, 1, 0, D, ctx)
    var ca_scalev = slice(ca_added, 1, D, D, ctx)
    var ca_gate = slice(ca_added, 1, 2 * D, D, ctx)

    var ca_ln = layer_norm(x_after_sa, ln_ones, ln_zeros, eps, ctx)
    var ca_xmod = _adaln_pre_dev(x_after_sa, ca_shift, ca_scalev, ln_ones, ln_zeros, B, D, eps, ctx)

    var ca_q_base = linear(ca_xmod, w.ca_q[], Optional[Tensor](None), ctx)
    var ca_k_base = linear(ctx_t_arc[], w.ca_k[], Optional[Tensor](None), ctx)
    var ca_v_base = linear(ctx_t_arc[], w.ca_v[], Optional[Tensor](None), ctx)
    var ca_q_f = anima_lora_apply_device(ca_q_base^, ca_xmod, lora.ca_q, B * S_IMG, ctx)
    var ca_k_f = anima_lora_apply_device(ca_k_base^, ctx_t_arc[], lora.ca_k, B * S_TXT, ctx)
    var ca_v_f = anima_lora_apply_device(ca_v_base^, ctx_t_arc[], lora.ca_v, B * S_TXT, ctx)

    var caq4s = List[Int](); caq4s.append(B); caq4s.append(S_IMG); caq4s.append(H); caq4s.append(Dh)
    var cak4s = List[Int](); cak4s.append(B); cak4s.append(S_TXT); cak4s.append(H); cak4s.append(Dh)
    var ca_q4 = reshape(ca_q_f, caq4s.copy(), ctx)
    var ca_k4 = reshape(ca_k_f, cak4s.copy(), ctx)
    var ca_v4 = reshape(ca_v_f, cak4s.copy(), ctx)
    var ca_qrms = _rms_per_head(ca_q4, w.ca_qn[], ctx)
    var ca_krms = _rms_per_head(ca_k4, w.ca_kn[], ctx)
    var ca_att = sdxl_sdpa[1, S_IMG, S_TXT, H, Dh](ca_qrms, ca_krms, ca_v4, sa_scale, ctx)
    var caf = List[Int](); caf.append(B); caf.append(S_IMG); caf.append(D)
    var ca_attflat = reshape(ca_att, caf.copy(), ctx)

    var ca_out_base = linear(ca_attflat, w.ca_out[], Optional[Tensor](None), ctx)
    var ca_out = anima_lora_apply_device(ca_out_base^, ca_attflat, lora.ca_out, B * S_IMG, ctx)

    var ca_gate3 = reshape(ca_gate, g3.copy(), ctx)
    var ca_gated = mul(ca_out, ca_gate3, ctx)
    var x_after_ca = add(x_after_sa, ca_gated, ctx)

    # ── sub-block 3: MLP (GELU) ──────────────────────────────────────────────
    var mlp_h_ = linear(t_silu_arc[], w.mlp_mod1[], Optional[Tensor](None), ctx)
    var mlp_modout = linear(mlp_h_, w.mlp_mod2[], Optional[Tensor](None), ctx)
    var mlp_added = add(mlp_modout, base_t_arc[], ctx)
    var mlp_shift = slice(mlp_added, 1, 0, D, ctx)
    var mlp_scalev = slice(mlp_added, 1, D, D, ctx)
    var mlp_gate = slice(mlp_added, 1, 2 * D, D, ctx)

    var mlp_ln = layer_norm(x_after_ca, ln_ones, ln_zeros, eps, ctx)
    var mlp_xmod = _adaln_pre_dev(x_after_ca, mlp_shift, mlp_scalev, ln_ones, ln_zeros, B, D, eps, ctx)

    var mlp_h1_base = linear(mlp_xmod, w.mlp1[], Optional[Tensor](None), ctx)
    var mlp_h1 = anima_lora_apply_device(mlp_h1_base^, mlp_xmod, lora.mlp1, B * S_IMG, ctx)
    var mlp_ha = gelu(mlp_h1, ctx)

    var mlp_out_base = linear(mlp_ha, w.mlp2[], Optional[Tensor](None), ctx)
    var mlp_out = anima_lora_apply_device(mlp_out_base^, mlp_ha, lora.mlp2, B * S_IMG, ctx)

    var mlp_gate3 = reshape(mlp_gate, g3.copy(), ctx)
    var mlp_gated = mul(mlp_out, mlp_gate3, ctx)
    var x_final = add(x_after_ca, mlp_gated, ctx)

    # ── save activations (LoRA-MODIFIED) — same struct contract as base block ──
    var sa_xmod_a = TArc(sa_xmod^)
    var ca_xmod_a = TArc(ca_xmod^)
    var mlp_xmod_a = TArc(mlp_xmod^)
    var t_silu_a = t_silu_arc.copy()

    var sa_sub = _SubSaved(
        x_arc.copy(), TArc(sa_ln^), sa_xmod_a.copy(),
        TArc(sa_shift^), TArc(sa_scalev^), TArc(sa_gate^),
        t_silu_a.copy(), TArc(sa_h^), TArc(sa_out^),
    )
    var ca_sub = _SubSaved(
        TArc(x_after_sa^), TArc(ca_ln^), ca_xmod_a.copy(),
        TArc(ca_shift^), TArc(ca_scalev^), TArc(ca_gate^),
        t_silu_a.copy(), TArc(ca_h^), TArc(ca_out^),
    )
    var mlp_sub = _SubSaved(
        TArc(x_after_ca^), TArc(mlp_ln^), mlp_xmod_a.copy(),
        TArc(mlp_shift^), TArc(mlp_scalev^), TArc(mlp_gate^),
        t_silu_a.copy(), TArc(mlp_h_^), TArc(mlp_out^),
    )
    var ctx_t_a = ctx_t_arc.copy()
    var sa_attn = _AttnSaved(
        TArc(sa_qrope^), TArc(sa_krope^), TArc(sa_v4^),
        TArc(sa_q4^), TArc(sa_k4^), TArc(sa_qrms^), TArc(sa_krms^),
        TArc(sa_attflat^), sa_xmod_a.copy(), sa_xmod_a.copy(),
    )
    var ca_qrms_a = TArc(ca_qrms^)
    var ca_krms_a = TArc(ca_krms^)
    var ca_attn = _AttnSaved(
        ca_qrms_a.copy(), ca_krms_a.copy(), TArc(ca_v4^),
        TArc(ca_q4^), TArc(ca_k4^), ca_qrms_a.copy(), ca_krms_a.copy(),
        TArc(ca_attflat^), ca_xmod_a.copy(), ctx_t_a.copy(),
    )
    var saved = AnimaBlockSaved(
        sa_sub^, ca_sub^, mlp_sub^, sa_attn^, ca_attn^, TArc(mlp_h1^), TArc(mlp_ha^),
    )
    return AnimaBlockForwardLoraTensor(TArc(x_final^), saved^)


# AdaLN-pre on device tensors: (1+scale)*LayerNorm(x,no-affine,eps)+shift.
# Same math as block._adaln_pre but takes B,D explicitly (no host hop).
def _adaln_pre_dev(
    x_f32: Tensor, shift: Tensor, scale: Tensor,
    ln_ones: Tensor, ln_zeros: Tensor, B: Int, D: Int, eps: Float32, ctx: DeviceContext,
) raises -> Tensor:
    var normed = layer_norm(x_f32, ln_ones, ln_zeros, eps, ctx)
    var s3 = List[Int](); s3.append(B); s3.append(1); s3.append(D)
    var scale_3d = reshape(scale, s3.copy(), ctx)
    var shift_3d = reshape(shift, s3.copy(), ctx)
    var one = _t(_ones(B * D), [B, 1, D], ctx)
    var factor = add(scale_3d, one, ctx)
    var scaled = mul(normed, factor, ctx)
    return add(scaled, shift_3d, ctx)


# ── LoRA-aware block backward ─────────────────────────────────────────────────
# Mirrors anima_block_backward EXACTLY (the INLINED hand-chained backward). At each
# of the 10 trained projections the d_y flowing INTO that projection (the same grad
# the base linear_backward sees) is fed to anima_lora_bwd to produce d_A/d_B + the
# LoRA d_x contribution, which is SUMMED into the projection-input grad. For sa
# q/k/v this contributes to d_sa_xmod; for sa out to d_sa_attflat; for ca q to
# d_ca_xmod; for ca out to d_ca_attflat; for mlp1 to d_mlp_xmod; for mlp2 to
# d_mlp_ha. For ca k/v the projection input is FROZEN context, so the LoRA d_x is
# DISCARDED (matching the base block, which discards d_input on those linears).
struct AnimaBlockLoraBackward(Movable):
    var base: AnimaBlockGrads
    var lora: AnimaBlockLoraGrads

    def __init__(out self, var base: AnimaBlockGrads, var lora: AnimaBlockLoraGrads):
        self.base = base^
        self.lora = lora^


# helper: given the base linear_backward d_x [M,in] (host) and a LoRA slot, append
# the LoRA d_a/d_b to the slot lists and return the SUMMED d_x. `keep_dx` controls
# whether the LoRA d_x is summed into the returned d_x (False for frozen-context
# inputs where the base d_x is itself discarded by the caller).
def _proj_lora_grads(
    d_y_h: List[Float32], x_in_h: List[Float32], base_dx_h: List[Float32],
    lo: Optional[LoraAdapter], slot: Int,
    M: Int, in_f: Int,
    mut d_a_slots: List[List[Float32]], mut d_b_slots: List[List[Float32]],
    keep_dx: Bool, ctx: DeviceContext,
) raises -> List[Float32]:
    if not lo:
        return base_dx_h.copy()
    var lg = anima_lora_bwd(d_y_h, x_in_h, lo.value(), M, ctx)
    d_a_slots[slot] = lg.d_a.copy()
    d_b_slots[slot] = lg.d_b.copy()
    if keep_dx:
        return _add_lists(base_dx_h, lg.d_x)
    return base_dx_h.copy()


def anima_block_lora_backward[
    H: Int, Dh: Int, S_IMG: Int, S_TXT: Int
](
    d_out: List[Float32],      # [B,S_img,D]
    saved: AnimaBlockSaved,
    w: AnimaBlockWeights, lora: AnimaBlockLora,
    cos: Tensor, sin: Tensor,  # [B*S_img*H, Dh/2] F32
    B: Int, D: Int, JOINT: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> AnimaBlockLoraBackward:
    var ln_ones = _t(_ones(D), [D], ctx)
    var sa_scale = Float32(1.0) / fsqrt(Float32(Dh))

    var d_a_slots = List[List[Float32]]()
    var d_b_slots = List[List[Float32]]()
    for _ in range(ANIMA_SLOTS):
        d_a_slots.append(List[Float32]())
        d_b_slots.append(List[Float32]())

    var d_x = _t(d_out, [B, S_IMG, D], ctx)
    var d_t_silu_acc = _t(_zeros(B * ANIMA_HIDDEN), [B, ANIMA_HIDDEN], ctx)

    # ─────────────────────────────── MLP sub-block (last in forward) ──────────
    var mlp_gg = _gate_residual_bwd(d_x, saved.mlp.gate[], saved.mlp.sub_out[], B, S_IMG, D, ctx)
    var d_x_after_ca = d_x^   # residual passthrough
    # mlp2: mlp_out = linear(mlp_ha, mlp2)[+LoRA(mlp2)]   in=F out=D ; input mlp_ha
    var mlp_lb2 = linear_backward(mlp_gg.d_sub_out, saved.mlp_ha[], w.mlp2[], B * S_IMG, F, D, ctx)
    var d_mlp2 = mlp_lb2.d_w.to_host(ctx)
    var mlp2_dy_h = mlp_gg.d_sub_out.to_host(ctx)
    var mlp_ha_h = saved.mlp_ha[].to_host(ctx)
    var d_mlp_ha_base = mlp_lb2.d_x.to_host(ctx)
    var d_mlp_ha_h = _proj_lora_grads(
        mlp2_dy_h, mlp_ha_h, d_mlp_ha_base, lora.mlp2, SLOT_MLP2,
        B * S_IMG, F, d_a_slots, d_b_slots, True, ctx,
    )
    var d_mlp_ha = _t(d_mlp_ha_h, [B, S_IMG, F], ctx)
    var d_mlp_h1 = gelu_backward(d_mlp_ha, saved.mlp_h[], ctx)
    # mlp1: mlp_h1 = linear(mlp_xmod, mlp1)[+LoRA(mlp1)]  in=D out=F ; input mlp_xmod
    var mlp_lb1 = linear_backward(d_mlp_h1, saved.mlp.x_mod[], w.mlp1[], B * S_IMG, D, F, ctx)
    var d_mlp1 = mlp_lb1.d_w.to_host(ctx)
    var mlp1_dy_h = d_mlp_h1.to_host(ctx)
    var mlp_xmod_h = saved.mlp.x_mod[].to_host(ctx)
    var d_mlp_xmod_base = mlp_lb1.d_x.to_host(ctx)
    var d_mlp_xmod_h = _proj_lora_grads(
        mlp1_dy_h, mlp_xmod_h, d_mlp_xmod_base, lora.mlp1, SLOT_MLP1,
        B * S_IMG, D, d_a_slots, d_b_slots, True, ctx,
    )
    var d_mlp_xmod = _t(d_mlp_xmod_h, [B, S_IMG, D], ctx)
    # adaln-pre + adaln-mod backward (identical to base block)
    var mlp_mb = modulate_backward(d_mlp_xmod, saved.mlp.ln[], saved.mlp.scale[], ctx)
    var mlp_dxin = layer_norm_backward_dx(mlp_mb.d_x, saved.mlp.x_in[], ln_ones, eps, ctx)
    d_x_after_ca = add(d_x_after_ca, mlp_dxin, ctx)
    var mlp_dadd = concat(1, ctx, mlp_mb.d_shift, mlp_mb.d_scale, mlp_gg.d_gate)
    var mlp_mlb2 = linear_backward(mlp_dadd, saved.mlp.mod_h[], w.mlp_mod2[], B, 256, 3 * D, ctx)
    var d_mlp_mod2 = mlp_mlb2.d_w.to_host(ctx)
    var mlp_mlb1 = linear_backward(mlp_mlb2.d_x, saved.mlp.t_silu[], w.mlp_mod1[], B, ANIMA_HIDDEN, 256, ctx)
    var d_mlp_mod1 = mlp_mlb1.d_w.to_host(ctx)
    d_t_silu_acc = add(d_t_silu_acc, mlp_mlb1.d_x, ctx)

    # ─────────────────────────────── CROSS-ATTENTION sub-block ────────────────
    var ca_gg = _gate_residual_bwd(d_x_after_ca, saved.ca.gate[], saved.ca.sub_out[], B, S_IMG, D, ctx)
    var d_x_after_sa = d_x_after_ca^
    # ca out: ca_out = linear(attn_flat, ca_out)[+LoRA(ca_out)]  in=D out=D ; input attn_flat
    var ca_lbout = linear_backward(ca_gg.d_sub_out, saved.ca_attn.attn_flat[], w.ca_out[], B * S_IMG, D, D, ctx)
    var d_ca_out = ca_lbout.d_w.to_host(ctx)
    var ca_out_dy_h = ca_gg.d_sub_out.to_host(ctx)
    var ca_attflat_h = saved.ca_attn.attn_flat[].to_host(ctx)
    var d_ca_attflat_base = ca_lbout.d_x.to_host(ctx)
    var d_ca_attflat_h = _proj_lora_grads(
        ca_out_dy_h, ca_attflat_h, d_ca_attflat_base, lora.ca_out, SLOT_CA_O,
        B * S_IMG, D, d_a_slots, d_b_slots, True, ctx,
    )
    var ca_af4 = List[Int](); ca_af4.append(B); ca_af4.append(S_IMG); ca_af4.append(H); ca_af4.append(Dh)
    var d_ca_att4 = reshape(_t(d_ca_attflat_h, [B, S_IMG, D], ctx), ca_af4.copy(), ctx)
    var ca_sb = sdpa_backward_rect[1, S_IMG, S_TXT, H, Dh](
        saved.ca_attn.q_sdpa[], saved.ca_attn.k_sdpa[], saved.ca_attn.v4[],
        d_ca_att4, sa_scale, ctx,
    )
    var ca_q_flat = List[Int](); ca_q_flat.append(B * S_IMG * H); ca_q_flat.append(Dh)
    var ca_k_flat = List[Int](); ca_k_flat.append(B * S_TXT * H); ca_k_flat.append(Dh)
    var d_ca_q_rms_f = reshape(ca_sb.d_q, ca_q_flat.copy(), ctx)
    var d_ca_k_rms_f = reshape(ca_sb.d_k, ca_k_flat.copy(), ctx)
    var ca_qpre_f = reshape(saved.ca_attn.q_pre[], ca_q_flat.copy(), ctx)
    var ca_kpre_f = reshape(saved.ca_attn.k_pre[], ca_k_flat.copy(), ctx)
    var ca_rbq = rms_norm_backward(d_ca_q_rms_f, ca_qpre_f, w.ca_qn[], Float32(1e-6), ctx)
    var ca_rbk = rms_norm_backward(d_ca_k_rms_f, ca_kpre_f, w.ca_kn[], Float32(1e-6), ctx)
    var d_ca_qn = ca_rbq.d_g.to_host(ctx)
    var d_ca_kn = ca_rbk.d_g.to_host(ctx)
    var ca_q3 = List[Int](); ca_q3.append(B); ca_q3.append(S_IMG); ca_q3.append(D)
    var ca_kv3 = List[Int](); ca_kv3.append(B); ca_kv3.append(S_TXT); ca_kv3.append(D)
    var d_ca_q_proj = reshape(ca_rbq.d_x, ca_q3.copy(), ctx)
    var d_ca_k_proj = reshape(ca_rbk.d_x, ca_kv3.copy(), ctx)
    var d_ca_v_proj = reshape(ca_sb.d_v, ca_kv3.copy(), ctx)
    # ca q: ca_q = linear(ca_xmod, ca_q)[+LoRA(ca_q)]  in=D out=D ; input ca_xmod (trained)
    var ca_lbq = linear_backward(d_ca_q_proj, saved.ca_attn.q_ctx_in[], w.ca_q[], B * S_IMG, D, D, ctx)
    var d_ca_q = ca_lbq.d_w.to_host(ctx)
    # ca k/v: linear(context, ca_k/ca_v)[+LoRA]  in=JOINT out=D ; input FROZEN context.
    #   Base d_x flows into context (discarded). LoRA d_a/d_b are STILL trained; their
    #   d_x is also into frozen context -> keep_dx=False.
    var ca_lbk = linear_backward(d_ca_k_proj, saved.ca_attn.kv_ctx_in[], w.ca_k[], B * S_TXT, JOINT, D, ctx)
    var ca_lbv = linear_backward(d_ca_v_proj, saved.ca_attn.kv_ctx_in[], w.ca_v[], B * S_TXT, JOINT, D, ctx)
    var d_ca_k = ca_lbk.d_w.to_host(ctx)
    var d_ca_v = ca_lbv.d_w.to_host(ctx)
    var ca_xmod_h = saved.ca_attn.q_ctx_in[].to_host(ctx)
    var context_h = saved.ca_attn.kv_ctx_in[].to_host(ctx)
    var ca_q_dy_h = d_ca_q_proj.to_host(ctx)
    var ca_k_dy_h = d_ca_k_proj.to_host(ctx)
    var ca_v_dy_h = d_ca_v_proj.to_host(ctx)
    var ca_lbq_dx_h = ca_lbq.d_x.to_host(ctx)
    var d_ca_xmod_h = _proj_lora_grads(
        ca_q_dy_h, ca_xmod_h, ca_lbq_dx_h, lora.ca_q, SLOT_CA_Q,
        B * S_IMG, D, d_a_slots, d_b_slots, True, ctx,
    )
    var ca_zero_kv = _zeros(B * S_TXT * JOINT)   # frozen-context d_x is discarded
    _ = _proj_lora_grads(
        ca_k_dy_h, context_h, ca_zero_kv.copy(), lora.ca_k, SLOT_CA_K,
        B * S_TXT, JOINT, d_a_slots, d_b_slots, False, ctx,
    )
    _ = _proj_lora_grads(
        ca_v_dy_h, context_h, ca_zero_kv.copy(), lora.ca_v, SLOT_CA_V,
        B * S_TXT, JOINT, d_a_slots, d_b_slots, False, ctx,
    )
    var d_ca_xmod = _t(d_ca_xmod_h, [B, S_IMG, D], ctx)
    var ca_mb = modulate_backward(d_ca_xmod, saved.ca.ln[], saved.ca.scale[], ctx)
    var ca_dxin = layer_norm_backward_dx(ca_mb.d_x, saved.ca.x_in[], ln_ones, eps, ctx)
    d_x_after_sa = add(d_x_after_sa, ca_dxin, ctx)
    var ca_dadd = concat(1, ctx, ca_mb.d_shift, ca_mb.d_scale, ca_gg.d_gate)
    var ca_mlb2 = linear_backward(ca_dadd, saved.ca.mod_h[], w.ca_mod2[], B, 256, 3 * D, ctx)
    var d_ca_mod2 = ca_mlb2.d_w.to_host(ctx)
    var ca_mlb1 = linear_backward(ca_mlb2.d_x, saved.ca.t_silu[], w.ca_mod1[], B, ANIMA_HIDDEN, 256, ctx)
    var d_ca_mod1 = ca_mlb1.d_w.to_host(ctx)
    d_t_silu_acc = add(d_t_silu_acc, ca_mlb1.d_x, ctx)

    # ─────────────────────────────── SELF-ATTENTION sub-block ─────────────────
    var sa_gg = _gate_residual_bwd(d_x_after_sa, saved.sa.gate[], saved.sa.sub_out[], B, S_IMG, D, ctx)
    var d_x_in = d_x_after_sa^
    # sa out: sa_out = linear(attn_flat, sa_out)[+LoRA(sa_out)]  in=D out=D ; input attn_flat
    var sa_lbout = linear_backward(sa_gg.d_sub_out, saved.sa_attn.attn_flat[], w.sa_out[], B * S_IMG, D, D, ctx)
    var d_sa_out = sa_lbout.d_w.to_host(ctx)
    var sa_out_dy_h = sa_gg.d_sub_out.to_host(ctx)
    var sa_attflat_h = saved.sa_attn.attn_flat[].to_host(ctx)
    var d_sa_attflat_base = sa_lbout.d_x.to_host(ctx)
    var d_sa_attflat_h = _proj_lora_grads(
        sa_out_dy_h, sa_attflat_h, d_sa_attflat_base, lora.sa_out, SLOT_SA_O,
        B * S_IMG, D, d_a_slots, d_b_slots, True, ctx,
    )
    var sa_af4 = List[Int](); sa_af4.append(B); sa_af4.append(S_IMG); sa_af4.append(H); sa_af4.append(Dh)
    var d_sa_att4 = reshape(_t(d_sa_attflat_h, [B, S_IMG, D], ctx), sa_af4.copy(), ctx)
    var sa_sb = sdpa_backward[1, S_IMG, H, Dh](
        saved.sa_attn.q_sdpa[], saved.sa_attn.k_sdpa[], saved.sa_attn.v4[],
        d_sa_att4, sa_scale, ctx,
    )
    var sa_q_flat = List[Int](); sa_q_flat.append(B * S_IMG * H); sa_q_flat.append(Dh)
    var d_sa_q_rope_f = reshape(sa_sb.d_q, sa_q_flat.copy(), ctx)
    var d_sa_k_rope_f = reshape(sa_sb.d_k, sa_q_flat.copy(), ctx)
    var d_sa_q_rms_f = rope_backward(d_sa_q_rope_f, cos, sin, False, ctx)
    var d_sa_k_rms_f = rope_backward(d_sa_k_rope_f, cos, sin, False, ctx)
    var sa_qpre_f = reshape(saved.sa_attn.q_pre[], sa_q_flat.copy(), ctx)
    var sa_kpre_f = reshape(saved.sa_attn.k_pre[], sa_q_flat.copy(), ctx)
    var sa_rbq = rms_norm_backward(d_sa_q_rms_f, sa_qpre_f, w.sa_qn[], Float32(1e-6), ctx)
    var sa_rbk = rms_norm_backward(d_sa_k_rms_f, sa_kpre_f, w.sa_kn[], Float32(1e-6), ctx)
    var d_sa_qn = sa_rbq.d_g.to_host(ctx)
    var d_sa_kn = sa_rbk.d_g.to_host(ctx)
    var sa3 = List[Int](); sa3.append(B); sa3.append(S_IMG); sa3.append(D)
    var d_sa_q_proj = reshape(sa_rbq.d_x, sa3.copy(), ctx)
    var d_sa_k_proj = reshape(sa_rbk.d_x, sa3.copy(), ctx)
    var d_sa_v_proj = reshape(sa_sb.d_v, sa3.copy(), ctx)
    # sa q/k/v: linear(sa_xmod, sa_q/k/v)[+LoRA]  in=D out=D ; input sa_xmod (trained).
    # All three feed sa_xmod -> sum base d_x then sum each LoRA d_x via _proj_lora_grads.
    var sa_lbq = linear_backward(d_sa_q_proj, saved.sa_attn.q_ctx_in[], w.sa_q[], B * S_IMG, D, D, ctx)
    var sa_lbk = linear_backward(d_sa_k_proj, saved.sa_attn.q_ctx_in[], w.sa_k[], B * S_IMG, D, D, ctx)
    var sa_lbv = linear_backward(d_sa_v_proj, saved.sa_attn.q_ctx_in[], w.sa_v[], B * S_IMG, D, D, ctx)
    var d_sa_q = sa_lbq.d_w.to_host(ctx)
    var d_sa_k = sa_lbk.d_w.to_host(ctx)
    var d_sa_v = sa_lbv.d_w.to_host(ctx)
    var sa_xmod_h = saved.sa_attn.q_ctx_in[].to_host(ctx)
    var sa_q_dy_h = d_sa_q_proj.to_host(ctx)
    var sa_k_dy_h = d_sa_k_proj.to_host(ctx)
    var sa_v_dy_h = d_sa_v_proj.to_host(ctx)
    var d_sa_q_dx = _proj_lora_grads(
        sa_q_dy_h, sa_xmod_h, sa_lbq.d_x.to_host(ctx), lora.sa_q, SLOT_SA_Q,
        B * S_IMG, D, d_a_slots, d_b_slots, True, ctx,
    )
    var d_sa_k_dx = _proj_lora_grads(
        sa_k_dy_h, sa_xmod_h, sa_lbk.d_x.to_host(ctx), lora.sa_k, SLOT_SA_K,
        B * S_IMG, D, d_a_slots, d_b_slots, True, ctx,
    )
    var d_sa_v_dx = _proj_lora_grads(
        sa_v_dy_h, sa_xmod_h, sa_lbv.d_x.to_host(ctx), lora.sa_v, SLOT_SA_V,
        B * S_IMG, D, d_a_slots, d_b_slots, True, ctx,
    )
    var d_sa_xmod = add(add(_t(d_sa_q_dx, [B, S_IMG, D], ctx), _t(d_sa_k_dx, [B, S_IMG, D], ctx), ctx),
                        _t(d_sa_v_dx, [B, S_IMG, D], ctx), ctx)
    var sa_mb = modulate_backward(d_sa_xmod, saved.sa.ln[], saved.sa.scale[], ctx)
    var sa_dxin = layer_norm_backward_dx(sa_mb.d_x, saved.sa.x_in[], ln_ones, eps, ctx)
    d_x_in = add(d_x_in, sa_dxin, ctx)
    var sa_dadd = concat(1, ctx, sa_mb.d_shift, sa_mb.d_scale, sa_gg.d_gate)
    var sa_mlb2 = linear_backward(sa_dadd, saved.sa.mod_h[], w.sa_mod2[], B, 256, 3 * D, ctx)
    var d_sa_mod2 = sa_mlb2.d_w.to_host(ctx)
    var sa_mlb1 = linear_backward(sa_mlb2.d_x, saved.sa.t_silu[], w.sa_mod1[], B, ANIMA_HIDDEN, 256, ctx)
    var d_sa_mod1 = sa_mlb1.d_w.to_host(ctx)
    d_t_silu_acc = add(d_t_silu_acc, sa_mlb1.d_x, ctx)

    var d_x_host = d_x_in.to_host(ctx)
    var d_t_silu_host = d_t_silu_acc.to_host(ctx)

    var base = AnimaBlockGrads(
        d_x_host^, d_t_silu_host^,
        d_sa_q^, d_sa_k^, d_sa_v^, d_sa_out^, d_sa_qn^, d_sa_kn^,
        d_ca_q^, d_ca_k^, d_ca_v^, d_ca_out^, d_ca_qn^, d_ca_kn^,
        d_mlp1^, d_mlp2^,
        d_sa_mod1^, d_sa_mod2^, d_ca_mod1^, d_ca_mod2^, d_mlp_mod1^, d_mlp_mod2^,
    )
    return AnimaBlockLoraBackward(base^, AnimaBlockLoraGrads(d_a_slots^, d_b_slots^))


# ══════════════════════════════════════════════════════════════════════════════
# DEVICE-RESIDENT LoRA-aware block backward. Mirrors anima_block_lora_backward but:
#   * d_out enters as a DEVICE Tensor (no host carrier);
#   * frozen base projection d_x via linear_backward_dx (no discarded d_w matmul);
#   * frozen per-head RMSNorm (q/k_norm) via rms_norm_backward_dx (no d_g reduction);
#   * frozen AdaLN LayerNorm-no-affine via layer_norm_backward_dx;
#   * the AdaLN-mod Linears (sa/ca/mlp_mod1/2) use linear_backward_dx only (their
#     d_w is discarded for LoRA) — d_t_silu still threaded for the parity gate;
#   * LoRA d_a/d_b returned as DEVICE TArc lists (10 slots, slot order canonical);
#   * d_x (block input) + d_t_silu returned as DEVICE Tensors — no per-block to_host.
# `trace=True` prints per-phase timing (mlp / cross / self) for profiling.
# ══════════════════════════════════════════════════════════════════════════════
struct AnimaBlockLoraTensorBackward(Movable):
    var d_x: TArc          # [B,S_img,D] grad into block input
    var d_t_silu: TArc     # [B,2048] grad into shared silu(t_cond)
    var d_a: List[TArc]    # 10 slots
    var d_b: List[TArc]

    def __init__(
        out self, var d_x: TArc, var d_t_silu: TArc,
        var d_a: List[TArc], var d_b: List[TArc],
    ):
        self.d_x = d_x^
        self.d_t_silu = d_t_silu^
        self.d_a = d_a^
        self.d_b = d_b^


def anima_block_lora_backward_device_tensors[
    H: Int, Dh: Int, S_IMG: Int, S_TXT: Int
](
    d_out: Tensor,             # [B,S_img,D] DEVICE
    saved: AnimaBlockSaved,
    w: AnimaBlockWeights, lora: AnimaBlockLoraDevice,
    cos: Tensor, sin: Tensor,
    B: Int, D: Int, JOINT: Int, F: Int, eps: Float32,
    ctx: DeviceContext,
    trace: Bool = False,
) raises -> AnimaBlockLoraTensorBackward:
    var ln_ones = _t(_ones(D), [D], ctx)
    var sa_scale = Float32(1.0) / fsqrt(Float32(Dh))
    var ts0 = perf_counter_ns()

    # per-slot LoRA grad device tensors (filled below).
    var d_a = List[TArc]()
    var d_b = List[TArc]()
    for _ in range(ANIMA_SLOTS):
        d_a.append(TArc(_t(_zeros(1), [1], ctx)))
        d_b.append(TArc(_t(_zeros(1), [1], ctx)))

    var d_t_silu_acc = _t(_zeros(B * ANIMA_HIDDEN), [B, ANIMA_HIDDEN], ctx)

    # ─────────────────────────────── MLP sub-block (last in forward) ──────────
    var mlp_gg = _gate_residual_bwd(d_out, saved.mlp.gate[], saved.mlp.sub_out[], B, S_IMG, D, ctx)
    var d_x_after_ca = add(d_out, _t(_zeros(B * S_IMG * D), [B, S_IMG, D], ctx), ctx)  # residual copy of d_out
    # mlp2: mlp_out = linear(mlp_ha, mlp2)[+LoRA]  in=F out=D ; input mlp_ha
    var pg_mlp2 = _proj_bwd_with_lora_device_tensors(
        mlp_gg.d_sub_out, saved.mlp_ha[], w.mlp2[], lora.mlp2, B * S_IMG, F, D, True, ctx,
    )
    d_a[SLOT_MLP2] = pg_mlp2.d_a.copy(); d_b[SLOT_MLP2] = pg_mlp2.d_b.copy()
    var d_mlp_ha = reshape(pg_mlp2.d_x, [B, S_IMG, F], ctx)
    var d_mlp_h1 = gelu_backward(d_mlp_ha, saved.mlp_h[], ctx)
    # mlp1: mlp_h1 = linear(mlp_xmod, mlp1)[+LoRA]  in=D out=F ; input mlp_xmod
    var pg_mlp1 = _proj_bwd_with_lora_device_tensors(
        d_mlp_h1, saved.mlp.x_mod[], w.mlp1[], lora.mlp1, B * S_IMG, D, F, True, ctx,
    )
    d_a[SLOT_MLP1] = pg_mlp1.d_a.copy(); d_b[SLOT_MLP1] = pg_mlp1.d_b.copy()
    var d_mlp_xmod = reshape(pg_mlp1.d_x, [B, S_IMG, D], ctx)
    var mlp_mb = modulate_backward(d_mlp_xmod, saved.mlp.ln[], saved.mlp.scale[], ctx)
    var mlp_dxin = layer_norm_backward_dx(mlp_mb.d_x, saved.mlp.x_in[], ln_ones, eps, ctx)
    d_x_after_ca = add(d_x_after_ca, mlp_dxin, ctx)
    var mlp_dadd = concat(1, ctx, mlp_mb.d_shift, mlp_mb.d_scale, mlp_gg.d_gate)
    var d_mlp_modh = linear_backward_dx(mlp_dadd, w.mlp_mod2[], B, 256, 3 * D, ctx)
    var d_mlp_tsilu = linear_backward_dx(d_mlp_modh, w.mlp_mod1[], B, ANIMA_HIDDEN, 256, ctx)
    d_t_silu_acc = add(d_t_silu_acc, d_mlp_tsilu, ctx)
    if trace:
        ctx.synchronize()
    var ts_mlp = perf_counter_ns()

    # ─────────────────────────────── CROSS-ATTENTION sub-block ────────────────
    var ca_gg = _gate_residual_bwd(d_x_after_ca, saved.ca.gate[], saved.ca.sub_out[], B, S_IMG, D, ctx)
    var d_x_after_sa = add(d_x_after_ca, _t(_zeros(B * S_IMG * D), [B, S_IMG, D], ctx), ctx)
    # ca out: ca_out = linear(attn_flat, ca_out)[+LoRA]  in=D out=D ; input attn_flat
    var pg_caout = _proj_bwd_with_lora_device_tensors(
        ca_gg.d_sub_out, saved.ca_attn.attn_flat[], w.ca_out[], lora.ca_out,
        B * S_IMG, D, D, True, ctx,
    )
    d_a[SLOT_CA_O] = pg_caout.d_a.copy(); d_b[SLOT_CA_O] = pg_caout.d_b.copy()
    var d_ca_att4 = reshape(pg_caout.d_x, [B, S_IMG, H, Dh], ctx)
    var ca_sb = sdpa_backward_rect[1, S_IMG, S_TXT, H, Dh](
        saved.ca_attn.q_sdpa[], saved.ca_attn.k_sdpa[], saved.ca_attn.v4[],
        d_ca_att4, sa_scale, ctx,
    )
    var d_ca_q_rms_f = reshape(ca_sb.d_q, [B * S_IMG * H, Dh], ctx)
    var d_ca_k_rms_f = reshape(ca_sb.d_k, [B * S_TXT * H, Dh], ctx)
    var ca_qpre_f = reshape(saved.ca_attn.q_pre[], [B * S_IMG * H, Dh], ctx)
    var ca_kpre_f = reshape(saved.ca_attn.k_pre[], [B * S_TXT * H, Dh], ctx)
    var d_ca_q_pre = rms_norm_backward_dx(d_ca_q_rms_f, ca_qpre_f, w.ca_qn[], Float32(1e-6), ctx)
    var d_ca_k_pre = rms_norm_backward_dx(d_ca_k_rms_f, ca_kpre_f, w.ca_kn[], Float32(1e-6), ctx)
    var d_ca_q_proj = reshape(d_ca_q_pre, [B, S_IMG, D], ctx)
    var d_ca_k_proj = reshape(d_ca_k_pre, [B, S_TXT, D], ctx)
    var d_ca_v_proj = reshape(ca_sb.d_v, [B, S_TXT, D], ctx)
    # ca q: input ca_xmod (trained); ca k/v: input FROZEN context (keep_dx=False).
    var pg_caq = _proj_bwd_with_lora_device_tensors(
        d_ca_q_proj, saved.ca_attn.q_ctx_in[], w.ca_q[], lora.ca_q, B * S_IMG, D, D, True, ctx,
    )
    d_a[SLOT_CA_Q] = pg_caq.d_a.copy(); d_b[SLOT_CA_Q] = pg_caq.d_b.copy()
    var pg_cak = _proj_bwd_with_lora_device_tensors(
        d_ca_k_proj, saved.ca_attn.kv_ctx_in[], w.ca_k[], lora.ca_k, B * S_TXT, JOINT, D, False, ctx,
    )
    d_a[SLOT_CA_K] = pg_cak.d_a.copy(); d_b[SLOT_CA_K] = pg_cak.d_b.copy()
    var pg_cav = _proj_bwd_with_lora_device_tensors(
        d_ca_v_proj, saved.ca_attn.kv_ctx_in[], w.ca_v[], lora.ca_v, B * S_TXT, JOINT, D, False, ctx,
    )
    d_a[SLOT_CA_V] = pg_cav.d_a.copy(); d_b[SLOT_CA_V] = pg_cav.d_b.copy()
    var d_ca_xmod = reshape(pg_caq.d_x, [B, S_IMG, D], ctx)
    var ca_mb = modulate_backward(d_ca_xmod, saved.ca.ln[], saved.ca.scale[], ctx)
    var ca_dxin = layer_norm_backward_dx(ca_mb.d_x, saved.ca.x_in[], ln_ones, eps, ctx)
    d_x_after_sa = add(d_x_after_sa, ca_dxin, ctx)
    var ca_dadd = concat(1, ctx, ca_mb.d_shift, ca_mb.d_scale, ca_gg.d_gate)
    var d_ca_modh = linear_backward_dx(ca_dadd, w.ca_mod2[], B, 256, 3 * D, ctx)
    var d_ca_tsilu = linear_backward_dx(d_ca_modh, w.ca_mod1[], B, ANIMA_HIDDEN, 256, ctx)
    d_t_silu_acc = add(d_t_silu_acc, d_ca_tsilu, ctx)
    if trace:
        ctx.synchronize()
    var ts_ca = perf_counter_ns()

    # ─────────────────────────────── SELF-ATTENTION sub-block ─────────────────
    var sa_gg = _gate_residual_bwd(d_x_after_sa, saved.sa.gate[], saved.sa.sub_out[], B, S_IMG, D, ctx)
    var d_x_in = add(d_x_after_sa, _t(_zeros(B * S_IMG * D), [B, S_IMG, D], ctx), ctx)
    var pg_saout = _proj_bwd_with_lora_device_tensors(
        sa_gg.d_sub_out, saved.sa_attn.attn_flat[], w.sa_out[], lora.sa_out,
        B * S_IMG, D, D, True, ctx,
    )
    d_a[SLOT_SA_O] = pg_saout.d_a.copy(); d_b[SLOT_SA_O] = pg_saout.d_b.copy()
    var d_sa_att4 = reshape(pg_saout.d_x, [B, S_IMG, H, Dh], ctx)
    var sa_sb = sdpa_backward[1, S_IMG, H, Dh](
        saved.sa_attn.q_sdpa[], saved.sa_attn.k_sdpa[], saved.sa_attn.v4[],
        d_sa_att4, sa_scale, ctx,
    )
    var d_sa_q_rope_f = reshape(sa_sb.d_q, [B * S_IMG * H, Dh], ctx)
    var d_sa_k_rope_f = reshape(sa_sb.d_k, [B * S_IMG * H, Dh], ctx)
    var d_sa_q_rms_f = rope_backward(d_sa_q_rope_f, cos, sin, False, ctx)
    var d_sa_k_rms_f = rope_backward(d_sa_k_rope_f, cos, sin, False, ctx)
    var sa_qpre_f = reshape(saved.sa_attn.q_pre[], [B * S_IMG * H, Dh], ctx)
    var sa_kpre_f = reshape(saved.sa_attn.k_pre[], [B * S_IMG * H, Dh], ctx)
    var d_sa_q_pre = rms_norm_backward_dx(d_sa_q_rms_f, sa_qpre_f, w.sa_qn[], Float32(1e-6), ctx)
    var d_sa_k_pre = rms_norm_backward_dx(d_sa_k_rms_f, sa_kpre_f, w.sa_kn[], Float32(1e-6), ctx)
    var d_sa_q_proj = reshape(d_sa_q_pre, [B, S_IMG, D], ctx)
    var d_sa_k_proj = reshape(d_sa_k_pre, [B, S_IMG, D], ctx)
    var d_sa_v_proj = reshape(sa_sb.d_v, [B, S_IMG, D], ctx)
    # q/k/v all from sa_xmod (trained) -> sum the 3 (base+LoRA) d_x contributions.
    var pg_saq = _proj_bwd_with_lora_device_tensors(
        d_sa_q_proj, saved.sa_attn.q_ctx_in[], w.sa_q[], lora.sa_q, B * S_IMG, D, D, True, ctx,
    )
    d_a[SLOT_SA_Q] = pg_saq.d_a.copy(); d_b[SLOT_SA_Q] = pg_saq.d_b.copy()
    var pg_sak = _proj_bwd_with_lora_device_tensors(
        d_sa_k_proj, saved.sa_attn.q_ctx_in[], w.sa_k[], lora.sa_k, B * S_IMG, D, D, True, ctx,
    )
    d_a[SLOT_SA_K] = pg_sak.d_a.copy(); d_b[SLOT_SA_K] = pg_sak.d_b.copy()
    var pg_sav = _proj_bwd_with_lora_device_tensors(
        d_sa_v_proj, saved.sa_attn.q_ctx_in[], w.sa_v[], lora.sa_v, B * S_IMG, D, D, True, ctx,
    )
    d_a[SLOT_SA_V] = pg_sav.d_a.copy(); d_b[SLOT_SA_V] = pg_sav.d_b.copy()
    var d_sa_xmod = add(add(pg_saq.d_x, pg_sak.d_x, ctx), pg_sav.d_x, ctx)
    var sa_mb = modulate_backward(d_sa_xmod, saved.sa.ln[], saved.sa.scale[], ctx)
    var sa_dxin = layer_norm_backward_dx(sa_mb.d_x, saved.sa.x_in[], ln_ones, eps, ctx)
    d_x_in = add(d_x_in, sa_dxin, ctx)
    var sa_dadd = concat(1, ctx, sa_mb.d_shift, sa_mb.d_scale, sa_gg.d_gate)
    var d_sa_modh = linear_backward_dx(sa_dadd, w.sa_mod2[], B, 256, 3 * D, ctx)
    var d_sa_tsilu = linear_backward_dx(d_sa_modh, w.sa_mod1[], B, ANIMA_HIDDEN, 256, ctx)
    d_t_silu_acc = add(d_t_silu_acc, d_sa_tsilu, ctx)
    if trace:
        ctx.synchronize()
        var ts_sa = perf_counter_ns()
        print("[ANIMA_BWD] mlp=", Float32(Float64(ts_mlp - ts0) / 1.0e9),
              " ca=", Float32(Float64(ts_ca - ts_mlp) / 1.0e9),
              " sa=", Float32(Float64(ts_sa - ts_ca) / 1.0e9))

    return AnimaBlockLoraTensorBackward(TArc(d_x_in^), TArc(d_t_silu_acc^), d_a^, d_b^)
