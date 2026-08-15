# serenitymojo/models/ltx2/ltx2_av_backward.mojo
#
# LTX-2.3 22B JOINT AUDIO-VIDEO BLOCK — activation-saving TRAIN forward +
# hand-chained BACKWARD for the production T2V LoRA surface.
#
# TENET (train/infer same math): the train forward below is an op-for-op mirror
# of the INFERENCE SPINE `serenitymojo/models/dit/ltx2_dit.mojo
# ltx2_block_forward_av` (the forward proven by frames + the block-0 parity
# smoke, video cos 0.9999943). It reuses the SAME weights struct
# (LTX2AVBlockWeights, factorized LoRA via `_linear_b`/`_linear_lora_delta`)
# and the SAME helper math (_ada_row_pertok / _modulate_bc / _kv_modulate /
# _compute_cross_mod / _rms_norm_opt / apply_ltx2_rope). The only differences:
#   * activations are SAVED (per-block recompute discipline: the acts struct is
#     transient — at stack level each block recomputes this forward right
#     before its backward, Klein-style),
#   * rectangular sdpa_cross_nomask is used for ALL six attentions (the spine's
#     square sdpa_nomask fast path is the same softmax-attention math; the
#     backward partner is ops/attention_backward.sdpa_backward_rect),
#   * no `skip_cross_modal` (that perturbation is inference-only guidance; the
#     trained path is the full block),
#   * the debug-only `v2a_delta` third output is not returned (it is an
#     inference probe; its math — the v2a addend — is inside audio_out).
#
# BACKWARD SCOPE (LoRA training; base FROZEN):
#   outputs d_hidden (video stream input grad), d_ahs (audio stream input
#   grad), and d_A/d_B for every attached factorized LoRA adapter — the
#   production surface is 24 pairs/block: {to_q,to_k,to_v,to_out.0} x
#   {attn1, attn2, audio_attn1, audio_attn2, audio_to_video_attn,
#    video_to_audio_attn} (torchref LTX2_INCLUDE_PATTERNS_T2V).
#   Base weight grads / modulation-vector grads / text-context grads are NOT
#   produced (frozen / upstream-shared / untrained leaves).
#
# Every backward arm is a pre-existing gated op:
#   linear_backward(_dx)  ops/linalg_backward.mojo
#   rms_norm_backward_dx  ops/norm_backward.mojo
#   sdpa_backward_rect    ops/attention_backward.mojo (rectangular, Sq!=Skv ok)
#   rope_backward         ops/rope_struct_backward.mojo (halfsplit tables
#                         [rows, Dh/2] in the same (s,h) row order as
#                         apply_ltx2_rope = rope_halfsplit)
#   gelu_backward / sigmoid_backward  ops/activation_backward.mojo
# Broadcast modulate/gate backwards are plain tensor_algebra mul/add with the
# SAME broadcast shapes the forward uses (no reductions needed because
# scale/shift/gate grads are not in scope).
#
# Parity gate: serenitymojo/models/ltx2/parity/ltx2_av_bwd_parity.mojo vs the
# torch.autograd oracle scripts/ltx2_av_block_bwd_oracle.py (real block-0
# weights from the dequant-bf16 export, non-degenerate seeded inputs, REAL head
# counts 32x128 / 32x64). The gate runs F32 (repo pattern for synthetic-dims
# gates); the production stack stage will run this bf16-carrier (every op used
# here has a native BF16 path).
#
# Mojo 1.0.0b1: `def` not `fn`; Tensor is move-only (Movable result structs);
# struct fields moved with `^`.

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.memory import ArcPointer
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

# forward ops (the spine's ops)
from serenitymojo.ops.linear import linear, linear_slab
from serenitymojo.ops.norm import rms_norm, rms_norm_slab
from serenitymojo.ops.activations import gelu, sigmoid, sigmoid_slab
from serenitymojo.ops.attention import sdpa_cross_nomask, sdpa_cross_nomask_slab
from serenitymojo.ops.tensor_algebra import (
    reshape, add, mul, mul_scalar, add_scalar,
    reshape_slab, add_slab, mul_slab, mul_scalar_slab, add_scalar_slab,
    full_device_slab, zeros_device,
)
from serenitymojo.models.dit.ltx2_rope import apply_ltx2_rope, apply_ltx2_rope_slab
from serenitymojo.models.dit.ltx2_dit import (
    LTX2AVBlockWeights,
    _ada_row_pertok,
    _modulate_bc,
    _kv_modulate,
    _compute_cross_mod,
    _rms_norm_opt,
    _shape3,
    _shape4,
)

# backward arms (all pre-built + gated)
from serenitymojo.ops.linalg_backward import (
    linear_backward, linear_backward_dx,
    linear_backward_slab, linear_backward_dx_slab,
)
from serenitymojo.ops.norm_backward import rms_norm_backward_dx, rms_norm_backward_dx_slab
from serenitymojo.ops.activation_backward import (
    gelu_backward, sigmoid_backward, sigmoid_backward_slab,
)
from serenitymojo.ops.attention_backward import sdpa_backward_rect, sdpa_backward_rect_slab
from serenitymojo.ops.rope_struct_backward import rope_backward, rope_backward_slab
from serenitymojo.ops.ltx2_gate_backward import ltx2_gate_dgates, ltx2_gate_dgates_slab
from serenitymojo.autograd_v2.step_slab import StepSlab


# ── small helpers ────────────────────────────────────────────────────────────
def _sh2(a: Int, b: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    return s^


def _dummy_t(dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    """1-element placeholder for absent rope tables (mirrors the spine)."""
    var d = List[Float32]()
    d.append(Float32(1.0))
    var sh = List[Int]()
    sh.append(1)
    sh.append(1)
    return Tensor.from_host(d, sh^, dtype, ctx)


def _ones_t(d: Int, dtype: STDtype, ctx: DeviceContext) raises -> Tensor:
    var o = List[Float32]()
    for _ in range(d):
        o.append(Float32(1.0))
    var sh = List[Int]()
    sh.append(d)
    return Tensor.from_host(o, sh^, dtype, ctx)


def _one_plus(t: Tensor, ctx: DeviceContext) raises -> Tensor:
    return add_scalar(t, Float32(1.0), ctx)


# rms-norm backward with the spine's OPTIONAL affine (_rms_norm_opt partner).
# Returns d_x only (affine weight is frozen base; d_g discarded).
def _rms_bwd_opt(
    d: Tensor,                  # [.., D] grad wrt rms output
    x: Tensor,                  # [.., D] saved pre-norm input
    weights: LTX2AVBlockWeights,
    w_key: String,
    eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    if weights._has(w_key):
        return rms_norm_backward_dx(d, x, weights._w(w_key), eps, ctx)
    var xs = x.shape()
    var ones = _ones_t(xs[len(xs) - 1], x.dtype(), ctx)
    return rms_norm_backward_dx(d, x, ones, eps, ctx)


# ── StepSlab twins (autograd_v2 contract C8) ─────────────────────────────────
def _one_plus_slab(t: Tensor, ctx: DeviceContext, mut slab: StepSlab) raises -> Tensor:
    return add_scalar_slab(t, Float32(1.0), ctx, slab)


def _rms_bwd_opt_slab(
    d: Tensor,
    x: Tensor,
    weights: LTX2AVBlockWeights,
    w_key: String,
    eps: Float32,
    ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> Tensor:
    """StepSlab variant of `_rms_bwd_opt` (this file :124). The no-affine ones
    weight is filled ON-DEVICE from the slab (full_device_slab — no host upload,
    no off-slab alloc; captured region stays alloc-clean, byte-identical since
    1.0 is exact)."""
    if weights._has(w_key):
        return rms_norm_backward_dx_slab(d, x, weights._w(w_key), eps, ctx, slab)
    var xs = x.shape()
    var od = List[Int]()
    od.append(xs[len(xs) - 1])
    var ones = full_device_slab(od^, Float32(1.0), x.dtype(), ctx, slab)
    return rms_norm_backward_dx_slab(d, x, ones, eps, ctx, slab)


# ── LoRA grads (factorized; matches add_lora_factor: y += scale*B(A x)) ──────
struct LoraPairGrad(Copyable, Movable):
    """d_A [rank,in] / d_B [out,rank] for one adapter, host F32, keyed by the
    canonical weight name (e.g. "attn1.to_q.weight")."""

    var name: String
    var d_a: List[Float32]
    var d_b: List[Float32]

    def __init__(
        out self, var name: String,
        var d_a: List[Float32], var d_b: List[Float32],
    ):
        self.name = name^
        self.d_a = d_a^
        self.d_b = d_b^


# Backward of the factorized delta y += s*B(A x) for the adapter attached to
# `w_key` (if any). Appends d_A/d_B to `grads`; returns the d_x contribution
# [M, in] (None when no adapter is attached).
#   t   = A x            [M, rank]   (recomputed — cheaper than saving)
#   dB  = (s*d_y)^T t    [out, rank]
#   d_t = (s*d_y) B      [M, rank]
#   dA  = d_t^T x        [rank, in]
#   d_x = d_t A          [M, in]
def _lora_pair_bwd(
    weights: LTX2AVBlockWeights,
    w_key: String,
    d_y2d: Tensor,   # [M, out]
    x2d: Tensor,     # [M, in]
    M: Int,
    ctx: DeviceContext,
    mut grads: List[LoraPairGrad],
) raises -> Optional[Tensor]:
    for i in range(len(weights.lora_names)):
        if weights.lora_names[i] == w_key:
            ref a = weights.lora_a[i][]
            ref b = weights.lora_b[i][]
            var rank = a.shape()[0]
            var in_f = a.shape()[1]
            var out_f = b.shape()[0]
            var nb = Optional[Tensor](None)
            var t = linear(x2d, a, nb^, ctx)                          # [M,rank]
            var d_dy = mul_scalar(d_y2d, weights.lora_scales[i], ctx)  # [M,out]
            var lb_b = linear_backward(d_dy, t, b, M, rank, out_f, ctx)
            var d_b = lb_b.d_w.to_host(ctx)                            # [out,rank]
            var lb_a = linear_backward(lb_b.d_x, x2d, a, M, in_f, rank, ctx)
            var d_a = lb_a.d_w.to_host(ctx)                            # [rank,in]
            grads.append(LoraPairGrad(String(w_key), d_a^, d_b^))
            return Optional[Tensor](lb_a.d_x.clone(ctx))               # [M,in]
    return Optional[Tensor](None)


def _lora_pair_bwd_slab(
    weights: LTX2AVBlockWeights,
    w_key: String,
    d_y2d: Tensor,   # [M, out]
    x2d: Tensor,     # [M, in]
    M: Int,
    ctx: DeviceContext,
    mut grads: List[LoraPairGrad],
    mut slab: StepSlab,
) raises -> Optional[Tensor]:
    """StepSlab variant of `_lora_pair_bwd` (this file :165) — byte-identical
    (same recompute t=Ax, same linear_backward folds); device scratch routes
    through the slab. d_A/d_B stay host F32 (C12; the LoRA-grad readback is
    out-of-band and unchanged). The returned d_x contribution is a slab clone."""
    for i in range(len(weights.lora_names)):
        if weights.lora_names[i] == w_key:
            ref a = weights.lora_a[i][]
            ref b = weights.lora_b[i][]
            var rank = a.shape()[0]
            var in_f = a.shape()[1]
            var out_f = b.shape()[0]
            var t = linear_slab(x2d, a, None, ctx, slab)                # [M,rank]
            var d_dy = mul_scalar_slab(d_y2d, weights.lora_scales[i], ctx, slab)
            var lb_b = linear_backward_slab(d_dy, t, b, M, rank, out_f, ctx, slab)
            var d_b = lb_b.d_w.to_host(ctx)                             # [out,rank]
            var lb_a = linear_backward_slab(lb_b.d_x, x2d, a, M, in_f, rank, ctx, slab)
            var d_a = lb_a.d_w.to_host(ctx)                             # [rank,in]
            grads.append(LoraPairGrad(String(w_key), d_a^, d_b^))
            return Optional[Tensor](
                reshape_slab(lb_a.d_x, lb_a.d_x.shape(), ctx, slab))   # [M,in]
    return Optional[Tensor](None)


# ── per-attention saved activations ──────────────────────────────────────────
struct AVAttnActs(Movable):
    var q_src: Tensor      # [1,SQ,qdim]   module Q-input (modulated hidden)
    var kv_src: Tensor     # [1,SKV,kvdim] module KV-input
    var q_pre: Tensor      # [SQ,inner]    q after to_q(+lora), pre QK-rms
    var k_pre: Tensor      # [SKV,inner]
    var v4: Tensor         # [1,SKV,H,DH]
    var q_sd: Tensor       # [1,SQ,H,DH]   sdpa Q (post rope if any)
    var k_sd: Tensor       # [1,SKV,H,DH]
    var att_flat: Tensor   # [1,SQ,inner]  sdpa out, pre per-head gate
    var gl: Tensor         # [1,SQ,H]      gate logits (dummy if no gate)
    var gates: Tensor      # [1,SQ,H]      2*sigmoid(gl)
    var att_g: Tensor      # [1,SQ,inner]  post-gate = to_out input
    var out: Tensor        # [1,SQ,out_dim] module output (forward result)
    var has_gate: Bool

    def __init__(
        out self,
        var q_src: Tensor, var kv_src: Tensor,
        var q_pre: Tensor, var k_pre: Tensor, var v4: Tensor,
        var q_sd: Tensor, var k_sd: Tensor,
        var att_flat: Tensor, var gl: Tensor, var gates: Tensor,
        var att_g: Tensor, var out: Tensor, has_gate: Bool,
    ):
        self.q_src = q_src^
        self.kv_src = kv_src^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.v4 = v4^
        self.q_sd = q_sd^
        self.k_sd = k_sd^
        self.att_flat = att_flat^
        self.gl = gl^
        self.gates = gates^
        self.att_g = att_g^
        self.out = out^
        self.has_gate = has_gate


# ── attention TRAIN forward (mirror of ltx2_dit._av_attention, saving acts) ──
def _av_attention_train[SQ: Int, SKV: Int, H: Int, DH: Int](
    weights: LTX2AVBlockWeights,
    mod_name: String,
    hidden: Tensor,            # [1,SQ,qdim]
    kv: Tensor,                # [1,SKV,kvdim]
    has_q_rope: Bool, q_cos: Tensor, q_sin: Tensor,
    has_k_rope: Bool, k_cos: Tensor, k_sin: Tensor,
    eps: Float32,
    ctx: DeviceContext,
) raises -> AVAttnActs:
    var inner = H * DH
    var scale = Float32(1.0) / sqrt(Float32(DH))

    # projections (base + attached factorized LoRA, exactly the spine's path)
    var q = weights._linear_b(
        hidden, mod_name + ".to_q.weight", mod_name + ".to_q.bias", ctx)
    var k = weights._linear_b(
        kv, mod_name + ".to_k.weight", mod_name + ".to_k.bias", ctx)
    var v = weights._linear_b(
        kv, mod_name + ".to_v.weight", mod_name + ".to_v.bias", ctx)

    var q_pre = reshape(q, _sh2(SQ, inner), ctx)
    var k_pre = reshape(k, _sh2(SKV, inner), ctx)

    var q_rms = rms_norm(q, weights._w(mod_name + ".norm_q.weight"), eps, ctx)
    var k_rms = rms_norm(k, weights._w(mod_name + ".norm_k.weight"), eps, ctx)

    var q4 = reshape(q_rms, _shape4(1, SQ, H, DH), ctx)
    var k4 = reshape(k_rms, _shape4(1, SKV, H, DH), ctx)
    var v4 = reshape(v, _shape4(1, SKV, H, DH), ctx)

    if has_q_rope:
        q4 = apply_ltx2_rope(q4, q_cos, q_sin, ctx)
    if has_k_rope:
        k4 = apply_ltx2_rope(k4, k_cos, k_sin, ctx)
    elif has_q_rope:
        # spine fallback: key_rope.or(query_rope)
        k4 = apply_ltx2_rope(k4, q_cos, q_sin, ctx)

    var attn = sdpa_cross_nomask[1, SQ, SKV, H, DH](q4, k4, v4, scale, ctx)
    var att_flat = reshape(attn, _shape3(1, SQ, inner), ctx)

    var has_gate = weights._has(mod_name + ".to_gate_logits.weight")
    var gl: Tensor
    var gates: Tensor
    var att_g: Tensor
    if has_gate:
        gl = weights._linear_b(
            hidden,
            mod_name + ".to_gate_logits.weight",
            mod_name + ".to_gate_logits.bias",
            ctx,
        )                                                  # [1,SQ,H]
        gates = mul_scalar(sigmoid(gl, ctx), Float32(2.0), ctx)
        var g4 = reshape(gates, _shape4(1, SQ, H, 1), ctx)
        var a4 = reshape(att_flat, _shape4(1, SQ, H, DH), ctx)
        att_g = reshape(mul(a4, g4, ctx), _shape3(1, SQ, inner), ctx)
    else:
        gl = _dummy_t(hidden.dtype(), ctx)
        gates = _dummy_t(hidden.dtype(), ctx)
        att_g = att_flat.clone(ctx)

    var out = weights._linear_b(
        att_g, mod_name + ".to_out.0.weight", mod_name + ".to_out.0.bias", ctx)

    return AVAttnActs(
        hidden.clone(ctx), kv.clone(ctx),
        q_pre^, k_pre^, v4^, q4^, k4^,
        att_flat^, gl^, gates^, att_g^, out^, has_gate,
    )


def _av_attention_train_slab[SQ: Int, SKV: Int, H: Int, DH: Int](
    weights: LTX2AVBlockWeights,
    mod_name: String,
    hidden: Tensor,            # [1,SQ,qdim]
    kv: Tensor,                # [1,SKV,kvdim]
    has_q_rope: Bool, q_cos: Tensor, q_sin: Tensor,
    has_k_rope: Bool, k_cos: Tensor, k_sin: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    mut slab: StepSlab,
) raises -> AVAttnActs:
    """StepSlab variant of `_av_attention_train` (this file :233) — op-for-op
    identical, every op swapped for its *_slab twin (_linear_b -> _linear_b_slab,
    rms_norm -> rms_norm_slab, rope -> apply_ltx2_rope_slab, sdpa_cross_nomask ->
    sdpa_cross_nomask_slab, clones -> reshape_slab). Saved acts are slab-backed;
    the block twin keeps them alive across fwd->bwd within one mark/rewind."""
    var inner = H * DH
    var scale = Float32(1.0) / sqrt(Float32(DH))

    var q = weights._linear_b_slab(
        hidden, mod_name + ".to_q.weight", mod_name + ".to_q.bias", ctx, slab)
    var k = weights._linear_b_slab(
        kv, mod_name + ".to_k.weight", mod_name + ".to_k.bias", ctx, slab)
    var v = weights._linear_b_slab(
        kv, mod_name + ".to_v.weight", mod_name + ".to_v.bias", ctx, slab)

    var q_pre = reshape_slab(q, _sh2(SQ, inner), ctx, slab)
    var k_pre = reshape_slab(k, _sh2(SKV, inner), ctx, slab)

    var q_rms = rms_norm_slab(q, weights._w(mod_name + ".norm_q.weight"), eps, ctx, slab)
    var k_rms = rms_norm_slab(k, weights._w(mod_name + ".norm_k.weight"), eps, ctx, slab)

    var q4 = reshape_slab(q_rms, _shape4(1, SQ, H, DH), ctx, slab)
    var k4 = reshape_slab(k_rms, _shape4(1, SKV, H, DH), ctx, slab)
    var v4 = reshape_slab(v, _shape4(1, SKV, H, DH), ctx, slab)

    if has_q_rope:
        q4 = apply_ltx2_rope_slab(q4, q_cos, q_sin, ctx, slab)
    if has_k_rope:
        k4 = apply_ltx2_rope_slab(k4, k_cos, k_sin, ctx, slab)
    elif has_q_rope:
        k4 = apply_ltx2_rope_slab(k4, q_cos, q_sin, ctx, slab)

    var attn = sdpa_cross_nomask_slab[1, SQ, SKV, H, DH](q4, k4, v4, scale, ctx, slab)
    var att_flat = reshape_slab(attn, _shape3(1, SQ, inner), ctx, slab)

    var has_gate = weights._has(mod_name + ".to_gate_logits.weight")
    var gl: Tensor
    var gates: Tensor
    var att_g: Tensor
    if has_gate:
        gl = weights._linear_b_slab(
            hidden,
            mod_name + ".to_gate_logits.weight",
            mod_name + ".to_gate_logits.bias",
            ctx, slab,
        )                                                  # [1,SQ,H]
        gates = mul_scalar_slab(sigmoid_slab(gl, ctx, slab), Float32(2.0), ctx, slab)
        var g4 = reshape_slab(gates, _shape4(1, SQ, H, 1), ctx, slab)
        var a4 = reshape_slab(att_flat, _shape4(1, SQ, H, DH), ctx, slab)
        att_g = reshape_slab(mul_slab(a4, g4, ctx, slab), _shape3(1, SQ, inner), ctx, slab)
    else:
        gl = _dummy_t(hidden.dtype(), ctx)
        gates = _dummy_t(hidden.dtype(), ctx)
        att_g = reshape_slab(att_flat, att_flat.shape(), ctx, slab)

    var out = weights._linear_b_slab(
        att_g, mod_name + ".to_out.0.weight", mod_name + ".to_out.0.bias", ctx, slab)

    return AVAttnActs(
        reshape_slab(hidden, hidden.shape(), ctx, slab),
        reshape_slab(kv, kv.shape(), ctx, slab),
        q_pre^, k_pre^, v4^, q4^, k4^,
        att_flat^, gl^, gates^, att_g^, out^, has_gate,
    )


# ── attention BACKWARD (reverse of _av_attention_train) ─────────────────────
struct AVAttnGrads(Movable):
    var d_q_src: Tensor    # [1,SQ,qdim]
    var d_kv_src: Tensor   # [1,SKV,kvdim]

    def __init__(out self, var d_q_src: Tensor, var d_kv_src: Tensor):
        self.d_q_src = d_q_src^
        self.d_kv_src = d_kv_src^


def _av_attention_bwd[SQ: Int, SKV: Int, H: Int, DH: Int](
    weights: LTX2AVBlockWeights,
    mod_name: String,
    acts: AVAttnActs,
    d_out: Tensor,             # [1,SQ,out_dim] grad wrt the module output
    has_q_rope: Bool, q_cos: Tensor, q_sin: Tensor,
    has_k_rope: Bool, k_cos: Tensor, k_sin: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    mut lora_grads: List[LoraPairGrad],
) raises -> AVAttnGrads:
    var inner = H * DH
    var scale = Float32(1.0) / sqrt(Float32(DH))
    var q_dim = acts.q_src.shape()[2]
    var kv_dim = acts.kv_src.shape()[2]
    var out_dim = d_out.shape()[len(d_out.shape()) - 1]

    var q_src2 = reshape(acts.q_src, _sh2(SQ, q_dim), ctx)
    var kv_src2 = reshape(acts.kv_src, _sh2(SKV, kv_dim), ctx)

    # ── to_out: y = W_o att_g + b (+ s*B_o(A_o att_g)) ──
    var d_out2 = reshape(d_out, _sh2(SQ, out_dim), ctx)
    var d_att_g = linear_backward_dx(
        d_out2, weights._w(mod_name + ".to_out.0.weight"),
        SQ, inner, out_dim, ctx)
    var att_g2 = reshape(acts.att_g, _sh2(SQ, inner), ctx)
    var lo = _lora_pair_bwd(
        weights, mod_name + ".to_out.0.weight", d_out2, att_g2, SQ, ctx,
        lora_grads)
    if lo:
        d_att_g = add(d_att_g, lo.value(), ctx)

    # ── per-head gate: att_g = att_flat * gates (head-broadcast) ──
    #   d_att_flat[s,h*DH+d] = d_att_g[s,h*DH+d] * gates[s,h]
    #   d_gates[s,h]         = sum_d d_att_g[s,h*DH+d] * att_flat[s,h*DH+d]
    #   gates = 2*sigmoid(gl); gl = linear(q_src, gate_w, gate_b)
    var d_att_flat: Tensor
    var d_gate_path = Optional[Tensor](None)
    if acts.has_gate:
        var d4 = reshape(d_att_g, _shape4(1, SQ, H, DH), ctx)
        var g4 = reshape(acts.gates, _shape4(1, SQ, H, 1), ctx)
        d_att_flat = reshape(mul(d4, g4, ctx), _sh2(SQ, inner), ctx)

        # DEVICE gate-grad (ops/ltx2_gate_backward) — BIT-EXACT to the old host
        # round-trip (parity gate ops/tests/ltx2_gate_backward_parity: bf16 AND
        # F32 n_mismatch=0), so wired UNCONDITIONALLY (no anchor shift). Kills the
        # d_att_g.to_host + att_flat.to_host + from_host = the L5 capture blocker.
        var d_gates = ltx2_gate_dgates(d_att_g, acts.att_flat, SQ, H, DH, ctx)
        var gl2 = reshape(acts.gl, _sh2(SQ, H), ctx)
        var d_gl = sigmoid_backward(
            mul_scalar(d_gates, Float32(2.0), ctx), gl2, ctx)
        d_gate_path = Optional[Tensor](linear_backward_dx(
            d_gl, weights._w(mod_name + ".to_gate_logits.weight"),
            SQ, q_dim, H, ctx))
    else:
        d_att_flat = reshape(d_att_g, _sh2(SQ, inner), ctx)

    # ── sdpa (rectangular, recompute-softmax backward) ──
    var d_att4 = reshape(d_att_flat, _shape4(1, SQ, H, DH), ctx)
    var sg = sdpa_backward_rect[1, SQ, SKV, H, DH](
        acts.q_sd, acts.k_sd, acts.v4, d_att4, scale, ctx)

    # ── rope backward (halfsplit; tables in the same (s,h) row order) ──
    var d_q4: Tensor
    if has_q_rope:
        d_q4 = rope_backward(sg.d_q, q_cos, q_sin, False, ctx)
    else:
        d_q4 = sg.d_q.clone(ctx)
    var d_k4: Tensor
    if has_k_rope:
        d_k4 = rope_backward(sg.d_k, k_cos, k_sin, False, ctx)
    elif has_q_rope:
        d_k4 = rope_backward(sg.d_k, q_cos, q_sin, False, ctx)
    else:
        d_k4 = sg.d_k.clone(ctx)

    # ── QK-RMSNorm over full inner dim (weights frozen; d_g discarded) ──
    var d_q_flat = reshape(d_q4, _sh2(SQ, inner), ctx)
    var d_k_flat = reshape(d_k4, _sh2(SKV, inner), ctx)
    var d_q_rms = rms_norm_backward_dx(
        d_q_flat, acts.q_pre, weights._w(mod_name + ".norm_q.weight"), eps, ctx)
    var d_k_rms = rms_norm_backward_dx(
        d_k_flat, acts.k_pre, weights._w(mod_name + ".norm_k.weight"), eps, ctx)
    var d_v_flat = reshape(sg.d_v, _sh2(SKV, inner), ctx)

    # ── projections (base dx only — frozen; + LoRA d_A/d_B/d_x) ──
    var d_q_src = linear_backward_dx(
        d_q_rms, weights._w(mod_name + ".to_q.weight"), SQ, q_dim, inner, ctx)
    var lq = _lora_pair_bwd(
        weights, mod_name + ".to_q.weight", d_q_rms, q_src2, SQ, ctx,
        lora_grads)
    if lq:
        d_q_src = add(d_q_src, lq.value(), ctx)
    if d_gate_path:
        d_q_src = add(d_q_src, d_gate_path.value(), ctx)

    var d_kv_src = linear_backward_dx(
        d_k_rms, weights._w(mod_name + ".to_k.weight"), SKV, kv_dim, inner, ctx)
    var lk = _lora_pair_bwd(
        weights, mod_name + ".to_k.weight", d_k_rms, kv_src2, SKV, ctx,
        lora_grads)
    if lk:
        d_kv_src = add(d_kv_src, lk.value(), ctx)
    d_kv_src = add(d_kv_src, linear_backward_dx(
        d_v_flat, weights._w(mod_name + ".to_v.weight"), SKV, kv_dim, inner,
        ctx), ctx)
    var lv = _lora_pair_bwd(
        weights, mod_name + ".to_v.weight", d_v_flat, kv_src2, SKV, ctx,
        lora_grads)
    if lv:
        d_kv_src = add(d_kv_src, lv.value(), ctx)

    return AVAttnGrads(
        reshape(d_q_src, _shape3(1, SQ, q_dim), ctx),
        reshape(d_kv_src, _shape3(1, SKV, kv_dim), ctx),
    )


def _av_attention_bwd_slab[SQ: Int, SKV: Int, H: Int, DH: Int](
    weights: LTX2AVBlockWeights,
    mod_name: String,
    acts: AVAttnActs,
    d_out: Tensor,             # [1,SQ,out_dim]
    has_q_rope: Bool, q_cos: Tensor, q_sin: Tensor,
    has_k_rope: Bool, k_cos: Tensor, k_sin: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    mut lora_grads: List[LoraPairGrad],
    mut slab: StepSlab,
) raises -> AVAttnGrads:
    """StepSlab variant of `_av_attention_bwd` (this file :315) — op-for-op
    identical, every op swapped for its *_slab twin (linear_backward_dx ->
    linear_backward_dx_slab, _lora_pair_bwd -> _lora_pair_bwd_slab, sdpa/rope/
    rms/gate backward -> their _slab twins, clones -> reshape_slab). d_A/d_B
    stay host F32 (C12)."""
    var inner = H * DH
    var scale = Float32(1.0) / sqrt(Float32(DH))
    var q_dim = acts.q_src.shape()[2]
    var kv_dim = acts.kv_src.shape()[2]
    var out_dim = d_out.shape()[len(d_out.shape()) - 1]

    var q_src2 = reshape_slab(acts.q_src, _sh2(SQ, q_dim), ctx, slab)
    var kv_src2 = reshape_slab(acts.kv_src, _sh2(SKV, kv_dim), ctx, slab)

    # ── to_out ──
    var d_out2 = reshape_slab(d_out, _sh2(SQ, out_dim), ctx, slab)
    var d_att_g = linear_backward_dx_slab(
        d_out2, weights._w(mod_name + ".to_out.0.weight"),
        SQ, inner, out_dim, ctx, slab)
    var att_g2 = reshape_slab(acts.att_g, _sh2(SQ, inner), ctx, slab)
    var lo = _lora_pair_bwd_slab(
        weights, mod_name + ".to_out.0.weight", d_out2, att_g2, SQ, ctx,
        lora_grads, slab)
    if lo:
        d_att_g = add_slab(d_att_g, lo.value(), ctx, slab)

    # ── per-head gate ──
    var d_att_flat: Tensor
    var d_gate_path = Optional[Tensor](None)
    if acts.has_gate:
        var d4 = reshape_slab(d_att_g, _shape4(1, SQ, H, DH), ctx, slab)
        var g4 = reshape_slab(acts.gates, _shape4(1, SQ, H, 1), ctx, slab)
        d_att_flat = reshape_slab(mul_slab(d4, g4, ctx, slab), _sh2(SQ, inner), ctx, slab)

        var d_gates = ltx2_gate_dgates_slab(d_att_g, acts.att_flat, SQ, H, DH, ctx, slab)
        var gl2 = reshape_slab(acts.gl, _sh2(SQ, H), ctx, slab)
        var d_gl = sigmoid_backward_slab(
            mul_scalar_slab(d_gates, Float32(2.0), ctx, slab), gl2, ctx, slab)
        d_gate_path = Optional[Tensor](linear_backward_dx_slab(
            d_gl, weights._w(mod_name + ".to_gate_logits.weight"),
            SQ, q_dim, H, ctx, slab))
    else:
        d_att_flat = reshape_slab(d_att_g, _sh2(SQ, inner), ctx, slab)

    # ── sdpa (rectangular, recompute-softmax backward) ──
    var d_att4 = reshape_slab(d_att_flat, _shape4(1, SQ, H, DH), ctx, slab)
    var sg = sdpa_backward_rect_slab[1, SQ, SKV, H, DH](
        acts.q_sd, acts.k_sd, acts.v4, d_att4, scale, ctx, slab)

    # ── rope backward ──
    var d_q4: Tensor
    if has_q_rope:
        d_q4 = rope_backward_slab(sg.d_q, q_cos, q_sin, False, ctx, slab)
    else:
        d_q4 = reshape_slab(sg.d_q, sg.d_q.shape(), ctx, slab)
    var d_k4: Tensor
    if has_k_rope:
        d_k4 = rope_backward_slab(sg.d_k, k_cos, k_sin, False, ctx, slab)
    elif has_q_rope:
        d_k4 = rope_backward_slab(sg.d_k, q_cos, q_sin, False, ctx, slab)
    else:
        d_k4 = reshape_slab(sg.d_k, sg.d_k.shape(), ctx, slab)

    # ── QK-RMSNorm (weights frozen; d_g discarded) ──
    var d_q_flat = reshape_slab(d_q4, _sh2(SQ, inner), ctx, slab)
    var d_k_flat = reshape_slab(d_k4, _sh2(SKV, inner), ctx, slab)
    var d_q_rms = rms_norm_backward_dx_slab(
        d_q_flat, acts.q_pre, weights._w(mod_name + ".norm_q.weight"), eps, ctx, slab)
    var d_k_rms = rms_norm_backward_dx_slab(
        d_k_flat, acts.k_pre, weights._w(mod_name + ".norm_k.weight"), eps, ctx, slab)
    var d_v_flat = reshape_slab(sg.d_v, _sh2(SKV, inner), ctx, slab)

    # ── projections (base dx only — frozen; + LoRA d_A/d_B/d_x) ──
    var d_q_src = linear_backward_dx_slab(
        d_q_rms, weights._w(mod_name + ".to_q.weight"), SQ, q_dim, inner, ctx, slab)
    var lq = _lora_pair_bwd_slab(
        weights, mod_name + ".to_q.weight", d_q_rms, q_src2, SQ, ctx,
        lora_grads, slab)
    if lq:
        d_q_src = add_slab(d_q_src, lq.value(), ctx, slab)
    if d_gate_path:
        d_q_src = add_slab(d_q_src, d_gate_path.value(), ctx, slab)

    var d_kv_src = linear_backward_dx_slab(
        d_k_rms, weights._w(mod_name + ".to_k.weight"), SKV, kv_dim, inner, ctx, slab)
    var lk = _lora_pair_bwd_slab(
        weights, mod_name + ".to_k.weight", d_k_rms, kv_src2, SKV, ctx,
        lora_grads, slab)
    if lk:
        d_kv_src = add_slab(d_kv_src, lk.value(), ctx, slab)
    d_kv_src = add_slab(d_kv_src, linear_backward_dx_slab(
        d_v_flat, weights._w(mod_name + ".to_v.weight"), SKV, kv_dim, inner,
        ctx, slab), ctx, slab)
    var lv = _lora_pair_bwd_slab(
        weights, mod_name + ".to_v.weight", d_v_flat, kv_src2, SKV, ctx,
        lora_grads, slab)
    if lv:
        d_kv_src = add_slab(d_kv_src, lv.value(), ctx, slab)

    return AVAttnGrads(
        reshape_slab(d_q_src, _shape3(1, SQ, q_dim), ctx, slab),
        reshape_slab(d_kv_src, _shape3(1, SKV, kv_dim), ctx, slab),
    )


# ═════════════════════════════════════════════════════════════════════════════
# RUNG 1 — device-resident LoRA grads. The device analog of the LoraPairGrad /
# _lora_pair_bwd_slab / _av_attention_bwd_slab host path above: byte-identical
# math + fold order, but d_A/d_B are copied OUT of the slab to resident device
# buffers (.clone = D2D, NO sync, NO DtoH) instead of the per-adapter .to_host,
# so the measured 768 DtoH+sync/step leave the per-block region. The SINGLE
# boundary readback (ltx2_lora_dev_readback) lands host F32 exactly as today
# (C12 — the optimizer input is unchanged). The host path stays reachable (C13).
# ═════════════════════════════════════════════════════════════════════════════
struct LoraPairGradDev(Copyable, Movable):
    """Device-resident F32 d_A [rank,in] / d_B [out,rank] for one adapter, keyed
    by the canonical weight name — the device analog of LoraPairGrad. Held as
    ArcPointer[Tensor] so it can live in a List (Tensor is move-only)."""

    var name: String
    var d_a: ArcPointer[Tensor]   # [rank, in]  F32 device
    var d_b: ArcPointer[Tensor]   # [out, rank] F32 device

    def __init__(
        out self, var name: String,
        var d_a: ArcPointer[Tensor], var d_b: ArcPointer[Tensor],
    ):
        self.name = name^
        self.d_a = d_a^
        self.d_b = d_b^


struct Ltx2LoraGradStore(Movable):
    """Stack-loop-owned RESIDENT device grad store (rung 3 capture prereq): per
    (block, slot) PRE-ALLOCATED F32 d_A [rank,in] / d_B [out,rank] device buffers,
    sized from the attached LoRA adapters. _lora_pair_bwd_slab_dev writes d_w in
    via enqueue_copy (NO alloc — capture-safe), RETIRING the rung-1 TEMPORARY
    .clone. Read to host F32 in ONE boundary pass (C12). accum=1 = overwrite =
    bit-identical to the rung-2 path; grad-accum add-into (accum>1) is a
    documented follow-up (needs an in-place F32 add + window-start zero)."""

    var d_a: List[ArcPointer[Tensor]]    # [num_layers*n_slots] F32 device, pre-alloc
    var d_b: List[ArcPointer[Tensor]]
    var names: List[String]              # slot s -> canonical weight name
    var num_layers: Int
    var n_slots: Int

    def __init__(
        out self, var d_a: List[ArcPointer[Tensor]], var d_b: List[ArcPointer[Tensor]],
        var names: List[String], num_layers: Int, n_slots: Int,
    ):
        self.d_a = d_a^
        self.d_b = d_b^
        self.names = names^
        self.num_layers = num_layers
        self.n_slots = n_slots

    def __init__(out self):
        """Empty store — the default for the .clone path (block_idx<0); never
        accessed, so it holds no buffers (C13: rung-1/2 .clone stays reachable)."""
        self.d_a = List[ArcPointer[Tensor]]()
        self.d_b = List[ArcPointer[Tensor]]()
        self.names = List[String]()
        self.num_layers = 0
        self.n_slots = 0

    def active(self) -> Bool:
        return self.n_slots > 0

    @staticmethod
    def create(
        num_layers: Int, names: List[String],
        la: List[ArcPointer[Tensor]], lb: List[ArcPointer[Tensor]],
        ctx: DeviceContext,
    ) raises -> Ltx2LoraGradStore:
        """Pre-allocate zeroed grad buffers matching each attached adapter's shape
        AND dtype (la[i]=[rank,in], lb[i]=[out,rank]). linear_backward_slab casts
        d_w to the weight's dtype (bf16 for bf16 adapters), so the store slots are
        the ADAPTER dtype — enqueue_copy byte-matches the grad, and readback's
        .to_host upcasts bf16→F32 exactly as the hand-chain does. la/lb are the
        flat [num_layers*n_slots] adapter lists the stack loop already holds."""
        var n_slots = len(names)
        var d_a = List[ArcPointer[Tensor]]()
        var d_b = List[ArcPointer[Tensor]]()
        for i in range(num_layers * n_slots):
            d_a.append(ArcPointer[Tensor](zeros_device(la[i][].shape(), la[i][].dtype(), ctx)))
            d_b.append(ArcPointer[Tensor](zeros_device(lb[i][].shape(), lb[i][].dtype(), ctx)))
        return Ltx2LoraGradStore(d_a^, d_b^, names.copy(), num_layers, n_slots)

    def slot_of(self, name: String) -> Int:
        for s in range(len(self.names)):
            if self.names[s] == name:
                return s
        return -1

    def a_arc(self, block: Int, slot: Int) -> ArcPointer[Tensor]:
        return self.d_a[block * self.n_slots + slot].copy()

    def b_arc(self, block: Int, slot: Int) -> ArcPointer[Tensor]:
        return self.d_b[block * self.n_slots + slot].copy()

    def write(
        self, block: Int, slot: Int, d_a_src: Tensor, d_b_src: Tensor,
        ctx: DeviceContext,
    ) raises:
        """Overwrite (accum=1) the (block,slot) F32 grad slots via D2D enqueue_copy
        — NO alloc, capture-safe. d_a_src/d_b_src are the F32 slab d_w (same
        shape/dtype as the slot). Rung 3 is accum=1 ONLY; the stack loop fails
        loud if the store path is active with accum>1 (device add-into is a
        separate gated rung — see zero()/the P4 note)."""
        var idx = block * self.n_slots + slot
        ctx.enqueue_copy(dst_buf=self.d_a[idx][].buf, src_buf=d_a_src.buf)
        ctx.enqueue_copy(dst_buf=self.d_b[idx][].buf, src_buf=d_b_src.buf)

    def zero(self, ctx: DeviceContext) raises:
        """DEAD hook for grad-accum add-into (accum>1): zero the whole store at a
        window start, then write() would become +=. NOT wired live in rung 3
        (accum=1 overwrite only); present so the device-accum rung can build on
        it without reshaping the store."""
        for i in range(len(self.d_a)):
            self.d_a[i][].buf.enqueue_fill(UInt8(0))
            self.d_b[i][].buf.enqueue_fill(UInt8(0))

    def readback_block(self, block: Int, ctx: DeviceContext) raises -> List[LoraPairGrad]:
        """Host F32 grads for one block, in slot order (boundary readback, C12)."""
        var out = List[LoraPairGrad]()
        var base = block * self.n_slots
        for s in range(self.n_slots):
            out.append(LoraPairGrad(String(self.names[s]),
                self.d_a[base + s][].to_host(ctx), self.d_b[base + s][].to_host(ctx)))
        return out^


def ltx2_lora_dev_readback(
    dev: List[LoraPairGradDev], ctx: DeviceContext
) raises -> List[LoraPairGrad]:
    """The SINGLE boundary readback (outside the per-block region): device F32
    d_A/d_B → host F32 LoraPairGrad, byte-preserving. This is where the DtoH now
    happen — batched at the boundary instead of scattered through the block
    backward. C12: the host-F32 optimizer input is identical."""
    var out = List[LoraPairGrad]()
    for ref g in dev:
        out.append(LoraPairGrad(String(g.name), g.d_a[].to_host(ctx), g.d_b[].to_host(ctx)))
    return out^


def _lora_pair_bwd_slab_dev(
    weights: LTX2AVBlockWeights,
    w_key: String,
    d_y2d: Tensor,   # [M, out]
    x2d: Tensor,     # [M, in]
    M: Int,
    ctx: DeviceContext,
    mut grads: List[LoraPairGradDev],
    mut slab: StepSlab,
    store: Ltx2LoraGradStore = Ltx2LoraGradStore(),
    block_idx: Int = -1,
) raises -> Optional[Tensor]:
    """Device-grad variant of `_lora_pair_bwd_slab` (this file) — identical math
    + fold order (same t=Ax recompute, same two linear_backward folds).

    TWO device-grad sinks, selected by `block_idx >= 0 and store.active()`:
      • RESIDENT STORE (rung 3, capture-safe): d_A/d_B enqueue_copy'd into the
        pre-allocated (block,slot) store buffers — NO alloc. The returned list
        points INTO the store.
      • .clone fallback (rung 1/2, C13-reachable, default): d_A/d_B .clone'd OUT
        of the slab (off-slab alloc — fine for the non-capture path).
    Both are bit-identical (same lb.d_w bytes)."""
    for i in range(len(weights.lora_names)):
        if weights.lora_names[i] == w_key:
            ref a = weights.lora_a[i][]
            ref b = weights.lora_b[i][]
            var rank = a.shape()[0]
            var in_f = a.shape()[1]
            var out_f = b.shape()[0]
            var t = linear_slab(x2d, a, None, ctx, slab)                # [M,rank]
            var d_dy = mul_scalar_slab(d_y2d, weights.lora_scales[i], ctx, slab)
            var lb_b = linear_backward_slab(d_dy, t, b, M, rank, out_f, ctx, slab)
            var lb_a = linear_backward_slab(lb_b.d_x, x2d, a, M, in_f, rank, ctx, slab)
            if block_idx >= 0 and store.active():
                var slot = store.slot_of(w_key)
                if slot < 0:
                    raise Error(
                        String("_lora_pair_bwd_slab_dev: store has no slot for ") + w_key)
                store.write(block_idx, slot, lb_a.d_w, lb_b.d_w, ctx)   # enqueue_copy, NO alloc
                grads.append(LoraPairGradDev(
                    String(w_key), store.a_arc(block_idx, slot), store.b_arc(block_idx, slot)))
            else:
                var d_b_dev = lb_b.d_w.clone(ctx)                       # [out,rank] F32 (off-slab, C13)
                var d_a_dev = lb_a.d_w.clone(ctx)                       # [rank,in] F32
                grads.append(LoraPairGradDev(
                    String(w_key), ArcPointer[Tensor](d_a_dev^), ArcPointer[Tensor](d_b_dev^)))
            return Optional[Tensor](
                reshape_slab(lb_a.d_x, lb_a.d_x.shape(), ctx, slab))   # [M,in]
    return Optional[Tensor](None)


def _av_attention_bwd_slab_dev[SQ: Int, SKV: Int, H: Int, DH: Int](
    weights: LTX2AVBlockWeights,
    mod_name: String,
    acts: AVAttnActs,
    d_out: Tensor,             # [1,SQ,out_dim]
    has_q_rope: Bool, q_cos: Tensor, q_sin: Tensor,
    has_k_rope: Bool, k_cos: Tensor, k_sin: Tensor,
    eps: Float32,
    ctx: DeviceContext,
    mut lora_grads: List[LoraPairGradDev],
    mut slab: StepSlab,
    store: Ltx2LoraGradStore = Ltx2LoraGradStore(),
    block_idx: Int = -1,
) raises -> AVAttnGrads:
    """Device-grad variant of `_av_attention_bwd_slab` (this file) — byte-for-byte
    identical except the 4 LoRA pair backwards go through _lora_pair_bwd_slab_dev
    (device d_A/d_B) and collect List[LoraPairGradDev]. All d_x math is the same
    slab twins, so d_q_src/d_kv_src are bit-identical to the host variant.
    (store, block_idx) route the grads to the resident store when active; else
    .clone (C13)."""
    var inner = H * DH
    var scale = Float32(1.0) / sqrt(Float32(DH))
    var q_dim = acts.q_src.shape()[2]
    var kv_dim = acts.kv_src.shape()[2]
    var out_dim = d_out.shape()[len(d_out.shape()) - 1]

    var q_src2 = reshape_slab(acts.q_src, _sh2(SQ, q_dim), ctx, slab)
    var kv_src2 = reshape_slab(acts.kv_src, _sh2(SKV, kv_dim), ctx, slab)

    var d_out2 = reshape_slab(d_out, _sh2(SQ, out_dim), ctx, slab)
    var d_att_g = linear_backward_dx_slab(
        d_out2, weights._w(mod_name + ".to_out.0.weight"),
        SQ, inner, out_dim, ctx, slab)
    var att_g2 = reshape_slab(acts.att_g, _sh2(SQ, inner), ctx, slab)
    var lo = _lora_pair_bwd_slab_dev(
        weights, mod_name + ".to_out.0.weight", d_out2, att_g2, SQ, ctx,
        lora_grads, slab, store, block_idx)
    if lo:
        d_att_g = add_slab(d_att_g, lo.value(), ctx, slab)

    var d_att_flat: Tensor
    var d_gate_path = Optional[Tensor](None)
    if acts.has_gate:
        var d4 = reshape_slab(d_att_g, _shape4(1, SQ, H, DH), ctx, slab)
        var g4 = reshape_slab(acts.gates, _shape4(1, SQ, H, 1), ctx, slab)
        d_att_flat = reshape_slab(mul_slab(d4, g4, ctx, slab), _sh2(SQ, inner), ctx, slab)

        var d_gates = ltx2_gate_dgates_slab(d_att_g, acts.att_flat, SQ, H, DH, ctx, slab)
        var gl2 = reshape_slab(acts.gl, _sh2(SQ, H), ctx, slab)
        var d_gl = sigmoid_backward_slab(
            mul_scalar_slab(d_gates, Float32(2.0), ctx, slab), gl2, ctx, slab)
        d_gate_path = Optional[Tensor](linear_backward_dx_slab(
            d_gl, weights._w(mod_name + ".to_gate_logits.weight"),
            SQ, q_dim, H, ctx, slab))
    else:
        d_att_flat = reshape_slab(d_att_g, _sh2(SQ, inner), ctx, slab)

    var d_att4 = reshape_slab(d_att_flat, _shape4(1, SQ, H, DH), ctx, slab)
    var sg = sdpa_backward_rect_slab[1, SQ, SKV, H, DH](
        acts.q_sd, acts.k_sd, acts.v4, d_att4, scale, ctx, slab)

    var d_q4: Tensor
    if has_q_rope:
        d_q4 = rope_backward_slab(sg.d_q, q_cos, q_sin, False, ctx, slab)
    else:
        d_q4 = reshape_slab(sg.d_q, sg.d_q.shape(), ctx, slab)
    var d_k4: Tensor
    if has_k_rope:
        d_k4 = rope_backward_slab(sg.d_k, k_cos, k_sin, False, ctx, slab)
    elif has_q_rope:
        d_k4 = rope_backward_slab(sg.d_k, q_cos, q_sin, False, ctx, slab)
    else:
        d_k4 = reshape_slab(sg.d_k, sg.d_k.shape(), ctx, slab)

    var d_q_flat = reshape_slab(d_q4, _sh2(SQ, inner), ctx, slab)
    var d_k_flat = reshape_slab(d_k4, _sh2(SKV, inner), ctx, slab)
    var d_q_rms = rms_norm_backward_dx_slab(
        d_q_flat, acts.q_pre, weights._w(mod_name + ".norm_q.weight"), eps, ctx, slab)
    var d_k_rms = rms_norm_backward_dx_slab(
        d_k_flat, acts.k_pre, weights._w(mod_name + ".norm_k.weight"), eps, ctx, slab)
    var d_v_flat = reshape_slab(sg.d_v, _sh2(SKV, inner), ctx, slab)

    var d_q_src = linear_backward_dx_slab(
        d_q_rms, weights._w(mod_name + ".to_q.weight"), SQ, q_dim, inner, ctx, slab)
    var lq = _lora_pair_bwd_slab_dev(
        weights, mod_name + ".to_q.weight", d_q_rms, q_src2, SQ, ctx,
        lora_grads, slab, store, block_idx)
    if lq:
        d_q_src = add_slab(d_q_src, lq.value(), ctx, slab)
    if d_gate_path:
        d_q_src = add_slab(d_q_src, d_gate_path.value(), ctx, slab)

    var d_kv_src = linear_backward_dx_slab(
        d_k_rms, weights._w(mod_name + ".to_k.weight"), SKV, kv_dim, inner, ctx, slab)
    var lk = _lora_pair_bwd_slab_dev(
        weights, mod_name + ".to_k.weight", d_k_rms, kv_src2, SKV, ctx,
        lora_grads, slab, store, block_idx)
    if lk:
        d_kv_src = add_slab(d_kv_src, lk.value(), ctx, slab)
    d_kv_src = add_slab(d_kv_src, linear_backward_dx_slab(
        d_v_flat, weights._w(mod_name + ".to_v.weight"), SKV, kv_dim, inner,
        ctx, slab), ctx, slab)
    var lv = _lora_pair_bwd_slab_dev(
        weights, mod_name + ".to_v.weight", d_v_flat, kv_src2, SKV, ctx,
        lora_grads, slab, store, block_idx)
    if lv:
        d_kv_src = add_slab(d_kv_src, lv.value(), ctx, slab)

    return AVAttnGrads(
        reshape_slab(d_q_src, _shape3(1, SQ, q_dim), ctx, slab),
        reshape_slab(d_kv_src, _shape3(1, SKV, kv_dim), ctx, slab),
    )


# ── block-level saved activations ────────────────────────────────────────────
struct LTX2AVBlockActs(Movable):
    var hidden: Tensor     # [1,S_V,4096] block video input
    var ahs: Tensor        # [1,S_A,2048] block audio input
    var at1: AVAttnActs    # attn1 (video self)
    var aat1: AVAttnActs   # audio_attn1
    var hs1: Tensor        # video post self-attn
    var ahss1: Tensor      # audio post self-attn
    var at2: AVAttnActs    # attn2 (video<-text)
    var aat2: AVAttnActs   # audio_attn2
    var hs2: Tensor        # video post text cross-attn
    var ahss2: Tensor      # audio post text cross-attn
    var a2v: AVAttnActs    # audio_to_video_attn
    var v2a: AVAttnActs    # video_to_audio_attn
    var hs3: Tensor        # video post a2v residual
    var ahss3: Tensor      # audio post v2a residual
    var h1_v: Tensor       # [1,S_V,16384] video FFN pre-gelu
    var h1_a: Tensor       # [1,S_A,8192]  audio FFN pre-gelu

    def __init__(
        out self,
        var hidden: Tensor, var ahs: Tensor,
        var at1: AVAttnActs, var aat1: AVAttnActs,
        var hs1: Tensor, var ahss1: Tensor,
        var at2: AVAttnActs, var aat2: AVAttnActs,
        var hs2: Tensor, var ahss2: Tensor,
        var a2v: AVAttnActs, var v2a: AVAttnActs,
        var hs3: Tensor, var ahss3: Tensor,
        var h1_v: Tensor, var h1_a: Tensor,
    ):
        self.hidden = hidden^
        self.ahs = ahs^
        self.at1 = at1^
        self.aat1 = aat1^
        self.hs1 = hs1^
        self.ahss1 = ahss1^
        self.at2 = at2^
        self.aat2 = aat2^
        self.hs2 = hs2^
        self.ahss2 = ahss2^
        self.a2v = a2v^
        self.v2a = v2a^
        self.hs3 = hs3^
        self.ahss3 = ahss3^
        self.h1_v = h1_v^
        self.h1_a = h1_a^


struct LTX2AVTrainForward(Movable):
    var video_out: Tensor  # [1,S_V,4096]
    var audio_out: Tensor  # [1,S_A,2048]
    var acts: LTX2AVBlockActs

    def __init__(
        out self, var video_out: Tensor, var audio_out: Tensor,
        var acts: LTX2AVBlockActs,
    ):
        self.video_out = video_out^
        self.audio_out = audio_out^
        self.acts = acts^


# ── TRAIN FORWARD (activation-saving mirror of ltx2_block_forward_av) ────────
def ltx2_block_forward_av_train[S_V: Int, S_A: Int, N_TXT: Int](
    weights: LTX2AVBlockWeights,
    hidden: Tensor, ahs: Tensor,
    enc: Tensor, aenc: Tensor,
    v_temb: Tensor, a_temb: Tensor,
    v_ca_ss: Tensor, a_ca_ss: Tensor,
    v_ca_gate: Tensor, a_ca_gate: Tensor,
    v_prompt_ts: Tensor, a_prompt_ts: Tensor,
    v_cos: Tensor, v_sin: Tensor,
    a_cos: Tensor, a_sin: Tensor,
    ca_v_cos: Tensor, ca_v_sin: Tensor,
    ca_a_cos: Tensor, ca_a_sin: Tensor,
    eps: Float32, ctx: DeviceContext,
) raises -> LTX2AVTrainForward:
    var VD = 4096
    var AD = 2048
    var dummy = _dummy_t(hidden.dtype(), ctx)

    # ---- 1. video self-attn (AdaLN rows 0..2, rope, gated residual) ----
    ref vtab = weights._w("scale_shift_table")
    var v_shift_msa = _ada_row_pertok(vtab, v_temb, 0, VD, S_V, ctx)
    var v_scale_msa = _ada_row_pertok(vtab, v_temb, 1, VD, S_V, ctx)
    var v_gate_msa = _ada_row_pertok(vtab, v_temb, 2, VD, S_V, ctx)
    var v_shift_mlp = _ada_row_pertok(vtab, v_temb, 3, VD, S_V, ctx)
    var v_scale_mlp = _ada_row_pertok(vtab, v_temb, 4, VD, S_V, ctx)
    var v_gate_mlp = _ada_row_pertok(vtab, v_temb, 5, VD, S_V, ctx)

    var mod_h = _modulate_bc(
        _rms_norm_opt(hidden, weights, "norm1.weight", eps, ctx),
        v_scale_msa, v_shift_msa, ctx,
    )
    var r_at1 = _av_attention_train[S_V, S_V, 32, 128](
        weights, "attn1", mod_h, mod_h,
        True, v_cos, v_sin, False, dummy, dummy, eps, ctx,
    )
    var hs1 = add(hidden, mul(v_gate_msa, r_at1.out, ctx), ctx)

    # ---- audio self-attn ----
    ref atab = weights._w("audio_scale_shift_table")
    var a_shift_msa = _ada_row_pertok(atab, a_temb, 0, AD, S_A, ctx)
    var a_scale_msa = _ada_row_pertok(atab, a_temb, 1, AD, S_A, ctx)
    var a_gate_msa = _ada_row_pertok(atab, a_temb, 2, AD, S_A, ctx)
    var a_shift_mlp = _ada_row_pertok(atab, a_temb, 3, AD, S_A, ctx)
    var a_scale_mlp = _ada_row_pertok(atab, a_temb, 4, AD, S_A, ctx)
    var a_gate_mlp = _ada_row_pertok(atab, a_temb, 5, AD, S_A, ctx)

    var mod_a = _modulate_bc(
        _rms_norm_opt(ahs, weights, "audio_norm1.weight", eps, ctx),
        a_scale_msa, a_shift_msa, ctx,
    )
    var r_aat1 = _av_attention_train[S_A, S_A, 32, 64](
        weights, "audio_attn1", mod_a, mod_a,
        True, a_cos, a_sin, False, dummy, dummy, eps, ctx,
    )
    var ahss1 = add(ahs, mul(a_gate_msa, r_aat1.out, ctx), ctx)

    # ---- 2. video text cross-attn (AdaLN rows 6..8, KV-modulated context) ----
    var v_shift_ca = _ada_row_pertok(vtab, v_temb, 6, VD, S_V, ctx)
    var v_scale_ca = _ada_row_pertok(vtab, v_temb, 7, VD, S_V, ctx)
    var v_gate_ca = _ada_row_pertok(vtab, v_temb, 8, VD, S_V, ctx)
    var mod_h2 = _modulate_bc(
        _rms_norm_opt(hs1, weights, "norm2.weight", eps, ctx),
        v_scale_ca, v_shift_ca, ctx,
    )
    var mv_ctx: Tensor
    if weights._has("prompt_scale_shift_table"):
        mv_ctx = _kv_modulate(
            enc, weights._w("prompt_scale_shift_table"), v_prompt_ts,
            N_TXT, VD, ctx,
        )
    else:
        mv_ctx = enc.clone(ctx)
    var r_at2 = _av_attention_train[S_V, N_TXT, 32, 128](
        weights, "attn2", mod_h2, mv_ctx,
        False, dummy, dummy, False, dummy, dummy, eps, ctx,
    )
    var hs2 = add(hs1, mul(v_gate_ca, r_at2.out, ctx), ctx)

    # ---- audio text cross-attn ----
    var a_shift_ca = _ada_row_pertok(atab, a_temb, 6, AD, S_A, ctx)
    var a_scale_ca = _ada_row_pertok(atab, a_temb, 7, AD, S_A, ctx)
    var a_gate_ca = _ada_row_pertok(atab, a_temb, 8, AD, S_A, ctx)
    var mod_a2 = _modulate_bc(
        _rms_norm_opt(ahss1, weights, "audio_norm2.weight", eps, ctx),
        a_scale_ca, a_shift_ca, ctx,
    )
    var ma_ctx: Tensor
    if weights._has("audio_prompt_scale_shift_table"):
        ma_ctx = _kv_modulate(
            aenc, weights._w("audio_prompt_scale_shift_table"), a_prompt_ts,
            N_TXT, AD, ctx,
        )
    else:
        ma_ctx = aenc.clone(ctx)
    var r_aat2 = _av_attention_train[S_A, N_TXT, 32, 64](
        weights, "audio_attn2", mod_a2, ma_ctx,
        False, dummy, dummy, False, dummy, dummy, eps, ctx,
    )
    var ahss2 = add(ahss1, mul(a_gate_ca, r_aat2.out, ctx), ctx)

    # ---- 3. cross-modal a2v / v2a (shared pre-norms off hs2/ahss2) ----
    var norm_a2v = _rms_norm_opt(hs2, weights, "audio_to_video_norm.weight", eps, ctx)
    var norm_v2a = _rms_norm_opt(ahss2, weights, "video_to_audio_norm.weight", eps, ctx)
    var cm = _compute_cross_mod(
        weights._w("scale_shift_table_a2v_ca_video"),
        weights._w("scale_shift_table_a2v_ca_audio"),
        v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate, VD, AD, ctx,
    )

    var mod_video_a2v = _modulate_bc(norm_a2v, cm.v_a2v_scale, cm.v_a2v_shift, ctx)
    var mod_audio_a2v = _modulate_bc(norm_v2a, cm.a_a2v_scale, cm.a_a2v_shift, ctx)
    var r_a2v = _av_attention_train[S_V, S_A, 32, 64](
        weights, "audio_to_video_attn", mod_video_a2v, mod_audio_a2v,
        True, ca_v_cos, ca_v_sin, True, ca_a_cos, ca_a_sin, eps, ctx,
    )
    var hs3 = add(hs2, mul(cm.a2v_gate, r_a2v.out, ctx), ctx)

    var mod_video_v2a = _modulate_bc(norm_a2v, cm.v_v2a_scale, cm.v_v2a_shift, ctx)
    var mod_audio_v2a = _modulate_bc(norm_v2a, cm.a_v2a_scale, cm.a_v2a_shift, ctx)
    var r_v2a = _av_attention_train[S_A, S_V, 32, 64](
        weights, "video_to_audio_attn", mod_audio_v2a, mod_video_v2a,
        True, ca_a_cos, ca_a_sin, True, ca_v_cos, ca_v_sin, eps, ctx,
    )
    var ahss3 = add(ahss2, mul(cm.v2a_gate, r_v2a.out, ctx), ctx)

    # ---- 4. FFNs (gated residual; spine has no clamp) ----
    var mod_ff = _modulate_bc(
        _rms_norm_opt(hs3, weights, "norm3.weight", eps, ctx),
        v_scale_mlp, v_shift_mlp, ctx,
    )
    var h1_v = weights._linear_b(
        mod_ff, "ff.net.0.proj.weight", "ff.net.0.proj.bias", ctx)
    var h1g_v = gelu(h1_v, ctx)
    var ff_v = weights._linear_b(h1g_v, "ff.net.2.weight", "ff.net.2.bias", ctx)
    var video_out = add(hs3, mul(v_gate_mlp, ff_v, ctx), ctx)

    var mod_aff = _modulate_bc(
        _rms_norm_opt(ahss3, weights, "audio_norm3.weight", eps, ctx),
        a_scale_mlp, a_shift_mlp, ctx,
    )
    var h1_a = weights._linear_b(
        mod_aff, "audio_ff.net.0.proj.weight", "audio_ff.net.0.proj.bias", ctx)
    var h1g_a = gelu(h1_a, ctx)
    var ff_a = weights._linear_b(
        h1g_a, "audio_ff.net.2.weight", "audio_ff.net.2.bias", ctx)
    var audio_out = add(ahss3, mul(a_gate_mlp, ff_a, ctx), ctx)

    var acts = LTX2AVBlockActs(
        hidden.clone(ctx), ahs.clone(ctx),
        r_at1^, r_aat1^,
        hs1^, ahss1^,
        r_at2^, r_aat2^,
        hs2^, ahss2^,
        r_a2v^, r_v2a^,
        hs3^, ahss3^,
        h1_v^, h1_a^,
    )
    return LTX2AVTrainForward(video_out^, audio_out^, acts^)


# ── BACKWARD result ───────────────────────────────────────────────────────────
struct LTX2AVBlockGrads(Movable):
    var d_hidden: Tensor              # [1,S_V,4096]
    var d_ahs: Tensor                 # [1,S_A,2048]
    var lora: List[LoraPairGrad]      # d_A/d_B per attached adapter

    def __init__(
        out self, var d_hidden: Tensor, var d_ahs: Tensor,
        var lora: List[LoraPairGrad],
    ):
        self.d_hidden = d_hidden^
        self.d_ahs = d_ahs^
        self.lora = lora^


def _has_lora_factor(weights: LTX2AVBlockWeights, w_key: String) -> Bool:
    for i in range(len(weights.lora_names)):
        if weights.lora_names[i] == w_key:
            return True
    return False


# ── BLOCK BACKWARD (hand-chained reverse of ltx2_block_forward_av_train) ─────
def ltx2_block_backward_av[S_V: Int, S_A: Int, N_TXT: Int](
    weights: LTX2AVBlockWeights,
    acts: LTX2AVBlockActs,
    d_video: Tensor, d_audio: Tensor,   # [1,S_V,4096] / [1,S_A,2048]
    v_temb: Tensor, a_temb: Tensor,
    v_ca_ss: Tensor, a_ca_ss: Tensor,
    v_ca_gate: Tensor, a_ca_gate: Tensor,
    v_cos: Tensor, v_sin: Tensor,
    a_cos: Tensor, a_sin: Tensor,
    ca_v_cos: Tensor, ca_v_sin: Tensor,
    ca_a_cos: Tensor, ca_a_sin: Tensor,
    eps: Float32, ctx: DeviceContext,
) raises -> LTX2AVBlockGrads:
    var VD = 4096
    var AD = 2048
    var FFV = 16384
    var FFA = 8192
    var dummy = _dummy_t(d_video.dtype(), ctx)
    var lora_grads = List[LoraPairGrad]()

    # Recompute the AdaLN rows (cheap slices/adds; the same helper math).
    ref vtab = weights._w("scale_shift_table")
    ref atab = weights._w("audio_scale_shift_table")
    var v_scale_msa = _ada_row_pertok(vtab, v_temb, 1, VD, S_V, ctx)
    var v_gate_msa = _ada_row_pertok(vtab, v_temb, 2, VD, S_V, ctx)
    var v_scale_mlp = _ada_row_pertok(vtab, v_temb, 4, VD, S_V, ctx)
    var v_gate_mlp = _ada_row_pertok(vtab, v_temb, 5, VD, S_V, ctx)
    var v_scale_ca = _ada_row_pertok(vtab, v_temb, 7, VD, S_V, ctx)
    var v_gate_ca = _ada_row_pertok(vtab, v_temb, 8, VD, S_V, ctx)
    var a_scale_msa = _ada_row_pertok(atab, a_temb, 1, AD, S_A, ctx)
    var a_gate_msa = _ada_row_pertok(atab, a_temb, 2, AD, S_A, ctx)
    var a_scale_mlp = _ada_row_pertok(atab, a_temb, 4, AD, S_A, ctx)
    var a_gate_mlp = _ada_row_pertok(atab, a_temb, 5, AD, S_A, ctx)
    var a_scale_ca = _ada_row_pertok(atab, a_temb, 7, AD, S_A, ctx)
    var a_gate_ca = _ada_row_pertok(atab, a_temb, 8, AD, S_A, ctx)
    var cm = _compute_cross_mod(
        weights._w("scale_shift_table_a2v_ca_video"),
        weights._w("scale_shift_table_a2v_ca_audio"),
        v_ca_ss, a_ca_ss, v_ca_gate, a_ca_gate, VD, AD, ctx,
    )

    # ---- 4r. video FFN: video_out = hs3 + v_gate_mlp * ff. FFN LoRA (P6.2 (a)
    # rider) on ff.net.2 (down) + ff.net.0.proj (up), CONDITIONAL on the factor
    # being attached (attention-only paths do ZERO extra work) — mirrors the
    # video-block P3.1 pattern (ltx2_video_backward.mojo:252-287); x is hs3 here. ----
    var v_ffn_lora = (
        _has_lora_factor(weights, "ff.net.2.weight")
        or _has_lora_factor(weights, "ff.net.0.proj.weight"))
    var d_hs3 = d_video.clone(ctx)
    var d_ff_v = reshape(mul(d_video, v_gate_mlp, ctx), _sh2(S_V, VD), ctx)
    var d_h1g_v = linear_backward_dx(
        d_ff_v, weights._w("ff.net.2.weight"), S_V, FFV, VD, ctx)
    if v_ffn_lora:
        var h1g_v2 = gelu(reshape(acts.h1_v, _sh2(S_V, FFV), ctx), ctx)
        var l2 = _lora_pair_bwd(
            weights, String("ff.net.2.weight"), d_ff_v, h1g_v2, S_V, ctx, lora_grads)
        if l2:
            d_h1g_v = add(d_h1g_v, l2.value(), ctx)
    var d_h1_v = gelu_backward(
        d_h1g_v, reshape(acts.h1_v, _sh2(S_V, FFV), ctx), ctx)
    var d_mod_ff = linear_backward_dx(
        d_h1_v, weights._w("ff.net.0.proj.weight"), S_V, VD, FFV, ctx)
    if v_ffn_lora:
        var v_shift_mlp = _ada_row_pertok(vtab, v_temb, 3, VD, S_V, ctx)
        var mod_ff2 = reshape(
            _modulate_bc(
                _rms_norm_opt(acts.hs3, weights, "norm3.weight", eps, ctx),
                v_scale_mlp, v_shift_mlp, ctx),
            _sh2(S_V, VD), ctx)
        var l0 = _lora_pair_bwd(
            weights, String("ff.net.0.proj.weight"), d_h1_v, mod_ff2, S_V, ctx, lora_grads)
        if l0:
            d_mod_ff = add(d_mod_ff, l0.value(), ctx)
    var d_norm3 = mul(
        reshape(d_mod_ff, _shape3(1, S_V, VD), ctx),
        _one_plus(v_scale_mlp, ctx), ctx)
    d_hs3 = add(
        d_hs3, _rms_bwd_opt(d_norm3, acts.hs3, weights, "norm3.weight", eps, ctx),
        ctx)

    # ---- audio FFN. FFN LoRA (P6.2 (a) TRUE-672) on audio_ff.net.2 +
    # audio_ff.net.0.proj, x is ahss3 — same pattern as the video FFN above. ----
    var a_ffn_lora = (
        _has_lora_factor(weights, "audio_ff.net.2.weight")
        or _has_lora_factor(weights, "audio_ff.net.0.proj.weight"))
    var d_ahss3 = d_audio.clone(ctx)
    var d_ff_a = reshape(mul(d_audio, a_gate_mlp, ctx), _sh2(S_A, AD), ctx)
    var d_h1g_a = linear_backward_dx(
        d_ff_a, weights._w("audio_ff.net.2.weight"), S_A, FFA, AD, ctx)
    if a_ffn_lora:
        var h1g_a2 = gelu(reshape(acts.h1_a, _sh2(S_A, FFA), ctx), ctx)
        var l2a = _lora_pair_bwd(
            weights, String("audio_ff.net.2.weight"), d_ff_a, h1g_a2, S_A, ctx, lora_grads)
        if l2a:
            d_h1g_a = add(d_h1g_a, l2a.value(), ctx)
    var d_h1_a = gelu_backward(
        d_h1g_a, reshape(acts.h1_a, _sh2(S_A, FFA), ctx), ctx)
    var d_mod_aff = linear_backward_dx(
        d_h1_a, weights._w("audio_ff.net.0.proj.weight"), S_A, AD, FFA, ctx)
    if a_ffn_lora:
        var a_shift_mlp = _ada_row_pertok(atab, a_temb, 3, AD, S_A, ctx)
        var mod_aff2 = reshape(
            _modulate_bc(
                _rms_norm_opt(acts.ahss3, weights, "audio_norm3.weight", eps, ctx),
                a_scale_mlp, a_shift_mlp, ctx),
            _sh2(S_A, AD), ctx)
        var l0a = _lora_pair_bwd(
            weights, String("audio_ff.net.0.proj.weight"), d_h1_a, mod_aff2, S_A, ctx, lora_grads)
        if l0a:
            d_mod_aff = add(d_mod_aff, l0a.value(), ctx)
    var d_anorm3 = mul(
        reshape(d_mod_aff, _shape3(1, S_A, AD), ctx),
        _one_plus(a_scale_mlp, ctx), ctx)
    d_ahss3 = add(
        d_ahss3,
        _rms_bwd_opt(d_anorm3, acts.ahss3, weights, "audio_norm3.weight", eps, ctx),
        ctx)

    # ---- 3r. cross-modal. Both branches hang off (hs2, ahss2) via the SHARED
    # pre-norms norm_a2v=rms(hs2), norm_v2a=rms(ahss2):
    #   hs3   = hs2   + a2v_gate * a2v(Q=mod(norm_a2v), KV=mod(norm_v2a))
    #   ahss3 = ahss2 + v2a_gate * v2a(Q=mod(norm_v2a), KV=mod(norm_a2v))
    var d_hs2 = d_hs3.clone(ctx)
    var d_ahss2 = d_ahss3.clone(ctx)
    var d_a2v_out = mul(d_hs3, cm.a2v_gate, ctx)
    var d_v2a_out = mul(d_ahss3, cm.v2a_gate, ctx)

    var g_a2v = _av_attention_bwd[S_V, S_A, 32, 64](
        weights, "audio_to_video_attn", acts.a2v, d_a2v_out,
        True, ca_v_cos, ca_v_sin, True, ca_a_cos, ca_a_sin, eps, ctx,
        lora_grads)
    var g_v2a = _av_attention_bwd[S_A, S_V, 32, 64](
        weights, "video_to_audio_attn", acts.v2a, d_v2a_out,
        True, ca_a_cos, ca_a_sin, True, ca_v_cos, ca_v_sin, eps, ctx,
        lora_grads)

    # modulate backward (broadcast [1,1,D] scales; scale/shift grads not in scope)
    var d_norm_a2v = add(
        mul(g_a2v.d_q_src, _one_plus(cm.v_a2v_scale, ctx), ctx),
        mul(g_v2a.d_kv_src, _one_plus(cm.v_v2a_scale, ctx), ctx), ctx)
    var d_norm_v2a = add(
        mul(g_a2v.d_kv_src, _one_plus(cm.a_a2v_scale, ctx), ctx),
        mul(g_v2a.d_q_src, _one_plus(cm.a_v2a_scale, ctx), ctx), ctx)
    d_hs2 = add(
        d_hs2,
        _rms_bwd_opt(d_norm_a2v, acts.hs2, weights, "audio_to_video_norm.weight", eps, ctx),
        ctx)
    d_ahss2 = add(
        d_ahss2,
        _rms_bwd_opt(d_norm_v2a, acts.ahss2, weights, "video_to_audio_norm.weight", eps, ctx),
        ctx)

    # ---- 2r. video text cross-attn: hs2 = hs1 + v_gate_ca * attn2(...) ----
    # KV grads (text context) are dropped — untrained leaves outside the block.
    var d_hs1 = d_hs2.clone(ctx)
    var d_vca = mul(d_hs2, v_gate_ca, ctx)
    var g_at2 = _av_attention_bwd[S_V, N_TXT, 32, 128](
        weights, "attn2", acts.at2, d_vca,
        False, dummy, dummy, False, dummy, dummy, eps, ctx, lora_grads)
    var d_norm2 = mul(g_at2.d_q_src, _one_plus(v_scale_ca, ctx), ctx)
    d_hs1 = add(
        d_hs1, _rms_bwd_opt(d_norm2, acts.hs1, weights, "norm2.weight", eps, ctx),
        ctx)

    # ---- audio text cross-attn ----
    var d_ahss1 = d_ahss2.clone(ctx)
    var d_aca = mul(d_ahss2, a_gate_ca, ctx)
    var g_aat2 = _av_attention_bwd[S_A, N_TXT, 32, 64](
        weights, "audio_attn2", acts.aat2, d_aca,
        False, dummy, dummy, False, dummy, dummy, eps, ctx, lora_grads)
    var d_anorm2 = mul(g_aat2.d_q_src, _one_plus(a_scale_ca, ctx), ctx)
    d_ahss1 = add(
        d_ahss1,
        _rms_bwd_opt(d_anorm2, acts.ahss1, weights, "audio_norm2.weight", eps, ctx),
        ctx)

    # ---- 1r. video self-attn: hs1 = hidden + v_gate_msa * attn1(mod_h) ----
    # Q-source and KV-source are the SAME tensor (mod_h): sum both grads.
    var d_hidden = d_hs1.clone(ctx)
    var d_vsa = mul(d_hs1, v_gate_msa, ctx)
    var g_at1 = _av_attention_bwd[S_V, S_V, 32, 128](
        weights, "attn1", acts.at1, d_vsa,
        True, v_cos, v_sin, False, dummy, dummy, eps, ctx, lora_grads)
    var d_mod_h = add(g_at1.d_q_src, g_at1.d_kv_src, ctx)
    var d_norm1 = mul(d_mod_h, _one_plus(v_scale_msa, ctx), ctx)
    d_hidden = add(
        d_hidden,
        _rms_bwd_opt(d_norm1, acts.hidden, weights, "norm1.weight", eps, ctx),
        ctx)

    # ---- audio self-attn ----
    var d_ahs = d_ahss1.clone(ctx)
    var d_asa = mul(d_ahss1, a_gate_msa, ctx)
    var g_aat1 = _av_attention_bwd[S_A, S_A, 32, 64](
        weights, "audio_attn1", acts.aat1, d_asa,
        True, a_cos, a_sin, False, dummy, dummy, eps, ctx, lora_grads)
    var d_mod_a = add(g_aat1.d_q_src, g_aat1.d_kv_src, ctx)
    var d_anorm1 = mul(d_mod_a, _one_plus(a_scale_msa, ctx), ctx)
    d_ahs = add(
        d_ahs,
        _rms_bwd_opt(d_anorm1, acts.ahs, weights, "audio_norm1.weight", eps, ctx),
        ctx)

    return LTX2AVBlockGrads(d_hidden^, d_ahs^, lora_grads^)
