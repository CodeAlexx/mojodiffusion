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
#    video_to_audio_attn} (musubi LTX2_INCLUDE_PATTERNS_T2V).
#   Base weight grads / modulation-vector grads / text-context grads are NOT
#   produced (frozen / upstream-shared / untrained leaves).
#
# Every backward arm is a pre-existing gated op:
#   linear_backward(_dx)  ops/linalg_backward.mojo
#   rms_norm_backward     ops/norm_backward.mojo
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

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype

# forward ops (the spine's ops)
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import gelu, sigmoid
from serenitymojo.ops.attention import sdpa_cross_nomask
from serenitymojo.ops.tensor_algebra import (
    reshape, add, mul, mul_scalar, add_scalar,
)
from serenitymojo.models.dit.ltx2_rope import apply_ltx2_rope
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
from serenitymojo.ops.linalg_backward import linear_backward, linear_backward_dx
from serenitymojo.ops.norm_backward import rms_norm_backward
from serenitymojo.ops.activation_backward import gelu_backward, sigmoid_backward
from serenitymojo.ops.attention_backward import sdpa_backward_rect
from serenitymojo.ops.rope_struct_backward import rope_backward


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
        var rb = rms_norm_backward(d, x, weights._w(w_key), eps, ctx)
        return rb.d_x.clone(ctx)
    var xs = x.shape()
    var ones = _ones_t(xs[len(xs) - 1], x.dtype(), ctx)
    var rb = rms_norm_backward(d, x, ones, eps, ctx)
    return rb.d_x.clone(ctx)


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

        var dag_h = d_att_g.to_host(ctx)        # [SQ*inner]
        var af_h = acts.att_flat.to_host(ctx)   # [SQ*inner]
        var dg = List[Float32]()
        for s in range(SQ):
            for h in range(H):
                var acc = Float32(0.0)
                var base = s * inner + h * DH
                for d in range(DH):
                    acc += dag_h[base + d] * af_h[base + d]
                dg.append(acc)
        var d_gates = Tensor.from_host(dg, _sh2(SQ, H), d_att_g.dtype(), ctx)
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
    var rb_q = rms_norm_backward(
        d_q_flat, acts.q_pre, weights._w(mod_name + ".norm_q.weight"), eps, ctx)
    var rb_k = rms_norm_backward(
        d_k_flat, acts.k_pre, weights._w(mod_name + ".norm_k.weight"), eps, ctx)
    var d_v_flat = reshape(sg.d_v, _sh2(SKV, inner), ctx)

    # ── projections (base dx only — frozen; + LoRA d_A/d_B/d_x) ──
    var d_q_src = linear_backward_dx(
        rb_q.d_x, weights._w(mod_name + ".to_q.weight"), SQ, q_dim, inner, ctx)
    var lq = _lora_pair_bwd(
        weights, mod_name + ".to_q.weight", rb_q.d_x, q_src2, SQ, ctx,
        lora_grads)
    if lq:
        d_q_src = add(d_q_src, lq.value(), ctx)
    if d_gate_path:
        d_q_src = add(d_q_src, d_gate_path.value(), ctx)

    var d_kv_src = linear_backward_dx(
        rb_k.d_x, weights._w(mod_name + ".to_k.weight"), SKV, kv_dim, inner, ctx)
    var lk = _lora_pair_bwd(
        weights, mod_name + ".to_k.weight", rb_k.d_x, kv_src2, SKV, ctx,
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

    # ---- 4r. video FFN: video_out = hs3 + v_gate_mlp * ff ----
    var d_hs3 = d_video.clone(ctx)
    var d_ff_v = reshape(mul(d_video, v_gate_mlp, ctx), _sh2(S_V, VD), ctx)
    var d_h1g_v = linear_backward_dx(
        d_ff_v, weights._w("ff.net.2.weight"), S_V, FFV, VD, ctx)
    var d_h1_v = gelu_backward(
        d_h1g_v, reshape(acts.h1_v, _sh2(S_V, FFV), ctx), ctx)
    var d_mod_ff = linear_backward_dx(
        d_h1_v, weights._w("ff.net.0.proj.weight"), S_V, VD, FFV, ctx)
    var d_norm3 = mul(
        reshape(d_mod_ff, _shape3(1, S_V, VD), ctx),
        _one_plus(v_scale_mlp, ctx), ctx)
    d_hs3 = add(
        d_hs3, _rms_bwd_opt(d_norm3, acts.hs3, weights, "norm3.weight", eps, ctx),
        ctx)

    # ---- audio FFN ----
    var d_ahss3 = d_audio.clone(ctx)
    var d_ff_a = reshape(mul(d_audio, a_gate_mlp, ctx), _sh2(S_A, AD), ctx)
    var d_h1g_a = linear_backward_dx(
        d_ff_a, weights._w("audio_ff.net.2.weight"), S_A, FFA, AD, ctx)
    var d_h1_a = gelu_backward(
        d_h1g_a, reshape(acts.h1_a, _sh2(S_A, FFA), ctx), ctx)
    var d_mod_aff = linear_backward_dx(
        d_h1_a, weights._w("audio_ff.net.0.proj.weight"), S_A, AD, FFA, ctx)
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
