# serenitymojo/models/ltx2/ltx2_block.mojo
#
# LTX-2 (22B video DiT) CORE VIDEO TRANSFORMER BLOCK: forward (saving
# activations) + hand-chained backward (training) + LoRA variants, packaged in
# the EXACT style proven by serenitymojo/models/klein/single_block.mojo.
#
# SCOPE (per mojo-train-port task): the LoRA-TRAINED core video block —
#   self-attn (attn1, AdaLN + QK-RMSNorm + halfsplit-RoPE + SDPA + per-head
#   gate) + FFN (AdaLN + GELU). This is the surface a LTX-2 LoRA targets
#   (to_q/to_k/to_v/to_out, EDv2 ltx2.rs:357-358 / attention_forward_lora slots).
#   The full AV joint block (ltx2_dit.mojo LTX2AVBlockWeights, 6 attention
#   paths: attn1/audio_attn1/attn2/audio_attn2/audio_to_video/video_to_audio)
#   is OMITTED — the audio + cross-modal streams are not the video-LoRA surface,
#   and the task explicitly says to gate the core video DiT block when the AV
#   joint block is complex.  attn2 (text cross-attn) is also omitted from the
#   parity surface (same linear/rms/sdpa machinery, no new backward arms).
#
# FORWARD GRAPH (mirrors models/dit/ltx2_dit.mojo ltx2_block_forward_video_only
#   self-attn + FFN halves; ltx2_model.rs:765-882 attn1, :1074-1086 FFN):
#   With AdaLN vectors (shift,scale,gate)_{msa,mlp} each [D] from scale_shift_table:
#     norm_h  = rms_norm(hidden)                 # NO affine (ltx2 norm1 has none)
#     mod_h   = modulate(norm_h, scale_msa, shift_msa)   # (1+scale)*norm+shift
#     q = linear(mod_h, Wq, bq)  (+lora_q)       # [S,D]
#     k = linear(mod_h, Wk, bk)  (+lora_k)
#     v = linear(mod_h, Wv, bv)  (+lora_v)
#     q = rms_norm(q, q_norm[D]) ; k = rms_norm(k, k_norm[D])   # over FULL inner
#     q4,k4,v4 = reshape [1,S,H,Dh]
#     q4 = rope_halfsplit(q4,cos,sin) ; k4 = rope_halfsplit(k4,cos,sin)
#     attn = sdpa(q4,k4,v4, 1/sqrt(Dh)) -> att_flat [S,D]
#     gl   = linear(mod_h, gate_w[H,D], gate_b[H])  # [S,H]  per-head gate logits
#     gates= 2*sigmoid(gl)                       # [S,H]
#     att_g= att_flat[:,h*Dh+d] * gates[:,h]     # per-head broadcast
#     ao   = linear(att_g, Wo, bo) (+lora_o)     # [S,D]
#     hs   = hidden + gate_msa * ao              # gated residual (per-channel)
#     norm_ff = rms_norm(hs)                     # NO affine
#     mod_ff  = modulate(norm_ff, scale_mlp, shift_mlp)
#     h1 = linear(mod_ff, Wff0, bff0)            # [S, FF]
#     h1g= gelu(h1)                              # tanh-approx (ltx2_model.rs:279)
#     ff = linear(h1g, Wff2, bff2)               # [S, D]
#     out= hs + gate_mlp * ff                    # gated residual
#
# Base weights FROZEN (LoRA training). LoRA on Wq/Wk/Wv/Wo. The per-head gate
# weights gate_w/gate_b and the AdaLN vectors are NOT LoRA targets; their grads
# are produced (gate_w/gate_b weight grads; modvec grads as block outputs) but
# gate logits are part of the frozen base in the LoRA recipe (we still compute
# their grads for the parity gate's completeness — base weight grads).
#
# Mojo 1.0.0b1: `def` not `fn`; Tensor move-only (return Movable structs); host
# List[Float32] at the API boundary (move-only Tensor cannot be a collection
# element across the seam); no-bias linear = linear(x, w, Optional(None), ctx).

from std.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import sqrt
from std.memory import ArcPointer
from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor

comptime TArc = ArcPointer[Tensor]

# ── forward ops (GPU) ────────────────────────────────────────────────────────
from serenitymojo.ops.linear import linear
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.activations import gelu, sigmoid
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.rope import rope_halfsplit
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.tensor_algebra import (
    reshape, reshape_owned, slice, mul, mul_scalar, add,
)

# ── backward arms (GPU; all pre-built + gated) ───────────────────────────────
from serenitymojo.ops.linalg_backward import linear_backward, LinearGrads
from serenitymojo.ops.norm_backward import rms_norm_backward, RmsNormBackward
from serenitymojo.ops.activation_backward import gelu_backward, sigmoid_backward
from serenitymojo.ops.attention_backward import sdpa_backward, SdpaGrads
from serenitymojo.ops.elementwise_backward import modulate_backward, ModulateBackward
from serenitymojo.ops.rope_struct_backward import (
    gate_residual_backward, GateResidualGrads, rope_backward,
)


# ── host helpers ─────────────────────────────────────────────────────────────
def _ones(d: Int) -> List[Float32]:
    var o = List[Float32]()
    for _ in range(d):
        o.append(1.0)
    return o^


def _t(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.F32, ctx)


# Upload an F32 host list to a native BF16 device tensor (F32→bf16 cast on
# upload). Used for compute inputs (modvecs, ones, d_out) so every downstream
# op runs the native bf16 path. flame-core contract: bf16 in/out, F32 accumulate.
def _tb(vals: List[Float32], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host(vals, shape^, STDtype.BF16, ctx)


# Re-upload a BF16 saved activation to a native BF16 device tensor (verbatim, no
# F32 widening) so backward matmuls also run native bf16 (bf16·bf16, F32 accum).
def _tbf16(vals: List[BFloat16], var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    return Tensor.from_host_bf16(vals, shape^, ctx)


def _add_lists(a: List[Float32], b: List[Float32]) -> List[Float32]:
    var o = List[Float32]()
    for i in range(len(a)):
        o.append(a[i] + b[i])
    return o^


# RMSNorm with NO affine (ones weight) over the last dim. The LTX-2 norm1/norm3
# have no learnable weight; rms_norm needs a [D] weight so we pass ones.
def _rms_no_affine(
    x: Tensor, ones_t: Tensor, eps: Float32, ctx: DeviceContext
) raises -> Tensor:
    return rms_norm(x, ones_t, eps, ctx)


# ── modulation vectors (each [D]: shift,scale,gate for msa + mlp) ─────────────
struct LTX2ModVecs(Copyable, Movable):
    var shift_msa: List[Float32]
    var scale_msa: List[Float32]
    var gate_msa: List[Float32]
    var shift_mlp: List[Float32]
    var scale_mlp: List[Float32]
    var gate_mlp: List[Float32]

    def __init__(
        out self,
        var shift_msa: List[Float32], var scale_msa: List[Float32], var gate_msa: List[Float32],
        var shift_mlp: List[Float32], var scale_mlp: List[Float32], var gate_mlp: List[Float32],
    ):
        self.shift_msa = shift_msa^
        self.scale_msa = scale_msa^
        self.gate_msa = gate_msa^
        self.shift_mlp = shift_mlp^
        self.scale_mlp = scale_mlp^
        self.gate_mlp = gate_mlp^


# ── trainable + frozen weights (device-resident, uploaded once) ──────────────
#   Wq/Wk/Wv/Wo : [D,D]   to_q/k/v/out.0   (LoRA targets when training)
#   bq/bk/bv/bo : [D]
#   q_norm/k_norm: [D]    QK RMS scale (full inner_dim)
#   gate_w: [H,D] gate_b: [H]   per-head gate logits
#   Wff0: [FF,D] bff0: [FF] ; Wff2: [D,FF] bff2: [D]
struct LTX2BlockWeights(Copyable, Movable):
    var wq: TArc
    var bq: TArc
    var wk: TArc
    var bk: TArc
    var wv: TArc
    var bv: TArc
    var wo: TArc
    var bo: TArc
    var q_norm: TArc
    var k_norm: TArc
    var gate_w: TArc
    var gate_b: TArc
    var wff0: TArc
    var bff0: TArc
    var wff2: TArc
    var bff2: TArc

    def __init__(
        out self,
        var wq: List[Float32], var bq: List[Float32],
        var wk: List[Float32], var bk: List[Float32],
        var wv: List[Float32], var bv: List[Float32],
        var wo: List[Float32], var bo: List[Float32],
        var q_norm: List[Float32], var k_norm: List[Float32],
        var gate_w: List[Float32], var gate_b: List[Float32],
        var wff0: List[Float32], var bff0: List[Float32],
        var wff2: List[Float32], var bff2: List[Float32],
        D: Int, H: Int, FF: Int, ctx: DeviceContext,
    ) raises:
        # Native BF16 weights AND biases: every linear runs bf16·bf16 (F32
        # accumulate inside the GEMM), matching flame-core's bf16 contract.
        # Biases are bf16 too so linear's bias-dtype == x-dtype check passes when
        # activations are bf16 (the standard bf16 path, not the F32×bf16 mixed_base).
        self.wq = TArc(Tensor.from_host(wq^, [D, D], STDtype.BF16, ctx))
        self.bq = TArc(Tensor.from_host(bq^, [D], STDtype.BF16, ctx))
        self.wk = TArc(Tensor.from_host(wk^, [D, D], STDtype.BF16, ctx))
        self.bk = TArc(Tensor.from_host(bk^, [D], STDtype.BF16, ctx))
        self.wv = TArc(Tensor.from_host(wv^, [D, D], STDtype.BF16, ctx))
        self.bv = TArc(Tensor.from_host(bv^, [D], STDtype.BF16, ctx))
        self.wo = TArc(Tensor.from_host(wo^, [D, D], STDtype.BF16, ctx))
        self.bo = TArc(Tensor.from_host(bo^, [D], STDtype.BF16, ctx))
        self.q_norm = TArc(Tensor.from_host(q_norm^, [D], STDtype.BF16, ctx))
        self.k_norm = TArc(Tensor.from_host(k_norm^, [D], STDtype.BF16, ctx))
        self.gate_w = TArc(Tensor.from_host(gate_w^, [H, D], STDtype.BF16, ctx))
        self.gate_b = TArc(Tensor.from_host(gate_b^, [H], STDtype.BF16, ctx))
        self.wff0 = TArc(Tensor.from_host(wff0^, [FF, D], STDtype.BF16, ctx))
        self.bff0 = TArc(Tensor.from_host(bff0^, [FF], STDtype.BF16, ctx))
        self.wff2 = TArc(Tensor.from_host(wff2^, [D, FF], STDtype.BF16, ctx))
        self.bff2 = TArc(Tensor.from_host(bff2^, [D], STDtype.BF16, ctx))


# ── LoRA adapter (A [rank,in], B [out,rank], scale=alpha/rank) ───────────────
# Same math/shape contract as training/train_step.mojo LoraAdapter. We keep a
# self-contained host-list adapter here so the block is independently gateable.
struct LTX2Lora(Copyable, Movable):
    var a: List[Float32]   # [rank, in]
    var b: List[Float32]   # [out, rank]
    var rank: Int
    var in_f: Int
    var out_f: Int
    var scale: Float32

    def __init__(
        out self, var a: List[Float32], var b: List[Float32],
        rank: Int, in_f: Int, out_f: Int, scale: Float32,
    ):
        self.a = a^
        self.b = b^
        self.rank = rank
        self.in_f = in_f
        self.out_f = out_f
        self.scale = scale


# adapter forward contribution on x [M,in] -> [M,out] (device tensors).
# x is bf16 (mod_h / att_g); A/B uploaded bf16 so the two linears run native bf16.
def _lora_contrib(x: Tensor, lo: LTX2Lora, M: Int, ctx: DeviceContext) raises -> Tensor:
    var nb1 = Optional[Tensor](None)
    var t = linear(
        x, _tb(lo.a.copy(), [lo.rank, lo.in_f], ctx), nb1^, ctx,
    )  # [M,rank]
    var nb2 = Optional[Tensor](None)
    var dy = linear(
        t, _tb(lo.b.copy(), [lo.out_f, lo.rank], ctx), nb2^, ctx,
    )  # [M,out]
    return mul_scalar(dy, lo.scale, ctx)


struct LTX2LoraGrads(Copyable, Movable):
    var d_a: List[Float32]
    var d_b: List[Float32]
    var d_x: List[Float32]   # input-grad contribution from this adapter [M,in]

    def __init__(out self, var d_a: List[Float32], var d_b: List[Float32], var d_x: List[Float32]):
        self.d_a = d_a^
        self.d_b = d_b^
        self.d_x = d_x^


# adapter backward given d(projection-output) [M,out] and the adapter input x [M,in].
# x/d_proj_out are bf16; A/B uploaded bf16 -> both linear_backward calls run
# native bf16 (F32 accumulate). Returned grads read via to_host -> F32.
def _lora_bwd(
    d_proj_out: Tensor, x: Tensor, lo: LTX2Lora, M: Int, ctx: DeviceContext,
) raises -> LTX2LoraGrads:
    var nb_t = Optional[Tensor](None)
    var t = linear(
        x, _tb(lo.a.copy(), [lo.rank, lo.in_f], ctx), nb_t^, ctx,
    )  # [M,rank]
    var d_dy = mul_scalar(d_proj_out, lo.scale, ctx)  # [M,out]
    var lbB = linear_backward(
        d_dy, t, _tb(lo.b.copy(), [lo.out_f, lo.rank], ctx),
        M, lo.rank, lo.out_f, ctx,
    )
    var d_t = lbB.d_x.clone(ctx)    # [M,rank]
    var d_b = lbB.d_w.to_host(ctx)  # [out,rank] -> F32
    var lbA = linear_backward(
        d_t, x, _tb(lo.a.copy(), [lo.rank, lo.in_f], ctx),
        M, lo.in_f, lo.rank, ctx,
    )
    var d_a = lbA.d_w.to_host(ctx)  # [rank,in] -> F32
    var d_x = lbA.d_x.to_host(ctx)  # [M,in]  input-grad contribution -> F32
    return LTX2LoraGrads(d_a^, d_b^, d_x^)


# ── saved activations (host BF16 — half working set vs F32 TArc) ─────────────
# flame-core contract: BF16 in/out, F32 only inside GEMM accumulate.
# All saved fields are List[BFloat16] (to_host_bf16). Backward re-uploads via
# Tensor.from_host_bf16. Modulation vectors, gradients, and block I/O stay F32.
struct LTX2BlockSaved(Movable):
    var hidden: List[BFloat16]     # [S,D] block input
    var norm_h: List[BFloat16]     # [S,D] rms(hidden)
    var mod_h: List[BFloat16]      # [S,D] modulate(norm_h, scale_msa, shift_msa)
    var q_pre: List[BFloat16]      # [1,S,H,Dh] q after linear (pre rms)
    var k_pre: List[BFloat16]      # [1,S,H,Dh]
    var q_rms: List[BFloat16]      # [1,S,H,Dh] rms_norm(q_pre, q_norm)
    var k_rms: List[BFloat16]      # [1,S,H,Dh]
    var v: List[BFloat16]          # [1,S,H,Dh]
    var q_rope: List[BFloat16]     # [1,S,H,Dh]
    var k_rope: List[BFloat16]     # [1,S,H,Dh]
    var att_flat: List[BFloat16]   # [S,D] reshape(sdpa)
    var gl: List[BFloat16]         # [S,H] gate logits
    var gates: List[BFloat16]      # [S,H] 2*sigmoid(gl)
    var att_g: List[BFloat16]      # [S,D] gated attention output
    var hs: List[BFloat16]         # [S,D] post self-attn residual
    var norm_ff: List[BFloat16]    # [S,D] rms(hs)
    var mod_ff: List[BFloat16]     # [S,D] modulate(norm_ff, scale_mlp, shift_mlp)
    var h1: List[BFloat16]         # [S,FF] linear(mod_ff, Wff0)
    var h1g: List[BFloat16]        # [S,FF] gelu(h1)
    var ff: List[BFloat16]         # [S,D] linear(h1g, Wff2)

    def __init__(
        out self,
        var hidden: List[BFloat16], var norm_h: List[BFloat16], var mod_h: List[BFloat16],
        var q_pre: List[BFloat16], var k_pre: List[BFloat16],
        var q_rms: List[BFloat16], var k_rms: List[BFloat16], var v: List[BFloat16],
        var q_rope: List[BFloat16], var k_rope: List[BFloat16], var att_flat: List[BFloat16],
        var gl: List[BFloat16], var gates: List[BFloat16],
        var att_g: List[BFloat16], var hs: List[BFloat16],
        var norm_ff: List[BFloat16], var mod_ff: List[BFloat16],
        var h1: List[BFloat16], var h1g: List[BFloat16], var ff: List[BFloat16],
    ):
        self.hidden = hidden^
        self.norm_h = norm_h^
        self.mod_h = mod_h^
        self.q_pre = q_pre^
        self.k_pre = k_pre^
        self.q_rms = q_rms^
        self.k_rms = k_rms^
        self.v = v^
        self.q_rope = q_rope^
        self.k_rope = k_rope^
        self.att_flat = att_flat^
        self.gl = gl^
        self.gates = gates^
        self.att_g = att_g^
        self.hs = hs^
        self.norm_ff = norm_ff^
        self.mod_ff = mod_ff^
        self.h1 = h1^
        self.h1g = h1g^
        self.ff = ff^


struct LTX2BlockForward(Movable):
    var out: List[Float32]   # [S,D] block output (host boundary readback)
    var saved: LTX2BlockSaved

    def __init__(out self, var out: List[Float32], var saved: LTX2BlockSaved):
        self.out = out^
        self.saved = saved^


# ── per-head gate application: att_flat[S,D] * gates[S,H] (head-broadcast) ────
#   att4 = reshape(att_flat, [1,S,H,Dh]); g4 = reshape(gates,[1,S,H,1]);
#   gated = att4 * g4 (broadcast over Dh); -> [S,D].
def _apply_head_gate[H: Int, Dh: Int, S: Int](
    att_flat: Tensor, gates: Tensor, ctx: DeviceContext
) raises -> Tensor:
    var att4 = reshape(att_flat, [1, S, H, Dh], ctx)
    var g4 = reshape(gates, [1, S, H, 1], ctx)
    var gated = mul(att4, g4, ctx)   # broadcast partner [1,S,H,1] over [1,S,H,Dh]
    return reshape_owned(gated^, [S, D_of(H, Dh)])


def D_of(H: Int, Dh: Int) -> Int:
    return H * Dh


# ── FORWARD ──────────────────────────────────────────────────────────────────
def ltx2_block_forward[
    H: Int, Dh: Int, S: Int
](
    hidden: List[Float32],
    w: LTX2BlockWeights, mv: LTX2ModVecs,
    cos: Tensor, sin: Tensor,
    lq: LTX2Lora, lk: LTX2Lora, lv: LTX2Lora, lo: LTX2Lora,
    use_lora: Bool,
    D: Int, FF: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> LTX2BlockForward:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    # Native bf16 compute: all forward inputs uploaded as bf16. ones_t (affine),
    # block input, modvecs, and the rope tables (local cast) are bf16 so every
    # op (rms_norm, modulate, linear, rope, sdpa, gelu, gate) runs bf16·bf16.
    var ones_t = _tb(_ones(D), [D], ctx)
    var cos_b = cast_tensor(cos, STDtype.BF16, ctx)
    var sin_b = cast_tensor(sin, STDtype.BF16, ctx)

    var x_t = _tb(hidden, [S, D], ctx)

    # self-attn AdaLN
    var norm_h = _rms_no_affine(x_t, ones_t, eps, ctx)
    var mod_h = modulate(
        norm_h, _tb(mv.scale_msa.copy(), [D], ctx), _tb(mv.shift_msa.copy(), [D], ctx), ctx
    )

    # q,k,v = linear(mod_h, W*, b*) (+ lora)
    var q = linear(mod_h, w.wq[], Optional[Tensor](_clone(w.bq[], ctx)), ctx)
    var k = linear(mod_h, w.wk[], Optional[Tensor](_clone(w.bk[], ctx)), ctx)
    var v_lin = linear(mod_h, w.wv[], Optional[Tensor](_clone(w.bv[], ctx)), ctx)
    if use_lora:
        q = add(q, _lora_contrib(mod_h, lq, S, ctx), ctx)
        k = add(k, _lora_contrib(mod_h, lk, S, ctx), ctx)
        v_lin = add(v_lin, _lora_contrib(mod_h, lv, S, ctx), ctx)

    # QK-RMSNorm over FULL inner_dim D
    var q_rms_flat = rms_norm(q, w.q_norm[], eps, ctx)
    var k_rms_flat = rms_norm(k, w.k_norm[], eps, ctx)

    # reshape to [1,S,H,Dh] (byte no-op)
    var q_pre = reshape_owned(q^, [1, S, H, Dh])
    var k_pre = reshape_owned(k^, [1, S, H, Dh])
    var v4 = reshape_owned(v_lin^, [1, S, H, Dh])
    var q_rms = reshape_owned(q_rms_flat^, [1, S, H, Dh])
    var k_rms = reshape_owned(k_rms_flat^, [1, S, H, Dh])

    # halfsplit rope (bf16 tables -> native bf16)
    var q_rope = rope_halfsplit(q_rms, cos_b, sin_b, ctx)
    var k_rope = rope_halfsplit(k_rms, cos_b, sin_b, ctx)

    var att = sdpa_nomask[1, S, H, Dh](q_rope, k_rope, v4, scale, ctx)
    var att_flat = reshape_owned(att^, [S, D])

    # per-head gate: gl=linear(mod_h,gate_w,gate_b); gates=2*sigmoid(gl)
    var gl = linear(mod_h, w.gate_w[], Optional[Tensor](_clone(w.gate_b[], ctx)), ctx)  # [S,H]
    var gates = mul_scalar(sigmoid(gl, ctx), Float32(2.0), ctx)  # [S,H]
    var att_g = _apply_head_gate[H, Dh, S](att_flat, gates, ctx)

    # to_out + lora
    var ao = linear(att_g, w.wo[], Optional[Tensor](_clone(w.bo[], ctx)), ctx)
    if use_lora:
        ao = add(ao, _lora_contrib(att_g, lo, S, ctx), ctx)

    # gated residual: hs = hidden + gate_msa * ao
    var hs = residual_gate(x_t, _tb(mv.gate_msa.copy(), [D], ctx), ao, ctx)

    # FFN AdaLN
    var norm_ff = _rms_no_affine(hs, ones_t, eps, ctx)
    var mod_ff = modulate(
        norm_ff, _tb(mv.scale_mlp.copy(), [D], ctx), _tb(mv.shift_mlp.copy(), [D], ctx), ctx
    )
    var h1 = linear(mod_ff, w.wff0[], Optional[Tensor](_clone(w.bff0[], ctx)), ctx)  # [S,FF]
    var h1g = gelu(h1, ctx)
    var ff = linear(h1g, w.wff2[], Optional[Tensor](_clone(w.bff2[], ctx)), ctx)     # [S,D]
    # Block output read back as F32 (to_host upcasts bf16->F32): public API stays F32.
    var out = residual_gate(hs, _tb(mv.gate_mlp.copy(), [D], ctx), ff, ctx).to_host(ctx)

    # Save activations as BF16 host lists (halves working set vs F32 device TArc).
    var saved = LTX2BlockSaved(
        x_t.to_host_bf16(ctx), norm_h.to_host_bf16(ctx), mod_h.to_host_bf16(ctx),
        q_pre.to_host_bf16(ctx), k_pre.to_host_bf16(ctx),
        q_rms.to_host_bf16(ctx), k_rms.to_host_bf16(ctx), v4.to_host_bf16(ctx),
        q_rope.to_host_bf16(ctx), k_rope.to_host_bf16(ctx), att_flat.to_host_bf16(ctx),
        gl.to_host_bf16(ctx), gates.to_host_bf16(ctx),
        att_g.to_host_bf16(ctx), hs.to_host_bf16(ctx),
        norm_ff.to_host_bf16(ctx), mod_ff.to_host_bf16(ctx),
        h1.to_host_bf16(ctx), h1g.to_host_bf16(ctx), ff.to_host_bf16(ctx),
    )
    return LTX2BlockForward(out^, saved^)


# D2D clone for a bias tensor we hand to a linear() Optional (consumed by move).
def _clone(x: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dev = ctx.enqueue_create_buffer[DType.uint8](x.nbytes())
    ctx.enqueue_copy(dst_buf=dev, src_buf=x.buf)
    ctx.synchronize()
    return Tensor(dev^, x.shape().copy(), x.dtype())


# ── backward result ──────────────────────────────────────────────────────────
struct LTX2BlockGrads(Copyable, Movable):
    var d_hidden: List[Float32]
    # base weight grads
    var d_wq: List[Float32]
    var d_bq: List[Float32]
    var d_wk: List[Float32]
    var d_bk: List[Float32]
    var d_wv: List[Float32]
    var d_bv: List[Float32]
    var d_wo: List[Float32]
    var d_bo: List[Float32]
    var d_q_norm: List[Float32]
    var d_k_norm: List[Float32]
    var d_gate_w: List[Float32]
    var d_gate_b: List[Float32]
    var d_wff0: List[Float32]
    var d_bff0: List[Float32]
    var d_wff2: List[Float32]
    var d_bff2: List[Float32]
    # modulation-vector grads (block outputs)
    var d_shift_msa: List[Float32]
    var d_scale_msa: List[Float32]
    var d_gate_msa: List[Float32]
    var d_shift_mlp: List[Float32]
    var d_scale_mlp: List[Float32]
    var d_gate_mlp: List[Float32]
    # LoRA grads
    var d_lq_a: List[Float32]
    var d_lq_b: List[Float32]
    var d_lk_a: List[Float32]
    var d_lk_b: List[Float32]
    var d_lv_a: List[Float32]
    var d_lv_b: List[Float32]
    var d_lo_a: List[Float32]
    var d_lo_b: List[Float32]

    def __init__(
        out self,
        var d_hidden: List[Float32],
        var d_wq: List[Float32], var d_bq: List[Float32],
        var d_wk: List[Float32], var d_bk: List[Float32],
        var d_wv: List[Float32], var d_bv: List[Float32],
        var d_wo: List[Float32], var d_bo: List[Float32],
        var d_q_norm: List[Float32], var d_k_norm: List[Float32],
        var d_gate_w: List[Float32], var d_gate_b: List[Float32],
        var d_wff0: List[Float32], var d_bff0: List[Float32],
        var d_wff2: List[Float32], var d_bff2: List[Float32],
        var d_shift_msa: List[Float32], var d_scale_msa: List[Float32], var d_gate_msa: List[Float32],
        var d_shift_mlp: List[Float32], var d_scale_mlp: List[Float32], var d_gate_mlp: List[Float32],
        var d_lq_a: List[Float32], var d_lq_b: List[Float32],
        var d_lk_a: List[Float32], var d_lk_b: List[Float32],
        var d_lv_a: List[Float32], var d_lv_b: List[Float32],
        var d_lo_a: List[Float32], var d_lo_b: List[Float32],
    ):
        self.d_hidden = d_hidden^
        self.d_wq = d_wq^; self.d_bq = d_bq^
        self.d_wk = d_wk^; self.d_bk = d_bk^
        self.d_wv = d_wv^; self.d_bv = d_bv^
        self.d_wo = d_wo^; self.d_bo = d_bo^
        self.d_q_norm = d_q_norm^; self.d_k_norm = d_k_norm^
        self.d_gate_w = d_gate_w^; self.d_gate_b = d_gate_b^
        self.d_wff0 = d_wff0^; self.d_bff0 = d_bff0^
        self.d_wff2 = d_wff2^; self.d_bff2 = d_bff2^
        self.d_shift_msa = d_shift_msa^; self.d_scale_msa = d_scale_msa^; self.d_gate_msa = d_gate_msa^
        self.d_shift_mlp = d_shift_mlp^; self.d_scale_mlp = d_scale_mlp^; self.d_gate_mlp = d_gate_mlp^
        self.d_lq_a = d_lq_a^; self.d_lq_b = d_lq_b^
        self.d_lk_a = d_lk_a^; self.d_lk_b = d_lk_b^
        self.d_lv_a = d_lv_a^; self.d_lv_b = d_lv_b^
        self.d_lo_a = d_lo_a^; self.d_lo_b = d_lo_b^


def _empty() -> List[Float32]:
    var o = List[Float32]()
    return o^


# accumulate b into a (host, same length).
def _acc(mut a: List[Float32], b: List[Float32]):
    for i in range(len(a)):
        a[i] = a[i] + b[i]


# ── BACKWARD (hand-chained, reverse of forward) ──────────────────────────────
def ltx2_block_backward[
    H: Int, Dh: Int, S: Int
](
    d_out: List[Float32],
    w: LTX2BlockWeights, mv: LTX2ModVecs, saved: LTX2BlockSaved,
    cos: Tensor, sin: Tensor,
    lq: LTX2Lora, lk: LTX2Lora, lv: LTX2Lora, lo: LTX2Lora,
    use_lora: Bool,
    D: Int, FF: Int, eps: Float32,
    ctx: DeviceContext,
) raises -> LTX2BlockGrads:
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    # d_out is the incoming gradient: kept F32 (gate_residual_backward requires
    # an F32 grad_out and an F32 per-channel gate). All other backward ops run
    # native bf16; the few F32-only boundaries are local-cast (noted at each site).
    var d_out_t = _t(d_out, [S, D], ctx)

    # Re-upload saved BF16 activations to NATIVE BF16 device tensors (verbatim,
    # no F32 widening) so backward matmuls run bf16·bf16, F32 accumulate.
    var sv_hs      = _tbf16(saved.hs.copy(),       [S, D],        ctx)
    var sv_ff      = _tbf16(saved.ff.copy(),       [S, D],        ctx)
    var sv_h1g     = _tbf16(saved.h1g.copy(),      [S, FF],       ctx)
    var sv_h1      = _tbf16(saved.h1.copy(),       [S, FF],       ctx)
    var sv_mod_ff  = _tbf16(saved.mod_ff.copy(),   [S, D],        ctx)
    var sv_norm_ff = _tbf16(saved.norm_ff.copy(),  [S, D],        ctx)
    var sv_hidden  = _tbf16(saved.hidden.copy(),   [S, D],        ctx)
    var sv_att_g   = _tbf16(saved.att_g.copy(),    [S, D],        ctx)
    var sv_att_flat= _tbf16(saved.att_flat.copy(), [S, D],        ctx)
    var sv_gates   = _tbf16(saved.gates.copy(),    [S, H],        ctx)
    var sv_gl      = _tbf16(saved.gl.copy(),       [S, H],        ctx)
    var sv_mod_h   = _tbf16(saved.mod_h.copy(),    [S, D],        ctx)
    var sv_q_rope  = _tbf16(saved.q_rope.copy(),   [1, S, H, Dh], ctx)
    var sv_k_rope  = _tbf16(saved.k_rope.copy(),   [1, S, H, Dh], ctx)
    var sv_v       = _tbf16(saved.v.copy(),        [1, S, H, Dh], ctx)
    var sv_q_pre   = _tbf16(saved.q_pre.copy(),    [1, S, H, Dh], ctx)
    var sv_k_pre   = _tbf16(saved.k_pre.copy(),    [1, S, H, Dh], ctx)
    var sv_norm_h  = _tbf16(saved.norm_h.copy(),   [S, D],        ctx)

    # bf16 rope tables for the bf16 _ao recompute (gate_residual y branch).
    var ones_t = _tb(_ones(D), [D], ctx)

    # ── FFN gated residual: out = hs + gate_mlp * ff ──
    # gate_residual_backward is an F32-grad-out boundary: d_out_t stays F32, the
    # per-channel gate is F32; it accepts bf16 x/y (sv_hs/sv_ff) and returns F32
    # grads. Cast its F32 d_x/d_y down to bf16 to rejoin the bf16 grad chain.
    var grg_mlp = gate_residual_backward(
        d_out_t, sv_hs, _t(mv.gate_mlp.copy(), [D], ctx), sv_ff, ctx
    )
    var d_gate_mlp = grg_mlp.d_g.to_host(ctx)   # [D] -> F32 returned grad
    var d_hs = cast_tensor(grg_mlp.d_x, STDtype.BF16, ctx)   # [S,D] bf16
    var d_ff = cast_tensor(grg_mlp.d_y, STDtype.BF16, ctx)   # [S,D] bf16

    # ff = linear(h1g, Wff2, bff2)
    var lb_ff2 = linear_backward(d_ff, sv_h1g, w.wff2[], S, FF, D, ctx)
    var d_wff2 = lb_ff2.d_w.to_host(ctx)
    var d_bff2 = lb_ff2.d_b.to_host(ctx)
    var d_h1g = lb_ff2.d_x.clone(ctx)               # [S,FF] bf16

    # h1g = gelu(h1)
    var d_h1 = gelu_backward(d_h1g, sv_h1, ctx)  # [S,FF] bf16

    # h1 = linear(mod_ff, Wff0, bff0)
    var lb_ff0 = linear_backward(d_h1, sv_mod_ff, w.wff0[], S, D, FF, ctx)
    var d_wff0 = lb_ff0.d_w.to_host(ctx)
    var d_bff0 = lb_ff0.d_b.to_host(ctx)
    var d_mod_ff = lb_ff0.d_x.clone(ctx)            # [S,D] bf16

    # mod_ff = modulate(norm_ff, scale_mlp, shift_mlp)
    var mb_mlp = modulate_backward(
        d_mod_ff, sv_norm_ff, _tb(mv.scale_mlp.copy(), [D], ctx), ctx
    )
    var d_scale_mlp = mb_mlp.d_scale.to_host(ctx)
    var d_shift_mlp = mb_mlp.d_shift.to_host(ctx)
    var d_norm_ff = mb_mlp.d_x.clone(ctx)           # [S,D] bf16

    # norm_ff = rms(hs)  (no affine -> discard d_g)
    var rb_ff = rms_norm_backward(d_norm_ff, sv_hs, ones_t, eps, ctx)
    var d_hs_from_ff = rb_ff.d_x.clone(ctx)         # [S,D] bf16

    # accumulate the two paths into d_hs (residual + ffn-norm), both bf16.
    var d_hs_total_b = add(d_hs, d_hs_from_ff, ctx)  # [S,D] bf16
    # gate_residual_backward needs an F32 grad_out -> local cast up for the call.
    var d_hs_total = cast_tensor(d_hs_total_b, STDtype.F32, ctx)

    # ── self-attn gated residual: hs = hidden + gate_msa * ao ──
    var grg_msa = gate_residual_backward(
        d_hs_total, sv_hidden, _t(mv.gate_msa.copy(), [D], ctx), _ao(saved, w, lo, use_lora, S, D, FF, ctx), ctx
    )
    var d_gate_msa = grg_msa.d_g.to_host(ctx)
    var d_hidden_res = cast_tensor(grg_msa.d_x, STDtype.BF16, ctx)  # [S,D] bf16
    var d_ao = cast_tensor(grg_msa.d_y, STDtype.BF16, ctx)          # [S,D] bf16

    # ao = linear(att_g, Wo, bo) (+ lora_o)
    var lb_o = linear_backward(d_ao, sv_att_g, w.wo[], S, D, D, ctx)
    var d_wo = lb_o.d_w.to_host(ctx)
    var d_bo = lb_o.d_b.to_host(ctx)
    var d_att_g = lb_o.d_x.clone(ctx)               # [S,D] bf16

    var d_lo_a = _empty()
    var d_lo_b = _empty()
    if use_lora:
        var lg = _lora_bwd(d_ao, sv_att_g, lo, S, ctx)
        d_lo_a = lg.d_a.copy()
        d_lo_b = lg.d_b.copy()
        # lora_o input is att_g -> add its d_x contribution (bf16 to match chain)
        d_att_g = add(d_att_g, _tb(lg.d_x.copy(), [S, D], ctx), ctx)

    # att_g = att_flat (head-gated by gates) : att4*g4
    # backward splits to d_att_flat and d_gates.
    var dag = _head_gate_backward[H, Dh, S](d_att_g, sv_att_flat, sv_gates, D, ctx)
    var d_att_flat = dag.d_att.clone(ctx)           # [S,D] bf16
    var d_gates = dag.d_gates.clone(ctx)            # [S,H] bf16

    # gates = 2*sigmoid(gl) -> d_gl = 2 * sigmoid'(gl) * d_gates
    var d_sig = mul_scalar(d_gates, Float32(2.0), ctx)
    var d_gl = sigmoid_backward(d_sig, sv_gl, ctx)   # [S,H] bf16

    # gl = linear(mod_h, gate_w, gate_b)
    var lb_gl = linear_backward(d_gl, sv_mod_h, w.gate_w[], S, D, H, ctx)
    var d_gate_w = lb_gl.d_w.to_host(ctx)
    var d_gate_b = lb_gl.d_b.to_host(ctx)
    var d_mod_h_gate = lb_gl.d_x.clone(ctx)         # [S,D] bf16 from gate path

    # att_flat = reshape(sdpa(q_rope,k_rope,v)) -> sdpa backward (native bf16:
    # q/k/v/d_out all bf16).
    var d_att4 = reshape(d_att_flat, [1, S, H, Dh], ctx)
    var sg = sdpa_backward[1, S, H, Dh](
        sv_q_rope, sv_k_rope, sv_v, d_att4, scale, ctx
    )
    var d_q_rope = sg.d_q.clone(ctx)   # bf16
    var d_k_rope = sg.d_k.clone(ctx)   # bf16
    var d_v4 = sg.d_v.clone(ctx)       # bf16

    # rope backward (halfsplit, interleaved=False). rope_backward is F32-ONLY
    # (grad_out/cos/sin): local-cast bf16 grads + tables up to F32, run, cast the
    # d_x result back to bf16 to rejoin the bf16 chain.
    var d_q_rope32 = cast_tensor(d_q_rope, STDtype.F32, ctx)
    var d_k_rope32 = cast_tensor(d_k_rope, STDtype.F32, ctx)
    var d_q_rms32 = rope_backward(d_q_rope32, cos, sin, False, ctx)  # [1,S,H,Dh] F32
    var d_k_rms32 = rope_backward(d_k_rope32, cos, sin, False, ctx)
    var d_q_rms = cast_tensor(d_q_rms32, STDtype.BF16, ctx)
    var d_k_rms = cast_tensor(d_k_rms32, STDtype.BF16, ctx)

    # reshape [1,S,H,Dh] -> [S,D] (byte no-op) for the rms over full D
    var d_q_rms_flat = reshape(d_q_rms, [S, D], ctx)
    var d_k_rms_flat = reshape(d_k_rms, [S, D], ctx)
    var d_v_flat = reshape(d_v4, [S, D], ctx)

    # q_rms = rms_norm(q, q_norm) over full D ; k_rms likewise (q_norm bf16)
    var q_flat = reshape(sv_q_pre, [S, D], ctx)
    var k_flat = reshape(sv_k_pre, [S, D], ctx)
    var rb_q = rms_norm_backward(d_q_rms_flat, q_flat, w.q_norm[], eps, ctx)
    var rb_k = rms_norm_backward(d_k_rms_flat, k_flat, w.k_norm[], eps, ctx)
    var d_q_norm = rb_q.d_g.to_host(ctx)
    var d_k_norm = rb_k.d_g.to_host(ctx)
    var d_q = rb_q.d_x.clone(ctx)   # [S,D] bf16
    var d_k = rb_k.d_x.clone(ctx)
    var d_v = d_v_flat.clone(ctx)   # [S,D] bf16 (no rms on v)

    # q = linear(mod_h, Wq, bq) (+lora_q) ; same for k,v
    var lb_q = linear_backward(d_q, sv_mod_h, w.wq[], S, D, D, ctx)
    var lb_k = linear_backward(d_k, sv_mod_h, w.wk[], S, D, D, ctx)
    var lb_v = linear_backward(d_v, sv_mod_h, w.wv[], S, D, D, ctx)
    var d_wq = lb_q.d_w.to_host(ctx); var d_bq = lb_q.d_b.to_host(ctx)
    var d_wk = lb_k.d_w.to_host(ctx); var d_bk = lb_k.d_b.to_host(ctx)
    var d_wv = lb_v.d_w.to_host(ctx); var d_bv = lb_v.d_b.to_host(ctx)

    # d_mod_h accumulates from q,k,v linears + the gate linear path (all bf16).
    var d_mod_h = add(lb_q.d_x, lb_k.d_x, ctx)
    d_mod_h = add(d_mod_h, lb_v.d_x, ctx)
    d_mod_h = add(d_mod_h, d_mod_h_gate, ctx)

    var d_lq_a = _empty(); var d_lq_b = _empty()
    var d_lk_a = _empty(); var d_lk_b = _empty()
    var d_lv_a = _empty(); var d_lv_b = _empty()
    if use_lora:
        var gq = _lora_bwd(d_q, sv_mod_h, lq, S, ctx)
        var gk = _lora_bwd(d_k, sv_mod_h, lk, S, ctx)
        var gv = _lora_bwd(d_v, sv_mod_h, lv, S, ctx)
        d_lq_a = gq.d_a.copy(); d_lq_b = gq.d_b.copy()
        d_lk_a = gk.d_a.copy(); d_lk_b = gk.d_b.copy()
        d_lv_a = gv.d_a.copy(); d_lv_b = gv.d_b.copy()
        d_mod_h = add(d_mod_h, _tb(gq.d_x.copy(), [S, D], ctx), ctx)
        d_mod_h = add(d_mod_h, _tb(gk.d_x.copy(), [S, D], ctx), ctx)
        d_mod_h = add(d_mod_h, _tb(gv.d_x.copy(), [S, D], ctx), ctx)

    # mod_h = modulate(norm_h, scale_msa, shift_msa)
    var mb_msa = modulate_backward(
        d_mod_h, sv_norm_h, _tb(mv.scale_msa.copy(), [D], ctx), ctx
    )
    var d_scale_msa = mb_msa.d_scale.to_host(ctx)
    var d_shift_msa = mb_msa.d_shift.to_host(ctx)
    var d_norm_h = mb_msa.d_x.clone(ctx)            # bf16

    # norm_h = rms(hidden) (no affine)
    var rb_h = rms_norm_backward(d_norm_h, sv_hidden, ones_t, eps, ctx)
    var d_hidden_from_norm = rb_h.d_x.clone(ctx)    # bf16

    # d_hidden = residual branch + norm branch (both bf16); read out as F32.
    var d_hidden = add(d_hidden_res, d_hidden_from_norm, ctx).to_host(ctx)

    return LTX2BlockGrads(
        d_hidden^,
        d_wq^, d_bq^, d_wk^, d_bk^, d_wv^, d_bv^, d_wo^, d_bo^,
        d_q_norm^, d_k_norm^, d_gate_w^, d_gate_b^,
        d_wff0^, d_bff0^, d_wff2^, d_bff2^,
        d_shift_msa^, d_scale_msa^, d_gate_msa^,
        d_shift_mlp^, d_scale_mlp^, d_gate_mlp^,
        d_lq_a^, d_lq_b^, d_lk_a^, d_lk_b^, d_lv_a^, d_lv_b^, d_lo_a^, d_lo_b^,
    )


# recompute `ao` (the gated `y` of the self-attn residual) for gate_residual_backward.
# att_g is now stored as List[BFloat16]; re-upload as BF16 (gate_residual_backward
# now accepts BF16 y via internal cast-up).
def _ao(
    saved: LTX2BlockSaved, w: LTX2BlockWeights, lo: LTX2Lora, use_lora: Bool,
    S: Int, D: Int, FF: Int, ctx: DeviceContext
) raises -> Tensor:
    var att_g_t = _tbf16(saved.att_g.copy(), [S, D], ctx)
    var ao = linear(att_g_t, w.wo[], Optional[Tensor](_clone(w.bo[], ctx)), ctx)
    if use_lora:
        ao = add(ao, _lora_contrib(att_g_t, lo, S, ctx), ctx)
    return ao^


# ── per-head gate backward ───────────────────────────────────────────────────
# forward: att_g[s, h*Dh+d] = att_flat[s,h*Dh+d] * gates[s,h]
#   d_att_flat[s,h*Dh+d] = d_att_g[s,h*Dh+d] * gates[s,h]
#   d_gates[s,h]         = sum_d d_att_g[s,h*Dh+d] * att_flat[s,h*Dh+d]
struct _HeadGateGrads(Movable):
    var d_att: Tensor   # [S,D]
    var d_gates: Tensor # [S,H]

    def __init__(out self, var d_att: Tensor, var d_gates: Tensor):
        self.d_att = d_att^
        self.d_gates = d_gates^


def _head_gate_backward[H: Int, Dh: Int, S: Int](
    d_att_g: Tensor, att_flat: Tensor, gates: Tensor, D: Int, ctx: DeviceContext
) raises -> _HeadGateGrads:
    # d_att = d_att_g * gates(head-broadcast)
    var d4 = reshape(d_att_g, [1, S, H, Dh], ctx)
    var g4 = reshape(gates, [1, S, H, 1], ctx)
    var d_att4 = mul(d4, g4, ctx)
    var d_att = reshape_owned(d_att4^, [S, D])

    # d_gates[s,h] = sum_d d_att_g[s,h*Dh+d]*att_flat[s,h*Dh+d]
    # compute prod then sum over Dh on host (H is small enough; S*D readback).
    var dag_h = d_att_g.to_host(ctx)
    var af_h = att_flat.to_host(ctx)
    var dg = List[Float32]()
    for s in range(S):
        for h in range(H):
            var acc = Float32(0.0)
            var base = s * D + h * Dh
            for d in range(Dh):
                acc += dag_h[base + d] * af_h[base + d]
            dg.append(acc)
    # bf16 d_gates so the downstream gate chain (mul_scalar -> sigmoid_backward
    # against bf16 sv_gl) runs native bf16 without a dtype mismatch.
    var d_gates = _tb(dg^, [S, H], ctx)
    return _HeadGateGrads(d_att^, d_gates^)
