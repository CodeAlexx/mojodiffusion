# MiniMax-H3 LoRA training block -- PARITY-ONLY device reference.
#
# Oracle: kohya-ss/musubi-tuner b8717864713c9e4e7ef3d56eba1fc695a9b626a5,
# `minimax_h3/model.py::DiTBlock.forward` and `networks/lora_minimax_h3.py`.
# Python is evidence only; this Mojo/MAX reference composes shared device
# forward/backward ops and contains no host-loop model math. It must never be
# imported by a training entrypoint. Product H3 requires a separate BF16 flash-
# attention implementation and gate.
#
# CHECKPOINT BOUNDARY (binding):
# * This module consumes the Musubi MODULE layout: fused QKV rows are
#   `[all-q; all-k; all-v]`; FC1 rows are raw `[gate; value]`.
# * The released frozen checkpoint stores fused QKV per-head interleaved and
#   raw FC1 `[gate; value]`.  A training loader must deinterleave QKV once and
#   must NOT apply Serenity inference's FC1 swap.
# * Serenity inference tensors are `[all-q; all-k; all-v]` plus transformed
#   FC1 `[value; gate]`; call `minimax_h3_training_swap_fc1_rows_device` on
#   both the frozen FC1 weight and FC1 LoRA B before constructing this block.
# * LoRA B/up rows must follow the same effective output-row layout as the base
#   projection presented to this module.  Save/load must invert any boundary
#   row transform exactly once; A/down rows are unaffected.
#
# Runtime guards restrict this module to the reduced F32 gate only
# (S=3,H=56,Dh=4,D=8,F=12,R=2,Rot=2).
# This is not released-geometry, BF16, stack, optimizer, or trainer evidence.

from max.gpu.host import DeviceContext
from std.collections import List, Optional
from std.math import isfinite, sqrt
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward_dx, linear_backward_dw
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.norm_backward import rms_norm_backward_dx
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward
from serenitymojo.ops.attention import sdpa_nomask
from serenitymojo.ops.attention_backward import sdpa_backward
from serenitymojo.ops.rope import rope_halfsplit_full_head_broadcast
from serenitymojo.ops.rope_struct_backward import (
    gate_residual_backward_dxdy,
    rope_halfsplit_full_backward,
)
from serenitymojo.ops.tensor_algebra import (
    add,
    concat,
    full_device,
    mul_scalar,
    reshape,
    reshape_owned,
    slice,
)


comptime TArc = ArcPointer[Tensor]


@fieldwise_init
struct MiniMaxH3TrainingBlockWeightsDevice(Copyable, Movable):
    var norm1: TArc       # [D]
    var qkv: TArc         # [3*H*Dh,D], module [all-q;all-k;all-v]
    var q_norm: TArc      # [Dh]
    var k_norm: TArc      # [Dh]
    var out_proj: TArc    # [D,H*Dh]
    var norm2: TArc       # [D]
    var fc1: TArc         # [2*F,D], raw [gate;value]
    var fc2: TArc         # [D,F]


@fieldwise_init
struct MiniMaxH3BlockModulationDevice(Copyable, Movable):
    # Already gathered per packed-sequence row.
    var shift_msa: TArc   # [S,D]
    var scale_msa: TArc   # [S,D]
    var gate_msa: TArc    # [S,D]
    var shift_mlp: TArc   # [S,D]
    var scale_mlp: TArc   # [S,D]
    var gate_mlp: TArc    # [S,D]


@fieldwise_init
struct MiniMaxH3LoraAdapterDevice(Copyable, Movable):
    var a: TArc           # [rank,in]
    var b: TArc           # [out,rank]
    var rank: Int
    var in_features: Int
    var out_features: Int
    var scale: Float32    # multiplier * alpha / rank


@fieldwise_init
struct MiniMaxH3TrainingBlockLoraDevice(Copyable, Movable):
    var qkv: MiniMaxH3LoraAdapterDevice
    var out_proj: MiniMaxH3LoraAdapterDevice
    var fc1: MiniMaxH3LoraAdapterDevice
    var fc2: MiniMaxH3LoraAdapterDevice


@fieldwise_init
struct MiniMaxH3LoraGradDevice(Copyable, Movable):
    var d_a: TArc
    var d_b: TArc


@fieldwise_init
struct MiniMaxH3TrainingBlockBackwardDevice(Copyable, Movable):
    var d_x: TArc
    var qkv: MiniMaxH3LoraGradDevice
    var out_proj: MiniMaxH3LoraGradDevice
    var fc1: MiniMaxH3LoraGradDevice
    var fc2: MiniMaxH3LoraGradDevice


@fieldwise_init
struct _MiniMaxH3ProjectionBackwardDevice(Copyable, Movable):
    var d_x: TArc
    var d_a: TArc
    var d_b: TArc


def _require_f32(label: String, t: Tensor) raises:
    if t.dtype() != STDtype.F32:
        raise Error(label + ": reduced MiniMax-H3 device slice requires F32")


def minimax_h3_training_swap_fc1_rows_device(
    runtime_or_raw: Tensor,
    ffn: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    """Swap `[value;gate] <-> [gate;value]` rows at the explicit checkpoint seam.

    The transform is an involution and applies to either the frozen FC1 matrix
    `[2F,D]` or FC1 LoRA B/up `[2F,R]`.  FC1 LoRA A/down is `[R,D]` and must
    never be transformed.  QKV's current inference runtime order already
    matches this training module's `[all-q;all-k;all-v]` order.
    """
    var sh = runtime_or_raw.shape()
    if len(sh) != 2 or sh[0] != 2 * ffn:
        raise Error("MiniMax-H3 FC1 boundary swap expects [2F,K]")
    var first = slice(runtime_or_raw, 0, 0, ffn, ctx)
    var second = slice(runtime_or_raw, 0, ffn, ffn, ctx)
    return concat(0, ctx, second, first)


def _validate_adapter(
    label: String,
    lo: MiniMaxH3LoraAdapterDevice,
    in_features: Int,
    out_features: Int,
) raises:
    if lo.rank != 2 or lo.in_features != in_features or lo.out_features != out_features:
        raise Error(label + ": MiniMax-H3 LoRA geometry mismatch")
    if not isfinite(lo.scale) or lo.scale <= 0.0:
        raise Error(label + ": MiniMax-H3 LoRA scale must be finite and positive")
    if lo.a[].shape() != [lo.rank, in_features] or lo.b[].shape() != [out_features, lo.rank]:
        raise Error(label + ": MiniMax-H3 LoRA tensor shape mismatch")
    _require_f32(label + ".A", lo.a[])
    _require_f32(label + ".B", lo.b[])


def _validate[
    S: Int, H: Int, Dh: Int, D: Int, F: Int, Rot: Int
](
    x: Tensor,
    weights: MiniMaxH3TrainingBlockWeightsDevice,
    mod: MiniMaxH3BlockModulationDevice,
    lora: MiniMaxH3TrainingBlockLoraDevice,
    cos: Tensor,
    sin: Tensor,
) raises:
    comptime assert S == 3 and H == 56 and Dh == 4 and D == 8 and F == 12
    comptime assert Rot == 2
    comptime I = H * Dh
    if x.shape() != [S, D]:
        raise Error("MiniMax-H3 device block x must be [S,D]")
    if weights.norm1[].shape() != [D] or weights.norm2[].shape() != [D]:
        raise Error("MiniMax-H3 device block norm shape mismatch")
    if weights.qkv[].shape() != [3 * I, D] or weights.out_proj[].shape() != [D, I]:
        raise Error("MiniMax-H3 device block attention weight shape mismatch")
    if weights.q_norm[].shape() != [Dh] or weights.k_norm[].shape() != [Dh]:
        raise Error("MiniMax-H3 device block q/k norm shape mismatch")
    if weights.fc1[].shape() != [2 * F, D] or weights.fc2[].shape() != [D, F]:
        raise Error("MiniMax-H3 device block MLP weight shape mismatch")
    if cos.shape() != [S, Rot] or sin.shape() != [S, Rot]:
        raise Error("MiniMax-H3 device block compact RoPE shape mismatch")
    for t in [
        mod.shift_msa, mod.scale_msa, mod.gate_msa,
        mod.shift_mlp, mod.scale_mlp, mod.gate_mlp,
    ]:
        if t[].shape() != [S, D]:
            raise Error("MiniMax-H3 device block per-row modulation shape mismatch")
    _require_f32("x", x)
    for t in [
        weights.norm1, weights.qkv, weights.q_norm, weights.k_norm,
        weights.out_proj, weights.norm2, weights.fc1, weights.fc2,
        mod.shift_msa, mod.scale_msa, mod.gate_msa,
        mod.shift_mlp, mod.scale_mlp, mod.gate_mlp,
    ]:
        _require_f32("MiniMax-H3 device tensor", t[])
    _require_f32("cos", cos)
    _require_f32("sin", sin)
    _validate_adapter("qkv", lora.qkv, D, 3 * I)
    _validate_adapter("out_proj", lora.out_proj, I, D)
    _validate_adapter("fc1", lora.fc1, D, 2 * F)
    _validate_adapter("fc2", lora.fc2, F, D)


def _lora_projection_forward(
    x: Tensor,
    base_weight: Tensor,
    lo: MiniMaxH3LoraAdapterDevice,
    ctx: DeviceContext,
) raises -> Tensor:
    var no_bias = Optional[Tensor](None)
    var base = linear(x, base_weight, no_bias^, ctx)
    var no_bias_a = Optional[Tensor](None)
    var t = linear(x, lo.a[], no_bias_a^, ctx)
    var no_bias_b = Optional[Tensor](None)
    var delta = linear(t, lo.b[], no_bias_b^, ctx)
    var scaled = mul_scalar(delta, lo.scale, ctx)
    return add(base, scaled, ctx)


def _lora_projection_backward(
    d_y: Tensor,
    x: Tensor,
    base_weight: Tensor,
    lo: MiniMaxH3LoraAdapterDevice,
    rows: Int,
    ctx: DeviceContext,
) raises -> _MiniMaxH3ProjectionBackwardDevice:
    var base_dx = linear_backward_dx(
        d_y, base_weight, rows, lo.in_features, lo.out_features, ctx
    )
    var no_bias = Optional[Tensor](None)
    var t = linear(x, lo.a[], no_bias^, ctx)
    var scaled_dy = mul_scalar(d_y, lo.scale, ctx)
    var d_t = linear_backward_dx(
        scaled_dy, lo.b[], rows, lo.rank, lo.out_features, ctx
    )
    var d_b = linear_backward_dw(
        scaled_dy, t, rows, lo.rank, lo.out_features, ctx, STDtype.F32
    )
    var d_x_lora = linear_backward_dx(
        d_t, lo.a[], rows, lo.in_features, lo.rank, ctx
    )
    var d_a = linear_backward_dw(
        d_t, x, rows, lo.in_features, lo.rank, ctx, STDtype.F32
    )
    var d_x = add(base_dx, d_x_lora, ctx)
    return _MiniMaxH3ProjectionBackwardDevice(
        TArc(d_x^), TArc(d_a^), TArc(d_b^)
    )


def _partial_rope_forward[
    H: Int, Rot: Int
](x: Tensor, cos: Tensor, sin: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dh = x.shape()[3]
    var xr = slice(x, 3, 0, Rot, ctx)
    var xp = slice(x, 3, Rot, dh - Rot, ctx)
    var yr = rope_halfsplit_full_head_broadcast(xr, cos, sin, H, ctx)
    return concat(3, ctx, yr, xp)


def _expand_rope[
    S: Int, H: Int, Rot: Int
](tbl: Tensor, ctx: DeviceContext) raises -> Tensor:
    var t3 = reshape(tbl, [S, 1, Rot], ctx)
    var zeros = full_device([S, H, Rot], Float32(0.0), tbl.dtype(), ctx)
    var expanded = add(t3, zeros, ctx)
    return reshape_owned(expanded^, [S * H, Rot])


def _partial_rope_backward[
    S: Int, H: Int, Dh: Int, Rot: Int
](d_y: Tensor, cos: Tensor, sin: Tensor, ctx: DeviceContext) raises -> Tensor:
    var d_rot = slice(d_y, 3, 0, Rot, ctx)
    var d_pass = slice(d_y, 3, Rot, Dh - Rot, ctx)
    var cos_h = _expand_rope[S, H, Rot](cos, ctx)
    var sin_h = _expand_rope[S, H, Rot](sin, ctx)
    var d_rot_x = rope_halfsplit_full_backward(d_rot, cos_h, sin_h, ctx)
    return concat(3, ctx, d_rot_x, d_pass)


def minimax_h3_training_block_forward_device[
    S: Int, H: Int, Dh: Int, D: Int, F: Int, Rot: Int
](
    x: Tensor,
    weights: MiniMaxH3TrainingBlockWeightsDevice,
    mod: MiniMaxH3BlockModulationDevice,
    lora: MiniMaxH3TrainingBlockLoraDevice,
    cos: Tensor,
    sin: Tensor,
    norm_eps: Float32,
    qk_eps: Float32,
    ctx: DeviceContext,
) raises -> Tensor:
    _validate[S, H, Dh, D, F, Rot](x, weights, mod, lora, cos, sin)
    if not isfinite(norm_eps) or norm_eps <= 0.0 \
            or not isfinite(qk_eps) or qk_eps <= 0.0:
        raise Error("MiniMax-H3 parity reference eps values must be finite and positive")
    comptime I = H * Dh
    var n1 = rms_norm(x, weights.norm1[], norm_eps, ctx)
    var a1 = modulate(n1, mod.scale_msa[], mod.shift_msa[], ctx)
    var qkv = _lora_projection_forward(a1, weights.qkv[], lora.qkv, ctx)
    var q = reshape_owned(slice(qkv, 1, 0, I, ctx), [1, S, H, Dh])
    var k = reshape_owned(slice(qkv, 1, I, I, ctx), [1, S, H, Dh])
    var v = reshape_owned(slice(qkv, 1, 2 * I, I, ctx), [1, S, H, Dh])
    var qn = rms_norm(q, weights.q_norm[], qk_eps, ctx)
    var kn = rms_norm(k, weights.k_norm[], qk_eps, ctx)
    var qr = _partial_rope_forward[H, Rot](qn, cos, sin, ctx)
    var kr = _partial_rope_forward[H, Rot](kn, cos, sin, ctx)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var att = sdpa_nomask[1, S, H, Dh](qr, kr, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, I])
    var ao = _lora_projection_forward(att_flat, weights.out_proj[], lora.out_proj, ctx)
    var x1 = residual_gate(x, mod.gate_msa[], ao, ctx)
    var n2 = rms_norm(x1, weights.norm2[], norm_eps, ctx)
    var a2 = modulate(n2, mod.scale_mlp[], mod.shift_mlp[], ctx)
    var fc1 = _lora_projection_forward(a2, weights.fc1[], lora.fc1, ctx)
    var gate = slice(fc1, 1, 0, F, ctx)
    var value = slice(fc1, 1, F, F, ctx)
    var act = swiglu(gate, value, ctx)
    var fc2 = _lora_projection_forward(act, weights.fc2[], lora.fc2, ctx)
    return residual_gate(x1, mod.gate_mlp[], fc2, ctx)


def minimax_h3_training_block_backward_device[
    S: Int, H: Int, Dh: Int, D: Int, F: Int, Rot: Int
](
    d_y: Tensor,
    x: Tensor,
    weights: MiniMaxH3TrainingBlockWeightsDevice,
    mod: MiniMaxH3BlockModulationDevice,
    lora: MiniMaxH3TrainingBlockLoraDevice,
    cos: Tensor,
    sin: Tensor,
    norm_eps: Float32,
    qk_eps: Float32,
    ctx: DeviceContext,
) raises -> MiniMaxH3TrainingBlockBackwardDevice:
    _validate[S, H, Dh, D, F, Rot](x, weights, mod, lora, cos, sin)
    if not isfinite(norm_eps) or norm_eps <= 0.0 \
            or not isfinite(qk_eps) or qk_eps <= 0.0:
        raise Error("MiniMax-H3 parity reference eps values must be finite and positive")
    _require_f32("d_y", d_y)
    if d_y.shape() != [S, D]:
        raise Error("MiniMax-H3 device block d_y must be [S,D]")
    comptime I = H * Dh

    # Per-block recompute: keep only the block input at the future stack seam.
    var n1 = rms_norm(x, weights.norm1[], norm_eps, ctx)
    var a1 = modulate(n1, mod.scale_msa[], mod.shift_msa[], ctx)
    var qkv = _lora_projection_forward(a1, weights.qkv[], lora.qkv, ctx)
    var q = reshape_owned(slice(qkv, 1, 0, I, ctx), [1, S, H, Dh])
    var k = reshape_owned(slice(qkv, 1, I, I, ctx), [1, S, H, Dh])
    var v = reshape_owned(slice(qkv, 1, 2 * I, I, ctx), [1, S, H, Dh])
    var qn = rms_norm(q, weights.q_norm[], qk_eps, ctx)
    var kn = rms_norm(k, weights.k_norm[], qk_eps, ctx)
    var qr = _partial_rope_forward[H, Rot](qn, cos, sin, ctx)
    var kr = _partial_rope_forward[H, Rot](kn, cos, sin, ctx)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var att = sdpa_nomask[1, S, H, Dh](qr, kr, v, scale, ctx)
    var att_flat = reshape_owned(att^, [S, I])
    var ao = _lora_projection_forward(att_flat, weights.out_proj[], lora.out_proj, ctx)
    var x1 = residual_gate(x, mod.gate_msa[], ao, ctx)
    var n2 = rms_norm(x1, weights.norm2[], norm_eps, ctx)
    var a2 = modulate(n2, mod.scale_mlp[], mod.shift_mlp[], ctx)
    var fc1 = _lora_projection_forward(a2, weights.fc1[], lora.fc1, ctx)
    var gate = slice(fc1, 1, 0, F, ctx)
    var value = slice(fc1, 1, F, F, ctx)
    var act = swiglu(gate, value, ctx)

    var rg2 = gate_residual_backward_dxdy(d_y, mod.gate_mlp[], ctx)
    var b_fc2 = _lora_projection_backward(
        rg2.d_y, act, weights.fc2[], lora.fc2, S, ctx
    )
    var rg2_dx = rg2.d_x.clone(ctx)
    var b_swiglu = swiglu_backward(b_fc2.d_x[], gate, value, ctx)
    var d_fc1 = concat(1, ctx, b_swiglu.d_gate, b_swiglu.d_up)
    var b_fc1 = _lora_projection_backward(
        d_fc1, a2, weights.fc1[], lora.fc1, S, ctx
    )
    var b_mod2 = modulate_backward(
        b_fc1.d_x[], n2, mod.scale_mlp[], ctx, False
    )
    var d_x1_norm = rms_norm_backward_dx(
        b_mod2.d_x, x1, weights.norm2[], norm_eps, ctx
    )
    var d_x1 = add(rg2_dx, d_x1_norm, ctx)

    var rg1 = gate_residual_backward_dxdy(d_x1, mod.gate_msa[], ctx)
    var b_out = _lora_projection_backward(
        rg1.d_y, att_flat, weights.out_proj[], lora.out_proj, S, ctx
    )
    var rg1_dx = rg1.d_x.clone(ctx)
    var d_att = reshape(b_out.d_x[], [1, S, H, Dh], ctx)
    var b_att = sdpa_backward[1, S, H, Dh](qr, kr, v, d_att, scale, ctx)
    var d_q_att = b_att.d_q.clone(ctx)
    var d_k_att = b_att.d_k.clone(ctx)
    var d_v_att = b_att.d_v.clone(ctx)
    var d_qn = _partial_rope_backward[S, H, Dh, Rot](d_q_att, cos, sin, ctx)
    var d_kn = _partial_rope_backward[S, H, Dh, Rot](d_k_att, cos, sin, ctx)
    var d_q = rms_norm_backward_dx(d_qn, q, weights.q_norm[], qk_eps, ctx)
    var d_k = rms_norm_backward_dx(d_kn, k, weights.k_norm[], qk_eps, ctx)
    var d_q2 = reshape_owned(d_q^, [S, I])
    var d_k2 = reshape_owned(d_k^, [S, I])
    var d_v2 = reshape_owned(d_v_att^, [S, I])
    var d_qkv = concat(1, ctx, d_q2, d_k2, d_v2)
    var b_qkv = _lora_projection_backward(
        d_qkv, a1, weights.qkv[], lora.qkv, S, ctx
    )
    var b_mod1 = modulate_backward(
        b_qkv.d_x[], n1, mod.scale_msa[], ctx, False
    )
    var d_x_norm = rms_norm_backward_dx(
        b_mod1.d_x, x, weights.norm1[], norm_eps, ctx
    )
    var d_x = add(rg1_dx, d_x_norm, ctx)

    return MiniMaxH3TrainingBlockBackwardDevice(
        TArc(d_x^),
        MiniMaxH3LoraGradDevice(b_qkv.d_a, b_qkv.d_b),
        MiniMaxH3LoraGradDevice(b_out.d_a, b_out.d_b),
        MiniMaxH3LoraGradDevice(b_fc1.d_a, b_fc1.d_b),
        MiniMaxH3LoraGradDevice(b_fc2.d_a, b_fc2.d_b),
    )
