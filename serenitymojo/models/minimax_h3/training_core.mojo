# MiniMax-H3 BF16-compute/F32-LoRA training core.
#
# Pure Mojo/MAX device math. Frozen base weights are BF16; LoRA A/B leaves and
# returned gradients are F32, with explicit BF16 projection views matching the
# pinned Musubi autocast boundary. Attention is noncausal shared cuDNN flash.
#
# INTEGRATION LAYOUT: this product core consumes the existing Serenity runtime
# convention: module QKV `[all-q;all-k;all-v]` and FC1 `[value;gate]`. The
# checkpoint loader owns raw QKV deinterleave and raw `[gate;value]` FC1 swaps.
# LoRA FC1 B/up must cross the same runtime boundary before entering this core.
#
# Scope: LoRA backward only. Frozen-base gradients, full finetune, and INT8
# backward are deliberately unsupported and fail before device math.

from max.gpu.host import DeviceContext
from std.collections import List
from std.math import isfinite, sqrt
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.linalg_backward import linear_backward_dx, linear_backward_dw
from serenitymojo.ops.norm import rms_norm
from serenitymojo.ops.norm_backward import rms_norm_backward_dx
from serenitymojo.ops.elementwise import modulate, residual_gate
from serenitymojo.ops.elementwise_backward import modulate_backward
from serenitymojo.ops.activations import swiglu
from serenitymojo.ops.loss_swiglu_backward import swiglu_backward
from serenitymojo.ops.attention_flash import sdpa_flash_train_fwd, sdpa_flash_backward
from serenitymojo.ops.rope import rope_halfsplit_full_head_broadcast
from serenitymojo.ops.rope_struct_backward import (
    gate_residual_backward_dxdy,
    rope_halfsplit_full_backward,
)
from serenitymojo.ops.tensor_algebra import (
    add,
    concat,
    full_device,
    gather_rows,
    mul_scalar,
    reshape,
    reshape_owned,
    slice,
)


comptime TArc = ArcPointer[Tensor]
comptime MINIMAX_H3_TRAIN_QKV_RUNTIME_ALL_Q_K_V = 2051
comptime MINIMAX_H3_TRAIN_FC1_RUNTIME_VALUE_GATE = 2052
comptime MINIMAX_H3_TRAIN_FC1_LORA_B_RUNTIME_VALUE_GATE = 2053


@fieldwise_init
struct MiniMaxH3TrainingWeightsDevice(Copyable, Movable):
    var norm1: TArc
    var qkv: TArc
    var q_norm: TArc
    var k_norm: TArc
    var out_proj: TArc
    var norm2: TArc
    var fc1: TArc
    var fc2: TArc


@fieldwise_init
struct MiniMaxH3TrainingModulationDevice(Copyable, Movable):
    var shift_msa: TArc
    var scale_msa: TArc
    var gate_msa: TArc
    var shift_mlp: TArc
    var scale_mlp: TArc
    var gate_mlp: TArc


@fieldwise_init
struct MiniMaxH3TrainingLoraAdapterDevice(Copyable, Movable):
    var a: TArc
    var b: TArc
    var rank: Int
    var in_features: Int
    var out_features: Int
    var scale: Float32


@fieldwise_init
struct MiniMaxH3TrainingBlockLoraDevice(Copyable, Movable):
    var qkv: MiniMaxH3TrainingLoraAdapterDevice
    var out_proj: MiniMaxH3TrainingLoraAdapterDevice
    var fc1: MiniMaxH3TrainingLoraAdapterDevice
    var fc2: MiniMaxH3TrainingLoraAdapterDevice


@fieldwise_init
struct MiniMaxH3TrainingLoraGradDevice(Copyable, Movable):
    var d_a: TArc
    var d_b: TArc


@fieldwise_init
struct MiniMaxH3TrainingCoreBackward(Copyable, Movable):
    var d_x: TArc
    var qkv: MiniMaxH3TrainingLoraGradDevice
    var out_proj: MiniMaxH3TrainingLoraGradDevice
    var fc1: MiniMaxH3TrainingLoraGradDevice
    var fc2: MiniMaxH3TrainingLoraGradDevice


@fieldwise_init
struct _MiniMaxH3ProjectionBackward(Copyable, Movable):
    var d_x: TArc
    var d_a: TArc
    var d_b: TArc


def minimax_h3_training_modulation_from_table[
    S: Int, D: Int
](
    table: Tensor,
    adaln_indices: List[Int],
    ctx: DeviceContext,
) raises -> MiniMaxH3TrainingModulationDevice:
    """Gather a frozen AdaLN cache `[rows,6D]` into six `[S,D]` tensors.

    `table` is exactly one `MiniMaxH3ModCache.block_mod[layer]`; indices are
    the packed-sequence `(timestep * 3 + modality_tag)` rows already produced
    by the shared H3 frontend.
    """
    var shape = table.shape()
    if len(shape) != 2 or shape[1] != 6 * D or len(adaln_indices) != S:
        raise Error("MiniMax-H3 training AdaLN table/index geometry mismatch")
    if table.dtype() != STDtype.BF16:
        raise Error("MiniMax-H3 training AdaLN modulation table must be BF16")
    var gathered = gather_rows(table, adaln_indices, ctx)
    return MiniMaxH3TrainingModulationDevice(
        TArc(slice(gathered, 1, 0 * D, D, ctx)),
        TArc(slice(gathered, 1, 1 * D, D, ctx)),
        TArc(slice(gathered, 1, 2 * D, D, ctx)),
        TArc(slice(gathered, 1, 3 * D, D, ctx)),
        TArc(slice(gathered, 1, 4 * D, D, ctx)),
        TArc(slice(gathered, 1, 5 * D, D, ctx)),
    )


def _require_bf16(label: String, tensor: Tensor) raises:
    if tensor.dtype() != STDtype.BF16:
        raise Error(label + String(": MiniMax-H3 training requires BF16"))


def _validate_adapter(
    label: String,
    adapter: MiniMaxH3TrainingLoraAdapterDevice,
    inf: Int,
    outf: Int,
) raises:
    if adapter.rank <= 0 or adapter.in_features != inf or adapter.out_features != outf:
        raise Error(label + String(": MiniMax-H3 LoRA geometry mismatch"))
    if adapter.a[].shape() != [adapter.rank, inf] or adapter.b[].shape() != [outf, adapter.rank]:
        raise Error(label + String(": MiniMax-H3 LoRA tensor shape mismatch"))
    if adapter.a[].dtype() != STDtype.F32 or adapter.b[].dtype() != STDtype.F32:
        raise Error(label + String(": MiniMax-H3 LoRA leaves must be F32"))
    if not isfinite(adapter.scale):
        raise Error(label + String(": MiniMax-H3 LoRA scale must be finite"))


def _validate[
    S: Int, H: Int, Dh: Int, D: Int, F: Int, Rot: Int
](
    x: Tensor,
    weights: MiniMaxH3TrainingWeightsDevice,
    modulation: MiniMaxH3TrainingModulationDevice,
    lora: MiniMaxH3TrainingBlockLoraDevice,
    cos: Tensor,
    sin: Tensor,
    norm_eps: Float32,
    qk_eps: Float32,
    qkv_layout_receipt: Int,
    fc1_layout_receipt: Int,
    fc1_lora_b_layout_receipt: Int,
) raises:
    # The reduced profile exists only for product-smoke coverage. The only
    # released geometry admitted by this API is H3 Base's exact block shape.
    comptime assert (
        (S == 3 and H == 56 and Dh == 8 and D == 8 and F == 12 and Rot == 4)
        or (S > 0 and H == 56 and Dh == 128 and D == 5376 and F == 14336 and Rot == 96)
    )
    if norm_eps != Float32(1.0e-5) or qk_eps != Float32(1.0e-5):
        raise Error("MiniMax-H3 training requires norm_eps=qk_eps=1e-5")
    if qkv_layout_receipt != MINIMAX_H3_TRAIN_QKV_RUNTIME_ALL_Q_K_V:
        raise Error("MiniMax-H3 training QKV must be runtime [all-q;all-k;all-v]")
    if fc1_layout_receipt != MINIMAX_H3_TRAIN_FC1_RUNTIME_VALUE_GATE:
        raise Error("MiniMax-H3 training FC1 base must be runtime [value;gate]")
    if fc1_lora_b_layout_receipt != MINIMAX_H3_TRAIN_FC1_LORA_B_RUNTIME_VALUE_GATE:
        raise Error("MiniMax-H3 training FC1 LoRA B must be runtime [value;gate]")
    comptime I = H * Dh
    if x.shape() != [S, D] or cos.shape() != [S, Rot] or sin.shape() != [S, Rot]:
        raise Error("MiniMax-H3 training x/RoPE shape mismatch")
    if weights.norm1[].shape() != [D] or weights.norm2[].shape() != [D]:
        raise Error("MiniMax-H3 training norm shape mismatch")
    if weights.qkv[].shape() != [3 * I, D] or weights.out_proj[].shape() != [D, I]:
        raise Error("MiniMax-H3 training attention weight shape mismatch")
    if weights.q_norm[].shape() != [Dh] or weights.k_norm[].shape() != [Dh]:
        raise Error("MiniMax-H3 training q/k norm shape mismatch")
    if weights.fc1[].shape() != [2 * F, D] or weights.fc2[].shape() != [D, F]:
        raise Error("MiniMax-H3 training MLP weight shape mismatch")
    for t in [
        weights.norm1, weights.qkv, weights.q_norm, weights.k_norm,
        weights.out_proj, weights.norm2, weights.fc1, weights.fc2,
        modulation.shift_msa, modulation.scale_msa, modulation.gate_msa,
        modulation.shift_mlp, modulation.scale_mlp, modulation.gate_mlp,
    ]:
        _require_bf16("MiniMax-H3 training tensor", t[])
    for t in [
        modulation.shift_msa, modulation.scale_msa, modulation.gate_msa,
        modulation.shift_mlp, modulation.scale_mlp, modulation.gate_mlp,
    ]:
        if t[].shape() != [S, D]:
            raise Error("MiniMax-H3 training per-row modulation shape mismatch")
    _require_bf16("x", x)
    _require_bf16("cos", cos)
    _require_bf16("sin", sin)
    _validate_adapter("qkv", lora.qkv, D, 3 * I)
    _validate_adapter("out_proj", lora.out_proj, I, D)
    _validate_adapter("fc1", lora.fc1, D, 2 * F)
    _validate_adapter("fc2", lora.fc2, F, D)


def _projection_forward(
    x: Tensor,
    weight: Tensor,
    adapter: MiniMaxH3TrainingLoraAdapterDevice,
    ctx: DeviceContext,
) raises -> Tensor:
    var base = linear(x, weight, None, ctx)
    var a_compute = cast_tensor(adapter.a[], STDtype.BF16, ctx)
    var b_compute = cast_tensor(adapter.b[], STDtype.BF16, ctx)
    var hidden = linear(x, a_compute, None, ctx)
    var delta = linear(hidden, b_compute, None, ctx)
    return add(base, mul_scalar(delta, adapter.scale, ctx), ctx)


def _projection_backward(
    d_y: Tensor,
    x: Tensor,
    weight: Tensor,
    adapter: MiniMaxH3TrainingLoraAdapterDevice,
    rows: Int,
    ctx: DeviceContext,
) raises -> _MiniMaxH3ProjectionBackward:
    var base_dx = linear_backward_dx(
        d_y, weight, rows, adapter.in_features, adapter.out_features, ctx
    )
    var a_compute = cast_tensor(adapter.a[], STDtype.BF16, ctx)
    var b_compute = cast_tensor(adapter.b[], STDtype.BF16, ctx)
    var hidden = linear(x, a_compute, None, ctx)
    var scaled_dy = mul_scalar(d_y, adapter.scale, ctx)
    var d_hidden = linear_backward_dx(
        scaled_dy, b_compute, rows, adapter.rank, adapter.out_features, ctx
    )
    var d_b_compute = linear_backward_dw(
        scaled_dy, hidden, rows, adapter.rank, adapter.out_features,
        ctx, STDtype.BF16,
    )
    var d_x_lora = linear_backward_dx(
        d_hidden, a_compute, rows, adapter.in_features, adapter.rank, ctx
    )
    var d_a_compute = linear_backward_dw(
        d_hidden, x, rows, adapter.in_features, adapter.rank,
        ctx, STDtype.BF16,
    )
    return _MiniMaxH3ProjectionBackward(
        TArc(add(base_dx, d_x_lora, ctx)),
        TArc(cast_tensor(d_a_compute, STDtype.F32, ctx)),
        TArc(cast_tensor(d_b_compute, STDtype.F32, ctx)),
    )


def _partial_rope_forward[
    H: Int, Rot: Int
](x: Tensor, cos: Tensor, sin: Tensor, ctx: DeviceContext) raises -> Tensor:
    var dh = x.shape()[3]
    var rotated = slice(x, 3, 0, Rot, ctx)
    var passthrough = slice(x, 3, Rot, dh - Rot, ctx)
    var out = rope_halfsplit_full_head_broadcast(rotated, cos, sin, H, ctx)
    return concat(3, ctx, out, passthrough)


def _expand_rope[
    S: Int, H: Int, Rot: Int
](table: Tensor, ctx: DeviceContext) raises -> Tensor:
    var table3 = reshape(table, [S, 1, Rot], ctx)
    var zeros = full_device([S, H, Rot], Float32(0.0), STDtype.BF16, ctx)
    var expanded = add(table3, zeros, ctx)
    return reshape_owned(expanded^, [S * H, Rot])


def _partial_rope_backward[
    S: Int, H: Int, Dh: Int, Rot: Int
](d_y: Tensor, cos: Tensor, sin: Tensor, ctx: DeviceContext) raises -> Tensor:
    var d_rotated = slice(d_y, 3, 0, Rot, ctx)
    var d_passthrough = slice(d_y, 3, Rot, Dh - Rot, ctx)
    var cos_h = _expand_rope[S, H, Rot](cos, ctx)
    var sin_h = _expand_rope[S, H, Rot](sin, ctx)
    var d_rotated_x = rope_halfsplit_full_backward(d_rotated, cos_h, sin_h, ctx)
    return concat(3, ctx, d_rotated_x, d_passthrough)


def minimax_h3_training_core_forward[
    S: Int, H: Int, Dh: Int, D: Int, F: Int, Rot: Int
](
    x: Tensor,
    weights: MiniMaxH3TrainingWeightsDevice,
    modulation: MiniMaxH3TrainingModulationDevice,
    lora: MiniMaxH3TrainingBlockLoraDevice,
    cos: Tensor,
    sin: Tensor,
    norm_eps: Float32,
    qk_eps: Float32,
    qkv_layout_receipt: Int,
    fc1_layout_receipt: Int,
    fc1_lora_b_layout_receipt: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    _validate[S, H, Dh, D, F, Rot](
        x, weights, modulation, lora, cos, sin, norm_eps, qk_eps,
        qkv_layout_receipt, fc1_layout_receipt, fc1_lora_b_layout_receipt,
    )
    comptime I = H * Dh
    var n1 = rms_norm(x, weights.norm1[], norm_eps, ctx)
    var a1 = modulate(n1, modulation.scale_msa[], modulation.shift_msa[], ctx)
    var qkv = _projection_forward(a1, weights.qkv[], lora.qkv, ctx)
    var q = reshape_owned(slice(qkv, 1, 0, I, ctx), [1, S, H, Dh])
    var k = reshape_owned(slice(qkv, 1, I, I, ctx), [1, S, H, Dh])
    var v = reshape_owned(slice(qkv, 1, 2 * I, I, ctx), [1, S, H, Dh])
    var qn = rms_norm(q, weights.q_norm[], qk_eps, ctx)
    var kn = rms_norm(k, weights.k_norm[], qk_eps, ctx)
    var qr = _partial_rope_forward[H, Rot](qn, cos, sin, ctx)
    var kr = _partial_rope_forward[H, Rot](kn, cos, sin, ctx)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var flash = sdpa_flash_train_fwd[1, S, H, Dh](qr, kr, v, scale, ctx)
    var attention = reshape(flash.o, [S, I], ctx)
    var projected = _projection_forward(attention, weights.out_proj[], lora.out_proj, ctx)
    var x1 = residual_gate(x, modulation.gate_msa[], projected, ctx)
    var n2 = rms_norm(x1, weights.norm2[], norm_eps, ctx)
    var a2 = modulate(n2, modulation.scale_mlp[], modulation.shift_mlp[], ctx)
    var fc1 = _projection_forward(a2, weights.fc1[], lora.fc1, ctx)
    # Runtime convention is [value;gate], unlike the raw Musubi fixture.
    var value = slice(fc1, 1, 0, F, ctx)
    var gate = slice(fc1, 1, F, F, ctx)
    var activation = swiglu(gate, value, ctx)
    var fc2 = _projection_forward(activation, weights.fc2[], lora.fc2, ctx)
    return residual_gate(x1, modulation.gate_mlp[], fc2, ctx)


def minimax_h3_training_core_backward[
    S: Int, H: Int, Dh: Int, D: Int, F: Int, Rot: Int
](
    d_y: Tensor,
    x: Tensor,
    weights: MiniMaxH3TrainingWeightsDevice,
    modulation: MiniMaxH3TrainingModulationDevice,
    lora: MiniMaxH3TrainingBlockLoraDevice,
    cos: Tensor,
    sin: Tensor,
    norm_eps: Float32,
    qk_eps: Float32,
    qkv_layout_receipt: Int,
    fc1_layout_receipt: Int,
    fc1_lora_b_layout_receipt: Int,
    ctx: DeviceContext,
) raises -> MiniMaxH3TrainingCoreBackward:
    _validate[S, H, Dh, D, F, Rot](
        x, weights, modulation, lora, cos, sin, norm_eps, qk_eps,
        qkv_layout_receipt, fc1_layout_receipt, fc1_lora_b_layout_receipt,
    )
    _require_bf16("d_y", d_y)
    if d_y.shape() != [S, D]:
        raise Error("MiniMax-H3 training d_y shape mismatch")
    comptime I = H * Dh

    # Per-core recompute: the 50-block stack retains only residual inputs.
    var n1 = rms_norm(x, weights.norm1[], norm_eps, ctx)
    var a1 = modulate(n1, modulation.scale_msa[], modulation.shift_msa[], ctx)
    var qkv = _projection_forward(a1, weights.qkv[], lora.qkv, ctx)
    var q = reshape_owned(slice(qkv, 1, 0, I, ctx), [1, S, H, Dh])
    var k = reshape_owned(slice(qkv, 1, I, I, ctx), [1, S, H, Dh])
    var v = reshape_owned(slice(qkv, 1, 2 * I, I, ctx), [1, S, H, Dh])
    var qn = rms_norm(q, weights.q_norm[], qk_eps, ctx)
    var kn = rms_norm(k, weights.k_norm[], qk_eps, ctx)
    var qr = _partial_rope_forward[H, Rot](qn, cos, sin, ctx)
    var kr = _partial_rope_forward[H, Rot](kn, cos, sin, ctx)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var flash = sdpa_flash_train_fwd[1, S, H, Dh](qr, kr, v, scale, ctx)
    var attention = reshape(flash.o, [S, I], ctx)
    var projected = _projection_forward(attention, weights.out_proj[], lora.out_proj, ctx)
    var x1 = residual_gate(x, modulation.gate_msa[], projected, ctx)
    var n2 = rms_norm(x1, weights.norm2[], norm_eps, ctx)
    var a2 = modulate(n2, modulation.scale_mlp[], modulation.shift_mlp[], ctx)
    var fc1 = _projection_forward(a2, weights.fc1[], lora.fc1, ctx)
    var value = slice(fc1, 1, 0, F, ctx)
    var gate = slice(fc1, 1, F, F, ctx)
    var activation = swiglu(gate, value, ctx)

    var residual2 = gate_residual_backward_dxdy(d_y, modulation.gate_mlp[], ctx)
    var residual2_dx = residual2.d_x.clone(ctx)
    var b_fc2 = _projection_backward(
        residual2.d_y, activation, weights.fc2[], lora.fc2, S, ctx
    )
    var b_swiglu = swiglu_backward(b_fc2.d_x[], gate, value, ctx)
    # Runtime FC1 is [value;gate]. swiglu backward names value `d_up`.
    var d_fc1 = concat(1, ctx, b_swiglu.d_up, b_swiglu.d_gate)
    var b_fc1 = _projection_backward(d_fc1, a2, weights.fc1[], lora.fc1, S, ctx)
    var b_mod2 = modulate_backward(
        b_fc1.d_x[], n2, modulation.scale_mlp[], ctx, False
    )
    var d_x1_norm = rms_norm_backward_dx(
        b_mod2.d_x, x1, weights.norm2[], norm_eps, ctx
    )
    var d_x1 = add(residual2_dx, d_x1_norm, ctx)

    var residual1 = gate_residual_backward_dxdy(d_x1, modulation.gate_msa[], ctx)
    var residual1_dx = residual1.d_x.clone(ctx)
    var b_out = _projection_backward(
        residual1.d_y, attention, weights.out_proj[], lora.out_proj, S, ctx
    )
    var d_attention = reshape(b_out.d_x[], [1, S, H, Dh], ctx)
    var b_attention = sdpa_flash_backward[1, S, H, Dh](flash, d_attention, scale, ctx)
    # Clone the three owned results before consuming them so the aggregate
    # flash-backward value retains a valid destruction path.
    var d_q_attention = b_attention.d_q.clone(ctx)
    var d_k_attention = b_attention.d_k.clone(ctx)
    var d_v_attention = b_attention.d_v.clone(ctx)
    var d_qn = _partial_rope_backward[S, H, Dh, Rot](d_q_attention, cos, sin, ctx)
    var d_kn = _partial_rope_backward[S, H, Dh, Rot](d_k_attention, cos, sin, ctx)
    var d_q = rms_norm_backward_dx(d_qn, q, weights.q_norm[], qk_eps, ctx)
    var d_k = rms_norm_backward_dx(d_kn, k, weights.k_norm[], qk_eps, ctx)
    var d_q2 = reshape_owned(d_q^, [S, I])
    var d_k2 = reshape_owned(d_k^, [S, I])
    var d_v2 = reshape_owned(d_v_attention^, [S, I])
    var d_qkv = concat(1, ctx, d_q2, d_k2, d_v2)
    var b_qkv = _projection_backward(d_qkv, a1, weights.qkv[], lora.qkv, S, ctx)
    var b_mod1 = modulate_backward(
        b_qkv.d_x[], n1, modulation.scale_msa[], ctx, False
    )
    var d_x_norm = rms_norm_backward_dx(
        b_mod1.d_x, x, weights.norm1[], norm_eps, ctx
    )
    var d_x = add(residual1_dx, d_x_norm, ctx)

    return MiniMaxH3TrainingCoreBackward(
        TArc(d_x^),
        MiniMaxH3TrainingLoraGradDevice(b_qkv.d_a, b_qkv.d_b),
        MiniMaxH3TrainingLoraGradDevice(b_out.d_a, b_out.d_b),
        MiniMaxH3TrainingLoraGradDevice(b_fc1.d_a, b_fc1.d_b),
        MiniMaxH3TrainingLoraGradDevice(b_fc2.d_a, b_fc2.d_b),
    )
