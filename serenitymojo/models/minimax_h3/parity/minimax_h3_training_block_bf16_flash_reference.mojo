# MiniMax-H3 mixed-dtype FLASH DiT block core -- PARITY-ONLY composition.
#
# Per-row AdaLN shifts/scales/gates are injected inputs: AdalnProj and segment
# gathering are outside this core. This is not evidence for a full DiTBlock.
# The base/input/modulation/compute path is BF16; pinned-Musubi LoRA A/B leaves
# and returned dA/dB are F32 while their projection kernels run BF16. All model
# math stays on device and attention is exclusively cuDNN flash fwd/bwd.

from max.gpu.host import DeviceContext
from std.collections import List, Optional
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
from serenitymojo.ops.attention_flash import (
    sdpa_flash_train_fwd,
    sdpa_flash_backward,
)
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
from serenitymojo.models.minimax_h3.parity.training_block_device_reference import (
    MiniMaxH3BlockModulationDevice,
    MiniMaxH3LoraAdapterDevice,
    MiniMaxH3TrainingBlockLoraDevice,
    MiniMaxH3TrainingBlockWeightsDevice,
)


comptime TArc = ArcPointer[Tensor]

# Callers must provide these directional receipts. They prevent the shape-equal
# Serenity inference FC1 `[value;gate]` layout from silently entering the raw
# Musubi training math. QKV runtime already equals module `[all-q;all-k;all-v]`;
# raw-checkpoint deinterleave is outside this core and is independently gated by
# `minimax_h3_loader_parity.mojo` section [3].
comptime MINIMAX_H3_BF16_QKV_ALL_Q_K_V = 1029
comptime MINIMAX_H3_BF16_FC1_GATE_VALUE = 1030
comptime MINIMAX_H3_BF16_FC1_LORA_B_GATE_VALUE = 1031


@fieldwise_init
struct MiniMaxH3BF16FlashCoreLoraGrad(Copyable, Movable):
    var d_a: TArc
    var d_b: TArc


@fieldwise_init
struct MiniMaxH3BF16FlashCoreBackward(Copyable, Movable):
    var d_x: TArc
    var qkv: MiniMaxH3BF16FlashCoreLoraGrad
    var out_proj: MiniMaxH3BF16FlashCoreLoraGrad
    var fc1: MiniMaxH3BF16FlashCoreLoraGrad
    var fc2: MiniMaxH3BF16FlashCoreLoraGrad


@fieldwise_init
struct _ProjectionBackward(Copyable, Movable):
    var d_x: TArc
    var d_a: TArc
    var d_b: TArc


def _require_bf16(label: String, t: Tensor) raises:
    if t.dtype() != STDtype.BF16:
        raise Error(label + ": BF16 flash parity slice requires BF16")


def _validate_adapter(
    label: String,
    lo: MiniMaxH3LoraAdapterDevice,
    inf: Int,
    outf: Int,
) raises:
    if lo.rank <= 0 or lo.in_features != inf or lo.out_features != outf:
        raise Error(label + ": BF16 H3 LoRA geometry mismatch")
    if lo.a[].shape() != [lo.rank, inf] or lo.b[].shape() != [outf, lo.rank]:
        raise Error(label + ": BF16 H3 LoRA tensor shape mismatch")
    if lo.rank != 2:
        raise Error(label + ": reduced BF16 H3 core requires exact rank 2")
    if not isfinite(lo.scale) or abs(lo.scale - Float32(0.7)) > 1.0e-7:
        raise Error(label + ": reduced BF16 H3 core requires exact LoRA scale 0.7")
    if lo.a[].dtype() != STDtype.F32 or lo.b[].dtype() != STDtype.F32:
        raise Error(label + ": pinned Musubi LoRA A/B leaves must be F32")


def _validate[
    S: Int, H: Int, Dh: Int, D: Int, F: Int, Rot: Int
](
    x: Tensor,
    weights: MiniMaxH3TrainingBlockWeightsDevice,
    mod: MiniMaxH3BlockModulationDevice,
    lora: MiniMaxH3TrainingBlockLoraDevice,
    cos: Tensor,
    sin: Tensor,
    qkv_layout_receipt: Int,
    fc1_layout_receipt: Int,
    fc1_lora_b_layout_receipt: Int,
    norm_eps: Float32,
    qk_eps: Float32,
) raises:
    # Closed evidence profiles only: the Torch-compared reduced oracle and the
    # released-geometry synthetic smoke. This parity-only module is not a
    # generic product surface.
    comptime assert (
        (S == 3 and H == 56 and Dh == 8 and D == 8 and F == 12 and Rot == 4)
        or (
            S == 2 and H == 56 and Dh == 128 and D == 5376
            and F == 14336 and Rot == 96
        )
    )
    if norm_eps != Float32(1.0e-5) or qk_eps != Float32(1.0e-5):
        raise Error("reduced BF16 H3 core requires norm_eps=qk_eps=1e-5")
    comptime I = H * Dh
    if qkv_layout_receipt != MINIMAX_H3_BF16_QKV_ALL_Q_K_V:
        raise Error("BF16 H3 QKV layout receipt must be [all-q;all-k;all-v]")
    if fc1_layout_receipt != MINIMAX_H3_BF16_FC1_GATE_VALUE:
        raise Error("BF16 H3 FC1 base layout receipt must be [gate;value]")
    if fc1_lora_b_layout_receipt != MINIMAX_H3_BF16_FC1_LORA_B_GATE_VALUE:
        raise Error("BF16 H3 FC1 LoRA B layout receipt must be [gate;value]")
    if x.shape() != [S, D] or cos.shape() != [S, Rot] or sin.shape() != [S, Rot]:
        raise Error("BF16 H3 x/RoPE shape mismatch")
    if weights.norm1[].shape() != [D] or weights.norm2[].shape() != [D]:
        raise Error("BF16 H3 norm shape mismatch")
    if weights.qkv[].shape() != [3 * I, D] or weights.out_proj[].shape() != [D, I]:
        raise Error("BF16 H3 attention weight shape mismatch")
    if weights.q_norm[].shape() != [Dh] or weights.k_norm[].shape() != [Dh]:
        raise Error("BF16 H3 q/k norm shape mismatch")
    if weights.fc1[].shape() != [2 * F, D] or weights.fc2[].shape() != [D, F]:
        raise Error("BF16 H3 MLP weight shape mismatch")
    for t in [
        mod.shift_msa, mod.scale_msa, mod.gate_msa,
        mod.shift_mlp, mod.scale_mlp, mod.gate_mlp,
    ]:
        if t[].shape() != [S, D]:
            raise Error("BF16 H3 per-row modulation shape mismatch")
    _require_bf16("x", x)
    _require_bf16("cos", cos)
    _require_bf16("sin", sin)
    for t in [
        weights.norm1, weights.qkv, weights.q_norm, weights.k_norm,
        weights.out_proj, weights.norm2, weights.fc1, weights.fc2,
        mod.shift_msa, mod.scale_msa, mod.gate_msa,
        mod.shift_mlp, mod.scale_mlp, mod.gate_mlp,
    ]:
        _require_bf16("BF16 H3 fixture tensor", t[])
    _validate_adapter("qkv", lora.qkv, D, 3 * I)
    _validate_adapter("out_proj", lora.out_proj, I, D)
    _validate_adapter("fc1", lora.fc1, D, 2 * F)
    _validate_adapter("fc2", lora.fc2, F, D)


def _projection_forward(
    x: Tensor,
    w: Tensor,
    lo: MiniMaxH3LoraAdapterDevice,
    ctx: DeviceContext,
) raises -> Tensor:
    var nb0 = Optional[Tensor](None)
    var base = linear(x, w, nb0^, ctx)
    # Pinned Musubi stores trainable leaves F32; CUDA BF16 autocast executes
    # their Linear kernels from BF16 views without changing parameter storage.
    var a_compute = cast_tensor(lo.a[], STDtype.BF16, ctx)
    var b_compute = cast_tensor(lo.b[], STDtype.BF16, ctx)
    var nb1 = Optional[Tensor](None)
    var t = linear(x, a_compute, nb1^, ctx)
    var nb2 = Optional[Tensor](None)
    var delta = linear(t, b_compute, nb2^, ctx)
    var scaled = mul_scalar(delta, lo.scale, ctx)
    return add(base, scaled, ctx)


def _projection_backward(
    d_y: Tensor,
    x: Tensor,
    w: Tensor,
    lo: MiniMaxH3LoraAdapterDevice,
    rows: Int,
    ctx: DeviceContext,
) raises -> _ProjectionBackward:
    var base_dx = linear_backward_dx(
        d_y, w, rows, lo.in_features, lo.out_features, ctx
    )
    var a_compute = cast_tensor(lo.a[], STDtype.BF16, ctx)
    var b_compute = cast_tensor(lo.b[], STDtype.BF16, ctx)
    var nb = Optional[Tensor](None)
    var t = linear(x, a_compute, nb^, ctx)
    var scaled_dy = mul_scalar(d_y, lo.scale, ctx)
    var d_t = linear_backward_dx(
        scaled_dy, b_compute, rows, lo.rank, lo.out_features, ctx
    )
    # Autocast computes these leaf gradients in BF16, then writes/casts them
    # into the F32 parameter-grad storage. The pinned Torch fixture confirms
    # every saved F32 dA/dB value is exactly BF16-representable.
    var d_b_compute = linear_backward_dw(
        scaled_dy, t, rows, lo.rank, lo.out_features, ctx, STDtype.BF16
    )
    var d_x_lo = linear_backward_dx(
        d_t, a_compute, rows, lo.in_features, lo.rank, ctx
    )
    var d_a_compute = linear_backward_dw(
        d_t, x, rows, lo.in_features, lo.rank, ctx, STDtype.BF16
    )
    var d_b = cast_tensor(d_b_compute, STDtype.F32, ctx)
    var d_a = cast_tensor(d_a_compute, STDtype.F32, ctx)
    var d_x = add(base_dx, d_x_lo, ctx)
    return _ProjectionBackward(TArc(d_x^), TArc(d_a^), TArc(d_b^))


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
    var zeros = full_device([S, H, Rot], Float32(0.0), STDtype.BF16, ctx)
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


def minimax_h3_bf16_flash_core_forward[
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
    qkv_layout_receipt: Int,
    fc1_layout_receipt: Int,
    fc1_lora_b_layout_receipt: Int,
    ctx: DeviceContext,
) raises -> Tensor:
    _validate[S, H, Dh, D, F, Rot](
        x, weights, mod, lora, cos, sin, qkv_layout_receipt,
        fc1_layout_receipt, fc1_lora_b_layout_receipt, norm_eps, qk_eps,
    )
    comptime I = H * Dh
    var n1 = rms_norm(x, weights.norm1[], norm_eps, ctx)
    var a1 = modulate(n1, mod.scale_msa[], mod.shift_msa[], ctx)
    var qkv = _projection_forward(a1, weights.qkv[], lora.qkv, ctx)
    var q = reshape_owned(slice(qkv, 1, 0, I, ctx), [1, S, H, Dh])
    var k = reshape_owned(slice(qkv, 1, I, I, ctx), [1, S, H, Dh])
    var v = reshape_owned(slice(qkv, 1, 2 * I, I, ctx), [1, S, H, Dh])
    var qn = rms_norm(q, weights.q_norm[], qk_eps, ctx)
    var kn = rms_norm(k, weights.k_norm[], qk_eps, ctx)
    var qr = _partial_rope_forward[H, Rot](qn, cos, sin, ctx)
    var kr = _partial_rope_forward[H, Rot](kn, cos, sin, ctx)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ff = sdpa_flash_train_fwd[1, S, H, Dh](qr, kr, v, scale, ctx)
    var att_flat = reshape(ff.o, [S, I], ctx)
    var ao = _projection_forward(att_flat, weights.out_proj[], lora.out_proj, ctx)
    var x1 = residual_gate(x, mod.gate_msa[], ao, ctx)
    var n2 = rms_norm(x1, weights.norm2[], norm_eps, ctx)
    var a2 = modulate(n2, mod.scale_mlp[], mod.shift_mlp[], ctx)
    var fc1 = _projection_forward(a2, weights.fc1[], lora.fc1, ctx)
    var gate = slice(fc1, 1, 0, F, ctx)
    var value = slice(fc1, 1, F, F, ctx)
    var act = swiglu(gate, value, ctx)
    var fc2 = _projection_forward(act, weights.fc2[], lora.fc2, ctx)
    return residual_gate(x1, mod.gate_mlp[], fc2, ctx)


def minimax_h3_bf16_flash_core_backward[
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
    qkv_layout_receipt: Int,
    fc1_layout_receipt: Int,
    fc1_lora_b_layout_receipt: Int,
    ctx: DeviceContext,
) raises -> MiniMaxH3BF16FlashCoreBackward:
    _validate[S, H, Dh, D, F, Rot](
        x, weights, mod, lora, cos, sin, qkv_layout_receipt,
        fc1_layout_receipt, fc1_lora_b_layout_receipt, norm_eps, qk_eps,
    )
    _require_bf16("d_y", d_y)
    if d_y.shape() != [S, D]:
        raise Error("BF16 H3 d_y must be [S,D]")
    comptime I = H * Dh

    var n1 = rms_norm(x, weights.norm1[], norm_eps, ctx)
    var a1 = modulate(n1, mod.scale_msa[], mod.shift_msa[], ctx)
    var qkv = _projection_forward(a1, weights.qkv[], lora.qkv, ctx)
    var q = reshape_owned(slice(qkv, 1, 0, I, ctx), [1, S, H, Dh])
    var k = reshape_owned(slice(qkv, 1, I, I, ctx), [1, S, H, Dh])
    var v = reshape_owned(slice(qkv, 1, 2 * I, I, ctx), [1, S, H, Dh])
    var qn = rms_norm(q, weights.q_norm[], qk_eps, ctx)
    var kn = rms_norm(k, weights.k_norm[], qk_eps, ctx)
    var qr = _partial_rope_forward[H, Rot](qn, cos, sin, ctx)
    var kr = _partial_rope_forward[H, Rot](kn, cos, sin, ctx)
    var scale = Float32(1.0) / sqrt(Float32(Dh))
    var ff = sdpa_flash_train_fwd[1, S, H, Dh](qr, kr, v, scale, ctx)
    var att_flat = reshape(ff.o, [S, I], ctx)
    var ao = _projection_forward(att_flat, weights.out_proj[], lora.out_proj, ctx)
    var x1 = residual_gate(x, mod.gate_msa[], ao, ctx)
    var n2 = rms_norm(x1, weights.norm2[], norm_eps, ctx)
    var a2 = modulate(n2, mod.scale_mlp[], mod.shift_mlp[], ctx)
    var fc1 = _projection_forward(a2, weights.fc1[], lora.fc1, ctx)
    var gate = slice(fc1, 1, 0, F, ctx)
    var value = slice(fc1, 1, F, F, ctx)
    var act = swiglu(gate, value, ctx)

    var rg2 = gate_residual_backward_dxdy(d_y, mod.gate_mlp[], ctx)
    var rg2_dx = rg2.d_x.clone(ctx)
    var b_fc2 = _projection_backward(rg2.d_y, act, weights.fc2[], lora.fc2, S, ctx)
    var b_swiglu = swiglu_backward(b_fc2.d_x[], gate, value, ctx)
    var d_fc1 = concat(1, ctx, b_swiglu.d_gate, b_swiglu.d_up)
    var b_fc1 = _projection_backward(d_fc1, a2, weights.fc1[], lora.fc1, S, ctx)
    var b_mod2 = modulate_backward(b_fc1.d_x[], n2, mod.scale_mlp[], ctx, False)
    var d_x1_norm = rms_norm_backward_dx(b_mod2.d_x, x1, weights.norm2[], norm_eps, ctx)
    var d_x1 = add(rg2_dx, d_x1_norm, ctx)

    var rg1 = gate_residual_backward_dxdy(d_x1, mod.gate_msa[], ctx)
    var rg1_dx = rg1.d_x.clone(ctx)
    var b_out = _projection_backward(rg1.d_y, att_flat, weights.out_proj[], lora.out_proj, S, ctx)
    var d_att = reshape(b_out.d_x[], [1, S, H, Dh], ctx)
    var b_att = sdpa_flash_backward[1, S, H, Dh](ff, d_att, scale, ctx)
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
    var b_qkv = _projection_backward(d_qkv, a1, weights.qkv[], lora.qkv, S, ctx)
    var b_mod1 = modulate_backward(b_qkv.d_x[], n1, mod.scale_msa[], ctx, False)
    var d_x_norm = rms_norm_backward_dx(b_mod1.d_x, x, weights.norm1[], norm_eps, ctx)
    var d_x = add(rg1_dx, d_x_norm, ctx)

    return MiniMaxH3BF16FlashCoreBackward(
        TArc(d_x^),
        MiniMaxH3BF16FlashCoreLoraGrad(b_qkv.d_a, b_qkv.d_b),
        MiniMaxH3BF16FlashCoreLoraGrad(b_out.d_a, b_out.d_b),
        MiniMaxH3BF16FlashCoreLoraGrad(b_fc1.d_a, b_fc1.d_b),
        MiniMaxH3BF16FlashCoreLoraGrad(b_fc2.d_a, b_fc2.d_b),
    )
