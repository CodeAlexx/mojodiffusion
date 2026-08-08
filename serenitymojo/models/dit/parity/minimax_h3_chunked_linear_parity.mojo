# MiniMax-H3 long-sequence BF16 projection parity.
#
# The production H3 block switches from the shared full-F32-accumulator
# `linear` to `minimax_h3_bf16_linear_chunked` above 16,384 sequence rows.
# This GPU gate straddles that boundary and compares the chunked result to the
# ordinary full-accumulator operator on identical BF16 inputs and weights.

from std.collections import Dict, List
from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.parity import ParityHarness
from serenitymojo.tensor import Tensor
from serenitymojo.ops.linear import linear
from serenitymojo.ops.random import randn
from serenitymojo.ops.activations import swiglu_packed_value_gate
from serenitymojo.ops.cast import cast_tensor
from serenitymojo.ops.elementwise import residual_gate
from serenitymojo.ops.int8_quant import (
    int8_dequant_groupwise_to_bf16,
    int8_encode_groupwise,
    int8_encode_perrow,
    int8_groupscale,
    int8_rowscale,
)
from serenitymojo.ops.rope import rope_halfsplit_full_head_broadcast
from serenitymojo.ops.tensor_algebra import concat, slice
from serenitymojo.ops.vec_rms_norm import vec_rms_norm
from serenitymojo.models.dit.minimax_h3_int8_linear import (
    minimax_h3_bf16_linear_chunked,
    minimax_h3_bf16_mlp_residual_inplace,
    minimax_h3_bf16_qkv_linear,
    minimax_h3_groupwise_mlp_residual_inplace,
    minimax_h3_groupwise_qkv_linear,
    minimax_h3_int8_linear,
    minimax_h3_int8_mlp_residual_inplace,
    minimax_h3_int8_qkv_linear,
    minimax_h3_int8_residual_linear,
    minimax_h3_int8_residual_linear_inplace,
    minimax_h3_int8_swiglu_linear,
)
from serenitymojo.models.dit.minimax_h3_qk_inplace import (
    minimax_h3_qk_norm_partial_rope_inplace,
)
from serenitymojo.models.dit.minimax_h3_dit import MiniMaxH3DiTConfig
from serenitymojo.models.dit.minimax_h3_frontend import (
    minimax_h3_final_layer,
    minimax_h3_final_layer_chunked,
)


comptime M = 17001
comptime N = 512
comptime K = 256


def main() raises:
    var ctx = DeviceContext()
    var x = randn([M, K], 20260805, STDtype.BF16, ctx)
    var weight = randn([N, K], 20260806, STDtype.BF16, ctx)
    var reference = linear(x, weight, None, ctx).to_host(ctx)
    var chunked = minimax_h3_bf16_linear_chunked(x, weight, ctx)
    var result = ParityHarness(0.999999).compare(chunked, reference, ctx)
    print("MiniMax-H3 chunked BF16 linear parity:", result)
    if not result.passed:
        raise Error("MiniMax-H3 chunked BF16 linear parity failed")

    # The long-sequence block fuses the direct-W8A8 FC1 dequant boundary and
    # SwiGLU so `[M, 2*ffn]` is never materialized. Compare it against the
    # original two-operation route with identical quantized inputs.
    var fc1_x = randn([1025, 128], 20260807, STDtype.BF16, ctx)
    var fc1_weight = randn([512, 128], 20260808, STDtype.BF16, ctx)
    var fc1_scale = int8_rowscale(fc1_weight, ctx)
    var fc1_int8 = int8_encode_perrow(fc1_weight, fc1_scale, ctx)
    var packed = minimax_h3_int8_linear(
        fc1_x, fc1_int8, fc1_scale, ctx
    )
    var swiglu_reference = swiglu_packed_value_gate(packed, ctx).to_host(ctx)
    var swiglu_fused = minimax_h3_int8_swiglu_linear(
        fc1_x, fc1_int8, fc1_scale, ctx
    )
    var swiglu_result = ParityHarness(1.0).compare(
        swiglu_fused, swiglu_reference, ctx
    )
    print("MiniMax-H3 fused W8A8 SwiGLU parity:", swiglu_result)
    if not swiglu_result.passed:
        raise Error("MiniMax-H3 fused W8A8 SwiGLU parity failed")

    # FC2 and its residual gate are fused for the same memory reason. The
    # dequantized projection must still round to BF16 before the F32 residual
    # expression, exactly like the original two-operator route.
    var fc2_x = randn([1025, 256], 20260809, STDtype.BF16, ctx)
    var fc2_weight = randn([128, 256], 20260810, STDtype.BF16, ctx)
    var fc2_scale = int8_rowscale(fc2_weight, ctx)
    var fc2_int8 = int8_encode_perrow(fc2_weight, fc2_scale, ctx)
    var residual = randn([1025, 128], 20260811, STDtype.BF16, ctx)
    var gate = randn([1025, 128], 20260812, STDtype.BF16, ctx)
    var fc2_projected = minimax_h3_int8_linear(
        fc2_x, fc2_int8, fc2_scale, ctx
    )
    var fc2_reference = residual_gate(
        residual, gate, fc2_projected, ctx
    ).to_host(ctx)
    var fc2_fused = minimax_h3_int8_residual_linear(
        fc2_x, fc2_int8, fc2_scale, residual, gate, ctx
    )
    var fc2_result = ParityHarness(1.0).compare(
        fc2_fused, fc2_reference, ctx
    )
    print("MiniMax-H3 fused W8A8 FC2 residual parity:", fc2_result)
    if not fc2_result.passed:
        raise Error("MiniMax-H3 fused W8A8 FC2 residual parity failed")

    var inplace_residual = randn([1025, 128], 20260811, STDtype.BF16, ctx)
    var fc2_inplace = minimax_h3_int8_residual_linear_inplace(
        fc2_x, fc2_int8, fc2_scale, inplace_residual^, gate, ctx
    )
    var fc2_inplace_result = ParityHarness(1.0).compare(
        fc2_inplace, fc2_reference, ctx
    )
    print("MiniMax-H3 in-place W8A8 FC2 residual parity:", fc2_inplace_result)
    if not fc2_inplace_result.passed:
        raise Error("MiniMax-H3 in-place W8A8 FC2 residual parity failed")

    var qkv_x = randn([1025, 128], 20260813, STDtype.BF16, ctx)
    var qkv_weight = randn([384, 128], 20260814, STDtype.BF16, ctx)
    var qkv_scale = int8_rowscale(qkv_weight, ctx)
    var qkv_int8 = int8_encode_perrow(qkv_weight, qkv_scale, ctx)
    var qkv_packed = minimax_h3_int8_linear(
        qkv_x, qkv_int8, qkv_scale, ctx
    )
    var q_ref = slice(qkv_packed, 1, 0, 128, ctx).to_host(ctx)
    var k_ref = slice(qkv_packed, 1, 128, 128, ctx).to_host(ctx)
    var v_ref = slice(qkv_packed, 1, 256, 128, ctx).to_host(ctx)
    var qkv_split = minimax_h3_int8_qkv_linear(
        qkv_x, qkv_int8, qkv_scale, ctx
    )
    # Cosine summation can land a few ulps below 1.0 even at max_abs=0.0.
    var q_result = ParityHarness(0.999999).compare(qkv_split.q, q_ref, ctx)
    var k_result = ParityHarness(0.999999).compare(qkv_split.k, k_ref, ctx)
    var v_result = ParityHarness(0.999999).compare(qkv_split.v, v_ref, ctx)
    print("MiniMax-H3 split W8A8 Q parity:", q_result)
    print("MiniMax-H3 split W8A8 K parity:", k_result)
    print("MiniMax-H3 split W8A8 V parity:", v_result)
    if not q_result.passed or not k_result.passed or not v_result.passed:
        raise Error("MiniMax-H3 split W8A8 QKV parity failed")

    var bf16_qkv_packed = minimax_h3_bf16_linear_chunked(
        qkv_x, qkv_weight, ctx
    )
    var bf16_q_ref = slice(bf16_qkv_packed, 1, 0, 128, ctx).to_host(ctx)
    var bf16_k_ref = slice(bf16_qkv_packed, 1, 128, 128, ctx).to_host(ctx)
    var bf16_v_ref = slice(bf16_qkv_packed, 1, 256, 128, ctx).to_host(ctx)
    var bf16_qkv_split = minimax_h3_bf16_qkv_linear(
        qkv_x, qkv_weight, ctx
    )
    var bf16_q_result = ParityHarness(0.999999).compare(
        bf16_qkv_split.q, bf16_q_ref, ctx
    )
    var bf16_k_result = ParityHarness(0.999999).compare(
        bf16_qkv_split.k, bf16_k_ref, ctx
    )
    var bf16_v_result = ParityHarness(0.999999).compare(
        bf16_qkv_split.v, bf16_v_ref, ctx
    )
    print("MiniMax-H3 split BF16 Q parity:", bf16_q_result)
    print("MiniMax-H3 split BF16 K parity:", bf16_k_result)
    print("MiniMax-H3 split BF16 V parity:", bf16_v_result)
    if not bf16_q_result.passed or not bf16_k_result.passed \
            or not bf16_v_result.passed:
        raise Error("MiniMax-H3 split BF16 QKV parity failed")

    var mlp_x = randn([1025, 128], 20260815, STDtype.BF16, ctx)
    var mlp_fc1 = randn([512, 128], 20260816, STDtype.BF16, ctx)
    var mlp_fc1_scale = int8_rowscale(mlp_fc1, ctx)
    var mlp_fc1_i8 = int8_encode_perrow(mlp_fc1, mlp_fc1_scale, ctx)
    var mlp_fc2 = randn([128, 256], 20260817, STDtype.BF16, ctx)
    var mlp_fc2_scale = int8_rowscale(mlp_fc2, ctx)
    var mlp_fc2_i8 = int8_encode_perrow(mlp_fc2, mlp_fc2_scale, ctx)
    var mlp_residual = randn([1025, 128], 20260818, STDtype.BF16, ctx)
    var mlp_residual_fused = randn(
        [1025, 128], 20260818, STDtype.BF16, ctx
    )
    var mlp_gate = randn([1025, 128], 20260819, STDtype.BF16, ctx)
    var mlp_packed = minimax_h3_int8_linear(
        mlp_x, mlp_fc1_i8, mlp_fc1_scale, ctx
    )
    var mlp_act = swiglu_packed_value_gate(mlp_packed, ctx)
    var mlp_projected = minimax_h3_int8_linear(
        mlp_act, mlp_fc2_i8, mlp_fc2_scale, ctx
    )
    var mlp_reference = residual_gate(
        mlp_residual, mlp_gate, mlp_projected, ctx
    ).to_host(ctx)
    var mlp_fused = minimax_h3_int8_mlp_residual_inplace(
        mlp_x, mlp_fc1_i8, mlp_fc1_scale, mlp_fc2_i8, mlp_fc2_scale,
        mlp_residual_fused^, mlp_gate, ctx,
    )
    var mlp_result = ParityHarness(0.999999).compare(
        mlp_fused, mlp_reference, ctx
    )
    print("MiniMax-H3 chunk-fused W8A8 MLP parity:", mlp_result)
    if not mlp_result.passed:
        raise Error("MiniMax-H3 chunk-fused W8A8 MLP parity failed")

    var bf16_mlp_packed = minimax_h3_bf16_linear_chunked(
        mlp_x, mlp_fc1, ctx
    )
    var bf16_mlp_act = swiglu_packed_value_gate(bf16_mlp_packed, ctx)
    var bf16_mlp_projected = minimax_h3_bf16_linear_chunked(
        bf16_mlp_act, mlp_fc2, ctx
    )
    var bf16_mlp_reference = residual_gate(
        mlp_residual, mlp_gate, bf16_mlp_projected, ctx
    ).to_host(ctx)
    var bf16_mlp_residual_fused = randn(
        [1025, 128], 20260818, STDtype.BF16, ctx
    )
    var bf16_mlp_fused = minimax_h3_bf16_mlp_residual_inplace(
        mlp_x, mlp_fc1, mlp_fc2, bf16_mlp_residual_fused^, mlp_gate, ctx
    )
    var bf16_mlp_result = ParityHarness(0.999999).compare(
        bf16_mlp_fused, bf16_mlp_reference, ctx
    )
    print("MiniMax-H3 chunk-fused BF16 MLP parity:", bf16_mlp_result)
    if not bf16_mlp_result.passed:
        raise Error("MiniMax-H3 chunk-fused BF16 MLP parity failed")

    # The long-sequence direct-W8A8 path owns split Q/K buffers, so Q/K norm
    # and partial RoPE can safely update them in place.  Gate the combined path
    # against the original allocating composition, including every BF16 round.
    var qk_input = randn([1, 257, 4, 128], 20260820, STDtype.BF16, ctx)
    var qk_inplace = qk_input.clone(ctx)
    var qk_weight = randn([128], 20260821, STDtype.BF16, ctx)
    var qk_cos = randn([257, 96], 20260822, STDtype.F32, ctx)
    var qk_sin = randn([257, 96], 20260823, STDtype.F32, ctx)
    var qk_norm = vec_rms_norm(qk_input, qk_weight, 1.0e-6, ctx)
    var qk_rot = slice(qk_norm, 3, 0, 96, ctx)
    var qk_pass = slice(qk_norm, 3, 96, 32, ctx)
    var qk_rotated = rope_halfsplit_full_head_broadcast(
        qk_rot, qk_cos, qk_sin, 4, ctx
    )
    var qk_reference = concat(3, ctx, qk_rotated, qk_pass).to_host(ctx)
    minimax_h3_qk_norm_partial_rope_inplace(
        qk_inplace, qk_weight, qk_cos, qk_sin, 4, 96, 1.0e-6, ctx
    )
    var qk_result = ParityHarness(0.999999).compare(
        qk_inplace, qk_reference, ctx
    )
    print("MiniMax-H3 in-place Q/K norm + partial RoPE parity:", qk_result)
    if not qk_result.passed:
        raise Error("MiniMax-H3 in-place Q/K norm + partial RoPE parity failed")

    var group_qkv_weight = randn([384, 128], 20260831, STDtype.BF16, ctx)
    var group_qkv_scale_f32 = int8_groupscale(group_qkv_weight, 16, ctx)
    var group_qkv_i8 = int8_encode_groupwise(
        group_qkv_weight, group_qkv_scale_f32, 16, ctx
    )
    var group_qkv_scale = cast_tensor(
        group_qkv_scale_f32, STDtype.F16, ctx
    )
    var group_qkv_bf16 = int8_dequant_groupwise_to_bf16(
        group_qkv_i8, group_qkv_scale, 16, ctx
    )
    var group_qkv_packed = minimax_h3_bf16_linear_chunked(
        qkv_x, group_qkv_bf16, ctx
    )
    var group_q_ref = slice(group_qkv_packed, 1, 0, 128, ctx).to_host(ctx)
    var group_k_ref = slice(group_qkv_packed, 1, 128, 128, ctx).to_host(ctx)
    var group_v_ref = slice(group_qkv_packed, 1, 256, 128, ctx).to_host(ctx)
    var group_qkv_split = minimax_h3_groupwise_qkv_linear(
        qkv_x, group_qkv_i8, group_qkv_scale, ctx
    )
    var group_q_result = ParityHarness(0.999999).compare(
        group_qkv_split.q, group_q_ref, ctx
    )
    var group_k_result = ParityHarness(0.999999).compare(
        group_qkv_split.k, group_k_ref, ctx
    )
    var group_v_result = ParityHarness(0.999999).compare(
        group_qkv_split.v, group_v_ref, ctx
    )
    print("MiniMax-H3 split groupwise Q parity:", group_q_result)
    print("MiniMax-H3 split groupwise K parity:", group_k_result)
    print("MiniMax-H3 split groupwise V parity:", group_v_result)
    if not group_q_result.passed or not group_k_result.passed \
            or not group_v_result.passed:
        raise Error("MiniMax-H3 split groupwise QKV parity failed")

    var group_fc1 = randn([512, 128], 20260832, STDtype.BF16, ctx)
    var group_fc1_scale_f32 = int8_groupscale(group_fc1, 16, ctx)
    var group_fc1_i8 = int8_encode_groupwise(
        group_fc1, group_fc1_scale_f32, 16, ctx
    )
    var group_fc1_scale = cast_tensor(
        group_fc1_scale_f32, STDtype.F16, ctx
    )
    var group_fc2 = randn([128, 256], 20260833, STDtype.BF16, ctx)
    var group_fc2_scale_f32 = int8_groupscale(group_fc2, 32, ctx)
    var group_fc2_i8 = int8_encode_groupwise(
        group_fc2, group_fc2_scale_f32, 32, ctx
    )
    var group_fc2_scale = cast_tensor(
        group_fc2_scale_f32, STDtype.F16, ctx
    )
    var group_fc1_bf16 = int8_dequant_groupwise_to_bf16(
        group_fc1_i8, group_fc1_scale, 16, ctx
    )
    var group_fc2_bf16 = int8_dequant_groupwise_to_bf16(
        group_fc2_i8, group_fc2_scale, 32, ctx
    )
    var group_packed = minimax_h3_bf16_linear_chunked(
        mlp_x, group_fc1_bf16, ctx
    )
    var group_act = swiglu_packed_value_gate(group_packed, ctx)
    var group_projected = minimax_h3_bf16_linear_chunked(
        group_act, group_fc2_bf16, ctx
    )
    var group_reference = residual_gate(
        mlp_residual, mlp_gate, group_projected, ctx
    ).to_host(ctx)
    var group_residual_fused = randn(
        [1025, 128], 20260818, STDtype.BF16, ctx
    )
    var group_fused = minimax_h3_groupwise_mlp_residual_inplace(
        mlp_x, group_fc1_i8, group_fc1_scale,
        group_fc2_i8, group_fc2_scale,
        group_residual_fused^, mlp_gate, ctx,
    )
    var group_result = ParityHarness(0.999999).compare(
        group_fused, group_reference, ctx
    )
    print("MiniMax-H3 chunk-fused groupwise MLP parity:", group_result)
    if not group_result.passed:
        raise Error("MiniMax-H3 chunk-fused groupwise MLP parity failed")

    var final_config = MiniMaxH3DiTConfig(
        128, 1, 1, 4, 32, 256, 24, 32, 64, 16, 64,
        2304, 256, 4, 1.0e-5, 1.0e-5, 1.0e-5,
    )
    var final_weights = Dict[String, ArcPointer[Tensor]]()
    final_weights["final_layer.norm.weight"] = ArcPointer(
        randn([128], 20260824, STDtype.BF16, ctx)
    )
    final_weights["final_layer.video_out.weight"] = ArcPointer(
        randn([96, 128], 20260825, STDtype.F32, ctx)
    )
    final_weights["final_layer.video_out.bias"] = ArcPointer(
        randn([96], 20260826, STDtype.F32, ctx)
    )
    final_weights["final_layer.audio_out.weight"] = ArcPointer(
        randn([32, 128], 20260827, STDtype.F32, ctx)
    )
    final_weights["final_layer.audio_out.bias"] = ArcPointer(
        randn([32], 20260828, STDtype.F32, ctx)
    )
    var final_hidden = randn([2053, 128], 20260829, STDtype.BF16, ctx)
    var final_mod = randn([3, 256], 20260830, STDtype.BF16, ctx)
    var final_timesteps = List[Int](capacity=2053)
    var final_video_indices = List[Int]()
    var final_audio_indices = List[Int]()
    for row in range(2053):
        final_timesteps.append(row % 3)
        if row % 3 == 0:
            final_video_indices.append(row)
        elif row % 3 == 1:
            final_audio_indices.append(row)
    var final_reference = minimax_h3_final_layer(
        final_hidden, final_mod, final_timesteps,
        final_video_indices, final_audio_indices,
        final_weights, final_config, ctx,
    )
    var final_video_reference = final_reference.video_out.to_host(ctx)
    var final_audio_reference = final_reference.audio_out.to_host(ctx)
    var final_chunked = minimax_h3_final_layer_chunked(
        final_hidden, final_mod, final_timesteps,
        final_video_indices, final_audio_indices,
        final_weights, final_config, ctx,
    )
    var final_video_result = ParityHarness(0.999999).compare(
        final_chunked.video_out, final_video_reference, ctx
    )
    var final_audio_result = ParityHarness(0.999999).compare(
        final_chunked.audio_out, final_audio_reference, ctx
    )
    print("MiniMax-H3 chunked final video parity:", final_video_result)
    print("MiniMax-H3 chunked final audio parity:", final_audio_result)
    if not final_video_result.passed or not final_audio_result.passed:
        raise Error("MiniMax-H3 chunked final layer parity failed")
