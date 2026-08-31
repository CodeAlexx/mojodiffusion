# MiniMax-H3 reusable 50-core LoRA training stack.
#
# The released path streams one transformed BF16 base block during forward and
# reloads it during reverse backward. Only 51 BF16 residual-stream states are
# retained. Frozen AdaLN projections are consumed through MiniMaxH3ModCache;
# per-block modulation gathers are recomputed rather than checkpointed.

from max.gpu.host import DeviceContext
from std.collections import Dict, List
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.dit.minimax_h3_dit import (
    MiniMaxH3DiTConfig,
    MINIMAX_H3_FC1_SWAPPED_MARKER,
    MINIMAX_H3_QKV_DEINTERLEAVED_MARKER,
    minimax_h3_block_prefix,
    minimax_h3_require_transformed_weights,
)
from serenitymojo.models.dit.minimax_h3_loader_device import (
    minimax_h3_block_uploader,
    minimax_h3_load_block_device_up,
)
from serenitymojo.models.dit.minimax_h3_modcache import MiniMaxH3ModCache
from serenitymojo.models.minimax_h3.training_core import (
    MINIMAX_H3_TRAIN_FC1_LORA_B_RUNTIME_VALUE_GATE,
    MINIMAX_H3_TRAIN_FC1_RUNTIME_VALUE_GATE,
    MINIMAX_H3_TRAIN_QKV_RUNTIME_ALL_Q_K_V,
    MiniMaxH3TrainingBlockLoraDevice,
    MiniMaxH3TrainingModulationDevice,
    MiniMaxH3TrainingWeightsDevice,
    minimax_h3_training_core_backward,
    minimax_h3_training_core_forward,
    minimax_h3_training_modulation_from_table,
)


comptime TArc = ArcPointer[Tensor]
comptime MINIMAX_H3_TRAINING_BLOCKS = 50


@fieldwise_init
struct MiniMaxH3TrainingLoraDeviceSet(Copyable, Movable):
    # Exact block-major order; each block is qkv,out_proj,fc1,fc2.
    var blocks: List[MiniMaxH3TrainingBlockLoraDevice]


@fieldwise_init
struct MiniMaxH3TrainingStackTape(Copyable, Movable):
    # x0 plus every block output. No attention/MLP internals are retained.
    var residual_states: List[TArc]

    def output(self) raises -> TArc:
        if len(self.residual_states) != MINIMAX_H3_TRAINING_BLOCKS + 1:
            raise Error("MiniMax-H3 training tape is incomplete")
        return self.residual_states[MINIMAX_H3_TRAINING_BLOCKS].copy()


@fieldwise_init
struct MiniMaxH3TrainingStackGrads(Copyable, Movable):
    var d_x: TArc
    # Block-major qkv,out_proj,fc1,fc2 order, matching lora_surface (200 each).
    var d_a: List[TArc]
    var d_b: List[TArc]

    def adapter_count(self) -> Int:
        return len(self.d_a)


def _validate_lora_set(lora: MiniMaxH3TrainingLoraDeviceSet) raises:
    if len(lora.blocks) != MINIMAX_H3_TRAINING_BLOCKS:
        raise Error("MiniMax-H3 training requires exactly 50 block LoRA sets")


def _validate_released_config(config: MiniMaxH3DiTConfig) raises:
    config.validate()
    if (
        config.num_layers != 50 or config.num_attention_heads != 56
        or config.attention_head_dim != 128 or config.hidden_size != 5376
        or config.ffn_hidden_size != 14336 or config.norm_eps != Float32(1.0e-5)
        or config.qk_norm_eps != Float32(1.0e-5)
    ):
        raise Error("MiniMax-H3 training stack requires exact released block geometry")


def minimax_h3_training_weights_from_runtime_dict(
    weights: Dict[String, ArcPointer[Tensor]],
    layer: Int,
    config: MiniMaxH3DiTConfig,
) raises -> MiniMaxH3TrainingWeightsDevice:
    """Bind the existing inference loader's transformed BF16 block to training.

    INT8 resident dictionaries are rejected: current backward kernels require
    BF16 frozen weights and must never dequantize silently.
    """
    minimax_h3_require_transformed_weights(weights, layer)
    if (
        MINIMAX_H3_QKV_DEINTERLEAVED_MARKER not in weights
        or MINIMAX_H3_FC1_SWAPPED_MARKER not in weights
    ):
        raise Error("MiniMax-H3 training transformed-layout markers are missing")
    var p = minimax_h3_block_prefix(layer)
    var out = MiniMaxH3TrainingWeightsDevice(
        weights[p + String("norm1.weight")].copy(),
        weights[p + String("attn.qkv_proj.weight")].copy(),
        weights[p + String("attn.q_norm.weight")].copy(),
        weights[p + String("attn.k_norm.weight")].copy(),
        weights[p + String("attn.out_proj.weight")].copy(),
        weights[p + String("norm2.weight")].copy(),
        weights[p + String("mlp.fc1.weight")].copy(),
        weights[p + String("mlp.fc2.weight")].copy(),
    )
    for tensor in [
        out.norm1, out.qkv, out.q_norm, out.k_norm,
        out.out_proj, out.norm2, out.fc1, out.fc2,
    ]:
        if tensor[].dtype() != STDtype.BF16:
            raise Error(
                String("MiniMax-H3 training layer ") + String(layer)
                + String(" contains INT8/non-BF16 base weights; backward unsupported")
            )
    return out


def minimax_h3_training_stack_forward_resident[
    S: Int, H: Int, Dh: Int, D: Int, F: Int, Rot: Int
](
    x: Tensor,
    weights: List[MiniMaxH3TrainingWeightsDevice],
    modulations: List[MiniMaxH3TrainingModulationDevice],
    lora: MiniMaxH3TrainingLoraDeviceSet,
    cos: Tensor,
    sin: Tensor,
    ctx: DeviceContext,
) raises -> MiniMaxH3TrainingStackTape:
    """Reusable resident-base composition used by bounded gates/small models."""
    _validate_lora_set(lora)
    if len(weights) != 50 or len(modulations) != 50:
        raise Error("MiniMax-H3 resident training stack requires exactly 50 blocks")
    var states = List[TArc](capacity=51)
    states.append(TArc(x.clone(ctx)))
    for block in range(50):
        var y = minimax_h3_training_core_forward[S, H, Dh, D, F, Rot](
            states[block][], weights[block], modulations[block], lora.blocks[block],
            cos, sin, 1.0e-5, 1.0e-5,
            MINIMAX_H3_TRAIN_QKV_RUNTIME_ALL_Q_K_V,
            MINIMAX_H3_TRAIN_FC1_RUNTIME_VALUE_GATE,
            MINIMAX_H3_TRAIN_FC1_LORA_B_RUNTIME_VALUE_GATE,
            ctx,
        )
        states.append(TArc(y^))
    return MiniMaxH3TrainingStackTape(states^)


def minimax_h3_training_stack_backward_resident[
    S: Int, H: Int, Dh: Int, D: Int, F: Int, Rot: Int
](
    d_y: Tensor,
    weights: List[MiniMaxH3TrainingWeightsDevice],
    modulations: List[MiniMaxH3TrainingModulationDevice],
    lora: MiniMaxH3TrainingLoraDeviceSet,
    cos: Tensor,
    sin: Tensor,
    tape: MiniMaxH3TrainingStackTape,
    ctx: DeviceContext,
) raises -> MiniMaxH3TrainingStackGrads:
    _validate_lora_set(lora)
    if len(weights) != 50 or len(modulations) != 50 or len(tape.residual_states) != 51:
        raise Error("MiniMax-H3 resident backward requires an exact 50-block tape")
    var d_a_reverse = List[TArc](capacity=200)
    var d_b_reverse = List[TArc](capacity=200)
    var handoff = d_y.clone(ctx)
    for reverse_index in range(50):
        var block = 49 - reverse_index
        var gradients = minimax_h3_training_core_backward[S, H, Dh, D, F, Rot](
            handoff, tape.residual_states[block][], weights[block],
            modulations[block], lora.blocks[block], cos, sin,
            1.0e-5, 1.0e-5,
            MINIMAX_H3_TRAIN_QKV_RUNTIME_ALL_Q_K_V,
            MINIMAX_H3_TRAIN_FC1_RUNTIME_VALUE_GATE,
            MINIMAX_H3_TRAIN_FC1_LORA_B_RUNTIME_VALUE_GATE,
            ctx,
        )
        # Reverse collection is reordered after the loop into lora_surface order.
        d_a_reverse.append(gradients.qkv.d_a.copy())
        d_b_reverse.append(gradients.qkv.d_b.copy())
        d_a_reverse.append(gradients.out_proj.d_a.copy())
        d_b_reverse.append(gradients.out_proj.d_b.copy())
        d_a_reverse.append(gradients.fc1.d_a.copy())
        d_b_reverse.append(gradients.fc1.d_b.copy())
        d_a_reverse.append(gradients.fc2.d_a.copy())
        d_b_reverse.append(gradients.fc2.d_b.copy())
        handoff = gradients.d_x[].clone(ctx)
    var d_a = List[TArc](capacity=200)
    var d_b = List[TArc](capacity=200)
    for block in range(50):
        var reverse_block = 49 - block
        for slot in range(4):
            var index = reverse_block * 4 + slot
            d_a.append(d_a_reverse[index].copy())
            d_b.append(d_b_reverse[index].copy())
    return MiniMaxH3TrainingStackGrads(TArc(handoff^), d_a^, d_b^)


def minimax_h3_training_stack_forward_streamed[
    S: Int
](
    x: Tensor,
    checkpoint: ShardedSafeTensors,
    modcache: MiniMaxH3ModCache,
    adaln_indices: List[Int],
    lora: MiniMaxH3TrainingLoraDeviceSet,
    cos: Tensor,
    sin: Tensor,
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
) raises -> MiniMaxH3TrainingStackTape:
    """Released 50-block forward: stream one BF16 base block at a time."""
    _validate_released_config(config)
    _validate_lora_set(lora)
    if modcache.num_layers() != 50 or len(adaln_indices) != S:
        raise Error("MiniMax-H3 training modcache/index count mismatch")
    var uploader = minimax_h3_block_uploader(ctx)
    var states = List[TArc](capacity=51)
    states.append(TArc(x.clone(ctx)))
    for block in range(50):
        var runtime = minimax_h3_load_block_device_up(
            checkpoint, block, config, uploader, ctx
        )
        var typed = minimax_h3_training_weights_from_runtime_dict(runtime, block, config)
        var modulation = minimax_h3_training_modulation_from_table[S, 5376](
            modcache.block_mod[block][], adaln_indices, ctx
        )
        var y = minimax_h3_training_core_forward[S, 56, 128, 5376, 14336, 96](
            states[block][], typed, modulation, lora.blocks[block], cos, sin,
            config.norm_eps, config.qk_norm_eps,
            MINIMAX_H3_TRAIN_QKV_RUNTIME_ALL_Q_K_V,
            MINIMAX_H3_TRAIN_FC1_RUNTIME_VALUE_GATE,
            MINIMAX_H3_TRAIN_FC1_LORA_B_RUNTIME_VALUE_GATE,
            ctx,
        )
        states.append(TArc(y^))
    return MiniMaxH3TrainingStackTape(states^)


def minimax_h3_training_stack_backward_streamed[
    S: Int
](
    d_y: Tensor,
    checkpoint: ShardedSafeTensors,
    modcache: MiniMaxH3ModCache,
    adaln_indices: List[Int],
    lora: MiniMaxH3TrainingLoraDeviceSet,
    cos: Tensor,
    sin: Tensor,
    tape: MiniMaxH3TrainingStackTape,
    config: MiniMaxH3DiTConfig,
    ctx: DeviceContext,
) raises -> MiniMaxH3TrainingStackGrads:
    """Released reverse loop: reload block49..block0 and chain every d_x."""
    _validate_released_config(config)
    _validate_lora_set(lora)
    if modcache.num_layers() != 50 or len(adaln_indices) != S \
            or len(tape.residual_states) != 51:
        raise Error("MiniMax-H3 streamed backward tape/modcache mismatch")
    var uploader = minimax_h3_block_uploader(ctx)
    var d_a_reverse = List[TArc](capacity=200)
    var d_b_reverse = List[TArc](capacity=200)
    var handoff = d_y.clone(ctx)
    for reverse_index in range(50):
        var block = 49 - reverse_index
        var runtime = minimax_h3_load_block_device_up(
            checkpoint, block, config, uploader, ctx
        )
        var typed = minimax_h3_training_weights_from_runtime_dict(runtime, block, config)
        var modulation = minimax_h3_training_modulation_from_table[S, 5376](
            modcache.block_mod[block][], adaln_indices, ctx
        )
        var gradients = minimax_h3_training_core_backward[
            S, 56, 128, 5376, 14336, 96
        ](
            handoff, tape.residual_states[block][], typed, modulation,
            lora.blocks[block], cos, sin, config.norm_eps, config.qk_norm_eps,
            MINIMAX_H3_TRAIN_QKV_RUNTIME_ALL_Q_K_V,
            MINIMAX_H3_TRAIN_FC1_RUNTIME_VALUE_GATE,
            MINIMAX_H3_TRAIN_FC1_LORA_B_RUNTIME_VALUE_GATE,
            ctx,
        )
        d_a_reverse.append(gradients.qkv.d_a.copy())
        d_b_reverse.append(gradients.qkv.d_b.copy())
        d_a_reverse.append(gradients.out_proj.d_a.copy())
        d_b_reverse.append(gradients.out_proj.d_b.copy())
        d_a_reverse.append(gradients.fc1.d_a.copy())
        d_b_reverse.append(gradients.fc1.d_b.copy())
        d_a_reverse.append(gradients.fc2.d_a.copy())
        d_b_reverse.append(gradients.fc2.d_b.copy())
        handoff = gradients.d_x[].clone(ctx)
    var d_a = List[TArc](capacity=200)
    var d_b = List[TArc](capacity=200)
    for block in range(50):
        var reverse_block = 49 - block
        for slot in range(4):
            var index = reverse_block * 4 + slot
            d_a.append(d_a_reverse[index].copy())
            d_b.append(d_b_reverse[index].copy())
    return MiniMaxH3TrainingStackGrads(TArc(handoff^), d_a^, d_b^)
