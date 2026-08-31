# MiniMax-H3 product LoRA state/device/optimizer seam.
#
# F32LoraState is the shared trainer-private checkpoint contract. This module
# binds its 200 Musubi-ordered adapters to the device stack's runtime layouts,
# consumes device F32 gradients in the same inventory order, and delegates the
# AdamW scalar update to the shared Mojo trainer implementation.

from max.gpu.host import DeviceContext
from std.collections import List
from std.math import isfinite
from std.memory import ArcPointer

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.models.minimax_h3.training_core import (
    MiniMaxH3TrainingBlockLoraDevice,
    MiniMaxH3TrainingLoraAdapterDevice,
)
from serenitymojo.models.minimax_h3.training_stack import (
    MiniMaxH3TrainingLoraDeviceSet,
    MiniMaxH3TrainingStackGrads,
)
from serenitymojo.training.lora_save import F32LoraState
from serenitymojo.training.train_step import _adamw_host_list_f32
from serenitymojo.training.minimax_h3.contract import (
    MiniMaxH3TrainingContract,
    validate_minimax_h3_policy,
)
from serenitymojo.training.minimax_h3.lora_layout import (
    minimax_h3_fc1_lora_up_musubi_to_runtime,
    minimax_h3_fc1_lora_up_runtime_to_musubi,
)
from serenitymojo.training.minimax_h3.lora_surface import minimax_h3_lora_surface


comptime TArc = ArcPointer[Tensor]


def validate_minimax_h3_device_training_contract(
    contract: MiniMaxH3TrainingContract,
) raises:
    """Execution preflight beyond the host policy/cache contract."""
    validate_minimax_h3_policy(contract)
    if contract.full_finetune:
        raise Error("MiniMax-H3 device core implements LoRA only; full FT unsupported")
    if contract.base_storage != String("bf16"):
        raise Error(
            "MiniMax-H3 device backward requires BF16 frozen base weights; "
            "INT8/convrot backward is unsupported"
        )


def _validate_states(states: List[F32LoraState]) raises:
    var surface = minimax_h3_lora_surface()
    if len(states) != 200 or len(surface) != 200:
        raise Error("MiniMax-H3 product LoRA state must contain exactly 200 adapters")
    var rank = states[0].rank
    if rank <= 0:
        raise Error("MiniMax-H3 product LoRA rank must be positive")
    for i in range(200):
        ref state = states[i]
        ref target = surface[i]
        if (
            state.prefix != target.musubi_prefix or state.rank != rank
            or state.in_f != target.in_features or state.out_f != target.out_features
            or len(state.a) != rank * state.in_f
            or len(state.b) != state.out_f * rank
            or len(state.ma) != len(state.a) or len(state.va) != len(state.a)
            or len(state.mb) != len(state.b) or len(state.vb) != len(state.b)
        ):
            raise Error(
                String("MiniMax-H3 LoRA private-state mismatch at target ") + String(i)
            )


def _device_adapter(
    state: F32LoraState,
    b_values: List[Float32],
    scale: Float32,
    ctx: DeviceContext,
) raises -> MiniMaxH3TrainingLoraAdapterDevice:
    return MiniMaxH3TrainingLoraAdapterDevice(
        TArc(Tensor.from_host(
            state.a.copy(), [state.rank, state.in_f], STDtype.F32, ctx
        )),
        TArc(Tensor.from_host(
            b_values, [state.out_f, state.rank], STDtype.F32, ctx
        )),
        state.rank,
        state.in_f,
        state.out_f,
        scale,
    )


def minimax_h3_lora_device_set_from_f32_state(
    states: List[F32LoraState],
    multiplier: Float32,
    alpha: Float32,
    ctx: DeviceContext,
) raises -> MiniMaxH3TrainingLoraDeviceSet:
    """Upload shared F32 private state into the runtime-layout device set.

    FC1 B/up is swapped raw `[gate;value]` -> runtime `[value;gate]`; A/down
    and all other adapters are layout-identical.
    """
    _validate_states(states)
    if not isfinite(multiplier) or not isfinite(alpha):
        raise Error("MiniMax-H3 LoRA multiplier/alpha must be finite")
    var blocks = List[MiniMaxH3TrainingBlockLoraDevice](capacity=50)
    for block in range(50):
        var base = 4 * block
        var rank = states[base].rank
        var scale = multiplier * alpha / Float32(rank)
        if not isfinite(scale):
            raise Error("MiniMax-H3 LoRA scale must be finite")
        var qkv_b = states[base + 0].b.copy()
        var out_b = states[base + 1].b.copy()
        var fc1_b = minimax_h3_fc1_lora_up_musubi_to_runtime(
            states[base + 2].b, 14336, rank
        )
        var fc2_b = states[base + 3].b.copy()
        blocks.append(MiniMaxH3TrainingBlockLoraDevice(
            _device_adapter(states[base + 0], qkv_b^, scale, ctx),
            _device_adapter(states[base + 1], out_b^, scale, ctx),
            _device_adapter(states[base + 2], fc1_b^, scale, ctx),
            _device_adapter(states[base + 3], fc2_b^, scale, ctx),
        ))
    return MiniMaxH3TrainingLoraDeviceSet(blocks^)


def minimax_h3_lora_adamw_step_from_device_grads(
    mut states: List[F32LoraState],
    grads: MiniMaxH3TrainingStackGrads,
    step: Int,
    learning_rate: Float32,
    ctx: DeviceContext,
    beta1: Float32 = Float32(0.9),
    beta2: Float32 = Float32(0.999),
    eps: Float32 = Float32(1.0e-8),
    weight_decay: Float32 = Float32(0.01),
) raises:
    """Apply one shared AdamW step to F32 masters/moments.

    This bounded first product seam performs the device-gradient -> host-F32
    state boundary once per optimizer step. A future resident-F32 optimizer may
    replace this transport without changing stack ordering or checkpoint keys.
    """
    _validate_states(states)
    if step < 1 or len(grads.d_a) != 200 or len(grads.d_b) != 200:
        raise Error("MiniMax-H3 AdamW step requires step>=1 and exactly 200 gradients")
    for i in range(200):
        if grads.d_a[i][].dtype() != STDtype.F32 or grads.d_b[i][].dtype() != STDtype.F32:
            raise Error(String("MiniMax-H3 optimizer received non-F32 gradient ") + String(i))
        var d_a = grads.d_a[i][].to_host(ctx)
        var d_b_runtime = grads.d_b[i][].to_host(ctx)
        var d_b: List[Float32]
        if i % 4 == 2:
            d_b = minimax_h3_fc1_lora_up_runtime_to_musubi(
                d_b_runtime, 14336, states[i].rank
            )
        else:
            d_b = d_b_runtime.copy()
        _adamw_host_list_f32(
            states[i].a, d_a, states[i].ma, states[i].va,
            step, learning_rate, beta1, beta2, eps, weight_decay,
        )
        _adamw_host_list_f32(
            states[i].b, d_b, states[i].mb, states[i].vb,
            step, learning_rate, beta1, beta2, eps, weight_decay,
        )
