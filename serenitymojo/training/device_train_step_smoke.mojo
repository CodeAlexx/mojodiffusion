# device_train_step_smoke.mojo — device train-step ABI smoke.
#
# Run:
#   pixi run mojo run -I . serenitymojo/training/device_train_step_smoke.mojo

from std.gpu.host import DeviceContext
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.training.device_train_step import (
    DeviceAdamWState,
    DeviceGradSet,
    DeviceOptimizerConfig,
    DeviceTrainableSet,
    device_adamw_train_step_update,
    device_grad_stats,
    device_optimizer_backend_name,
    device_optimizer_train_step_update,
    device_optimizer_train_step_update_with_arena,
    host_grad_compat_result,
    validate_device_optimizer_supported_for_fast_path,
)
from serenitymojo.training.training_arena import (
    TRAINING_ARENA_PHASE_OPTIMIZER,
    TrainingArena,
)
from serenitymojo.training.train_config import (
    TRAIN_OPTIMIZER_ADAFACTOR,
    TRAIN_OPTIMIZER_ADAMW,
    TRAIN_OPTIMIZER_ADAMW_8BIT,
    TRAIN_OPTIMIZER_AUTOMAGIC3,
    TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW,
)


comptime TArc = ArcPointer[Tensor]


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("device_train_step_smoke FAILED: ") + msg)


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _adamw_config() -> DeviceOptimizerConfig:
    return DeviceOptimizerConfig(
        TRAIN_OPTIMIZER_ADAMW,
        Float32(0.01),
        Float32(0.9),
        Float32(0.999),
        Float32(1.0e-8),
        Float32(0.0),
        Float32(1.0),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
    )


def _unsupported_config(optimizer: Int) -> DeviceOptimizerConfig:
    return DeviceOptimizerConfig(
        optimizer,
        Float32(0.01),
        Float32(0.9),
        Float32(0.999),
        Float32(1.0e-8),
        Float32(0.0),
        Float32(1.0),
        Float32(0.0),
        Float32(0.0),
        Float32(0.0),
    )


def _expect_non_fast_optimizer(optimizer: Int, backend: String) raises:
    _check(
        device_optimizer_backend_name(optimizer) == backend,
        String("registered backend label for ") + backend,
    )
    var saw_unported = False
    try:
        validate_device_optimizer_supported_for_fast_path(
            _unsupported_config(optimizer)
        )
    except e:
        saw_unported = True
        print("  raised as expected [device optimizer non-fast]:", String(e))
    _check(
        saw_unported,
        String("device optimizer fast path must fail loud for ") + backend,
    )


def main() raises:
    var ctx = DeviceContext()
    var shape: List[Int] = [4]
    var p0 = Tensor.from_host([Float32(1.0), Float32(2.0), Float32(3.0), Float32(4.0)], shape.copy(), STDtype.F32, ctx)
    var g0 = Tensor.from_host([Float32(10.0), Float32(10.0), Float32(10.0), Float32(10.0)], shape.copy(), STDtype.F32, ctx)
    var m0 = Tensor.from_host(_zeros(4), shape.copy(), STDtype.F32, ctx)
    var v0 = Tensor.from_host(_zeros(4), shape.copy(), STDtype.F32, ctx)

    var trainables = DeviceTrainableSet()
    trainables.append(String("adapter.0.weight"), TArc(p0^), String("rank=2"))
    var grads = DeviceGradSet()
    grads.append(String("adapter.0.weight"), TArc(g0^), String("loss=mse"))
    var state = DeviceAdamWState()
    state.append(TArc(m0^), TArc(v0^))

    var result = device_adamw_train_step_update(
        trainables,
        grads,
        state,
        Float32(1.25),
        1,
        Float32(0.01),
        Float32(0.9),
        Float32(0.999),
        Float32(1.0e-8),
        Float32(0.0),
        Float32(1.0),
        ctx,
    )
    result.validate()
    _check(result.is_fast_path(), "result should be device fast path")
    _check(result.full_tensor_readback_count == 0, "no full grad readback")
    _check(result.scalar_readback_count == 2, "global norm and nonfinite scalar readbacks expected")
    _check(result.nonfinite_grad_count == 0, "finite grads should report no nonfinite values")
    _check(result.clip_scale < 1.0, "clip scale should fold into AdamW")
    _check(result.grad_norm > 1.0, "grad norm should be measured")
    var updated = trainables.params[0][].to_host(ctx)
    _check(updated[0] < 1.0, "parameter should update in place")

    _check(
        device_optimizer_backend_name(TRAIN_OPTIMIZER_ADAMW)
        == String("fused_adamw_multitensor"),
        "AdamW backend name",
    )
    _expect_non_fast_optimizer(
        TRAIN_OPTIMIZER_ADAMW_8BIT, String("device-adamw8bit-unported")
    )
    _expect_non_fast_optimizer(
        TRAIN_OPTIMIZER_AUTOMAGIC3, String("device-automagic3-host-grad-compat")
    )
    _expect_non_fast_optimizer(
        TRAIN_OPTIMIZER_ADAFACTOR, String("device-adafactor-unported")
    )
    _expect_non_fast_optimizer(
        TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW,
        String("device-schedulefree-adamw-unported"),
    )

    var p1 = Tensor.from_host([Float32(1.0), Float32(2.0), Float32(3.0), Float32(4.0)], shape.copy(), STDtype.F32, ctx)
    var g1 = Tensor.from_host([Float32(10.0), Float32(10.0), Float32(10.0), Float32(10.0)], shape.copy(), STDtype.F32, ctx)
    var m1 = Tensor.from_host(_zeros(4), shape.copy(), STDtype.F32, ctx)
    var v1 = Tensor.from_host(_zeros(4), shape.copy(), STDtype.F32, ctx)
    var trainables_dispatch = DeviceTrainableSet()
    trainables_dispatch.append(String("adapter.1.weight"), TArc(p1^), String("rank=2"))
    var grads_dispatch = DeviceGradSet()
    grads_dispatch.append(String("adapter.1.weight"), TArc(g1^), String("loss=mse"))
    var state_dispatch = DeviceAdamWState()
    state_dispatch.append(TArc(m1^), TArc(v1^))
    var dispatch_result = device_optimizer_train_step_update(
        trainables_dispatch,
        grads_dispatch,
        state_dispatch,
        _adamw_config(),
        Float32(1.25),
        1,
        ctx,
    )
    dispatch_result.validate()
    _check(dispatch_result.is_fast_path(), "shared optimizer dispatch should be fast path for AdamW")
    _check(
        dispatch_result.optimizer_backend == String("fused_adamw_multitensor"),
        "shared optimizer dispatch should report AdamW backend",
    )

    var p2 = Tensor.from_host([Float32(1.0), Float32(2.0), Float32(3.0), Float32(4.0)], shape.copy(), STDtype.F32, ctx)
    var g2 = Tensor.from_host([Float32(10.0), Float32(10.0), Float32(10.0), Float32(10.0)], shape.copy(), STDtype.F32, ctx)
    var m2 = Tensor.from_host(_zeros(4), shape.copy(), STDtype.F32, ctx)
    var v2 = Tensor.from_host(_zeros(4), shape.copy(), STDtype.F32, ctx)
    var trainables_arena = DeviceTrainableSet()
    trainables_arena.append(String("adapter.2.weight"), TArc(p2^), String("rank=2"))
    var grads_arena = DeviceGradSet()
    grads_arena.append(String("adapter.2.weight"), TArc(g2^), String("loss=mse"))
    var state_arena = DeviceAdamWState()
    state_arena.append(TArc(m2^), TArc(v2^))
    var arena = TrainingArena(ctx, 4096, 1)
    var arena_mark = arena.mark(TRAINING_ARENA_PHASE_OPTIMIZER)
    var arena_result = device_optimizer_train_step_update_with_arena(
        trainables_arena,
        grads_arena,
        state_arena,
        _adamw_config(),
        Float32(1.25),
        1,
        arena,
        ctx,
    )
    arena_result.validate()
    _check(arena_result.is_fast_path(), "arena-backed AdamW result should be device fast path")
    _check(
        arena_result.optimizer_backend == String("fused_adamw_multitensor-arena-grad-stats-adamw-descriptors"),
        "arena-backed AdamW backend label",
    )
    var arena_stats = arena.stats()
    _check(arena_stats.allocation_count == 9, "arena optimizer should allocate grad-stats and AdamW descriptor scratch")
    _check(arena_stats.host_device_transfer_count == 9, "arena grad stats plus AdamW descriptor transfer accounting")
    _check(arena_stats.sync_count == 2, "arena grad stats plus AdamW sync accounting")
    _check(arena_stats.scalar_sync_count == 1, "arena grad stats scalar sync reason")
    _check(arena_stats.optimizer_sync_count == 1, "arena AdamW optimizer sync reason")
    _check(arena_result.sync_count == arena_stats.sync_count, "arena result sync count matches arena")
    arena.rewind(arena_mark)
    _check(arena.stats().current_used_bytes == 0, "arena grad stats rewind")

    var bad_grad = Float32(0.0) / Float32(0.0)
    var bad = Tensor.from_host([bad_grad, Float32(1.0), Float32(2.0), Float32(3.0)], shape.copy(), STDtype.F32, ctx)
    var bad_grads = DeviceGradSet()
    bad_grads.append(String("adapter.0.weight"), TArc(bad^), String("loss=mse"))
    var stats = device_grad_stats(bad_grads, ctx)
    _check(stats.nonfinite_count == 1, "device grad stats should count NaN grads")
    _check(stats.full_tensor_readback_count == 0, "nonfinite stats should not read full grads")

    var slow = host_grad_compat_result(Float32(0.5), Float32(1.0), 1, String("parity dump"))
    _check(not slow.is_fast_path(), "host grad compatibility must be slow path")
    print("PASS: device train-step ABI updates params with folded clip")
