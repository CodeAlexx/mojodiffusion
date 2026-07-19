# training/device_train_step.mojo — shared device-resident train-step ABI.
#
# New product trainers should return device grads through this contract and feed
# them directly into shared optimizers. Host grad extraction remains available
# for parity/debug/checkpoint surfaces, but any result carrying full tensor
# readbacks is marked as the compatibility slow path.

from std.gpu.host import DeviceContext
from std.memory import ArcPointer
from std.time import perf_counter_ns

from serenitymojo.io.dtype import STDtype
from serenitymojo.tensor import Tensor
from serenitymojo.training.fused_adamw_multitensor import (
    fused_adamw_step,
    fused_adamw_step_with_arena,
)
from serenitymojo.training.on_device_global_norm import (
    DeviceGradStats,
    on_device_grad_stats,
    on_device_grad_stats_with_arena,
)
from serenitymojo.training.perf_record import (
    PERF_FAST_PATH_DEVICE,
    PERF_FAST_PATH_HOST_GRAD_COMPAT,
    TrainingPhaseTimings,
)
from serenitymojo.training.training_arena import TrainingArena
from serenitymojo.training.train_config import (
    TRAIN_OPTIMIZER_ADAFACTOR,
    TRAIN_OPTIMIZER_ADAMW,
    TRAIN_OPTIMIZER_ADAMW_8BIT,
    TRAIN_OPTIMIZER_AUTOMAGIC3,
    TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW,
)


comptime TArc = ArcPointer[Tensor]


def _supported_device_storage(dt: STDtype) -> Bool:
    return dt == STDtype.F32 or dt == STDtype.BF16 or dt == STDtype.F16


struct DeviceTrainableSet(Movable, Writable):
    var keys: List[String]
    var params: List[TArc]
    var metadata: List[String]

    def __init__(out self):
        self.keys = List[String]()
        self.params = List[TArc]()
        self.metadata = List[String]()

    def append(mut self, key: String, param: TArc, metadata: String) raises:
        if key == String(""):
            raise Error("DeviceTrainableSet.append: empty key")
        if param[].numel() <= 0:
            raise Error(String("DeviceTrainableSet.append: empty tensor for ") + key)
        if not _supported_device_storage(param[].dtype()):
            raise Error(
                String("DeviceTrainableSet.append: unsupported dtype ")
                + param[].dtype().name()
                + String(" for ")
                + key
            )
        self.keys.append(key.copy())
        self.params.append(param)
        self.metadata.append(metadata.copy())

    def count(self) -> Int:
        return len(self.params)

    def total_numel(self) -> Int:
        var total = 0
        for i in range(len(self.params)):
            total += self.params[i][].numel()
        return total

    def dtype_summary(self) -> String:
        if len(self.params) == 0:
            return String("empty")
        var first = self.params[0][].dtype()
        for i in range(1, len(self.params)):
            if self.params[i][].dtype() != first:
                return String("mixed")
        return first.name()

    def validate(self) raises:
        if len(self.keys) != len(self.params) or len(self.metadata) != len(self.params):
            raise Error("DeviceTrainableSet: parallel list length mismatch")
        if len(self.params) == 0:
            raise Error("DeviceTrainableSet: empty trainable set")
        for i in range(len(self.params)):
            if self.keys[i] == String(""):
                raise Error("DeviceTrainableSet: empty key")
            if self.params[i][].numel() <= 0:
                raise Error(String("DeviceTrainableSet: empty tensor at ") + self.keys[i])
            if not _supported_device_storage(self.params[i][].dtype()):
                raise Error(
                    String("DeviceTrainableSet: unsupported dtype at ")
                    + self.keys[i]
                )

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "DeviceTrainableSet(count=",
            self.count(),
            ", dtype=",
            self.dtype_summary(),
            ", total_numel=",
            self.total_numel(),
            ")",
        )


struct DeviceGradSet(Movable, Writable):
    var keys: List[String]
    var grads: List[TArc]
    var metadata: List[String]

    def __init__(out self):
        self.keys = List[String]()
        self.grads = List[TArc]()
        self.metadata = List[String]()

    def append(mut self, key: String, grad: TArc, metadata: String) raises:
        if key == String(""):
            raise Error("DeviceGradSet.append: empty key")
        if grad[].numel() <= 0:
            raise Error(String("DeviceGradSet.append: empty grad for ") + key)
        if not _supported_device_storage(grad[].dtype()):
            raise Error(
                String("DeviceGradSet.append: unsupported dtype ")
                + grad[].dtype().name()
                + String(" for ")
                + key
            )
        self.keys.append(key.copy())
        self.grads.append(grad)
        self.metadata.append(metadata.copy())

    def count(self) -> Int:
        return len(self.grads)

    def total_numel(self) -> Int:
        var total = 0
        for i in range(len(self.grads)):
            total += self.grads[i][].numel()
        return total

    def dtype_summary(self) -> String:
        if len(self.grads) == 0:
            return String("empty")
        var first = self.grads[0][].dtype()
        for i in range(1, len(self.grads)):
            if self.grads[i][].dtype() != first:
                return String("mixed")
        return first.name()

    def validate(self) raises:
        if len(self.keys) != len(self.grads) or len(self.metadata) != len(self.grads):
            raise Error("DeviceGradSet: parallel list length mismatch")
        if len(self.grads) == 0:
            raise Error("DeviceGradSet: empty grad set")
        for i in range(len(self.grads)):
            if self.keys[i] == String(""):
                raise Error("DeviceGradSet: empty key")
            if self.grads[i][].numel() <= 0:
                raise Error(String("DeviceGradSet: empty tensor at ") + self.keys[i])
            if not _supported_device_storage(self.grads[i][].dtype()):
                raise Error(String("DeviceGradSet: unsupported dtype at ") + self.keys[i])

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "DeviceGradSet(count=",
            self.count(),
            ", dtype=",
            self.dtype_summary(),
            ", total_numel=",
            self.total_numel(),
            ")",
        )


struct DeviceAdamWState(Movable, Writable):
    var m: List[TArc]
    var v: List[TArc]

    def __init__(out self):
        self.m = List[TArc]()
        self.v = List[TArc]()

    def append(mut self, m: TArc, v: TArc) raises:
        if m[].dtype() != STDtype.F32 or v[].dtype() != STDtype.F32:
            raise Error("DeviceAdamWState.append: m/v must be F32 optimizer state")
        if m[].numel() != v[].numel():
            raise Error("DeviceAdamWState.append: m/v numel mismatch")
        self.m.append(m)
        self.v.append(v)

    def count(self) -> Int:
        return len(self.m)

    def validate_for(self, trainables: DeviceTrainableSet) raises:
        if len(self.m) != trainables.count() or len(self.v) != trainables.count():
            raise Error("DeviceAdamWState: state count != trainable count")
        for i in range(trainables.count()):
            if self.m[i][].dtype() != STDtype.F32 or self.v[i][].dtype() != STDtype.F32:
                raise Error(String("DeviceAdamWState: m/v must be F32 at ") + trainables.keys[i])
            if self.m[i][].numel() != trainables.params[i][].numel():
                raise Error(String("DeviceAdamWState: m numel mismatch at ") + trainables.keys[i])
            if self.v[i][].numel() != trainables.params[i][].numel():
                raise Error(String("DeviceAdamWState: v numel mismatch at ") + trainables.keys[i])

    def write_to(self, mut writer: Some[Writer]):
        writer.write("DeviceAdamWState(count=", self.count(), ")")


def device_optimizer_backend_name(optimizer: Int) -> String:
    if optimizer == TRAIN_OPTIMIZER_ADAMW:
        return String("fused_adamw_multitensor")
    if optimizer == TRAIN_OPTIMIZER_ADAMW_8BIT:
        return String("device-adamw8bit-unported")
    if optimizer == TRAIN_OPTIMIZER_AUTOMAGIC3:
        return String("device-automagic3-host-grad-compat")
    if optimizer == TRAIN_OPTIMIZER_ADAFACTOR:
        return String("device-adafactor-unported")
    if optimizer == TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW:
        return String("device-schedulefree-adamw-unported")
    return String("device-optimizer-unregistered")


def device_optimizer_tag_registered(optimizer: Int) -> Bool:
    return (
        optimizer == TRAIN_OPTIMIZER_ADAMW
        or optimizer == TRAIN_OPTIMIZER_ADAMW_8BIT
        or optimizer == TRAIN_OPTIMIZER_AUTOMAGIC3
        or optimizer == TRAIN_OPTIMIZER_ADAFACTOR
        or optimizer == TRAIN_OPTIMIZER_SCHEDULE_FREE_ADAMW
    )


@fieldwise_init
struct DeviceOptimizerConfig(Copyable, Movable, Writable):
    var optimizer: Int
    var lr: Float32
    var beta1: Float32
    var beta2: Float32
    var eps: Float32
    var weight_decay: Float32
    var max_grad_norm: Float32
    var optimizer_eps2: Float32
    var optimizer_clip_threshold: Float32
    var optimizer_decay_rate: Float32

    def validate(self) raises:
        if not device_optimizer_tag_registered(self.optimizer):
            raise Error(
                String("DeviceOptimizerConfig: optimizer tag ")
                + String(self.optimizer)
                + String(" is not registered in the shared device optimizer interface")
            )
        if self.lr <= 0.0:
            raise Error("DeviceOptimizerConfig: lr must be positive for a device update")
        if self.beta1 < 0.0 or self.beta1 >= 1.0:
            raise Error("DeviceOptimizerConfig: beta1 must be in [0, 1)")
        if self.beta2 < 0.0 or self.beta2 >= 1.0:
            raise Error("DeviceOptimizerConfig: beta2 must be in [0, 1)")
        if self.eps <= 0.0:
            raise Error("DeviceOptimizerConfig: eps must be positive")
        if self.weight_decay < 0.0:
            raise Error("DeviceOptimizerConfig: weight_decay must be nonnegative")
        if self.max_grad_norm < 0.0:
            raise Error("DeviceOptimizerConfig: max_grad_norm must be nonnegative")
        if self.optimizer_eps2 < 0.0:
            raise Error("DeviceOptimizerConfig: optimizer_eps2 must be nonnegative")
        if self.optimizer_clip_threshold < 0.0:
            raise Error("DeviceOptimizerConfig: optimizer_clip_threshold must be nonnegative")

    def backend_name(self) -> String:
        return device_optimizer_backend_name(self.optimizer)


def validate_device_optimizer_supported_for_fast_path(
    config: DeviceOptimizerConfig
) raises:
    config.validate()
    if config.optimizer == TRAIN_OPTIMIZER_ADAMW:
        return
    raise Error(
        String("device optimizer interface registered for ")
        + config.backend_name()
        + String(" but full device fast path is not ported; use host-grad-compatible or host levers path")
    )


@fieldwise_init
struct TrainStepDeviceResult(Copyable, Movable, Writable):
    var loss: Float32
    var grad_norm: Float32
    var clip_scale: Float32
    var phases: TrainingPhaseTimings
    var scalar_readback_count: Int
    var full_tensor_readback_count: Int
    var sync_count: Int
    var nonfinite_grad_count: Int
    var fast_path_kind: Int
    var optimizer_backend: String
    var debug_handle: String

    def is_fast_path(self) -> Bool:
        return (
            self.fast_path_kind == PERF_FAST_PATH_DEVICE
            and self.full_tensor_readback_count == 0
            and self.nonfinite_grad_count == 0
        )

    def validate(self) raises:
        if self.grad_norm < 0.0:
            raise Error("TrainStepDeviceResult: grad_norm must be nonnegative")
        if self.clip_scale < 0.0:
            raise Error("TrainStepDeviceResult: clip_scale must be nonnegative")
        if self.scalar_readback_count < 0 or self.full_tensor_readback_count < 0:
            raise Error("TrainStepDeviceResult: readback counts must be nonnegative")
        if self.sync_count < 0:
            raise Error("TrainStepDeviceResult: sync count must be nonnegative")
        if self.nonfinite_grad_count < 0:
            raise Error("TrainStepDeviceResult: nonfinite grad count must be nonnegative")
        if self.fast_path_kind != PERF_FAST_PATH_DEVICE and self.fast_path_kind != PERF_FAST_PATH_HOST_GRAD_COMPAT:
            raise Error("TrainStepDeviceResult: invalid fast path kind")
        if self.fast_path_kind == PERF_FAST_PATH_DEVICE and self.full_tensor_readback_count != 0:
            raise Error("TrainStepDeviceResult: device fast path cannot include full tensor readbacks")

    def write_to(self, mut writer: Some[Writer]):
        writer.write(
            "TrainStepDeviceResult(loss=",
            self.loss,
            ", grad_norm=",
            self.grad_norm,
            ", clip_scale=",
            self.clip_scale,
            ", scalar_readbacks=",
            self.scalar_readback_count,
            ", full_tensor_readbacks=",
            self.full_tensor_readback_count,
            ", syncs=",
            self.sync_count,
            ", nonfinite_grads=",
            self.nonfinite_grad_count,
            ", backend=",
            self.optimizer_backend,
            ")",
        )


def validate_device_grad_keys(
    trainables: DeviceTrainableSet, grads: DeviceGradSet
) raises:
    trainables.validate()
    grads.validate()
    if trainables.count() != grads.count():
        raise Error("device train-step ABI: trainable/grad count mismatch")
    for i in range(trainables.count()):
        if trainables.keys[i] != grads.keys[i]:
            raise Error(
                String("device train-step ABI: key mismatch at ")
                + String(i)
                + String(" trainable=")
                + trainables.keys[i]
                + String(" grad=")
                + grads.keys[i]
            )
        if trainables.params[i][].numel() != grads.grads[i][].numel():
            raise Error(String("device train-step ABI: numel mismatch at ") + trainables.keys[i])


def device_clip_scale(grad_norm: Float32, max_norm: Float32) -> Float32:
    if max_norm <= 0.0:
        return Float32(1.0)
    if grad_norm > max_norm and grad_norm > 0.0:
        return max_norm / grad_norm
    return Float32(1.0)


def device_grad_stats(grads: DeviceGradSet, ctx: DeviceContext) raises -> DeviceGradStats:
    grads.validate()
    return on_device_grad_stats(grads.grads, ctx)


def device_grad_stats_with_arena(
    grads: DeviceGradSet, mut arena: TrainingArena, ctx: DeviceContext
) raises -> DeviceGradStats:
    grads.validate()
    return on_device_grad_stats_with_arena(grads.grads, arena, ctx)


def device_grad_norm(grads: DeviceGradSet, ctx: DeviceContext) raises -> Float32:
    return device_grad_stats(grads, ctx).grad_norm


def device_grad_norm_with_arena(
    grads: DeviceGradSet, mut arena: TrainingArena, ctx: DeviceContext
) raises -> Float32:
    return device_grad_stats_with_arena(grads, arena, ctx).grad_norm


def device_adamw_train_step_update(
    trainables: DeviceTrainableSet,
    grads: DeviceGradSet,
    state: DeviceAdamWState,
    loss: Float32,
    t: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    max_grad_norm: Float32,
    ctx: DeviceContext,
) raises -> TrainStepDeviceResult:
    """Validate device-resident trainables/grads, compute global norm on device,
    fold clip_scale into fused AdamW, and update params/m/v in place."""
    validate_device_grad_keys(trainables, grads)
    state.validate_for(trainables)
    var norm_t0 = perf_counter_ns()
    var stats = device_grad_stats(grads, ctx)
    var gn = stats.grad_norm
    if stats.nonfinite_count != 0:
        raise Error("device_adamw_train_step_update: nonfinite device grads")
    var norm_t1 = perf_counter_ns()
    var clip = device_clip_scale(gn, max_grad_norm)
    fused_adamw_step(
        trainables.params,
        grads.grads,
        state.m,
        state.v,
        t,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        ctx,
        clip,
    )
    var opt_t1 = perf_counter_ns()
    var phases = TrainingPhaseTimings(
        0.0,
        0.0,
        0.0,
        Float64(norm_t1 - norm_t0) / 1.0e9,
        0.0,
        Float64(opt_t1 - norm_t1) / 1.0e9,
        0.0,
        0.0,
    )
    var result = TrainStepDeviceResult(
        loss,
        gn,
        clip,
        phases^,
        stats.scalar_readback_count,
        0,
        stats.sync_count + 1,  # grad stats and fused_adamw_step synchronize today
        stats.nonfinite_count,
        PERF_FAST_PATH_DEVICE,
        String("fused_adamw_multitensor"),
        String(""),
    )
    result.validate()
    return result^


def device_adamw_train_step_update_with_arena(
    trainables: DeviceTrainableSet,
    grads: DeviceGradSet,
    state: DeviceAdamWState,
    loss: Float32,
    t: Int,
    lr: Float32,
    beta1: Float32,
    beta2: Float32,
    eps: Float32,
    weight_decay: Float32,
    max_grad_norm: Float32,
    mut arena: TrainingArena,
    ctx: DeviceContext,
) raises -> TrainStepDeviceResult:
    """AdamW device update using arena-backed grad stats scratch."""
    validate_device_grad_keys(trainables, grads)
    state.validate_for(trainables)
    var arena_before = arena.stats()
    var norm_t0 = perf_counter_ns()
    var stats = device_grad_stats_with_arena(grads, arena, ctx)
    var gn = stats.grad_norm
    if stats.nonfinite_count != 0:
        raise Error("device_adamw_train_step_update_with_arena: nonfinite device grads")
    var norm_t1 = perf_counter_ns()
    var clip = device_clip_scale(gn, max_grad_norm)
    fused_adamw_step_with_arena(
        trainables.params,
        grads.grads,
        state.m,
        state.v,
        t,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        arena,
        ctx,
        clip,
    )
    var opt_t1 = perf_counter_ns()
    var arena_after = arena.stats()
    var sync_delta = arena_after.sync_count - arena_before.sync_count
    if sync_delta < stats.sync_count:
        raise Error("device_adamw_train_step_update_with_arena: arena sync accounting regressed")
    var phases = TrainingPhaseTimings(
        0.0,
        0.0,
        0.0,
        Float64(norm_t1 - norm_t0) / 1.0e9,
        0.0,
        Float64(opt_t1 - norm_t1) / 1.0e9,
        0.0,
        0.0,
    )
    var result = TrainStepDeviceResult(
        loss,
        gn,
        clip,
        phases^,
        stats.scalar_readback_count,
        0,
        sync_delta,
        stats.nonfinite_count,
        PERF_FAST_PATH_DEVICE,
        String("fused_adamw_multitensor-arena-grad-stats-adamw-descriptors"),
        String(""),
    )
    result.validate()
    return result^


def device_optimizer_train_step_update(
    trainables: DeviceTrainableSet,
    grads: DeviceGradSet,
    state: DeviceAdamWState,
    config: DeviceOptimizerConfig,
    loss: Float32,
    t: Int,
    ctx: DeviceContext,
) raises -> TrainStepDeviceResult:
    """Shared device optimizer dispatch.

    AdamW is the only GPU fast path currently wired. AdamW8bit, Adafactor, and
    schedule-free AdamW are registered here so product trainers fail loud
    instead of falling back to model-owned optimizer plumbing. Automagic3 has a
    separate host-grad-compatible device optimizer wrapper, but still fails this
    device-fast dispatch until grads and param mirrors stay device-resident.
    """
    validate_device_optimizer_supported_for_fast_path(config)
    if config.optimizer == TRAIN_OPTIMIZER_ADAMW:
        var result = device_adamw_train_step_update(
            trainables,
            grads,
            state,
            loss,
            t,
            config.lr,
            config.beta1,
            config.beta2,
            config.eps,
            config.weight_decay,
            config.max_grad_norm,
            ctx,
        )
        return result^
    raise Error("device_optimizer_train_step_update: unreachable optimizer dispatch")


def device_optimizer_train_step_update_with_arena(
    trainables: DeviceTrainableSet,
    grads: DeviceGradSet,
    state: DeviceAdamWState,
    config: DeviceOptimizerConfig,
    loss: Float32,
    t: Int,
    mut arena: TrainingArena,
    ctx: DeviceContext,
) raises -> TrainStepDeviceResult:
    """Shared device optimizer dispatch using arena-backed grad stats scratch."""
    validate_device_optimizer_supported_for_fast_path(config)
    if config.optimizer == TRAIN_OPTIMIZER_ADAMW:
        var result = device_adamw_train_step_update_with_arena(
            trainables,
            grads,
            state,
            loss,
            t,
            config.lr,
            config.beta1,
            config.beta2,
            config.eps,
            config.weight_decay,
            config.max_grad_norm,
            arena,
            ctx,
        )
        return result^
    raise Error("device_optimizer_train_step_update_with_arena: unreachable optimizer dispatch")


def host_grad_compat_result(
    loss: Float32,
    grad_norm: Float32,
    full_tensor_readbacks: Int,
    reason: String,
) raises -> TrainStepDeviceResult:
    var result = TrainStepDeviceResult(
        loss,
        grad_norm,
        Float32(1.0),
        TrainingPhaseTimings(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        0,
        full_tensor_readbacks,
        0,
        0,
        PERF_FAST_PATH_HOST_GRAD_COMPAT,
        String("host-grad-compat"),
        reason.copy(),
    )
    result.validate()
    return result^
