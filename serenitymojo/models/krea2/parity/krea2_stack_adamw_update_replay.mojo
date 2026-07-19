# Krea2 reduced-depth ai-toolkit stack AdamW update replay.
#
# Consumes the NBLOCKS=4 ai-toolkit SingleStreamDiT oracle dump from
# krea2_stack_oracle.py and replays one AdamW step over the reduced-depth block
# LoRA tensors through the shared device train-step ABI. This is optimizer
# parity for the already-gated block-stack gradient surface; it is not real-cache
# product parity, full 28-block parity, txtfusion parity, or convergence evidence.

from std.builtin.dtype import DType
from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import ArcPointer

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.tensor import Tensor
from serenitymojo.training.device_train_step import (
    DeviceAdamWState,
    DeviceGradSet,
    DeviceTrainableSet,
    device_adamw_train_step_update,
)


comptime TArc = ArcPointer[Tensor]
comptime ORACLE = "/home/alex/mojodiffusion/serenitymojo/models/krea2/parity/krea2_stack_oracle.safetensors"
comptime NBLOCKS = 4
comptime SLOTS_PER_BLOCK = 8
comptime LR = Float32(1.0e-3)
comptime BETA1 = Float32(0.9)
comptime BETA2 = Float32(0.999)
comptime EPS = Float32(1.0e-8)
comptime WEIGHT_DECAY = Float32(0.01)
comptime EXPECTED_COUNT = 64
comptime EXPECTED_NUMEL = 3833856
comptime EXPECTED_NONZERO_UPDATE = 3833856
comptime MAX_PARAM_ABS_TOL = Float32(1.0e-5)
comptime MAX_STATE_ABS_TOL = Float32(1.0e-6)


@fieldwise_init
struct ReplayStats(Copyable, Movable):
    var numel: Int
    var nonzero_update: Int
    var nonzero_param_error: Int
    var nonzero_state_error: Int
    var max_param_abs: Float32
    var max_state_abs: Float32
    var param_l2_sumsq: Float64
    var state_l2_sumsq: Float64


def _require(ok: Bool, msg: String) raises:
    if not ok:
        raise Error(msg)


def _abs(x: Float32) -> Float32:
    if x < Float32(0.0):
        return -x
    return x


def _empty_stats() -> ReplayStats:
    return ReplayStats(
        numel=0,
        nonzero_update=0,
        nonzero_param_error=0,
        nonzero_state_error=0,
        max_param_abs=Float32(0.0),
        max_state_abs=Float32(0.0),
        param_l2_sumsq=Float64(0.0),
        state_l2_sumsq=Float64(0.0),
    )


def _add_stats(mut total: ReplayStats, part: ReplayStats):
    total.numel += part.numel
    total.nonzero_update += part.nonzero_update
    total.nonzero_param_error += part.nonzero_param_error
    total.nonzero_state_error += part.nonzero_state_error
    total.param_l2_sumsq += part.param_l2_sumsq
    total.state_l2_sumsq += part.state_l2_sumsq
    if part.max_param_abs > total.max_param_abs:
        total.max_param_abs = part.max_param_abs
    if part.max_state_abs > total.max_state_abs:
        total.max_state_abs = part.max_state_abs


def _same_shape(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def _require_f32_tensor(st: SafeTensors, key: String) raises -> Int:
    _require(key in st.tensors, String("missing Krea2 oracle tensor ") + key)
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.F32, String("expected F32 tensor ") + key)
    _require(
        info.size % 4 == 0,
        String("F32 tensor byte size is not divisible by 4: ") + key,
    )
    return info.size // 4


def _require_same_shape(
    st: SafeTensors, key_a: String, key_b: String
) raises:
    var a = st.tensor_info(key_a)
    var b = st.tensor_info(key_b)
    _require(
        _same_shape(a.shape, b.shape),
        String("shape mismatch ") + key_a + String(" vs ") + key_b,
    )


def _load_f32_values(st: SafeTensors, key: String) raises -> List[Float32]:
    var n = _require_f32_tensor(st, key)
    var bytes = st.tensor_bytes(key)
    var ptr = bytes.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
    for i in range(n):
        out.append(ptr[i])
    return out^


def _zeros(n: Int) -> List[Float32]:
    var out = List[Float32]()
    for _ in range(n):
        out.append(Float32(0.0))
    return out^


def _tensor_from_loaded(
    loaded: SafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var info = loaded.tensor_info(name)
    var span = loaded.tensor_bytes(name)
    var nbytes = len(span)
    var host = ctx.enqueue_create_host_buffer[DType.uint8](nbytes)
    var hp = host.unsafe_ptr()
    for i in range(nbytes):
        hp[i] = span[i]
    var dev = ctx.enqueue_create_buffer[DType.uint8](nbytes)
    ctx.enqueue_copy(dst_buf=dev, src_buf=host)
    ctx.synchronize()
    return Tensor(dev^, info.shape.copy(), info.dtype)


def _slot_name(s: Int) -> String:
    if s == 0:
        return String("wq")
    if s == 1:
        return String("wk")
    if s == 2:
        return String("wv")
    if s == 3:
        return String("gate")
    if s == 4:
        return String("wo")
    if s == 5:
        return String("mlp_gate")
    if s == 6:
        return String("mlp_up")
    return String("mlp_down")


def _adapter_names() -> List[String]:
    var names = List[String]()
    for bi in range(NBLOCKS):
        for s in range(SLOTS_PER_BLOCK):
            var pfx = String("blk") + String(bi) + String(".") + _slot_name(s) + String(".")
            names.append(pfx + String("A"))
            names.append(pfx + String("B"))
    return names^


def _kadamw(name: String, suffix: String) -> String:
    return String("kadamw.") + name + String(".") + suffix


def _append_device_inputs(
    ref st: SafeTensors,
    name: String,
    mut trainables: DeviceTrainableSet,
    mut grads: DeviceGradSet,
    mut state: DeviceAdamWState,
    ctx: DeviceContext,
) raises -> Int:
    var k_before = _kadamw(name, String("before"))
    var k_grad = _kadamw(name, String("grad"))
    var k_after = _kadamw(name, String("after"))
    var k_m = _kadamw(name, String("exp_avg"))
    var k_v = _kadamw(name, String("exp_avg_sq"))

    var n = _require_f32_tensor(st, k_before)
    _require(_require_f32_tensor(st, k_grad) == n, String("grad numel mismatch for ") + name)
    _require(_require_f32_tensor(st, k_after) == n, String("after numel mismatch for ") + name)
    _require(_require_f32_tensor(st, k_m) == n, String("exp_avg numel mismatch for ") + name)
    _require(_require_f32_tensor(st, k_v) == n, String("exp_avg_sq numel mismatch for ") + name)
    _require_same_shape(st, k_before, k_grad)
    _require_same_shape(st, k_before, k_after)
    _require_same_shape(st, k_before, k_m)
    _require_same_shape(st, k_before, k_v)

    var info = st.tensor_info(k_before)
    trainables.append(
        name.copy(),
        TArc(_tensor_from_loaded(st, k_before, ctx)),
        String("krea2-reduced-depth-ai-toolkit-adapter-before"),
    )
    grads.append(
        name.copy(),
        TArc(_tensor_from_loaded(st, k_grad, ctx)),
        String("krea2-reduced-depth-ai-toolkit-adapter-grad"),
    )
    state.append(
        TArc(Tensor.from_host(_zeros(n), info.shape.copy(), STDtype.F32, ctx)),
        TArc(Tensor.from_host(_zeros(n), info.shape.copy(), STDtype.F32, ctx)),
    )
    return n


def _compare_one(
    ref st: SafeTensors,
    name: String,
    result_param: Tensor,
    result_m: Tensor,
    result_v: Tensor,
    ctx: DeviceContext,
) raises -> ReplayStats:
    var before = _load_f32_values(st, _kadamw(name, String("before")))
    var after = _load_f32_values(st, _kadamw(name, String("after")))
    var expected_m = _load_f32_values(st, _kadamw(name, String("exp_avg")))
    var expected_v = _load_f32_values(st, _kadamw(name, String("exp_avg_sq")))
    var got_p = result_param.to_host(ctx)
    var got_m = result_m.to_host(ctx)
    var got_v = result_v.to_host(ctx)
    var n = len(after)
    _require(len(before) == n, String("before numel mismatch for ") + name)
    _require(len(expected_m) == n, String("exp_avg numel mismatch for ") + name)
    _require(len(expected_v) == n, String("exp_avg_sq numel mismatch for ") + name)
    _require(len(got_p) == n, String("result param numel mismatch for ") + name)
    _require(len(got_m) == n, String("result exp_avg numel mismatch for ") + name)
    _require(len(got_v) == n, String("result exp_avg_sq numel mismatch for ") + name)

    var stats = _empty_stats()
    stats.numel = n
    for i in range(n):
        if after[i] != before[i]:
            stats.nonzero_update += 1

        var p_err = got_p[i] - after[i]
        var p_abs = _abs(p_err)
        if p_err != Float32(0.0):
            stats.nonzero_param_error += 1
        if p_abs > stats.max_param_abs:
            stats.max_param_abs = p_abs
        var pe64 = Float64(p_err)
        stats.param_l2_sumsq += pe64 * pe64

        var m_err = got_m[i] - expected_m[i]
        var v_err = got_v[i] - expected_v[i]
        var m_abs = _abs(m_err)
        var v_abs = _abs(v_err)
        if m_err != Float32(0.0):
            stats.nonzero_state_error += 1
        if v_err != Float32(0.0):
            stats.nonzero_state_error += 1
        if m_abs > stats.max_state_abs:
            stats.max_state_abs = m_abs
        if v_abs > stats.max_state_abs:
            stats.max_state_abs = v_abs
        var me64 = Float64(m_err)
        var ve64 = Float64(v_err)
        stats.state_l2_sumsq += me64 * me64 + ve64 * ve64

    _require(
        stats.max_param_abs <= MAX_PARAM_ABS_TOL,
        String("Krea2 shared device AdamW param max_abs too high for ")
        + name
        + String(": ")
        + String(stats.max_param_abs),
    )
    _require(
        stats.max_state_abs <= MAX_STATE_ABS_TOL,
        String("Krea2 shared device AdamW state max_abs too high for ")
        + name
        + String(": ")
        + String(stats.max_state_abs),
    )
    return stats^


def main() raises:
    var ctx = DeviceContext()
    var st = SafeTensors.open(String(ORACLE))
    var names = _adapter_names()
    _require(len(names) == EXPECTED_COUNT, String("Krea2 AdamW replay name count mismatch"))

    var trainables = DeviceTrainableSet()
    var grads = DeviceGradSet()
    var state = DeviceAdamWState()
    var loaded_numel = 0
    for i in range(len(names)):
        loaded_numel += _append_device_inputs(
            st, names[i], trainables, grads, state, ctx
        )
    _require(loaded_numel == EXPECTED_NUMEL, String("Krea2 AdamW replay numel mismatch"))

    var update_result = device_adamw_train_step_update(
        trainables,
        grads,
        state,
        Float32(0.0),
        1,
        LR,
        BETA1,
        BETA2,
        EPS,
        WEIGHT_DECAY,
        Float32(0.0),
        ctx,
    )
    _require(
        update_result.clip_scale == Float32(1.0),
        String("Krea2 shared device AdamW replay unexpectedly clipped grads"),
    )
    _require(
        update_result.optimizer_backend == String("fused_adamw_multitensor"),
        String("Krea2 shared device AdamW replay did not use fused optimizer backend"),
    )

    var total = _empty_stats()
    for i in range(len(names)):
        var stats = _compare_one(
            st,
            names[i],
            trainables.params[i][],
            state.m[i][],
            state.v[i][],
            ctx,
        )
        _add_stats(total, stats)

    _require(
        total.numel == EXPECTED_NUMEL,
        String("Krea2 shared device AdamW replay compare numel mismatch"),
    )
    _require(
        total.nonzero_update == EXPECTED_NONZERO_UPDATE,
        String("Krea2 shared device AdamW replay nonzero update count mismatch"),
    )

    var param_l2 = sqrt(total.param_l2_sumsq)
    var state_l2 = sqrt(total.state_l2_sumsq)
    print(
        "[krea2-stack-adamw-update-mojo] reduced_depth_shared_device_abi_replay PASS tensors=",
        len(names),
        " numel=",
        total.numel,
        " nonzero_update=",
        total.nonzero_update,
        " nonzero_param_error=",
        total.nonzero_param_error,
        " max_param_abs=",
        total.max_param_abs,
        " param_l2=",
        param_l2,
        " nonzero_state_error=",
        total.nonzero_state_error,
        " max_state_abs=",
        total.max_state_abs,
        " state_l2=",
        state_l2,
        " grad_norm=",
        update_result.grad_norm,
        " clip_scale=",
        update_result.clip_scale,
        " syncs=",
        update_result.sync_count,
    )
    print(
        "[krea2-stack-adamw-update-mojo] scope=reduced-depth ai-toolkit SingleStreamDiT block-stack gradient plus shared-device AdamW update replay; not real-cache, full-28-block, txtfusion, or convergence parity"
    )
