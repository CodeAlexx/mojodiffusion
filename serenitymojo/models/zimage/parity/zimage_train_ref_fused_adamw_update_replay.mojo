# ZImage OneTrainer train-ref fused AdamW update replay.
#
# Opens the real OneTrainer ZImage step000/step001 adapter dumps and replays
# all step001 AdamW adapter updates through the shared device train-step ABI.
# This is stronger than the scalar math bridge: params/grads/m/v are device
# tensors and the update runs through DeviceTrainableSet/DeviceGradSet/
# DeviceAdamWState before reaching the shared fused AdamW optimizer. It still
# does not run transformer forward/backward.

from std.math import sqrt
from std.pathlib import Path
from std.gpu.host import DeviceContext

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.tensor import Tensor
from serenitymojo.training.device_train_step import (
    DeviceAdamWState,
    DeviceGradSet,
    DeviceTrainableSet,
    TArc,
    device_adamw_train_step_update,
)


comptime META_JSON = "/home/alex/serenity-trainer/parity/zimage_train_ref_meta.json"
comptime STEP0_ADAPTERS = "/home/alex/serenity-trainer/parity/zimage_train_ref_step000_adapters.safetensors"
comptime STEP1_ADAPTERS = "/home/alex/serenity-trainer/parity/zimage_train_ref_step001_adapters.safetensors"

comptime LR = Float32(1.4999999999999998e-06)
comptime WEIGHT_DECAY = Float32(0.01)
comptime BETA1 = Float32(0.9)
comptime BETA2 = Float32(0.999)
comptime EPS = Float32(1.0e-08)
comptime MAX_PARAM_ABS_TOL = Float32(1.0e-7)
comptime MAX_STATE_ABS_TOL = Float32(1.0e-7)
comptime EXPECTED_COUNT = 420
comptime EXPECTED_NUMEL = 35020800
comptime EXPECTED_NONZERO_UPDATE = 19046400


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
    _require(key in st.tensors, String("missing tensor ") + key)
    var info = st.tensor_info(key)
    _require(info.dtype == STDtype.F32, String("expected F32 tensor ") + key)
    _require(
        info.size % 4 == 0,
        String("F32 tensor byte size is not divisible by 4: ") + key,
    )
    return info.size // 4


def _require_same_shape(
    st_a: SafeTensors, key_a: String, st_b: SafeTensors, key_b: String
) raises:
    var a = st_a.tensor_info(key_a)
    var b = st_b.tensor_info(key_b)
    _require(
        _same_shape(a.shape, b.shape),
        String("shape mismatch ") + key_a + String(" vs ") + key_b,
    )


def _require_same_values(
    st_a: SafeTensors, key_a: String, st_b: SafeTensors, key_b: String
) raises:
    var n = _require_f32_tensor(st_a, key_a)
    _require(
        _require_f32_tensor(st_b, key_b) == n,
        String("numel mismatch ") + key_a + String(" vs ") + key_b,
    )
    _require_same_shape(st_a, key_a, st_b, key_b)
    var a_bytes = st_a.tensor_bytes(key_a)
    var b_bytes = st_b.tensor_bytes(key_b)
    var ap = a_bytes.unsafe_ptr().bitcast[Float32]()
    var bp = b_bytes.unsafe_ptr().bitcast[Float32]()
    for i in range(n):
        if ap[i] != bp[i]:
            raise Error(
                String("phase payload mismatch ")
                + key_a
                + String(" vs ")
                + key_b
                + String(" at ")
                + String(i)
            )


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


def _load_f32_values(st: SafeTensors, key: String) raises -> List[Float32]:
    var n = _require_f32_tensor(st, key)
    var bytes = st.tensor_bytes(key)
    var ptr = bytes.unsafe_ptr().bitcast[Float32]()
    var out = List[Float32]()
    for i in range(n):
        out.append(ptr[i])
    return out^


def _sort_strings(mut xs: List[String]):
    for i in range(1, len(xs)):
        var key = xs[i]
        var j = i - 1
        while j >= 0 and xs[j] > key:
            xs[j + 1] = xs[j]
            j -= 1
        xs[j + 1] = key


def _adapter_names(ref step1: SafeTensors) raises -> List[String]:
    comptime PREFIX = "adapter_post_clip."
    var raw = step1.names()
    var names = List[String]()
    for i in range(len(raw)):
        if raw[i].startswith(String(PREFIX)):
            names.append(String(raw[i].removeprefix(String(PREFIX))))
    _sort_strings(names)
    _require(
        len(names) == EXPECTED_COUNT,
        String("adapter_post_clip tensor count mismatch: ") + String(len(names)),
    )
    return names^


def _append_device_inputs(
    ref step0: SafeTensors,
    ref step1: SafeTensors,
    name: String,
    mut trainables: DeviceTrainableSet,
    mut grads: DeviceGradSet,
    mut state: DeviceAdamWState,
    ctx: DeviceContext,
) raises -> Int:
    var k_g0 = String("adapter_post_clip_grad.") + name
    var k_post = String("adapter_post.") + name
    var k_before = String("adapter_post_clip.") + name
    var k_g1 = String("adapter_post_clip_grad.") + name
    var k_after = String("adapter_after.") + name

    var n = _require_f32_tensor(step0, k_g0)
    _require_same_values(step1, k_post, step1, k_before)
    _require(_require_f32_tensor(step1, k_before) == n, String("numel mismatch for ") + name)
    _require(_require_f32_tensor(step1, k_g1) == n, String("numel mismatch for ") + name)
    _require(_require_f32_tensor(step1, k_after) == n, String("numel mismatch for ") + name)
    _require_same_shape(step0, k_g0, step1, k_before)
    _require_same_shape(step1, k_before, step1, k_g1)
    _require_same_shape(step1, k_before, step1, k_after)

    var g0 = _load_f32_values(step0, k_g0)
    var m0 = List[Float32]()
    var v0 = List[Float32]()
    for i in range(n):
        var gi = g0[i]
        m0.append((Float32(1.0) - BETA1) * gi)
        v0.append((Float32(1.0) - BETA2) * gi * gi)

    var info = step0.tensor_info(k_g0)
    trainables.append(
        name.copy(),
        TArc(_tensor_from_loaded(step1, k_before, ctx)),
        String("zimage-step001-adapter_post_clip"),
    )
    grads.append(
        name.copy(),
        TArc(_tensor_from_loaded(step1, k_g1, ctx)),
        String("zimage-step001-adapter_post_clip_grad"),
    )
    state.append(
        TArc(Tensor.from_host(m0^, info.shape.copy(), STDtype.F32, ctx)),
        TArc(Tensor.from_host(v0^, info.shape.copy(), STDtype.F32, ctx)),
    )
    return n


def _compare_one(
    ref step0: SafeTensors,
    ref step1: SafeTensors,
    name: String,
    result_param: Tensor,
    result_m: Tensor,
    result_v: Tensor,
    ctx: DeviceContext,
) raises -> ReplayStats:
    var k_g0 = String("adapter_post_clip_grad.") + name
    var k_before = String("adapter_post_clip.") + name
    var k_g1 = String("adapter_post_clip_grad.") + name
    var k_after = String("adapter_after.") + name

    var n = _require_f32_tensor(step1, k_after)
    var before = _load_f32_values(step1, k_before)
    var g0 = _load_f32_values(step0, k_g0)
    var g1 = _load_f32_values(step1, k_g1)
    var after_bytes = step1.tensor_bytes(k_after)
    var afterp = after_bytes.unsafe_ptr().bitcast[Float32]()
    var got_p = result_param.to_host(ctx)
    var got_m = result_m.to_host(ctx)
    var got_v = result_v.to_host(ctx)

    _require(len(before) == n, String("before numel mismatch for ") + name)
    _require(len(g0) == n, String("g0 numel mismatch for ") + name)
    _require(len(g1) == n, String("g1 numel mismatch for ") + name)
    _require(len(got_p) == n, String("result param numel mismatch for ") + name)
    _require(len(got_m) == n, String("result m numel mismatch for ") + name)
    _require(len(got_v) == n, String("result v numel mismatch for ") + name)

    var stats = _empty_stats()
    stats.numel = n
    for i in range(n):
        var actual = afterp[i]
        if actual != before[i]:
            stats.nonzero_update += 1

        var p_err = got_p[i] - actual
        var p_abs = _abs(p_err)
        if p_err != Float32(0.0):
            stats.nonzero_param_error += 1
        if p_abs > stats.max_param_abs:
            stats.max_param_abs = p_abs
        var pe64 = Float64(p_err)
        stats.param_l2_sumsq += pe64 * pe64

        var m0 = (Float32(1.0) - BETA1) * g0[i]
        var v0 = (Float32(1.0) - BETA2) * g0[i] * g0[i]
        var expected_m = BETA1 * m0 + (Float32(1.0) - BETA1) * g1[i]
        var expected_v = BETA2 * v0 + (Float32(1.0) - BETA2) * g1[i] * g1[i]
        var m_err = got_m[i] - expected_m
        var v_err = got_v[i] - expected_v
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
        String("fused AdamW param max_abs too high for ")
        + name
        + String(": ")
        + String(stats.max_param_abs),
    )
    _require(
        stats.max_state_abs <= MAX_STATE_ABS_TOL,
        String("fused AdamW state max_abs too high for ")
        + name
        + String(": ")
        + String(stats.max_state_abs),
    )
    return stats^


def _check_meta_json() raises:
    var meta = Path(String(META_JSON)).read_text()
    _require(meta.byte_length() > 0, String("empty ZImage OneTrainer metadata JSON"))
    _require(
        meta.find(String("\"producer\": \"scripts/zimage_dump_train_ref.py\"")) >= 0,
        String("metadata producer mismatch"),
    )
    _require(meta.find(String("\"optimizer\": \"ADAMW\"")) >= 0, String("metadata optimizer mismatch"))
    _require(meta.find(String("\"lora_weight_dtype\": \"FLOAT_32\"")) >= 0, String("metadata LoRA dtype mismatch"))
    _require(meta.find(String("\"max_steps\": 2")) >= 0, String("metadata max_steps mismatch"))
    _require(meta.find(String("\"step_index\": 1")) >= 0, String("metadata step1 missing"))
    _require(
        meta.find(String("\"lr_before\": [\n        1.4999999999999998e-06")) >= 0,
        String("metadata step1 lr_before mismatch"),
    )
    _require(meta.find(String("\"parameter_entries\": 420")) >= 0, String("metadata optimizer state count mismatch"))
    _require(meta.find(String(STEP0_ADAPTERS)) >= 0, String("metadata step0 adapter path mismatch"))
    _require(meta.find(String(STEP1_ADAPTERS)) >= 0, String("metadata step1 adapter path mismatch"))


def main() raises:
    _check_meta_json()
    var ctx = DeviceContext()
    var step0 = SafeTensors.open(String(STEP0_ADAPTERS))
    var step1 = SafeTensors.open(String(STEP1_ADAPTERS))
    var names = _adapter_names(step1)

    var trainables = DeviceTrainableSet()
    var grads = DeviceGradSet()
    var state = DeviceAdamWState()
    var loaded_numel = 0
    for i in range(len(names)):
        loaded_numel += _append_device_inputs(
            step0, step1, names[i], trainables, grads, state, ctx
        )

    _require(
        loaded_numel == EXPECTED_NUMEL,
        String("full fused AdamW replay numel mismatch"),
    )
    var update_result = device_adamw_train_step_update(
        trainables,
        grads,
        state,
        Float32(0.0),
        2,
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
        String("device AdamW replay unexpectedly clipped grads"),
    )
    _require(
        update_result.optimizer_backend == String("fused_adamw_multitensor"),
        String("device AdamW replay did not use fused optimizer backend"),
    )

    var total = _empty_stats()
    for i in range(len(names)):
        var stats = _compare_one(
            step0,
            step1,
            names[i],
            trainables.params[i][],
            state.m[i][],
            state.v[i][],
            ctx,
        )
        _add_stats(total, stats)

    _require(
        total.nonzero_update == EXPECTED_NONZERO_UPDATE,
        String("full fused AdamW replay nonzero update count mismatch"),
    )
    _require(
        total.numel == EXPECTED_NUMEL,
        String("full fused AdamW replay numel mismatch after compare"),
    )
    var param_l2 = sqrt(total.param_l2_sumsq)
    var state_l2 = sqrt(total.state_l2_sumsq)
    print(
        "[zimage-fused-adamw-update-mojo] full_device_abi_replay PASS tensors=",
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
        "[zimage-fused-adamw-update-mojo] scope=all-420 optimizer-only replay through shared device train-step ABI from real OneTrainer adapter safetensors; not transformer forward/backward parity"
    )
