# ZImage OneTrainer train-ref AdamW update replay.
#
# Opens the real OneTrainer ZImage step000/step001 adapter dumps and replays
# selected step001 AdamW updates from dumped F32 adapter_post_clip tensors. This is a
# Mojo safetensors consumer plus scalar optimizer-math bridge. It does not run
# transformer forward/backward and does not exercise the fused device optimizer.

from std.math import sqrt
from std.pathlib import Path

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors


comptime META_JSON = "/home/alex/serenity-trainer/parity/zimage_train_ref_meta.json"
comptime STEP0_ADAPTERS = "/home/alex/serenity-trainer/parity/zimage_train_ref_step000_adapters.safetensors"
comptime STEP1_ADAPTERS = "/home/alex/serenity-trainer/parity/zimage_train_ref_step001_adapters.safetensors"

comptime LR = Float32(1.4999999999999998e-06)
comptime WEIGHT_DECAY = Float32(0.01)
comptime BETA1 = Float32(0.9)
comptime BETA2 = Float32(0.999)
comptime EPS = Float32(1.0e-08)
comptime BC1_STEP2 = Float32(1.0) - BETA1 * BETA1
comptime BC2_STEP2 = Float32(1.0) - BETA2 * BETA2
comptime SQRT_BC2_STEP2 = sqrt(BC2_STEP2)
comptime MAX_ABS_TOL = Float32(1.0e-9)
comptime EXPECTED_SAMPLED_NUMEL = 696320


@fieldwise_init
struct ReplayStats(Copyable, Movable):
    var numel: Int
    var nonzero_update: Int
    var nonzero_error: Int
    var max_abs: Float32
    var abs_sum: Float64
    var l2_sumsq: Float64


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
        nonzero_error=0,
        max_abs=Float32(0.0),
        abs_sum=Float64(0.0),
        l2_sumsq=Float64(0.0),
    )


def _add_stats(mut total: ReplayStats, part: ReplayStats):
    total.numel += part.numel
    total.nonzero_update += part.nonzero_update
    total.nonzero_error += part.nonzero_error
    total.abs_sum += part.abs_sum
    total.l2_sumsq += part.l2_sumsq
    if part.max_abs > total.max_abs:
        total.max_abs = part.max_abs


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
    _require(info.size % 4 == 0, String("F32 tensor byte size is not divisible by 4: ") + key)
    return info.size // 4


def _require_same_shape(st_a: SafeTensors, key_a: String, st_b: SafeTensors, key_b: String) raises:
    var a = st_a.tensor_info(key_a)
    var b = st_b.tensor_info(key_b)
    _require(
        _same_shape(a.shape, b.shape),
        String("shape mismatch ") + key_a + String(" vs ") + key_b,
    )


def _require_same_values(st_a: SafeTensors, key_a: String, st_b: SafeTensors, key_b: String) raises:
    var n = _require_f32_tensor(st_a, key_a)
    _require(_require_f32_tensor(st_b, key_b) == n, String("numel mismatch ") + key_a + String(" vs ") + key_b)
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


def _sample_names() -> List[String]:
    var names = List[String]()
    names.append(String("transformer.layers.0.attention.to_q.lora_up.weight"))
    names.append(String("transformer.layers.0.attention.to_k.lora_up.weight"))
    names.append(String("transformer.layers.0.attention.to_v.lora_up.weight"))
    names.append(String("transformer.layers.0.attention.to_out.0.lora_up.weight"))
    names.append(String("transformer.layers.0.feed_forward.w1.lora_up.weight"))
    names.append(String("transformer.layers.0.feed_forward.w2.lora_up.weight"))
    names.append(String("transformer.layers.0.feed_forward.w3.lora_up.weight"))
    names.append(String("transformer.layers.1.attention.to_q.lora_up.weight"))
    return names^


def _replay_one(ref step0: SafeTensors, ref step1: SafeTensors, name: String) raises -> ReplayStats:
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

    var g0_bytes = step0.tensor_bytes(k_g0)
    var before_bytes = step1.tensor_bytes(k_before)
    var g1_bytes = step1.tensor_bytes(k_g1)
    var after_bytes = step1.tensor_bytes(k_after)
    var g0p = g0_bytes.unsafe_ptr().bitcast[Float32]()
    var beforep = before_bytes.unsafe_ptr().bitcast[Float32]()
    var g1p = g1_bytes.unsafe_ptr().bitcast[Float32]()
    var afterp = after_bytes.unsafe_ptr().bitcast[Float32]()

    var stats = _empty_stats()
    stats.numel = n
    for i in range(n):
        var g0 = g0p[i]
        var before = beforep[i]
        var g1 = g1p[i]
        var actual = afterp[i]

        # Step0 had lr=0.0 and empty AdamW state, so its dumped gradient fully
        # determines the step1 optimizer_before moments.
        var m0 = (Float32(1.0) - BETA1) * g0
        var v0 = (Float32(1.0) - BETA2) * g0 * g0
        var m = BETA1 * m0 + (Float32(1.0) - BETA1) * g1
        var v = BETA2 * v0 + (Float32(1.0) - BETA2) * g1 * g1
        var expected = before * (Float32(1.0) - LR * WEIGHT_DECAY)
        expected = expected - (LR / BC1_STEP2) * m / (sqrt(v) / SQRT_BC2_STEP2 + EPS)

        var err = expected - actual
        var ae = _abs(err)
        if actual != before:
            stats.nonzero_update += 1
        if err != Float32(0.0):
            stats.nonzero_error += 1
        if ae > stats.max_abs:
            stats.max_abs = ae
        stats.abs_sum += Float64(ae)
        var e64 = Float64(err)
        stats.l2_sumsq += e64 * e64

    _require(
        stats.max_abs <= MAX_ABS_TOL,
        String("AdamW replay max_abs too high for ")
        + name
        + String(": ")
        + String(stats.max_abs),
    )
    return stats^


def _check_meta_json() raises:
    var meta = Path(String(META_JSON)).read_text()
    _require(meta.byte_length() > 0, String("empty ZImage OneTrainer metadata JSON"))
    _require(meta.find(String("\"producer\": \"scripts/zimage_dump_train_ref.py\"")) >= 0, String("metadata producer mismatch"))
    _require(meta.find(String("\"optimizer\": \"ADAMW\"")) >= 0, String("metadata optimizer mismatch"))
    _require(meta.find(String("\"lora_weight_dtype\": \"FLOAT_32\"")) >= 0, String("metadata LoRA dtype mismatch"))
    _require(meta.find(String("\"max_steps\": 2")) >= 0, String("metadata max_steps mismatch"))
    _require(meta.find(String("\"step_index\": 1")) >= 0, String("metadata step1 missing"))
    _require(meta.find(String("\"lr_before\": [\n        1.4999999999999998e-06")) >= 0, String("metadata step1 lr_before mismatch"))
    _require(meta.find(String("\"parameter_entries\": 420")) >= 0, String("metadata optimizer state count mismatch"))
    _require(meta.find(String(STEP0_ADAPTERS)) >= 0, String("metadata step0 adapter path mismatch"))
    _require(meta.find(String(STEP1_ADAPTERS)) >= 0, String("metadata step1 adapter path mismatch"))


def main() raises:
    _check_meta_json()
    var step0 = SafeTensors.open(String(STEP0_ADAPTERS))
    var step1 = SafeTensors.open(String(STEP1_ADAPTERS))
    var names = _sample_names()
    var total = _empty_stats()
    for i in range(len(names)):
        var stats = _replay_one(step0, step1, names[i])
        _add_stats(total, stats)

    _require(total.nonzero_update > 0, String("sampled AdamW replay update is all zero"))
    _require(total.numel == EXPECTED_SAMPLED_NUMEL, String("sampled AdamW replay numel mismatch"))
    var l2 = sqrt(total.l2_sumsq)
    print(
        "[zimage-adamw-update-mojo] sampled_replay PASS tensors=",
        len(names),
        " numel=",
        total.numel,
        " nonzero_update=",
        total.nonzero_update,
        " nonzero_error=",
        total.nonzero_error,
        " max_abs=",
        total.max_abs,
        " l2=",
        l2,
    )
    print(
        "[zimage-adamw-update-mojo] scope=sampled Mojo scalar AdamW replay from real OneTrainer adapter safetensors; not fused device optimizer parity"
    )
