# models/lens/lens_dit_math.mojo - small Microsoft Lens DiT math gates.
#
# This is deliberately CPU-side and sampled: it verifies real Lens transformer
# weights against existing parity captures without introducing shared ops or a
# full Lens DiT runtime. The covered paths are block-0 image QKV and sampled
# image Q/K RMSNorm + Lens 3-axis RoPE:
#   hs -> img_in -> RMSNorm(img_norm1) -> modulate(img_mod(silu(temb))) -> img_qkv
#   img_qkv -> split q/k -> QK RMSNorm -> Lens interleaved-pair RoPE
#   text smoke hidden layers -> txt_norm/txt_in -> txt Q/K RMSNorm + text RoPE

from std.math import sqrt, exp, cos as fcos, sin as fsin, pow as fpow

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.models.lens.lens_contract import (
    LENS_DIT_INNER_DIM,
    LENS_DIT_HEAD_DIM,
    LENS_DIT_HEADS,
    LENS_DIT_MLP_HIDDEN,
    LENS_IMAGE_TOKENS,
    LENS_LATENT_H,
    LENS_LATENT_W,
    LENS_GPT_OSS_HIDDEN,
    LENS_TEXT_SMOKE_HIDDEN_05,
    LENS_TEXT_SMOKE_HIDDEN_11,
    LENS_TEXT_SMOKE_HIDDEN_17,
    LENS_TEXT_SMOKE_HIDDEN_23,
    LENS_TEXT_SMOKE_SEQ_LEN,
    LENS_TRANSFORMER_DIR,
)


comptime LENS_CAPTURE_1024_DIR = (
    "/home/alex/EriDiffusion/inference-flame/lens/parity/captures"
)
comptime LENS_CAPTURE_HS_STEP0 = LENS_CAPTURE_1024_DIR + "/hidden_states_pre_step_00.safetensors"
comptime LENS_CAPTURE_TEMB_STEP0 = LENS_CAPTURE_1024_DIR + "/temb_step0.safetensors"
comptime LENS_CAPTURE_BLOCK0_IMG_QKV = (
    LENS_CAPTURE_1024_DIR + "/block_00_step0_img_qkv.safetensors"
)
comptime LENS_CAPTURE_BLOCK0_QK_AFTER_ROPE = (
    LENS_CAPTURE_1024_DIR + "/block_00_step0_qk_after_rope.safetensors"
)
comptime LENS_QKV_WIDTH = 3 * LENS_DIT_INNER_DIM
comptime LENS_QKV_SAMPLE_BATCH = 2
comptime LENS_QKV_SAMPLE_TOKEN_COUNT = 4
comptime LENS_ROPE_HALF_DIM = LENS_DIT_HEAD_DIM // 2
comptime LENS_ROPE_FRAME_HALF = 4
comptime LENS_ROPE_HEIGHT_HALF = 14
comptime LENS_ROPE_WIDTH_HALF = 14
comptime LENS_TEXT_QK_SAMPLE_TOKEN_COUNT = 4
comptime LENS_TEXT_FEATURE_LAYERS = 4
comptime LENS_TEXT_CAT_DIM = LENS_TEXT_FEATURE_LAYERS * LENS_GPT_OSS_HIDDEN


@fieldwise_init
struct LensBlock0QKVStats(Copyable, Movable):
    var samples: Int
    var qkv_values: Int
    var finite_values: Int
    var got_mean: Float64
    var got_std: Float64
    var got_absmax: Float64
    var ref_mean: Float64
    var ref_std: Float64
    var ref_absmax: Float64
    var mean_abs_diff: Float64
    var max_abs_diff: Float64


@fieldwise_init
struct LensBlock0QKRoPEStats(Copyable, Movable):
    var samples: Int
    var qk_values: Int
    var finite_values: Int
    var got_mean: Float64
    var got_std: Float64
    var got_absmax: Float64
    var ref_mean: Float64
    var ref_std: Float64
    var ref_absmax: Float64
    var mean_abs_diff: Float64
    var max_abs_diff: Float64


@fieldwise_init
struct LensBlock0TextQKRoPEStats(Copyable, Movable):
    var samples: Int
    var qk_values: Int
    var finite_values: Int
    var got_mean: Float64
    var got_std: Float64
    var got_absmax: Float64


@fieldwise_init
struct LensBlock0FullStats(Copyable, Movable):
    # Compile-only bounded full block-0 forward smoke statistics.
    var n_img: Int
    var n_txt: Int
    var img_values: Int
    var txt_values: Int
    var img_finite: Int
    var txt_finite: Int
    var img_mean: Float64
    var img_std: Float64
    var img_absmax: Float64
    var txt_mean: Float64
    var txt_std: Float64
    var txt_absmax: Float64


def _shape1(a: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    return out^


def _shape2(a: Int, b: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    return out^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(c)
    return out^


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var out = List[Int]()
    out.append(a)
    out.append(b)
    out.append(c)
    out.append(d)
    return out^


def _check_shape(name: String, got: List[Int], expected: List[Int]) raises:
    if len(got) != len(expected):
        raise Error(String("Lens QKV rank mismatch for ") + name)
    for i in range(len(expected)):
        if got[i] != expected[i]:
            raise Error(
                String("Lens QKV shape mismatch for ")
                + name
                + String(" at dim ")
                + String(i)
                + String(": got=")
                + String(got[i])
                + String(" expected=")
                + String(expected[i])
            )


def _check_tensor(
    ref st: SafeTensors, name: String, dtype: STDtype, expected_shape: List[Int]
) raises:
    var info = st.tensor_info(name)
    if info.dtype != dtype:
        raise Error(
            String("Lens QKV dtype mismatch for ")
            + name
            + String(": got=")
            + info.dtype.name()
            + String(" expected=")
            + dtype.name()
        )
    _check_shape(name, info.shape, expected_shape)


def _check_weight(
    ref st: ShardedSafeTensors, name: String, dtype: STDtype, expected_shape: List[Int]
) raises:
    var info = st.tensor_info(name)
    if info.dtype != dtype:
        raise Error(
            String("Lens QKV weight dtype mismatch for ")
            + name
            + String(": got=")
            + info.dtype.name()
            + String(" expected=")
            + dtype.name()
        )
    _check_shape(name, info.shape, expected_shape)


def _sample_tokens() -> List[Int]:
    var out = List[Int]()
    out.append(0)
    out.append(1)
    out.append(137)
    out.append(LENS_IMAGE_TOKENS - 1)
    return out^


def _text_sample_tokens() -> List[Int]:
    var out = List[Int]()
    out.append(0)
    out.append(1)
    out.append(17)
    out.append(LENS_TEXT_SMOKE_SEQ_LEN - 1)
    return out^


def _abs64(x: Float64) -> Float64:
    if x < 0.0:
        return -x
    return x


def _silu(x: Float32) -> Float32:
    var xf = Float64(x)
    if xf >= 0.0:
        return Float32(xf / (1.0 + exp(-xf)))
    var e = exp(xf)
    return Float32((xf * e) / (1.0 + e))


def _is_bad(x: Float64) -> Bool:
    if x != x:
        return True
    if x > 1.0e30 or x < -1.0e30:
        return True
    return False


def _lens_image_rope_angle(token: Int, pair: Int) -> Float64:
    var y = token // LENS_LATENT_W
    var x = token - y * LENS_LATENT_W
    if pair < LENS_ROPE_FRAME_HALF:
        return 0.0
    if pair < LENS_ROPE_FRAME_HALF + LENS_ROPE_HEIGHT_HALF:
        var k = pair - LENS_ROPE_FRAME_HALF
        var exponent = Float64(2 * k) / 28.0
        var freq = 1.0 / fpow(10000.0, exponent)
        return Float64(y - (LENS_LATENT_H // 2)) * freq
    var wk = pair - LENS_ROPE_FRAME_HALF - LENS_ROPE_HEIGHT_HALF
    var wexp = Float64(2 * wk) / 28.0
    var wfreq = 1.0 / fpow(10000.0, wexp)
    return Float64(x - (LENS_LATENT_W // 2)) * wfreq


def _lens_text_rope_angle(token: Int, pair: Int) -> Float64:
    var row = Float64((LENS_LATENT_H // 2) + token)
    if pair < LENS_ROPE_FRAME_HALF:
        var fk = pair
        var fexp = Float64(2 * fk) / 8.0
        var ffreq = 1.0 / fpow(10000.0, fexp)
        return row * ffreq
    if pair < LENS_ROPE_FRAME_HALF + LENS_ROPE_HEIGHT_HALF:
        var hk = pair - LENS_ROPE_FRAME_HALF
        var hexp = Float64(2 * hk) / 28.0
        var hfreq = 1.0 / fpow(10000.0, hexp)
        return row * hfreq
    var wk = pair - LENS_ROPE_FRAME_HALF - LENS_ROPE_HEIGHT_HALF
    var wexp = Float64(2 * wk) / 28.0
    var wfreq = 1.0 / fpow(10000.0, wexp)
    return row * wfreq


def validate_lens_block0_qkv_sample_gate() raises -> LensBlock0QKVStats:
    var transformer = ShardedSafeTensors.open(String(LENS_TRANSFORMER_DIR))
    var hs_st = SafeTensors.open(String(LENS_CAPTURE_HS_STEP0))
    var temb_st = SafeTensors.open(String(LENS_CAPTURE_TEMB_STEP0))
    var qkv_st = SafeTensors.open(String(LENS_CAPTURE_BLOCK0_IMG_QKV))
    var qk_rope_st = SafeTensors.open(String(LENS_CAPTURE_BLOCK0_QK_AFTER_ROPE))

    _check_tensor(hs_st, String("hs"), STDtype.BF16, _shape3(2, LENS_IMAGE_TOKENS, 128))
    _check_tensor(temb_st, String("temb"), STDtype.BF16, _shape2(2, LENS_DIT_INNER_DIM))
    _check_tensor(
        qkv_st,
        String("img_qkv"),
        STDtype.BF16,
        _shape3(2, LENS_IMAGE_TOKENS, LENS_QKV_WIDTH),
    )
    _check_tensor(
        qk_rope_st,
        String("img_q"),
        STDtype.BF16,
        _shape4(2, LENS_IMAGE_TOKENS, LENS_DIT_HEADS, LENS_DIT_HEAD_DIM),
    )
    _check_tensor(
        qk_rope_st,
        String("img_k"),
        STDtype.BF16,
        _shape4(2, LENS_IMAGE_TOKENS, LENS_DIT_HEADS, LENS_DIT_HEAD_DIM),
    )

    _check_weight(transformer, String("img_in.weight"), STDtype.F32, _shape2(1536, 128))
    _check_weight(transformer, String("img_in.bias"), STDtype.F32, _shape1(1536))
    _check_weight(
        transformer,
        String("transformer_blocks.0.img_mod.1.weight"),
        STDtype.F32,
        _shape2(6 * LENS_DIT_INNER_DIM, LENS_DIT_INNER_DIM),
    )
    _check_weight(
        transformer,
        String("transformer_blocks.0.img_mod.1.bias"),
        STDtype.F32,
        _shape1(6 * LENS_DIT_INNER_DIM),
    )
    _check_weight(
        transformer,
        String("transformer_blocks.0.img_norm1.weight"),
        STDtype.F32,
        _shape1(LENS_DIT_INNER_DIM),
    )
    _check_weight(
        transformer,
        String("transformer_blocks.0.attn.img_qkv.weight"),
        STDtype.F32,
        _shape2(LENS_QKV_WIDTH, LENS_DIT_INNER_DIM),
    )
    _check_weight(
        transformer,
        String("transformer_blocks.0.attn.img_qkv.bias"),
        STDtype.F32,
        _shape1(LENS_QKV_WIDTH),
    )

    var tokens = _sample_tokens()
    if len(tokens) != LENS_QKV_SAMPLE_TOKEN_COUNT:
        raise Error("Lens QKV sample token count drifted")
    var sample_count = LENS_QKV_SAMPLE_BATCH * len(tokens)

    var hs_bytes = hs_st.tensor_bytes(String("hs"))
    var temb_bytes = temb_st.tensor_bytes(String("temb"))
    var ref_qkv_bytes = qkv_st.tensor_bytes(String("img_qkv"))
    var img_in_w_bytes = transformer.tensor_bytes(String("img_in.weight"))
    var img_in_b_bytes = transformer.tensor_bytes(String("img_in.bias"))
    var img_mod_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.img_mod.1.weight"))
    var img_mod_b_bytes = transformer.tensor_bytes(String("transformer_blocks.0.img_mod.1.bias"))
    var img_norm_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.img_norm1.weight"))
    var qkv_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.img_qkv.weight"))
    var qkv_b_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.img_qkv.bias"))

    var hs = hs_bytes.unsafe_ptr().bitcast[BFloat16]()
    var temb = temb_bytes.unsafe_ptr().bitcast[BFloat16]()
    var ref_qkv = ref_qkv_bytes.unsafe_ptr().bitcast[BFloat16]()
    var img_in_w = img_in_w_bytes.unsafe_ptr().bitcast[Float32]()
    var img_in_b = img_in_b_bytes.unsafe_ptr().bitcast[Float32]()
    var img_mod_w = img_mod_w_bytes.unsafe_ptr().bitcast[Float32]()
    var img_mod_b = img_mod_b_bytes.unsafe_ptr().bitcast[Float32]()
    var img_norm_w = img_norm_w_bytes.unsafe_ptr().bitcast[Float32]()
    var qkv_w = qkv_w_bytes.unsafe_ptr().bitcast[Float32]()
    var qkv_b = qkv_b_bytes.unsafe_ptr().bitcast[Float32]()

    var temb_silu = List[Float32]()
    for b in range(LENS_QKV_SAMPLE_BATCH):
        for d in range(LENS_DIT_INNER_DIM):
            temb_silu.append(_silu(temb[b * LENS_DIT_INNER_DIM + d].cast[DType.float32]()))

    # Only the first two modulation chunks are needed for image attention QKV:
    # shift1 and scale1, each [B, dim].
    var shift = List[Float32]()
    var scale = List[Float32]()
    for b in range(LENS_QKV_SAMPLE_BATCH):
        for out_d in range(2 * LENS_DIT_INNER_DIM):
            var acc = img_mod_b[out_d]
            for in_d in range(LENS_DIT_INNER_DIM):
                acc += (
                    temb_silu[b * LENS_DIT_INNER_DIM + in_d]
                    * img_mod_w[out_d * LENS_DIT_INNER_DIM + in_d]
                )
            if out_d < LENS_DIT_INNER_DIM:
                shift.append(acc)
            else:
                scale.append(acc)

    var h = List[Float32]()
    for b in range(LENS_QKV_SAMPLE_BATCH):
        for ti in range(len(tokens)):
            var tok = tokens[ti]
            for out_d in range(LENS_DIT_INNER_DIM):
                var acc = img_in_b[out_d]
                for in_d in range(128):
                    acc += (
                        hs[(b * LENS_IMAGE_TOKENS + tok) * 128 + in_d].cast[DType.float32]()
                        * img_in_w[out_d * 128 + in_d]
                    )
                h.append(acc)

    var modulated = List[Float32]()
    for sample in range(sample_count):
        var b = sample // len(tokens)
        var sumsq = Float64(0.0)
        for d in range(LENS_DIT_INNER_DIM):
            var x = Float64(h[sample * LENS_DIT_INNER_DIM + d])
            sumsq += x * x
        var inv_rms = Float32(1.0 / sqrt(sumsq / Float64(LENS_DIT_INNER_DIM) + 1.0e-6))
        for d in range(LENS_DIT_INNER_DIM):
            var normed = h[sample * LENS_DIT_INNER_DIM + d] * inv_rms * img_norm_w[d]
            modulated.append(
                normed * (Float32(1.0) + scale[b * LENS_DIT_INNER_DIM + d])
                + shift[b * LENS_DIT_INNER_DIM + d]
            )

    var count = 0
    var finite = 0
    var got_sum = Float64(0.0)
    var got_sum2 = Float64(0.0)
    var got_absmax = Float64(0.0)
    var ref_sum = Float64(0.0)
    var ref_sum2 = Float64(0.0)
    var ref_absmax = Float64(0.0)
    var diff_sum = Float64(0.0)
    var diff_max = Float64(0.0)

    for sample in range(sample_count):
        var b = sample // len(tokens)
        var tok = tokens[sample - b * len(tokens)]
        for out_d in range(LENS_QKV_WIDTH):
            var acc = qkv_b[out_d]
            for in_d in range(LENS_DIT_INNER_DIM):
                acc += (
                    modulated[sample * LENS_DIT_INNER_DIM + in_d]
                    * qkv_w[out_d * LENS_DIT_INNER_DIM + in_d]
                )
            var got = Float64(acc)
            var expected = Float64(
                ref_qkv[
                    (b * LENS_IMAGE_TOKENS + tok) * LENS_QKV_WIDTH + out_d
                ].cast[DType.float32]()
            )
            count += 1
            if not _is_bad(got):
                finite += 1
            got_sum += got
            got_sum2 += got * got
            var got_abs = _abs64(got)
            if got_abs > got_absmax:
                got_absmax = got_abs
            ref_sum += expected
            ref_sum2 += expected * expected
            var ref_abs = _abs64(expected)
            if ref_abs > ref_absmax:
                ref_absmax = ref_abs
            var diff = _abs64(got - expected)
            diff_sum += diff
            if diff > diff_max:
                diff_max = diff

    if count == 0:
        raise Error("Lens QKV gate produced no values")
    if finite != count:
        raise Error("Lens QKV gate produced non-finite values")

    var n = Float64(count)
    var got_mean = got_sum / n
    var got_var = got_sum2 / n - got_mean * got_mean
    if got_var < 0.0:
        got_var = 0.0
    var ref_mean = ref_sum / n
    var ref_var = ref_sum2 / n - ref_mean * ref_mean
    if ref_var < 0.0:
        ref_var = 0.0
    var mean_abs_diff = diff_sum / n

    # CPU F32 math is compared to captured BF16 CUDA output. Keep this as a
    # finite-stats/parity guard, not a bit-exact kernel test.
    if mean_abs_diff > 0.05 or diff_max > 0.75:
        raise Error(
            String("Lens QKV sample drift too high: mean_abs_diff=")
            + String(mean_abs_diff)
            + String(" max_abs_diff=")
            + String(diff_max)
        )

    return LensBlock0QKVStats(
        sample_count,
        count,
        finite,
        got_mean,
        sqrt(got_var),
        got_absmax,
        ref_mean,
        sqrt(ref_var),
        ref_absmax,
        mean_abs_diff,
        diff_max,
    )


def validate_lens_block0_qk_rope_sample_gate() raises -> LensBlock0QKRoPEStats:
    var transformer = ShardedSafeTensors.open(String(LENS_TRANSFORMER_DIR))
    var hs_st = SafeTensors.open(String(LENS_CAPTURE_HS_STEP0))
    var temb_st = SafeTensors.open(String(LENS_CAPTURE_TEMB_STEP0))
    var qk_rope_st = SafeTensors.open(String(LENS_CAPTURE_BLOCK0_QK_AFTER_ROPE))

    _check_tensor(hs_st, String("hs"), STDtype.BF16, _shape3(2, LENS_IMAGE_TOKENS, 128))
    _check_tensor(temb_st, String("temb"), STDtype.BF16, _shape2(2, LENS_DIT_INNER_DIM))
    _check_tensor(
        qk_rope_st,
        String("img_q"),
        STDtype.BF16,
        _shape4(2, LENS_IMAGE_TOKENS, LENS_DIT_HEADS, LENS_DIT_HEAD_DIM),
    )
    _check_tensor(
        qk_rope_st,
        String("img_k"),
        STDtype.BF16,
        _shape4(2, LENS_IMAGE_TOKENS, LENS_DIT_HEADS, LENS_DIT_HEAD_DIM),
    )

    _check_weight(transformer, String("img_in.weight"), STDtype.F32, _shape2(1536, 128))
    _check_weight(transformer, String("img_in.bias"), STDtype.F32, _shape1(1536))
    _check_weight(
        transformer,
        String("transformer_blocks.0.img_mod.1.weight"),
        STDtype.F32,
        _shape2(6 * LENS_DIT_INNER_DIM, LENS_DIT_INNER_DIM),
    )
    _check_weight(
        transformer,
        String("transformer_blocks.0.img_mod.1.bias"),
        STDtype.F32,
        _shape1(6 * LENS_DIT_INNER_DIM),
    )
    _check_weight(
        transformer,
        String("transformer_blocks.0.img_norm1.weight"),
        STDtype.F32,
        _shape1(LENS_DIT_INNER_DIM),
    )
    _check_weight(
        transformer,
        String("transformer_blocks.0.attn.img_qkv.weight"),
        STDtype.F32,
        _shape2(LENS_QKV_WIDTH, LENS_DIT_INNER_DIM),
    )
    _check_weight(
        transformer,
        String("transformer_blocks.0.attn.img_qkv.bias"),
        STDtype.F32,
        _shape1(LENS_QKV_WIDTH),
    )
    _check_weight(
        transformer,
        String("transformer_blocks.0.attn.norm_q.weight"),
        STDtype.F32,
        _shape1(LENS_DIT_HEAD_DIM),
    )
    _check_weight(
        transformer,
        String("transformer_blocks.0.attn.norm_k.weight"),
        STDtype.F32,
        _shape1(LENS_DIT_HEAD_DIM),
    )

    var tokens = _sample_tokens()
    if len(tokens) != LENS_QKV_SAMPLE_TOKEN_COUNT:
        raise Error("Lens QK RoPE sample token count drifted")
    var sample_count = LENS_QKV_SAMPLE_BATCH * len(tokens)

    var hs_bytes = hs_st.tensor_bytes(String("hs"))
    var temb_bytes = temb_st.tensor_bytes(String("temb"))
    var ref_q_bytes = qk_rope_st.tensor_bytes(String("img_q"))
    var ref_k_bytes = qk_rope_st.tensor_bytes(String("img_k"))
    var img_in_w_bytes = transformer.tensor_bytes(String("img_in.weight"))
    var img_in_b_bytes = transformer.tensor_bytes(String("img_in.bias"))
    var img_mod_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.img_mod.1.weight"))
    var img_mod_b_bytes = transformer.tensor_bytes(String("transformer_blocks.0.img_mod.1.bias"))
    var img_norm_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.img_norm1.weight"))
    var qkv_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.img_qkv.weight"))
    var qkv_b_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.img_qkv.bias"))
    var norm_q_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.norm_q.weight"))
    var norm_k_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.norm_k.weight"))

    var hs = hs_bytes.unsafe_ptr().bitcast[BFloat16]()
    var temb = temb_bytes.unsafe_ptr().bitcast[BFloat16]()
    var ref_q = ref_q_bytes.unsafe_ptr().bitcast[BFloat16]()
    var ref_k = ref_k_bytes.unsafe_ptr().bitcast[BFloat16]()
    var img_in_w = img_in_w_bytes.unsafe_ptr().bitcast[Float32]()
    var img_in_b = img_in_b_bytes.unsafe_ptr().bitcast[Float32]()
    var img_mod_w = img_mod_w_bytes.unsafe_ptr().bitcast[Float32]()
    var img_mod_b = img_mod_b_bytes.unsafe_ptr().bitcast[Float32]()
    var img_norm_w = img_norm_w_bytes.unsafe_ptr().bitcast[Float32]()
    var qkv_w = qkv_w_bytes.unsafe_ptr().bitcast[Float32]()
    var qkv_b = qkv_b_bytes.unsafe_ptr().bitcast[Float32]()
    var norm_q_w = norm_q_w_bytes.unsafe_ptr().bitcast[Float32]()
    var norm_k_w = norm_k_w_bytes.unsafe_ptr().bitcast[Float32]()

    var temb_silu = List[Float32]()
    for b in range(LENS_QKV_SAMPLE_BATCH):
        for d in range(LENS_DIT_INNER_DIM):
            temb_silu.append(_silu(temb[b * LENS_DIT_INNER_DIM + d].cast[DType.float32]()))

    var shift = List[Float32]()
    var scale = List[Float32]()
    for b in range(LENS_QKV_SAMPLE_BATCH):
        for out_d in range(2 * LENS_DIT_INNER_DIM):
            var acc = img_mod_b[out_d]
            for in_d in range(LENS_DIT_INNER_DIM):
                acc += (
                    temb_silu[b * LENS_DIT_INNER_DIM + in_d]
                    * img_mod_w[out_d * LENS_DIT_INNER_DIM + in_d]
                )
            if out_d < LENS_DIT_INNER_DIM:
                shift.append(acc)
            else:
                scale.append(acc)

    var h = List[Float32]()
    for b in range(LENS_QKV_SAMPLE_BATCH):
        for ti in range(len(tokens)):
            var tok = tokens[ti]
            for out_d in range(LENS_DIT_INNER_DIM):
                var acc = img_in_b[out_d]
                for in_d in range(128):
                    acc += (
                        hs[(b * LENS_IMAGE_TOKENS + tok) * 128 + in_d].cast[DType.float32]()
                        * img_in_w[out_d * 128 + in_d]
                    )
                h.append(acc)

    var modulated = List[Float32]()
    for sample in range(sample_count):
        var b = sample // len(tokens)
        var sumsq = Float64(0.0)
        for d in range(LENS_DIT_INNER_DIM):
            var x = Float64(h[sample * LENS_DIT_INNER_DIM + d])
            sumsq += x * x
        var inv_rms = Float32(1.0 / sqrt(sumsq / Float64(LENS_DIT_INNER_DIM) + 1.0e-6))
        for d in range(LENS_DIT_INNER_DIM):
            var normed = h[sample * LENS_DIT_INNER_DIM + d] * inv_rms * img_norm_w[d]
            modulated.append(
                normed * (Float32(1.0) + scale[b * LENS_DIT_INNER_DIM + d])
                + shift[b * LENS_DIT_INNER_DIM + d]
            )

    var raw_q = List[Float32]()
    var raw_k = List[Float32]()
    for sample in range(sample_count):
        for head in range(LENS_DIT_HEADS):
            for d in range(LENS_DIT_HEAD_DIM):
                var q_out_d = head * LENS_DIT_HEAD_DIM + d
                var q_acc = qkv_b[q_out_d]
                for in_d in range(LENS_DIT_INNER_DIM):
                    q_acc += (
                        modulated[sample * LENS_DIT_INNER_DIM + in_d]
                        * qkv_w[q_out_d * LENS_DIT_INNER_DIM + in_d]
                    )
                raw_q.append(q_acc)

                var k_out_d = LENS_DIT_INNER_DIM + head * LENS_DIT_HEAD_DIM + d
                var k_acc = qkv_b[k_out_d]
                for in_d in range(LENS_DIT_INNER_DIM):
                    k_acc += (
                        modulated[sample * LENS_DIT_INNER_DIM + in_d]
                        * qkv_w[k_out_d * LENS_DIT_INNER_DIM + in_d]
                    )
                raw_k.append(k_acc)

    var count = 0
    var finite = 0
    var got_sum = Float64(0.0)
    var got_sum2 = Float64(0.0)
    var got_absmax = Float64(0.0)
    var ref_sum = Float64(0.0)
    var ref_sum2 = Float64(0.0)
    var ref_absmax = Float64(0.0)
    var diff_sum = Float64(0.0)
    var diff_max = Float64(0.0)

    for sample in range(sample_count):
        var b = sample // len(tokens)
        var tok = tokens[sample - b * len(tokens)]
        for head in range(LENS_DIT_HEADS):
            var q_sumsq = Float64(0.0)
            var k_sumsq = Float64(0.0)
            var base = (sample * LENS_DIT_HEADS + head) * LENS_DIT_HEAD_DIM
            for d in range(LENS_DIT_HEAD_DIM):
                var qv = Float64(raw_q[base + d])
                var kv = Float64(raw_k[base + d])
                q_sumsq += qv * qv
                k_sumsq += kv * kv
            var q_inv = 1.0 / sqrt(q_sumsq / Float64(LENS_DIT_HEAD_DIM) + 1.0e-5)
            var k_inv = 1.0 / sqrt(k_sumsq / Float64(LENS_DIT_HEAD_DIM) + 1.0e-5)

            for pair in range(LENS_ROPE_HALF_DIM):
                var d0 = 2 * pair
                var d1 = d0 + 1
                var angle = _lens_image_rope_angle(tok, pair)
                var cv = Float64(fcos(angle))
                var sv = Float64(fsin(angle))

                var q0 = Float64(raw_q[base + d0]) * q_inv * Float64(norm_q_w[d0])
                var q1 = Float64(raw_q[base + d1]) * q_inv * Float64(norm_q_w[d1])
                var got_q0 = q0 * cv - q1 * sv
                var got_q1 = q0 * sv + q1 * cv

                var k0 = Float64(raw_k[base + d0]) * k_inv * Float64(norm_k_w[d0])
                var k1 = Float64(raw_k[base + d1]) * k_inv * Float64(norm_k_w[d1])
                var got_k0 = k0 * cv - k1 * sv
                var got_k1 = k0 * sv + k1 * cv

                var ref_base = (
                    (b * LENS_IMAGE_TOKENS + tok) * LENS_DIT_HEADS + head
                ) * LENS_DIT_HEAD_DIM
                var ref_q0 = Float64(ref_q[ref_base + d0].cast[DType.float32]())
                var ref_q1 = Float64(ref_q[ref_base + d1].cast[DType.float32]())
                var ref_k0 = Float64(ref_k[ref_base + d0].cast[DType.float32]())
                var ref_k1 = Float64(ref_k[ref_base + d1].cast[DType.float32]())

                var got_vals = List[Float64]()
                got_vals.append(got_q0)
                got_vals.append(got_q1)
                got_vals.append(got_k0)
                got_vals.append(got_k1)
                var ref_vals = List[Float64]()
                ref_vals.append(ref_q0)
                ref_vals.append(ref_q1)
                ref_vals.append(ref_k0)
                ref_vals.append(ref_k1)
                for vi in range(4):
                    var got = got_vals[vi]
                    var expected = ref_vals[vi]
                    count += 1
                    if not _is_bad(got):
                        finite += 1
                    got_sum += got
                    got_sum2 += got * got
                    var got_abs = _abs64(got)
                    if got_abs > got_absmax:
                        got_absmax = got_abs
                    ref_sum += expected
                    ref_sum2 += expected * expected
                    var ref_abs = _abs64(expected)
                    if ref_abs > ref_absmax:
                        ref_absmax = ref_abs
                    var diff = _abs64(got - expected)
                    diff_sum += diff
                    if diff > diff_max:
                        diff_max = diff

    if count == 0:
        raise Error("Lens QK RoPE gate produced no values")
    if finite != count:
        raise Error("Lens QK RoPE gate produced non-finite values")

    var n = Float64(count)
    var got_mean = got_sum / n
    var got_var = got_sum2 / n - got_mean * got_mean
    if got_var < 0.0:
        got_var = 0.0
    var ref_mean = ref_sum / n
    var ref_var = ref_sum2 / n - ref_mean * ref_mean
    if ref_var < 0.0:
        ref_var = 0.0
    var mean_abs_diff = diff_sum / n

    # This compares CPU F64/F32 accumulation to captured BF16 CUDA output.
    # It guards the Lens RoPE geometry and QK-norm order without pretending to
    # be a bit-exact kernel parity test.
    if mean_abs_diff > 0.005 or diff_max > 0.08:
        raise Error(
            String("Lens QK RoPE sample drift too high: mean_abs_diff=")
            + String(mean_abs_diff)
            + String(" max_abs_diff=")
            + String(diff_max)
        )

    return LensBlock0QKRoPEStats(
        sample_count,
        count,
        finite,
        got_mean,
        sqrt(got_var),
        got_absmax,
        ref_mean,
        sqrt(ref_var),
        ref_absmax,
        mean_abs_diff,
        diff_max,
    )


def validate_lens_block0_text_qk_rope_sample_gate() raises -> LensBlock0TextQKRoPEStats:
    var transformer = ShardedSafeTensors.open(String(LENS_TRANSFORMER_DIR))
    var temb_st = SafeTensors.open(String(LENS_CAPTURE_TEMB_STEP0))
    var layer_05_st = SafeTensors.open(String(LENS_TEXT_SMOKE_HIDDEN_05))
    var layer_11_st = SafeTensors.open(String(LENS_TEXT_SMOKE_HIDDEN_11))
    var layer_17_st = SafeTensors.open(String(LENS_TEXT_SMOKE_HIDDEN_17))
    var layer_23_st = SafeTensors.open(String(LENS_TEXT_SMOKE_HIDDEN_23))

    _check_tensor(temb_st, String("temb"), STDtype.BF16, _shape2(2, LENS_DIT_INNER_DIM))
    _check_tensor(
        layer_05_st,
        String("tensor"),
        STDtype.BF16,
        _shape3(1, LENS_TEXT_SMOKE_SEQ_LEN, LENS_GPT_OSS_HIDDEN),
    )
    _check_tensor(
        layer_11_st,
        String("tensor"),
        STDtype.BF16,
        _shape3(1, LENS_TEXT_SMOKE_SEQ_LEN, LENS_GPT_OSS_HIDDEN),
    )
    _check_tensor(
        layer_17_st,
        String("tensor"),
        STDtype.BF16,
        _shape3(1, LENS_TEXT_SMOKE_SEQ_LEN, LENS_GPT_OSS_HIDDEN),
    )
    _check_tensor(
        layer_23_st,
        String("tensor"),
        STDtype.BF16,
        _shape3(1, LENS_TEXT_SMOKE_SEQ_LEN, LENS_GPT_OSS_HIDDEN),
    )

    _check_weight(transformer, String("txt_in.weight"), STDtype.F32, _shape2(1536, LENS_TEXT_CAT_DIM))
    _check_weight(transformer, String("txt_in.bias"), STDtype.F32, _shape1(1536))
    _check_weight(transformer, String("txt_norm.0.weight"), STDtype.F32, _shape1(LENS_GPT_OSS_HIDDEN))
    _check_weight(transformer, String("txt_norm.1.weight"), STDtype.F32, _shape1(LENS_GPT_OSS_HIDDEN))
    _check_weight(transformer, String("txt_norm.2.weight"), STDtype.F32, _shape1(LENS_GPT_OSS_HIDDEN))
    _check_weight(transformer, String("txt_norm.3.weight"), STDtype.F32, _shape1(LENS_GPT_OSS_HIDDEN))
    _check_weight(
        transformer,
        String("transformer_blocks.0.txt_mod.1.weight"),
        STDtype.F32,
        _shape2(6 * LENS_DIT_INNER_DIM, LENS_DIT_INNER_DIM),
    )
    _check_weight(
        transformer,
        String("transformer_blocks.0.txt_mod.1.bias"),
        STDtype.F32,
        _shape1(6 * LENS_DIT_INNER_DIM),
    )
    _check_weight(
        transformer,
        String("transformer_blocks.0.txt_norm1.weight"),
        STDtype.F32,
        _shape1(LENS_DIT_INNER_DIM),
    )
    _check_weight(
        transformer,
        String("transformer_blocks.0.attn.txt_qkv.weight"),
        STDtype.F32,
        _shape2(LENS_QKV_WIDTH, LENS_DIT_INNER_DIM),
    )
    _check_weight(
        transformer,
        String("transformer_blocks.0.attn.txt_qkv.bias"),
        STDtype.F32,
        _shape1(LENS_QKV_WIDTH),
    )
    _check_weight(
        transformer,
        String("transformer_blocks.0.attn.norm_added_q.weight"),
        STDtype.F32,
        _shape1(LENS_DIT_HEAD_DIM),
    )
    _check_weight(
        transformer,
        String("transformer_blocks.0.attn.norm_added_k.weight"),
        STDtype.F32,
        _shape1(LENS_DIT_HEAD_DIM),
    )

    var tokens = _text_sample_tokens()
    if len(tokens) != LENS_TEXT_QK_SAMPLE_TOKEN_COUNT:
        raise Error("Lens text QK RoPE sample token count drifted")

    var temb_bytes = temb_st.tensor_bytes(String("temb"))
    var layer_05_bytes = layer_05_st.tensor_bytes(String("tensor"))
    var layer_11_bytes = layer_11_st.tensor_bytes(String("tensor"))
    var layer_17_bytes = layer_17_st.tensor_bytes(String("tensor"))
    var layer_23_bytes = layer_23_st.tensor_bytes(String("tensor"))
    var txt_in_w_bytes = transformer.tensor_bytes(String("txt_in.weight"))
    var txt_in_b_bytes = transformer.tensor_bytes(String("txt_in.bias"))
    var txt_norm0_w_bytes = transformer.tensor_bytes(String("txt_norm.0.weight"))
    var txt_norm1_w_bytes = transformer.tensor_bytes(String("txt_norm.1.weight"))
    var txt_norm2_w_bytes = transformer.tensor_bytes(String("txt_norm.2.weight"))
    var txt_norm3_w_bytes = transformer.tensor_bytes(String("txt_norm.3.weight"))
    var txt_mod_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.txt_mod.1.weight"))
    var txt_mod_b_bytes = transformer.tensor_bytes(String("transformer_blocks.0.txt_mod.1.bias"))
    var txt_block_norm_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.txt_norm1.weight"))
    var txt_qkv_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.txt_qkv.weight"))
    var txt_qkv_b_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.txt_qkv.bias"))
    var norm_added_q_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.norm_added_q.weight"))
    var norm_added_k_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.norm_added_k.weight"))

    var temb = temb_bytes.unsafe_ptr().bitcast[BFloat16]()
    var layer_05 = layer_05_bytes.unsafe_ptr().bitcast[BFloat16]()
    var layer_11 = layer_11_bytes.unsafe_ptr().bitcast[BFloat16]()
    var layer_17 = layer_17_bytes.unsafe_ptr().bitcast[BFloat16]()
    var layer_23 = layer_23_bytes.unsafe_ptr().bitcast[BFloat16]()
    var txt_in_w = txt_in_w_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_in_b = txt_in_b_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_norm0_w = txt_norm0_w_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_norm1_w = txt_norm1_w_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_norm2_w = txt_norm2_w_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_norm3_w = txt_norm3_w_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_mod_w = txt_mod_w_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_mod_b = txt_mod_b_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_block_norm_w = txt_block_norm_w_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_qkv_w = txt_qkv_w_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_qkv_b = txt_qkv_b_bytes.unsafe_ptr().bitcast[Float32]()
    var norm_added_q_w = norm_added_q_w_bytes.unsafe_ptr().bitcast[Float32]()
    var norm_added_k_w = norm_added_k_w_bytes.unsafe_ptr().bitcast[Float32]()

    var temb_silu = List[Float32]()
    for d in range(LENS_DIT_INNER_DIM):
        temb_silu.append(_silu(temb[d].cast[DType.float32]()))

    var shift = List[Float32]()
    var scale = List[Float32]()
    for out_d in range(2 * LENS_DIT_INNER_DIM):
        var acc = txt_mod_b[out_d]
        for in_d in range(LENS_DIT_INNER_DIM):
            acc += temb_silu[in_d] * txt_mod_w[out_d * LENS_DIT_INNER_DIM + in_d]
        if out_d < LENS_DIT_INNER_DIM:
            shift.append(acc)
        else:
            scale.append(acc)

    var normed_text = List[Float32]()
    for ti in range(len(tokens)):
        var tok = tokens[ti]
        for layer in range(LENS_TEXT_FEATURE_LAYERS):
            var sumsq = Float64(0.0)
            for d in range(LENS_GPT_OSS_HIDDEN):
                if layer == 0:
                    var x0 = Float64(
                        layer_05[tok * LENS_GPT_OSS_HIDDEN + d].cast[DType.float32]()
                    )
                    sumsq += x0 * x0
                elif layer == 1:
                    var x1 = Float64(
                        layer_11[tok * LENS_GPT_OSS_HIDDEN + d].cast[DType.float32]()
                    )
                    sumsq += x1 * x1
                elif layer == 2:
                    var x2 = Float64(
                        layer_17[tok * LENS_GPT_OSS_HIDDEN + d].cast[DType.float32]()
                    )
                    sumsq += x2 * x2
                else:
                    var x3 = Float64(
                        layer_23[tok * LENS_GPT_OSS_HIDDEN + d].cast[DType.float32]()
                    )
                    sumsq += x3 * x3
            var inv_rms = Float32(
                1.0 / sqrt(sumsq / Float64(LENS_GPT_OSS_HIDDEN) + 1.0e-5)
            )
            for d in range(LENS_GPT_OSS_HIDDEN):
                if layer == 0:
                    var raw0 = layer_05[tok * LENS_GPT_OSS_HIDDEN + d].cast[DType.float32]()
                    normed_text.append(raw0 * inv_rms * txt_norm0_w[d])
                elif layer == 1:
                    var raw1 = layer_11[tok * LENS_GPT_OSS_HIDDEN + d].cast[DType.float32]()
                    normed_text.append(raw1 * inv_rms * txt_norm1_w[d])
                elif layer == 2:
                    var raw2 = layer_17[tok * LENS_GPT_OSS_HIDDEN + d].cast[DType.float32]()
                    normed_text.append(raw2 * inv_rms * txt_norm2_w[d])
                else:
                    var raw3 = layer_23[tok * LENS_GPT_OSS_HIDDEN + d].cast[DType.float32]()
                    normed_text.append(raw3 * inv_rms * txt_norm3_w[d])

    var text_h = List[Float32]()
    for sample in range(len(tokens)):
        for out_d in range(LENS_DIT_INNER_DIM):
            var acc = txt_in_b[out_d]
            for in_d in range(LENS_TEXT_CAT_DIM):
                acc += (
                    normed_text[sample * LENS_TEXT_CAT_DIM + in_d]
                    * txt_in_w[out_d * LENS_TEXT_CAT_DIM + in_d]
                )
            text_h.append(acc)

    var modulated = List[Float32]()
    for sample in range(len(tokens)):
        var sumsq = Float64(0.0)
        for d in range(LENS_DIT_INNER_DIM):
            var x = Float64(text_h[sample * LENS_DIT_INNER_DIM + d])
            sumsq += x * x
        var inv_rms = Float32(1.0 / sqrt(sumsq / Float64(LENS_DIT_INNER_DIM) + 1.0e-6))
        for d in range(LENS_DIT_INNER_DIM):
            var normed = text_h[sample * LENS_DIT_INNER_DIM + d] * inv_rms * txt_block_norm_w[d]
            modulated.append(normed * (Float32(1.0) + scale[d]) + shift[d])

    var raw_q = List[Float32]()
    var raw_k = List[Float32]()
    for sample in range(len(tokens)):
        for head in range(LENS_DIT_HEADS):
            for d in range(LENS_DIT_HEAD_DIM):
                var q_out_d = head * LENS_DIT_HEAD_DIM + d
                var q_acc = txt_qkv_b[q_out_d]
                for in_d in range(LENS_DIT_INNER_DIM):
                    q_acc += (
                        modulated[sample * LENS_DIT_INNER_DIM + in_d]
                        * txt_qkv_w[q_out_d * LENS_DIT_INNER_DIM + in_d]
                    )
                raw_q.append(q_acc)

                var k_out_d = LENS_DIT_INNER_DIM + head * LENS_DIT_HEAD_DIM + d
                var k_acc = txt_qkv_b[k_out_d]
                for in_d in range(LENS_DIT_INNER_DIM):
                    k_acc += (
                        modulated[sample * LENS_DIT_INNER_DIM + in_d]
                        * txt_qkv_w[k_out_d * LENS_DIT_INNER_DIM + in_d]
                    )
                raw_k.append(k_acc)

    var count = 0
    var finite = 0
    var got_sum = Float64(0.0)
    var got_sum2 = Float64(0.0)
    var got_absmax = Float64(0.0)

    for sample in range(len(tokens)):
        var tok = tokens[sample]
        for head in range(LENS_DIT_HEADS):
            var q_sumsq = Float64(0.0)
            var k_sumsq = Float64(0.0)
            var base = (sample * LENS_DIT_HEADS + head) * LENS_DIT_HEAD_DIM
            for d in range(LENS_DIT_HEAD_DIM):
                var qv = Float64(raw_q[base + d])
                var kv = Float64(raw_k[base + d])
                q_sumsq += qv * qv
                k_sumsq += kv * kv
            var q_inv = 1.0 / sqrt(q_sumsq / Float64(LENS_DIT_HEAD_DIM) + 1.0e-5)
            var k_inv = 1.0 / sqrt(k_sumsq / Float64(LENS_DIT_HEAD_DIM) + 1.0e-5)

            for pair in range(LENS_ROPE_HALF_DIM):
                var d0 = 2 * pair
                var d1 = d0 + 1
                var angle = _lens_text_rope_angle(tok, pair)
                var cv = Float64(fcos(angle))
                var sv = Float64(fsin(angle))

                var q0 = Float64(raw_q[base + d0]) * q_inv * Float64(norm_added_q_w[d0])
                var q1 = Float64(raw_q[base + d1]) * q_inv * Float64(norm_added_q_w[d1])
                var k0 = Float64(raw_k[base + d0]) * k_inv * Float64(norm_added_k_w[d0])
                var k1 = Float64(raw_k[base + d1]) * k_inv * Float64(norm_added_k_w[d1])

                var got_vals = List[Float64]()
                got_vals.append(q0 * cv - q1 * sv)
                got_vals.append(q0 * sv + q1 * cv)
                got_vals.append(k0 * cv - k1 * sv)
                got_vals.append(k0 * sv + k1 * cv)
                for vi in range(4):
                    var got = got_vals[vi]
                    count += 1
                    if not _is_bad(got):
                        finite += 1
                    got_sum += got
                    got_sum2 += got * got
                    var got_abs = _abs64(got)
                    if got_abs > got_absmax:
                        got_absmax = got_abs

    if count == 0:
        raise Error("Lens text QK RoPE gate produced no values")
    if finite != count:
        raise Error("Lens text QK RoPE gate produced non-finite values")
    if got_absmax <= 1.0e-6:
        raise Error("Lens text QK RoPE gate produced only near-zero values")
    if got_absmax > 1000.0:
        raise Error(String("Lens text QK RoPE absmax too high: ") + String(got_absmax))

    var n = Float64(count)
    var got_mean = got_sum / n
    var got_var = got_sum2 / n - got_mean * got_mean
    if got_var < 0.0:
        got_var = 0.0

    return LensBlock0TextQKRoPEStats(
        len(tokens),
        count,
        finite,
        got_mean,
        sqrt(got_var),
        got_absmax,
    )


# ---------------------------------------------------------------------------
# Microsoft Lens block-0 FULL forward (bounded, compile-only, code-only).
#
# Implements one dual-stream LensTransformerBlock forward end-to-end in pure
# Mojo, using REAL block-0 weights loaded via ShardedSafeTensors and SYNTHETIC,
# deterministic, bounded image and text inputs. This is intentionally a
# compile-time/structure gate. It is NOT a parity gate against a capture.
#
# Mirrors `LensDiTBlock::forward` in `lens_dit.rs`:
#   1) silu(temb) -> img_mod / txt_mod -> chunk(2) -> chunk(3) ->
#      (shift1, scale1, gate1), (shift2, scale2, gate2)
#   2) RMSNorm(hidden, img_norm1, eps=1e-6) -> modulate(shift1, scale1)
#   3) joint_attention:
#        img_qkv, txt_qkv -> split [Q,K,V] per head ->
#        per-head Q/K RMSNorm (norm_q/norm_k for img; norm_added_q/k for txt) ->
#        Lens 3-axis interleaved-pair RoPE on Q/K (image rope from frame/H/W;
#        text rope from `max(h/2, w/2) + token_index` slice of pos table) ->
#        concat [img, txt] along seq -> SDPA -> split -> img_out / txt_out
#   4) residual + gate1
#   5) RMSNorm2 -> modulate(shift2, scale2) -> SwiGLU MLP (w2(silu(w1) * w3))
#   6) residual + gate2
#
# Bounded synthetic sizes:
#   - N_IMG = 64 (8x8 image grid; full Lens uses 64x64=4096)
#   - N_TXT = 64
#   - HIDDEN = 1536, HEADS = 24, HEAD_DIM = 64
#
# A single batch row is used to keep build-time bounded. Synthetic image
# patches `[N_IMG, 128]` and synthetic concatenated text features
# `[N_TXT, 4*2880=11520]` are generated deterministically.

comptime LENS_FULL_N_IMG = 64
comptime LENS_FULL_N_TXT = 64
comptime LENS_FULL_H = 8
comptime LENS_FULL_W = 8


def _det_f32(i: Int, j: Int, seed: Int) -> Float32:
    # Deterministic, bounded synthetic value generator. Avoids `random.mojo`
    # entirely; this is a compile/structure gate, not a parity gate.
    var k = (i * 1315423911) ^ (j * 2654435761) ^ (seed * 0x9E3779B9)
    var kk = k & 0xFFFFFFFF
    var f = Float32(Float64(kk) * (1.0 / 4294967296.0))
    return f * Float32(2.0) - Float32(1.0)


def _lens_image_rope_angle_hw(token: Int, pair: Int, h: Int, w: Int) -> Float64:
    # Variant of `_lens_image_rope_angle` for arbitrary (h, w) image grids,
    # so the bounded smoke can use h=w=8 instead of the full 64x64 grid.
    var y = token // w
    var x = token - y * w
    if pair < LENS_ROPE_FRAME_HALF:
        return 0.0
    if pair < LENS_ROPE_FRAME_HALF + LENS_ROPE_HEIGHT_HALF:
        var k = pair - LENS_ROPE_FRAME_HALF
        var exponent = Float64(2 * k) / 28.0
        var freq = 1.0 / fpow(10000.0, exponent)
        return Float64(y - (h // 2)) * freq
    var wk = pair - LENS_ROPE_FRAME_HALF - LENS_ROPE_HEIGHT_HALF
    var wexp = Float64(2 * wk) / 28.0
    var wfreq = 1.0 / fpow(10000.0, wexp)
    return Float64(x - (w // 2)) * wfreq


def _lens_text_rope_angle_hw(token: Int, pair: Int, h: Int, w: Int) -> Float64:
    # Variant of `_lens_text_rope_angle` for arbitrary (h, w) image grids.
    var hh = h // 2
    var ww = w // 2
    var row_offset = hh
    if ww > hh:
        row_offset = ww
    var row = Float64(row_offset + token)
    if pair < LENS_ROPE_FRAME_HALF:
        var fk = pair
        var fexp = Float64(2 * fk) / 8.0
        var ffreq = 1.0 / fpow(10000.0, fexp)
        return row * ffreq
    if pair < LENS_ROPE_FRAME_HALF + LENS_ROPE_HEIGHT_HALF:
        var hk = pair - LENS_ROPE_FRAME_HALF
        var hexp = Float64(2 * hk) / 28.0
        var hfreq = 1.0 / fpow(10000.0, hexp)
        return row * hfreq
    var wk = pair - LENS_ROPE_FRAME_HALF - LENS_ROPE_HEIGHT_HALF
    var wexp = Float64(2 * wk) / 28.0
    var wfreq = 1.0 / fpow(10000.0, wexp)
    return row * wfreq


def validate_lens_block0_full_smoke_gate() raises -> LensBlock0FullStats:
    # Bounded full block-0 forward, compile-only structural gate.
    var transformer = ShardedSafeTensors.open(String(LENS_TRANSFORMER_DIR))

    # ---- Header dtype/shape checks for every block-0 tensor referenced. ----
    _check_weight(transformer, String("img_in.weight"), STDtype.F32, _shape2(LENS_DIT_INNER_DIM, 128))
    _check_weight(transformer, String("img_in.bias"), STDtype.F32, _shape1(LENS_DIT_INNER_DIM))
    _check_weight(transformer, String("txt_in.weight"), STDtype.F32, _shape2(LENS_DIT_INNER_DIM, LENS_TEXT_CAT_DIM))
    _check_weight(transformer, String("txt_in.bias"), STDtype.F32, _shape1(LENS_DIT_INNER_DIM))
    _check_weight(transformer, String("txt_norm.0.weight"), STDtype.F32, _shape1(LENS_GPT_OSS_HIDDEN))
    _check_weight(transformer, String("txt_norm.1.weight"), STDtype.F32, _shape1(LENS_GPT_OSS_HIDDEN))
    _check_weight(transformer, String("txt_norm.2.weight"), STDtype.F32, _shape1(LENS_GPT_OSS_HIDDEN))
    _check_weight(transformer, String("txt_norm.3.weight"), STDtype.F32, _shape1(LENS_GPT_OSS_HIDDEN))

    _check_weight(transformer, String("transformer_blocks.0.img_mod.1.weight"), STDtype.F32, _shape2(6 * LENS_DIT_INNER_DIM, LENS_DIT_INNER_DIM))
    _check_weight(transformer, String("transformer_blocks.0.img_mod.1.bias"), STDtype.F32, _shape1(6 * LENS_DIT_INNER_DIM))
    _check_weight(transformer, String("transformer_blocks.0.txt_mod.1.weight"), STDtype.F32, _shape2(6 * LENS_DIT_INNER_DIM, LENS_DIT_INNER_DIM))
    _check_weight(transformer, String("transformer_blocks.0.txt_mod.1.bias"), STDtype.F32, _shape1(6 * LENS_DIT_INNER_DIM))
    _check_weight(transformer, String("transformer_blocks.0.img_norm1.weight"), STDtype.F32, _shape1(LENS_DIT_INNER_DIM))
    _check_weight(transformer, String("transformer_blocks.0.img_norm2.weight"), STDtype.F32, _shape1(LENS_DIT_INNER_DIM))
    _check_weight(transformer, String("transformer_blocks.0.txt_norm1.weight"), STDtype.F32, _shape1(LENS_DIT_INNER_DIM))
    _check_weight(transformer, String("transformer_blocks.0.txt_norm2.weight"), STDtype.F32, _shape1(LENS_DIT_INNER_DIM))
    _check_weight(transformer, String("transformer_blocks.0.attn.img_qkv.weight"), STDtype.F32, _shape2(LENS_QKV_WIDTH, LENS_DIT_INNER_DIM))
    _check_weight(transformer, String("transformer_blocks.0.attn.img_qkv.bias"), STDtype.F32, _shape1(LENS_QKV_WIDTH))
    _check_weight(transformer, String("transformer_blocks.0.attn.txt_qkv.weight"), STDtype.F32, _shape2(LENS_QKV_WIDTH, LENS_DIT_INNER_DIM))
    _check_weight(transformer, String("transformer_blocks.0.attn.txt_qkv.bias"), STDtype.F32, _shape1(LENS_QKV_WIDTH))
    _check_weight(transformer, String("transformer_blocks.0.attn.norm_q.weight"), STDtype.F32, _shape1(LENS_DIT_HEAD_DIM))
    _check_weight(transformer, String("transformer_blocks.0.attn.norm_k.weight"), STDtype.F32, _shape1(LENS_DIT_HEAD_DIM))
    _check_weight(transformer, String("transformer_blocks.0.attn.norm_added_q.weight"), STDtype.F32, _shape1(LENS_DIT_HEAD_DIM))
    _check_weight(transformer, String("transformer_blocks.0.attn.norm_added_k.weight"), STDtype.F32, _shape1(LENS_DIT_HEAD_DIM))
    _check_weight(transformer, String("transformer_blocks.0.attn.to_out.0.weight"), STDtype.F32, _shape2(LENS_DIT_INNER_DIM, LENS_DIT_INNER_DIM))
    _check_weight(transformer, String("transformer_blocks.0.attn.to_out.0.bias"), STDtype.F32, _shape1(LENS_DIT_INNER_DIM))
    _check_weight(transformer, String("transformer_blocks.0.attn.to_add_out.weight"), STDtype.F32, _shape2(LENS_DIT_INNER_DIM, LENS_DIT_INNER_DIM))
    _check_weight(transformer, String("transformer_blocks.0.attn.to_add_out.bias"), STDtype.F32, _shape1(LENS_DIT_INNER_DIM))
    _check_weight(transformer, String("transformer_blocks.0.img_mlp.w1.weight"), STDtype.F32, _shape2(LENS_DIT_MLP_HIDDEN, LENS_DIT_INNER_DIM))
    _check_weight(transformer, String("transformer_blocks.0.img_mlp.w2.weight"), STDtype.F32, _shape2(LENS_DIT_INNER_DIM, LENS_DIT_MLP_HIDDEN))
    _check_weight(transformer, String("transformer_blocks.0.img_mlp.w3.weight"), STDtype.F32, _shape2(LENS_DIT_MLP_HIDDEN, LENS_DIT_INNER_DIM))
    _check_weight(transformer, String("transformer_blocks.0.txt_mlp.w1.weight"), STDtype.F32, _shape2(LENS_DIT_MLP_HIDDEN, LENS_DIT_INNER_DIM))
    _check_weight(transformer, String("transformer_blocks.0.txt_mlp.w2.weight"), STDtype.F32, _shape2(LENS_DIT_INNER_DIM, LENS_DIT_MLP_HIDDEN))
    _check_weight(transformer, String("transformer_blocks.0.txt_mlp.w3.weight"), STDtype.F32, _shape2(LENS_DIT_MLP_HIDDEN, LENS_DIT_INNER_DIM))

    # ---- Acquire tensor byte views (all F32). ----
    var img_in_w_bytes = transformer.tensor_bytes(String("img_in.weight"))
    var img_in_b_bytes = transformer.tensor_bytes(String("img_in.bias"))
    var txt_in_w_bytes = transformer.tensor_bytes(String("txt_in.weight"))
    var txt_in_b_bytes = transformer.tensor_bytes(String("txt_in.bias"))
    var txt_norm0_w_bytes = transformer.tensor_bytes(String("txt_norm.0.weight"))
    var txt_norm1_w_bytes = transformer.tensor_bytes(String("txt_norm.1.weight"))
    var txt_norm2_w_bytes = transformer.tensor_bytes(String("txt_norm.2.weight"))
    var txt_norm3_w_bytes = transformer.tensor_bytes(String("txt_norm.3.weight"))
    var img_mod_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.img_mod.1.weight"))
    var img_mod_b_bytes = transformer.tensor_bytes(String("transformer_blocks.0.img_mod.1.bias"))
    var txt_mod_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.txt_mod.1.weight"))
    var txt_mod_b_bytes = transformer.tensor_bytes(String("transformer_blocks.0.txt_mod.1.bias"))
    var img_norm1_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.img_norm1.weight"))
    var img_norm2_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.img_norm2.weight"))
    var txt_norm1blk_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.txt_norm1.weight"))
    var txt_norm2blk_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.txt_norm2.weight"))
    var img_qkv_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.img_qkv.weight"))
    var img_qkv_b_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.img_qkv.bias"))
    var txt_qkv_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.txt_qkv.weight"))
    var txt_qkv_b_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.txt_qkv.bias"))
    var norm_q_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.norm_q.weight"))
    var norm_k_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.norm_k.weight"))
    var norm_added_q_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.norm_added_q.weight"))
    var norm_added_k_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.norm_added_k.weight"))
    var img_to_out_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.to_out.0.weight"))
    var img_to_out_b_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.to_out.0.bias"))
    var txt_to_out_w_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.to_add_out.weight"))
    var txt_to_out_b_bytes = transformer.tensor_bytes(String("transformer_blocks.0.attn.to_add_out.bias"))
    var img_mlp_w1_bytes = transformer.tensor_bytes(String("transformer_blocks.0.img_mlp.w1.weight"))
    var img_mlp_w2_bytes = transformer.tensor_bytes(String("transformer_blocks.0.img_mlp.w2.weight"))
    var img_mlp_w3_bytes = transformer.tensor_bytes(String("transformer_blocks.0.img_mlp.w3.weight"))
    var txt_mlp_w1_bytes = transformer.tensor_bytes(String("transformer_blocks.0.txt_mlp.w1.weight"))
    var txt_mlp_w2_bytes = transformer.tensor_bytes(String("transformer_blocks.0.txt_mlp.w2.weight"))
    var txt_mlp_w3_bytes = transformer.tensor_bytes(String("transformer_blocks.0.txt_mlp.w3.weight"))

    var img_in_w = img_in_w_bytes.unsafe_ptr().bitcast[Float32]()
    var img_in_b = img_in_b_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_in_w = txt_in_w_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_in_b = txt_in_b_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_norm0_w = txt_norm0_w_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_norm1_w = txt_norm1_w_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_norm2_w = txt_norm2_w_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_norm3_w = txt_norm3_w_bytes.unsafe_ptr().bitcast[Float32]()
    var img_mod_w = img_mod_w_bytes.unsafe_ptr().bitcast[Float32]()
    var img_mod_b = img_mod_b_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_mod_w = txt_mod_w_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_mod_b = txt_mod_b_bytes.unsafe_ptr().bitcast[Float32]()
    var img_norm1_w = img_norm1_w_bytes.unsafe_ptr().bitcast[Float32]()
    var img_norm2_w = img_norm2_w_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_norm1blk_w = txt_norm1blk_w_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_norm2blk_w = txt_norm2blk_w_bytes.unsafe_ptr().bitcast[Float32]()
    var img_qkv_w = img_qkv_w_bytes.unsafe_ptr().bitcast[Float32]()
    var img_qkv_b = img_qkv_b_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_qkv_w = txt_qkv_w_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_qkv_b = txt_qkv_b_bytes.unsafe_ptr().bitcast[Float32]()
    var norm_q_w = norm_q_w_bytes.unsafe_ptr().bitcast[Float32]()
    var norm_k_w = norm_k_w_bytes.unsafe_ptr().bitcast[Float32]()
    var norm_added_q_w = norm_added_q_w_bytes.unsafe_ptr().bitcast[Float32]()
    var norm_added_k_w = norm_added_k_w_bytes.unsafe_ptr().bitcast[Float32]()
    var img_to_out_w = img_to_out_w_bytes.unsafe_ptr().bitcast[Float32]()
    var img_to_out_b = img_to_out_b_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_to_out_w = txt_to_out_w_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_to_out_b = txt_to_out_b_bytes.unsafe_ptr().bitcast[Float32]()
    var img_mlp_w1 = img_mlp_w1_bytes.unsafe_ptr().bitcast[Float32]()
    var img_mlp_w2 = img_mlp_w2_bytes.unsafe_ptr().bitcast[Float32]()
    var img_mlp_w3 = img_mlp_w3_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_mlp_w1 = txt_mlp_w1_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_mlp_w2 = txt_mlp_w2_bytes.unsafe_ptr().bitcast[Float32]()
    var txt_mlp_w3 = txt_mlp_w3_bytes.unsafe_ptr().bitcast[Float32]()

    # ---- Build synthetic deterministic inputs. ----
    # Synthetic temb [HIDDEN], silu'd.
    var temb_silu = List[Float32]()
    for d in range(LENS_DIT_INNER_DIM):
        temb_silu.append(_silu(_det_f32(0, d, 7)))

    # img_mod / txt_mod -> 6 chunks. Indexing per Rust:
    #   chunk(2, -1) -> halves [0:3*dim], [3*dim:6*dim].
    #   each half -> chunk(3, -1) -> [shift, scale, gate].
    # So out_d in [0, dim) = img_shift1; [dim, 2*dim) = img_scale1;
    #    [2*dim, 3*dim) = img_gate1; [3*dim, 4*dim) = img_shift2;
    #    [4*dim, 5*dim) = img_scale2; [5*dim, 6*dim) = img_gate2.
    var img_mod_out = List[Float32]()
    for out_d in range(6 * LENS_DIT_INNER_DIM):
        var acc = img_mod_b[out_d]
        for in_d in range(LENS_DIT_INNER_DIM):
            acc += temb_silu[in_d] * img_mod_w[out_d * LENS_DIT_INNER_DIM + in_d]
        img_mod_out.append(acc)
    var txt_mod_out = List[Float32]()
    for out_d in range(6 * LENS_DIT_INNER_DIM):
        var acc = txt_mod_b[out_d]
        for in_d in range(LENS_DIT_INNER_DIM):
            acc += temb_silu[in_d] * txt_mod_w[out_d * LENS_DIT_INNER_DIM + in_d]
        txt_mod_out.append(acc)

    # ---- Synthetic image hidden states [N_IMG, HIDDEN] from img_in([N_IMG, 128]). ----
    var img_hidden = List[Float32]()
    for tok in range(LENS_FULL_N_IMG):
        for out_d in range(LENS_DIT_INNER_DIM):
            var acc = img_in_b[out_d]
            for in_d in range(128):
                acc += _det_f32(tok, in_d, 11) * img_in_w[out_d * 128 + in_d]
            img_hidden.append(acc)

    # ---- Synthetic text hidden states [N_TXT, HIDDEN]. ----
    # Build cat'd [N_TXT, 4 * LENS_GPT_OSS_HIDDEN] from synthetic per-layer
    # tokens through per-layer RMSNorm and then txt_in.
    var txt_normed = List[Float32]()
    for tok in range(LENS_FULL_N_TXT):
        for layer in range(LENS_TEXT_FEATURE_LAYERS):
            var sumsq = Float64(0.0)
            for d in range(LENS_GPT_OSS_HIDDEN):
                var x = Float64(_det_f32(tok, d, 13 + layer))
                sumsq += x * x
            var inv_rms = Float32(1.0 / sqrt(sumsq / Float64(LENS_GPT_OSS_HIDDEN) + 1.0e-5))
            for d in range(LENS_GPT_OSS_HIDDEN):
                var raw = _det_f32(tok, d, 13 + layer)
                var w_d: Float32
                if layer == 0:
                    w_d = txt_norm0_w[d]
                elif layer == 1:
                    w_d = txt_norm1_w[d]
                elif layer == 2:
                    w_d = txt_norm2_w[d]
                else:
                    w_d = txt_norm3_w[d]
                txt_normed.append(raw * inv_rms * w_d)

    var txt_hidden = List[Float32]()
    for tok in range(LENS_FULL_N_TXT):
        for out_d in range(LENS_DIT_INNER_DIM):
            var acc = txt_in_b[out_d]
            for in_d in range(LENS_TEXT_CAT_DIM):
                acc += (
                    txt_normed[tok * LENS_TEXT_CAT_DIM + in_d]
                    * txt_in_w[out_d * LENS_TEXT_CAT_DIM + in_d]
                )
            txt_hidden.append(acc)

    # ---- Step 1: img/txt RMSNorm1 + modulate(shift1, scale1). ----
    var img_n1 = List[Float32]()
    for tok in range(LENS_FULL_N_IMG):
        var sumsq = Float64(0.0)
        for d in range(LENS_DIT_INNER_DIM):
            var x = Float64(img_hidden[tok * LENS_DIT_INNER_DIM + d])
            sumsq += x * x
        var inv_rms = Float32(1.0 / sqrt(sumsq / Float64(LENS_DIT_INNER_DIM) + 1.0e-6))
        for d in range(LENS_DIT_INNER_DIM):
            var normed = img_hidden[tok * LENS_DIT_INNER_DIM + d] * inv_rms * img_norm1_w[d]
            var shift1 = img_mod_out[d]
            var scale1 = img_mod_out[LENS_DIT_INNER_DIM + d]
            img_n1.append(normed * (Float32(1.0) + scale1) + shift1)

    var txt_n1 = List[Float32]()
    for tok in range(LENS_FULL_N_TXT):
        var sumsq = Float64(0.0)
        for d in range(LENS_DIT_INNER_DIM):
            var x = Float64(txt_hidden[tok * LENS_DIT_INNER_DIM + d])
            sumsq += x * x
        var inv_rms = Float32(1.0 / sqrt(sumsq / Float64(LENS_DIT_INNER_DIM) + 1.0e-6))
        for d in range(LENS_DIT_INNER_DIM):
            var normed = txt_hidden[tok * LENS_DIT_INNER_DIM + d] * inv_rms * txt_norm1blk_w[d]
            var shift1 = txt_mod_out[d]
            var scale1 = txt_mod_out[LENS_DIT_INNER_DIM + d]
            txt_n1.append(normed * (Float32(1.0) + scale1) + shift1)

    # ---- Step 2: QKV projections per stream. ----
    # Layout: [tok, HEAD, HEAD_DIM] flattened per Q/K/V. We materialise raw_q,
    # raw_k, raw_v separately to make RoPE/RMSNorm bookkeeping clearer.
    var img_q = List[Float32]()
    var img_k = List[Float32]()
    var img_v = List[Float32]()
    for tok in range(LENS_FULL_N_IMG):
        for out_d in range(LENS_QKV_WIDTH):
            var acc = img_qkv_b[out_d]
            for in_d in range(LENS_DIT_INNER_DIM):
                acc += (
                    img_n1[tok * LENS_DIT_INNER_DIM + in_d]
                    * img_qkv_w[out_d * LENS_DIT_INNER_DIM + in_d]
                )
            if out_d < LENS_DIT_INNER_DIM:
                img_q.append(acc)
            elif out_d < 2 * LENS_DIT_INNER_DIM:
                img_k.append(acc)
            else:
                img_v.append(acc)

    var txt_q = List[Float32]()
    var txt_k = List[Float32]()
    var txt_v = List[Float32]()
    for tok in range(LENS_FULL_N_TXT):
        for out_d in range(LENS_QKV_WIDTH):
            var acc = txt_qkv_b[out_d]
            for in_d in range(LENS_DIT_INNER_DIM):
                acc += (
                    txt_n1[tok * LENS_DIT_INNER_DIM + in_d]
                    * txt_qkv_w[out_d * LENS_DIT_INNER_DIM + in_d]
                )
            if out_d < LENS_DIT_INNER_DIM:
                txt_q.append(acc)
            elif out_d < 2 * LENS_DIT_INNER_DIM:
                txt_k.append(acc)
            else:
                txt_v.append(acc)

    # ---- Step 3: per-head Q/K RMSNorm and Lens 3-axis interleaved-pair RoPE. ----
    # Out layout per head-then-channel: [tok][head][d].
    var img_qn = List[Float32]()
    var img_kn = List[Float32]()
    for _ in range(LENS_FULL_N_IMG * LENS_DIT_INNER_DIM):
        img_qn.append(Float32(0.0))
        img_kn.append(Float32(0.0))
    for tok in range(LENS_FULL_N_IMG):
        for head in range(LENS_DIT_HEADS):
            var base = (tok * LENS_DIT_HEADS + head) * LENS_DIT_HEAD_DIM
            var q_sumsq = Float64(0.0)
            var k_sumsq = Float64(0.0)
            for d in range(LENS_DIT_HEAD_DIM):
                var qv = Float64(img_q[base + d])
                var kv = Float64(img_k[base + d])
                q_sumsq += qv * qv
                k_sumsq += kv * kv
            var q_inv = Float32(1.0 / sqrt(q_sumsq / Float64(LENS_DIT_HEAD_DIM) + 1.0e-5))
            var k_inv = Float32(1.0 / sqrt(k_sumsq / Float64(LENS_DIT_HEAD_DIM) + 1.0e-5))
            for pair in range(LENS_ROPE_HALF_DIM):
                var d0 = 2 * pair
                var d1 = d0 + 1
                var angle = _lens_image_rope_angle_hw(tok, pair, LENS_FULL_H, LENS_FULL_W)
                var cv = Float32(fcos(angle))
                var sv = Float32(fsin(angle))
                var q0 = img_q[base + d0] * q_inv * norm_q_w[d0]
                var q1 = img_q[base + d1] * q_inv * norm_q_w[d1]
                var k0 = img_k[base + d0] * k_inv * norm_k_w[d0]
                var k1 = img_k[base + d1] * k_inv * norm_k_w[d1]
                img_qn[base + d0] = q0 * cv - q1 * sv
                img_qn[base + d1] = q0 * sv + q1 * cv
                img_kn[base + d0] = k0 * cv - k1 * sv
                img_kn[base + d1] = k0 * sv + k1 * cv

    var txt_qn = List[Float32]()
    var txt_kn = List[Float32]()
    for _ in range(LENS_FULL_N_TXT * LENS_DIT_INNER_DIM):
        txt_qn.append(Float32(0.0))
        txt_kn.append(Float32(0.0))
    for tok in range(LENS_FULL_N_TXT):
        for head in range(LENS_DIT_HEADS):
            var base = (tok * LENS_DIT_HEADS + head) * LENS_DIT_HEAD_DIM
            var q_sumsq = Float64(0.0)
            var k_sumsq = Float64(0.0)
            for d in range(LENS_DIT_HEAD_DIM):
                var qv = Float64(txt_q[base + d])
                var kv = Float64(txt_k[base + d])
                q_sumsq += qv * qv
                k_sumsq += kv * kv
            var q_inv = Float32(1.0 / sqrt(q_sumsq / Float64(LENS_DIT_HEAD_DIM) + 1.0e-5))
            var k_inv = Float32(1.0 / sqrt(k_sumsq / Float64(LENS_DIT_HEAD_DIM) + 1.0e-5))
            for pair in range(LENS_ROPE_HALF_DIM):
                var d0 = 2 * pair
                var d1 = d0 + 1
                var angle = _lens_text_rope_angle_hw(tok, pair, LENS_FULL_H, LENS_FULL_W)
                var cv = Float32(fcos(angle))
                var sv = Float32(fsin(angle))
                var q0 = txt_q[base + d0] * q_inv * norm_added_q_w[d0]
                var q1 = txt_q[base + d1] * q_inv * norm_added_q_w[d1]
                var k0 = txt_k[base + d0] * k_inv * norm_added_k_w[d0]
                var k1 = txt_k[base + d1] * k_inv * norm_added_k_w[d1]
                txt_qn[base + d0] = q0 * cv - q1 * sv
                txt_qn[base + d1] = q0 * sv + q1 * cv
                txt_kn[base + d0] = k0 * cv - k1 * sv
                txt_kn[base + d1] = k0 * sv + k1 * cv

    # ---- Step 4: joint SDPA over concat([img, txt], seq_dim). ----
    # Python order: image first, then text (lens_dit.rs:842). No mask in this
    # smoke (mask=None semantics; full keep).
    # SDPA shapes per head: Q=[S_total, D], K=[S_total, D], V=[S_total, D].
    # Output per head: [S_total, D]; concat heads -> [S_total, HIDDEN]; split
    # back to img [N_IMG, HIDDEN] and txt [N_TXT, HIDDEN].
    var s_total = LENS_FULL_N_IMG + LENS_FULL_N_TXT
    var scale = Float64(1.0) / sqrt(Float64(LENS_DIT_HEAD_DIM))
    # Concatenated per-head Q/K/V views are read directly from img_qn/img_kn/
    # img_v and txt_qn/txt_kn/txt_v with explicit token-stream switches.
    var attn_concat = List[Float32]()
    for _ in range(s_total * LENS_DIT_INNER_DIM):
        attn_concat.append(Float32(0.0))

    for head in range(LENS_DIT_HEADS):
        # Compute attention output per head over the joint sequence.
        for qi in range(s_total):
            # Gather Q[qi, head, :].
            var q_row = List[Float64]()
            for d in range(LENS_DIT_HEAD_DIM):
                var qv: Float32
                if qi < LENS_FULL_N_IMG:
                    qv = img_qn[(qi * LENS_DIT_HEADS + head) * LENS_DIT_HEAD_DIM + d]
                else:
                    var ti = qi - LENS_FULL_N_IMG
                    qv = txt_qn[(ti * LENS_DIT_HEADS + head) * LENS_DIT_HEAD_DIM + d]
                q_row.append(Float64(qv))

            # Compute logits L[k] = (q . k_k) * scale for k in s_total.
            var logits = List[Float64]()
            var max_logit = Float64(-1.0e38)
            for kj in range(s_total):
                var dot = Float64(0.0)
                for d in range(LENS_DIT_HEAD_DIM):
                    var kv: Float32
                    if kj < LENS_FULL_N_IMG:
                        kv = img_kn[(kj * LENS_DIT_HEADS + head) * LENS_DIT_HEAD_DIM + d]
                    else:
                        var tj = kj - LENS_FULL_N_IMG
                        kv = txt_kn[(tj * LENS_DIT_HEADS + head) * LENS_DIT_HEAD_DIM + d]
                    dot += q_row[d] * Float64(kv)
                var lv = dot * scale
                logits.append(lv)
                if lv > max_logit:
                    max_logit = lv

            # Softmax in F64.
            var sum_exp = Float64(0.0)
            for kj in range(s_total):
                var e = exp(logits[kj] - max_logit)
                logits[kj] = e
                sum_exp += e
            var inv_sum = Float64(1.0) / sum_exp
            for kj in range(s_total):
                logits[kj] = logits[kj] * inv_sum

            # out[qi, head, :] = sum_kj softmax[kj] * V[kj, head, :].
            for d in range(LENS_DIT_HEAD_DIM):
                var acc = Float64(0.0)
                for kj in range(s_total):
                    var vv: Float32
                    if kj < LENS_FULL_N_IMG:
                        vv = img_v[(kj * LENS_DIT_HEADS + head) * LENS_DIT_HEAD_DIM + d]
                    else:
                        var tj = kj - LENS_FULL_N_IMG
                        vv = txt_v[(tj * LENS_DIT_HEADS + head) * LENS_DIT_HEAD_DIM + d]
                    acc += logits[kj] * Float64(vv)
                attn_concat[(qi * LENS_DIT_HEADS + head) * LENS_DIT_HEAD_DIM + d] = Float32(acc)

    # ---- Step 5: split attn_concat into img_attn, txt_attn; out projections. ----
    var img_attn_proj = List[Float32]()
    for tok in range(LENS_FULL_N_IMG):
        for out_d in range(LENS_DIT_INNER_DIM):
            var acc = img_to_out_b[out_d]
            for in_d in range(LENS_DIT_INNER_DIM):
                acc += (
                    attn_concat[tok * LENS_DIT_INNER_DIM + in_d]
                    * img_to_out_w[out_d * LENS_DIT_INNER_DIM + in_d]
                )
            img_attn_proj.append(acc)
    var txt_attn_proj = List[Float32]()
    for ti in range(LENS_FULL_N_TXT):
        var qi = LENS_FULL_N_IMG + ti
        for out_d in range(LENS_DIT_INNER_DIM):
            var acc = txt_to_out_b[out_d]
            for in_d in range(LENS_DIT_INNER_DIM):
                acc += (
                    attn_concat[qi * LENS_DIT_INNER_DIM + in_d]
                    * txt_to_out_w[out_d * LENS_DIT_INNER_DIM + in_d]
                )
            txt_attn_proj.append(acc)

    # ---- Step 6: residual + gate1. ----
    # hidden = hidden + gate1.unsqueeze(seq) * attn_proj.
    var img_after_attn = List[Float32]()
    for tok in range(LENS_FULL_N_IMG):
        for d in range(LENS_DIT_INNER_DIM):
            var gate1 = img_mod_out[2 * LENS_DIT_INNER_DIM + d]
            var v = (
                img_hidden[tok * LENS_DIT_INNER_DIM + d]
                + gate1 * img_attn_proj[tok * LENS_DIT_INNER_DIM + d]
            )
            img_after_attn.append(v)
    var txt_after_attn = List[Float32]()
    for tok in range(LENS_FULL_N_TXT):
        for d in range(LENS_DIT_INNER_DIM):
            var gate1 = txt_mod_out[2 * LENS_DIT_INNER_DIM + d]
            var v = (
                txt_hidden[tok * LENS_DIT_INNER_DIM + d]
                + gate1 * txt_attn_proj[tok * LENS_DIT_INNER_DIM + d]
            )
            txt_after_attn.append(v)

    # ---- Step 7: RMSNorm2 + modulate(shift2, scale2) + SwiGLU MLP + gate2 + residual. ----
    var img_out_list = List[Float32]()
    for tok in range(LENS_FULL_N_IMG):
        var sumsq = Float64(0.0)
        for d in range(LENS_DIT_INNER_DIM):
            var x = Float64(img_after_attn[tok * LENS_DIT_INNER_DIM + d])
            sumsq += x * x
        var inv_rms = Float32(1.0 / sqrt(sumsq / Float64(LENS_DIT_INNER_DIM) + 1.0e-6))
        var modulated2 = List[Float32]()
        for d in range(LENS_DIT_INNER_DIM):
            var normed = img_after_attn[tok * LENS_DIT_INNER_DIM + d] * inv_rms * img_norm2_w[d]
            var shift2 = img_mod_out[3 * LENS_DIT_INNER_DIM + d]
            var scale2 = img_mod_out[4 * LENS_DIT_INNER_DIM + d]
            modulated2.append(normed * (Float32(1.0) + scale2) + shift2)

        # SwiGLU: w2(silu(w1(x)) * w3(x)).
        var gate_proj = List[Float32]()
        var up_proj = List[Float32]()
        for h_out in range(LENS_DIT_MLP_HIDDEN):
            var g_acc = Float32(0.0)
            var u_acc = Float32(0.0)
            for in_d in range(LENS_DIT_INNER_DIM):
                g_acc += modulated2[in_d] * img_mlp_w1[h_out * LENS_DIT_INNER_DIM + in_d]
                u_acc += modulated2[in_d] * img_mlp_w3[h_out * LENS_DIT_INNER_DIM + in_d]
            gate_proj.append(g_acc)
            up_proj.append(u_acc)
        var activated = List[Float32]()
        for h_out in range(LENS_DIT_MLP_HIDDEN):
            activated.append(_silu(gate_proj[h_out]) * up_proj[h_out])
        var mlp_out = List[Float32]()
        for out_d in range(LENS_DIT_INNER_DIM):
            var acc = Float32(0.0)
            for in_d in range(LENS_DIT_MLP_HIDDEN):
                acc += activated[in_d] * img_mlp_w2[out_d * LENS_DIT_MLP_HIDDEN + in_d]
            mlp_out.append(acc)
        for d in range(LENS_DIT_INNER_DIM):
            var gate2 = img_mod_out[5 * LENS_DIT_INNER_DIM + d]
            img_out_list.append(img_after_attn[tok * LENS_DIT_INNER_DIM + d] + gate2 * mlp_out[d])

    var txt_out_list = List[Float32]()
    for tok in range(LENS_FULL_N_TXT):
        var sumsq = Float64(0.0)
        for d in range(LENS_DIT_INNER_DIM):
            var x = Float64(txt_after_attn[tok * LENS_DIT_INNER_DIM + d])
            sumsq += x * x
        var inv_rms = Float32(1.0 / sqrt(sumsq / Float64(LENS_DIT_INNER_DIM) + 1.0e-6))
        var modulated2 = List[Float32]()
        for d in range(LENS_DIT_INNER_DIM):
            var normed = txt_after_attn[tok * LENS_DIT_INNER_DIM + d] * inv_rms * txt_norm2blk_w[d]
            var shift2 = txt_mod_out[3 * LENS_DIT_INNER_DIM + d]
            var scale2 = txt_mod_out[4 * LENS_DIT_INNER_DIM + d]
            modulated2.append(normed * (Float32(1.0) + scale2) + shift2)

        var gate_proj = List[Float32]()
        var up_proj = List[Float32]()
        for h_out in range(LENS_DIT_MLP_HIDDEN):
            var g_acc = Float32(0.0)
            var u_acc = Float32(0.0)
            for in_d in range(LENS_DIT_INNER_DIM):
                g_acc += modulated2[in_d] * txt_mlp_w1[h_out * LENS_DIT_INNER_DIM + in_d]
                u_acc += modulated2[in_d] * txt_mlp_w3[h_out * LENS_DIT_INNER_DIM + in_d]
            gate_proj.append(g_acc)
            up_proj.append(u_acc)
        var activated = List[Float32]()
        for h_out in range(LENS_DIT_MLP_HIDDEN):
            activated.append(_silu(gate_proj[h_out]) * up_proj[h_out])
        var mlp_out = List[Float32]()
        for out_d in range(LENS_DIT_INNER_DIM):
            var acc = Float32(0.0)
            for in_d in range(LENS_DIT_MLP_HIDDEN):
                acc += activated[in_d] * txt_mlp_w2[out_d * LENS_DIT_MLP_HIDDEN + in_d]
            mlp_out.append(acc)
        for d in range(LENS_DIT_INNER_DIM):
            var gate2 = txt_mod_out[5 * LENS_DIT_INNER_DIM + d]
            txt_out_list.append(txt_after_attn[tok * LENS_DIT_INNER_DIM + d] + gate2 * mlp_out[d])

    # ---- Stats over both outputs. ----
    var img_n = LENS_FULL_N_IMG * LENS_DIT_INNER_DIM
    var txt_n = LENS_FULL_N_TXT * LENS_DIT_INNER_DIM
    var img_sum = Float64(0.0)
    var img_sum2 = Float64(0.0)
    var img_absmax = Float64(0.0)
    var img_finite = 0
    for i in range(img_n):
        var v = Float64(img_out_list[i])
        if not _is_bad(v):
            img_finite += 1
        img_sum += v
        img_sum2 += v * v
        var av = _abs64(v)
        if av > img_absmax:
            img_absmax = av
    var img_mean = img_sum / Float64(img_n)
    var img_var = img_sum2 / Float64(img_n) - img_mean * img_mean
    if img_var < 0.0:
        img_var = 0.0

    var txt_sum = Float64(0.0)
    var txt_sum2 = Float64(0.0)
    var txt_absmax = Float64(0.0)
    var txt_finite = 0
    for i in range(txt_n):
        var v = Float64(txt_out_list[i])
        if not _is_bad(v):
            txt_finite += 1
        txt_sum += v
        txt_sum2 += v * v
        var av = _abs64(v)
        if av > txt_absmax:
            txt_absmax = av
    var txt_mean = txt_sum / Float64(txt_n)
    var txt_var = txt_sum2 / Float64(txt_n) - txt_mean * txt_mean
    if txt_var < 0.0:
        txt_var = 0.0

    if img_finite != img_n:
        raise Error("Lens full block-0 image output contains non-finite values")
    if txt_finite != txt_n:
        raise Error("Lens full block-0 text output contains non-finite values")

    return LensBlock0FullStats(
        LENS_FULL_N_IMG,
        LENS_FULL_N_TXT,
        img_n,
        txt_n,
        img_finite,
        txt_finite,
        img_mean,
        sqrt(img_var),
        img_absmax,
        txt_mean,
        sqrt(txt_var),
        txt_absmax,
    )
