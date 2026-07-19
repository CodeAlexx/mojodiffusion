# ltx2_creator_phase_parity_smoke.mojo — real creator/Desktop phase parity.
#
# This is not a render. It consumes the real creator fast-distilled dump written
# by scripts/ltx2_creator_phase_dump.py and recomputes the phase math in Mojo:
#   * distilled stage schedules
#   * GaussianNoiser stage handoffs using captured PyTorch BF16 noise
#   * first/last raw transformer velocity capture
#   * first/last Euler velocity and next-latent updates for both modalities

from std.gpu.host import DeviceContext
from std.math import sqrt

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.ops import (
    torch_bf16_eager_add_scaled,
    torch_bf16_eager_blend_with_f32_mask,
    torch_bf16_eager_velocity_from_x0,
)
from serenitymojo.sampling.ltx2_sampling import (
    ltx2_distilled_sigmas,
    ltx2_stage2_distilled_sigmas,
)
from serenitymojo.tensor import Tensor


comptime REF = (
    "/home/alex/mojodiffusion/output/ltx2_creator_phase_dumps/"
    "creator_960x512_121f_seed42/creator_phase_tensors.safetensors"
)
comptime EXPECTED_TENSOR_COUNT = 222
comptime RAW_VELOCITY_X0_ROUNDTRIP_COS_GATE = Float64(0.9998)
comptime RAW_VELOCITY_X0_ROUNDTRIP_MAX_ABS_GATE = Float64(0.03125)

def _load(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var tv = st.tensor_view(name)
    return Tensor.from_view(tv, ctx)


def _abs64(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _shape_str(shape: List[Int]) -> String:
    var out = String("[")
    for i in range(len(shape)):
        if i != 0:
            out += String(",")
        out += String(shape[i])
    out += String("]")
    return out^


def _assert_same_shape_dtype(name: String, a: Tensor, b: Tensor) raises:
    if a.dtype() != b.dtype():
        raise Error(
            name + String(": dtype mismatch ")
            + a.dtype().name() + String(" vs ") + b.dtype().name()
        )
    var ash = a.shape()
    var bsh = b.shape()
    if len(ash) != len(bsh):
        raise Error(
            name + String(": rank mismatch ")
            + _shape_str(ash) + String(" vs ") + _shape_str(bsh)
        )
    for i in range(len(ash)):
        if ash[i] != bsh[i]:
            raise Error(
                name + String(": shape mismatch ")
                + _shape_str(ash) + String(" vs ") + _shape_str(bsh)
            )


def _compare_tensors(
    name: String,
    got: Tensor,
    expected: Tensor,
    ctx: DeviceContext,
    cos_gate: Float64,
    max_abs_gate: Float64,
) raises:
    _assert_same_shape_dtype(name, got, expected)
    var gh = got.to_host(ctx)
    var rh = expected.to_host(ctx)
    if len(gh) != len(rh):
        raise Error(name + String(": length mismatch"))
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    var max_abs = Float64(0.0)
    var max_idx = -1
    var first_idx = -1
    var first_a = Float64(0.0)
    var first_b = Float64(0.0)
    var finite = True
    for i in range(len(gh)):
        var a = Float64(gh[i])
        var b = Float64(rh[i])
        if not (a == a):
            finite = False
        dot += a * b
        na += a * a
        nb += b * b
        var d = _abs64(a - b)
        if d != 0.0 and first_idx < 0:
            first_idx = i
            first_a = a
            first_b = b
        if d > max_abs:
            max_abs = d
            max_idx = i
    var cos = Float64(1.0)
    if na > 0.0 or nb > 0.0:
        if na == 0.0 or nb == 0.0:
            cos = 0.0
        else:
            cos = dot / (sqrt(na) * sqrt(nb))
    print("[cmp]", name, "cos=", Float32(cos), "max_abs=", Float32(max_abs), "finite=", finite)
    if max_abs != 0.0:
        print(
            "      first_diff_idx=", first_idx,
            "got=", Float32(first_a),
            "expected=", Float32(first_b),
            "max_idx=", max_idx,
        )
    if (not finite) or cos < cos_gate or max_abs > max_abs_gate:
        raise Error(String("creator phase parity failed: ") + name)


def _compare_schedule(
    name: String,
    got: Tensor,
    expected: List[Float32],
    ctx: DeviceContext,
) raises:
    if got.dtype() != STDtype.F32:
        raise Error(name + String(": expected F32 schedule tensor"))
    var gh = got.to_host(ctx)
    if len(gh) != len(expected):
        raise Error(name + String(": schedule length mismatch"))
    var max_abs = Float64(0.0)
    for i in range(len(gh)):
        var d = _abs64(Float64(gh[i]) - Float64(expected[i]))
        if d > max_abs:
            max_abs = d
    print("[cmp]", name, "max_abs=", Float32(max_abs))
    if max_abs > 1.0e-7:
        raise Error(String("creator schedule parity failed: ") + name)


def _check_noiser(
    st: ShardedSafeTensors,
    stage: String,
    modality: String,
    ctx: DeviceContext,
) raises:
    var base = stage + String("__") + modality + String("__")
    var clean = _load(st, base + String("pre_noise_clean_latent"), ctx)
    var noise = _load(st, base + String("torch_randn_noise"), ctx)
    var mask = _load(st, base + String("scaled_mask"), ctx)
    var expected = _load(st, base + String("initial_noised_latent"), ctx)
    var got = torch_bf16_eager_blend_with_f32_mask(noise, clean, mask, ctx)
    _compare_tensors(
        stage + String(" ") + modality + String(" GaussianNoiser init"),
        got,
        expected,
        ctx,
        0.999999,
        0.0,
    )


def _check_euler_step(
    st: ShardedSafeTensors,
    stage: String,
    modality: String,
    step: Int,
    sigma: Float32,
    sigma_next: Float32,
    ctx: DeviceContext,
) raises:
    var prefix = (
        stage + String("__") + modality + String("__step_")
        + ("0" if step < 10 else "") + String(step) + String("__")
    )
    var inp = _load(st, prefix + String("input_latent"), ctx)
    var den = _load(st, prefix + String("denoised_post_process"), ctx)
    var expected_vel = _load(st, prefix + String("velocity_from_x0"), ctx)
    var expected_next = _load(st, prefix + String("next_latent"), ctx)
    var vel = torch_bf16_eager_velocity_from_x0(inp, den, sigma, ctx)
    _compare_tensors(
        stage + String(" ") + modality + String(" step ") + String(step) + String(" velocity"),
        vel,
        expected_vel,
        ctx,
        0.999999,
        0.0,
    )
    var next = torch_bf16_eager_add_scaled(inp, expected_vel, sigma_next - sigma, ctx)
    _compare_tensors(
        stage + String(" ") + modality + String(" step ") + String(step) + String(" next_latent"),
        next,
        expected_next,
        ctx,
        0.999999,
        0.0,
    )


def _check_transformer_capture(
    st: ShardedSafeTensors,
    stage: String,
    modality: String,
    step: Int,
    sigma: Float32,
    ctx: DeviceContext,
) raises:
    var prefix = (
        stage + String("__") + modality + String("__step_")
        + ("0" if step < 10 else "") + String(step) + String("__")
    )
    var transformer_prefix = prefix + String("transformer__")
    var inp = _load(st, prefix + String("input_latent"), ctx)
    var transformer_latent = _load(st, transformer_prefix + String("latent"), ctx)
    _compare_tensors(
        stage + String(" ") + modality + String(" step ") + String(step) + String(" transformer latent"),
        transformer_latent,
        inp,
        ctx,
        0.999999,
        0.0,
    )

    var timesteps = _load(st, transformer_prefix + String("timesteps"), ctx)
    if timesteps.dtype() != STDtype.F32:
        raise Error(stage + String(" ") + modality + String(" transformer timesteps: expected F32"))
    var ish = inp.shape()
    var tsh = timesteps.shape()
    if len(tsh) != 3 or tsh[0] != ish[0] or tsh[1] != ish[1] or tsh[2] != 1:
        raise Error(
            stage + String(" ") + modality + String(" transformer timesteps shape mismatch ")
            + _shape_str(tsh)
        )
    var th = timesteps.to_host(ctx)
    var max_abs = Float64(0.0)
    for i in range(len(th)):
        var d = _abs64(Float64(th[i]) - Float64(sigma))
        if d > max_abs:
            max_abs = d
    print(
        "[cmp]", stage, modality, "step", step,
        "transformer timesteps sigma max_abs=", Float32(max_abs),
    )
    if max_abs > 1.0e-7:
        raise Error(stage + String(" ") + modality + String(" transformer timestep mismatch"))

    var positions = _load(st, transformer_prefix + String("positions"), ctx)
    var expected_pos_dims = 1
    var expected_pos_dtype = STDtype.F32
    if modality == String("video"):
        expected_pos_dims = 3
        expected_pos_dtype = STDtype.BF16
    if positions.dtype() != expected_pos_dtype:
        raise Error(
            stage + String(" ") + modality + String(" transformer positions: unexpected dtype ")
            + positions.dtype().name()
        )
    var psh = positions.shape()
    if len(psh) != 4 or psh[0] != ish[0] or psh[1] != expected_pos_dims or psh[2] != ish[1] or psh[3] != 2:
        raise Error(
            stage + String(" ") + modality + String(" transformer positions shape mismatch ")
            + _shape_str(psh)
        )

    var raw_velocity = _load(st, transformer_prefix + String("raw_velocity"), ctx)
    var expected_vel = _load(st, prefix + String("velocity_from_x0"), ctx)
    _compare_tensors(
        stage + String(" ") + modality + String(" step ") + String(step)
            + String(" raw_velocity vs x0 roundtrip"),
        raw_velocity,
        expected_vel,
        ctx,
        RAW_VELOCITY_X0_ROUNDTRIP_COS_GATE,
        RAW_VELOCITY_X0_ROUNDTRIP_MAX_ABS_GATE,
    )


def _assert_shape_dtype(
    name: String,
    tensor: Tensor,
    dtype: STDtype,
    expected_shape: List[Int],
) raises:
    if tensor.dtype() != dtype:
        raise Error(
            name + String(": dtype mismatch ")
            + tensor.dtype().name() + String(" vs ") + dtype.name()
        )
    var sh = tensor.shape()
    if len(sh) != len(expected_shape):
        raise Error(
            name + String(": rank mismatch ")
            + _shape_str(sh) + String(" vs ") + _shape_str(expected_shape)
        )
    for i in range(len(sh)):
        if sh[i] != expected_shape[i]:
            raise Error(
                name + String(": shape mismatch ")
                + _shape_str(sh) + String(" vs ") + _shape_str(expected_shape)
            )


def _shape4(a: Int, b: Int, c: Int, d: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    s.append(d)
    return s^


def _shape3(a: Int, b: Int, c: Int) -> List[Int]:
    var s = List[Int]()
    s.append(a)
    s.append(b)
    s.append(c)
    return s^


def _check_transformer_args(
    st: ShardedSafeTensors,
    stage: String,
    modality: String,
    step: Int,
    seq: Int,
    dim: Int,
    ctx: DeviceContext,
) raises:
    var prefix = (
        stage + String("__") + modality + String("__step_")
        + ("0" if step < 10 else "") + String(step)
        + String("__transformer_args__")
    )
    var rope_half = 32
    if modality == String("video"):
        rope_half = 64
    var name = stage + String(" ") + modality + String(" step ") + String(step)
    _assert_shape_dtype(
        name + String(" transformer_args x"),
        _load(st, prefix + String("x"), ctx),
        STDtype.BF16,
        _shape3(1, seq, dim),
    )
    _assert_shape_dtype(
        name + String(" transformer_args context"),
        _load(st, prefix + String("context"), ctx),
        STDtype.BF16,
        _shape3(1, 1024, dim),
    )
    _assert_shape_dtype(
        name + String(" transformer_args timesteps"),
        _load(st, prefix + String("timesteps"), ctx),
        STDtype.BF16,
        _shape3(1, seq, dim * 9),
    )
    _assert_shape_dtype(
        name + String(" transformer_args embedded_timestep"),
        _load(st, prefix + String("embedded_timestep"), ctx),
        STDtype.BF16,
        _shape3(1, seq, dim),
    )
    _assert_shape_dtype(
        name + String(" transformer_args rope_cos"),
        _load(st, prefix + String("rope_cos"), ctx),
        STDtype.BF16,
        _shape4(1, 32, seq, rope_half),
    )
    _assert_shape_dtype(
        name + String(" transformer_args rope_sin"),
        _load(st, prefix + String("rope_sin"), ctx),
        STDtype.BF16,
        _shape4(1, 32, seq, rope_half),
    )
    _assert_shape_dtype(
        name + String(" transformer_args cross_rope_cos"),
        _load(st, prefix + String("cross_rope_cos"), ctx),
        STDtype.BF16,
        _shape4(1, 32, seq, 32),
    )
    _assert_shape_dtype(
        name + String(" transformer_args cross_rope_sin"),
        _load(st, prefix + String("cross_rope_sin"), ctx),
        STDtype.BF16,
        _shape4(1, 32, seq, 32),
    )
    _assert_shape_dtype(
        name + String(" transformer_args cross_scale_shift_timestep"),
        _load(st, prefix + String("cross_scale_shift_timestep"), ctx),
        STDtype.BF16,
        _shape3(1, 1, dim * 4),
    )
    _assert_shape_dtype(
        name + String(" transformer_args cross_gate_timestep"),
        _load(st, prefix + String("cross_gate_timestep"), ctx),
        STDtype.BF16,
        _shape3(1, 1, dim),
    )
    _assert_shape_dtype(
        name + String(" transformer_args prompt_timestep"),
        _load(st, prefix + String("prompt_timestep"), ctx),
        STDtype.BF16,
        _shape3(1, 1, dim * 2),
    )
    print("[check]", name, "transformer_args shape/dtype PASS")


def main() raises:
    var ctx = DeviceContext()
    print("=== LTX2 creator fast-distilled phase parity ===")
    print("oracle:", String(REF))
    var st = ShardedSafeTensors.open(String(REF))
    print("tensors:", st.num_tensors())
    if st.num_tensors() != EXPECTED_TENSOR_COUNT:
        raise Error("creator phase dump: expected 222 tensors")

    var s1 = ltx2_distilled_sigmas()
    var s2 = ltx2_stage2_distilled_sigmas()
    _compare_schedule(String("stage1 sigmas"), _load(st, String("stage1__sigmas"), ctx), s1, ctx)
    _compare_schedule(String("stage2 sigmas"), _load(st, String("stage2__sigmas"), ctx), s2, ctx)

    _check_noiser(st, String("stage1"), String("video"), ctx)
    _check_noiser(st, String("stage1"), String("audio"), ctx)
    _check_noiser(st, String("stage2"), String("video"), ctx)
    _check_noiser(st, String("stage2"), String("audio"), ctx)

    _check_transformer_capture(st, String("stage1"), String("video"), 0, s1[0], ctx)
    _check_transformer_capture(st, String("stage1"), String("audio"), 0, s1[0], ctx)
    _check_transformer_capture(st, String("stage1"), String("video"), 7, s1[7], ctx)
    _check_transformer_capture(st, String("stage1"), String("audio"), 7, s1[7], ctx)
    _check_transformer_capture(st, String("stage2"), String("video"), 0, s2[0], ctx)
    _check_transformer_capture(st, String("stage2"), String("audio"), 0, s2[0], ctx)
    _check_transformer_capture(st, String("stage2"), String("video"), 2, s2[2], ctx)
    _check_transformer_capture(st, String("stage2"), String("audio"), 2, s2[2], ctx)

    _check_transformer_args(st, String("stage1"), String("video"), 0, 1920, 4096, ctx)
    _check_transformer_args(st, String("stage1"), String("audio"), 0, 126, 2048, ctx)
    _check_transformer_args(st, String("stage1"), String("video"), 7, 1920, 4096, ctx)
    _check_transformer_args(st, String("stage1"), String("audio"), 7, 126, 2048, ctx)
    _check_transformer_args(st, String("stage2"), String("video"), 0, 7680, 4096, ctx)
    _check_transformer_args(st, String("stage2"), String("audio"), 0, 126, 2048, ctx)
    _check_transformer_args(st, String("stage2"), String("video"), 2, 7680, 4096, ctx)
    _check_transformer_args(st, String("stage2"), String("audio"), 2, 126, 2048, ctx)

    _check_euler_step(st, String("stage1"), String("video"), 0, s1[0], s1[1], ctx)
    _check_euler_step(st, String("stage1"), String("audio"), 0, s1[0], s1[1], ctx)
    _check_euler_step(st, String("stage1"), String("video"), 7, s1[7], s1[8], ctx)
    _check_euler_step(st, String("stage1"), String("audio"), 7, s1[7], s1[8], ctx)
    _check_euler_step(st, String("stage2"), String("video"), 0, s2[0], s2[1], ctx)
    _check_euler_step(st, String("stage2"), String("audio"), 0, s2[0], s2[1], ctx)
    _check_euler_step(st, String("stage2"), String("video"), 2, s2[2], s2[3], ctx)
    _check_euler_step(st, String("stage2"), String("audio"), 2, s2[2], s2[3], ctx)

    print("LTX2 creator fast-distilled phase PARITY PASS")
