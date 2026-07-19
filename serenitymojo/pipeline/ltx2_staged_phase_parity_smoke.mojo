# ltx2_staged_phase_parity_smoke.mojo — staged-HQ phase handoff parity.
#
# This gate is deliberately not a render.  It proves the staged pipeline joins:
#   stage-1 final denoise -> spatial-upscaler input contract
#   spatial-upscaler output -> stage-2 GaussianNoiser init
#   stage-2 video/audio latents -> first HQ res2s/SDE/bong step
#
# Python only generates fixed oracle tensors.  The phase math is recomputed here
# in Mojo using the same sampling functions as ltx2_t2v_av_hq.mojo.

from std.gpu.host import DeviceContext
from std.math import sqrt
from std.memory import alloc

from serenitymojo.io.dtype import STDtype
from serenitymojo.io.ffi import O_RDONLY, file_size, sys_close, sys_open, sys_pread
from serenitymojo.ops.tensor_algebra import add, mul_scalar, sub
from serenitymojo.sampling.ltx2_sampling import (
    res2s_bong_active,
    res2s_bong_refine,
    res2s_coefficients,
    res2s_combine,
    res2s_sde_step,
    res2s_substep,
)
from serenitymojo.tensor import Tensor


comptime REF = "/home/alex/mojodiffusion/output/ltx2_staged_phase"
comptime F32 = STDtype.F32
comptime TERMINAL_SIGMA = Float32(0.0011)
comptime S2_SIGMA = Float32(0.909375)
comptime S2_SIGMA_NEXT = Float32(0.725)
comptime BONG_ITERS = 100


def _shape_video_s1() -> List[Int]:
    var sh = List[Int]()
    sh.append(1); sh.append(128); sh.append(2); sh.append(8); sh.append(8)
    return sh^


def _shape_video_s2() -> List[Int]:
    var sh = List[Int]()
    sh.append(1); sh.append(128); sh.append(2); sh.append(16); sh.append(16)
    return sh^


def _shape_audio() -> List[Int]:
    var sh = List[Int]()
    sh.append(1); sh.append(8); sh.append(8); sh.append(16)
    return sh^


def _read_f32_bin(path: String) raises -> List[Float32]:
    var fd = sys_open(path, O_RDONLY)
    if fd < 0:
        raise Error(String("cannot open ") + path)
    var n = file_size(fd)
    var buf = alloc[UInt8](n)
    var done = 0
    while done < n:
        var got = sys_pread(fd, buf + done, n - done, done)
        if got <= 0:
            break
        done += got
    _ = sys_close(fd)
    if done != n:
        buf.free()
        raise Error(String("short read ") + path)
    var count = n // 4
    var fp = buf.bitcast[Float32]()
    var out = List[Float32]()
    for i in range(count):
        out.append(fp[i])
    buf.free()
    return out^


def _load_tensor(name: String, var shape: List[Int], ctx: DeviceContext) raises -> Tensor:
    var data = _read_f32_bin(String(REF) + "/" + name + ".bin")
    return Tensor.from_host(data, shape^, F32, ctx)


def _denoise_from_vel(x: Tensor, vel: Tensor, sigma: Float32, ctx: DeviceContext) raises -> Tensor:
    return sub(x, mul_scalar(vel, sigma, ctx), ctx)


def _abs64(x: Float64) -> Float64:
    return x if x >= 0.0 else -x


def _compare(
    name: String,
    got: Tensor,
    ref_name: String,
    ctx: DeviceContext,
    cos_gate: Float64,
    max_abs_gate: Float32,
) raises:
    var gh = got.to_host(ctx)
    var rh = _read_f32_bin(String(REF) + "/" + ref_name + ".bin")
    if len(gh) != len(rh):
        raise Error(String("length mismatch for ") + name)
    var dot = Float64(0.0)
    var na = Float64(0.0)
    var nb = Float64(0.0)
    var max_abs = Float32(0.0)
    var finite = True
    for i in range(len(gh)):
        var a = Float64(gh[i])
        var b = Float64(rh[i])
        if not (a == a):
            finite = False
        dot += a * b
        na += a * a
        nb += b * b
        var d = gh[i] - rh[i]
        if d < 0.0:
            d = -d
        if d > max_abs:
            max_abs = d
    var cos = Float64(0.0)
    if na > 0.0 and nb > 0.0:
        cos = dot / (sqrt(na) * sqrt(nb))
    print("[cmp]", name, "cos=", Float32(cos), "max_abs=", max_abs, "finite=", finite)
    if (not finite) or cos < cos_gate or max_abs > max_abs_gate:
        raise Error(String("staged phase parity failed: ") + name)


def _stage2_res2s_first_step(
    x: Tensor,
    vel1: Tensor,
    vel2: Tensor,
    noise_sub: Tensor,
    noise_step: Tensor,
    ctx: DeviceContext,
) raises -> Tensor:
    var c = res2s_coefficients(S2_SIGMA, S2_SIGMA_NEXT)
    var den1 = _denoise_from_vel(x, vel1, S2_SIGMA, ctx)
    var mid = res2s_substep(x, den1, c.h, c.a21, ctx)
    mid = res2s_sde_step(
        x, mid, Float64(S2_SIGMA), Float64(c.sub_sigma), noise_sub, ctx
    )
    var bong_iters = BONG_ITERS if res2s_bong_active(c.h, S2_SIGMA) else 0
    var anchor = res2s_bong_refine(x, mid, den1, c.h, c.a21, bong_iters, ctx)
    var den2 = _denoise_from_vel(mid, vel2, c.sub_sigma, ctx)
    var next = res2s_combine(anchor, den1, den2, c.h, c.b1, c.b2, ctx)
    return res2s_sde_step(
        anchor, next, Float64(S2_SIGMA), Float64(S2_SIGMA_NEXT), noise_step, ctx
    )


def main() raises:
    var ctx = DeviceContext()
    print("=== LTX2 staged-HQ phase handoff parity ===")
    print("oracle:", String(REF))

    # Stage-1 final denoise.  The video result must equal the spatial-upscaler
    # reference input; audio is the latent carried into stage 2 before noising.
    var v_s1_x = _load_tensor(String("stage1_video_x"), _shape_video_s1(), ctx)
    var v_s1_vel = _load_tensor(String("stage1_video_vel"), _shape_video_s1(), ctx)
    var v_s1_final = _denoise_from_vel(v_s1_x, v_s1_vel, TERMINAL_SIGMA, ctx)
    _compare(
        String("stage1 video final denoise -> upsampler input"),
        v_s1_final,
        String("stage1_video_final_ref"),
        ctx,
        0.999999,
        2.0e-6,
    )

    var a_s1_x = _load_tensor(String("stage1_audio_x"), _shape_audio(), ctx)
    var a_s1_vel = _load_tensor(String("stage1_audio_vel"), _shape_audio(), ctx)
    var a_s1_final = _denoise_from_vel(a_s1_x, a_s1_vel, TERMINAL_SIGMA, ctx)
    _compare(
        String("stage1 audio final denoise -> stage2 audio base"),
        a_s1_final,
        String("stage1_audio_final_ref"),
        ctx,
        0.999999,
        2.0e-6,
    )

    # Stage-2 GaussianNoiser init.  Video base is the spatial-upsampler oracle
    # output; the upsampler itself is covered by latent_upsampler in the matrix.
    var v_up = _load_tensor(String("stage2_video_upscaled_ref"), _shape_video_s2(), ctx)
    var v_init_noise = _load_tensor(String("stage2_video_init_noise"), _shape_video_s2(), ctx)
    var v_stage2_x = add(
        mul_scalar(v_init_noise, S2_SIGMA, ctx),
        mul_scalar(v_up, Float32(1.0) - S2_SIGMA, ctx),
        ctx,
    )
    _compare(
        String("stage2 video GaussianNoiser init"),
        v_stage2_x,
        String("stage2_video_x_ref"),
        ctx,
        0.999999,
        2.0e-6,
    )

    var a_init_noise = _load_tensor(String("stage2_audio_init_noise"), _shape_audio(), ctx)
    var a_stage2_x = add(
        mul_scalar(a_init_noise, S2_SIGMA, ctx),
        mul_scalar(a_s1_final, Float32(1.0) - S2_SIGMA, ctx),
        ctx,
    )
    _compare(
        String("stage2 audio GaussianNoiser init"),
        a_stage2_x,
        String("stage2_audio_x_ref"),
        ctx,
        0.999999,
        2.0e-6,
    )

    # First stage-2 HQ res2s step for both video and audio, using fixed synthetic
    # model velocities and fixed SDE noise tensors.
    var v_next = _stage2_res2s_first_step(
        v_stage2_x,
        _load_tensor(String("stage2_video_vel1"), _shape_video_s2(), ctx),
        _load_tensor(String("stage2_video_vel2"), _shape_video_s2(), ctx),
        _load_tensor(String("stage2_video_noise_sub"), _shape_video_s2(), ctx),
        _load_tensor(String("stage2_video_noise_step"), _shape_video_s2(), ctx),
        ctx,
    )
    _compare(
        String("stage2 video first res2s/SDE/bong step"),
        v_next,
        String("stage2_video_next_ref"),
        ctx,
        0.99999,
        1.0e-5,
    )

    var a_next = _stage2_res2s_first_step(
        a_stage2_x,
        _load_tensor(String("stage2_audio_vel1"), _shape_audio(), ctx),
        _load_tensor(String("stage2_audio_vel2"), _shape_audio(), ctx),
        _load_tensor(String("stage2_audio_noise_sub"), _shape_audio(), ctx),
        _load_tensor(String("stage2_audio_noise_step"), _shape_audio(), ctx),
        ctx,
    )
    _compare(
        String("stage2 audio first res2s/SDE/bong step"),
        a_next,
        String("stage2_audio_next_ref"),
        ctx,
        0.99999,
        1.0e-5,
    )

    print("LTX2 staged-HQ phase handoff PARITY PASS")
