# pipeline/ltx2_audiosync_profile_smoke.mojo
#
# Mojo-owned contract for the LTX-Desktop/Lightricks LTX-2.3 AudioSync profile.
# This is intentionally fast: it proves the production geometry/scheduler/audio
# profile without spending hours on a render. This gate encodes the shape math
# the Mojo runner must obey.


comptime AS_WIDTH = 768
comptime AS_HEIGHT = 512
comptime AS_FRAMES = 97
comptime AS_FPS = 24
comptime AS_AUDIO_DURATION_TENTHS = 65  # 6.5s

comptime VAE_T_SCALE = 8
comptime VAE_H_SCALE = 32
comptime VAE_W_SCALE = 32
comptime AUDIO_LATENTS_PER_SECOND = 25
comptime TEXT_TOKENS = 1024
comptime HEADS = 32
comptime F32_BYTES = 4
comptime MIB = 1024 * 1024
comptime LARGE_ATTENTION_THRESHOLD = 4096
comptime GPU_BUDGET_MIB = 22000

comptime SCHED_STEPS = 20
comptime SCHED_MAX_SHIFT_HUNDREDTHS = 205
comptime SCHED_BASE_SHIFT_HUNDREDTHS = 95
comptime SCHED_TERMINAL_TENTHS = 1

comptime LORA_DISTILLED_TENTHS = 10
comptime LORA_CAMERA_STATIC_TENTHS = 3
comptime LORA_DETAILER_TENTHS = 6


def _ceil_latent_frames(frames: Int) -> Int:
    return (frames - 1) // VAE_T_SCALE + 1


def _round_half_even(numer: Int, denom: Int) -> Int:
    var q = numer // denom
    var r = numer % denom
    var twice = r * 2
    if twice < denom:
        return q
    if twice > denom:
        return q + 1
    if (q % 2) == 0:
        return q
    return q + 1


def _max3(a: Int, b: Int, c: Int) -> Int:
    var m = a if a >= b else b
    return m if m >= c else c


def _score_mib(tokens: Int) -> Int:
    return (HEADS * tokens * tokens * F32_BYTES) // MIB


def _check_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            name + String(" mismatch: got ") + String(got)
            + String(" expected ") + String(expected)
        )


def _check_bool(name: String, got: Bool) raises:
    if not got:
        raise Error(name + String(" failed"))


def main() raises:
    var nf = _ceil_latent_frames(AS_FRAMES)
    var decoded_frames = 1 + (nf - 1) * VAE_T_SCALE
    var nh = AS_HEIGHT // VAE_H_SCALE
    var nw = AS_WIDTH // VAE_W_SCALE
    var sv = nf * nh * nw

    var s2_h = nh * 2
    var s2_w = nw * 2
    var sv2 = nf * s2_h * s2_w
    var out_w2 = AS_WIDTH * 2
    var out_h2 = AS_HEIGHT * 2

    var audio_tokens_from_video = _round_half_even(
        AS_FRAMES * AUDIO_LATENTS_PER_SECOND, AS_FPS,
    )
    var audio_tokens_from_fixture = _round_half_even(
        AS_AUDIO_DURATION_TENTHS * AUDIO_LATENTS_PER_SECOND, 10,
    )

    var vpad = _max3(sv, TEXT_TOKENS, audio_tokens_from_fixture)
    var vpad2 = _max3(sv2, TEXT_TOKENS, audio_tokens_from_fixture)
    var apad = TEXT_TOKENS if TEXT_TOKENS >= audio_tokens_from_fixture else audio_tokens_from_fixture

    print("=== LTX2 AudioSync profile contract ===")
    print("pixel input:", AS_WIDTH, "x", AS_HEIGHT, "frames", AS_FRAMES, "fps", AS_FPS)
    print("stage1 latent: nf/nh/nw =", nf, nh, nw, "video_tokens =", sv)
    print("stage2 latent: nf/nh/nw =", nf, s2_h, s2_w, "video_tokens =", sv2)
    print("stage2 pixels:", out_w2, "x", out_h2)
    print("decoded frames from nf:", decoded_frames)
    print("audio tokens from video duration:", audio_tokens_from_video)
    print("audio tokens from 6.5s fixture:", audio_tokens_from_fixture)
    print("padding: stage1 video", vpad, "stage2 video", vpad2, "audio", apad)
    print("scheduler: steps", SCHED_STEPS, "max_shift 2.05 base_shift 0.95 terminal 0.1")
    print("samplers: stage1 gradient_estimation, stage2 euler")
    print("loras: distilled 1.0, camera-static 0.3, detailer 0.6 (IC-required)")

    _check_bool(String("non-square pixel geometry"), AS_WIDTH != AS_HEIGHT)
    _check_int(String("latent frames"), nf, 13)
    _check_int(String("decoded frames"), decoded_frames, AS_FRAMES)
    _check_bool(String("non-square latent geometry"), nh != nw)
    _check_int(String("stage1 latent height"), nh, 16)
    _check_int(String("stage1 latent width"), nw, 24)
    _check_int(String("stage1 video tokens"), sv, 4992)
    _check_int(String("stage2 latent height"), s2_h, 32)
    _check_int(String("stage2 latent width"), s2_w, 48)
    _check_int(String("stage2 video tokens"), sv2, 19968)
    _check_int(String("video-duration audio tokens"), audio_tokens_from_video, 101)
    _check_int(String("fixture audio tokens"), audio_tokens_from_fixture, 162)
    _check_int(String("stage1 video pad"), vpad, 4992)
    _check_int(String("stage2 video pad"), vpad2, 19968)
    _check_int(String("audio pad"), apad, 1024)
    _check_int(String("scheduler steps"), SCHED_STEPS, 20)
    _check_int(String("scheduler max shift x100"), SCHED_MAX_SHIFT_HUNDREDTHS, 205)
    _check_int(String("scheduler base shift x100"), SCHED_BASE_SHIFT_HUNDREDTHS, 95)
    _check_int(String("scheduler terminal x10"), SCHED_TERMINAL_TENTHS, 1)
    _check_int(String("distilled lora x10"), LORA_DISTILLED_TENTHS, 10)
    _check_int(String("camera-static lora x10"), LORA_CAMERA_STATIC_TENTHS, 3)
    _check_int(String("detailer lora x10"), LORA_DETAILER_TENTHS, 6)

    var stage1_score_mib = _score_mib(sv)
    var stage2_score_mib = _score_mib(sv2)
    print("naive F32 attention score slab: stage1", stage1_score_mib,
          "MiB, stage2", stage2_score_mib, "MiB")
    _check_bool(String("stage1 requires large attention path"), sv >= LARGE_ATTENTION_THRESHOLD)
    _check_bool(String("stage2 requires large attention path"), sv2 >= LARGE_ATTENTION_THRESHOLD)
    _check_bool(String("stage2 naive scores exceed GPU budget"), stage2_score_mib > GPU_BUDGET_MIB)

    print("LTX2 AUDIOSYNC PROFILE PASS")
