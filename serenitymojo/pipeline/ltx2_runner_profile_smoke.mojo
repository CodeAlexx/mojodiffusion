# pipeline/ltx2_runner_profile_smoke.mojo
#
# Fast contract for renderer knobs that must not silently drift from the LTX2
# AudioSync production profile.

from serenitymojo.pipeline.ltx2_t2v_av_hq import (
    HQ_STEPS,
    LORA_CAMERA_STATIC_MULT,
    LORA_DETAILER_MULT,
    LORA_DISTILLED_MULT,
    NH,
    NW,
    NUM_FRAMES_AUDIOSYNC,
    S_V_AUDIOSYNC,
)


def _check_bool(name: String, got: Bool) raises:
    if not got:
        raise Error(name + String(" failed"))


def _check_int(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            name + String(" mismatch: got ") + String(got)
            + String(" expected ") + String(expected)
        )


def _check_scale(name: String, got: Float32, expected: Float32) raises:
    var delta = got - expected
    if delta < Float32(0.0):
        delta = -delta
    if delta > Float32(0.0001):
        raise Error(name + String(" scale mismatch"))


def main() raises:
    print("=== LTX2 runner profile contract ===")
    print("audiosync frames:", NUM_FRAMES_AUDIOSYNC)
    print("hq denoise steps:", HQ_STEPS)
    print("latent geometry nh/nw:", NH, NW)
    print("audiosync video tokens:", S_V_AUDIOSYNC)
    print("loras: distilled", LORA_DISTILLED_MULT,
          "camera-static", LORA_CAMERA_STATIC_MULT,
          "detailer", LORA_DETAILER_MULT, "(disabled until IC)")

    _check_int(String("audiosync frames"), NUM_FRAMES_AUDIOSYNC, 97)
    _check_int(String("hq denoise steps"), HQ_STEPS, 20)
    _check_bool(String("non-square latent geometry"), NH != NW)
    _check_int(String("audiosync video tokens"), S_V_AUDIOSYNC, 4992)
    _check_scale(String("distilled lora"), LORA_DISTILLED_MULT, Float32(1.0))
    _check_scale(String("camera-static lora"), LORA_CAMERA_STATIC_MULT, Float32(0.3))
    _check_scale(String("detailer lora raw-runtime"), LORA_DETAILER_MULT, Float32(0.0))

    print("LTX2 RUNNER PROFILE PASS")
