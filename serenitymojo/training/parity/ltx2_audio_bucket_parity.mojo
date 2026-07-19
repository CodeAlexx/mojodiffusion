# training/parity/ltx2_audio_bucket_parity.mojo
#
# Musubi LTX2 AV audio bucket policy gate.
#
# Run:
#   pixi run mojo run -I . serenitymojo/training/parity/ltx2_audio_bucket_parity.mojo

from serenitymojo.training.ltx2.audio_buckets import (
    AUDIO_BUCKET_PAD,
    AUDIO_BUCKET_TRUNCATE,
    AUDIO_SAMPLER_DISABLED,
    AUDIO_SAMPLER_PROBABILITY,
    AUDIO_SAMPLER_QUOTA,
    append_audio_bucket_key,
    audio_bucket_step_frames,
    audio_bucket_strategy_from_string,
    audio_bucket_strategy_name,
    audio_sampler_mode,
    audio_sampler_mode_name,
    quantize_audio_latent_time,
)
from serenitymojo.training.ltx2.cache_records import (
    ARCHITECTURE_LTX2,
    ARCHITECTURE_LTX2_FULL,
)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("LTX2 audio bucket parity failed: ") + msg)


def _expect_raise_bad_strategy() raises -> Bool:
    try:
        _ = audio_bucket_strategy_from_string("nearest")
    except e:
        return True
    return False


def _expect_raise_sampler_conflict() raises -> Bool:
    try:
        _ = audio_sampler_mode(4, 1, True, Float32(0.5), True)
    except e:
        return True
    return False


def _expect_raise_sampler_probability_range() raises -> Bool:
    try:
        _ = audio_sampler_mode(4, 0, True, Float32(1.25), True)
    except e:
        return True
    return False


def _expect_raise_missing_audio_batches() raises -> Bool:
    try:
        _ = audio_sampler_mode(4, 2, False, Float32(0.0), False)
    except e:
        return True
    return False


def run_ltx2_audio_bucket_parity(print_details: Bool = True) raises:
    if print_details:
        print("--- Musubi audio bucket parity ---")

    _check(audio_bucket_strategy_from_string("pad") == AUDIO_BUCKET_PAD, "pad strategy parser")
    _check(audio_bucket_strategy_from_string("truncate") == AUDIO_BUCKET_TRUNCATE, "truncate strategy parser")
    _check(audio_bucket_strategy_name(AUDIO_BUCKET_PAD) == "pad", "pad strategy name")
    _check(audio_bucket_strategy_name(AUDIO_BUCKET_TRUNCATE) == "truncate", "truncate strategy name")
    _check(_expect_raise_bad_strategy(), "invalid strategy must fail closed")

    _check(audio_bucket_step_frames(Float32(2.0)) == 50, "default 2s LTX2 bucket step")
    _check(audio_bucket_step_frames(Float32(0.01)) == 1, "minimum bucket step")
    _check(audio_bucket_step_frames(Float32(1.2), Float32(25.0)) == 30, "rounded bucket step")

    var step = audio_bucket_step_frames(Float32(2.0))
    _check(quantize_audio_latent_time(24, step, AUDIO_BUCKET_PAD) == 50, "pad min bucket")
    _check(quantize_audio_latent_time(124, step, AUDIO_BUCKET_PAD) == 100, "pad lower nearest bucket")
    _check(quantize_audio_latent_time(126, step, AUDIO_BUCKET_PAD) == 150, "pad upper nearest bucket")
    _check(quantize_audio_latent_time(124, step, AUDIO_BUCKET_TRUNCATE) == 100, "truncate floor bucket")
    _check(quantize_audio_latent_time(24, step, AUDIO_BUCKET_TRUNCATE) == 50, "truncate min bucket")

    _check(
        append_audio_bucket_key("768x512x49", String(ARCHITECTURE_LTX2), True, True)
        == "768x512x49|audio=1",
        "LTX2 audio-bearing bucket suffix",
    )
    _check(
        append_audio_bucket_key("768x512x49", String(ARCHITECTURE_LTX2_FULL), True, False)
        == "768x512x49|audio=0",
        "LTX2 full non-audio bucket suffix",
    )
    _check(
        append_audio_bucket_key("768x512x49", "wan", True, True) == "768x512x49",
        "non-LTX2 architecture must not split audio buckets",
    )
    _check(
        append_audio_bucket_key("768x512x49", String(ARCHITECTURE_LTX2), False, True)
        == "768x512x49",
        "separate_audio_buckets disabled",
    )

    _check(
        audio_sampler_mode(4, 0, False, Float32(0.0), True) == AUDIO_SAMPLER_DISABLED,
        "sampler disabled by default",
    )
    _check(
        audio_sampler_mode(4, 2, False, Float32(0.0), True) == AUDIO_SAMPLER_QUOTA,
        "quota sampler mode",
    )
    _check(
        audio_sampler_mode(4, 0, True, Float32(0.35), True) == AUDIO_SAMPLER_PROBABILITY,
        "probability sampler mode",
    )
    _check(audio_sampler_mode_name(AUDIO_SAMPLER_DISABLED) == "disabled", "disabled sampler name")
    _check(audio_sampler_mode_name(AUDIO_SAMPLER_QUOTA) == "quota", "quota sampler name")
    _check(audio_sampler_mode_name(AUDIO_SAMPLER_PROBABILITY) == "probability", "probability sampler name")
    _check(_expect_raise_sampler_conflict(), "quota/probability mutual exclusion")
    _check(_expect_raise_sampler_probability_range(), "probability range validation")
    _check(_expect_raise_missing_audio_batches(), "missing audio batches must fail closed")

    if print_details:
        print("  PASS audio bucket policy")


def main() raises:
    print("=== LTX2 audio bucket parity ===")
    run_ltx2_audio_bucket_parity()
    print("LTX2 audio bucket parity PASS")
