# audio_buckets.mojo -- LTX-2 AV audio bucket policy contract.
#
# Mirrors the Musubi LTX2 dataset/bucket behavior without implementing the
# full dataloader. The future collation path should use these functions as its
# readiness surface before it batches video/audio/text tensors.

from serenitymojo.training.ltx2.cache_records import (
    ARCHITECTURE_LTX2,
    ARCHITECTURE_LTX2_FULL,
)


comptime AUDIO_BUCKET_PAD = 0
comptime AUDIO_BUCKET_TRUNCATE = 1

comptime AUDIO_SAMPLER_DISABLED = 0
comptime AUDIO_SAMPLER_QUOTA = 1
comptime AUDIO_SAMPLER_PROBABILITY = 2

comptime LTX2_TARGET_FPS = Float32(25.0)


def audio_bucket_strategy_from_string(strategy: String) raises -> Int:
    if strategy == "pad":
        return AUDIO_BUCKET_PAD
    if strategy == "truncate":
        return AUDIO_BUCKET_TRUNCATE
    raise Error(String("audio_bucket_strategy must be 'pad' or 'truncate', got '") + strategy + String("'"))


def audio_bucket_strategy_name(strategy: Int) raises -> String:
    if strategy == AUDIO_BUCKET_PAD:
        return String("pad")
    if strategy == AUDIO_BUCKET_TRUNCATE:
        return String("truncate")
    raise Error("invalid audio bucket strategy id")


def _round_positive_to_int(x: Float32) -> Int:
    return Int(Float64(x) + Float64(0.5))


def audio_bucket_step_frames(
    audio_bucket_interval_seconds: Float32,
    target_fps: Float32 = LTX2_TARGET_FPS,
) -> Int:
    var step = _round_positive_to_int(audio_bucket_interval_seconds * target_fps)
    if step < 1:
        return 1
    return step


def quantize_audio_latent_time(audio_t: Int, bucket_step: Int, strategy: Int) raises -> Int:
    if bucket_step < 1:
        raise Error("audio bucket step must be >= 1")
    if audio_t < 1:
        raise Error("audio latent time must be >= 1")

    if strategy == AUDIO_BUCKET_TRUNCATE:
        var q = (audio_t // bucket_step) * bucket_step
        if q < bucket_step:
            return bucket_step
        return q
    if strategy == AUDIO_BUCKET_PAD:
        # Musubi's current LTX2 audio bucket path adds half a bucket before
        # integer division. This is a round-to-nearest bucket policy.
        var q = ((audio_t + bucket_step // 2) // bucket_step) * bucket_step
        if q < bucket_step:
            return bucket_step
        return q
    raise Error("invalid audio bucket strategy id")


def append_audio_bucket_key(
    base_key: String,
    architecture: String,
    separate_audio_buckets: Bool,
    has_audio: Bool,
) -> String:
    if not separate_audio_buckets:
        return base_key
    if architecture != String(ARCHITECTURE_LTX2) and architecture != String(ARCHITECTURE_LTX2_FULL):
        return base_key
    if has_audio:
        return base_key + String("|audio=1")
    return base_key + String("|audio=0")


def audio_sampler_mode(
    gradient_accumulation_steps: Int,
    min_audio_batches_per_accum: Int,
    audio_batch_probability_is_set: Bool,
    audio_batch_probability: Float32,
    has_audio_batches: Bool,
) raises -> Int:
    var min_audio = min_audio_batches_per_accum
    if min_audio < 0:
        min_audio = 0

    if audio_batch_probability_is_set:
        if audio_batch_probability < Float32(0.0) or audio_batch_probability > Float32(1.0):
            raise Error("audio_batch_probability must be in [0, 1]")

    if min_audio > 0 and audio_batch_probability_is_set:
        raise Error("--min_audio_batches_per_accum and --audio_batch_probability are mutually exclusive")

    if min_audio <= 0 and not audio_batch_probability_is_set:
        return AUDIO_SAMPLER_DISABLED

    if min_audio > 0 and min_audio > gradient_accumulation_steps:
        raise Error("min_audio_batches_per_accum must be <= gradient_accumulation_steps")

    if not has_audio_batches:
        if min_audio > 0:
            raise Error("--min_audio_batches_per_accum is set, but no audio-bearing batches were found")
        raise Error("--audio_batch_probability is set, but no audio-bearing batches were found")

    if min_audio > 0:
        return AUDIO_SAMPLER_QUOTA
    return AUDIO_SAMPLER_PROBABILITY


def audio_sampler_mode_name(mode: Int) raises -> String:
    if mode == AUDIO_SAMPLER_DISABLED:
        return String("disabled")
    if mode == AUDIO_SAMPLER_QUOTA:
        return String("quota")
    if mode == AUDIO_SAMPLER_PROBABILITY:
        return String("probability")
    raise Error("invalid audio sampler mode")
