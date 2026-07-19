# audio_ref_ic.mojo -- LTX2 audio-reference IC conditioning contracts.

from std.collections import List


def _require_positive(name: String, value: Int) raises:
    if value <= 0:
        raise Error(name + String(" must be > 0"))


def _require_nonnegative(name: String, value: Int) raises:
    if value < 0:
        raise Error(name + String(" must be >= 0"))


def _require_len(name: String, got: Int, expected: Int) raises:
    if got != expected:
        raise Error(
            name + String(" length mismatch: got ") + String(got)
            + String(", expected ") + String(expected)
        )


def audio_ref_total_time(ref_time: Int, target_time: Int) raises -> Int:
    _require_nonnegative("ref_time", ref_time)
    _require_positive("target_time", target_time)
    return ref_time + target_time


def prepend_audio_ref_latents(
    ref_latents: List[Float32],
    target_latents: List[Float32],
    batch: Int,
    channels: Int,
    ref_time: Int,
    target_time: Int,
    mel: Int,
) raises -> List[Float32]:
    _require_positive("batch", batch)
    _require_positive("channels", channels)
    _require_nonnegative("ref_time", ref_time)
    _require_positive("target_time", target_time)
    _require_positive("mel", mel)
    _require_len("ref_latents", len(ref_latents), batch * channels * ref_time * mel)
    _require_len("target_latents", len(target_latents), batch * channels * target_time * mel)

    var total_time = ref_time + target_time
    var out = List[Float32]()
    for b in range(batch):
        for c in range(channels):
            for t in range(total_time):
                for m in range(mel):
                    if t < ref_time:
                        var ri = (((b * channels + c) * ref_time + t) * mel + m)
                        out.append(ref_latents[ri])
                    else:
                        var tt = t - ref_time
                        var ti = (((b * channels + c) * target_time + tt) * mel + m)
                        out.append(target_latents[ti])
    return out^


def prepend_zero_ref_target(
    target: List[Float32],
    batch: Int,
    channels: Int,
    ref_time: Int,
    target_time: Int,
    mel: Int,
) raises -> List[Float32]:
    _require_len("target", len(target), batch * channels * target_time * mel)
    var zeros = List[Float32]()
    for _ in range(batch * channels * ref_time * mel):
        zeros.append(Float32(0.0))
    return prepend_audio_ref_latents(zeros, target, batch, channels, ref_time, target_time, mel)


def prepend_zero_ref_timesteps(
    target_timesteps: List[Float32],
    batch: Int,
    ref_time: Int,
    target_time: Int,
) raises -> List[Float32]:
    _require_positive("batch", batch)
    _require_nonnegative("ref_time", ref_time)
    _require_positive("target_time", target_time)
    if len(target_timesteps) != batch and len(target_timesteps) != batch * target_time:
        raise Error("target_timesteps must be [B] scalar-per-batch or [B,T]")

    var out = List[Float32]()
    for b in range(batch):
        for _ in range(ref_time):
            out.append(Float32(0.0))
        for t in range(target_time):
            if len(target_timesteps) == batch:
                out.append(target_timesteps[b])
            else:
                out.append(target_timesteps[b * target_time + t])
    return out^


def prepend_ref_false_loss_mask(
    target_mask: List[Bool],
    batch: Int,
    ref_time: Int,
    target_time: Int,
) raises -> List[Bool]:
    _require_len("target_mask", len(target_mask), batch * target_time)
    var out = List[Bool]()
    for b in range(batch):
        for _ in range(ref_time):
            out.append(False)
        for t in range(target_time):
            out.append(target_mask[b * target_time + t])
    return out^


def audio_ref_position_times(
    ref_time: Int,
    target_time: Int,
    use_negative_ref_positions: Bool,
    time_per_latent: Float32,
) raises -> List[Float32]:
    _require_nonnegative("ref_time", ref_time)
    _require_positive("target_time", target_time)
    if time_per_latent <= Float32(0.0):
        raise Error("time_per_latent must be > 0")

    var out = List[Float32]()
    var ref_shift = Float32(0.0)
    if use_negative_ref_positions and ref_time > 0:
        ref_shift = Float32(ref_time - 1) * time_per_latent + time_per_latent
    for t in range(ref_time):
        out.append(Float32(t) * time_per_latent - ref_shift)
    for t in range(target_time):
        out.append(Float32(t) * time_per_latent)
    return out^


def audio_ref_a2v_reference_mask(
    batch: Int,
    video_tokens: Int,
    ref_time: Int,
    target_time: Int,
) raises -> List[Bool]:
    _require_positive("batch", batch)
    _require_positive("video_tokens", video_tokens)
    var total_audio = audio_ref_total_time(ref_time, target_time)
    var out = List[Bool]()
    for _b in range(batch):
        for _v in range(video_tokens):
            for a in range(total_audio):
                out.append(a < ref_time)
    return out^


def audio_ref_text_reference_mask(
    text_valid: List[Bool],
    batch: Int,
    ref_time: Int,
    target_time: Int,
    text_tokens: Int,
) raises -> List[Bool]:
    _require_positive("batch", batch)
    _require_positive("text_tokens", text_tokens)
    _require_len("text_valid", len(text_valid), batch * text_tokens)
    var total_audio = audio_ref_total_time(ref_time, target_time)
    var out = List[Bool]()
    for b in range(batch):
        for a in range(total_audio):
            for txt in range(text_tokens):
                out.append(a < ref_time or not text_valid[b * text_tokens + txt])
    return out^
