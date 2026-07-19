# schedule.mojo -- LTX-2 sigma/noise/loss schedule helpers.

from std.math import exp
from serenitymojo.training.loss_weight import apply_loss_weight, combined_loss_value
from serenitymojo.training.lr_schedule import lr_for_step
from serenitymojo.training.noise_modifiers import apply_noise_modifiers_host
from serenitymojo.training.timestep_bias import apply_bias


def clamp01(x: Float32) -> Float32:
    if x < Float32(0.0):
        return Float32(0.0)
    if x > Float32(1.0):
        return Float32(1.0)
    return x


def sigmoid32(x: Float32) -> Float32:
    return Float32(1.0) / (Float32(1.0) + exp(-x))


def shift_for_sequence_length(
    seq_length: Int,
    min_tokens: Int = 1024,
    max_tokens: Int = 4096,
    min_shift: Float32 = 0.95,
    max_shift: Float32 = 2.05,
) -> Float32:
    var m = (max_shift - min_shift) / Float32(max_tokens - min_tokens)
    var b = min_shift - m * Float32(min_tokens)
    return m * Float32(seq_length) + b


def shifted_logit_normal_sigma_legacy(normal_z: Float32, shift: Float32, std: Float32 = 1.0) -> Float32:
    return sigmoid32(normal_z * std + shift)


def shifted_logit_normal_sigma_stretched(
    normal_z: Float32,
    uniform01: Float32,
    branch01: Float32,
    shift: Float32,
    std: Float32 = 1.0,
    eps: Float32 = 1.0e-3,
    uniform_prob: Float32 = 0.1,
) -> Float32:
    var e = eps
    if e < Float32(0.0):
        e = Float32(0.0)
    if e > Float32(0.499):
        e = Float32(0.499)
    var up = clamp01(uniform_prob)
    var raw = shifted_logit_normal_sigma_legacy(normal_z, shift, std)
    var p999 = sigmoid32(shift + Float32(3.0902) * std)
    var p005 = sigmoid32(shift - Float32(2.5758) * std)
    var denom = p999 - p005
    if denom < Float32(1.0e-6):
        denom = Float32(1.0e-6)
    var stretched = (raw - p005) / denom
    if stretched < e:
        stretched = Float32(2.0) * e - stretched
    stretched = clamp01(stretched)
    var uniform = (Float32(1.0) - e) * clamp01(uniform01) + e
    if up <= Float32(0.0):
        return stretched
    if up >= Float32(1.0):
        return uniform
    if clamp01(branch01) > up:
        return stretched
    return uniform


def apply_timestep_range(sigma: Float32, min_timestep: Float32, max_timestep: Float32) -> Float32:
    var min_sigma = min_timestep / Float32(1000.0)
    var max_sigma = max_timestep / Float32(1000.0)
    return sigma * (max_sigma - min_sigma) + min_sigma


def ltx2_lr_for_step(
    base_lr: Float32,
    step: Int,
    warmup_steps: Int,
    total_steps: Int,
    kind: Int,
    min_factor: Float32,
    cycles: Float32,
    power: Float32,
) -> Float32:
    return lr_for_step(base_lr, step, warmup_steps, total_steps, kind, min_factor, cycles, power)


def ltx2_apply_timestep_bias(
    sigma: Float32,
    strategy: Int,
    multiplier: Float32,
    range_min: Float32,
    range_max: Float32,
) -> Float32:
    return apply_bias(sigma, Float32(1.0), strategy, multiplier, range_min, range_max)


def sigma_to_model_timestep(sigma: Float32) -> Float32:
    return sigma * Float32(1000.0)


def model_timestep_to_sigma(timestep: Float32) -> Float32:
    return timestep / Float32(1000.0)


@fieldwise_init
struct LTX2AVTimestepHandoff(Copyable, Movable):
    var video_head_sigma: Float32
    var audio_sigma: Float32
    var coupled_audio_timestep: Bool


def normalize_musubi_training_timestep(timestep: Float32) -> Float32:
    # Musubi emits training timesteps as sigma * 1000 and normalizes before DiT.
    return model_timestep_to_sigma(timestep)


def independent_audio_sigma_from_uniform(
    uniform01: Float32,
    min_timestep: Float32 = 0.0,
    max_timestep: Float32 = 1000.0,
) raises -> Float32:
    if uniform01 < Float32(0.0) or uniform01 > Float32(1.0):
        raise Error("audio timestep uniform sample must be in [0, 1]")
    var min_sigma = normalize_musubi_training_timestep(min_timestep)
    var max_sigma = normalize_musubi_training_timestep(max_timestep)
    if max_sigma < min_sigma:
        raise Error("invalid independent audio timestep range")
    return uniform01 * (max_sigma - min_sigma) + min_sigma


def coupled_audio_sigma_from_timestep_row(raw_video_timesteps: List[Float32]) raises -> Float32:
    if len(raw_video_timesteps) == 0:
        raise Error("expected at least one video timestep")
    return normalize_musubi_training_timestep(raw_video_timesteps[0])


def prepare_av_model_sigmas_from_scalar_timestep(
    raw_video_timestep: Float32,
    independent_audio_timestep: Bool,
    audio_uniform01: Float32 = 0.0,
    min_timestep: Float32 = 0.0,
    max_timestep: Float32 = 1000.0,
) raises -> LTX2AVTimestepHandoff:
    var video_sigma = normalize_musubi_training_timestep(raw_video_timestep)
    if independent_audio_timestep:
        return LTX2AVTimestepHandoff(
            video_sigma,
            independent_audio_sigma_from_uniform(audio_uniform01, min_timestep, max_timestep),
            False,
        )
    return LTX2AVTimestepHandoff(video_sigma, video_sigma, True)


def prepare_av_model_sigmas_from_token_timesteps(
    raw_video_timesteps: List[Float32],
    independent_audio_timestep: Bool,
    audio_uniform01: Float32 = 0.0,
    min_timestep: Float32 = 0.0,
    max_timestep: Float32 = 1000.0,
) raises -> LTX2AVTimestepHandoff:
    var video_sigma = coupled_audio_sigma_from_timestep_row(raw_video_timesteps)
    if independent_audio_timestep:
        return LTX2AVTimestepHandoff(
            video_sigma,
            independent_audio_sigma_from_uniform(audio_uniform01, min_timestep, max_timestep),
            False,
        )
    return LTX2AVTimestepHandoff(video_sigma, video_sigma, True)


def flow_match_noisy_value(latent: Float32, noise: Float32, sigma: Float32) -> Float32:
    return (Float32(1.0) - sigma) * latent + sigma * noise


def flow_match_target_value(noise: Float32, latent: Float32) -> Float32:
    return noise - latent


def velocity_to_x0_value(latent: Float32, velocity: Float32, sigma: Float32) -> Float32:
    return latent - velocity * sigma


def apply_ltx2_noise_modifiers_host(
    mut noise: List[Float32],
    n_tokens: Int,
    channels: Int,
    offset_weight: Float32,
    offset_prob: Float32,
    input_perturb_gamma: Float32,
    multires_iterations: Int,
    multires_discount: Float32,
    step_seed: UInt64,
) raises -> Bool:
    return apply_noise_modifiers_host(
        noise,
        n_tokens,
        channels,
        offset_weight,
        offset_prob,
        input_perturb_gamma,
        multires_iterations,
        multires_discount,
        step_seed,
    )


def weighted_combined_loss(
    pred: List[Float32],
    target: List[Float32],
    sigma: Float32,
    min_snr_gamma: Float32,
    debiased: Bool,
    mse_strength: Float32,
    mae_strength: Float32,
    huber_strength: Float32,
) raises -> Float32:
    var base = combined_loss_value(pred, target, mse_strength, mae_strength, huber_strength)
    var w = apply_loss_weight(sigma, min_snr_gamma, debiased, True)
    return base * w


def av_weighted_loss_value(
    video_loss: Float32,
    audio_loss: Float32,
    has_audio: Bool,
    video_weight: Float32,
    audio_weight: Float32,
) -> Float32:
    if has_audio:
        return video_loss * video_weight + audio_loss * audio_weight
    return video_loss * video_weight
