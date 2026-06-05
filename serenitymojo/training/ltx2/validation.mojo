# validation.mojo -- LTX-2 validation sampling contract stubs.

from serenitymojo.training.ltx2.config import LTX2TrainerConfig


@fieldwise_init
struct ValidationContract(Copyable, Movable):
    var prompt_cache_path: String
    var sample_latents_cache_path: String
    var sample_every: Int
    var baseline_control: Bool
    var merge_audio: Bool
    var two_stage: Bool
    var tiled_vae: Bool

    def enabled(self) -> Bool:
        return self.sample_every > 0 and self.prompt_cache_path != ""


def default_validation_contract(cfg: LTX2TrainerConfig) -> ValidationContract:
    return ValidationContract(
        cfg.validation_prompts_cache,
        cfg.sample_latents_cache,
        cfg.sample_every,
        True,
        True,
        True,
        True,
    )


def sample_stem(step: Int) -> String:
    var raw = String(step)
    var out = String("")
    var pads = 6 - raw.byte_length()
    for _ in range(pads):
        out += String("0")
    return out + raw


def sample_video_path(output_dir: String, step: Int) -> String:
    return output_dir + String("/samples/") + sample_stem(step) + String(".mp4")


def sample_audio_path(output_dir: String, step: Int) -> String:
    return output_dir + String("/samples/") + sample_stem(step) + String(".wav")


def sample_muxed_av_path(output_dir: String, step: Int) -> String:
    return output_dir + String("/samples/") + sample_stem(step) + String("_av.mp4")


def validation_sampling_ready(has_av_sampler: Bool, has_video_vae_decode: Bool, has_vocoder: Bool) -> Bool:
    return has_av_sampler and has_video_vae_decode and has_vocoder


def run_validation_sample_stub() raises:
    raise Error("LTX2 validation sampling is a contract stub until train-time AV LoRA application is wired")
