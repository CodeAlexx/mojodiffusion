# acceptance.mojo -- contract acceptance runner for the LTX-2 AV trainer spine.

from std.math import abs
from serenitymojo.training.ltx2.cache_records import (
    audio_latents_key,
    audio_lengths_key,
    video_latents_key,
    video_prompt_embeds_key,
    audio_prompt_embeds_key,
)
from serenitymojo.training.ltx2.conditioning import select_conditioning
from serenitymojo.training.ltx2.config import MODE_AV, PRESET_T2V, LTX2TrainerConfig
from serenitymojo.training.ltx2.lora_surface import (
    DEFAULT_T2V_TARGETS_TOTAL,
    diffusion_lora_a_key,
    modules_per_block_for_preset,
    target_count_for_preset,
)
from serenitymojo.training.ltx2.schedule import (
    flow_match_noisy_value,
    flow_match_target_value,
    shifted_logit_normal_sigma_legacy,
    weighted_combined_loss,
)
from serenitymojo.training.ltx2.readiness import LTX2Readiness, default_readiness


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("LTX2 acceptance failed: ") + msg)


def _close(a: Float32, b: Float32, tol: Float32 = 1.0e-5) -> Bool:
    return abs(a - b) <= tol


def run_acceptance(print_details: Bool = True) raises -> LTX2Readiness:
    var cfg = LTX2TrainerConfig.default()
    cfg.validate()
    _check(cfg.ltx_mode == MODE_AV, "default trainer mode must be AV")

    _check(
        video_latents_key(9, 16, 16, "bf16") == "latents_9x16x16_bf16",
        "video latent key contract",
    )
    _check(
        audio_latents_key(100, 80, 8, "bf16") == "audio_latents_100x80x8_bf16",
        "audio latent key contract",
    )
    _check(audio_lengths_key() == "audio_lengths_int32", "audio length key contract")
    _check(video_prompt_embeds_key("bf16") == "video_prompt_embeds_bf16", "video text key")
    _check(audio_prompt_embeds_key("bf16") == "audio_prompt_embeds_bf16", "audio text key")

    var sel_av = select_conditioning(MODE_AV, True, True, True, 6144, 4096, 2048, False)
    _check(sel_av.video_enabled and sel_av.audio_enabled, "AV batch must enable both branches")
    _check(sel_av.split_combined_context, "AV combined context must split")
    var sel_no_audio = select_conditioning(MODE_AV, False, True, False, 4096, 4096, 2048, False)
    _check(sel_no_audio.video_enabled and not sel_no_audio.audio_enabled, "AV no-audio batch must fall back to video")

    _check(target_count_for_preset(PRESET_T2V) == DEFAULT_T2V_TARGETS_TOTAL, "T2V LoRA target count")
    _check(len(modules_per_block_for_preset(PRESET_T2V)) == 24, "T2V modules per block")
    _check(
        diffusion_lora_a_key(0, "attn1.to_q")
        == "diffusion_model.transformer_blocks.0.attn1.to_q.lora_A.weight",
        "diffusion LoRA key",
    )

    _check(_close(flow_match_noisy_value(2.0, 10.0, 0.25), 4.0), "flow noisy value")
    _check(_close(flow_match_target_value(10.0, 2.0), 8.0), "flow target value")
    var sigma = shifted_logit_normal_sigma_legacy(0.0, 0.0, 1.0)
    _check(_close(sigma, 0.5), "legacy shifted-logit sigma")
    var pred = List[Float32]()
    pred.append(1.0)
    pred.append(2.0)
    var target = List[Float32]()
    target.append(0.0)
    target.append(4.0)
    var loss = weighted_combined_loss(pred^, target^, 0.5, -1.0, False, 1.0, 0.0, 0.0)
    _check(_close(loss, 2.5), "weighted MSE default-off")

    var report = default_readiness()
    if print_details:
        print("LTX2 acceptance: PASS foundation contracts")
        print("  T2V target adapters:", target_count_for_preset(PRESET_T2V))
        print("  production_training_ready:", report.production_training_ready())
        print("  blocker: full AV backward and train-time AV LoRA runtime are not wired")
    return report^
