# training/parity/ltx2_trainer_parity_audit.mojo
#
# Executable parity audit for the LTX-2 trainer surface.
#
# This gate compares the current Mojo LTX2 trainer contracts against the
# Musubi/LTX2 behavior that must be covered before the trainer can be called
# production AV-ready. It intentionally passes when foundation contracts match
# and production blockers are explicitly tracked.
#
# Run:
#   pixi run mojo run -I . serenitymojo/training/parity/ltx2_trainer_parity_audit.mojo
#
# Production fail gate:
#   pixi run mojo run -I . serenitymojo/training/parity/ltx2_trainer_parity_audit.mojo --expect-production

from std.sys import argv
from std.collections import List
from std.math import abs

from serenitymojo.training.ltx2.audio_buckets import (
    AUDIO_BUCKET_PAD,
    AUDIO_BUCKET_TRUNCATE,
    AUDIO_SAMPLER_DISABLED,
    AUDIO_SAMPLER_PROBABILITY,
    AUDIO_SAMPLER_QUOTA,
    append_audio_bucket_key,
    audio_bucket_step_frames,
    audio_bucket_strategy_from_string,
    audio_sampler_mode,
    quantize_audio_latent_time,
)
from serenitymojo.training.ltx2.cache_records import (
    AUDIO_CACHE_SUFFIX,
    ARCHITECTURE_LTX2,
    ARCHITECTURE_LTX2_FULL,
    DINO_CACHE_SUFFIX,
    FORMAT_VERSION,
    TEXT_CACHE_SUFFIX,
    VIDEO_CACHE_SUFFIX,
    audio_latents_key,
    audio_lengths_key,
    default_record,
    latent_metadata_contract,
    legacy_text_key,
    prompt_attention_mask_key,
    text_metadata_contract,
    video_clean_latents_key,
    video_latents_key,
)
from serenitymojo.training.ltx2.checkpointing import (
    lobo_step_lora_filename,
    lora_emergency_path,
    lora_final_path,
    lora_latest_path,
    musubi_step_lora_filename,
    resume_checkpoint_contract,
    resume_token_is_latest,
    save_checkpoint_contract,
    train_state_filename,
)
from serenitymojo.training.ltx2.conditioning import (
    TEXT_SOURCE_COMBINED_AV,
    TEXT_SOURCE_SPLIT_VIDEO,
    can_split_combined_context,
    combined_context_audio_dim,
    combined_context_video_dim,
    select_conditioning,
)
from serenitymojo.training.ltx2.config import (
    MODE_AUDIO,
    MODE_AV,
    PRESET_AUDIO,
    PRESET_AUDIO_REF_ONLY_IC,
    PRESET_FULL,
    PRESET_T2V,
    PRESET_V2V,
    SHIFT_STRETCHED,
    SIGMA_SHIFTED_LOGIT_NORMAL,
    VERSION_23,
    LTX2TrainerConfig,
)
from serenitymojo.training.ltx2.lora_surface import (
    DEFAULT_T2V_TARGETS_TOTAL,
    audio_modules_per_block,
    audio_ref_only_ic_modules_per_block,
    diffusion_lora_a_key,
    diffusion_lora_b_key,
    modules_per_block_for_preset,
    target_count_for_preset,
    v2v_modules_per_block,
)
from serenitymojo.training.ltx2.masked_loss import (
    LTX2_LOSS_MSE,
    masked_loss_audio_bt_mask,
    masked_loss_video_bf_mask,
)
from serenitymojo.training.ltx2.schedule import (
    apply_timestep_range,
    av_weighted_loss_value,
    flow_match_noisy_value,
    flow_match_target_value,
    independent_audio_sigma_from_uniform,
    normalize_musubi_training_timestep,
    prepare_av_model_sigmas_from_scalar_timestep,
    prepare_av_model_sigmas_from_token_timesteps,
    shifted_logit_normal_sigma_legacy,
    shifted_logit_normal_sigma_stretched,
    shift_for_sequence_length,
    sigma_to_model_timestep,
)
from serenitymojo.training.ltx2.validation import (
    default_validation_contract,
    sample_audio_path,
    sample_muxed_av_path,
    sample_video_path,
    validation_sampling_ready,
)


def _check(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(String("LTX2 trainer parity audit failed: ") + msg)


def _close(a: Float32, b: Float32, tol: Float32 = 1.0e-5) -> Bool:
    return abs(a - b) <= tol


def _make_args() -> List[String]:
    var result = List[String]()
    result.append("ltx2_trainer_parity_audit")
    result.append("--ltx2_checkpoint")
    result.append("/models/ltx-2.3-22b-dev.safetensors")
    result.append("--dataset_cache_dir=/cache/ltx2")
    result.append("--output_dir")
    result.append("/runs/ltx2")
    result.append("--resume=latest")
    result.append("--validation_prompts_cache")
    result.append("/cache/ltx2/ltx2_sample_prompts_cache.pt")
    result.append("--sample_latents_cache=/cache/ltx2/ltx2_sample_latents_cache.pt")
    result.append("--ltx2_mode")
    result.append("av")
    result.append("--ltx_version=2.3")
    result.append("--lora_target_preset")
    result.append("v2v")
    result.append("--network_dim")
    result.append("32")
    result.append("--network_alpha=32")
    result.append("--learning_rate")
    result.append("1e-4")
    result.append("--max_train_steps=3000")
    result.append("--save_every_n_steps")
    result.append("500")
    result.append("--sample_every_n_steps=250")
    result.append("--video_loss_weight=1.0")
    result.append("--audio_loss_weight=0.3")
    result.append("--independent_audio_timestep")
    result.append("--timestep_sampling")
    result.append("shifted_logit_normal")
    result.append("--shifted_logit_mode=stretched")
    result.append("--audio_only_sequence_resolution")
    result.append("64")
    return result^


def _gate_config_cli() raises:
    print("--- config / CLI parity foundation ---")
    var cfg = LTX2TrainerConfig.from_args(_make_args())
    _check(cfg.ltx2_checkpoint == "/models/ltx-2.3-22b-dev.safetensors", "checkpoint arg")
    _check(cfg.dataset_cache_dir == "/cache/ltx2", "dataset cache arg")
    _check(cfg.output_dir == "/runs/ltx2", "output dir arg")
    _check(cfg.resume_from == "latest", "resume latest arg")
    _check(cfg.validation_prompts_cache == "/cache/ltx2/ltx2_sample_prompts_cache.pt", "sample prompt cache arg")
    _check(cfg.sample_latents_cache == "/cache/ltx2/ltx2_sample_latents_cache.pt", "sample latents cache arg")
    _check(cfg.ltx_mode == MODE_AV, "AV mode arg")
    _check(cfg.ltx_version == VERSION_23, "LTX 2.3 arg")
    _check(cfg.lora_target_preset == PRESET_V2V, "V2V preset arg")
    _check(cfg.lora_rank == 32, "network_dim arg")
    _check(_close(cfg.lora_alpha, 32.0), "network_alpha arg")
    _check(_close(cfg.learning_rate, 1.0e-4), "learning_rate arg")
    _check(cfg.max_steps == 3000, "max steps arg")
    _check(cfg.save_every == 500, "save every arg")
    _check(cfg.sample_every == 250, "sample every arg")
    _check(_close(cfg.video_loss_weight, 1.0), "video loss weight arg")
    _check(_close(cfg.audio_loss_weight, 0.3), "audio loss weight arg")
    _check(cfg.independent_audio_timestep, "independent audio timestep arg")
    _check(cfg.timestep_sampling == SIGMA_SHIFTED_LOGIT_NORMAL, "shifted logit sampling arg")
    _check(cfg.shifted_logit_mode == SHIFT_STRETCHED, "stretched sampler arg")
    _check(cfg.audio_only_sequence_resolution == 64, "audio only virtual sequence resolution")
    print("  PASS config / CLI foundation")


def _gate_cache_records() raises:
    print("--- data / cache record parity foundation ---")
    _check(String(FORMAT_VERSION) == "1.0.1", "Musubi cache format version")
    _check(String(VIDEO_CACHE_SUFFIX) == "_ltx2.safetensors", "video cache suffix")
    _check(String(AUDIO_CACHE_SUFFIX) == "_ltx2_audio.safetensors", "audio cache suffix")
    _check(String(TEXT_CACHE_SUFFIX) == "_ltx2_te.safetensors", "text cache suffix")
    _check(String(DINO_CACHE_SUFFIX) == "_ltx2_dino.safetensors", "DINO cache suffix")
    _check(video_latents_key(17, 64, 96, "bf16") == "latents_17x64x96_bf16", "video latent key")
    _check(video_clean_latents_key(1, 64, 96, "bf16") == "latents_clean_1x64x96_bf16", "clean reference latent key")
    _check(audio_latents_key(320, 80, 8, "bf16") == "audio_latents_320x80x8_bf16", "audio latent key")
    _check(audio_lengths_key() == "audio_lengths_int32", "audio lengths key")
    _check(legacy_text_key("bf16") == "text_bf16", "legacy text key")
    _check(prompt_attention_mask_key() == "prompt_attention_mask", "prompt attention mask key")
    _check(latent_metadata_contract() == "architecture,width,height,format_version,frame_count", "latent metadata contract")
    _check(text_metadata_contract() == "architecture,caption1,format_version", "text metadata contract")
    var rec = default_record(String("/cache/item001"))
    _check(rec.latent_cache_path == "/cache/item001_ltx2.safetensors", "default video cache path")
    _check(rec.audio_latent_cache_path == "/cache/item001_ltx2_audio.safetensors", "default audio cache path")
    _check(not rec.cache_ready() and not rec.av_ready(), "default record must start not-ready")
    print("  PASS cache naming foundation")


def _gate_audio_buckets() raises:
    print("--- audio bucket / AV batch policy parity ---")
    _check(audio_bucket_strategy_from_string("pad") == AUDIO_BUCKET_PAD, "pad strategy parser")
    _check(audio_bucket_strategy_from_string("truncate") == AUDIO_BUCKET_TRUNCATE, "truncate strategy parser")
    var step = audio_bucket_step_frames(Float32(2.0))
    _check(step == 50, "LTX2 default audio bucket step")
    _check(quantize_audio_latent_time(124, step, AUDIO_BUCKET_PAD) == 100, "pad lower nearest bucket")
    _check(quantize_audio_latent_time(126, step, AUDIO_BUCKET_PAD) == 150, "pad upper nearest bucket")
    _check(quantize_audio_latent_time(124, step, AUDIO_BUCKET_TRUNCATE) == 100, "truncate floor bucket")
    _check(
        append_audio_bucket_key("768x512x49", String(ARCHITECTURE_LTX2), True, True)
        == "768x512x49|audio=1",
        "LTX2 audio-bearing bucket split",
    )
    _check(
        append_audio_bucket_key("768x512x49", String(ARCHITECTURE_LTX2_FULL), True, False)
        == "768x512x49|audio=0",
        "LTX2 full non-audio bucket split",
    )
    _check(
        audio_sampler_mode(4, 0, False, Float32(0.0), True) == AUDIO_SAMPLER_DISABLED,
        "audio sampler disabled default",
    )
    _check(
        audio_sampler_mode(4, 2, False, Float32(0.0), True) == AUDIO_SAMPLER_QUOTA,
        "audio quota sampler mode",
    )
    _check(
        audio_sampler_mode(4, 0, True, Float32(0.35), True) == AUDIO_SAMPLER_PROBABILITY,
        "audio probability sampler mode",
    )
    print("  PASS audio bucket / AV batch policy")


def _gate_conditioning() raises:
    print("--- conditioning / modality parity foundation ---")
    _check(can_split_combined_context(6144, 4096, 2048), "2.3 combined context split")
    _check(combined_context_video_dim(6144, 4096, 2048) == 4096, "video context dim")
    _check(combined_context_audio_dim(6144, 4096, 2048) == 2048, "audio context dim")
    var av = select_conditioning(MODE_AV, True, True, True, 6144, 4096, 2048, False)
    _check(av.text_source == TEXT_SOURCE_COMBINED_AV, "AV combined context source")
    _check(av.video_enabled and av.audio_enabled, "AV enables both branches")
    _check(av.split_combined_context, "AV context split enabled")
    _check(av.coupled_timesteps, "AV coupled_timesteps should reflect independent_audio_timestep=false inversion")
    var audio = select_conditioning(MODE_AUDIO, True, False, True, 2048, 4096, 2048, True)
    _check(audio.audio_enabled and not audio.video_enabled, "audio mode branch selection")
    _check(audio.requires_audio_latents, "audio mode requires audio latents")
    var fallback = select_conditioning(MODE_AV, False, False, False, 4096, 4096, 2048, False)
    _check(fallback.text_source == TEXT_SOURCE_SPLIT_VIDEO, "AV no-audio fallback split video")
    _check(fallback.video_enabled and not fallback.audio_enabled, "AV no-audio fallback branches")
    print("  PASS conditioning foundation")


def _gate_schedule_loss() raises:
    print("--- schedule / noise / loss parity foundation ---")
    _check(_close(shift_for_sequence_length(1024), 0.95), "shift at min tokens")
    _check(_close(shift_for_sequence_length(4096), 2.05), "shift at max tokens")
    _check(_close(shifted_logit_normal_sigma_legacy(0.0, 0.0, 1.0), 0.5), "legacy logit-normal sigma")
    var stretched = shifted_logit_normal_sigma_stretched(0.0, 0.5, 0.0, 0.0, 1.0, 1.0e-3, 0.1)
    _check(_close(stretched, 0.5005), "stretched uniform fallback branch")
    _check(_close(apply_timestep_range(0.5, 100.0, 900.0), 0.5), "timestep range remap")
    _check(_close(sigma_to_model_timestep(0.25), 250.0), "sigma to model timestep")
    _check(_close(flow_match_noisy_value(2.0, 10.0, 0.25), 4.0), "flow noisy input")
    _check(_close(flow_match_target_value(10.0, 2.0), 8.0), "flow target")
    _check(_close(av_weighted_loss_value(2.0, 3.0, True, 0.5, 0.25), 1.75), "AV weighted loss")
    _check(_close(av_weighted_loss_value(2.0, 3.0, False, 0.5, 0.25), 1.0), "video-only weighted loss")

    var pred = List[Float32]()
    pred.append(1.0)
    pred.append(3.0)
    var target = List[Float32]()
    target.append(0.0)
    target.append(0.0)
    var vmask = List[Bool]()
    vmask.append(True)
    vmask.append(False)
    _check(
        _close(masked_loss_video_bf_mask(pred, target, vmask, 1, 1, 2, 1, 1, LTX2_LOSS_MSE), 1.0),
        "masked video loss denominator",
    )

    var apred = List[Float32]()
    apred.append(1.0)
    apred.append(2.0)
    apred.append(3.0)
    var atarget = List[Float32]()
    atarget.append(0.0)
    atarget.append(0.0)
    atarget.append(0.0)
    var amask = List[Bool]()
    amask.append(True)
    amask.append(False)
    amask.append(True)
    _check(
        _close(masked_loss_audio_bt_mask(apred, atarget, amask, 1, 1, 3, 1, LTX2_LOSS_MSE), 5.0),
        "masked audio loss denominator",
    )
    print("  PASS schedule / loss foundation")


def _gate_timestep_handoff() raises:
    print("--- Musubi timestep handoff parity ---")
    _check(_close(normalize_musubi_training_timestep(250.0), 0.25), "normalize raw timestep to sigma")

    var scalar = prepare_av_model_sigmas_from_scalar_timestep(375.0, False)
    _check(_close(scalar.video_head_sigma, 0.375), "scalar video sigma")
    _check(_close(scalar.audio_sigma, 0.375), "scalar coupled audio sigma")
    _check(scalar.coupled_audio_timestep, "scalar coupled audio flag")

    var token_timesteps = List[Float32]()
    token_timesteps.append(900.0)
    token_timesteps.append(125.0)
    token_timesteps.append(500.0)
    var tokenwise = prepare_av_model_sigmas_from_token_timesteps(token_timesteps, False)
    _check(_close(tokenwise.video_head_sigma, 0.9), "token-wise video head sigma")
    _check(_close(tokenwise.audio_sigma, 0.9), "token-wise audio uses first video timestep")
    _check(tokenwise.coupled_audio_timestep, "token-wise coupled audio flag")

    var independent = prepare_av_model_sigmas_from_token_timesteps(token_timesteps, True, 0.25, 100.0, 500.0)
    _check(_close(independent.video_head_sigma, 0.9), "independent audio preserves video head sigma")
    _check(_close(independent.audio_sigma, 0.2), "independent audio uniform range")
    _check(not independent.coupled_audio_timestep, "independent audio flag")
    _check(_close(independent_audio_sigma_from_uniform(0.75, 200.0, 600.0), 0.5), "independent audio scalar helper")
    print("  PASS timestep handoff parity")


def _gate_lora_surface() raises:
    print("--- LoRA surface parity foundation ---")
    _check(target_count_for_preset(PRESET_T2V) == DEFAULT_T2V_TARGETS_TOTAL, "T2V target count")
    _check(target_count_for_preset(PRESET_T2V) == 1152, "T2V Musubi count")
    _check(target_count_for_preset(PRESET_V2V) == 1344, "V2V IC-LoRA count")
    _check(target_count_for_preset(PRESET_AUDIO) == 672, "audio-only count")
    _check(target_count_for_preset(PRESET_AUDIO_REF_ONLY_IC) == 864, "audio-ref IC count")
    _check(target_count_for_preset(PRESET_FULL) == -1, "full preset requires checkpoint inspection")
    _check(len(modules_per_block_for_preset(PRESET_T2V)) == 24, "T2V modules per block")
    _check(len(v2v_modules_per_block()) == 28, "V2V modules per block")
    _check(len(audio_modules_per_block()) == 14, "audio modules per block")
    _check(len(audio_ref_only_ic_modules_per_block()) == 18, "audio-ref modules per block")
    _check(
        diffusion_lora_a_key(47, "video_to_audio_attn.to_out.0")
        == "diffusion_model.transformer_blocks.47.video_to_audio_attn.to_out.0.lora_A.weight",
        "diffusion LoRA A key",
    )
    _check(
        diffusion_lora_b_key(0, "audio_ff.net.2")
        == "diffusion_model.transformer_blocks.0.audio_ff.net.2.lora_B.weight",
        "diffusion LoRA B key",
    )
    print("  PASS LoRA target surface foundation")


def _gate_checkpoint_validation() raises:
    print("--- checkpoint / resume / validation parity foundation ---")
    _check(lobo_step_lora_filename(25) == "lora_step_000025.safetensors", "Lobo step filename")
    _check(musubi_step_lora_filename("ltx23_lora", 25) == "ltx23_lora-step00000025.safetensors", "Musubi step filename")
    _check(train_state_filename(25) == "train_state_step_000025.train_state.safetensors", "train state filename")
    _check(resume_token_is_latest("latest") and resume_token_is_latest("auto"), "resume latest tokens")
    var cfg = LTX2TrainerConfig.default()
    cfg.output_dir = String("/runs/ltx2")
    cfg.resume_from = String("latest")
    _check(lora_latest_path(cfg.output_dir) == "/runs/ltx2/lora_latest.safetensors", "latest path")
    _check(lora_final_path(cfg.output_dir) == "/runs/ltx2/lora_final.safetensors", "final path")
    _check(lora_emergency_path(cfg.output_dir) == "/runs/ltx2/lora_emergency.safetensors", "emergency path")
    _check(resume_checkpoint_contract(cfg).startswith("resume latest"), "resume latest contract")
    _check(save_checkpoint_contract(cfg, 500).find("train_state_step_000500") >= 0, "checkpoint save state contract")
    cfg.validation_prompts_cache = String("/cache/ltx2/ltx2_sample_prompts_cache.pt")
    cfg.sample_latents_cache = String("/cache/ltx2/ltx2_sample_latents_cache.pt")
    cfg.sample_every = 250
    var vc = default_validation_contract(cfg)
    _check(vc.enabled(), "validation contract enabled")
    _check(vc.baseline_control and vc.merge_audio and vc.two_stage and vc.tiled_vae, "validation HQ defaults")
    _check(sample_video_path("/runs/ltx2", 25) == "/runs/ltx2/samples/000025.mp4", "sample video path")
    _check(sample_audio_path("/runs/ltx2", 25) == "/runs/ltx2/samples/000025.wav", "sample audio path")
    _check(sample_muxed_av_path("/runs/ltx2", 25) == "/runs/ltx2/samples/000025_av.mp4", "sample muxed path")
    _check(validation_sampling_ready(True, True, True), "validation sampling ready predicate")
    _check(not validation_sampling_ready(True, True, False), "validation sampling vocoder gate")
    print("  PASS checkpoint / validation foundation")


def _print_production_gates():
    print("")
    print("=== production parity gates still required ===")
    print("DATA/CACHE")
    print("  PENDING dataset manifest loader and source-free train path")
    print("  PENDING safetensors tensor-shape validation for video/text/audio/DINO caches")
    print("  PENDING reference image/video cache records for IC-LoRA/V2V")
    print("  GATED audio_ref_only_ic composition contract; PENDING reference-audio cache records")
    print("  PENDING GPU audio bucket collation using gated separate_audio_buckets/pad/truncate/quota policy")
    print("  PENDING FPS resampling and audio time-stretch parity with Musubi cache behavior")
    print("SCHEDULE/NOISE/LOSS")
    print("  PENDING device RNG parity for per-sample video/audio independent timesteps")
    print("  GATED masked video/audio scalar parity; PENDING GPU/backward integration and reference-token exclusions")
    print("  GATED masked mse/mae/huber scalar parity; PENDING GPU weighted/backward integration")
    print("  PENDING audio_loss_balance modes: inv_freq, ema_mag, uncertainty")
    print("  PENDING audio_silence_regularizer, audio_dop, supervision monitor, modality freezing, CTS, Self-Flow, CREPA")
    print("LORA/ADAPTERS")
    print("  PENDING full preset checkpoint-inspected linear target enumeration")
    print("  PENDING train-time AV LoRA runtime/storage/save/load on the full LTX2 DiT")
    print("  PENDING per-module audio_dim/audio_alpha, audio_lr, lr_args grouping")
    print("  GATED audio-ref IC concat/masks/positions; PENDING V2V refs, first-frame conditioning, reference_downscale runtime")
    print("  PENDING DoRA/LyCORIS/LoHA/LoKR/LoCon policy and explicit compatibility gates for LTX2")
    print("BACKWARD/OPTIMIZER")
    print("  PENDING full BasicAVTransformerBlock backward with video/audio/cross-modal gradients")
    print("  PENDING GPU-heavy hot loop with CPU-light orchestration and no Python runtime")
    print("  PENDING gradient checkpointing/block swap backward parity")
    print("  PENDING optimizer state coverage for AdamW8Bit-compatible, AdamW, Prodigy/Lion/Adafactor/StableAdamW/ScheduleFree where enabled")
    print("CHECKPOINT/RESUME")
    print("  PENDING original Musubi LoRA format plus Comfy conversion sidecar")
    print("  PENDING optimizer, scheduler, RNG, dataloader, EMA/uncertainty/projector state resume")
    print("  PENDING rotation, latest/final/emergency, autoresume, reset_optimizer/reset_dataloader")
    print("VALIDATION/SAMPLING")
    print("  PENDING train-time HQ AV validation sampler with prompt cache, LoRA applied, two-stage, tiled VAE, audio mux")
    print("  PENDING baseline-control samples at step 0 and LensTrainer-style simple sample cadence")
    print("  PENDING IC-LoRA sample refs: --v, --ra, sample_include_reference, audio-only previews")


def production_ready() -> Bool:
    # Foundation contracts above are necessary but not sufficient. The full AV
    # backward/runtime/validation/save path is not present yet.
    return False


def main() raises:
    var args = argv()
    var expect_production = False
    for i in range(1, len(args)):
        if String(args[i]) == "--expect-production":
            expect_production = True

    print("=== LTX2 trainer parity audit ===")
    _gate_config_cli()
    _gate_cache_records()
    _gate_audio_buckets()
    _gate_conditioning()
    _gate_schedule_loss()
    _gate_timestep_handoff()
    _gate_lora_surface()
    _gate_checkpoint_validation()
    _print_production_gates()

    if expect_production and not production_ready():
        raise Error("LTX2 trainer parity audit: production parity is not achieved")

    print("")
    print("LTX2 trainer parity audit PASS: foundation contracts match; production gates remain open")
