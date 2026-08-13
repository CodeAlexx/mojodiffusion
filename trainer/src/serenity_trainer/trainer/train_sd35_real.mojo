# train_sd35_real.mojo — SD3.5-Large LoRA training loop (block-swap offload).
#
# TRANSLATION of the proven Chroma block-swap pattern onto SD3.5-Large.
# Real SD3.5-Large base weights (streamed block-by-block via TurboPlannedLoader),
# real SerenityTrainer cache (latent_image + split CLIP/T5 hidden/pooled fields),
# full 38 joint-block depth.
# No synthetic tensors. Mirrors train_chroma_real.mojo's loop structure.
#
# SD3.5 vs CHROMA (the deltas):
#   - NO frozen approximator. Modulation comes from per-block adaLN_modulation.1
#     (streamed with each block), conditioned on c = t_embed(sigma*1000) + y_embed(pooled).
#   - JOINT BLOCKS ONLY: 38 joint blocks, no single-stream blocks.
#   - SerenityTrainer cache keys:
#       "latent_image" [1,16,128,128]
#       "text_encoder_1_hidden_state" [1,77,768]
#       "text_encoder_2_hidden_state" [1,77,1280]
#       "text_encoder_3_hidden_state" [1,77,4096]
#       "text_encoder_1_pooled_state" [1,768]
#       "text_encoder_2_pooled_state" [1,1280]
#     The legacy local combined keys "latent", "text_embedding", and "pooled"
#     are accepted only as a compatibility fallback.
#   - NO RoPE (pos_embed added once at patchify, before blocks, in inference;
#     for training the patchify linear already encodes position via weight layout).
#   - LoRA: SD35LoraSet with 8 adapters/block (4 ctx + 4 x: qkv, proj, fc1, fc2).
#
# Per step:
#   1. Load cached SerenityTrainer {latent_image, split hidden/pooled text fields}
#   2. latent_scaled = (latent_image - VAE_SHIFT) * VAE_SCALE
#   3. pack_latents([16,128,128]) -> [N_IMG=4096, 64] channel-major patchify
#   4. sigma_idx = floor(logit_normal_sigma(shift=1.0) * 1000) clamp;
#      sig=(idx+1)/1000 ; sigma_cont=sig (passed to t_embedder as sigma*1000)
#   5. noisy = noise*sig + latent_packed*(1-sig) ; target = noise - latent_packed
#   6. sd35_stack_lora_forward_offload(noisy, txt, pooled, sigma, ...) -> pred [N_IMG,64]
#   7. loss = MSE(pred, target); d_loss = (2/N)(pred - target)
#   8. sd35_stack_lora_backward_offload -> LoRA grads; global-norm clip(1.0)
#   9. sd35_lora_adamw_step; print shared progress display
#
# Recipe (from EriDiffusion-v2 prepare_sd35.rs / SerenityTrainer SD3.5 LoRA preset):
#   lr=1e-4, rank=16, alpha=16, timestep_shift=1.0, clip_grad_norm=1.0
#   VAE shift=0.0609 scale=1.5305
#
# FIXED_SIGMA_SMOKE: every step uses the SAME cache sample AND a fixed
# timestep+noise so a correct LoRA backward MUST drive loss DOWN monotonically.
#
# Run (real smoke):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/training/train_sd35_real.mojo -o /tmp/train_sd35_real && \
#     /tmp/train_sd35_real [steps]

from std.sys import argv
from std.collections import List, Optional
from max.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from std.time import perf_counter_ns
from std.os import listdir, makedirs

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts

from serenitymojo.models.sd35.weights import load_sd35_stack_base
# DEVICE-RESIDENT stack (MJ-1069 rung 3): activations on GPU across the 38
# blocks, recompute-in-backward (input tapes only). False = host oracle.
comptime SD35_DEVICE_STACK = True

from serenitymojo.training.lora_adamw_plain_fused import (
    LoraAdamWPlainDeviceState,
    lora_adamw_plain_device_state_sync_params,
    lora_adamw_plain_device_state_sync_moments,
)
from serenitymojo.training.lora_save import lora_train_state_has_moments
from serenitymojo.models.sd35.sd35_stack_lora import (
    sd35_lora_adamw_step_resident, sd35_lora_adamw_state_init,
    SD35LoraSet, SD35LoraGradSet, SD35StackBase,
    build_sd35_lora_set, sd35_lora_adamw_step,
    save_sd35_lora, save_sd35_lora_state, total_adapters,
    save_sd35_lora_state_with_meta, load_sd35_lora_state, load_sd35_lora_resume,
    sd35_stack_lora_forward_offload, sd35_stack_lora_backward_offload,
    sd35_stack_lora_forward_offload_device, sd35_stack_lora_backward_offload_device,
    sd35_stack_lora_forward_offload_device_b2, sd35_stack_lora_backward_offload_device_b2,
    SD35_DIRECT_24_GIB,
    empty_sd35_direct_dora_set, empty_sd35_direct_oft_set,
    sd35_direct_dense_carrier_bytes,
    sd35_direct_dora_preflight, sd35_direct_oft_preflight,
    build_sd35_direct_dora_set_from_offload,
    build_sd35_direct_oft_set_for_stack,
    sd35_direct_dora_trainable_bytes, sd35_direct_oft_trainable_bytes,
    sd35_stack_direct_dora_forward_offload,
    sd35_stack_direct_dora_backward_offload,
    sd35_stack_direct_oft_forward_offload,
    sd35_stack_direct_oft_backward_offload,
    sd35_direct_dora_grad_norm, sd35_direct_dora_clip_grads,
    sd35_direct_dora_adamw_step, sd35_direct_dora_zero_leg_l1,
    sd35_direct_oft_grad_norm, sd35_direct_oft_clip_grads,
    sd35_direct_oft_adamw_step, sd35_direct_oft_vec_l1,
    save_sd35_direct_dora, save_sd35_direct_oft,
)
from serenitymojo.models.sd35.sd35_lycoris_stack import (
    SD35LoKrSet, empty_sd35_lokr_set, build_sd35_lokr_set,
    sd35_lokr_carrier_set, sd35_lokr_carrier_total_bytes,
    sd35_lokr_chain_all, sd35_lokr_adamw_step, sd35_lokr_grad_norm,
    sd35_lokr_clip_grads, sd35_lokr_zero_leg_l1, save_sd35_lokr,
    SD35LoHaSet, empty_sd35_loha_set, build_sd35_loha_set,
    sd35_loha_carrier_set, sd35_loha_carrier_total_bytes,
    sd35_loha_chain_all, sd35_loha_adamw_step, sd35_loha_grad_norm,
    sd35_loha_clip_grads, sd35_loha_zero_leg_l1, save_sd35_loha,
)
from serenitymojo.offload.plan import build_sd35_large_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.schedule import sample_timestep_logit_normal
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.training.sample_prompt_config import (
    SampleCadence, SamplePrompt, SamplePromptConfig,
    read_sample_cadence_config, read_sample_prompt_config,
    validate_step_sample_cadence, should_sample_completed_step,
    next_sample_completed_step, sample_time_unit_name,
    SAMPLE_UNIT_STEP, SAMPLE_UNIT_NEVER,
)
from serenitymojo.sampling.product_sampler_harness import (
    build_product_sampler_run_contract,
    validate_product_sampler_run_contract,
)
from serenitymojo.training.serenity_trainer_train_loop_policy import (
    SERENITY_GRAD_POLICY_ON_ONLY,
    serenity_cache_dir_from_train_config,
    serenity_lr_for_optimizer_step,
    serenity_output_lora_path_from_train_config,
    serenity_sample_cadence_from_train_config,
    serenity_sampling_enabled,
    serenity_should_save_before_sample,
    serenity_should_save_checkpoint,
    serenity_step_lora_path,
    validate_serenity_gradient_checkpointing_policy,
    validate_serenity_train_math_policy,
)
from serenitymojo.training.train_config import (
    TrainConfig, GRADIENT_CHECKPOINTING_ON,
    TRAIN_ADAPTER_ALGO_LORA, TRAIN_ADAPTER_ALGO_FULL,
    TRAIN_ADAPTER_ALGO_LOCON, TRAIN_ADAPTER_ALGO_LOHA,
    TRAIN_ADAPTER_ALGO_DORA, TRAIN_ADAPTER_ALGO_LOKR,
    TRAIN_ADAPTER_ALGO_OFT, TRAIN_ADAPTER_ALGO_BOFT,
)
from serenitymojo.training.adapter_algo_policy import adapter_algo_name
from serenitymojo.training.trainer_core import (
    GradAccumWindow, trainer_resolve_resume_path, trainer_warn_warm_resume,
    trainer_prune_target_step, trainer_prune_step_checkpoint,
)
from serenitymojo.training.lokr_stack import LOKR_CARRIER_MAX_DEVICE_BYTES
from serenitymojo.training.serenity_trainer_cache_preflight import (
    create_serenity_trainer_cache_preflight_plan,
    validate_serenity_trainer_cache_preflight_plan,
)
from serenitymojo.training.sd35_sample_resident import (
    sd35_sample_resident, sd35_decode_latent_to_png,
)
from serenitymojo.models.dit.sd3_contract import (
    SD3_LARGE_NUM_STEPS, sd3_large_schedule_shift,
)


# ── arch (sd3.5-large; H/Dh/D fixed comptime, verified vs the checkpoint) ────
comptime H = 38
comptime Dh = 64
comptime D = H * Dh            # 2432
comptime FMLP = 9728           # mlp_hidden = D*4 (approximately; real=9728)
comptime IN_CH = 64            # patch_dim = 16ch * 2*2
comptime TXT_CH = 4096         # combined CLIP-L/G + T5
comptime OUT_CH = 64
comptime NUM_JOINT = 38
comptime EPS = Float32(1e-06)
comptime QK_EPS = Float32(1e-06)
comptime TIMESTEP_DIM = 256    # sinusoidal embedding dim for t_embedder
comptime POOLED_DIM = 2048     # clip_l + clip_g pooled

# ── resolution (1024px): latent [16,128,128] -> pack2 -> 64x64=4096 img tokens ─
comptime LAT_C = 16
comptime LAT_H = 128
comptime LAT_W = 128
comptime PATCH = 2
comptime HT = LAT_H // PATCH   # 64
comptime WT = LAT_W // PATCH   # 64
comptime N_IMG = HT * WT       # 4096
comptime N_TXT = 154           # 77 CLIP-LG + 77 T5 (locked per prepare_sd35_cache.py)
comptime S = N_TXT + N_IMG     # 4250

# ── recipe ──────────────────────────────────────────────────────────────────────
comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime LR = Float32(1.0e-4)
comptime TIMESTEP_SHIFT = Float32(1.0)
comptime VAE_SHIFT = Float32(0.0609)
comptime VAE_SCALE = Float32(1.5305)
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime CLIP_GRAD_NORM = Float32(1.0)
comptime SEED_BASE = UInt64(42)
comptime SD35_OFT_BLOCK_SIZE = 4

comptime FIXED_SIGMA_SMOKE = True
comptime FIXED_SIGMA_IDX = 500

comptime CKPT = "/home/alex/.serenity/models/checkpoints/sd3.5_large.safetensors"
comptime CACHE_DIR = "/home/alex/datasets/andrsd35_sd35_cache"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/sd35_lora"
comptime DEFAULT_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/sd35.json"
comptime DEFAULT_RUN_STEPS = 5

# ── sample-during-training (v1; sd35_sample_resident) ────────────────────────
# When the existing SampleCadence fires (should_sample_completed_step), denoise a
# sample from the CURRENT frozen base + streamed joint blocks + LIVE LoRA, decode
# with the SD3.5 embedded VAE, and write <LORA_DIR>/samples/step_<N>.png. Geometry
# is the trainer's 1024px latent (LAT_H=LAT_W=128 -> 8x VAE -> 1024x1024 image).
#   SAMPLE_STEPS / SAMPLE_CFG / SAMPLE_SHIFT : denoise loop length + CFG + FlowMatch
#                               static shift (sampler defaults 28 / 4.5 / 3.0 —
#                               sd3_sample_cli.mojo NUM_STEPS/CFG_SCALE/SHIFT).
#   SAMPLE_SEED               : base RNG seed for the t=1 packed init noise.
# v1 CONDITIONING (flagged): no in-tree SD3 triple-encoder runtime, so the COND
#   text is the CURRENT step's cached caption embeds (txt_tokens + pooled_h);
#   UNCOND is a zero vector. See sd35_sample_resident.mojo header for the why +
#   drop-in path.
comptime SAMPLE_STEPS = SD3_LARGE_NUM_STEPS   # 28
comptime SAMPLE_CFG = Float32(4.5)            # sd3_sample_cli.mojo CFG_SCALE
# SAMPLE_SHIFT comes from sd3_large_schedule_shift() (3.0) at the callsite so the
# schedule shift is single-sourced with the inference scheduler.
comptime SAMPLE_SEED = UInt64(0x5D35_5A91)


def _is_nonnegative_int(s: String) -> Bool:
    if s.byte_length() == 0:
        return False
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        if bs[i] < 0x30 or bs[i] > 0x39:
            return False
    return True


def _parse_nonnegative_int(s: String) raises -> Int:
    if not _is_nonnegative_int(s):
        raise Error(String("expected non-negative integer, got ") + s)
    var out = 0
    var bs = s.as_bytes()
    for i in range(s.byte_length()):
        out = out * 10 + Int(bs[i] - 0x30)
    return out


def _close_f32(a: Float32, b: Float32, tol: Float32 = Float32(1.0e-7)) -> Bool:
    var d = a - b
    if d < Float32(0.0):
        d = -d
    return d <= tol


def sd35_checkpoint_from_train_config(cfg: TrainConfig) -> String:
    if cfg.checkpoint != String(""):
        return cfg.checkpoint.copy()
    if cfg.base_model_name != String(""):
        return cfg.base_model_name.copy()
    return String(CKPT)


def validate_sd35_train_config(cfg: TrainConfig) raises:
    if cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOCON:
        print("[SD35-locon] network_algorithm=locon: using the linear LoRA-compatible down/up path")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR:
        print("[SD35-lokr] network_algorithm=lokr: using carrier dispatch through the LoRA stack")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA:
        print("[SD35-loha] network_algorithm=loha: using carrier dispatch through the LoRA stack")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA:
        print("[SD35-dora] network_algorithm=dora: using direct W_eff stack dispatch with streamed base-weight init")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT:
        print("[SD35-oft] network_algorithm=oft: using direct W_eff stack dispatch with SerenityTrainer-OFT block_size=4")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_BOFT:
        raise Error("SD3.5 trainer: BOFT is intentionally excluded; use lora, locon, loha, lokr, dora, or oft where wired")
    elif cfg.adapter_algo == TRAIN_ADAPTER_ALGO_FULL:
        raise Error("SD3.5 trainer: full finetune is not wired; supported here: lora, locon, loha, lokr, dora, oft")
    elif cfg.adapter_algo != TRAIN_ADAPTER_ALGO_LORA:
        raise Error(
            String("SD3.5 trainer: network_algorithm=")
            + adapter_algo_name(cfg.adapter_algo)
            + String(" is not wired; supported here: lora, locon, loha, lokr, dora, oft")
        )
    if (
        cfg.name != String("STABLE_DIFFUSION_35")
        and cfg.name != String("sd35")
        and cfg.name != String("sd3.5")
        and cfg.name != String("sd3-5")
    ):
        raise Error(
            String("SD3.5 trainer only supports STABLE_DIFFUSION_35/sd35; plain SD3 is not a port target")
        )
    if cfg.checkpoint == String("") and cfg.base_model_name == String(""):
        raise Error("SD3.5 trainer config must set checkpoint or base_model_name")
    var ckpt = sd35_checkpoint_from_train_config(cfg)
    if not ckpt.endswith(String(".safetensors")):
        raise Error(
            String("SD3.5 trainer currently requires a single safetensors checkpoint; ")
            + String("sharded transformer dirs need a dedicated SD3.5 loader")
        )
    if cfg.n_heads != H:
        raise Error(String("SD3.5 config n_heads ") + String(cfg.n_heads) + String(" != H ") + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("SD3.5 config head_dim ") + String(cfg.head_dim) + String(" != Dh ") + String(Dh))
    if cfg.d_model != D:
        raise Error(String("SD3.5 config d_model ") + String(cfg.d_model) + String(" != D ") + String(D))
    if cfg.in_channels != IN_CH:
        raise Error(String("SD3.5 config in_channels ") + String(cfg.in_channels) + String(" != IN_CH ") + String(IN_CH))
    if cfg.joint_attention_dim != TXT_CH:
        raise Error(String("SD3.5 config joint_attention_dim ") + String(cfg.joint_attention_dim) + String(" != TXT_CH ") + String(TXT_CH))
    if cfg.out_channels != OUT_CH:
        raise Error(String("SD3.5 config out_channels ") + String(cfg.out_channels) + String(" != OUT_CH ") + String(OUT_CH))
    if cfg.num_double != NUM_JOINT or cfg.num_single != 0:
        raise Error(
            String("SD3.5 Large trainer requires joint blocks=") + String(NUM_JOINT)
            + String(" and no single-stream blocks; got num_double=")
            + String(cfg.num_double)
            + String(" num_single=")
            + String(cfg.num_single)
        )
    if cfg.mlp_hidden != FMLP:
        raise Error(String("SD3.5 config mlp_hidden ") + String(cfg.mlp_hidden) + String(" != FMLP ") + String(FMLP))
    if cfg.timestep_dim != TIMESTEP_DIM:
        raise Error(String("SD3.5 config timestep_dim ") + String(cfg.timestep_dim) + String(" != TIMESTEP_DIM ") + String(TIMESTEP_DIM))
    if cfg.lora_rank != RANK:
        raise Error(
            String("SD3.5 trainer is compiled for lora_rank=")
            + String(RANK)
            + String("; parsed ")
            + String(cfg.lora_rank)
        )
    if not _close_f32(cfg.lora_alpha, ALPHA):
        raise Error("SD3.5 trainer lora_alpha does not match compiled constant")
    if not _close_f32(cfg.lr, LR, Float32(1.0e-9)):
        raise Error("SD3.5 trainer learning_rate does not match compiled constant")
    if not _close_f32(cfg.timestep_shift, TIMESTEP_SHIFT):
        raise Error("SD3.5 trainer timestep_shift does not match compiled constant")
    if not _close_f32(cfg.max_grad_norm, CLIP_GRAD_NORM):
        raise Error("SD3.5 trainer max_grad_norm does not match compiled constant")
    validate_serenity_train_math_policy(cfg, String("SD3.5 trainer"))
    validate_serenity_gradient_checkpointing_policy(
        cfg, String("SD3.5 trainer"), SERENITY_GRAD_POLICY_ON_ONLY
    )


def sd35_cache_dir_from_train_config(cfg: TrainConfig) -> String:
    return serenity_cache_dir_from_train_config(cfg, String(CACHE_DIR))


def sd35_output_lora_path_from_train_config(cfg: TrainConfig, completed_step: Int) -> String:
    return serenity_output_lora_path_from_train_config(
        cfg, String(LORA_DIR), String("sd35_lora"), completed_step
    )


def _substr(s: String, start: Int, end: Int) -> String:
    var out = String("")
    var i = 0
    for ch in s.codepoint_slices():
        if i >= start and i < end:
            out += String(ch)
        i += 1
    return out^


def _dirname(path: String) -> String:
    var last = -1
    var i = 0
    for ch in path.codepoint_slices():
        if String(ch) == String("/"):
            last = i
        i += 1
    if last <= 0:
        return String(".")
    return _substr(path, 0, last)


def _mkdir_parent(path: String) raises:
    var parent_dir = _dirname(path)
    if parent_dir != String("."):
        makedirs(parent_dir, exist_ok=True)


def sd35_sample_cadence_from_train_config(
    cfg_path: String, cfg: TrainConfig,
) raises -> SampleCadence:
    return serenity_sample_cadence_from_train_config(cfg_path, cfg)


def sd35_sampling_enabled(cadence: SampleCadence) -> Bool:
    return serenity_sampling_enabled(cadence)


def sd35_should_save_checkpoint(cfg: TrainConfig, completed_step: Int) -> Bool:
    return serenity_should_save_checkpoint(cfg, completed_step)


def sd35_should_save_before_sample(
    cadence: SampleCadence, completed_step: Int, saved_this_step: Bool,
) raises -> Bool:
    return serenity_should_save_before_sample(cadence, completed_step, saved_this_step)


def _step_lora_path(base_path: String, step: Int) -> String:
    return serenity_step_lora_path(base_path, step)


# Rolling checkpoint retention (audit item #4), pruned AFTER a periodic save —
# krea2's discipline, thin wrapper over the shared trainer_core machinery. Reuses
# the shared keep-count decision (trainer_prune_target_step), but builds the
# pruned path with sd35's OWN step-path helper (reference-policy naming under LORA_DIR,
# not krea2's workspace/stem), then removes it + its `.state.safetensors` sidecar.
# keep_default/milestone=0 ⇒ NO prune until the webui sets save_max_keep, so
# keep-all stays byte-unchanged when it is unset.
def _sd35_prune_old_checkpoints(cfg: TrainConfig, run_steps: Int, saved_step: Int) raises:
    var old = trainer_prune_target_step(cfg, saved_step, 0, 0)
    if old > 0:
        trainer_prune_step_checkpoint(
            _step_lora_path(sd35_output_lora_path_from_train_config(cfg, run_steps), old),
            String(".state.safetensors"),
        )


def sd35_sample_prompt_config_for_sampler(
    cadence: SampleCadence,
) raises -> SamplePromptConfig:
    if cadence.sample_definition_file_name == String(""):
        raise Error("SD3.5 trainer sampling requires validation_prompts_file or sample_definition_file_name")
    var cfg = read_sample_prompt_config(cadence.sample_definition_file_name)
    if len(cfg.prompts) == 0:
        raise Error("SD3.5 trainer requires at least one validation prompt when sampling is enabled")
    return cfg^


def _sd35_sample_png_path(completed_step: Int, label: String) -> String:
    return (
        String(LORA_DIR) + String("/samples/sd35_sample_step")
        + String(completed_step) + String("_") + label + String(".png")
    )


def _validate_sd35_sampler_prompt(p: SamplePrompt) raises:
    if p.frames != 1:
        raise Error(String("SD3.5 image sampler expects frames=1 for ") + p.label)
    if p.sample_inpainting:
        raise Error(String("SD3.5 trainer sample prompt ") + p.label + String(" requests inpainting; SD3.5 sampler inpaint runtime is not wired"))
    if p.width < 1024 or p.height < 1024:
        raise Error(
            String("SD3.5 sample prompt ") + p.label
            + String(" is ") + String(p.width) + String("x") + String(p.height)
            + String("; image validation samples must be 1024x1024 or larger")
        )


# Preflight the sample prompts BEFORE the train loop: every enabled prompt must
# be a valid 1024+ square image prompt (no video, no inpaint) and produce a valid
# product-sampler run contract. This is the fail-loud geometry/contract gate.
#
# NOTE: the v1 sample-during-training denoise+decode+PNG runtime is now WIRED
# (sd35_sample_resident / sd35_decode_latent_to_png), so this preflight no longer
# raises on product_sampler_harness's deliberately-False scaffold stage flags
# (text_conditioning / transformer_denoise / vae_decode / postprocess_save /
# callbacks / timing / vram). Those flags gate SPEED/IMAGE PARITY ACCEPTANCE, not
# functional wiring: the harness is a measurement contract, not the denoiser.
# Parity acceptance (SerenityTrainer speed/VRAM/trajectory evidence) remains a separate,
# unmet milestone — see sd35_sample_resident.mojo header and the campaign doc.
def sd35_validate_sample_prompts_geometry(
    sample_cfg: SamplePromptConfig, completed_step: Int,
) raises:
    var checked = 0
    for i in range(len(sample_cfg.prompts)):
        var prompt = sample_cfg.prompts[i].copy()
        if not prompt.enabled:
            continue
        _validate_sd35_sampler_prompt(prompt)
        var run = build_product_sampler_run_contract(
            String("STABLE_DIFFUSION_35"),
            prompt,
            _sd35_sample_png_path(completed_step, prompt.label),
        )
        validate_product_sampler_run_contract(run)
        checked += 1
    if checked == 0:
        raise Error("SD3.5 trainer requires at least one enabled validation prompt when sampling is enabled")


# ── deterministic host gaussian noise (Box-Muller PCG; per-step seed) ─────────
def _host_noise(n: Int, seed: UInt64) -> List[Float32]:
    var out = List[Float32]()
    var state = seed
    var i = 0
    while i < n:
        state = state * 6364136223846793005 + 1442695040888963407
        var u1f = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        state = state * 6364136223846793005 + 1442695040888963407
        var u2f = Float64(Int((state >> 12) & 0xFFFFFFFFFFFFF)) * (1.0 / 4503599627370496.0)
        if u1f < 1.0e-12:
            u1f = 1.0e-12
        var r = sqrt(-2.0 * flog(Float64(u1f)))
        var theta = 6.283185307179586 * u2f
        out.append(Float32(r * fcos(Float64(theta))))
        if i + 1 < n:
            out.append(Float32(r * fsin(Float64(theta))))
        i += 2
    return out^


def _absum(v: List[Float32]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i]
        s += x if x >= 0.0 else -x
    return s


def _absum(v: List[BFloat16]) -> Float32:
    var s = Float32(0.0)
    for i in range(len(v)):
        var x = v[i].cast[DType.float32]()
        s += x if x >= 0.0 else -x
    return s


def _global_norm(grads: SD35LoraGradSet) -> Float64:
    var ss = 0.0
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    return sqrt(ss)


def _clip(mut grads: SD35LoraGradSet, max_norm: Float32) -> Float64:
    var gn = _global_norm(grads)
    if gn <= Float64(max_norm) or gn == 0.0:
        return gn
    var s = Float32(Float64(max_norm) / gn)
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            grads.d_a[i][j] = grads.d_a[i][j] * s
        for j in range(len(grads.d_b[i])):
            grads.d_b[i][j] = grads.d_b[i][j] * s
    return gn


def _list_cache(dir: String) raises -> List[String]:
    var raw = listdir(dir)
    var fs = List[String]()
    for i in range(len(raw)):
        if raw[i].endswith(".safetensors"):
            fs.append(dir + String("/") + raw[i])
    if len(fs) == 0:
        raise Error(String("sd35 cache: no .safetensors in ") + dir)
    for i in range(1, len(fs)):
        var j = i
        while j > 0 and fs[j - 1] > fs[j]:
            var tmp = fs[j - 1]
            fs[j - 1] = fs[j]
            fs[j] = tmp
            j -= 1
    return fs^


def _load_cache_preserving_dtype(
    st: SafeTensors, name: String, ctx: DeviceContext
) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def _cache_has_tensor(st: SafeTensors, name: String) -> Bool:
    return name in st.tensors


def _load_cache_preferred(
    st: SafeTensors, preferred: String, legacy: String, ctx: DeviceContext
) raises -> Tensor:
    if _cache_has_tensor(st, preferred):
        return _load_cache_preserving_dtype(st, preferred, ctx)
    if legacy != String("") and _cache_has_tensor(st, legacy):
        return _load_cache_preserving_dtype(st, legacy, ctx)
    raise Error(
        String("SD3.5 cache missing required tensor ")
        + preferred
        + String(" (legacy fallback ")
        + legacy
        + String(" not found)")
    )


def _cache_tensor_to_stack_f32(
    t: Tensor, device_ctx: DeviceContext
) raises -> List[Float32]:
    # The current SD3.5 stack interface is still host List[Float32]. Keep cache
    # tensors device-resident and stage through their stored dtype at this
    # explicit host-list handoff.
    if t.dtype() == STDtype.BF16:
        var bf = t.to_host_bf16(device_ctx)
        var out = List[Float32]()
        for i in range(len(bf)):
            out.append(bf[i].cast[DType.float32]())
        return out^
    if t.dtype() == STDtype.F16:
        var hf = t.to_host_f16(device_ctx)
        var out = List[Float32]()
        for i in range(len(hf)):
            out.append(hf[i].cast[DType.float32]())
        return out^
    return t.to_host(device_ctx)


def _append_padding(mut out: List[Float32], count: Int):
    for _ in range(count):
        out.append(Float32(0.0))


def _stage_sd35_context_for_stack(
    st: SafeTensors, ctx: DeviceContext
) raises -> List[Float32]:
    # Legacy local cache stored SerenityTrainer's combined text handoff directly.
    if _cache_has_tensor(st, String("text_embedding")):
        var te_info = st.tensor_info(String("text_embedding"))
        var te_seq = Int(te_info.shape[1])
        var te_tensor = _load_cache_preserving_dtype(
            st, String("text_embedding"), ctx
        )
        var te_flat = _cache_tensor_to_stack_f32(te_tensor, ctx)
        var tokens = List[Float32]()
        for r in range(N_TXT):
            if r < te_seq:
                for c in range(TXT_CH):
                    tokens.append(te_flat[r * TXT_CH + c])
            else:
                _append_padding(tokens, TXT_CH)
        return tokens^

    var te1_tensor = _load_cache_preserving_dtype(
        st, String("text_encoder_1_hidden_state"), ctx
    )
    var te2_tensor = _load_cache_preserving_dtype(
        st, String("text_encoder_2_hidden_state"), ctx
    )
    var te3_tensor = _load_cache_preserving_dtype(
        st, String("text_encoder_3_hidden_state"), ctx
    )
    var te1 = _cache_tensor_to_stack_f32(te1_tensor, ctx)
    var te2 = _cache_tensor_to_stack_f32(te2_tensor, ctx)
    var te3 = _cache_tensor_to_stack_f32(te3_tensor, ctx)

    var tokens = List[Float32]()
    for r in range(77):
        for c in range(768):
            tokens.append(te1[r * 768 + c])
        for c in range(1280):
            tokens.append(te2[r * 1280 + c])
        _append_padding(tokens, TXT_CH - 2048)
    for r in range(77):
        for c in range(TXT_CH):
            tokens.append(te3[r * TXT_CH + c])
    return tokens^


def _stage_sd35_pooled_for_stack(
    st: SafeTensors, ctx: DeviceContext
) raises -> List[Float32]:
    # Legacy local cache stored cat([clip_l_pool, clip_g_pool]) as "pooled".
    if _cache_has_tensor(st, String("pooled")):
        var pooled_tensor = _load_cache_preserving_dtype(st, String("pooled"), ctx)
        var pooled_raw = _cache_tensor_to_stack_f32(pooled_tensor, ctx)
        var pooled_h = List[Float32]()
        for i in range(POOLED_DIM):
            if i < len(pooled_raw):
                pooled_h.append(pooled_raw[i])
            else:
                pooled_h.append(Float32(0.0))
        return pooled_h^

    var pooled_1_tensor = _load_cache_preserving_dtype(
        st, String("text_encoder_1_pooled_state"), ctx
    )
    var pooled_2_tensor = _load_cache_preserving_dtype(
        st, String("text_encoder_2_pooled_state"), ctx
    )
    var pooled_1 = _cache_tensor_to_stack_f32(pooled_1_tensor, ctx)
    var pooled_2 = _cache_tensor_to_stack_f32(pooled_2_tensor, ctx)
    var pooled_h = List[Float32]()
    for i in range(768):
        pooled_h.append(pooled_1[i])
    for i in range(1280):
        pooled_h.append(pooled_2[i])
    return pooled_h^


# pack_latents: [16, LAT_H, LAT_W] flat (CHW) -> [N_IMG, IN_CH] channel-major patchify.
# Each patch token aggregates a 2x2 spatial region across all 16 channels.
# Token (ih, iw) -> 64 elements: for c in 16, for ph in 2, for pw in 2.
def _pack_latents(lat: List[Float32]) -> List[Float32]:
    var out = List[Float32]()
    for ih in range(HT):
        for iw in range(WT):
            for c in range(LAT_C):
                for ph in range(PATCH):
                    for pw in range(PATCH):
                        var hh = ih * PATCH + ph
                        var ww = iw * PATCH + pw
                        var idx = c * LAT_H * LAT_W + hh * LAT_W + ww
                        out.append(lat[idx])
    return out^


# ── deterministic host gaussian PACKED init noise [N_IMG*IN_CH] ──────────────
# Reuses the train loop's _host_noise Box-Muller PCG so the sample's t=1 packed
# latent is drawn the same way the training noise is. seed makes it deterministic
# per sampled step.
def _sample_init_noise(seed: UInt64) -> List[Float32]:
    return _host_noise(N_IMG * IN_CH, seed)


# ── _sd35_run_sample — one sample-during-training image ──────────────────────
#   cond text   : the current step's cached caption embeds (txt_tokens, v1; header).
#   cond pooled : the current step's cached pooled embeds (pooled_h, v1).
#   uncond      : zeroed [N_CTX*CTX_CH] / [POOLED_DIM] vectors (CFG empty cond).
#   init noise  : packed gaussian [N_IMG*IN_CH], seed = SAMPLE_SEED + step.
#   denoise     : sd35_sample_resident (frozen base + streamed joint blocks + live
#                 LoRA), 28-step shifted-flow CFG Euler.
#   decode+write: sd35_decode_latent_to_png -> <samples_dir>/step_<N>.png (embedded
#                 SD3.5 VAE decoder).
# Fail-loud: any raise propagates (no silent skip), matching the trainer's
# fail-loud cadence contract.
def _sd35_run_sample(
    base: SD35StackBase,
    mut loader: TurboPlannedLoader,
    lora: SD35LoraSet,
    cond_txt: List[Float32],      # [N_CTX*CTX_CH] — the step's cached caption embeds
    cond_pooled: List[Float32],   # [POOLED_DIM]  — the step's cached pooled embeds
    ckpt_path: String,            # SD3.5 checkpoint (embedded VAE decoder)
    samples_dir: String,
    step: Int,
    ctx: DeviceContext,
) raises:
    # UNCOND: zeroed text + pooled features (same shape as the cond conditioning).
    var uncond_txt = List[Float32]()
    for _ in range(N_TXT * TXT_CH):
        uncond_txt.append(Float32(0.0))
    var uncond_pooled = List[Float32]()
    for _ in range(POOLED_DIM):
        uncond_pooled.append(Float32(0.0))

    var init_noise = _sample_init_noise(SAMPLE_SEED + UInt64(step))

    var latent = sd35_sample_resident[H, Dh, N_IMG, N_TXT, S](
        base, loader, lora,
        cond_txt.copy(), cond_pooled.copy(),
        uncond_txt^, uncond_pooled^, init_noise^,
        SAMPLE_STEPS, SAMPLE_CFG, sd3_large_schedule_shift(),
        D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
        EPS, QK_EPS, ctx,
    )

    var out_path = samples_dir + String("/step_") + String(step) + String(".png")
    sd35_decode_latent_to_png[LAT_C, LAT_H, LAT_W, HT, WT, PATCH, N_IMG, IN_CH](
        latent, ckpt_path, out_path, ctx,
    )
    print("[SD35-lora] sample step=", step, " -> ", out_path)


# ── FULL-state resume resolution + loud warm-resume guard (MJ-1088 / MJ-1077) ──
# Probe the `<ckpt>.state.safetensors` sidecar (the naming save_sd35_lora_state
# writes) so a user who passes the PEFT weights still gets a FULL (moment-
# preserving) resume; only warm-restart (loud warning) when there is genuinely no
# moment state. The probe prefix matches _sd35_lora_prefixes[0].
# Thin wrapper → shared trainer_core (binds sd35's first-adapter probe prefix +
# sd35's `.state.safetensors` sidecar naming). Probe order + return logic are the
# trainer_core skeleton; the sidecar-suffix param keeps sd35's sibling name byte-
# exact while krea2 keeps its default `.state`.
def _sd35_resolve_resume_path(path: String) raises -> String:
    var probe = String("transformer.joint_blocks.0.context_block.attn.qkv")
    return trainer_resolve_resume_path(path, probe, String(".state.safetensors"))


# Thin wrapper → shared trainer_core (binds the sd35 log tag + sd35's
# `.state.safetensors` FULL-resume sidecar hint). Every banner line is byte-
# identical to the pre-extraction sd35 output.
def _sd35_warn_warm_resume(path: String):
    trainer_warn_warm_resume(String("sd35"), path, String(".state.safetensors"))


def main() raises:
    var a = argv()
    var cfg_path = String(DEFAULT_CONFIG)
    var arg_base = 1
    if len(a) >= 2:
        var first = String(a[1])
        if first.endswith(String(".json")):
            cfg_path = first.copy()
            arg_base = 2

    var train_cfg = read_model_config(cfg_path)
    validate_sd35_train_config(train_cfg)
    var cache_preflight = create_serenity_trainer_cache_preflight_plan(train_cfg)
    validate_serenity_trainer_cache_preflight_plan(cache_preflight)

    var run_steps = DEFAULT_RUN_STEPS
    if len(a) > arg_base:
        run_steps = _parse_nonnegative_int(String(a[arg_base]))
    elif train_cfg.only_cache:
        run_steps = 0

    # Optional argv[arg_base+1]: resume checkpoint (plain-LoRA arm). Pass either the
    # `<ckpt>.safetensors.state.safetensors` sidecar (FULL moment resume) or the
    # plain PEFT `.safetensors` (WARM start — the .state sibling is auto-probed
    # first so the PEFT path still full-resumes). MJ-1088.
    var resume_path = String("")
    if len(a) > arg_base + 1:
        resume_path = String(a[arg_base + 1])
    if len(a) > arg_base + 2:
        raise Error(
            String("SD3.5 trainer accepts [config.json] [steps] [resume_ckpt] only")
        )

    var ckpt = sd35_checkpoint_from_train_config(train_cfg)
    var cache_dir = sd35_cache_dir_from_train_config(train_cfg)
    var sample_cadence = sd35_sample_cadence_from_train_config(cfg_path, train_cfg)
    var sample_enabled = sd35_sampling_enabled(sample_cadence)
    var dora_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_DORA
    var oft_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_OFT
    var direct_active = dora_active or oft_active
    if direct_active and sample_enabled:
        raise Error("SD3.5 direct DoRA/OFT sample-during-training is not wired; disable sample cadence for this runtime gate")

    print("=== SD3.5-Large REAL LoRA training loop (block-swap offload) ===")
    print("  config:", cfg_path)
    print("  arch: D=", D, " H=", H, " Dh=", Dh, " Fmlp=", FMLP, " out_ch=", OUT_CH)
    print("  depth: NUM_JOINT=", NUM_JOINT)
    print("  tokens: N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S)
    print("  resolution: LAT_H=", LAT_H, " LAT_W=", LAT_W, " patch=", PATCH)
    print("  recipe: rank=", train_cfg.lora_rank, " alpha=", train_cfg.lora_alpha,
          " lr=", train_cfg.lr, " shift=", train_cfg.timestep_shift,
          " vae_shift=", VAE_SHIFT, " vae_scale=", VAE_SCALE)
    print("  run_steps=", run_steps, " config_max_steps=", train_cfg.max_steps)
    print(
        "  cadence: save_every=", train_cfg.save_every,
        " sample_after=", sample_cadence.sample_after,
        " unit=", sample_time_unit_name(sample_cadence.sample_after_unit),
        " skip_first=", sample_cadence.sample_skip_first,
        " sample_file=", sample_cadence.sample_definition_file_name,
    )
    print("  fixed_sigma_smoke=", FIXED_SIGMA_SMOKE)
    print("  ckpt:", ckpt)
    print("  cache:", cache_dir)
    if train_cfg.enable_async_offloading:
        print("[offload] async offload requested by config; SD3.5 trainer currently uses synchronous TurboPlannedLoader")
    if train_cfg.only_cache:
        print("[SD35-lora] only_cache requested; no train steps will run in this trainer")
        return
    var sample_cfg = SamplePromptConfig()
    if sample_enabled:
        sample_cfg = sd35_sample_prompt_config_for_sampler(sample_cadence)
        print(
            "  sample_prompts=", sample_cadence.sample_definition_file_name,
            " count=", len(sample_cfg.prompts),
        )
        if should_sample_completed_step(sample_cadence, 0):
            sd35_validate_sample_prompts_geometry(sample_cfg, 0)

    var ctx = DeviceContext()

    # ── stack-level base (frozen; embedders + final layer) ───────────────────
    print("[load] SD35StackBase (x_embedder, context_embedder, t_embedder, y_embedder, final_layer)")
    var base_st = SafeTensors.open(ckpt)
    var base = load_sd35_stack_base(base_st, ctx)
    print("[load] base resident")

    # ── block-swap offload loader ────────────────────────────────────────────
    var plan = build_sd35_large_block_plan()
    var cfg = OffloadConfig.synchronous_single()
    # fp8_e4m3 pins EVERY block device-resident — skip the whole-DiT pinned
    # host block store (~16 GB, never read again; 2× concurrent = host OOM).
    var loader = TurboPlannedLoader.open(
        ckpt, plan^, cfg, ctx,
        fill_block_store=train_cfg.quantized_resident != String("fp8_e4m3"),
    )
    print("[load] offload loader opened (", loader.block_count(), "blocks)")

    # ── T2.B residency policy (MJ-1065, user directive 2026-07-03) ────────────
    # SD3.5-Large is a ~16GB bf16 DiT; the streamed base re-reads the WHOLE DiT
    # from the host pin pool EVERY forward (a per-step DISK read forbidden by
    # policy). fp8_e4m3 = quantize the 2-D matmul weights ONCE to E4M3 + per-row
    # F32 scale, held resident (~8GB, fits 24GB with LoRA state), dequant per
    # block in the step → ZERO per-step disk. streamed_base_opt_in = the explicit
    # slow bf16 disk-stream arm (quality-control A/Bs only). Any other value
    # (incl the "OFF" default) FAILS LOUD. All adapter paths (lora/dora/oft) +
    # the inline sampler go through loader.await_block, so pinning here converts
    # every one at once.
    var quant_tag = train_cfg.quantized_resident.copy()
    if quant_tag == String("fp8_e4m3"):
        print("[quant] fp8_e4m3-resident base: quantizing",
              loader.block_count(), "blocks ONCE at load ...")
        var pinned = loader.pin_residents_fp8(20 * 1024 * 1024 * 1024, ctx)
        if pinned != loader.block_count():
            raise Error(
                String("SD3.5 fp8-resident: pinned ") + String(pinned) + String(" of ")
                + String(loader.block_count())
                + String(" blocks (budget too small) — MJ-1065 forbids a partial-")
                + String("resident base that would per-step disk-stream the rest")
            )
        print("[quant] fp8_e4m3-resident base: DONE (", pinned,
              "blocks; NO per-step disk read in the step).")
    elif quant_tag == String("streamed_base_opt_in"):
        print("[quant] streamed_base_opt_in: per-step bf16 DISK stream",
              "(EXPLICIT slow experiment arm).")
    else:
        raise Error(
            String("SD3.5: quantized_resident='") + quant_tag
            + String("' selects the per-step DISK-STREAM base, forbidden by policy ")
            + String("MJ-1065. Use \"fp8_e4m3\" (resident base) or ")
            + String("\"streamed_base_opt_in\" to explicitly run the slow streamed arm.")
        )

    # ── LoRA set (B=0 init -> identity at step 0) ────────────────────────────
    var lora = build_sd35_lora_set(NUM_JOINT, D, FMLP, RANK, ALPHA)
    var n_adapters = total_adapters(lora)
    var lokr_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOKR
    var loha_active = train_cfg.adapter_algo == TRAIN_ADAPTER_ALGO_LOHA
    var carrier_active = lokr_active or loha_active
    var lokr_masters = empty_sd35_lokr_set()
    var loha_masters = empty_sd35_loha_set()
    var dora_masters = empty_sd35_direct_dora_set()
    var oft_masters = empty_sd35_direct_oft_set()
    if dora_active:
        var dense_bytes = sd35_direct_dense_carrier_bytes(NUM_JOINT, D, FMLP, train_cfg.lokr_targets)
        var direct_bytes = sd35_direct_dora_preflight(
            NUM_JOINT, D, FMLP, RANK, train_cfg.lokr_targets,
            SD35_DIRECT_24_GIB, False,
        )
        print("[SD35-dora] dense carrier bytes would be:", dense_bytes,
              " direct trainable-state preflight bytes:", direct_bytes,
              " budget:", SD35_DIRECT_24_GIB)
        dora_masters = build_sd35_direct_dora_set_from_offload(
            loader, NUM_JOINT, D, FMLP, Dh, RANK, ALPHA,
            train_cfg.lokr_targets, UInt64(350003), False, ctx,
        )
        print("[SD35-dora] direct slots:", len(dora_masters.ad),
              " trainable bytes:", sd35_direct_dora_trainable_bytes(dora_masters))
    elif oft_active:
        var dense_bytes = sd35_direct_dense_carrier_bytes(NUM_JOINT, D, FMLP, train_cfg.lokr_targets)
        var direct_bytes = sd35_direct_oft_preflight(
            NUM_JOINT, D, FMLP, SD35_OFT_BLOCK_SIZE,
            train_cfg.lokr_targets, SD35_DIRECT_24_GIB,
        )
        print("[SD35-oft] dense carrier bytes would be:", dense_bytes,
              " direct trainable-state preflight bytes:", direct_bytes,
              " budget:", SD35_DIRECT_24_GIB)
        oft_masters = build_sd35_direct_oft_set_for_stack(
            NUM_JOINT, D, FMLP, SD35_OFT_BLOCK_SIZE, train_cfg.lokr_targets,
        )
        print("[SD35-oft] direct slots:", len(oft_masters.ad),
              " trainable bytes:", sd35_direct_oft_trainable_bytes(oft_masters))
    elif lokr_active:
        lokr_masters = build_sd35_lokr_set(
            NUM_JOINT, D, FMLP, RANK, ALPHA,
            train_cfg.lokr_factor, train_cfg.lokr_factor_attn,
            train_cfg.lokr_factor_ff,
            train_cfg.lokr_decompose_both, train_cfg.lokr_full_matrix,
            train_cfg.lokr_targets, UInt64(350001),
        )
        var carrier_bytes = sd35_lokr_carrier_total_bytes(lokr_masters, D, FMLP)
        print("[SD35-lokr] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("SD3.5 LoKr: carrier set needs ") + String(carrier_bytes)
                + String(" bytes (> budget). Use a smaller lokr_factor/rank or restrict lokr_targets.")
            )
        lora = sd35_lokr_carrier_set(lokr_masters, D, FMLP)
        print("[SD35-lokr] carrier set materialized:", len(lora.ad), "adapters")
    elif loha_active:
        loha_masters = build_sd35_loha_set(
            NUM_JOINT, D, FMLP, RANK, ALPHA,
            train_cfg.lokr_targets, UInt64(350002),
        )
        var carrier_bytes = sd35_loha_carrier_total_bytes(loha_masters, D, FMLP)
        print("[SD35-loha] carrier device bytes:", carrier_bytes, " budget:", LOKR_CARRIER_MAX_DEVICE_BYTES)
        if carrier_bytes > LOKR_CARRIER_MAX_DEVICE_BYTES:
            raise Error(
                String("SD3.5 LoHa: carrier set needs ") + String(carrier_bytes)
                + String(" bytes (> budget). Reduce lora_rank or restrict lokr_targets.")
            )
        lora = sd35_loha_carrier_set(loha_masters, D, FMLP)
        print("[SD35-loha] carrier set materialized:", len(lora.ad), "adapters")
    print("[lora] adapters:", n_adapters, " (8 per joint block x", NUM_JOINT, "blocks)")

    var files = _list_cache(cache_dir)
    print("[cache] samples:", len(files))

    var b_absum_init = Float32(0.0)
    for i in range(n_adapters):
        b_absum_init += _absum(lora.ad[i].b)
    if carrier_active:
        print("[lora] carrier LoRA-B |.|_1 at init =", b_absum_init)
    elif direct_active:
        print("[lora] direct DoRA/OFT run: LoRA carrier state is bypassed")
    else:
        print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")
    var carrier_zero_init = Float64(0.0)
    if lokr_active:
        carrier_zero_init = sd35_lokr_zero_leg_l1(lokr_masters)
        print("[SD35-lokr] zero-leg L1 at init =", carrier_zero_init)
    elif loha_active:
        carrier_zero_init = sd35_loha_zero_leg_l1(loha_masters)
        print("[SD35-loha] zero-leg L1 at init =", carrier_zero_init)
    elif dora_active:
        carrier_zero_init = sd35_direct_dora_zero_leg_l1(dora_masters)
        print("[SD35-dora] zero-leg L1 at init =", carrier_zero_init)
    elif oft_active:
        carrier_zero_init = sd35_direct_oft_vec_l1(oft_masters)
        print("[SD35-oft] vec L1 at init =", carrier_zero_init)

    var next_sample = next_sample_completed_step(sample_cadence, 0, train_cfg.max_steps)
    print("[cadence] next sample completed_step=", next_sample)

    # ── sample-during-training output dir (created once when sampling is on) ──
    var samples_dir = String(LORA_DIR) + String("/samples")
    if sample_enabled:
        makedirs(samples_dir, exist_ok=True)
        print("[cadence] sample-during-training WIRED -> ", samples_dir,
              " (", SAMPLE_STEPS, "-step CFG=", SAMPLE_CFG, " v1 cond=cached-caption)")
    var output_lora_path = sd35_output_lora_path_from_train_config(train_cfg, run_steps)
    _mkdir_parent(output_lora_path)

    var adamw_dev_state = Optional[LoraAdamWPlainDeviceState](None)
    var adamw_state_ready = False
    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)

    # ── gradient accumulation (SerenityTrainer micro-batch; default-off == 1) ─────────
    # Each loop iteration is one MICRO-step. SUM the two AdamW-fed LoRA grad
    # groups (d_a/d_b) across `accum_steps` micro-steps, then MEAN (÷N) and run
    # clip+AdamW once on accumulation boundaries. accum_steps=1 => every step is a
    # boundary, mean=÷1 => byte-identical to the per-step path. (The 3 pre_only
    # dead context slots on block 37 carry zero grads; accumulating zeros is a
    # no-op — no special-casing.)
    # Wired for PLAIN LoRA only this wave; LyCORIS arms fail loud (mirrors Klein).
    var accum_steps = train_cfg.grad_accum_steps
    if accum_steps < 1:
        accum_steps = 1
    var use_grad_accum = accum_steps > 1
    if use_grad_accum and (lokr_active or loha_active or dora_active or oft_active):
        raise Error(
            "SD3.5 trainer: grad_accum_steps>1 is wired for plain LoRA only this "
            + "wave; LoKr/LoHa/DoRA/OFT fail loud (mirrors Klein's honest scope). "
            + "Use adapter_algo=0 (plain LoRA) with gradient accumulation."
        )
    # window buffers + micro counter live in the shared trainer_core struct (wraps
    # the grad_accum.mojo SUM/MEAN primitives). accum_steps==1 => every step is a
    # boundary => byte-identical to the per-step path.
    var accum_window = GradAccumWindow(accum_steps)
    if use_grad_accum:
        print("  grad accumulation: accum_steps=", accum_steps, " (mean over micro-steps)")

    # ── TRUE batch-2 (row-stacked device stack): 2 samples/step -> mean gradient.
    #    Fenced to PLAIN LoRA + device stack + accum=1 (mirrors the fleet pattern,
    #    chroma/qwen). Each per-sample d_out is 0.5-scaled so the b2 backward's
    #    in-GEMM LoRA sum = mean(g0, g1) = the 2-sample batch grad; loss=0.5*(L0+L1).
    var use_b2 = train_cfg.batch_size == 2
    if train_cfg.batch_size < 1 or train_cfg.batch_size > 2:
        raise Error(
            "SD3.5 trainer: only batch_size 1 or 2 supported (TRUE batch-2 max); got "
            + String(train_cfg.batch_size)
        )
    if use_b2:
        if lokr_active or loha_active or dora_active or oft_active:
            raise Error(
                "SD3.5 trainer: batch_size=2 (TRUE batch-2) is wired for PLAIN LoRA "
                "only — not the LyCORIS/DoRA/OFT arms. Use adapter_algo=0."
            )
        if use_grad_accum:
            raise Error(
                "SD3.5 trainer: batch_size=2 + grad_accum_steps>1 not wired — "
                "batch_size=2 IS a 2-sample batch; set grad_accum_steps=1."
            )
        print("  TRUE batch-2 (row-stacked device stack): 2 samples/step,",
              "0.5-scaled per-sample d_out -> mean gradient")

    # ── optional resume (plain-LoRA arm only): reload A/B (+ AdamW moments if a
    # FULL `.state` exists) BEFORE the loop so the lazy resident-AdamW init
    # (sd35_lora_adamw_state_init) seeds dev_m/dev_v from the restored moments —
    # full-moment fidelity (MJ-1088).
    if resume_path != String("") and not carrier_active and not direct_active:
        var resolved = _sd35_resolve_resume_path(resume_path)
        var is_full = lora_train_state_has_moments(
            resolved, String("transformer.joint_blocks.0.context_block.attn.qkv")
        )
        if is_full:
            lora = load_sd35_lora_state(NUM_JOINT, D, FMLP, RANK, ALPHA, resolved, ctx)
            print("[sd35-resume] FULL resume (A/B + AdamW moments) from", resolved)
        else:
            _sd35_warn_warm_resume(resolved)
            lora = load_sd35_lora_resume(NUM_JOINT, D, FMLP, RANK, ALPHA, resolved, ctx)
        print("[sd35-resume] reloaded", total_adapters(lora), "adapters")

    var train_start = perf_counter_ns()
    for k in range(1, run_steps + 1):
        var t0 = perf_counter_ns()

        var slot = 0 if FIXED_SIGMA_SMOKE else (k - 1) % len(files)
        var step_seed = UInt64(1) if FIXED_SIGMA_SMOKE else UInt64(k)
        var st = SafeTensors.open(files[slot])

        # latent_image: [1, 16, 128, 128] -> flat [1*16*128*128] = [262144].
        # SerenityTrainer caches raw VAE posterior mean here and applies
        # (latent_image - shift) * scale inside BaseStableDiffusion3Setup.predict.
        var latent_tensor = _load_cache_preferred(
            st, String("latent_image"), String("latent"), ctx
        )
        var latent_raw = _cache_tensor_to_stack_f32(latent_tensor, ctx)

        # SerenityTrainer caches split CLIP-L, CLIP-G, and T5 fields. The legacy
        # local combined text cache is accepted only as a compatibility fallback.
        var txt_tokens = _stage_sd35_context_for_stack(st, ctx)
        var pooled_h = _stage_sd35_pooled_for_stack(st, ctx)

        # ── VAE shift/scale then pack_latents ──
        # latent_raw is flat [1, 16, 128, 128] in CHW; drop batch dim (offset 0).
        # Scale: latent_scaled = (latent_image - VAE_SHIFT) * VAE_SCALE
        var latent_scaled_chw = List[Float32]()
        for i in range(LAT_C * LAT_H * LAT_W):
            latent_scaled_chw.append((latent_raw[i] - VAE_SHIFT) * VAE_SCALE)
        var latent_packed = _pack_latents(latent_scaled_chw)   # [N_IMG=4096, 64]

        # ── timestep ──
        var sigma_idx: Int
        if FIXED_SIGMA_SMOKE:
            sigma_idx = FIXED_SIGMA_IDX
        else:
            var sigma = sample_timestep_logit_normal(SEED_BASE + step_seed, TIMESTEP_SHIFT)
            sigma_idx = Int(sigma * Float32(NUM_TRAIN_TIMESTEPS))
            if sigma_idx > NUM_TRAIN_TIMESTEPS - 1:
                sigma_idx = NUM_TRAIN_TIMESTEPS - 1
        var sig = Float32(sigma_idx + 1) / Float32(NUM_TRAIN_TIMESTEPS)
        # sigma for t_embedder: the conditioning input is sigma * 1000 (done inside _build_conditioning)
        var sigma_cont = sig   # [0,1] range; _build_conditioning multiplies by 1000

        # ── flow-match in PACKED latent space ──
        var noise = _host_noise(N_IMG * IN_CH, SEED_BASE * UInt64(7919) + step_seed)
        var noisy = List[Float32]()
        var target = List[Float32]()
        for i in range(len(latent_packed)):
            noisy.append(noise[i] * sig + latent_packed[i] * (Float32(1.0) - sig))
            target.append(noise[i] - latent_packed[i])

        if dora_active:
            var fwd_dora = sd35_stack_direct_dora_forward_offload[H, Dh, N_IMG, N_TXT, S](
                noisy.copy(), txt_tokens.copy(), pooled_h.copy(), sigma_cont,
                base, loader, dora_masters, NUM_JOINT, train_cfg.lokr_targets,
                D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
                EPS, QK_EPS, ctx,
            )
            var nout_d = len(fwd_dora.out)
            var d_loss_d = List[Float32]()
            var sse_d = 0.0
            var inv_n_d = Float32(2.0) / Float32(nout_d)
            for i in range(nout_d):
                var diff = fwd_dora.out[i] - target[i]
                sse_d += Float64(diff) * Float64(diff)
                d_loss_d.append(inv_n_d * diff)
            var loss_d = Float32(sse_d / Float64(nout_d))
            if k == 1:
                first_loss = loss_d
            last_loss = loss_d

            var dg = sd35_stack_direct_dora_backward_offload[H, Dh, N_IMG, N_TXT, S](
                d_loss_d, noisy.copy(), txt_tokens.copy(),
                base, loader, dora_masters, fwd_dora,
                NUM_JOINT, train_cfg.lokr_targets,
                D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
                EPS, QK_EPS, ctx,
            )
            var mnorm = sd35_direct_dora_grad_norm(dg.grads)
            if mnorm > Float64(CLIP_GRAD_NORM):
                sd35_direct_dora_clip_grads(dg.grads, CLIP_GRAD_NORM / Float32(mnorm))
            var step_lr_d = serenity_lr_for_optimizer_step(train_cfg, k)
            sd35_direct_dora_adamw_step(
                dora_masters, dg.grads, k, step_lr_d,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay,
            )

            var t1_d = perf_counter_ns()
            var secs_d = Float64(t1_d - t0) / 1.0e9
            print_trainer_progress(
                String("SD35-dora"), k, run_steps, 1,
                loss_d, mnorm, secs_d, 0.0,
                Float64(t1_d - train_start) / 1.0e9,
            )
            print("[SD35-dora] step=", k, " master_grad_norm=", Float32(mnorm),
                  " zero_leg_l1=", sd35_direct_dora_zero_leg_l1(dora_masters))
            if dg.nonfinite_grads != 0:
                print("[SD35-dora] warning nonfinite=", dg.nonfinite_grads)
            if sd35_should_save_checkpoint(train_cfg, k):
                var save_path_d = _step_lora_path(
                    sd35_output_lora_path_from_train_config(train_cfg, run_steps), k
                )
                _ = save_sd35_direct_dora(dora_masters, save_path_d, ctx)
                print("[SD35-dora] save step=", k, " path=", save_path_d)
                _sd35_prune_old_checkpoints(train_cfg, run_steps, k)
            continue

        if oft_active:
            var fwd_oft = sd35_stack_direct_oft_forward_offload[H, Dh, N_IMG, N_TXT, S](
                noisy.copy(), txt_tokens.copy(), pooled_h.copy(), sigma_cont,
                base, loader, oft_masters, NUM_JOINT, train_cfg.lokr_targets,
                D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
                EPS, QK_EPS, ctx,
            )
            var nout_o = len(fwd_oft.out)
            var d_loss_o = List[Float32]()
            var sse_o = 0.0
            var inv_n_o = Float32(2.0) / Float32(nout_o)
            for i in range(nout_o):
                var diff = fwd_oft.out[i] - target[i]
                sse_o += Float64(diff) * Float64(diff)
                d_loss_o.append(inv_n_o * diff)
            var loss_o = Float32(sse_o / Float64(nout_o))
            if k == 1:
                first_loss = loss_o
            last_loss = loss_o

            var og = sd35_stack_direct_oft_backward_offload[H, Dh, N_IMG, N_TXT, S](
                d_loss_o, noisy.copy(), txt_tokens.copy(),
                base, loader, oft_masters, fwd_oft,
                NUM_JOINT, train_cfg.lokr_targets,
                D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
                EPS, QK_EPS, ctx,
            )
            var onorm = sd35_direct_oft_grad_norm(og.grads)
            if onorm > Float64(CLIP_GRAD_NORM):
                sd35_direct_oft_clip_grads(og.grads, CLIP_GRAD_NORM / Float32(onorm))
            var step_lr_o = serenity_lr_for_optimizer_step(train_cfg, k)
            sd35_direct_oft_adamw_step(
                oft_masters, og.grads, k, step_lr_o,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay,
            )

            var t1_o = perf_counter_ns()
            var secs_o = Float64(t1_o - t0) / 1.0e9
            print_trainer_progress(
                String("SD35-oft"), k, run_steps, 1,
                loss_o, onorm, secs_o, 0.0,
                Float64(t1_o - train_start) / 1.0e9,
            )
            print("[SD35-oft] step=", k, " master_grad_norm=", Float32(onorm),
                  " vec_l1=", sd35_direct_oft_vec_l1(oft_masters))
            if og.nonfinite_grads != 0:
                print("[SD35-oft] warning nonfinite=", og.nonfinite_grads)
            if sd35_should_save_checkpoint(train_cfg, k):
                var save_path_o = _step_lora_path(
                    sd35_output_lora_path_from_train_config(train_cfg, run_steps), k
                )
                _ = save_sd35_direct_oft(oft_masters, save_path_o, ctx)
                print("[SD35-oft] save step=", k, " path=", save_path_o)
                _sd35_prune_old_checkpoints(train_cfg, run_steps, k)
            continue

        # ── forward+loss+backward (DEVICE arm default, MJ-1069 rung 3; host
        # arm retained as the small-scale oracle — SD35_DEVICE_STACK=False) ──
        # sd3.5-large block 37 context stream is pre_only IN THE FILE (qkv only,
        # no proj/mlp — verified from the safetensors header 2026-07-04).
        var loss: Float32
        var grads: SD35LoraGradSet
        comptime if SD35_DEVICE_STACK:
            if use_b2:
                # ── TRUE batch-2: sample-0 is prepped above; prep sample-1 (its
                #    own cache slot + noise/sigma stream), then run the row-stacked
                #    b2 device stack. Each per-sample d_out is 0.5-scaled so the b2
                #    backward's in-GEMM LoRA sum = mean(g0, g1) = the 2-sample
                #    batch grad; loss = 0.5*(L0 + L1). ──
                var slot1 = 0 if FIXED_SIGMA_SMOKE else (k % len(files))
                var step_seed1 = UInt64(2) if FIXED_SIGMA_SMOKE else (UInt64(k) + UInt64(7000003))
                var st1 = SafeTensors.open(files[slot1])
                var latent_tensor1 = _load_cache_preferred(
                    st1, String("latent_image"), String("latent"), ctx
                )
                var latent_raw1 = _cache_tensor_to_stack_f32(latent_tensor1, ctx)
                var txt_tokens1 = _stage_sd35_context_for_stack(st1, ctx)
                var pooled_h1 = _stage_sd35_pooled_for_stack(st1, ctx)
                var latent_scaled_chw1 = List[Float32]()
                for i in range(LAT_C * LAT_H * LAT_W):
                    latent_scaled_chw1.append((latent_raw1[i] - VAE_SHIFT) * VAE_SCALE)
                var latent_packed1 = _pack_latents(latent_scaled_chw1)
                var sigma_idx1: Int
                if FIXED_SIGMA_SMOKE:
                    sigma_idx1 = FIXED_SIGMA_IDX
                else:
                    var sigma_s1 = sample_timestep_logit_normal(SEED_BASE + step_seed1, TIMESTEP_SHIFT)
                    sigma_idx1 = Int(sigma_s1 * Float32(NUM_TRAIN_TIMESTEPS))
                    if sigma_idx1 > NUM_TRAIN_TIMESTEPS - 1:
                        sigma_idx1 = NUM_TRAIN_TIMESTEPS - 1
                var sig1 = Float32(sigma_idx1 + 1) / Float32(NUM_TRAIN_TIMESTEPS)
                var sigma_cont1 = sig1
                var noise1 = _host_noise(N_IMG * IN_CH, SEED_BASE * UInt64(7919) + step_seed1)
                var noisy1 = List[Float32]()
                var target1 = List[Float32]()
                for i in range(len(latent_packed1)):
                    noisy1.append(noise1[i] * sig1 + latent_packed1[i] * (Float32(1.0) - sig1))
                    target1.append(noise1[i] - latent_packed1[i])

                var fwd = sd35_stack_lora_forward_offload_device_b2[H, Dh, N_IMG, N_TXT, S](
                    noisy.copy(), txt_tokens.copy(), pooled_h.copy(), sigma_cont,
                    noisy1.copy(), txt_tokens1.copy(), pooled_h1.copy(), sigma_cont1,
                    base, loader, lora,
                    D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
                    EPS, QK_EPS, ctx,
                    last_ctx_preonly=True,
                )
                var nout = len(fwd.out0)
                var inv_n = Float32(2.0) / Float32(nout)
                var sse0 = 0.0
                var sse1 = 0.0
                var d_out0 = List[Float32]()
                var d_out1 = List[Float32]()
                for i in range(nout):
                    var diff0 = fwd.out0[i] - target[i]
                    var diff1 = fwd.out1[i] - target1[i]
                    sse0 += Float64(diff0) * Float64(diff0)
                    sse1 += Float64(diff1) * Float64(diff1)
                    d_out0.append(Float32(0.5) * inv_n * diff0)
                    d_out1.append(Float32(0.5) * inv_n * diff1)
                loss = Float32(0.5) * (Float32(sse0 / Float64(nout)) + Float32(sse1 / Float64(nout)))
                grads = sd35_stack_lora_backward_offload_device_b2[H, Dh, N_IMG, N_TXT, S](
                    d_out0, d_out1,
                    noisy.copy(), txt_tokens.copy(), noisy1.copy(), txt_tokens1.copy(),
                    base, loader, lora, fwd,
                    D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
                    EPS, QK_EPS, ctx,
                    last_ctx_preonly=True,
                )
            else:
                var fwd = sd35_stack_lora_forward_offload_device[H, Dh, N_IMG, N_TXT, S](
                    noisy.copy(), txt_tokens.copy(), pooled_h.copy(), sigma_cont,
                    base, loader, lora,
                    D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
                    EPS, QK_EPS, ctx,
                    last_ctx_preonly=True,
                )
                var nout = len(fwd.out)
                var d_loss = List[Float32]()
                var sse = 0.0
                var inv_n = Float32(2.0) / Float32(nout)
                for i in range(nout):
                    var diff = fwd.out[i] - target[i]
                    sse += Float64(diff) * Float64(diff)
                    d_loss.append(inv_n * diff)
                loss = Float32(sse / Float64(nout))
                grads = sd35_stack_lora_backward_offload_device[H, Dh, N_IMG, N_TXT, S](
                    d_loss, noisy.copy(), txt_tokens.copy(),
                    base, loader, lora, fwd,
                    D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
                    EPS, QK_EPS, ctx,
                    last_ctx_preonly=True,
                )
        else:
            if use_b2:
                raise Error(
                    "SD3.5 trainer: batch_size=2 (TRUE batch-2) requires the DEVICE"
                    " stack — compile with SD35_DEVICE_STACK=True."
                )
            var fwd = sd35_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
                noisy.copy(), txt_tokens.copy(), pooled_h.copy(), sigma_cont,
                base, loader, lora,
                D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
                EPS, QK_EPS, ctx,
                last_ctx_preonly=True,
            )
            var nout = len(fwd.out)
            var d_loss = List[Float32]()
            var sse = 0.0
            var inv_n = Float32(2.0) / Float32(nout)
            for i in range(nout):
                var diff = fwd.out[i] - target[i]
                sse += Float64(diff) * Float64(diff)
                d_loss.append(inv_n * diff)
            loss = Float32(sse / Float64(nout))
            grads = sd35_stack_lora_backward_offload[H, Dh, N_IMG, N_TXT, S](
                d_loss, noisy.copy(), txt_tokens.copy(),
                base, loader, lora, fwd,
                D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
                EPS, QK_EPS, ctx,
                last_ctx_preonly=True,
            )
        if k == 1:
            first_loss = loss
        last_loss = loss

        # ── gradient accumulation (SerenityTrainer semantics; default-off when N==1) ─
        # Fast path: accum_steps==1 uses `grads` directly (no zero-clone/copy).
        if use_grad_accum:
            # Treat this loop iteration as one micro-step: SUM its LoRA grad groups
            # into the shared trainer_core window; on a non-boundary micro-step print
            # progress and skip clip/AdamW/save/sample; on the boundary MEAN (÷N)
            # back into `grads` before clip+AdamW. The window math lives in
            # trainer_core.GradAccumWindow (byte-op-identical to the prior inline
            # block); the boundary control flow (progress print + continue) stays
            # here. k==run_steps force-flushes the tail partial window.
            var is_boundary = accum_window.accumulate(grads.d_a, grads.d_b, k == run_steps)
            if not is_boundary:
                # mid-window: skip clip/AdamW/save/sample, keep accumulating.
                var t1m = perf_counter_ns()
                var secsm = Float64(t1m - t0) / 1.0e9
                print_trainer_progress(
                    String("SD35-lora"), k, run_steps, 1,
                    loss, 0.0, secsm, 0.0,
                    Float64(t1m - train_start) / 1.0e9,
                )
                continue
            # boundary: MEAN the window back into grads' groups. The norm is
            # recomputed by _clip below from the meaned grads, so finalize_mean's
            # returned norm is discarded (behavior-identical to the inline block).
            _ = accum_window.finalize_mean(grads.d_a, grads.d_b)

        # ── grad norm + clip(1.0) ──
        var gn_before = _clip(grads, CLIP_GRAD_NORM)

        # ── AdamW ──
        # grad-accum: LR schedule + AdamW step counter advance per OPTIMIZER step,
        # not per micro-step (N==1 => optimizer_step==k, byte-identical).
        var optimizer_step = ((k - 1) // accum_steps) + 1
        var step_lr = serenity_lr_for_optimizer_step(train_cfg, optimizer_step)
        if lokr_active:
            var mg = sd35_lokr_chain_all(lokr_masters, grads.d_a, grads.d_b)
            var mnorm = sd35_lokr_grad_norm(mg)
            if mnorm > Float64(CLIP_GRAD_NORM):
                sd35_lokr_clip_grads(mg, CLIP_GRAD_NORM / Float32(mnorm))
            sd35_lokr_adamw_step(
                lokr_masters, mg, optimizer_step, step_lr,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay,
            )
            lora = sd35_lokr_carrier_set(lokr_masters, D, FMLP)
            print("[SD35-lokr] step=", k, " master_grad_norm=", Float32(mnorm),
                  " zero_leg_l1=", sd35_lokr_zero_leg_l1(lokr_masters))
        elif loha_active:
            var mg = sd35_loha_chain_all(loha_masters, grads.d_a, grads.d_b)
            var mnorm = sd35_loha_grad_norm(mg)
            if mnorm > Float64(CLIP_GRAD_NORM):
                sd35_loha_clip_grads(mg, CLIP_GRAD_NORM / Float32(mnorm))
            sd35_loha_adamw_step(
                loha_masters, mg, optimizer_step, step_lr,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay,
            )
            lora = sd35_loha_carrier_set(loha_masters, D, FMLP)
            print("[SD35-loha] step=", k, " master_grad_norm=", Float32(mnorm),
                  " zero_leg_l1=", sd35_loha_zero_leg_l1(loha_masters))
        else:
            # MJ-1085: resident fused AdamW — persistent device P/M/V with
            # ONE-TIME pinned staging (init lazily after resume load); the
            # per-step fused arm segfaulted here (MJ-1070 unmapped-staging
            # mechanism) and stays retired.
            if not adamw_state_ready:
                adamw_dev_state = Optional[LoraAdamWPlainDeviceState](
                    sd35_lora_adamw_state_init(lora, ctx)
                )
                adamw_state_ready = True
                print("[sd35-adamw] resident fused state initialized (",
                      len(lora.ad), "adapters )")
            sd35_lora_adamw_step_resident(
                adamw_dev_state.value(), lora, grads, optimizer_step, step_lr, ctx,
                train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
                train_cfg.weight_decay,
            )

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        print_trainer_progress(
            String("SD35-lora"), k, run_steps, 1,
            loss, Float64(gn_before), secs, 0.0,
            Float64(t1 - train_start) / 1.0e9,
        )
        if grads.nonfinite != 0:
            print("[SD35-lora] warning nonfinite=", grads.nonfinite)

        var saved_this_step = False
        if sd35_should_save_checkpoint(train_cfg, k):
            var save_path = _step_lora_path(
                sd35_output_lora_path_from_train_config(train_cfg, run_steps), k
            )
            if lokr_active:
                _ = save_sd35_lokr(lokr_masters, save_path, ctx)
            elif loha_active:
                _ = save_sd35_loha(loha_masters, save_path, ctx)
            else:
                _ = save_sd35_lora(lora, save_path, ctx)
                # MJ-1088: pull the resident device m/v back to host before writing
                # the `.state` (the resident step syncs params each step, not
                # moments) so the saved AdamW moments are LIVE, not stale-init.
                if adamw_state_ready:
                    lora_adamw_plain_device_state_sync_params(
                        adamw_dev_state.value(), lora.ad, ctx
                    )
                    lora_adamw_plain_device_state_sync_moments(
                        adamw_dev_state.value(), lora.ad, ctx
                    )
                var state_path = save_path + String(".state.safetensors")
                var state_meta = List[Float32]()
                state_meta.append(Float32(k))
                state_meta.append(Float32(Int(train_cfg.seed)))
                _ = save_sd35_lora_state_with_meta(lora, state_path, ctx, state_meta^)
            saved_this_step = True
            print("[SD35-lora] save step=", k, " path=", save_path)
            _sd35_prune_old_checkpoints(train_cfg, run_steps, k)
        if sample_enabled and should_sample_completed_step(sample_cadence, k):
            if sd35_should_save_before_sample(sample_cadence, k, saved_this_step):
                var sample_path = _step_lora_path(
                    sd35_output_lora_path_from_train_config(train_cfg, run_steps), k
                )
                if lokr_active:
                    _ = save_sd35_lokr(lokr_masters, sample_path, ctx)
                elif loha_active:
                    _ = save_sd35_loha(loha_masters, sample_path, ctx)
                else:
                    _ = save_sd35_lora(lora, sample_path, ctx)
                    # MJ-1088: sync resident m/v to host so the `.state` carries LIVE
                    # AdamW moments (resident step syncs params, not moments).
                    if adamw_state_ready:
                        lora_adamw_plain_device_state_sync_params(
                            adamw_dev_state.value(), lora.ad, ctx
                        )
                        lora_adamw_plain_device_state_sync_moments(
                            adamw_dev_state.value(), lora.ad, ctx
                        )
                    var sample_state = sample_path + String(".state.safetensors")
                    var sample_meta = List[Float32]()
                    sample_meta.append(Float32(k))
                    sample_meta.append(Float32(Int(train_cfg.seed)))
                    _ = save_sd35_lora_state_with_meta(lora, sample_state, ctx, sample_meta^)
                print("[SD35-lora] save_before_sample step=", k, " path=", sample_path)
            # Geometry/contract preflight (fail-loud on bad prompts), then the real
            # v1 sample-during-training run: denoise from the CURRENT frozen base +
            # streamed joint blocks + LIVE LoRA, decode, write the PNG.
            sd35_validate_sample_prompts_geometry(sample_cfg, k)
            # v1 conditioning: this step's cached caption embeds (txt_tokens +
            # pooled_h) as COND, zeros as UNCOND. See sd35_sample_resident header.
            print(
                "[cadence] sample due at completed_step=", k,
                " sample_file=", sample_cadence.sample_definition_file_name,
            )
            _sd35_run_sample(
                base, loader, lora, txt_tokens.copy(), pooled_h.copy(),
                ckpt, samples_dir, k, ctx,
            )

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)
    var b_absum_final = Float32(0.0)
    for i in range(n_adapters):
        b_absum_final += _absum(lora.ad[i].b)
    var trains: Bool
    var carrier_zero_final = Float64(0.0)
    if lokr_active:
        carrier_zero_final = sd35_lokr_zero_leg_l1(lokr_masters)
        trains = carrier_zero_final > carrier_zero_init
    elif loha_active:
        carrier_zero_final = sd35_loha_zero_leg_l1(loha_masters)
        trains = carrier_zero_final > carrier_zero_init
    elif dora_active:
        carrier_zero_final = sd35_direct_dora_zero_leg_l1(dora_masters)
        trains = carrier_zero_final > carrier_zero_init
    elif oft_active:
        carrier_zero_final = sd35_direct_oft_vec_l1(oft_masters)
        trains = carrier_zero_final > carrier_zero_init
    else:
        trains = (b_absum_init == 0.0) and (b_absum_final > 0.0)
    if trains and (last_loss == last_loss):
        if carrier_active or direct_active:
            print("RESULT: REAL run OK — LyCORIS zero-leg grew ",
                  carrier_zero_init, " -> ", carrier_zero_final,
                  "; loss", first_loss, "->", last_loss,
                  (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        else:
            print("RESULT: REAL run OK — LoRA-B grew 0 ->", b_absum_final,
                  "; loss", first_loss, "->", last_loss,
                  (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        var lora_out = sd35_output_lora_path_from_train_config(train_cfg, run_steps)
        if lokr_active:
            _ = save_sd35_lokr(lokr_masters, lora_out, ctx)
        elif loha_active:
            _ = save_sd35_loha(loha_masters, lora_out, ctx)
        elif dora_active:
            _ = save_sd35_direct_dora(dora_masters, lora_out, ctx)
        elif oft_active:
            _ = save_sd35_direct_oft(oft_masters, lora_out, ctx)
        else:
            _ = save_sd35_lora(lora, lora_out, ctx)
            # MJ-1088: sync resident m/v to host so the final `.state` carries LIVE
            # AdamW moments (resident step syncs params, not moments).
            if adamw_state_ready:
                lora_adamw_plain_device_state_sync_params(
                    adamw_dev_state.value(), lora.ad, ctx
                )
                lora_adamw_plain_device_state_sync_moments(
                    adamw_dev_state.value(), lora.ad, ctx
                )
            var state_out = lora_out + String(".state.safetensors")
            var final_meta = List[Float32]()
            final_meta.append(Float32(run_steps))
            final_meta.append(Float32(Int(train_cfg.seed)))
            _ = save_sd35_lora_state_with_meta(lora, state_out, ctx, final_meta^)
            print("[SD35-lora] save_state step=", run_steps, " path=", state_out)
    else:
        print("RESULT: FAIL trains=", trains)
