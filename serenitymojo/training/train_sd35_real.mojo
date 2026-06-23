# train_sd35_real.mojo — SD3.5-Large LoRA training loop (block-swap offload).
#
# TRANSLATION of the proven Chroma block-swap pattern onto SD3.5-Large.
# Real SD3.5-Large base weights (streamed block-by-block via TurboPlannedLoader),
# real OneTrainer cache (latent_image + split CLIP/T5 hidden/pooled fields),
# full 38 joint-block depth.
# No synthetic tensors. Mirrors train_chroma_real.mojo's loop structure.
#
# SD3.5 vs CHROMA (the deltas):
#   - NO frozen approximator. Modulation comes from per-block adaLN_modulation.1
#     (streamed with each block), conditioned on c = t_embed(sigma*1000) + y_embed(pooled).
#   - JOINT BLOCKS ONLY: 38 joint blocks, no single-stream blocks.
#   - OneTrainer cache keys:
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
#   1. Load cached OneTrainer {latent_image, split hidden/pooled text fields}
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
# Recipe (from EriDiffusion-v2 prepare_sd35.rs / OneTrainer SD3.5 LoRA preset):
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
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from std.time import perf_counter_ns
from std.os import listdir, makedirs

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.tensor_view import from_parts

from serenitymojo.models.sd35.weights import load_sd35_stack_base
from serenitymojo.models.sd35.sd35_stack_lora import (
    SD35LoraSet, SD35LoraGradSet, SD35StackBase,
    build_sd35_lora_set, sd35_lora_adamw_step,
    save_sd35_lora, save_sd35_lora_state, total_adapters,
    sd35_stack_lora_forward_offload, sd35_stack_lora_backward_offload,
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
from serenitymojo.training.onetrainer_train_loop_policy import (
    OT_GRAD_POLICY_ON_ONLY,
    ot_cache_dir_from_train_config,
    ot_lr_for_optimizer_step,
    ot_output_lora_path_from_train_config,
    ot_sample_cadence_from_train_config,
    ot_sampling_enabled,
    ot_should_save_before_sample,
    ot_should_save_checkpoint,
    ot_step_lora_path,
    validate_ot_gradient_checkpointing_policy,
    validate_ot_train_math_policy,
)
from serenitymojo.training.train_config import (
    TrainConfig, GRADIENT_CHECKPOINTING_ON,
)
from serenitymojo.training.onetrainer_cache_preflight import (
    create_onetrainer_cache_preflight_plan,
    validate_onetrainer_cache_preflight_plan,
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
    validate_ot_train_math_policy(cfg, String("SD3.5 trainer"))
    validate_ot_gradient_checkpointing_policy(
        cfg, String("SD3.5 trainer"), OT_GRAD_POLICY_ON_ONLY
    )


def sd35_cache_dir_from_train_config(cfg: TrainConfig) -> String:
    return ot_cache_dir_from_train_config(cfg, String(CACHE_DIR))


def sd35_output_lora_path_from_train_config(cfg: TrainConfig, completed_step: Int) -> String:
    return ot_output_lora_path_from_train_config(
        cfg, String(LORA_DIR), String("sd35_lora"), completed_step
    )


def sd35_sample_cadence_from_train_config(
    cfg_path: String, cfg: TrainConfig,
) raises -> SampleCadence:
    return ot_sample_cadence_from_train_config(cfg_path, cfg)


def sd35_sampling_enabled(cadence: SampleCadence) -> Bool:
    return ot_sampling_enabled(cadence)


def sd35_should_save_checkpoint(cfg: TrainConfig, completed_step: Int) -> Bool:
    return ot_should_save_checkpoint(cfg, completed_step)


def sd35_should_save_before_sample(
    cadence: SampleCadence, completed_step: Int, saved_this_step: Bool,
) raises -> Bool:
    return ot_should_save_before_sample(cadence, completed_step, saved_this_step)


def _step_lora_path(base_path: String, step: Int) -> String:
    return ot_step_lora_path(base_path, step)


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
# Parity acceptance (OneTrainer speed/VRAM/trajectory evidence) remains a separate,
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
    # Legacy local cache stored OneTrainer's combined text handoff directly.
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
    var cache_preflight = create_onetrainer_cache_preflight_plan(train_cfg)
    validate_onetrainer_cache_preflight_plan(cache_preflight)

    var run_steps = DEFAULT_RUN_STEPS
    if len(a) > arg_base:
        run_steps = _parse_nonnegative_int(String(a[arg_base]))
    elif train_cfg.only_cache:
        run_steps = 0

    if len(a) > arg_base + 1:
        raise Error(
            String("SD3.5 trainer accepts [config.json] [steps] only; ")
            + String("start_step/state resume args are not wired for this loop")
        )

    var ckpt = sd35_checkpoint_from_train_config(train_cfg)
    var cache_dir = sd35_cache_dir_from_train_config(train_cfg)
    var sample_cadence = sd35_sample_cadence_from_train_config(cfg_path, train_cfg)
    var sample_enabled = sd35_sampling_enabled(sample_cadence)

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
    var loader = TurboPlannedLoader.open(ckpt, plan^, cfg, ctx)
    print("[load] offload loader opened (", loader.block_count(), "blocks)")

    # ── LoRA set (B=0 init -> identity at step 0) ────────────────────────────
    var lora = build_sd35_lora_set(NUM_JOINT, D, FMLP, RANK, ALPHA)
    var n_adapters = total_adapters(lora)
    print("[lora] adapters:", n_adapters, " (8 per joint block x", NUM_JOINT, "blocks)")

    var files = _list_cache(cache_dir)
    print("[cache] samples:", len(files))

    var b_absum_init = Float32(0.0)
    for i in range(n_adapters):
        b_absum_init += _absum(lora.ad[i].b)
    print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    var next_sample = next_sample_completed_step(sample_cadence, 0, train_cfg.max_steps)
    print("[cadence] next sample completed_step=", next_sample)

    # ── sample-during-training output dir (created once when sampling is on) ──
    var samples_dir = String(LORA_DIR) + String("/samples")
    if sample_enabled:
        makedirs(samples_dir, exist_ok=True)
        print("[cadence] sample-during-training WIRED -> ", samples_dir,
              " (", SAMPLE_STEPS, "-step CFG=", SAMPLE_CFG, " v1 cond=cached-caption)")

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)

    var train_start = perf_counter_ns()
    for k in range(1, run_steps + 1):
        var t0 = perf_counter_ns()

        var slot = 0 if FIXED_SIGMA_SMOKE else (k - 1) % len(files)
        var step_seed = UInt64(1) if FIXED_SIGMA_SMOKE else UInt64(k)
        var st = SafeTensors.open(files[slot])

        # latent_image: [1, 16, 128, 128] -> flat [1*16*128*128] = [262144].
        # OneTrainer caches raw VAE posterior mean here and applies
        # (latent_image - shift) * scale inside BaseStableDiffusion3Setup.predict.
        var latent_tensor = _load_cache_preferred(
            st, String("latent_image"), String("latent"), ctx
        )
        var latent_raw = _cache_tensor_to_stack_f32(latent_tensor, ctx)

        # OneTrainer caches split CLIP-L, CLIP-G, and T5 fields. The legacy
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

        # ── forward (offload, full depth) -> velocity [N_IMG, OUT_CH] ──
        var fwd = sd35_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
            noisy.copy(), txt_tokens.copy(), pooled_h.copy(), sigma_cont,
            base, loader, lora,
            D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
            EPS, QK_EPS, ctx,
        )

        # ── loss = MSE(pred, target) ; d_loss = (2/N)(pred - target) ──
        var nout = len(fwd.out)
        var d_loss = List[Float32]()
        var sse = 0.0
        var inv_n = Float32(2.0) / Float32(nout)
        for i in range(nout):
            var diff = fwd.out[i] - target[i]
            sse += Float64(diff) * Float64(diff)
            d_loss.append(inv_n * diff)
        var loss = Float32(sse / Float64(nout))
        if k == 1:
            first_loss = loss
        last_loss = loss

        # ── backward (offload, full depth) ──
        var grads = sd35_stack_lora_backward_offload[H, Dh, N_IMG, N_TXT, S](
            d_loss, noisy.copy(), txt_tokens.copy(),
            base, loader, lora, fwd,
            D, FMLP, IN_CH, TXT_CH, OUT_CH, TIMESTEP_DIM, POOLED_DIM,
            EPS, QK_EPS, ctx,
        )

        # ── grad norm + clip(1.0) ──
        var gn_before = _clip(grads, CLIP_GRAD_NORM)

        # ── AdamW ──
        var step_lr = ot_lr_for_optimizer_step(train_cfg, k)
        sd35_lora_adamw_step(
            lora, grads, k, step_lr, ctx,
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
            _ = save_sd35_lora(lora, save_path, ctx)
            var state_path = save_path + String(".state.safetensors")
            _ = save_sd35_lora_state(lora, state_path, ctx)
            saved_this_step = True
            print("[SD35-lora] save_state step=", k, " path=", state_path)
        if sample_enabled and should_sample_completed_step(sample_cadence, k):
            if sd35_should_save_before_sample(sample_cadence, k, saved_this_step):
                var sample_path = _step_lora_path(
                    sd35_output_lora_path_from_train_config(train_cfg, run_steps), k
                )
                _ = save_sd35_lora(lora, sample_path, ctx)
                var sample_state = sample_path + String(".state.safetensors")
                _ = save_sd35_lora_state(lora, sample_state, ctx)
                print("[SD35-lora] save_before_sample step=", k, " path=", sample_state)
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
    var trains = (b_absum_init == 0.0) and (b_absum_final > 0.0)
    if trains and (last_loss == last_loss):
        print("RESULT: REAL run OK — LoRA-B grew 0 ->", b_absum_final,
              "; loss", first_loss, "->", last_loss,
              (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        var lora_out = sd35_output_lora_path_from_train_config(train_cfg, run_steps)
        _ = save_sd35_lora(lora, lora_out, ctx)
        var state_out = lora_out + String(".state.safetensors")
        _ = save_sd35_lora_state(lora, state_out, ctx)
        print("[SD35-lora] save_state step=", run_steps, " path=", state_out)
    else:
        print("RESULT: FAIL trains=", trains)
