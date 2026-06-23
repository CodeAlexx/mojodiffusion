# train_chroma_real.mojo — Chroma1-HD LoRA training loop (block-swap offload).
#
# TRANSLATION of EriDiffusion-v2 chroma.rs onto the parity-verified Mojo Chroma
# LoRA OFFLOAD stack (models/chroma/chroma_stack_lora.mojo). Real Chroma1-HD base
# weights (streamed block-by-block via TurboPlannedLoader), real prepared cache
# (latent + T5), full 19+38 block depth. No synthetic tensors. Mirrors
# train_flux_real.mojo's loop structure (timing, grad clip, shared progress
# display) and chroma.rs's recipe.
#
# CHROMA vs FLUX (the deltas; see chroma_stack_lora.mojo header):
#   - NO guidance / CLIP-pooled vector. Modulation comes from the FROZEN
#     distilled_guidance_layer APPROXIMATOR (models/dit/chroma_dit.mojo
#     ChromaDitCache.approximator_forward), producing a per-step pooled_temb
#     table [mod_index=344, D=3072]; each block's ModVecs are sliced rows.
#   - Block math IS the proven Flux block (after the loader's separate->fused
#     row-stack), so the per-block LoRA fwd/bwd is REUSED verbatim and the LoRA
#     carrier / AdamW is shared while save uses Chroma's OneTrainer raw-key API.
#
# Per step:
#   1. load cached {latent [1,16,64,64] RAW, t5_embed [1,seq,4096]}
#   2. latent_scaled = (latent - SHIFT) * SCALE  (Chroma VAE shift/scale)
#   3. pack_latents([16,64,64]) -> [N_IMG=1024, 64] channel-major patchify
#   4. sigma_idx = floor(logit_normal_sigma(shift=1.15) * 1000) clamp;
#      sig=(idx+1)/1000 ; t_model=idx/1000
#   5. noisy = noise*sig + latent_packed*(1-sig) ; target = noise - latent_packed
#   6. pooled_temb = approximator(t_model)  (frozen; once per step)
#   7. chroma_stack_lora_forward_offload(noisy, txt, pooled, ...) -> pred [N_IMG,64]
#   8. loss = MSE(pred, target); d_loss = (2/N)(pred - target)
#   9. chroma_stack_lora_backward_offload -> LoRA grads; global-norm clip(1.0)
#  10. flux_lora_adamw_step; print shared progress display
#
# Recipe scalars (configs/chroma.json / chroma.rs): lr=1e-4, rank=16, alpha=16,
#   timestep_shift=1.15, clip_grad_norm=1.0, VAE shift=0.1159 scale=0.3611.
#
# MEMORY: the Chroma transformer is ~8.9B params (BF16 17.8GB on disk). The
# OFFLOAD path streams one block at a time + holds the resident base
# (x_embedder/context_embedder/proj_out ~tiny) + the approximator (~loaded once)
# + LoRA optimizer state. FULL 19+38 depth is the default.
#
# FIXED_SIGMA_SMOKE: every step uses the SAME cache sample AND a fixed
# timestep+noise so a correct LoRA backward MUST drive loss DOWN monotonically
# (the canonical trainer-correctness gate, same probe as train_flux_real).
#
# Run (real smoke):
#   cd /home/alex/mojodiffusion && rm -f serenitymojo.mojopkg && \
#     pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#       serenitymojo/training/train_chroma_real.mojo -o /tmp/train_chroma_real && \
#     /tmp/train_chroma_real [steps]

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

from serenitymojo.models.chroma.weights import load_chroma_stack_base
from serenitymojo.models.chroma.chroma_stack_lora import (
    ChromaStackBase,
    chroma_stack_lora_forward_offload, chroma_stack_lora_backward_offload,
    save_chroma_lora, save_chroma_lora_state,
)
from serenitymojo.models.flux.flux_stack_lora import (
    FluxLoraSet, FluxLoraGradSet, build_flux_lora_set,
    flux_lora_adamw_step, total_adapters,
)
from serenitymojo.models.flux.lora_block import DBL_STREAM_SLOTS, SGL_SLOTS
from serenitymojo.models.dit.flux1_dit import build_flux1_rope_tables
from serenitymojo.models.dit.chroma_dit import ChromaDitCache
from serenitymojo.offload.plan import build_chroma1_hd_block_plan, OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.schedule import sample_timestep_logit_normal
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.training.sample_prompt_config import (
    SampleCadence, read_sample_cadence_config,
    validate_step_sample_cadence, should_sample_completed_step,
    next_sample_completed_step, sample_time_unit_name,
    SAMPLE_UNIT_STEP, SAMPLE_UNIT_NEVER,
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
from serenitymojo.training.chroma_sample_resident import (
    chroma_sample_resident, chroma_decode_latent_to_png,
)


# ── arch (chroma1-hd; H/Dh/D fixed comptime, verified vs the checkpoint) ─────
comptime H = 24
comptime Dh = 128
comptime D = H * Dh            # 3072
comptime FMLP = 12288          # mlp_hidden = D*4
comptime IN_CH = 64            # patch_dim = 16ch * 2*2
comptime TXT_CH = 4096         # T5-XXL hidden
comptime OUT_CH = 64
comptime NUM_DOUBLE = 19
comptime NUM_SINGLE = 38
comptime MOD_INDEX = 3 * NUM_SINGLE + 2 * 6 * NUM_DOUBLE + 2   # 344
comptime EPS = Float32(1e-06)

# ── resolution (512px): latent [16,64,64] -> pack2 -> 32x32=1024 img tokens ──
comptime LAT_C = 16
comptime LAT_H = 64
comptime LAT_W = 64
comptime PATCH = 2
comptime HT = LAT_H // PATCH   # 32
comptime WT = LAT_W // PATCH   # 32
comptime N_IMG = HT * WT       # 1024
comptime N_TXT = 512           # T5 padded length
comptime S = N_TXT + N_IMG     # 1536

# ── recipe (configs/chroma.json) ─────────────────────────────────────────────
comptime RANK = 16
comptime ALPHA = Float32(16.0)
comptime LR = Float32(1.0e-4)
comptime TIMESTEP_SHIFT = Float32(1.15)
comptime VAE_SHIFT = Float32(0.1159)
comptime VAE_SCALE = Float32(0.3611)
comptime NUM_TRAIN_TIMESTEPS = 1000
comptime CLIP_GRAD_NORM = Float32(1.0)
comptime SEED_BASE = UInt64(42)

comptime FIXED_SIGMA_SMOKE = True
comptime FIXED_SIGMA_IDX = 500

comptime CKPT = "/home/alex/.serenity/models/checkpoints/chroma1_hd_bf16.safetensors"
comptime CACHE_DIR = "/home/alex/datasets/boxjana_chroma_edv2_512"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/chroma_boxjana"
comptime DEFAULT_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/chroma.json"
comptime DEFAULT_RUN_STEPS = 5

# ── sample-during-training (v1; chroma_sample_resident) ──────────────────────
# When the existing SampleCadence fires (should_sample_completed_step), denoise a
# sample from the CURRENT frozen base + streamed blocks + LIVE LoRA, decode with
# the FLUX VAE, and write <LORA_DIR>/samples/step_<N>.png. Geometry is the
# trainer's 512px latent (LAT_H=LAT_W=64 -> 8x VAE -> 512x512 image).
#   SAMPLE_STEPS / SAMPLE_CFG : denoise loop length + CFG (sampler defaults 30/4.0
#                               — chroma_sample_cli.mojo NUM_STEPS/GUIDANCE).
#   SAMPLE_SEED               : base RNG seed for the t=1 packed init noise.
# v1 CONDITIONING (flagged): no in-tree T5 tokenizer, so the COND text is the
#   CURRENT step's cached caption T5 embeds (the loop's txt_tokens); UNCOND is a
#   zero vector. See chroma_sample_resident.mojo header for the why + drop-in path.
comptime SAMPLE_STEPS = 30
comptime SAMPLE_CFG = Float32(4.0)
comptime SAMPLE_SEED = UInt64(0xC4_303A_5A91)
comptime SAMPLE_VAE_PATH = "/home/alex/.serenity/models/vaes/ae.safetensors"


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


def validate_chroma_train_config(cfg: TrainConfig) raises:
    if cfg.checkpoint == String(""):
        raise Error("Chroma trainer config must set checkpoint")
    if not cfg.checkpoint.endswith(String(".safetensors")):
        raise Error(
            String("Chroma trainer currently requires a single safetensors checkpoint; ")
            + String("sharded transformer dirs need a dedicated product loader")
        )
    if cfg.n_heads != H:
        raise Error(String("Chroma config n_heads ") + String(cfg.n_heads) + String(" != H ") + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("Chroma config head_dim ") + String(cfg.head_dim) + String(" != Dh ") + String(Dh))
    if cfg.d_model != D:
        raise Error(String("Chroma config d_model ") + String(cfg.d_model) + String(" != D ") + String(D))
    if cfg.in_channels != IN_CH:
        raise Error(String("Chroma config in_channels ") + String(cfg.in_channels) + String(" != IN_CH ") + String(IN_CH))
    if cfg.joint_attention_dim != TXT_CH:
        raise Error(String("Chroma config joint_attention_dim ") + String(cfg.joint_attention_dim) + String(" != TXT_CH ") + String(TXT_CH))
    if cfg.out_channels != OUT_CH:
        raise Error(String("Chroma config out_channels ") + String(cfg.out_channels) + String(" != OUT_CH ") + String(OUT_CH))
    if cfg.num_double != NUM_DOUBLE or cfg.num_single != NUM_SINGLE:
        raise Error(
            String("Chroma trainer requires double=") + String(NUM_DOUBLE)
            + String(" single=") + String(NUM_SINGLE)
            + String("; got double=") + String(cfg.num_double)
            + String(" single=") + String(cfg.num_single)
        )
    if cfg.mlp_hidden != FMLP:
        raise Error(String("Chroma config mlp_hidden ") + String(cfg.mlp_hidden) + String(" != FMLP ") + String(FMLP))
    if cfg.lora_rank != RANK:
        raise Error(
            String("Chroma trainer is compiled for lora_rank=")
            + String(RANK)
            + String("; parsed ")
            + String(cfg.lora_rank)
        )
    if not _close_f32(cfg.lora_alpha, ALPHA):
        raise Error("Chroma trainer lora_alpha does not match compiled constant")
    if not _close_f32(cfg.lr, LR, Float32(1.0e-9)):
        raise Error("Chroma trainer learning_rate does not match compiled constant")
    if not _close_f32(cfg.timestep_shift, TIMESTEP_SHIFT):
        raise Error("Chroma trainer timestep_shift does not match compiled constant")
    if not _close_f32(cfg.max_grad_norm, CLIP_GRAD_NORM):
        raise Error("Chroma trainer max_grad_norm does not match compiled constant")
    validate_ot_train_math_policy(cfg, String("Chroma trainer"))
    validate_ot_gradient_checkpointing_policy(
        cfg, String("Chroma trainer"), OT_GRAD_POLICY_ON_ONLY
    )


def chroma_checkpoint_from_train_config(cfg: TrainConfig) -> String:
    if cfg.checkpoint != String(""):
        return cfg.checkpoint.copy()
    return String(CKPT)


def chroma_cache_dir_from_train_config(cfg: TrainConfig) -> String:
    return ot_cache_dir_from_train_config(cfg, String(CACHE_DIR))


def chroma_output_lora_path_from_train_config(cfg: TrainConfig, completed_step: Int) -> String:
    return ot_output_lora_path_from_train_config(
        cfg, String(LORA_DIR), String("chroma_lora"), completed_step
    )


def chroma_sample_cadence_from_train_config(
    cfg_path: String, cfg: TrainConfig,
) raises -> SampleCadence:
    return ot_sample_cadence_from_train_config(cfg_path, cfg)


def chroma_sampling_enabled(cadence: SampleCadence) -> Bool:
    return ot_sampling_enabled(cadence)


def chroma_should_save_checkpoint(cfg: TrainConfig, completed_step: Int) -> Bool:
    return ot_should_save_checkpoint(cfg, completed_step)


def chroma_should_save_before_sample(
    cadence: SampleCadence, completed_step: Int, saved_this_step: Bool,
) raises -> Bool:
    return ot_should_save_before_sample(cadence, completed_step, saved_this_step)


def _step_lora_path(base_path: String, step: Int) -> String:
    return ot_step_lora_path(base_path, step)


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


def _global_norm(grads: FluxLoraGradSet) -> Float64:
    var ss = 0.0
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    return sqrt(ss)


def _clip(mut grads: FluxLoraGradSet, max_norm: Float32) -> Float64:
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
        raise Error(String("chroma cache: no .safetensors in ") + dir)
    for i in range(1, len(fs)):
        var j = i
        while j > 0 and fs[j - 1] > fs[j]:
            var tmp = fs[j - 1]
            fs[j - 1] = fs[j]
            fs[j] = tmp
            j -= 1
    return fs^


def _load_chroma_cache_tensor(st: SafeTensors, name: String, ctx: DeviceContext) raises -> Tensor:
    var info = st.tensor_info(name)
    var bytes = st.tensor_bytes(name)
    var tv = from_parts(info.dtype, info.shape.copy(), bytes)
    return Tensor.from_view(tv, ctx)


def _host_f32_for_step_math(t: Tensor, ctx: DeviceContext) raises -> List[Float32]:
    """Stage tensors through their stored dtype before host step math."""
    if t.dtype() == STDtype.BF16:
        var bf = t.to_host_bf16(ctx)
        var out = List[Float32]()
        for i in range(len(bf)):
            out.append(bf[i].cast[DType.float32]())
        return out^
    if t.dtype() == STDtype.F16:
        var hf = t.to_host_f16(ctx)
        var out = List[Float32]()
        for i in range(len(hf)):
            out.append(hf[i].cast[DType.float32]())
        return out^
    return t.to_host(ctx)


# pack_latents: [16,LAT_H,LAT_W] flat -> [N_IMG, 64] channel-major patchify.
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


# Build the frozen per-step modulation table [MOD_INDEX*D] as a Tensor.
def _pooled_modulation_tensor(
    approx: ChromaDitCache, t_model: Float32, ctx: DeviceContext,
) raises -> Tensor:
    var approx_in = approx._approximator_input(t_model, ctx)
    return approx.approximator_forward(approx_in, ctx)   # [1, MOD_INDEX, D] BF16


# ── deterministic host gaussian PACKED init noise [N_IMG*IN_CH] ──────────────
# Reuses the same Box-Muller PCG as the train loop's _host_noise (so the sample's
# t=1 latent is drawn the same way the training noise is). seed makes it
# deterministic per sampled step.
def _sample_init_noise(seed: UInt64) -> List[Float32]:
    return _host_noise(N_IMG * IN_CH, seed)


# ── _chroma_run_sample — one sample-during-training image ────────────────────
#   cond text    : the current step's cached caption T5 embeds (v1; see header).
#   uncond text  : a zeroed [N_TXT*TXT_CH] vector (CFG empty cond).
#   init noise   : packed gaussian [N_IMG*IN_CH], seed = SAMPLE_SEED + step.
#   denoise      : chroma_sample_resident (frozen base + streamed blocks + live
#                  LoRA + frozen approximator).
#   decode+write : chroma_decode_latent_to_png -> <samples_dir>/step_<N>.png.
# Fail-loud: any raise propagates (no silent skip), matching the trainer's
# fail-loud cadence contract.
def _chroma_run_sample(
    base: ChromaStackBase,
    approx: ChromaDitCache,
    mut loader: TurboPlannedLoader,
    lora: FluxLoraSet,
    cond_txt: List[Float32],     # [N_TXT*TXT_CH] — the step's cached caption embeds
    cos: List[Float32],
    sin: List[Float32],
    samples_dir: String,
    step: Int,
    ctx: DeviceContext,
) raises:
    # UNCOND: zeroed text features (same dtype/shape as cond_txt).
    var uncond_txt = List[Float32]()
    for _ in range(N_TXT * TXT_CH):
        uncond_txt.append(Float32(0.0))

    var init_noise = _sample_init_noise(SAMPLE_SEED + UInt64(step))

    var latent = chroma_sample_resident[H, Dh, N_IMG, N_TXT, S](
        base, approx, loader, lora,
        cond_txt.copy(), uncond_txt^, init_noise^,
        cos.copy(), sin.copy(),
        SAMPLE_STEPS, SAMPLE_CFG, MOD_INDEX,
        D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
    )

    var out_path = samples_dir + String("/step_") + String(step) + String(".png")
    chroma_decode_latent_to_png[LAT_C, LAT_H, LAT_W, HT, WT, PATCH, N_IMG, IN_CH](
        latent, String(SAMPLE_VAE_PATH), out_path, ctx,
    )
    print("[Chroma-lora] sample step=", step, " -> ", out_path)


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
    validate_chroma_train_config(train_cfg)
    var cache_preflight = create_onetrainer_cache_preflight_plan(train_cfg)
    validate_onetrainer_cache_preflight_plan(cache_preflight)

    var run_steps = DEFAULT_RUN_STEPS
    if len(a) > arg_base:
        run_steps = _parse_nonnegative_int(String(a[arg_base]))
    elif train_cfg.only_cache:
        run_steps = 0

    var ckpt = chroma_checkpoint_from_train_config(train_cfg)
    var cache_dir = chroma_cache_dir_from_train_config(train_cfg)
    var sample_cadence = chroma_sample_cadence_from_train_config(cfg_path, train_cfg)
    var sample_enabled = chroma_sampling_enabled(sample_cadence)

    print("=== Chroma (chroma1-hd) REAL LoRA training loop (block-swap offload) ===")
    print("  config:", cfg_path)
    print("  arch: D=", D, " H=", H, " Dh=", Dh, " Fmlp=", FMLP, " out_ch=", OUT_CH)
    print("  depth: NUM_DOUBLE=", NUM_DOUBLE, " NUM_SINGLE=", NUM_SINGLE, " mod_index=", MOD_INDEX)
    print("  tokens: N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S)
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
        print("[offload] async offload requested by config; Chroma trainer currently uses synchronous TurboPlannedLoader")
    if train_cfg.only_cache:
        print("[Chroma] only_cache requested; no train steps will run in this trainer")
        return

    var ctx = DeviceContext()

    # ── stack-level base (frozen; x_embedder/context_embedder/proj_out) ──────
    print("[load] ChromaStackBase (x_embedder, context_embedder, proj_out)")
    var base_st = SafeTensors.open(ckpt)
    var base = load_chroma_stack_base(base_st, NUM_DOUBLE, NUM_SINGLE, ctx)
    print("[load] base resident")

    # ── frozen approximator (distilled_guidance_layer) ───────────────────────
    print("[load] approximator (distilled_guidance_layer)")
    var approx = ChromaDitCache.load(ckpt, ctx)
    print("[load] approximator resident")

    # ── block-swap offload loader ────────────────────────────────────────────
    var plan = build_chroma1_hd_block_plan()
    var cfg = OffloadConfig.synchronous_single()
    var loader = TurboPlannedLoader.open(ckpt, plan^, cfg, ctx)
    print("[load] offload loader opened (", loader.block_count(), "blocks)")

    # ── 3-axis RoPE tables (positions fixed for 512px; built once, BF16) ─────
    var rope = build_flux1_rope_tables[N_IMG, N_TXT, H, Dh](HT, WT, ctx, STDtype.BF16)
    var cos = rope[0].to_host(ctx)
    var sin = rope[1].to_host(ctx)
    print("[load] chroma 3-axis rope tables built (S*H x Dh/2)")

    # ── LoRA set (B=0 init -> identity at step 0) ────────────────────────────
    var lora = build_flux_lora_set(NUM_DOUBLE, NUM_SINGLE, D, FMLP, RANK, ALPHA)
    var n_adapters = total_adapters(lora)
    print("[lora] adapters:", n_adapters,
          " (", DBL_STREAM_SLOTS * 2, "x", NUM_DOUBLE, "double +",
          SGL_SLOTS, "x", NUM_SINGLE, "single)")

    var files = _list_cache(cache_dir)
    print("[cache] samples:", len(files))

    var b_absum_init = Float32(0.0)
    for i in range(n_adapters):
        b_absum_init += _absum(lora.ad[i].b)
    print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    # samples dir for sample-during-training PNGs (created once if enabled).
    var samples_dir = String(LORA_DIR) + String("/samples")
    if sample_enabled:
        makedirs(samples_dir, exist_ok=True)
        print("[cadence] sample-during-training WIRED -> ", samples_dir,
              " (steps=", SAMPLE_STEPS, " cfg=", SAMPLE_CFG, ")")
    if sample_enabled and should_sample_completed_step(sample_cadence, 0):
        print("[cadence] step 0 sample due (fires after the first completed step in this bounded loop)")
    var next_sample = next_sample_completed_step(sample_cadence, 0, train_cfg.max_steps)
    print("[cadence] next sample completed_step=", next_sample)

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)

    var train_start = perf_counter_ns()
    for k in range(1, run_steps + 1):
        var t0 = perf_counter_ns()

        var slot = 0 if FIXED_SIGMA_SMOKE else (k - 1) % len(files)
        var step_seed = UInt64(1) if FIXED_SIGMA_SMOKE else UInt64(k)
        var st = SafeTensors.open(files[slot])
        var latent_cache = _load_chroma_cache_tensor(st, String("latent"), ctx)
        var lat_raw = _host_f32_for_step_math(latent_cache, ctx)         # [16*64*64]

        # t5_embed [1, seq, 4096] -> pad/truncate to [N_TXT, 4096] (zero pad rows).
        var t5_info = st.tensor_info(String("t5_embed"))
        var t5_seq = Int(t5_info.shape[1])
        var t5_cache = _load_chroma_cache_tensor(st, String("t5_embed"), ctx)
        var t5_flat = _host_f32_for_step_math(t5_cache, ctx)
        var txt_tokens = List[Float32]()
        for r in range(N_TXT):
            if r < t5_seq:
                for c in range(TXT_CH):
                    txt_tokens.append(t5_flat[r * TXT_CH + c])
            else:
                for _ in range(TXT_CH):
                    txt_tokens.append(Float32(0.0))

        # ── VAE shift/scale then pack_latents ──
        for i in range(len(lat_raw)):
            lat_raw[i] = (lat_raw[i] - VAE_SHIFT) * VAE_SCALE
        var latent_packed = _pack_latents(lat_raw)

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
        var t_model = Float32(sigma_idx) / Float32(NUM_TRAIN_TIMESTEPS)

        # ── flow-match in PACKED latent space ──
        var noise = _host_noise(N_IMG * IN_CH, SEED_BASE * UInt64(7919) + step_seed)
        var noisy = List[Float32]()
        var target = List[Float32]()
        for i in range(len(latent_packed)):
            noisy.append(noise[i] * sig + latent_packed[i] * (Float32(1.0) - sig))
            target.append(noise[i] - latent_packed[i])

        # ── frozen approximator -> modulation table ──
        var pooled_tensor = _pooled_modulation_tensor(approx, t_model, ctx)
        var pooled = _host_f32_for_step_math(pooled_tensor, ctx)

        # ── forward (offload, full depth) -> velocity [N_IMG, OUT_CH] ──
        var fwd = chroma_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
            noisy.copy(), txt_tokens.copy(), pooled.copy(), MOD_INDEX,
            base, loader, lora, cos.copy(), sin.copy(),
            D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
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
        var grads = chroma_stack_lora_backward_offload[H, Dh, N_IMG, N_TXT, S](
            d_loss, noisy.copy(), txt_tokens.copy(), base, loader, lora,
            cos.copy(), sin.copy(), fwd,
            D, FMLP, IN_CH, TXT_CH, OUT_CH, EPS, ctx,
        )

        # ── grad norm + clip(1.0) ──
        var gn_before = _clip(grads, CLIP_GRAD_NORM)

        # ── AdamW ──
        var step_lr = ot_lr_for_optimizer_step(train_cfg, k)
        flux_lora_adamw_step(
            lora, grads, k, step_lr, ctx,
            train_cfg.beta1, train_cfg.beta2, train_cfg.eps,
            train_cfg.weight_decay,
        )

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        print_trainer_progress(
            String("Chroma-lora"), k, run_steps, 1,
            loss, Float64(gn_before), secs, 0.0,
            Float64(t1 - train_start) / 1.0e9,
        )
        if grads.nonfinite_lora_grads != 0:
            print("[Chroma-lora] warning nonfinite_lora_grads=", grads.nonfinite_lora_grads)

        var saved_this_step = False
        if chroma_should_save_checkpoint(train_cfg, k):
            var save_path = _step_lora_path(
                chroma_output_lora_path_from_train_config(train_cfg, run_steps), k
            )
            _ = save_chroma_lora(lora, save_path, ctx)
            var state_path = save_path + String(".state.safetensors")
            _ = save_chroma_lora_state(lora, state_path, ctx)
            saved_this_step = True
            print("[Chroma-lora] save_state step=", k, " path=", state_path)
        if sample_enabled and should_sample_completed_step(sample_cadence, k):
            if chroma_should_save_before_sample(sample_cadence, k, saved_this_step):
                var sample_path = _step_lora_path(
                    chroma_output_lora_path_from_train_config(train_cfg, run_steps), k
                )
                _ = save_chroma_lora(lora, sample_path, ctx)
                var sample_state = sample_path + String(".state.safetensors")
                _ = save_chroma_lora_state(lora, sample_state, ctx)
                print("[Chroma-lora] save_before_sample step=", k, " path=", sample_state)
            # Sample from the CURRENT frozen base + streamed blocks + LIVE LoRA.
            # v1 conditioning: this step's cached caption T5 embeds (txt_tokens)
            # as COND, zeros as UNCOND. See chroma_sample_resident.mojo header.
            print(
                "[cadence] sample due at completed_step=", k,
                " sample_file=", sample_cadence.sample_definition_file_name,
            )
            _chroma_run_sample(
                base, approx, loader, lora, txt_tokens.copy(),
                cos.copy(), sin.copy(), samples_dir, k, ctx,
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
        var lora_out = chroma_output_lora_path_from_train_config(train_cfg, run_steps)
        _ = save_chroma_lora(lora, lora_out, ctx)
        var state_out = lora_out + String(".state.safetensors")
        _ = save_chroma_lora_state(lora, state_out, ctx)
        print("[Chroma-lora] save_state step=", run_steps, " path=", state_out)
    else:
        print("RESULT: FAIL trains=", trains)
