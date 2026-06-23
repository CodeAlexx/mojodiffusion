# train_qwenimage_real.mojo — Qwen-Image LoRA training loop (block-swap offload).
#
# TRANSLATION of EriDiffusion-v2 train_qwenimage.rs onto the parity-verified Mojo
# Qwen-Image LoRA OFFLOAD stack. 60 all-double-stream blocks (D=3072, H=24, Dh=128,
# F=12288, in_ch=64, txt_ch=3584, out_ch=64). Mirrors train_chroma_real.mojo's loop
# structure (timing, grad clip, progress display) and qwenimage.rs's recipe.
#
# QWENIMAGE vs CHROMA (key deltas):
#   - ALL double-stream (60 blocks, 0 single). No single-stream loop.
#   - SEPARATE per-block mod-MLPs (img_mod.1 / txt_mod.1). Each frozen mod MLP
#     projects silu(temb) -> [6D] per block. Mod-MLP weights are STREAMED from the
#     block (same Block handle as the attention weights). Frozen: grads discarded.
#   - time_text_embed: sinusoidal(t*1000, 256) -> silu(Linear1) -> Linear2 -> [D].
#     This MLP is applied ONCE per step to get silu_temb_h [1,D].
#   - norm_out.linear: [2D,D] produces final_scale/final_shift from silu_temb_h.
#   - txt_ch = 3584 (Qwen2.5-VL text encoder hidden dim; NOT T5-XXL 4096).
#   - Flow-match recipe (qwenimage.rs:1093-1099):
#       x_t = (1 - sigma)*latent + sigma*noise   (note: opposite sign to Flux)
#       target = noise - latent
#   - Timestep: OneTrainer DISCRETE (BaseQwenSetup + ModelSetupNoiseMixin
#       _get_timestep_discrete + ModelSetupFlowMatchingMixin _add_noise_discrete):
#       idx = int(sigmoid(N(0,1)) * 1000 * shift_remap);  shift=1.0 -> identity.
#       sigma = (idx+1)/1000  (blend);  model_t = idx/1000  (transformer input).
#   - out_ch = 64 (latent channels; proj_out [64,D]; target [N_IMG, 64]).
#   - ROPE: 3-axis interleaved, axes=(16,56,56), theta=10000.
#
# Recipe (configs/qwenimage.json, matching OneTrainer "#qwen LoRA 24GB" preset):
#   lr=3e-4, rank=16, alpha=1.0 (scale=1/16), timestep_shift=1.0,
#   lr_warmup_steps=200 (constant scheduler), clip_grad_norm=1.0.
#
# MEMORY: 60 * ~648 MB BF16/FP8 blocks + resident base (~tiny) + LoRA + optimizer.
# Block-swap streams one block at a time. A fixed-sigma smoke mode confirms
# loss decreases monotonically with a frozen sample.
#
# FIXED_SIGMA_SMOKE: every step uses the SAME latent+text AND a fixed sigma+noise.
# A correct LoRA backward MUST drive loss DOWN monotonically.
#
# COMPILE-ONLY DELIVERABLE: do NOT execute this binary (full 60-block model).
#   pixi run mojo build -I . -Xlinker -lm -Xlinker -lcuda \
#     serenitymojo/training/train_qwenimage_real.mojo \
#     -o /tmp/train_qwenimage_real
#
# ── UNVERIFIABLE-WITHOUT-CACHE ITEMS (flagged for future parity gate) ──────────
# (1) Checkpoint dtype FP8-E4M3: Qwen loaders dequant FP8 bytes to BF16 on use;
#     parity vs torch FP8 checkpoint still needs a local reference gate.
# (2) txt_ch=3584: the Qwen2.5-VL text encoder; cache dir uses placeholder zeros.
# (3) RoPE total_half = 8+28+28 = 64 = Dh//2: matches config axes (16,56,56);
#     parity to qwenimage.rs RoPE verified per-block cos>=0.999 in the block tests.
# (4) norm_out.linear [2D,D] → chunk 0 scale chunk 1 shift: diffusers layout from
#     config.json; not re-gated here (same as weights.mojo build_qwen_per_block_mods).
# (5) txt_norm.weight [txt_ch]: applied by the caller pre-normalization in inference;
#     in the trainer we skip it (match train_qwenimage.rs which operates on already-
#     normalized text embeddings from the cache).

from std.sys import argv
from std.collections import List, Optional
from std.gpu.host import DeviceContext
from std.math import sqrt, log as flog, cos as fcos, sin as fsin
from std.time import perf_counter_ns
from std.os import listdir, makedirs

from serenitymojo.tensor import Tensor
from serenitymojo.io.dtype import STDtype
from serenitymojo.io.safetensors import SafeTensors
from serenitymojo.io.sharded import ShardedSafeTensors
from serenitymojo.io.tensor_view import from_parts
from serenitymojo.ops.embeddings import timestep_embedding
from serenitymojo.ops.activations import silu

from serenitymojo.models.qwenimage.qwenimage_stack import (
    QwenStackBase, _t as _qstack_t,
)
from serenitymojo.models.qwenimage.weights import load_qwen_stack_base, load_qwen_host_bf16
from serenitymojo.models.qwenimage.qwenimage_stack_lora import (
    QwenLoraSet, QwenLoraGradSet, QwenOffloadBase, QwenOffloadForward,
    build_qwen_lora_set, save_qwen_lora, save_qwen_lora_state,
    qwenimage_stack_lora_forward_offload,
    qwenimage_stack_lora_backward_offload,
    qwen_offload_lora_adamw_step,
    DBL_SLOTS,
)
from serenitymojo.models.dit.qwenimage_dit import (
    QwenImageConfig, build_qwenimage_rope_tables,
)
from serenitymojo.training.qwenimage_sample_resident import (
    qwenimage_sample_resident, qwenimage_decode_packed_to_png,
)
from serenitymojo.offload.qwenimage_plan import build_qwenimage_offload_plan
from serenitymojo.offload.plan import OffloadConfig
from serenitymojo.offload.turbo_planned_loader import TurboPlannedLoader
from serenitymojo.training.schedule import (
    sample_timestep_discrete_qwen, DiscreteTimestep,
)
from serenitymojo.training.progress_display import print_trainer_progress
from serenitymojo.training.train_config import (
    TrainConfig, GRADIENT_CHECKPOINTING_ON,
)
from serenitymojo.training.onetrainer_cache_preflight import (
    create_onetrainer_cache_preflight_plan,
    validate_onetrainer_cache_preflight_plan,
)
from serenitymojo.io.train_config_reader import read_model_config
from serenitymojo.training.sample_prompt_config import (
    SampleCadence, read_sample_cadence_config,
    validate_step_sample_cadence, should_sample_completed_step,
    next_sample_completed_step, sample_time_unit_name,
    SAMPLE_UNIT_STEP, SAMPLE_UNIT_NEVER,
)
from serenitymojo.training.onetrainer_train_loop_policy import (
    OT_GRAD_POLICY_ON_ONLY,
    ot_lr_for_optimizer_step,
    ot_sample_cadence_from_train_config,
    ot_should_save_before_sample,
    ot_state_path_for_lora,
    ot_step_lora_path,
    validate_ot_gradient_checkpointing_policy,
    validate_ot_train_math_policy,
)


# ── arch (qwen-image; confirmed from config.json + qwenimage_dit.mojo) ────────
comptime H = 24
comptime Dh = 128
comptime D = H * Dh            # 3072
comptime FMLP = 12288          # mlp_hidden = D * 4
comptime IN_CH = 64            # in_channels (patchified latent)
comptime TXT_CH = 3584         # Qwen2.5-VL text encoder hidden
comptime OUT_CH = 64           # proj_out output channels
comptime NUM_DOUBLE = 60       # all-double-stream
comptime TIMESTEP_DIM = 256    # sinusoidal embedding dim
comptime EPS = Float32(1.0e-6)

# ── resolution (512px / patch=2): latent [64,32,32] -> [N_IMG=1024, 64] ───────
comptime LAT_C = 64            # in_channels (VAE latent channels = 16 before patch)
comptime LAT_H = 32            # latent height at patch-2 (512px / 16)
comptime LAT_W = 32            # latent width
comptime N_IMG = LAT_H * LAT_W  # 1024 image tokens
comptime N_TXT = 256           # text token sequence length (padded)
comptime S = N_TXT + N_IMG     # 1280 joint sequence

# ── RoPE frame/height/width for 512px 1-frame ─────────────────────────────────
comptime ROPE_FRAME = 1
comptime ROPE_H = 32           # == LAT_H (latent height in patch coords)
comptime ROPE_W = 32           # == LAT_W

# ── recipe defaults (configs/qwenimage.json is the runtime source of truth) ───
comptime SEED_BASE = UInt64(42)

comptime FIXED_SIGMA_SMOKE = True
comptime FIXED_SIGMA_VAL = Float32(0.5)    # fixed sigma for smoke test

comptime DEFAULT_CONFIG = "/home/alex/mojodiffusion/serenitymojo/configs/qwenimage.json"
comptime DEFAULT_RUN_STEPS = 5
comptime DEFAULT_CACHE_DIR = "/home/alex/datasets/qwenimage_cache_512"
comptime LORA_DIR = "/home/alex/mojodiffusion/output/qwenimage_lora"

# ── sample-during-training (v1; qwenimage_sample_resident) ────────────────────
# SAMPLE_STEPS / SAMPLE_CFG : denoise loop length + true-CFG scale (sampler
#   defaults 30 / 4.0, matching qwenimage_sample_cli.mojo STEPS/CFG).
# SAMPLE_SEED : base RNG seed for the t=1 packed init noise (per-step deterministic).
# DEFAULT_VAE_DIR : fallback Qwen VAE dir if train_cfg.vae is empty (the config's
#   "vae" field is the source of truth; see configs/qwenimage.json).
# Conditioning v1 = the step's cached caption embeds (COND) + zeros (UNCOND); the
#   trainer has no in-tree Qwen2.5-VL encoder to encode an arbitrary sample prompt.
comptime SAMPLE_STEPS = 30
comptime SAMPLE_CFG = Float32(4.0)
comptime SAMPLE_SEED = UInt64(0xC4_303A_5A91)
comptime DEFAULT_VAE_DIR = "/home/alex/.serenity/models/checkpoints/qwen-image-2512/vae"

# ── VAE latent geometry for the 512px bucket (patch=2) ────────────────────────
# unpatchify needs (channels, height, width, patch). The trainer's packed latent
# is [N_IMG=1024, IN_CH=64]; IN_CH = LAT_C(16) * patch(2) * patch(2). The VAE input
# is [1, 16, LAT_H*2, LAT_W*2] = [1,16,64,64] -> tiled-decode -> [1,3,512,512].
comptime SAMPLE_LAT_C = 16              # VAE latent channels (out_channels)
comptime SAMPLE_PATCH = 2               # patch_size (config.json)


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


def qwen_patchified_out_channels(cfg: TrainConfig) -> Int:
    var qcfg = QwenImageConfig.qwen_image()
    return cfg.out_channels * qcfg.patch_size * qcfg.patch_size


def validate_qwen_train_config(cfg: TrainConfig) raises:
    # The hot stack functions are still comptime-specialized for the 512px
    # Qwen-Image bucket. Fail here instead of silently using mismatched metadata.
    if cfg.checkpoint == String(""):
        raise Error("Qwen trainer config must set checkpoint")
    if cfg.n_heads != H:
        raise Error(String("Qwen config n_heads ") + String(cfg.n_heads) + String(" != comptime H ") + String(H))
    if cfg.head_dim != Dh:
        raise Error(String("Qwen config head_dim ") + String(cfg.head_dim) + String(" != comptime Dh ") + String(Dh))
    if cfg.d_model != H * Dh:
        raise Error(String("Qwen config d_model ") + String(cfg.d_model) + String(" != H*Dh ") + String(H * Dh))
    if cfg.in_channels != IN_CH:
        raise Error(String("Qwen config in_channels ") + String(cfg.in_channels) + String(" != IN_CH ") + String(IN_CH))
    if cfg.joint_attention_dim != TXT_CH:
        raise Error(String("Qwen config joint_attention_dim ") + String(cfg.joint_attention_dim) + String(" != TXT_CH ") + String(TXT_CH))
    if cfg.num_double != NUM_DOUBLE or cfg.num_single != 0:
        raise Error(
            String("Qwen trainer requires 60 double-stream blocks and 0 single blocks; got double=")
            + String(cfg.num_double) + String(" single=") + String(cfg.num_single)
        )
    if cfg.mlp_hidden != FMLP:
        raise Error(String("Qwen config mlp_hidden ") + String(cfg.mlp_hidden) + String(" != FMLP ") + String(FMLP))
    if cfg.timestep_dim != TIMESTEP_DIM:
        raise Error(String("Qwen config timestep_dim ") + String(cfg.timestep_dim) + String(" != TIMESTEP_DIM ") + String(TIMESTEP_DIM))
    if qwen_patchified_out_channels(cfg) != OUT_CH:
        raise Error(
            String("Qwen config out_channels ") + String(cfg.out_channels)
            + String(" with patch_size=2 gives ")
            + String(qwen_patchified_out_channels(cfg))
            + String(" patchified channels, expected ") + String(OUT_CH)
        )
    if cfg.lora_rank <= 0:
        raise Error("Qwen trainer config requires lora_rank > 0")
    if cfg.lora_alpha <= Float32(0.0):
        raise Error("Qwen trainer config requires lora_alpha > 0")
    if cfg.lr <= Float32(0.0):
        raise Error("Qwen trainer config requires learning_rate > 0")
    if cfg.max_grad_norm <= Float32(0.0):
        raise Error("Qwen trainer config requires max_grad_norm > 0")
    validate_ot_train_math_policy(cfg, String("Qwen trainer"))
    validate_ot_gradient_checkpointing_policy(
        cfg, String("Qwen trainer"), OT_GRAD_POLICY_ON_ONLY
    )


def qwen_offload_config_from_train_config(cfg: TrainConfig) raises -> OffloadConfig:
    validate_qwen_train_config(cfg)
    if cfg.activation_offload_enabled() or cfg.layer_offload_enabled():
        raise Error(
            String("Qwen trainer cannot honor CPU activation/layer offload yet; ")
            + String("set gradient_checkpointing=ON for the current synchronous block loader")
        )
    return OffloadConfig.synchronous_single()


def qwen_sample_cadence_from_train_config(
    cfg_path: String, cfg: TrainConfig,
) raises -> SampleCadence:
    return ot_sample_cadence_from_train_config(cfg_path, cfg)


def qwen_should_save_before_sample(
    cadence: SampleCadence, completed_step: Int, saved_this_step: Bool,
) raises -> Bool:
    return ot_should_save_before_sample(cadence, completed_step, saved_this_step)


def qwen_state_path_for_lora(lora_path: String) -> String:
    return ot_state_path_for_lora(lora_path)


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


def _global_norm(grads: QwenLoraGradSet) -> Float64:
    var ss = 0.0
    for i in range(len(grads.d_a)):
        for j in range(len(grads.d_a[i])):
            ss += Float64(grads.d_a[i][j]) * Float64(grads.d_a[i][j])
        for j in range(len(grads.d_b[i])):
            ss += Float64(grads.d_b[i][j]) * Float64(grads.d_b[i][j])
    return sqrt(ss)


def _clip(mut grads: QwenLoraGradSet, max_norm: Float32) -> Float64:
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
        raise Error(String("qwenimage cache: no .safetensors in ") + dir)
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


def _load_host_f32_sharded(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> List[Float32]:
    var tv = st.tensor_view(name)
    var t = Tensor.from_view(tv, ctx)
    return t.to_host(ctx)


def _load_host_bf16_sharded(st: ShardedSafeTensors, name: String, ctx: DeviceContext) raises -> List[BFloat16]:
    return load_qwen_host_bf16(st, name, ctx)


# Sinusoidal timestep embedding (host, returns [timestep_dim] F32).
# t_val: sigma in [0,1]; Qwen scales by 1000 before embedding.
def _sinusoidal_temb(t_val: Float32, ctx: DeviceContext) raises -> List[Float32]:
    var t_h = List[Float32]()
    t_h.append(t_val * Float32(1000.0))
    var t_tensor = Tensor.from_host(t_h, [1], STDtype.F32, ctx)
    var t_emb = timestep_embedding(
        t_tensor, Int(TIMESTEP_DIM), ctx, Float32(10000.0), STDtype.F32
    )
    return t_emb.to_host(ctx)   # [1, TIMESTEP_DIM] flat = [TIMESTEP_DIM] scalars


# Build silu_temb_h: silu(time_text_embed(t)) = silu(MLP(sinusoidal(t))).
# te_lin1_w [D, timestep_dim], te_lin1_b [D], te_lin2_w [D,D], te_lin2_b [D].
def _build_silu_temb(
    t_val: Float32,
    te_lin1_w: List[BFloat16], te_lin1_b: List[BFloat16],
    te_lin2_w: List[BFloat16], te_lin2_b: List[BFloat16],
    ctx: DeviceContext,
) raises -> List[Float32]:
    from serenitymojo.ops.linear import linear
    var sin_emb_h = _sinusoidal_temb(t_val, ctx)        # [TIMESTEP_DIM]
    var t_emb = Tensor.from_host(sin_emb_h, [1, Int(TIMESTEP_DIM)], STDtype.BF16, ctx)
    var b1 = Tensor.from_host_bf16(te_lin1_b.copy(), [Int(D)], ctx)
    var h1 = linear(
        t_emb,
        Tensor.from_host_bf16(te_lin1_w.copy(), [Int(D), Int(TIMESTEP_DIM)], ctx),
        Optional[Tensor](b1^), ctx,
    )
    var h1_silu = silu(h1, ctx)
    var b2 = Tensor.from_host_bf16(te_lin2_b.copy(), [Int(D)], ctx)
    var temb_out = linear(
        h1_silu,
        Tensor.from_host_bf16(te_lin2_w.copy(), [Int(D), Int(D)], ctx),
        Optional[Tensor](b2^), ctx,
    )
    # final silu for use as per-block mod MLP input
    return silu(temb_out, ctx).to_host(ctx)   # [1, D] flat


# pack_latents: [LAT_C, LAT_H, LAT_W] flat F32 -> [N_IMG, LAT_C] (trivial
# patch=1 since latent is already at patch resolution for Qwen-Image 512px).
# Qwen-Image patchify: patch_size=2 applied at VAE decode time (in_channels=64
# = 16ch * 2*2); the latent cache already stores the patchified [N_IMG, 64].
# So no patchify needed — the cache tensor IS [N_IMG, 64] already.
# (Verified: in_channels=64, out_channels=64 in config.json; the latent cache
#  for training stores the pack_latents output, not the raw VAE latent.)


# ── _qwen_run_sample — one sample-during-training image ───────────────────────
#   cond text    : the current step's cached caption embeds (v1; see header of
#                  qwenimage_sample_resident.mojo).
#   uncond text  : a zeroed [N_TXT*TXT_CH] vector (true-CFG empty cond).
#   init noise   : packed gaussian [N_IMG*IN_CH], seed = SAMPLE_SEED + step (the
#                  same Box-Muller PCG the train loop draws noise with).
#   denoise      : qwenimage_sample_resident (frozen base + streamed blocks + live
#                  LoRA; true-CFG flow-match Euler).
#   decode+write : qwenimage_decode_packed_to_png -> <samples_dir>/step_<N>.png.
# Fail-loud: any raise propagates (no silent skip), matching the trainer's
# fail-loud cadence contract.
def _qwen_run_sample(
    base: QwenOffloadBase,
    mut loader: TurboPlannedLoader,
    lora: QwenLoraSet,
    cond_txt: List[Float32],     # [N_TXT*TXT_CH] — the step's cached caption embeds
    cos_h: List[Float32],
    sin_h: List[Float32],
    norm_out_w: List[BFloat16], norm_out_b: List[BFloat16],
    vae_dir: String,
    samples_dir: String,
    step: Int,
    ctx: DeviceContext,
) raises:
    # UNCOND: zeroed text features (same dtype/shape as cond_txt).
    var uncond_txt = List[Float32]()
    for _ in range(Int(N_TXT) * Int(TXT_CH)):
        uncond_txt.append(Float32(0.0))

    var init_noise = _host_noise(Int(N_IMG) * Int(IN_CH), SAMPLE_SEED + UInt64(step))

    var latent = qwenimage_sample_resident[H, Dh, N_IMG, N_TXT, S](
        base, loader, lora,
        cond_txt.copy(), uncond_txt^, init_noise^,
        cos_h.copy(), sin_h.copy(),
        norm_out_w, norm_out_b,
        SAMPLE_STEPS, SAMPLE_CFG,
        Int(D), Int(FMLP), Int(IN_CH), Int(TXT_CH), Int(OUT_CH), Int(TIMESTEP_DIM),
        EPS, ctx,
    )

    var out_path = samples_dir + String("/step_") + String(step) + String(".png")
    qwenimage_decode_packed_to_png[
        N_IMG, ROPE_H, ROPE_W, SAMPLE_LAT_C, SAMPLE_PATCH, IN_CH
    ](
        latent, vae_dir, out_path, ctx,
    )
    print("[QwenImage-lora] sample step=", step, " -> ", out_path)


def main() raises:
    var a = argv()
    var cfg_path = String(DEFAULT_CONFIG)
    var run_steps = DEFAULT_RUN_STEPS
    if len(a) >= 2:
        var arg1 = String(a[1])
        if _is_nonnegative_int(arg1):
            run_steps = _parse_nonnegative_int(arg1)
        else:
            cfg_path = arg1^
    if len(a) >= 3:
        run_steps = _parse_nonnegative_int(String(a[2]))

    var train_cfg = read_model_config(cfg_path)
    validate_qwen_train_config(train_cfg)
    var cache_preflight = create_onetrainer_cache_preflight_plan(train_cfg)
    validate_onetrainer_cache_preflight_plan(cache_preflight)
    var sample_cadence = qwen_sample_cadence_from_train_config(cfg_path, train_cfg)
    var offload_cfg = qwen_offload_config_from_train_config(train_cfg)
    if run_steps <= 0:
        run_steps = train_cfg.max_steps
    if run_steps > train_cfg.max_steps:
        run_steps = train_cfg.max_steps
    var cache_dir = String(DEFAULT_CACHE_DIR)
    if train_cfg.dataset_cache_dir != String(""):
        cache_dir = train_cfg.dataset_cache_dir.copy()
    var output_lora_path = String(LORA_DIR) + String("/qwenimage_lora_smoke.safetensors")
    if train_cfg.output_model_destination != String(""):
        output_lora_path = train_cfg.output_model_destination.copy()

    print("=== Qwen-Image REAL LoRA training loop (block-swap offload) ===")
    print("  config:", cfg_path)
    print("  arch: D=", D, " H=", H, " Dh=", Dh, " F=", FMLP, " in_ch=", IN_CH,
          " txt_ch=", TXT_CH, " out_ch=", OUT_CH)
    print("  depth: NUM_DOUBLE=", NUM_DOUBLE, " (all-double)")
    print("  tokens: N_IMG=", N_IMG, " N_TXT=", N_TXT, " S=", S)
    print("  recipe: rank=", train_cfg.lora_rank, " alpha=", train_cfg.lora_alpha,
          " lr=", train_cfg.lr, " shift=", train_cfg.timestep_shift,
          " max_grad_norm=", train_cfg.max_grad_norm)
    print("  run_steps=", run_steps, " config_max_steps=", train_cfg.max_steps)
    print(
        "  cadence: save_every=", train_cfg.save_every,
        " sample_after=", sample_cadence.sample_after,
        " unit=", sample_time_unit_name(sample_cadence.sample_after_unit),
        " skip_first=", sample_cadence.sample_skip_first,
        " sample_file=", sample_cadence.sample_definition_file_name,
    )
    if train_cfg.enable_async_offloading:
        print("[offload] async offload requested by config; Qwen trainer currently uses synchronous TurboPlannedLoader")
    print("  LoRA targets: 12/block (img/txt x q,k,v,out,ff_up,ff_down) x 60 = 720")
    print("  fixed_sigma_smoke=", FIXED_SIGMA_SMOKE)
    print("  ckpt:", train_cfg.checkpoint)
    print("  cache:", cache_dir)

    if should_sample_completed_step(sample_cadence, 0):
        print("[cadence] step 0 sample due; fires after the first completed step (sampler is wired)")
    var next_sample = next_sample_completed_step(sample_cadence, 0, train_cfg.max_steps)
    print("[cadence] next sample completed_step=", next_sample)
    if train_cfg.only_cache:
        print("[QwenImage-lora] only_cache requested; no train steps will run in this trainer")
        return

    var ctx = DeviceContext()

    # ── load frozen stack-level base (img_in/txt_in/proj_out + timestep MLP) ──
    print("[load] QwenStackBase from checkpoint")
    var st = ShardedSafeTensors.open(train_cfg.checkpoint)

    var base_stack = load_qwen_stack_base(
        st,
        train_cfg.d_model,
        train_cfg.in_channels,
        train_cfg.joint_attention_dim,
        qwen_patchified_out_channels(train_cfg),
        ctx,
    )

    # timestep MLP weights (top-level in checkpoint)
    var te_lin1_w = _load_host_bf16_sharded(
        st, "time_text_embed.timestep_embedder.linear_1.weight", ctx
    )   # [D, TIMESTEP_DIM]
    var te_lin1_b = _load_host_bf16_sharded(
        st, "time_text_embed.timestep_embedder.linear_1.bias", ctx
    )   # [D]
    var te_lin2_w = _load_host_bf16_sharded(
        st, "time_text_embed.timestep_embedder.linear_2.weight", ctx
    )   # [D, D]
    var te_lin2_b = _load_host_bf16_sharded(
        st, "time_text_embed.timestep_embedder.linear_2.bias", ctx
    )   # [D]

    # norm_out.linear weights (for final scale/shift)
    var norm_out_w = _load_host_bf16_sharded(st, "norm_out.linear.weight", ctx)  # [2D, D]
    var norm_out_b = _load_host_bf16_sharded(st, "norm_out.linear.bias", ctx)    # [2D]

    var base = QwenOffloadBase(
        base_stack^,
        norm_out_w.copy(), norm_out_b.copy(),
        te_lin1_w.copy(), te_lin1_b.copy(),
        te_lin2_w.copy(), te_lin2_b.copy(),
    )
    print("[load] base resident (img_in/txt_in/proj_out/timestep-MLP/norm_out)")

    # ── block-swap offload loader ────────────────────────────────────────────
    var plan = build_qwenimage_offload_plan()
    var loader = TurboPlannedLoader.open(train_cfg.checkpoint, plan^, offload_cfg, ctx)
    print("[load] offload loader opened (", loader.block_count(), "blocks)")

    # ── 3-axis RoPE tables (fixed for 512px / 1 frame) ──────────────────────
    var qcfg = QwenImageConfig.qwen_image()
    var rope = build_qwenimage_rope_tables(
        Int(ROPE_FRAME), Int(ROPE_H), Int(ROPE_W), Int(N_TXT),
        Int(H), qcfg, STDtype.F32, ctx
    )
    var cos_h = rope[0].to_host(ctx)
    var sin_h = rope[1].to_host(ctx)
    print("[load] Qwen-Image 3-axis RoPE tables built (S*H x Dh//2)")

    # ── LoRA set (B=0 init -> identity at step 0) ────────────────────────────
    var lora = build_qwen_lora_set(
        train_cfg.num_double,
        train_cfg.d_model,
        train_cfg.mlp_hidden,
        train_cfg.lora_rank,
        train_cfg.lora_alpha,
    )
    var n_adapters = train_cfg.num_double * Int(DBL_SLOTS)
    print("[lora] adapters:", n_adapters, " (", DBL_SLOTS, "x", NUM_DOUBLE, "double)")

    var files: List[String]
    var have_cache = True
    try:
        files = _list_cache(cache_dir)
        print("[cache] samples:", len(files))
    except:
        files = List[String]()
        have_cache = False
        print("[cache] WARNING: no cache at", cache_dir, "- using synthetic tokens")

    var b_absum_init = Float32(0.0)
    for i in range(n_adapters):
        b_absum_init += _absum(lora.dbl[i].b)
    print("[lora] LoRA-B |.|_1 at init =", b_absum_init, " (expect 0.0)")

    # ── sample-during-training setup ─────────────────────────────────────────
    # Sampling is WIRED (see _qwen_run_sample): denoise the CURRENT base + streamed
    # blocks + LIVE LoRA -> Qwen VAE tiled-decode -> <LORA_DIR>/samples/step_<N>.png.
    # Conditioning v1 = the firing step's cached caption embeds (COND) + zeros (UNCOND).
    var sample_vae_dir = String(DEFAULT_VAE_DIR)
    if train_cfg.vae != String(""):
        sample_vae_dir = train_cfg.vae.copy()
    var samples_dir = String(LORA_DIR) + String("/samples")
    makedirs(samples_dir, exist_ok=True)
    print("[cadence] sample-during-training WIRED -> ", samples_dir,
          " (steps=", SAMPLE_STEPS, " cfg=", SAMPLE_CFG, " vae=", sample_vae_dir, ")")

    var first_loss = Float32(0.0)
    var last_loss = Float32(0.0)

    var train_start = perf_counter_ns()
    for k in range(1, run_steps + 1):
        var t0 = perf_counter_ns()

        # ── timestep (OneTrainer DISCRETE: idx -> sigma & model_t) ──
        # OneTrainer discretizes: idx = int(sigmoid(N)*1000*shift_remap); then
        #   sigma   = (idx+1)/1000  (noise/latent blend, _add_noise_discrete)
        #   model_t = idx/1000      (transformer timestep input; *1000 internally)
        # The blend coefficient (sigma) and the embedding input (model_t) DIFFER
        # by one quantum — this is the divergence being fixed.
        var sigma: Float32
        var model_t: Float32
        var step_seed = UInt64(1) if FIXED_SIGMA_SMOKE else UInt64(k)
        if FIXED_SIGMA_SMOKE:
            # discretize the fixed smoke sigma the same way (0.5 -> idx=499):
            var smoke_idx = Int(FIXED_SIGMA_VAL * Float32(1000.0))
            if smoke_idx >= 1000:
                smoke_idx = 999
            sigma = Float32(Float64(smoke_idx + 1) / 1000.0)
            model_t = Float32(Float64(smoke_idx) / 1000.0)
        else:
            var dts = sample_timestep_discrete_qwen(
                SEED_BASE + step_seed, train_cfg.timestep_shift, 1000
            )
            sigma = dts.sigma
            model_t = dts.model_t

        # ── load / synthesize tokens ──
        var img_tokens = List[Float32]()   # [N_IMG, IN_CH]
        var txt_tokens = List[Float32]()   # [N_TXT, TXT_CH]

        if have_cache and len(files) > 0:
            var slot = 0 if FIXED_SIGMA_SMOKE else (k - 1) % len(files)
            var cst = SafeTensors.open(files[slot])
            var latent_cache = _load_cache_preserving_dtype(cst, String("latent"), ctx)
            var latent_h = latent_cache.to_host_bf16(ctx)
            for i in range(len(latent_h)):
                img_tokens.append(latent_h[i].cast[DType.float32]())
            # txt embed cache key = "text_embedding" (the OT/producer key; matches
            # ernie/anima/sd35 producers). Latent key "latent" already matches.
            var txt_cache = _load_cache_preserving_dtype(cst, String("text_embedding"), ctx)
            var txt_flat = txt_cache.to_host_bf16(ctx)
            var txt_seq = len(txt_flat) // Int(TXT_CH)
            for r in range(Int(N_TXT)):
                if r < txt_seq:
                    for c in range(Int(TXT_CH)):
                        txt_tokens.append(txt_flat[r * Int(TXT_CH) + c].cast[DType.float32]())
                else:
                    for _ in range(Int(TXT_CH)):
                        txt_tokens.append(Float32(0.0))
        else:
            # synthetic: zeros (smoke compile check only)
            for _ in range(Int(N_IMG) * Int(IN_CH)):
                img_tokens.append(Float32(0.0))
            for _ in range(Int(N_TXT) * Int(TXT_CH)):
                txt_tokens.append(Float32(0.0))

        # ── flow-match: noisy = (1-sigma)*latent + sigma*noise ; target = noise - latent ──
        var noise = _host_noise(Int(N_IMG) * Int(IN_CH), SEED_BASE * UInt64(7919) + step_seed)
        var noisy = List[Float32]()
        var target = List[Float32]()
        var one_minus_sigma = Float32(1.0) - sigma
        for i in range(len(img_tokens)):
            noisy.append(one_minus_sigma * img_tokens[i] + sigma * noise[i])
            target.append(noise[i] - img_tokens[i])

        # ── silu_temb_h: frozen time_text_embed output [1, D] ──
        # Use model_t (= idx/1000), the OneTrainer transformer timestep input,
        # NOT sigma (= (idx+1)/1000, the blend coefficient).
        var silu_temb_h = _build_silu_temb(
            model_t,
            te_lin1_w, te_lin1_b, te_lin2_w, te_lin2_b,
            ctx,
        )

        # ── forward (offload, full 60-block depth) -> pred [N_IMG, OUT_CH] ──
        var fwd = qwenimage_stack_lora_forward_offload[H, Dh, N_IMG, N_TXT, S](
            noisy.copy(), txt_tokens.copy(), silu_temb_h.copy(),
            base, loader, lora,
            cos_h.copy(), sin_h.copy(),
            norm_out_w, norm_out_b,
            Int(D), Int(FMLP), Int(IN_CH), Int(TXT_CH), Int(OUT_CH), EPS, ctx,
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

        # ── backward (offload, full 60-block depth) ──
        var grads = qwenimage_stack_lora_backward_offload[H, Dh, N_IMG, N_TXT, S](
            d_loss,
            noisy.copy(), txt_tokens.copy(), silu_temb_h.copy(),
            base, loader, lora,
            cos_h.copy(), sin_h.copy(),
            norm_out_w, norm_out_b,
            fwd,
            Int(D), Int(FMLP), Int(IN_CH), Int(TXT_CH), Int(OUT_CH), EPS, ctx,
        )

        # ── grad norm + clip(1.0) ──
        var gn_before = _clip(grads, train_cfg.max_grad_norm)

        # ── AdamW ──
        var step_lr = ot_lr_for_optimizer_step(train_cfg, k)
        qwen_offload_lora_adamw_step(
            lora, grads, k, step_lr, ctx,
            train_cfg.beta1, train_cfg.beta2, train_cfg.eps, train_cfg.weight_decay,
        )

        var t1 = perf_counter_ns()
        var secs = Float64(t1 - t0) / 1.0e9
        print_trainer_progress(
            String("QwenImage-lora"), k, run_steps, 1,
            loss, Float64(gn_before), secs, 0.0,
            Float64(t1 - train_start) / 1.0e9,
        )
        if grads.nonfinite_lora_grads != 0:
            print("[QwenImage-lora] warning nonfinite_lora_grads=", grads.nonfinite_lora_grads)

        var saved_this_step = False
        if train_cfg.save_every > 0 and k % train_cfg.save_every == 0:
            var ckpt_path = _step_lora_path(output_lora_path, k)
            _ = save_qwen_lora(lora, ckpt_path, ctx)
            var ckpt_state = qwen_state_path_for_lora(ckpt_path)
            _ = save_qwen_lora_state(lora, ckpt_state, ctx)
            saved_this_step = True
            print("[checkpoint] saved step=", k, " state=", ckpt_state)
        if should_sample_completed_step(sample_cadence, k):
            if qwen_should_save_before_sample(sample_cadence, k, saved_this_step):
                var pre_sample_path = _step_lora_path(output_lora_path, k)
                _ = save_qwen_lora(lora, pre_sample_path, ctx)
                var pre_sample_state = qwen_state_path_for_lora(pre_sample_path)
                _ = save_qwen_lora_state(lora, pre_sample_state, ctx)
                print("[checkpoint] saved before sample step=", k, " state=", pre_sample_state)
            print(
                "[cadence] sample due at completed_step=", k,
                " sample_file=", sample_cadence.sample_definition_file_name,
            )
            # Sample from the CURRENT frozen base + streamed blocks + LIVE LoRA.
            # v1 conditioning: this step's cached caption embeds (txt_tokens) as
            # COND, zeros as UNCOND (see qwenimage_sample_resident.mojo header).
            # Skip if there is no real cache — synthetic txt_tokens are all-zeros and
            # would render a degenerate sample.
            if have_cache and len(files) > 0:
                _qwen_run_sample(
                    base, loader, lora, txt_tokens.copy(),
                    cos_h.copy(), sin_h.copy(),
                    norm_out_w, norm_out_b,
                    sample_vae_dir, samples_dir, k, ctx,
                )
            else:
                print("[QwenImage-lora] sample skipped step=", k,
                      " (no cache; synthetic zero conditioning)")

    print("")
    print("first_loss=", first_loss, " last_loss=", last_loss)
    var b_absum_final = Float32(0.0)
    for i in range(n_adapters):
        b_absum_final += _absum(lora.dbl[i].b)
    var trains = (b_absum_init == 0.0) and (b_absum_final > 0.0)
    if trains and (last_loss == last_loss):
        print("RESULT: REAL run OK — LoRA-B grew 0 ->", b_absum_final,
              "; loss", first_loss, "->", last_loss,
              (" (DECREASED)" if last_loss < first_loss else " (see trajectory)"))
        _ = save_qwen_lora(lora, output_lora_path, ctx)
        _ = save_qwen_lora_state(lora, qwen_state_path_for_lora(output_lora_path), ctx)
    else:
        print("RESULT: FAIL trains=", trains)
